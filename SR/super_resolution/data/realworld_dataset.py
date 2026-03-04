import cv2
import random
import torch
import math
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import normalize

from ..utils.data_util import DataUtil, paths_from_folder, paired_paths_from_folder
from ..utils.matlab_functions import imresize
from ..utils.color_util import rgb2ycbcr
from ..utils.registry import DATASET_REGISTRY
from ..utils.degradations import circular_lowpass_kernel, random_mixed_kernels

@DATASET_REGISTRY.register()
class RealWorldDataset(Dataset):
    def __init__(self, opt):
        super(RealWorldDataset, self).__init__()

        # Initialization parameters
        self.opt = opt  # dataset options
        self.io_backend_opt = None  # io backed options
        self.phase = opt['phase']  # train, val or test
        self.scale = opt['scale']  # scale index
        self.gt_size = opt.get('gt_size', None)  # maximum size of gt images, large ones will be randomly cropped.
        self.gt_folder = opt.get('dataroot_gt', None)  # high resolution dir
        self.lq_folder = opt.get('dataroot_lq', None)  # low resolution dir
        self.suffix = tuple(opt.get('suffix', ['.tif', '.png']))  # selected file suffix type
        self.is_paired_dataset = None  # both gt, lq are available, this dataset can be used for train, val and test
        self.is_gt_dataset = None  # only gt are available, this dataset can be used for train, val and test
        self.is_lq_dataset = None  # only lq are available, this dataset can be used for prediction
        self.mean, self.std = opt.get('mean', None), opt.get('std', None)  # mean & std for normalizing the input images
        self.use_hflip, self.use_rot = opt.get('use_hflip', False), opt.get('use_rot', False)  # data augmentation
        self.paths = []
        # The format of self.paths:
        # if is_paired_dataset: [{'gt_path': gt_path, 'lq_path': lq_path}, ...]
        # if is_gt_dataset: [{'gt_path': gt_path}, ...]
        # if is_lq_dataset: [{'lq_path': lq_path}, ...]
        # (Note: The value gt_path or lq_path corresponding to the key 'gt_path' or 'lq_path' can contain one or
        #   more paths, which is a string type for a single path and a list type for multiple paths..)
        self.keywords_gt = opt.get('keywords_gt', None)  # Use these keywords to filter and sort gt paths.
        self.keywords_lq = opt.get('keywords_lq', None)  # Use these keywords to filter and sort lq paths.

        # Set the paired | gt | lq dataset flag
        if self.gt_folder and self.lq_folder:
            self.is_paired_dataset = True
            self.is_gt_dataset = False
            self.is_lq_dataset = False
        elif self.gt_folder and self.lq_folder is None:
            self.is_paired_dataset = False
            self.is_gt_dataset = True
            self.is_lq_dataset = False
        elif self.gt_folder is None and self.lq_folder:
            self.is_paired_dataset = False
            self.is_gt_dataset = False
            self.is_lq_dataset = True
        else:
            raise ValueError('gt_folder and lq_folder cannot be empty at the same time.')

        # Get dataset paths
        self.io_backend_opt = opt['io_backend']
        if self.io_backend_opt['type'] == 'disk':  # disk type
            if self.opt.get('meta_info_file') is not None:  # With file: meta_info_file
                self.meta_info_file = self.opt['meta_info_file']
                with open(self.meta_info_file, 'r') as fin:
                    paths = [line.strip() for line in fin]
                    if self.is_paired_dataset:
                        for path in paths:
                            # gt_path and lq_path are divided from a line of txt, the delimiter is '; '
                            gt_path, lq_path = path.split('; ')
                            # gt(lq)_path_split are divided from gt(lq)_path, the delimiter is ', '
                            gt_path_split, lq_path_split = gt_path.split(', '), lq_path.split(', ')
                            gt_paths = [osp.join(self.gt_folder, path_split) for path_split in gt_path_split]
                            lq_paths = [osp.join(self.lq_folder, path_split) for path_split in lq_path_split]
                            gt_paths = gt_paths[0] if len(gt_paths) == 1 else gt_paths
                            lq_paths = lq_paths[0] if len(lq_paths) == 1 else lq_paths
                            self.paths.append({'gt_path': gt_paths, 'lq_path': lq_paths})
                    elif self.is_gt_dataset:
                        for gt_path in paths:
                            # gt_path_split are divided from gt_path, the delimiter is ', '
                            gt_path_split = gt_path.split(', ')
                            gt_paths = [osp.join(self.gt_folder, path_split) for path_split in gt_path_split]
                            gt_paths = gt_paths[0] if len(gt_paths) == 1 else gt_paths
                            self.paths.append({'gt_path': gt_paths})
                    elif self.is_lq_dataset:
                        for lq_path in paths:
                            # lq_path_split are divided from lq_path, the delimiter is ', '
                            lq_path_split = lq_path.split(', ')
                            lq_paths = [osp.join(self.lq_folder, path_split) for path_split in lq_path_split]
                            lq_paths = lq_paths[0] if len(lq_paths) == 1 else lq_paths
                            self.paths.append({'lq_path': lq_paths})
            else:  # Without file: meta_info_file
                if self.is_paired_dataset:
                    self.paths = paired_paths_from_folder(folders=[self.gt_folder, self.lq_folder], keys=['gt', 'lq'],
                                                          suffix=self.suffix, recursive=True)
                elif self.is_gt_dataset:
                    self.paths = paths_from_folder(folder=self.gt_folder, key='gt',
                                                   suffix=self.suffix, recursive=True)
                elif self.is_lq_dataset:
                    self.paths = paths_from_folder(folder=self.lq_folder, key='lq',
                                                   suffix=self.suffix, recursive=True)
        else:
            raise Exception(f'{self.io_backend_opt} is an unsupported io backend.')

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        # Get the paths specified by index.
        gt_path, lq_path = self.paths[index].get('gt_path', None), self.paths[index].get('lq_path', None)
        if self.is_paired_dataset:
            assert (gt_path is not None) and (lq_path is not None), 'gt_path and lq_path do not match the dataset type.'
        elif self.is_gt_dataset:
            assert (gt_path is not None) and (lq_path is None), 'gt_path and lq_path do not match the dataset type.'
        elif self.is_lq_dataset:
            assert (gt_path is None) and (lq_path is not None), 'gt_path and lq_path do not match the dataset type.'
        else:
            raise Exception('gt_path and lq_path cannot be empty at the same time.')

        # Load image from gt path and lq path.
        gt_array, lq_array = None, None
        # The dimensions of gt_array | lq_array is (c, h, w), and when the number of channels is one, it is (1, h, w).
        if gt_path is not None:
            gt_array = DataUtil.image_read_by_rasterio(gt_path)
            if isinstance(gt_array, list):
                gt_array = np.concatenate(gt_array, axis=0)
        if lq_path is not None:
            lq_array = DataUtil.image_read_by_rasterio(lq_path)
            if isinstance(lq_array, list):
                lq_array = np.concatenate(lq_array, axis=0)

        # We put the the channel dimension at the end, it should have the form (h, w, c).
        if gt_array is not None:
            gt_array = np.moveaxis(gt_array, source=0, destination=2)
        if lq_array is not None:
            lq_array = np.moveaxis(lq_array, source=0, destination=2)

        # Data augmentation
        phase, scale, gt_size, use_hflip, use_rot = self.phase, self.scale, self.gt_size, self.use_hflip, self.use_rot
        # If the following condition is established, lq_array and gt_array can finally be obtained.
        if phase == 'train' or phase == 'val' or (phase == 'test' and (self.is_gt_dataset or self.is_paired_dataset)):
            assert gt_array is not None, f'the gt_array should not be empty in {phase} phase.'
            gt_height, gt_width = gt_array.shape[0:2]

            # Get lq_array if lq_array doesn't exist, lq_array is calculated from gt_array.
            if lq_array is not None and self.is_paired_dataset is True:
                lq_height, lq_width = lq_array.shape[0:2]
                # Make sure the height and width of gt_array is an integer multiple of scale when compared with lq_array.
                if (lq_height * scale != gt_height) or (lq_width * scale != gt_width):
                    raise ValueError(f'scale mismatches. {lq_height}x{scale} is not {gt_height} '
                                     f'or {lq_width}x{scale} is not {gt_width}')
                else:
                    # If there is 'gt_size' parameter in the dataset configuration and
                    # the size of gt_array, lq_array is smaller than gt_size,
                    # then the size of gt_array, lq_array are adjusted and enlarged according gt_size.
                    if gt_size is not None and gt_size > min(gt_height, gt_width):
                        small_size = gt_height if gt_height <= gt_width else gt_width
                        scale_factor = gt_size / small_size
                        gt_height, gt_width = (
                            max(gt_size, int(gt_height * scale_factor) - int(gt_height * scale_factor) % scale),
                            max(gt_size, int(gt_width * scale_factor) - int(gt_width * scale_factor) % scale))
                        gt_array = cv2.resize(src=gt_array, dsize=(gt_width, gt_height), interpolation=cv2.INTER_LINEAR)
                        lq_height, lq_width = gt_height // scale, gt_width // scale
                        lq_array = cv2.resize(src=lq_array, dsize=(lq_width, lq_height), interpolation=cv2.INTER_LINEAR)
                        # Convert to continuous array.
                        gt_array = np.ascontiguousarray(gt_array, dtype=gt_array.dtype)
                        lq_array = np.ascontiguousarray(lq_array, dtype=lq_array.dtype)

            # lq_array does not exist and is calculated from gt_array.
            elif lq_array is None and self.is_gt_dataset is True:
                # Crop gt_array to an integer multiple of scale.
                gt_height, gt_width = gt_height - gt_height % scale, gt_width - gt_width % scale
                gt_array = gt_array[0:gt_height, 0:gt_width, :]
                # If there is 'gt_size' parameter in the dataset configuration and
                # the size of gt_array is smaller than gt_size,
                # then the size of gt_array, lq_array are adjusted and enlarged according gt_size.
                if gt_size is not None and gt_size > min(gt_height, gt_width):
                    small_size = gt_height if gt_height <= gt_width else gt_width
                    scale_factor = gt_size / small_size
                    gt_height, gt_width = (
                        max(gt_size, int(gt_height * scale_factor) - int(gt_height * scale_factor) % scale),
                        max(gt_size, int(gt_width * scale_factor) - int(gt_width * scale_factor) % scale))
                    gt_array = cv2.resize(src=gt_array, dsize=(gt_width, gt_height), interpolation=cv2.INTER_LINEAR)
                # Use matlab_functions get lq from gt.
                lq_height, lq_width = gt_height // scale, gt_width // scale
                lq_array = np.clip(a=imresize(img=gt_array, scale=1 / scale), a_min=0, a_max=255).astype(np.uint8)
                # Convert to continuous array.
                gt_array = np.ascontiguousarray(gt_array, dtype=gt_array.dtype)
                lq_array = np.ascontiguousarray(lq_array, dtype=lq_array.dtype)

            else:
                raise Exception('An unacceptable error occurred.')

            # If during the training phase, then use the following data augmentation operation.
            if phase == 'train':
                # If there is 'gt_size' parameter in the dataset configuration and
                # gt_size is smaller than the size of gt_array,
                # then gt_array and lq_array will be randomly cropped.
                if gt_size is not None and gt_size < max(gt_height, gt_width):
                    lq_size = gt_size // scale
                    top_lq, left_lq = random.randint(0, lq_height - lq_size), random.randint(0, lq_width - lq_size)
                    top_gt, left_gt = int(top_lq * scale), int(left_lq * scale)
                    gt_array = gt_array[top_gt:top_gt + gt_size, left_gt:left_gt + gt_size, :]
                    lq_array = lq_array[top_lq:top_lq + lq_size, left_lq:left_lq + lq_size, :]

                # Use horizontally flip, vertically flip and rotate 0, 90, 180, 270 degrees. If the parameters
                # use_hflip and use_rot are used at the same time, there are a total of 8 different situations with
                # probability 1/8.
                hflip = use_hflip and (random.random() < 0.5)
                vflip = use_rot and (random.random() < 0.5)
                rot = use_rot and (random.random() < 0.5)
                if hflip:  # Use horizontally flip
                    gt_array, lq_array = cv2.flip(src=gt_array, flipCode=1), cv2.flip(src=lq_array, flipCode=1)
                if vflip:  # Use vertically flip
                    gt_array, lq_array = cv2.flip(src=gt_array, flipCode=0), cv2.flip(src=lq_array, flipCode=0)
                if rot:  # Use rotation
                    gt_array, lq_array = gt_array.transpose(1, 0, 2), lq_array.transpose(1, 0, 2)

        # If the following condition is established, only lq_array can finally be obtained.
        elif phase == 'test' and self.is_lq_dataset:
            # Don't need to do any other processing of lq_array, just pass it.
            pass

        else:
            raise Exception('An unacceptable error occurred.')

        # Color space transform. Only supports RGB images.
        if self.opt.get('color', None) == 'y':
            if gt_array is not None:
                gt_array = rgb2ycbcr(img=gt_array, y_only=True)[..., None]
            if lq_array is not None:
                lq_array = rgb2ycbcr(img=lq_array, y_only=True)[..., None]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # Convert ndarray to tensor. tensor dimension order: (c, h, w), range: [0, 1], dtype: float32
        if gt_array is not None:
            gt_array = DataUtil.image_to_tensor(image=gt_array, channel_axis=2, bgr_to_rgb=False, normalize=True)
        if lq_array is not None:
            lq_array = DataUtil.image_to_tensor(image=lq_array, channel_axis=2, bgr_to_rgb=False, normalize=True)

        # Normalize.
        mean, std = self.mean, self.std
        if mean or std:
            if gt_array is not None:
                normalize(inpt=gt_array, mean=mean, std=std, inplace=True)
            if lq_array is not None:
                normalize(inpt=lq_array, mean=mean, std=std, inplace=True)

        # kernel 2 tensor
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        # Return.
        if self.is_lq_dataset:  # For predicton.
            return {'lq': lq_array, 'lq_path': lq_path,
                    'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel}
        elif self.is_gt_dataset:  # For train, val and test.
            return {'gt': gt_array, 'lq': lq_array, 'gt_path': gt_path,
                    'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel}
        elif self.is_paired_dataset:  # For train, val and test.
            return {'gt': gt_array, 'lq': lq_array, 'gt_path': gt_path, 'lq_path': lq_path,
                    'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel}

    def __len__(self):
        return len(self.paths)
