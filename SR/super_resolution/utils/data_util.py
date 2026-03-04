import os
import cv2
import math
import torch
import random
import rasterio
import threading
import numpy as np
import os.path as osp
import queue as Queue
from functools import partial
from collections import OrderedDict
from rasterio.enums import Compression
from torch.utils.data import DataLoader, Sampler

from .basicsr_util import scandir, get_root_logger, get_dist_info


def paths_from_folder(folder, key, mapping_keywords_func=None, mapping_order=None, **kwargs):
    """Retrieve files from a root directory. When using mapping functions, multiple paths with the same keyword
    can be combined. The output result is a list containing multiple dict. The key of each dict is a path
    type and the value are paths. When a single path is used, the type of the value is a path string.
    When there are multiple paths, the type of the value is a list containing multiple path strings.

    Args:
        folder: The root directory for retrieving paths.
        key: Path types of the root directory, can be 'gt' or 'lq'.
        mapping_keywords_func: A function to extract some keywords from a single path. Used to combine
            multiple paths with the same keyword.
        mapping_order: A list of some keywords. Can be used to filter paths and specify the
            order of combined paths.
        kwargs: Some options for retrieving paths.
    Returns:
        list[dict]: Returns a list containing multiple dict. The keys of the dict is path type
            and the values are paths.
            e.g., [{'gt_path': gt_path}, ...] or
                  [{'lq_path': [lq_path3, lq_path2, lq_path1]}, ...]
    """
    folder = osp.abspath(folder)
    all_paths = list(scandir(dir_path=folder, suffix=kwargs['suffix'], recursive=kwargs['recursive'],
                             full_path=True))
    paths = []
    if mapping_keywords_func is not None and mapping_order is not None:
        mapping_dict = {}
        for path in all_paths:
            mapping_keywords_1, mapping_keywords_2 = mapping_keywords_func(path, key, mapping_order)
            if mapping_keywords_1 is not None:
                if mapping_dict.get(mapping_keywords_1, None) is None:
                    mapping_dict[mapping_keywords_1] = OrderedDict()
                mapping_dict[mapping_keywords_1][mapping_keywords_2] = path
        del all_paths

        for mapping_1, mapping_2 in mapping_dict.items():
            mapping_paths = [mapping_2[mapping_i] for mapping_i in mapping_order]  # Sort paths by using mapping order.
            mapping_paths = mapping_paths[0] if len(mapping_paths) == 1 else mapping_paths
            paths.append({f'{key}_path': mapping_paths})
    else:
        for path in all_paths:
            paths.append({f'{key}_path': path})
    return paths


def paired_paths_from_folder(folders, keys, mapping_keywords_func=None, mapping_order=None, **kwargs):
    """Retrieve files from two root directories. When using mapping functions, multiple paths with the same keyword
    can be combined. The output result is a list containing multiple dict. The key of each dict are path
    type and the value are paths. When a single path is used, the type of the value is a path string.
    When there are multiple paths, the type of the value is a list containing multiple path strings.

    Args:
        folders: A list containing two root directories for retrieving paths.
        keys: A list containing path types for two root directories, which can be 'gt' and 'lq'.
        mapping_keywords_func: A function to extract some keywords from a single path. Used to combine
            multiple paths with the same keyword.
        mapping_order: A list contains two lists with some keywords. Can be used to filter paths and specify the
            order of combined paths.
        kwargs: Some options for retrieving paths.
    Returns:
        list[dict]: Returns a list containing multiple dict. The keys of the dict are path types
            and the values are paths.
            e.g., [{'gt_path': gt_path, 'lq_path': lq_path}, ...] or
                  [{'gt_path': [gt_path3, gt_path2, gt_path1], 'lq_path': lq_path}, ...]
    """
    assert len(folders) == 2, f'The len of folders should be 2 with [gt_folder, lq_folder]. But got {len(folders)}'
    assert len(keys) == 2, f'The len of keys should be 2 with [gt_key, lq_key]. But got {len(keys)}'
    gt_folder, lq_folder = folders
    gt_folder, lq_folder = osp.abspath(gt_folder), osp.abspath(lq_folder)
    gt_key, lq_key = keys
    gt_paths = list(scandir(dir_path=gt_folder, suffix=kwargs['suffix'], recursive=kwargs['recursive'],
                            full_path=True))
    lq_paths = list(scandir(dir_path=lq_folder, suffix=kwargs['suffix'], recursive=kwargs['recursive'],
                            full_path=True))
    paths = []
    if mapping_keywords_func is not None and mapping_order is not None:
        gt_mapping_order, lq_mapping_order = mapping_order

        gt_mapping_dict = {}
        for gt_path in gt_paths:
            gt_mapping_keywords_1, gt_mapping_keywords_2 = mapping_keywords_func(gt_path, gt_key, gt_mapping_order)
            if gt_mapping_keywords_1 is not None:
                if gt_mapping_dict.get(gt_mapping_keywords_1, None) is None:
                    gt_mapping_dict[gt_mapping_keywords_1] = OrderedDict()
                gt_mapping_dict[gt_mapping_keywords_1][gt_mapping_keywords_2] = gt_path
        del gt_paths

        lq_mapping_dict = {}
        for lq_path in lq_paths:
            lq_mapping_keywords_1, lq_mapping_keywords_2 = mapping_keywords_func(lq_path, lq_key, lq_mapping_order)
            if lq_mapping_keywords_1 is not None:
                if lq_mapping_dict.get(lq_mapping_keywords_1, None) is None:
                    lq_mapping_dict[lq_mapping_keywords_1] = OrderedDict()
                lq_mapping_dict[lq_mapping_keywords_1][lq_mapping_keywords_2] = lq_path
        del lq_paths

        gt_mapping_keys, lq_mapping_keys = set(gt_mapping_dict.keys()), set(lq_mapping_dict.keys())
        assert gt_mapping_keys == lq_mapping_keys, 'There is an error in the mapping relationship.'
        for mapping_key in gt_mapping_keys:
            gt_mapping_value = gt_mapping_dict[mapping_key]
            lq_mapping_value = lq_mapping_dict[mapping_key]
            gt_mapping_paths = [gt_mapping_value[gt_mapping_i] for gt_mapping_i in gt_mapping_order]  # Sort paths.
            lq_mapping_paths = [lq_mapping_value[lq_mapping_i] for lq_mapping_i in lq_mapping_order]  # Sort paths.
            gt_mapping_paths = gt_mapping_paths[0] if len(gt_mapping_paths) == 1 else gt_mapping_paths
            lq_mapping_paths = lq_mapping_paths[0] if len(lq_mapping_paths) == 1 else lq_mapping_paths
            paths.append({f'{gt_key}_path': gt_mapping_paths, f'{lq_key}_path': lq_mapping_paths})
    else:
        assert len(gt_paths) == len(lq_paths), \
            f'{gt_key} and {lq_key} datasets have different number of images: {len(gt_paths)}, {len(lq_paths)}.'
        for gt_path in gt_paths:
            gt_basename = osp.relpath(osp.abspath(gt_path), gt_folder)
            lq_path = osp.abspath(osp.join(lq_folder, gt_basename))
            assert lq_path in lq_paths, f'{lq_path} is not in {lq_key}_paths.'
            assert osp.exists(lq_path) and osp.exists(gt_path), f'{lq_path}, {gt_path} are not exist.'
            paths.append({f'{gt_key}_path': gt_path, f'{lq_key}_path': lq_path})
    return paths


class DataUtil:
    @staticmethod
    def image_path_to_bytes(file_path, is_text=False):
        """Read a file as bytes from the disk.

        Args:
            file_path (str): A path string that exists at a certain location on the disk.
            is_text (bool): True indicates textual data, false indicates images or
                other data. Default: False.
        Returns:
            bytes: image or text bytes.
        """
        if not osp.exists(file_path) or not osp.isfile(file_path):
            raise FileNotFoundError(f"file {file_path} does not exist")
        if not is_text:
            with open(file_path, 'rb') as fin:
                value_buf = fin.read()
        else:
            with open(file_path, 'r') as fin:
                value_buf = fin.read()
        return value_buf

    @staticmethod
    def image_from_bytes_by_cv2(content, flag='unchanged', bgr_to_rgb=False, normalize=False):
        """Read an image from bytes.
        this function is referenced from basicsr.utils.img_util.imfrombytes

        Args:
            content (bytes): Image bytes got from files or other streams.
            flag (str): Flags specifying the color type of a loaded image,
                candidates are `color`, `grayscale` and `unchanged`.
            bgr_to_rgb (bool): If true, convert BGR format image to RGB format
                or BGRA format image to RGBA format. Default: False.
            normalize (bool): Whether to change to float32 and norm
                to [0, 1]. Default: False.
        Returns:
            ndarray: Loaded image array.
        """
        # bytes to ndarray
        image_ndarray = np.frombuffer(buffer=content, dtype=np.uint8)  # convert bytes to one-dimensional ndarray
        image_read_flag = {'color': cv2.IMREAD_COLOR,
                           'grayscale': cv2.IMREAD_GRAYSCALE,
                           'unchanged': cv2.IMREAD_UNCHANGED}
        image = cv2.imdecode(buf=image_ndarray, flags=image_read_flag[flag])
        assert image.dtype == np.uint8, f'Type {image.dtype} is not supported, only uint8 type is supported.'
        # convert color channels
        if bgr_to_rgb and len(image.shape) == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(src=image, code=cv2.COLOR_BGRA2RGBA)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
        # normalized
        if normalize:
            image = image.astype(np.float32) / 255.
        return image

    @staticmethod
    def image_read_by_cv2(file_path, flag='unchanged', bgr_to_rgb=False, normalize=False):
        """Read an image from a disk path.

        Args:
            file_path (str): A path string that exists at a certain location on the disk.
            flag (str): Flags specifying the color type of a loaded image,
                candidates are `color`, `grayscale` and `unchanged`.
            bgr_to_rgb (bool): If true, convert BGR format image to RGB format
                or BGRA format image to RGBA format. Default: False.
            normalize (bool): Whether to change to float32 and norm
                to [0, 1]. Default: False.
        Returns:
            ndarray: Loaded image array.
        """
        if not osp.exists(file_path) or not osp.isfile(file_path):
            raise FileNotFoundError(f"file {file_path} does not exist")
        image_read_flag = {'color': cv2.IMREAD_COLOR,
                           'grayscale': cv2.IMREAD_GRAYSCALE,
                           'unchanged': cv2.IMREAD_UNCHANGED}
        image = cv2.imread(filename=file_path, flags=image_read_flag[flag])
        assert image.dtype == np.uint8, f'Type {image.dtype} is not supported, only uint8 type is supported.'
        # convert color channels
        if bgr_to_rgb and len(image.shape) == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(src=image, code=cv2.COLOR_BGRA2RGBA)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
        # normalized
        if normalize:
            image = image.astype(np.float32) / 255.
        return image

    @staticmethod
    def image_write_by_cv2(image, file_path, rgb_to_bgr=False, params=None):
        """Write image to file.
        this function is referenced from basicsr.utils.img_util.imfrombytes

        Args:
            image (ndarray): Image array to be written.
            file_path (str): Image file path.
            rgb_to_bgr (bool): If true, convert RGB format image to BGR format
                or RGBA format image to BGRA format. Default: False.
            params (None or list): Same as opencv's :func:`imwrite` interface.
        Returns:
            bool: Successful or not.
        """
        # mkdir
        os.makedirs(osp.abspath(osp.dirname(file_path)), exist_ok=True)
        # convert color channels
        if rgb_to_bgr and len(image.shape) == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(src=image, code=cv2.COLOR_RGBA2BGRA)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(src=image, code=cv2.COLOR_RGB2BGR)
        # save
        ok = cv2.imwrite(filename=file_path, img=image, params=params)
        if not ok:
            raise IOError(f'Error writing image array to file `{file_path}`.')

    @staticmethod
    def image_to_tensor(image, channel_axis=2, bgr_to_rgb=False, normalize=False):
        """Numpy array to tensor.

        Args:
            image (ndarray): Image ndarray to be converted to tensor.
            channel_axis (int): Channel dimensions of the input image.
            bgr_to_rgb (bool): If true, convert BGR(A) format image to RGB(A) format.
            normalize (bool): Whether to change to and norm.
        Returns:
            tensor: Tensor image.
        """
        if image.dtype == np.float64:
            image = image.astype(np.float32)
        # Channel process.
        # The dimensions are transformed into (h, w, c), and when the number of channels is one, it is (h, w, 1).
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=channel_axis)
        if channel_axis != 2:
            image = np.moveaxis(image, source=channel_axis, destination=2)
        # For the data read with cv2, we can choose to convert the BGR(A) image color channel order to RGB(A).
        heigth, width, channel = image.shape
        if bgr_to_rgb:
            if channel == 3:
                image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
            elif channel == 4:
                image = cv2.cvtColor(src=image, code=cv2.COLOR_BGRA2RGBA)
            else:
                raise Exception(
                    'The parameter bgr_to_rgb is only available when the input image is of BGR(A) type.')
        # Convert ndarray to tensor.
        tensor = torch.from_numpy(image)
        # Convert (h, w, c) to (c, h, w).
        tensor = torch.permute(input=tensor, dims=(2, 0, 1))
        # Normalized to [0, 1].
        if normalize:
            tensor = tensor.float() / 255.
        return tensor

    @staticmethod
    def images_to_tensors(image, channel_axis=2, bgr_to_rgb=False, normalize=False):
        """Numpy array to tensor.

        Args:
            image (ndarray|list[ndarray]): An image ndarray or a list filled with ndarray
                to be converted to tensors.
            channel_axis (int): Channel dimensions of the input image.
            bgr_to_rgb (bool): If true, convert BGR(A) format image to RGB(A) format.
            normalize (bool): Whether to change to and norm.
        Returns:
            (tensor|list[tensor]): Tensor images. If returned results only have
                one element, just return tensor.
        """
        # Batch processing.
        images_list = [image] if not isinstance(image, list) else image
        tensors = []
        for img in images_list:
            assert isinstance(img, np.ndarray), 'The data to be converted is not ndarray type.'
            tensor = DataUtil.image_to_tensor(
                image=img, channel_axis=channel_axis, bgr_to_rgb=bgr_to_rgb, normalize=normalize)
            tensors.append(tensor)
        tensors = tensors[0] if len(tensors) == 1 else tensors
        return tensors

    @staticmethod
    def _tensor_to_image(tensor, out_type=np.uint8, rgb_to_bgr=False, min_max=(0, 1)):
        # tensor preprocessing.
        tensor = tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        # Normalized to min_max.
        tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])

        def process(_tensor):
            # Processing in different dimensions.
            n_dim = _tensor.dim()
            if n_dim == 4:
                batch = _tensor.shape[0]
                for b in range(batch):
                    yield from process(_tensor[b, :, :, :])
            elif n_dim == 3:
                _array = _tensor.numpy()
                _array = np.moveaxis(_array, source=0, destination=2)
                # For the data read with cv2, we can choose to convert the RGB(A) image color channel order to BGR(A).
                heigth, width, channel = _array.shape
                if rgb_to_bgr:
                    if channel == 3:
                        _array = cv2.cvtColor(src=_array, code=cv2.COLOR_RGB2BGR)
                    elif channel == 4:
                        _array = cv2.cvtColor(src=_array, code=cv2.COLOR_RGBA2BGRA)
                    else:
                        raise Exception(
                            'The parameter rgb_to_bgr is only available when the input image is of RGB(A) type.')
                if channel == 1:
                    _array = np.squeeze(_array, axis=2)  # (h, w, 1) -> (h, w)

                if out_type == np.uint8:
                    _array = (_array * 255.).round().astype(out_type)
                elif out_type == 'sentinel2':
                    _array = (_array * 10000.).round().astype(np.uint16)
                else:
                    _array = _array.astype(out_type)
                yield _array
            elif n_dim == 2:
                _array = _tensor.numpy()

                if out_type == np.uint8:
                    _array = (_array * 255.).round().astype(out_type)
                elif out_type == 'sentinel2':
                    _array = (_array * 10000.).round().astype(np.uint16)
                else:
                    _array = _array.astype(out_type)
                yield _array
            else:
                raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')

        output = list(process(tensor))
        return output[0] if len(output) == 1 else output

    @staticmethod
    def tensor_to_image(tensor, out_type=np.uint8, rgb_to_bgr=False, min_max=(0, 1)):
        """Convert torch Tensor into image numpy array.
        If output type is np.uint8, transform ndarray output to uint8 type with range [0, 255].
        Note that when the number of channels of the output ndarray is 1, the channel is canceled, for example,
        (H, W, 1) is converted to (H, W).

        Args:
            tensor (Tensor): Accept shapes:
                1) 4D Tensor of shape (B x C x H x W);
                2) 3D Tensor of shape (C x H x W);
                3) 2D Tensor of shape (H x W).
            out_type (numpy type): output types. If np.uint8, transform outputs
                to uint8 type with range [0, 255]; otherwise, float type with
                range [0, 1]. Default: np.uint8.
            rgb_to_bgr (bool): If true, convert RGB(A) format image to BGR(A) format.
            min_max (tuple[int]): min and max values for clamp.
        Returns:
            (ndarray|list[ndarray]):
                1) For 4D input Tensor: Return a list filled with ndarrays of shape (H x W x C). e.g.[ndarray_1, ...]
                2) For 3D input Tensor: Return a ndarray of shape (H x W x C).
                3) For 2D input Tensor: Return a ndarray of shape (H x W).
        """
        # tensor preprocessing.
        tensor = tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        # Normalized to min_max.
        tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
        # Processing in different dimensions.
        n_dim = tensor.dim()
        array_list = []  # For 4D, 3D, 2D tensors.
        if n_dim == 4:
            batch = tensor.shape[0]
            for b in range(batch):
                array = tensor[b, :, :, :].numpy()
                array = np.moveaxis(array, source=0, destination=2)
                # For the data read with cv2, we can choose to convert the RGB(A) image color channel order to BGR(A).
                heigth, width, channel = array.shape
                if rgb_to_bgr:
                    if channel == 3:
                        array = cv2.cvtColor(src=array, code=cv2.COLOR_RGB2BGR)
                    elif channel == 4:
                        array = cv2.cvtColor(src=array, code=cv2.COLOR_RGBA2BGRA)
                    else:
                        raise Exception(
                            'The parameter rgb_to_bgr is only available when the input image is of RGB(A) type.')
                if channel == 1:
                    array = np.squeeze(array, axis=2)  # (h, w, 1) -> (h, w)
                array_list.append(array)
        elif n_dim == 3:
            array = tensor.numpy()
            array = np.moveaxis(array, source=0, destination=2)
            # For the data read with cv2, we can choose to convert the RGB(A) image color channel order to BGR(A).
            heigth, width, channel = array.shape
            if rgb_to_bgr:
                if channel == 3:
                    array = cv2.cvtColor(src=array, code=cv2.COLOR_RGB2BGR)
                elif channel == 4:
                    array = cv2.cvtColor(src=array, code=cv2.COLOR_RGBA2BGRA)
                else:
                    raise Exception(
                        'The parameter rgb_to_bgr is only available when the input image is of RGB(A) type.')
            if channel == 1:
                array = np.squeeze(array, axis=2)  # (h, w, 1) -> (h, w)
            array_list.append(array)
        elif n_dim == 2:
            array = tensor.numpy()
            array_list.append(array)
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        # Convert data types and ranges.
        if out_type == np.uint8:
            output = [(array * 255.).round().astype(out_type) for array in array_list]
        else:
            output = [array.astype(out_type) for array in array_list]
        output = output[0] if len(output) == 1 else output
        return output

    @staticmethod
    def tensors_to_images(tensor, out_type=np.uint8, rgb_to_bgr=False, min_max=(0, 1)):
        """Convert torch Tensors into image numpy arrays.
        If output type is np.uint8, transform ndarray outputs to uint8 type with range [0, 255].
        Note that when the number of channels of the output ndarray is 1, the channel is canceled, for example,
        (H, W, 1) is converted to (H, W).

        Args:
            tensor (Tensor|list[Tensor]): Can be a tensor or a list full of tensors. Accept shapes:
                1) 4D Tensor of shape (B x C x H x W);
                2) 3D Tensor of shape (C x H x W);
                3) 2D Tensor of shape (H x W).
            out_type (numpy type): output types. If np.uint8, transform outputs
                to uint8 type with range [0, 255]; otherwise, float type with
                range [0, 1]. Default: np.uint8.
            rgb_to_bgr (bool): If true, convert RGB(A) format image to BGR(A) format.
            min_max (tuple[int]): min and max values for clamp.
        Returns:
            (ndarray|list[ ndarray|list[ndarray] ]): When the number of elements in the input tensor list is greater
            than one, a list containing the following elements is returned; when the number of elements in the input
            tensor or input list is equal to one, the following elements are returned directly.
                1) For 4D input Tensor: Return a list filled with ndarrays of shape (H x W x C). e.g.[ndarray_1, ...]
                2) For 3D input Tensor: Return a ndarray of shape (H x W x C).
                3) For 2D input Tensor: Return a ndarray of shape (H x W).
        """
        # Batch processing.
        tensors_list = [tensor] if not isinstance(tensor, list) else tensor
        images = []
        for tsr in tensors_list:
            assert torch.is_tensor(tsr) is True, 'The data to be converted is not tensor type.'
            image = DataUtil.tensor_to_image(
                tensor=tensor, out_type=out_type, rgb_to_bgr=rgb_to_bgr, min_max=min_max)
            images.append(image)
        images = images[0] if len(images) == 1 else images
        return images

    @staticmethod
    def image_read_by_rasterio(input_image, channel_axis=0):
        input_image = [input_image] if not isinstance(input_image, list) else input_image
        output_image = []
        for image in input_image:
            # Start read.
            with rasterio.open(image, mode='r') as src:
                # The dimensions of image_data is (c, h, w), and when the number of channels is one, it is (1, h, w).
                image_data = src.read()
                if channel_axis != 0:
                    image_data = np.moveaxis(image_data, 0, channel_axis)
                output_image.append(image_data)
        output_image = output_image[0] if len(output_image) == 1 else output_image
        return output_image

    @staticmethod
    def image_write_by_rasterio(input_image, output_image, channel_axis=0,
                                transform=None, crs=None, origin=None, nodata=None, lzw=False):
        # Make dir.
        os.makedirs(os.path.dirname(output_image), exist_ok=True)

        # Channel process.
        if len(input_image.shape) == 2:
            input_image = np.expand_dims(input_image, axis=channel_axis)
        if channel_axis != 0:
            input_image = np.moveaxis(input_image, channel_axis, 0)

        # Get meta.
        channel, height, width = input_image.shape
        dtype = input_image.dtype

        if transform is None and origin is not None:  # origin = (west, north, xsize, ysize)
            assert len(origin) == 4, "origin(west, north, xsize, ysize) should be a list or tuple of length 4"
            transform = rasterio.transform.from_origin(*origin)

        if lzw:
            compress = Compression.lzw
        else:
            compress = Compression.none

        driver = "PNG" if os.path.splitext(output_image)[1].lower() == ".png" else "GTiff"

        # Start write.
        with rasterio.open(output_image, mode='w', driver=driver,
                           count=channel, height=height, width=width, dtype=dtype,
                           transform=transform, crs=crs,
                           nodata=nodata, compress=compress) as dst:
            dst.write(input_image)

    @staticmethod
    def get_crs_and_transform(input_image):
        with rasterio.open(input_image, mode='r') as src:
            crs = src.crs
            transform = src.transform
        return crs, transform


def build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
    """Build dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None
    """
    phase = dataset_opt['phase']
    rank, _ = get_dist_info()
    if phase == 'train':
        if dist:  # distributed training
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:  # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True)
        if sampler is None:
            dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
    elif phase in ['val', 'test']:  # validation
        dataloader_args = dict(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    else:
        raise ValueError(f"Wrong dataset phase: {phase}. Supported ones are 'train', 'val' and 'test'.")

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)
    dataloader_args['persistent_workers'] = dataset_opt.get('persistent_workers', False)

    prefetch_mode = dataset_opt.get('prefetch_mode')
    if prefetch_mode == 'cpu':  # CPUPrefetcher
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        logger = get_root_logger()
        logger.info(f'Use {prefetch_mode} prefetch dataloader: num_prefetch_queue = {num_prefetch_queue}')
        return PrefetchDataLoader(num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    else:
        # prefetch_mode=None: Normal dataloader
        # prefetch_mode='cuda': dataloader for CUDAPrefetcher
        return torch.utils.data.DataLoader(**dataloader_args)


def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class EnlargedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    Modified from torch.utils.data.distributed.DistributedSampler
    Support enlarging the dataset for iteration-based training, for saving
    time when restart the dataloader after each epoch

    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        num_replicas (int | None): Number of processes participating in
            the training. It is usually the world_size.
        rank (int | None): Rank of the current process within num_replicas.
        ratio (int): Enlarging ratio. Default: 1.
    """

    def __init__(self, dataset, num_replicas, rank, ratio):
        super(EnlargedSampler, self).__init__()
        self.dataset = dataset
        self.num_replicas = num_replicas  # world size(GPU number)
        self.rank = rank
        self.epoch = 0
        self.num_samples = math.ceil((len(dataset) * ratio) / num_replicas)  # number of samples in a single GPU
        self.total_size = self.num_samples * self.num_replicas  # number of samples in all GPUs

    def __iter__(self):
        # use epoch as the seed of the generator
        g = torch.Generator()
        g.manual_seed(self.epoch)
        # generate a random permutation of 0 to total_size - 1
        indices = torch.randperm(n=self.total_size, generator=g).tolist()
        # correct the indice value to the appropriate dataset index range 0 to data_size - 1
        data_size = len(self.dataset)
        indices = [v % data_size for v in indices]
        # get the indices of current rank
        rank_indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(rank_indices) == self.num_samples

        return iter(rank_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class PrefetchGenerator(threading.Thread):
    """A general prefetch generator.

    Ref:
    https://stackoverflow.com/questions/7323664/python-generator-pre-fetch

    Args:
        generator: Python generator.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, generator, num_prefetch_queue):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(num_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """Prefetch version of dataloader.

    Ref:
    https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#

    TODO:
    Need to test on single gpu and ddp (multi-gpu). There is a known issue in
    ddp.

    Args:
        num_prefetch_queue (int): Number of prefetch queue.
        kwargs (dict): Other arguments for dataloader.
    """

    def __init__(self, num_prefetch_queue, **kwargs):
        self.num_prefetch_queue = num_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_prefetch_queue)


class CPUPrefetcher(object):
    """CPU prefetcher.

    Args:
        loader: Dataloader.
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self):
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def reset(self):
        self.loader = iter(self.ori_loader)


class CUDAPrefetcher(object):
    """CUDA prefetcher.

    Ref:
    https://github.com/NVIDIA/apex/issues/304#

    It may consums more GPU memory.

    Args:
        loader: Dataloader.
        opt (dict): Options.
    """

    def __init__(self, loader, opt):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)  # self.batch is a dict
        except StopIteration:
            self.batch = None
            return None
        # put tensors to gpu
        with torch.cuda.stream(self.stream):
            for k, v in self.batch.items():
                if torch.is_tensor(v):
                    self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self):
        self.loader = iter(self.ori_loader)
        self.preload()
