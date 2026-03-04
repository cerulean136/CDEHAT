import numpy as np
import torch
import torch.nn as nn
import torchvision.io
from scipy import linalg
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from ..utils.basicsr_util import scandir
from ..archs.inception import InceptionV3


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.

    Args:
        mu1 (np.array): The sample mean over activations.
        sigma1 (np.array): The covariance matrix over activations for
            generated samples.
        mu2 (np.array): The sample mean over activations, precalculated on an
               representative data set.
        sigma2 (np.array): The covariance matrix over activations,
            precalculated on an representative data set.

    Returns:
        float: The Frechet Distance.
    """
    assert mu1.shape == mu2.shape, 'Two mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Two covariances have different dimensions'

    cov_sqrt, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Product might be almost singular
    if not np.isfinite(cov_sqrt).all():
        print('Product of cov matrices is singular. Adding {eps} to diagonal of cov estimates')
        offset = np.eye(sigma1.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))
            raise ValueError(f'Imaginary component {m}')
        cov_sqrt = cov_sqrt.real

    mean_diff = mu1 - mu2
    mean_norm = mean_diff @ mean_diff
    trace = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(cov_sqrt)
    fid = mean_norm + trace
    return fid


def load_patched_inception_v3(device='cuda', resize_input=True, normalize_input=False):
    """Load pretrained inception v3 model

    Args:
        device (str, optional): Device to load model. Defaults to 'cuda'.
        resize_input (bool, optional): Whether to resize input. Defaults to True. If true,
            bilinearly resizes input to width and height 299 before feeding input to model.
        normalize_input (bool, optional): Whether to normalize input. Defaults to False.
            If true, scales the input from range (0, 1) to the range the pretrained
            Inception network expects, namely (-1, 1).
    """
    # we may not resize the input, but in [rosinality/stylegan2-pytorch] it
    # does resize the input.
    inception = InceptionV3([3], resize_input=resize_input, normalize_input=normalize_input)
    inception = nn.DataParallel(inception).eval().to(device)
    return inception


class ImageDataset(Dataset):
    def __init__(self, image_folder, recursive=False):
        self.image_folder = image_folder
        self.image_paths = list(scandir(dir_path=image_folder, recursive=recursive, full_path=True))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = torchvision.io.read_image(image_path)
        image = image.float() / 255.
        return image


def setup_dataloader(image_folder, batch_size=32):
    """Setup a dataloader for the image folder."""

    dataset = ImageDataset(image_folder=image_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


@torch.no_grad()
def extract_inception_features(data_generator, inception, len_generator=None, device='cuda'):
    """Extract inception features.

    Args:
        data_generator (generator): A data generator.
        inception (nn.Module): Inception model.
        len_generator (int): Length of the data_generator to show the
            progressbar. Default: None.
        device (str): Device. Default: cuda.

    Returns:
        Tensor: Extracted features.
    """
    if len_generator is not None:
        pbar = tqdm(total=len_generator, unit='batch', desc='Extract')
    else:
        pbar = None
    features = []

    for data in data_generator:
        if pbar:
            pbar.update(1)
        data = data.to(device)
        feature = inception(data)[0].view(data.shape[0], -1)
        features.append(feature.to('cpu'))
    if pbar:
        pbar.close()
    features = torch.cat(features, 0)
    return features


def calculate_statistics(features):
    """Calculate the mean and covariance matrix of features."""

    features = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
    mu = np.mean(features, axis=0)
    if features.shape[0] > 1:
        sigma = np.cov(features, rowvar=False)
    else:
        sigma = np.zeros((features.shape[1], features.shape[1]))
    return mu, sigma


def main(folder1, folder2):
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

    inception_v3 = load_patched_inception_v3(device=device_str, resize_input=True, normalize_input=True)

    dataloader1 = setup_dataloader(image_folder=folder1, batch_size=32)
    dataloader2 = setup_dataloader(image_folder=folder2, batch_size=32)

    # assert len(dataloader1) == len(dataloader2), 'The amount of data in the two folders is different.'

    features1 = extract_inception_features(data_generator=dataloader1, inception=inception_v3, device=device_str)
    features2 = extract_inception_features(data_generator=dataloader2, inception=inception_v3, device=device_str)

    mu1, sigma1 = calculate_statistics(features1)
    mu2, sigma2 = calculate_statistics(features2)

    fid = calculate_fid(mu1=mu1, sigma1=sigma1, mu2=mu2, sigma2=sigma2)
    print(f'\t# fid: {fid:7.4f}')
