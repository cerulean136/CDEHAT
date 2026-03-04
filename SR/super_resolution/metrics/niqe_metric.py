import cv2
import math
import numpy as np
import os.path as osp
from scipy.special import gamma
from scipy.ndimage.filters import convolve

from ..utils.matlab_functions import imresize
from ..utils.metric_util import reorder_image, to_y_channel
from ..utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_niqe(img, crop_border, input_order='HWC', convert_to='Y', color_format='RGB', **kwargs):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.

    Ref: https://www.mathworks.com/help/images/ref/niqe.html
         http://live.ece.utexas.edu/research/quality/niqe_release.zip
         https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/metrics/niqe.py

    Args:
        img (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HW', 'HWC' or 'CHW'. Default: 'HWC'.
        convert_to (str): Y: Convert to Y channel of YCbCr; GRAY: Convert to grayscale. Default: Y.
        color_format (str): Image color format. When converting to Y channel or graysclae, it can only be 'RGB' or 'BGR'.
            Otherwise, it can be ignored.

    Returns:
        float: NIQE result.
    """
    niqe_pris_params = np.load(osp.abspath(osp.join(osp.dirname(__file__), 'niqe_pris_params.npz')))
    mu_pris_param = niqe_pris_params['mu_pris_param']
    cov_pris_param = niqe_pris_params['cov_pris_param']
    gaussian_window = niqe_pris_params['gaussian_window']

    img = img.astype(np.float32)
    input_order = input_order.upper()
    if input_order not in ['HW', 'HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are HW, HWC and CHW.')
    if input_order in ['HWC', 'CHW']:
        img = reorder_image(img, input_order=input_order)
        convert_to, color_format = convert_to.upper(), color_format.upper()
        if convert_to == 'Y':
            assert color_format in ['RGB', 'BGR'], 'When converting to Y channel, only RGB or BGR images are supported.'
            img = to_y_channel(img, color_format=color_format)
        elif convert_to == 'GRAY':
            assert color_format in ['RGB', 'BGR'], 'When converting to grayscale, only RGB or BGR images are supported.'
            if color_format == 'RGB':
                img = cv2.cvtColor(img / 255., cv2.COLOR_RGB2GRAY) * 255.
            elif color_format == 'BGR':
                img = cv2.cvtColor(img / 255., cv2.COLOR_BGR2GRAY) * 255.
            else:
                raise ValueError(f'Wrong color_format: {color_format}.')
        else:
            raise ValueError(f'Wrong convert_type: {convert_to}.')
        img = np.squeeze(img)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border]

    img = img.round().astype(np.float64)
    niqe_result = niqe(img, mu_pris_param, cov_pris_param, gaussian_window, block_size_h=96, block_size_w=96)
    return niqe_result


def niqe(img, mu_pris_param, cov_pris_param, gaussian_window, block_size_h=96, block_size_w=96):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.

    Args:
        img (ndarray): Input image whose quality needs to be computed. The
            image must be a gray or Y (of YCbCr) image with shape (h, w).
            Range [0, 255] with float type.
        mu_pris_param (ndarray): Mean of a pre-defined multivariate Gaussian
            model calculated on the pristine dataset.
        cov_pris_param (ndarray): Covariance of a pre-defined multivariate
            Gaussian model calculated on the pristine dataset.
        gaussian_window (ndarray): A 7x7 Gaussian window used for smoothing the
            image.
        block_size_h (int): Height of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
        block_size_w (int): Width of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
    """
    assert img.ndim == 2, 'Input image must be a grayscale or Y (of YCbCr) image with shape (H, W).'
    # crop image
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

    distparam = []  # dist param is actually the multiscale features
    for scale in (1, 2):  # perform on two scales (1, 2)
        mu = convolve(img, gaussian_window, mode='nearest')
        sigma = np.sqrt(np.abs(convolve(np.square(img), gaussian_window, mode='nearest') - np.square(mu)))
        # normalize, as in Eq. 1 in the paper
        img_nomalized = (img - mu) / (sigma + 1)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                # process ecah block
                block = img_nomalized[idx_h * block_size_h // scale:(idx_h + 1) * block_size_h // scale,
                        idx_w * block_size_w // scale:(idx_w + 1) * block_size_w // scale]
                feat.append(compute_feature(block))

        distparam.append(np.array(feat))

        if scale == 1:
            img = imresize(img / 255., scale=0.5, antialiasing=True)
            img = img * 255.

    distparam = np.concatenate(distparam, axis=1)

    # fit a MVG (multivariate Gaussian) model to distorted patch features
    mu_distparam = np.nanmean(distparam, axis=0)
    # use nancov. ref: https://ww2.mathworks.cn/help/stats/nancov.html
    distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
    cov_distparam = np.cov(distparam_no_nan, rowvar=False)

    # compute niqe quality, Eq. 10 in the paper
    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    quality = np.matmul(
        np.matmul((mu_pris_param - mu_distparam), invcov_param), np.transpose((mu_pris_param - mu_distparam)))

    quality = np.sqrt(quality)
    quality = float(np.squeeze(quality))
    return quality


def compute_feature(block):
    """Compute features.

    Args:
        block (ndarray): 2D Image block.

    Returns:
        list: Features with length of 18.
    """
    feat = []
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])

    # distortions disturb the fairly regular structure of natural images.
    # This deviation can be captured by analyzing the sample distribution of
    # the products of pairs of adjacent coefficients computed along
    # horizontal, vertical and diagonal orientations.
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for i in range(len(shifts)):
        shifted_block = np.roll(block, shifts[i], axis=(0, 1))
        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted_block)
        # Eq. 8
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])
    return feat


def estimate_aggd_param(block):
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters.

    Args:
        block (ndarray): 2D Image block.

    Returns:
        tuple: alpha (float), beta_l (float) and beta_r (float) for the AGGD
            distribution (Estimating the parames in Equation 7 in the paper).
    """
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)  # len = 9801
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))

    left_std = np.sqrt(np.mean(block[block < 0] ** 2))
    right_std = np.sqrt(np.mean(block[block > 0] ** 2))
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(block))) ** 2 / np.mean(block ** 2)
    rhatnorm = (rhat * (gammahat ** 3 + 1) * (gammahat + 1)) / ((gammahat ** 2 + 1) ** 2)
    array_position = np.argmin((r_gam - rhatnorm) ** 2)

    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return (alpha, beta_l, beta_r)
