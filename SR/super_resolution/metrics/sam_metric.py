import numpy as np

from ..utils.metric_util import reorder_image
from ..utils.registry import METRIC_REGISTRY

"""
from torchmetrics.image import SpectralAngleMapper

sam_cal = SpectralAngleMapper()
sam_new = sam_cal(sr_tensor, hr_tensor)
"""


@METRIC_REGISTRY.register()
def calculate_sam(img, img2, crop_border, input_order='HWC', **kwargs):
    """Calculate SAM (Spectral Angle Mapper).

    Ref: https://www.mathworks.com/help/images/ref/sam.html

    It works by calculating the angle between the spectra, where small angles between indicate high similarity and
    high angles indicate low similarity.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.

    Returns:
        float: SAM result.
    """
    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'
    input_order = input_order.upper()
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are HWC and CHW.')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    dot_product = np.sum(img * img2, axis=2)
    norm_img = np.sqrt(np.sum(img ** 2, axis=2))
    norm_img2 = np.sqrt(np.sum(img2 ** 2, axis=2))
    norm_product = norm_img * norm_img2
    norm_product = np.clip(norm_product, a_min=1e-10, a_max=None)
    cosine = dot_product / norm_product
    cosine = np.clip(cosine, a_min=-1, a_max=1)
    sam = np.arccos(cosine)
    return np.mean(sam)
