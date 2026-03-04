import numpy as np
import torch
from scipy.stats import pearsonr

from ..utils.metric_util import reorder_image
from ..utils.registry import METRIC_REGISTRY

from torchmetrics.image import SpatialCorrelationCoefficient as SCC

"""
scc_cal = SCC()
scc_new = scc_cal(sr_tensor, hr_tensor)
"""


@METRIC_REGISTRY.register()
def calculate_scc(img, img2, crop_border, input_order='HWC', **kwargs):
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

    sr_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    hr_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)
    scc_cal = SCC()
    scc = scc_cal(sr_tensor, hr_tensor)

    # print(scc_new)
    # img_flat = img.flatten()
    # img2_flat = img2.flatten()
    # scc, _ = pearsonr(img_flat, img2_flat)
    return scc
