import numpy as np

from ..utils.metric_util import reorder_image
from ..utils.registry import METRIC_REGISTRY

"""
from torchmetrics.image import ErrorRelativeGlobalDimensionlessSynthesis

ergas_cal = ErrorRelativeGlobalDimensionlessSynthesis()
ergas = ergas_cal(sr_tensor, hr_tensor)
"""


@METRIC_REGISTRY.register()
def calculate_ergas(img, img2, crop_border, input_order='HWC', ratio=4, **kwargs):
    """Calculate ERGAS.
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

    h, w, c = img.shape
    preds = img.reshape(h * w, c)
    target = img2.reshape(h * w, c)

    diff = preds - target
    sum_squared_error = np.sum(diff ** 2, axis=0)
    rmse_per_band = np.sqrt(sum_squared_error / (h * w))
    mean_target = np.mean(target, axis=0)

    ergas_score = 100 / ratio * np.sqrt(np.sum((rmse_per_band / mean_target) ** 2) / c)

    return ergas_score
