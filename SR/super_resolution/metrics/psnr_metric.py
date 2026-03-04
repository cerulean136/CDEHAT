import numpy as np

from ..utils.metric_util import reorder_image, to_y_channel
from ..utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=False, color_format='RGB', **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
        color_format (str): Image color format. When testing the Y channel, it can only be 'RGB' or 'BGR'.
            When not testing the Y channel, it can be any value.

    Returns:
        float: PSNR result.
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

    if test_y_channel:
        color_format = color_format.upper()
        assert color_format in ['RGB', 'BGR'], 'When testing the Y channel, only RGB or BGR images are supported.'
        img = to_y_channel(img, color_format=color_format)
        img2 = to_y_channel(img2, color_format=color_format)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10. * np.log10(255. * 255. / mse)
