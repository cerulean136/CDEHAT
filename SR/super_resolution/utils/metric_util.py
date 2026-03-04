import numpy as np

from .color_util import rgb2ycbcr, bgr2ycbcr


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img, color_format='RGB'):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].
        color_format (str): Image color format. When testing the Y channel, it can only be 'RGB' or 'BGR'.

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        if color_format == 'RGB':
            img = rgb2ycbcr(img, y_only=True)  # Only get Y channel.
            img = img[..., None]  # h, w -> h, w, 1
        elif color_format == 'BGR':
            img = bgr2ycbcr(img, y_only=True)  # Only get Y channel.
            img = img[..., None]  # h, w -> h, w, 1
        else:
            raise Exception('When changing to the Y channel, only RGB or BGR images are supported.')
    return img * 255.
