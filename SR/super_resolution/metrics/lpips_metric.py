import torch

from ..utils.metric_util import reorder_image
from ..utils.registry import METRIC_REGISTRY

try:
    from lpips import LPIPS
except ImportError:
    print(f"Package lpips is not installed. Installing...")

    # Check whl file.
    import os.path as osp

    whl_file = osp.join(osp.dirname(osp.abspath(__file__)), 'lpips-0.1.4-py3-none-any.whl')
    if not osp.exists(whl_file):
        print(f"Could not find wheel file {osp.basename(whl_file)}. Please check if the file exists.")
        exit()

    # Install the package using pip.
    import sys
    import subprocess

    install_command = [sys.executable, '-m', 'pip', 'install', whl_file]
    subprocess.check_call(install_command)

    # Try importing the package again.
    try:
        from lpips import LPIPS
    except ImportError:
        print(f"Failed to import lpips even after installation.")
        exit()


@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, crop_border, input_order='HWC', net='alex', model_path=None, device='cuda', **kwargs):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Ref: https://github.com/seungho-snu/SROOE/blob/main/codes/PerceptualSimilarity/models/__init__.py

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        net (str): 'alex', 'vgg', 'squeeze' are the base/trunk networks available.
        model_path (str): Path to the pretrained weights. Default: 'None'.
        device (str): 'cuda' or 'cpu'. Default: 'cuda'.

    Returns:
        float: LPIPS result.
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

    device = torch.device(device)
    tensor = torch.as_tensor(img).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.
    tensor2 = torch.as_tensor(img2).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.

    b, c, height, width = tensor.shape

    if height > 1000 or width > 1000:
        crop_size = 1000
        stride = crop_size  # # no opverlap

        range_list_height = list(range(0, height - crop_size, stride))
        range_list_width = list(range(0, width - crop_size, stride))
        if len(range_list_height) != 0:
            range_list_height.append(height - crop_size)
        else:
            range_list_height.append(0)
        if len(range_list_width) != 0:
            range_list_width.append(width - crop_size)
        else:
            range_list_width.append(0)
        count = 0
        lpips_total = 0.
        with torch.no_grad():
            perceptual_loss_fn = LPIPS(net=net, model_path=model_path, verbose=False).to(device)

            for h in range_list_height:
                for w in range_list_width:
                    count += 1
                    h_start, h_end = h, min(h + crop_size, height)
                    w_start, w_end = w, min(w + crop_size, width)

                    croped_tensor = tensor[:, :, h_start:h_end, w_start:w_end]
                    croped_tensor2 = tensor2[:, :, h_start:h_end, w_start:w_end]

                    lpips_result = perceptual_loss_fn(croped_tensor, croped_tensor2).item()
                    lpips_total += lpips_result
        return lpips_total / count
    else:
        with torch.no_grad():
            perceptual_loss_fn = LPIPS(net=net, model_path=model_path, verbose=False).to(device)
            lpips_result = perceptual_loss_fn(tensor, tensor2).item()
        return lpips_result
