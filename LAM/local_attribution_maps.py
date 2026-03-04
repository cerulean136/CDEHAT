import os.path

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F
from PIL import Image
import random

sys.path.append('./LAM_Demo')

import ModelZoo
import ModelZoo.utils
from LAM_Demo.ModelZoo.matlab_functions import imresize

import SaliencyModel.attributes
import SaliencyModel.BackProp
import SaliencyModel.utils

from SaliencyModel.attributes import attr_grad
from SaliencyModel.utils import (cv2_to_pil, pil_to_cv2, gini,
                                 grad_norm, IG_baseline, interpolation, isotropic_gaussian_kernel)


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def LAM(input_model, input_image, block=False):
    model = ModelZoo.load_model(input_model)

    window_size = 16
    scale = 4
    img_lr, img_hr = SaliencyModel.utils.prepare_images(input_image, scale=scale)
    tensor_lr = ModelZoo.utils.PIL2Tensor(img_lr)[:3]
    tensor_hr = ModelZoo.utils.PIL2Tensor(img_hr)[:3]

    np_hr = np.array(img_hr)
    np_lr = np.clip(imresize(np_hr, scale=1 / scale), 0, 255).astype(np.uint8)
    lr_pil = Image.fromarray(np_lr)
    tensor_lr = torch.from_numpy(np_lr.transpose(2, 0, 1)).float() / 255.

    # todo: pad
    window_size = 16
    scale = 4
    c, old_h_lq, old_w_lq = tensor_lr.shape
    lqpad_size_in_h = window_size - old_h_lq % window_size if old_h_lq % window_size != 0 else 0
    lqpad_size_in_w = window_size - old_w_lq % window_size if old_w_lq % window_size != 0 else 0
    tensor_lr = F.pad(input=tensor_lr, pad=(0, lqpad_size_in_w, 0, lqpad_size_in_h),
                      mode='reflect')
    tensor_hr = F.pad(input=tensor_hr, pad=(0, lqpad_size_in_w * scale, 0, lqpad_size_in_h * scale),
                      mode='reflect')

    cv2_lr = np.moveaxis(tensor_lr.numpy(), 0, 2)
    cv2_hr = np.moveaxis(tensor_hr.numpy(), 0, 2)

    # fig, axes = plt.subplots(1, 4)
    # axes[0].imshow(cv2_hr)
    # axes[1].imshow(cv2_lr)

    draw_img = SaliencyModel.utils.pil_to_cv2(img_hr)
    cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
    position_pil = SaliencyModel.utils.cv2_to_pil(draw_img)
    # axes[2].imshow(position_pil)

    sigma = 1.2
    fold = 50
    l = 9
    alpha = 0.5
    attr_objective = SaliencyModel.BackProp.attribution_objective(attr_grad, h, w, window=window_size)
    gaus_blur_path_func = SaliencyModel.BackProp.GaussianBlurPath(sigma, fold, l)
    interpolated_grad_numpy, result_numpy, interpolated_numpy = SaliencyModel.BackProp.Path_gradient(tensor_lr.numpy(),
                                                                                                     tensor_hr,
                                                                                                     model,
                                                                                                     attr_objective,
                                                                                                     gaus_blur_path_func,
                                                                                                     cuda=True)
    grad_numpy, result = SaliencyModel.BackProp.saliency_map_PG(interpolated_grad_numpy, result_numpy)

    # axes[3].imshow(ModelZoo.utils.Tensor2PIL(torch.clamp(torch.tensor(result), min=0., max=1.)))

    abs_normed_grad_numpy = SaliencyModel.utils.grad_abs_norm(grad_numpy)

    saliency_image_abs = SaliencyModel.utils.vis_saliency(abs_normed_grad_numpy, zoomin=4)
    saliency_image_kde = SaliencyModel.utils.vis_saliency_kde(abs_normed_grad_numpy)

    blend_abs_and_input = SaliencyModel.utils.cv2_to_pil(
        pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    blend_kde_and_input = SaliencyModel.utils.cv2_to_pil(
        pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)

    sr_res = ModelZoo.utils.Tensor2PIL(torch.clamp(torch.tensor(result), min=0., max=1.))
    pil = SaliencyModel.utils.make_pil_grid(
        [position_pil,
         saliency_image_abs,
         saliency_image_kde,
         blend_abs_and_input,
         blend_kde_and_input,
         sr_res])

    # save image
    position_pil.save(input_image.replace('.png', F'_position.png'))
    saliency_image_abs.save(input_image.replace('.png', F'_{input_model}_attribution.png'))
    saliency_image_kde.save(input_image.replace('.png', F'_{input_model}_attribution_blend.png'))
    blend_abs_and_input.save(input_image.replace('.png', F'_{input_model}_contribution.png'))
    blend_kde_and_input.save(input_image.replace('.png', F'_{input_model}_contribution_blend.png'))
    sr_res.save(input_image.replace('.png', F'_{input_model}_sr.png'))
    lr_pil.save(input_image.replace('.png', F'_lr.png'))

    plt.figure()
    plt.imshow(pil)
    plt.axis('off')

    gini_index = SaliencyModel.utils.gini(abs_normed_grad_numpy)
    diffusion_index = (1 - gini_index) * 100
    print(f"The `DI` of this case is {diffusion_index}")

    # plt.tight_layout()
    plt.show(block=block)


if __name__ == '__main__':
    set_random_seed(42)

    input_image = r"LAM_Demo/test_images/bridge_250.png"
    w = 71
    h = 86

    LAM(input_model='EDSR@Base',
        input_image=input_image)
    LAM(input_model='RCAN@Base',
        input_image=input_image)
    # LAM(input_model='RRDBNet@Base',
    #     input_image=input_image)
    # LAM(input_model='SwinIR@Base',
    #     input_image=input_image)
    LAM(input_model='HAT@Base',
        input_image=input_image)
    LAM(input_model='TTST@Base',
        input_image=input_image)
    LAM(input_model='CDEHAT@Base',
        input_image=input_image,
        block=True)
