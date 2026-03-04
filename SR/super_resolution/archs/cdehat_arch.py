import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from inspect import isfunction
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from ..utils.registry import ARCH_REGISTRY

use_cdp = None
cdp_feat = None
DropPath.extra_repr = lambda self: "" if hasattr(self, 'extra_repr') else None


class CDEM(nn.Module):
    def __init__(self, dim, num_feat):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, dim, kernel_size=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x, cdp):
        _, _, h, w = cdp.shape
        cdp = self.act(self.conv1(cdp))
        x_cdp = torch.cat([rearrange(x, 'b (h w) c -> b c h w', h=h, w=w), cdp], dim=1)
        x = x + rearrange(self.conv2(x_cdp), 'b c h w -> b (h w) c')
        return x


class CSAM(nn.Module):
    def __init__(self, dim, compress_ratio=3, reduction_factor=30, kernel=(3, 5, 7)):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim // compress_ratio, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // compress_ratio, dim, kernel_size=3, stride=1, padding=1))
        self.ca = ChannelAttention(dim, reduction_factor=reduction_factor)
        self.sa = SpatialAttention(kernel=kernel)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv_block(x)
        ca = self.ca(x)
        ca = rearrange(ca, 'b c h w -> b (h w) c')
        sa = self.sa(x)
        sa = rearrange(sa, 'b c h w -> b (h w) c')
        return ca * 0.01 + sa


class GTAM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cta = CTA(in_dim=dim, hidden_dim=dim)
        self.pta = PTA(in_dim=dim, hidden_dim=dim)

    def forward(self, x):
        b, c, h, w = x.shape
        cta = self.cta(x)  # b, c, h, w
        cta = rearrange(cta, 'b c h w -> b (h w) c')
        pta = self.pta(x)  # b, c, h, w
        pta = rearrange(pta, 'b c h w -> b (h w) c')
        return cta * 0.01 + pta


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, reduction_factor=30):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // reduction_factor, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // reduction_factor, num_feat, kernel_size=1, stride=1, padding=0))
        self.fc2 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // reduction_factor, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // reduction_factor, num_feat, kernel_size=1, stride=1, padding=0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_out = self.fc1(avg_out)
        max_out = self.fc2(max_out)
        out = avg_out + max_out
        out = self.sigmoid(out)
        out = x * out
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel=(3, 5, 7)):
        super().__init__()
        kernel = [kernel] if not isinstance(kernel, (list, tuple)) else kernel
        self.convs = nn.ModuleList()
        for k in kernel:
            assert k in (1, 3, 5, 7, 9), 'kernel size must be 1, 3, 5, 7 or 9'
            self.convs.append(nn.Conv2d(2, 1, kernel_size=k, stride=1, padding=k // 2))
        self.proj = nn.Conv2d(len(kernel), 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        conv_output = torch.cat([conv(out) for conv in self.convs], dim=1)
        out = self.proj(conv_output)
        out = self.sigmoid(out)
        out = x * out
        return out


class CTA(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.qkv_conv = nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(in_channels=hidden_dim * 3, out_channels=hidden_dim * 3,
                                    kernel_size=3, stride=1, padding=1, groups=hidden_dim * 3)
        self.proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, C, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv_conv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')
        dots = (q @ k.transpose(-2, -1))  # b, c, c
        attn = self.softmax(dots)
        out = attn @ v  # b, c, hw
        out = rearrange(out, 'b c (h w) -> b c h w', h=h, w=w)
        out = self.proj(out)
        return out


class PTA(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.qkv_conv = nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(in_channels=hidden_dim * 3, out_channels=hidden_dim * 3,
                                    kernel_size=3, stride=1, padding=1, groups=hidden_dim * 3)
        self.proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, h_patch_size=160, w_patch_size=160):
        b, C, h, w = x.shape

        if h < h_patch_size:
            h_patch_size = h
        if w < w_patch_size:
            w_patch_size = w


        pad_h = (h_patch_size - h % h_patch_size) % h_patch_size
        pad_w = (w_patch_size - w % w_patch_size) % w_patch_size
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))


        qkv = self.qkv_dwconv(self.qkv_conv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b c (h_n h_ps) (w_n w_ps) -> (b h_n w_n) (h_ps w_ps) c',
                      h_n=(h + pad_h) // h_patch_size, w_n=(w + pad_w) // w_patch_size,
                      h_ps=h_patch_size, w_ps=w_patch_size)
        k = rearrange(k, 'b c (h_n h_ps) (w_n w_ps) -> (b h_n w_n) (h_ps w_ps) c',
                      h_n=(h + pad_h) // h_patch_size, w_n=(w + pad_w) // w_patch_size,
                      h_ps=h_patch_size, w_ps=w_patch_size)
        v = rearrange(v, 'b c (h_n h_ps) (w_n w_ps) -> (b h_n w_n) (h_ps w_ps) c',
                      h_n=(h + pad_h) // h_patch_size, w_n=(w + pad_w) // w_patch_size,
                      h_ps=h_patch_size, w_ps=w_patch_size)
        dots = (q @ k.transpose(-2, -1))
        attn = self.softmax(dots)
        out = attn @ v
        out = rearrange(out, '(b h_n w_n) (h_ps w_ps) c -> b c (h_n h_ps) (w_n w_ps)',
                        h_n=(h + pad_h) // h_patch_size, w_n=(w + pad_w) // w_patch_size,
                        h_ps=h_patch_size, w_ps=w_patch_size)
        out = self.proj(out)


        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return out


def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class SENet(nn.Module):
    def __init__(self, num_feat, reduction_factor=30):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_downsample = nn.Conv2d(
            num_feat, num_feat // reduction_factor, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU(inplace=True)
        self.conv_upsample = nn.Conv2d(
            num_feat // reduction_factor, num_feat, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # b, c, h, w
        y = self.avg_pool(x)  # b, c, 1, 1
        y = self.conv_downsample(y)  # b, c // reduction_factor, 1, 1
        y = self.act(y)  # b, c // reduction_factor, 1, 1
        y = self.conv_upsample(y)  # b, c, 1, 1
        y = self.sigmoid(y)  # b, c, 1, 1
        x = x * y  # b, c, h, w
        return x


class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, reduction_factor=30):
        super().__init__()
        self.conv_compress = nn.Conv2d(
            num_feat, num_feat // compress_ratio, kernel_size=3, stride=1, padding=1)
        self.act = nn.GELU()
        self.conv_uncompress = nn.Conv2d(
            num_feat // compress_ratio, num_feat, kernel_size=3, stride=1, padding=1)
        self.attention = SENet(num_feat=num_feat, reduction_factor=reduction_factor)

    def forward(self, x):  # b, c, h, w
        x = self.conv_compress(x)  # b, c // compress_factor, h, w
        x = self.act(x)  # b, c // compress_factor, h, w
        x = self.conv_uncompress(x)  # b, c, h, w
        x = self.attention(x)  # b, c, h, w
        return x


class ResidualChannelAttentionGroup(nn.Module):
    def __init__(self, num_feat, num_block):
        super().__init__()
        self.residual_group = make_layer(ResidualChannelAttentionBlock, num_block,
                                         num_feat=num_feat, compress_ratio=1, reduction_factor=16)
        self.conv = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = self.conv(self.residual_group(x))
        x = res + x
        return x


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat, num_grow_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class ResidualMLP(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.resmlp = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        return self.resmlp(x)


class UNet(nn.Module):
    def __init__(self, num_in_ch, num_out_ch=1, num_feat=64, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1)
        self.conv2 = nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1)
        self.conv3 = nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1)
        # upsample
        self.conv4 = nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1)
        self.conv6 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        # extra convolutions
        self.conv7 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv8 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv9 = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


class EncoderHR(nn.Module):
    def __init__(self, in_chans_gt, out_chans, feats=16, scale=4):
        super().__init__()
        in_chans = in_chans_gt + in_chans_gt * (scale ** 2)
        self.pixel_unshuffle = nn.PixelUnshuffle(scale)

        # Degradation
        self.D = nn.Sequential(
            nn.Conv2d(in_chans, feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            ResidualChannelAttentionGroup(num_feat=feats, num_block=9),

            ResidualDenseBlock(num_feat=feats, num_grow_ch=feats // 2),
            ResidualDenseBlock(num_feat=feats, num_grow_ch=feats // 2),
            ResidualDenseBlock(num_feat=feats, num_grow_ch=feats // 2))

        # Content
        self.C = nn.Sequential(
            nn.Conv2d(in_chans_gt, feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            ResidualChannelAttentionGroup(num_feat=feats, num_block=9),

            ResidualDenseBlock(num_feat=feats, num_grow_ch=feats // 2),
            ResidualDenseBlock(num_feat=feats, num_grow_ch=feats // 2),
            ResidualDenseBlock(num_feat=feats, num_grow_ch=feats // 2))

        self.conv = nn.Conv2d(feats + feats * (scale ** 2), out_chans, kernel_size=3, stride=1, padding=1)

    def forward(self, x, gt):
        gt_down = self.pixel_unshuffle(gt)
        x = torch.cat([x, gt_down], dim=1)
        x1 = self.D(x)
        x2 = self.C(gt)
        x2 = self.pixel_unshuffle(x2)
        x = self.conv(torch.cat([x1, x2], dim=1))
        return x


class EncoderLR(nn.Module):
    def __init__(self, in_chans_lq, out_chans, feats=16):
        super().__init__()

        # Extraction
        self.E = nn.Sequential(
            nn.Conv2d(in_chans_lq, feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            ResidualChannelAttentionGroup(num_feat=feats, num_block=9),

            ResidualDenseBlock(num_feat=feats, num_grow_ch=feats // 2),
            ResidualDenseBlock(num_feat=feats, num_grow_ch=feats // 2),
            ResidualDenseBlock(num_feat=feats, num_grow_ch=feats // 2))

        self.conv = nn.Conv2d(feats, out_chans, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.E(x)
        x = self.conv(x)
        return x


class Denoise(nn.Module):
    def __init__(self, feats, timesteps=4):
        super().__init__()
        self.max_period = timesteps * 10.

        self.unet = UNet(num_in_ch=feats * 2 + 1, num_out_ch=feats)

    def forward(self, x, t, condition):
        t = ((t.float()) / self.max_period).reshape(-1, 1, 1, 1)
        t = t.expand(-1, 1, x.shape[2], x.shape[3])
        x = self.unet(torch.cat([x, t, condition], dim=1))
        return x


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
        betas = linear_end * torch.ones(n_timestep, dtype=np.float64)
        warmup_time = int(n_timestep * warmup_frac)
        betas[:warmup_time] = torch.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
        return betas

    if schedule == 'linear':
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == 'quad':
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac=0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac=0.5)
    elif schedule == 'const':
        betas = linear_end * torch.ones(n_timestep, dtype=torch.float64)
    elif schedule == 'jsd':
        betas = 1. / torch.linspace(n_timestep, 1, n_timestep, dtype=torch.float64)
    elif schedule == 'cosine':
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(min=0, max=0.999)
    else:
        raise NotImplementedError(schedule)
    betas = betas.numpy()
    return betas


class GaussianDiffusion(nn.Module):
    def __init__(self, denoise, condition, timesteps=1000, given_betas=None, beta_schedule='quad',
                 linear_start=0.1, linear_end=0.99, cosine_s=8e-3,
                 v_posterior=0., clip_denoised=False, feats=64, parameterization='x0'):
        super().__init__()
        self.denoise = denoise
        self.condition = condition
        self.v_posterior = v_posterior
        self.clip_denoised = clip_denoised
        self.feats = feats
        assert parameterization in ['eps', 'x0'], "Currently only supporting 'eps' and 'x0'."
        self.parameterization = parameterization  # all assuming fixed variance schedules

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

    def register_schedule(self, given_betas=None, beta_schedule='linear', timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                          extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, c, clip_denoised: bool):
        model_out = self.denoise(x, t, c)
        x_recon = None
        if self.parameterization == 'eps':
            x_recon = self.predict_start_from_noise(x_t=x, t=t, noise=model_out)
        elif self.parameterization == 'x0':
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        model_mean, _, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, _, posterior_log_variance

    def p_sample(self, x, t, c, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, c=c, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))  # no noise when t == 0
        model_mean = model_mean + noise * (0.5 * model_log_variance).exp() * nonzero_mask
        return model_mean

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def forward(self, img, x=None):
        device = self.betas.device
        b = img.shape[0]
        if self.training:
            assert exists(x), 'x must be defined for training.'
            t = torch.full((b,), self.num_timesteps - 1, device=device, dtype=torch.long)
            noise = torch.randn_like(x)
            x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
            condition = self.condition(x=img)
            cdp = x_noisy
            for i in reversed(range(0, self.num_timesteps)):
                cdp = self.p_sample(x=cdp, t=torch.full((b,), i, device=device, dtype=torch.long), c=condition,
                                    clip_denoised=self.clip_denoised)
            return cdp
        else:
            shape = (b, self.feats, img.shape[2], img.shape[3])
            x_noisy = torch.randn(shape, device=device)
            condition = self.condition(x=img)
            cdp = x_noisy
            for i in reversed(range(0, self.num_timesteps)):
                cdp = self.p_sample(x=cdp, t=torch.full((b,), i, device=device, dtype=torch.long), c=condition,
                                    clip_denoised=self.clip_denoised)
            return cdp


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cast_tuple(val, depth=1):
    return val if isinstance(val, tuple) else (val,) * depth


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)


class WindowsHandling(object):
    @staticmethod
    def unfold_partition(x, kernel_size, stride, padding=0):
        unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
        b, h, w, c = x.shape
        x = x.permute(0, 3, 1, 2)  # b, c, h, w
        x = unfold(x)  # b, c * kernel_size * kernel_size, nW
        x = x.permute(0, 2, 1).reshape(b, -1, kernel_size, kernel_size, c)
        return x

    @staticmethod
    def windows_partition(x, window_size):
        b, h, w, c = x.shape  # b, h, w, c
        x = x.reshape(b, h // window_size, window_size, w // window_size, window_size,
                      c)  # b, h // Ws, Ws, w // Ws, Ws, c
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # b, h // Ws, w // Ws, Ws, Ws, c
        windows = x.reshape(-1, window_size, window_size, c)  # b * nW, Ws, Ws, c = b_, Ws, Ws, c
        return windows

    @staticmethod
    def windows_reverse(windows, window_size, h, w):
        b_, _, _, _ = windows.shape  # b_, Ws, Ws, c
        b = b_ // ((h // window_size) * (w // window_size))  # b
        x = windows.reshape(b, h // window_size, w // window_size, window_size, window_size,
                            -1)  # b, h // Ws, w // Ws, Ws, Ws, c
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # b, h // Ws, Ws, w // Ws, Ws, c
        x = x.reshape(b, h, w, -1)  # b, h, w, c
        return x

    @staticmethod
    def get_relative_position_index(window_size):
        coords_h = torch.arange(window_size)  # Ws
        coords_w = torch.arange(window_size)  # Ws
        grid_h, gird_w = torch.meshgrid([coords_h, coords_w])  # Ws * Ws; Ws * Ws
        coords = torch.stack([grid_h, gird_w])  # 2, Ws, Ws
        coords_flatten = torch.flatten(coords, start_dim=1)  # 2, Ws * Ws
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Ws * Ws, Ws * Ws
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Ws * Ws, Ws * Ws, 2
        relative_coords[:, :, 0] += (window_size - 1)  # Shift to start from 0.
        relative_coords[:, :, 1] += (window_size - 1)  # Shift to start from 0.
        relative_coords[:, :, 0] *= (2 * window_size - 1)  # Make sure to map to different index.
        relative_position_index = relative_coords.sum(dim=-1)  # Ws * Ws, Ws * Ws
        return relative_position_index

    @staticmethod
    def get_relative_position_index_for_overlap_window(window_size, overlap_win_size):
        window_size_ori = window_size
        window_size_ext = overlap_win_size
        coords_h = torch.arange(window_size_ori)  # Ws
        coords_w = torch.arange(window_size_ori)  # Ws
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Ws, Ws
        coords_ori_flatten = torch.flatten(coords_ori, start_dim=1)  # 2, Ws * Ws
        coords_h = torch.arange(window_size_ext)  # oWs
        coords_w = torch.arange(window_size_ext)  # oWs
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, oWs, oWs
        coords_ext_flatten = torch.flatten(coords_ext, start_dim=1)  # 2, oWs * oWs
        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]  # 2, Ws * Ws, oWs * oWs
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Ws * Ws, oWs * oWs, 2
        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1  # Shift to start from 0.
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1  # Shift to start from 0.
        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1  # Make sure to map to different index.
        relative_position_index = relative_coords.sum(dim=-1)  # Ws * Ws, oWs * oWs
        return relative_position_index

    @staticmethod  # swin
    def get_attention_mask(input_resolution, window_size, shift_size):
        h, w = input_resolution  # h, w
        img_mask = torch.zeros(size=(1, h, w, 1))  # 1, h, w, 1
        h_slices = w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        # 0:-Ws:1, -Ws:-Ss:1, -Ss::1

        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = count  # 1, h, w, 1
                count += 1

        img_mask_windows = WindowsHandling.windows_partition(x=img_mask, window_size=window_size)  # nW, Ws, Ws, 1
        img_mask_windows = img_mask_windows.reshape(-1, window_size * window_size)  # nW, Ws * Ws

        attn_mask = img_mask_windows[:, None, :] - img_mask_windows[:, :, None]  # nW, Ws * Ws, Ws * Ws
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.)).masked_fill(attn_mask == 0, float(0.))
        return attn_mask

    @staticmethod  # sowin
    def get_attention_mask_for_overlap_window(input_resolution, window_size, overlap_win_size, shift_size):
        overlap_pad_size = (overlap_win_size - window_size) // 2

        h, w = input_resolution  # h, w
        ph, pw = h + overlap_pad_size * 2, w + overlap_pad_size * 2  # ph, pw

        img_mask = torch.zeros(size=(1, h, w, 1))  # 1, h, w, 1
        img_pad_mask = torch.zeros(size=(1, ph, pw, 1))  # 1, ph, pw, 1

        if shift_size > 0:
            # cross-boundary mask
            h_slices = w_slices = (
                slice(0, -shift_size - overlap_pad_size), slice(-shift_size - overlap_pad_size, None))
            # 0:-Ss:1, -Ss::1

            count = 0
            for h_slice in h_slices:
                for w_slice in w_slices:
                    img_pad_mask[:, h_slice, w_slice, :] = count
                    count += 1
            img_mask[...] = img_pad_mask[:, overlap_pad_size:-overlap_pad_size, overlap_pad_size:-overlap_pad_size, :]

            img_mask_windows = WindowsHandling.windows_partition(img_mask, window_size)  # nW, Ws, Ws, 1
            img_mask_windows = img_mask_windows.reshape(-1, window_size * window_size)  # nW, Ws * Ws
            k_mask_windows = WindowsHandling.unfold_partition(
                x=img_pad_mask, kernel_size=overlap_win_size, stride=window_size)  # 1, nW, oWs, oWs, 1
            k_mask_windows = k_mask_windows.reshape(-1, overlap_win_size * overlap_win_size)  # nW, oWs * oWs
            attn_mask = img_mask_windows.unsqueeze(2) - k_mask_windows.unsqueeze(1)  # nW, Ws * Ws, oWs * oWs
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.)).masked_fill(attn_mask == 0, float(0.))

        else:
            # edge-extension mask
            valid_mask = torch.ones((1, h, w, 1))
            q_valid_mask_windows = WindowsHandling.windows_partition(valid_mask, window_size)  # nW, Ws, Ws, 1
            q_valid_mask_windows = q_valid_mask_windows.reshape(-1, window_size * window_size)  # nW, Ws * Ws
            k_valid_mask_windows = WindowsHandling.unfold_partition(
                x=valid_mask, kernel_size=overlap_win_size, stride=window_size,
                padding=overlap_pad_size)  # 1, nW, oWs, oWs, 1
            k_valid_mask_windows = k_valid_mask_windows.reshape(-1, overlap_win_size * overlap_win_size)  # nW, oWs*oWs
            attn_mask = q_valid_mask_windows.unsqueeze(2) * k_valid_mask_windows.unsqueeze(1)  # nW, Ws * Ws, oWs*oWs
            attn_mask = attn_mask.masked_fill((attn_mask == 0), float(-100.)).masked_fill(attn_mask == 1, float(0.))

        return attn_mask


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x, x_size=None):
        b, c, h, w = x.shape  # b, c, h, w
        if not exists(x_size):
            assert h == self.img_size[0] and w == self.img_size[1], \
                f"Input image size ({h} * {w}) doesn't match model ({self.img_size[0]} * {self.img_size[1]})."
        else:
            assert h == x_size[0] and w == x_size[1], \
                f"Input image size ({h} * {w}) doesn't match model ({x_size[0]} * {x_size[1]})."

        x = x.flatten(2).transpose(1, 2)  # b, h * w, c
        if exists(self.norm):
            x = self.norm(x)  # b, h * w, c
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size=None):
        b, l, c = x.shape  # b, l, c = b, h * w, c
        if not exists(x_size):
            assert l == self.img_size[0] * self.img_size[1], \
                f"Input size {l} doesn't match model ({self.img_size[0]} * {self.img_size[1]})."
        else:
            assert l == x_size[0] * x_size[1], \
                f"Input size {l} doesn't match model ({x_size[0]} * {x_size[1]})."

        if not exists(x_size):
            x = x.transpose(1, 2).reshape(b, c, self.img_size[0], self.img_size[1])  # b, c, h, w
        else:
            x = x.transpose(1, 2).reshape(b, c, x_size[0], x_size[1])  # b, c, h, w
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.norm_layer = norm_layer(4 * dim)
        self.redution = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        h, w = self.input_resolution  # h, w
        b, l, c = x.shape  # b, h * w, c
        assert l == h * w, "Input feature x has wrong size."
        assert h % 2 == 0 and w % 2 == 0, f"Input x size ({h} * {w}) are not even."
        x = x.reshape(b, h, w, c)  # b, h, w, c
        # Feature x is divided into 4 parts, as follows:
        #   x0,     x2,     x0,     x2,     ...
        #   x1,     x3,     x1,     x3,     ...
        #   x0,     x2,     x0,     x2,     ...
        #   x1,     x3,     x1,     x3,     ...
        #   ...,    ...,    ...,    ...,    ...
        x0 = x[:, 0::2, 0::2, :]  # b, h // 2, w // 2, c
        x1 = x[:, 1::2, 0::2, :]  # b, h // 2, w // 2, c
        x2 = x[:, 0::2, 1::2, :]  # b, h // 2, w // 2, c
        x3 = x[:, 1::2, 1::2, :]  # b, h // 2, w // 2, c
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # b, h // 2, w // 2, c * 4
        x = x.reshape(b, -1, 4 * c)  # b, (h // 2) * (w // 2), c * 4
        x = self.norm_layer(x)  # b, (h // 2) * (w // 2), c * 4
        x = self.redution(x)  # b, (h // 2) * (w // 2), c * 2
        return x


class PixelUnShuffle(nn.Module):
    def __init__(self, downscale, num_feat):
        super().__init__()

        m = []
        if (downscale & (downscale - 1)) == 0:  # downscale = 2^n (n is a natural number). e.g., 1, 2, 4, 8, ...
            num_pixelshuffle = int(math.log(downscale, 2))
            for _ in range(1, num_pixelshuffle + 1):
                m.append(nn.PixelUnshuffle(downscale_factor=2))
                m.append(nn.Conv2d(num_feat * 4, num_feat, kernel_size=3, stride=1, padding=1))
        elif downscale == 3:  # When downscale = 3.
            m.append(nn.PixelUnshuffle(downscale_factor=3))
            m.append(nn.Conv2d(num_feat * 9, num_feat, kernel_size=3, stride=1, padding=1))
        else:
            raise ValueError(f"Scale factor {downscale} is not supported. Supported scaling factors are 2^n or 3.")
        self.pixelshuffle = nn.Sequential(*m)

    def forward(self, x):
        x = self.pixelunshuffle(x)
        return x


class PixelShuffle(nn.Module):
    def __init__(self, upscale, num_feat):
        super().__init__()

        m = []
        if (upscale & (upscale - 1)) == 0:  # upscale = 2^n (n is a natural number). e.g., 1, 2, 4, 8, ...
            num_pixelshuffle = int(math.log(upscale, 2))
            for _ in range(1, num_pixelshuffle + 1):
                m.append(nn.Conv2d(num_feat, num_feat * 4, kernel_size=3, stride=1, padding=1))
                m.append(nn.PixelShuffle(upscale_factor=2))
        elif upscale == 3:  # When upscale = 3.
            m.append(nn.Conv2d(num_feat, num_feat * 9, kernel_size=3, stride=1, padding=1))
            m.append(nn.PixelShuffle(upscale_factor=3))
        else:
            raise ValueError(f"Scale factor {upscale} is not supported. Supported scaling factors are 2^n or 3.")
        self.pixelshuffle = nn.Sequential(*m)

    def forward(self, x):
        x = self.pixelshuffle(x)
        return x


class PixelShuffleDirect(nn.Module):
    def __init__(self, upscale, num_feat, out_chans):
        super().__init__()

        m = []
        m.append(nn.Conv2d(num_feat, (upscale ** 2) * out_chans, kernel_size=3, stride=1, padding=1))
        m.append(nn.PixelShuffle(upscale_factor=upscale))
        self.pixelshuffle = nn.Sequential(*m)

    def forward(self, x):
        x = self.pixelshuffle(x)
        return x


class WindowsAttention(nn.Module):
    def __init__(self,
                 dim, num_heads, window_size,
                 qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Note: The data type of Ws is int, not tuple.
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position_bias_table = nn.Parameter(torch.zeros(
            (2 * window_size - 1) * (2 * window_size - 1), num_heads))  # (2 * Ws - 1) * (2 * Ws - 1), nH
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.gating = nn.Linear(dim, dim)
        nn.init.constant_(self.gating.weight, 0.)
        nn.init.constant_(self.gating.bias, 1.)

    def forward(self, x, rpi, attn_mask=None):
        gates = self.gating(x)

        b_, n, c = x.shape  # nW * b, Ws * Ws, c = b_, n, c
        qkv = self.qkv(x)  # b_, n, 3 * c
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)  # 3, b_, nH, n, c // nH
        q, k, v = qkv[0], qkv[1], qkv[2]  # b_, nH, n, c // nH
        q = q * self.scale  # b_, nH, n, c // nH
        attn = (q @ k.transpose(-2, -1))  # b_, nH, n, n

        relative_position_index = rpi.reshape(-1)
        relative_position_bias = self.relative_position_bias_table[relative_position_index].reshape(
            self.window_size * self.window_size, self.window_size * self.window_size,
            -1)  # Ws * Ws, Ws * Ws, nH = n, n, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Ws * Ws, Ws * Ws = nH, n, n
        relative_position_bias = relative_position_bias.unsqueeze(0)  # 1, nH, Ws * Ws, Ws * Ws = 1, nH, n, n

        attn = attn + relative_position_bias  # b_, nH, n, n
        if exists(attn_mask):
            num_windows, _, _ = attn_mask.shape  # nW, n, n
            attn_mask = attn_mask[None, :, None, :, :]  # 1, nW, 1, n, n
            attn = attn.reshape(b_ // num_windows, num_windows, self.num_heads, n, n) + attn_mask  # b, nW, nH, n, n
            attn = attn.reshape(-1, self.num_heads, n, n)  # b * nW, nH, n, n = b_, nH, n, n

        attn = self.softmax(attn)  # b_, nH, n, n
        attn = self.attn_drop(attn)  # b_, nH, n, n

        x = (attn @ v)  # b_, nH, n, c // nH
        x = x.transpose(1, 2).reshape(b_, n, c)  # b_, n, c

        x = x * gates.sigmoid()

        x = self.proj(x)  # b_, n, c
        x = self.proj_drop(x)  # b_, n, c
        return x


# S(W)-MSA
class SwinTransformerBlock(nn.Module):
    def __init__(self,
                 dim, input_resolution,
                 num_heads, window_size,
                 shift_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 group_idx=None,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # h, w
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # If window size is larger than input resolution, we don't partition windows.
        if min(input_resolution) < window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowsAttention(
            dim=dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        if use_cdp:
            self.cdem1 = CDEM(dim=dim, num_feat=cdp_feat)

        self.conv_scale = 1
        self.csam = CSAM(dim=dim,
                         compress_ratio=kwargs['compress_ratio'],
                         reduction_factor=kwargs['reduction_factor'],
                         kernel=[3, 5, 7])

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_ratio = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_ratio, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, rpi, attn_mask, cdp):
        b, l, c = x.shape  # b, l, c = b, h * w, c
        h, w = x_size
        assert h * w == l, "input feature has wrong size"

        short_cut = x  # b, l, c
        x = self.norm1(x)  # b, l, c
        if use_cdp:
            x = self.cdem1(x, cdp)
        x = x.reshape(b, h, w, c)  # b, h, w, c

        csa = self.csam(rearrange(x, 'b h w c -> b c h w'))

        if self.shift_size > 0:
            shifted_x = torch.roll(input=x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))  # b, h, w, c
            attn_mask = attn_mask
        else:
            shifted_x = x  # b, h, w, c
            attn_mask = None

        sparse_window = False
        interval = 2
        gh, gw = h // interval, w // interval
        if sparse_window:
            shifted_x = shifted_x.reshape(b, gh, interval, gw, interval, c).permute(0, 2, 4, 1, 3, 5).contiguous()
            shifted_x = shifted_x.reshape(b * interval * interval, gh, gw, c)
            attn_mask = None

        x_windows = WindowsHandling.windows_partition(shifted_x, window_size=self.window_size)  # b_, Ws, Ws, c
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, c)  # b_, Ws * Ws, c = b_, n, c

        # W-MSA or SW-MSA
        attn_windows = self.attn(x_windows, rpi=rpi, attn_mask=attn_mask)  # b_, n, c = b_, Ws * Ws, c

        # merge windows
        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, c)  # b_, Ws, Ws, c
        if not sparse_window:
            shifted_x = WindowsHandling.windows_reverse(windows=attn_windows, window_size=self.window_size,
                                                        h=h, w=w)  # b, h, w, c

        if sparse_window:
            shifted_x = WindowsHandling.windows_reverse(windows=attn_windows, window_size=self.window_size,
                                                        h=gh, w=gw)  # b * interval * interval, gh, gw, c
            shifted_x = shifted_x.reshape(b, interval, interval, gh, gw, c).permute(0, 3, 1, 4, 2, 5).contiguous()
            shifted_x = shifted_x.reshape(b, h, w, c)

        if self.shift_size > 0:
            x = torch.roll(input=shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))  # b, h, w, c
        else:
            x = shifted_x  # b, h, w, c
        x = x.reshape(b, h * w, c)  # b, h * w, c = b, l, c
        x = short_cut + self.drop_path(x)  # b, l, c

        x = x + csa * self.conv_scale

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# LEHAB
class LocalEnhancedHybirdAttentionTransformerBlock(nn.Module):
    def __init__(self,
                 dim, input_resolution,
                 depth, num_heads, window_size,
                 mlp_ratio, qkv_bias, qk_scale,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, use_checkpoint=False,
                 compress_ratio=3, reduction_factor=30, conv_scale=0.01,
                 group_idx=None):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2) == 0 else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, act_layer=act_layer,
                compress_ratio=compress_ratio, reduction_factor=reduction_factor, conv_scale=conv_scale,
                group_idx=group_idx)
            self.blocks.append(block)

    def forward(self, x, x_size, rpi, attn_mask, cdp):
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(block, x, x_size, rpi, attn_mask, cdp)
            else:
                x = block(x, x_size, rpi, attn_mask, cdp)
        return x


class OverlapWindowAttention(nn.Module):
    def __init__(self,
                 overlap_ratio,
                 dim,
                 num_heads, window_size,
                 shift_size,
                 qkv_bias, qk_scale,
                 attn_drop, proj_drop):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Note: The data type of Ws is int, not tuple.
        self.overlap_win_size = window_size + int(window_size * overlap_ratio)
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.overlap_pad_size = (self.overlap_win_size - self.window_size) // 2
        if shift_size > 0:
            assert self.shift_size >= self.overlap_pad_size, 'shift_size must be greater than overlap pad size'

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.unfold = nn.Unfold(kernel_size=self.overlap_win_size,
                                stride=window_size,
                                padding=0)

        self.relative_position_bias_table = nn.Parameter(torch.zeros(
            (window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1),
            num_heads))  # (Ws + oWs - 1) * (Ws + oWs - 1), nH
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.gating = nn.Linear(dim, dim)
        nn.init.constant_(self.gating.weight, 0.)
        nn.init.constant_(self.gating.bias, 1.)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rpi, attn_mask=None):
        b, h, w, c = x.shape

        if self.shift_size > 0:
            # pad, shift and drop
            pad_size = self.shift_size + self.overlap_pad_size
            drop_size = self.shift_size - self.overlap_pad_size
            x_pad = torch.zeros(b, h + pad_size, w + pad_size, c, device=x.device)
            x_pad[:, 0:h, 0:w, :] = x
            x_pad[:, -pad_size:, :, :] = x_pad[:, 0:pad_size, :, :]
            x_pad[:, :, -pad_size:, :] = x_pad[:, :, 0:pad_size, :]
            x_pad = x_pad[:, drop_size:, drop_size:, :]  # drop left, drop top
            shifted_x = x_pad[:, self.overlap_pad_size:-self.overlap_pad_size,
                        self.overlap_pad_size:-self.overlap_pad_size, :]
        else:
            # pad
            pad_size = self.overlap_pad_size * 2
            x_pad = torch.zeros(b, h + pad_size, w + pad_size, c, device=x.device)
            x_pad[:, self.overlap_pad_size:-self.overlap_pad_size, self.overlap_pad_size:-self.overlap_pad_size, :] = x
            shifted_x = x

        b, ph, pw, c = x_pad.shape
        x_windows = WindowsHandling.windows_partition(shifted_x, window_size=self.window_size)  # b * nW, Ws, Ws, c
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, c)  # b * nW, Ws * Ws, c
        x_pad_windows = self.unfold(x_pad.permute(0, 3, 1, 2))  # b, c * oWs * oWs, nW
        x_pad_windows = rearrange(x_pad_windows,
                                  pattern='b (c oWh oWw) nW -> (b nW) (oWh oWw) c',
                                  oWh=self.overlap_win_size, oWw=self.overlap_win_size
                                  ).contiguous()  # b * nW, oWs * oWs, c

        gates = self.gating(x_windows)

        q_windows = self.q(x_windows)  # b * nW, Ws * Ws, c
        kv_windows = self.kv(x_pad_windows)  # b * nW, oWs * oWs, 2 * c

        b_, nq, _ = q_windows.shape  # b_ = b * nW, nq = Ws * Ws
        _, nkv, _ = kv_windows.shape  # nkv = oWs * oWs
        d = self.head_dim

        kv_windows = kv_windows.reshape(b_, nkv, 2, c).permute(2, 0, 1, 3)  # 2, b * nW, oWs * oWs, c
        k_windows, v_windows = kv_windows[0], kv_windows[1]  # b * nW, oWs * oWs, c

        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3)  # b * nW, nH, nq, d
        k = k_windows.reshape(b_, nkv, self.num_heads, d).permute(0, 2, 1, 3)  # b * nW, nH, nkv, d
        v = v_windows.reshape(b_, nkv, self.num_heads, d).permute(0, 2, 1, 3)  # b * nW, nH, nkv, d

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_index = rpi.reshape(-1)
        relative_position_bias = self.relative_position_bias_table[relative_position_index].reshape(
            self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size,
            -1)  # Ws * Ws, oWs * oWs, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Ws * Ws, oWs * oWs
        attn = attn + relative_position_bias.unsqueeze(0)

        if exists(attn_mask):  # nW, Ws * Ws, oWs * oWs
            nW = attn_mask.shape[0]
            attn = attn.reshape(b_ // nW, nW, self.num_heads, nq, nkv) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape(-1, self.num_heads, nq, nkv)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

        attn_windows = attn_windows * gates.sigmoid()

        attn_windows = self.proj(attn_windows)
        attn_windows = self.proj_drop(attn_windows)

        # attn window reverse
        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, c)
        shifted_x = WindowsHandling.windows_reverse(windows=attn_windows, window_size=self.window_size,
                                                    h=h, w=w)  # b, h, w, c

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.reshape(b, h * w, c)  # b, h * w, c = b, l, c
        return x


# SO(W)-MSA
class SowinTransformerBlock(nn.Module):
    def __init__(self,
                 overlap_ratio,
                 dim, input_resolution,
                 num_heads, window_size,
                 shift_size,
                 mlp_ratio, qkv_bias, qk_scale,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 group_idx=None,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.overlap_win_size = window_size + int(window_size * overlap_ratio)
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # If window size is larger than input resolution, we don't partition windows.
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = OverlapWindowAttention(
            overlap_ratio=overlap_ratio,
            dim=dim, num_heads=num_heads, window_size=window_size,
            shift_size=shift_size,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        if use_cdp:
            self.cdem1 = CDEM(dim=dim, num_feat=cdp_feat)

        self.conv_scale = 1
        self.gtam = GTAM(dim=dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, rpi, attn_mask, cdp):
        b, l, c = x.shape  # b, l, c = b, h * w, c
        h, w = x_size
        assert h * w == l, "input feature has wrong size"

        short_cut = x
        x = self.norm1(x)
        if use_cdp:
            x = self.cdem1(x, cdp)
        x = x.reshape(b, h, w, c)

        gta = self.gtam(rearrange(x, 'b h w c -> b c h w'))  # b, l, c

        if self.shift_size > 0:
            attn_mask = attn_mask[1]
        else:
            attn_mask = attn_mask[0]

        x = self.attn(x, rpi=rpi, attn_mask=attn_mask)
        x = short_cut + self.drop_path(x)  # b, l, c

        x = x + gta * self.conv_scale

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# GEHAB
class GlobalEnhancedHybirdAttentionTransformerBlock(nn.Module):
    def __init__(self,
                 overlap_ratio,
                 depth, dim, input_resolution,
                 num_heads, window_size,
                 mlp_ratio, qkv_bias, qk_scale,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, use_checkpoint=False,
                 group_idx=None):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = SowinTransformerBlock(
                overlap_ratio=overlap_ratio,
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                norm_layer=norm_layer, act_layer=act_layer,
                group_idx=group_idx)
            self.blocks.append(block)

    def forward(self, x, x_size, rpi, attn_mask, cdp):
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(block, x, x_size, rpi, attn_mask, cdp)
            else:
                x = block(x, x_size, rpi, attn_mask, cdp)
        return x


class BasicBlock(nn.Module):
    def __init__(self,
                 img_size, patch_size,
                 dim, input_resolution,
                 depth, num_heads, window_size,
                 mlp_ratio, qkv_bias, qk_scale,
                 drop_rate, attn_drop_rate, drop_path_rate,
                 norm_layer, use_checkpoint,
                 compress_ratio, reduction_factor, conv_scale,
                 overlap_ratio,
                 group_idx):
        super().__init__()

        # LEHAB
        self.lehab = LocalEnhancedHybirdAttentionTransformerBlock(
            dim=dim, input_resolution=input_resolution,
            depth=depth, num_heads=num_heads, window_size=window_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
            norm_layer=norm_layer, act_layer=nn.GELU, use_checkpoint=use_checkpoint,
            compress_ratio=compress_ratio, reduction_factor=reduction_factor, conv_scale=conv_scale,
            group_idx=group_idx)

        # GEHAB
        self.gehab = GlobalEnhancedHybirdAttentionTransformerBlock(
            overlap_ratio=overlap_ratio,
            dim=dim, input_resolution=input_resolution,
            depth=2, num_heads=num_heads, window_size=window_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
            norm_layer=norm_layer, act_layer=nn.GELU, use_checkpoint=use_checkpoint,
            group_idx=group_idx)

    def forward(self, x, x_size, params, cdp=None):
        x = self.lehab(x, x_size=x_size, rpi=params['rpi_swsa'], attn_mask=params['attn_mask_swsa'], cdp=cdp)
        x = self.gehab(x, x_size=x_size, rpi=params['rpi_sowsa'], attn_mask=params['attn_mask_sowsa'], cdp=cdp)
        return x


class DeepResidualAttentionGroup(nn.Module):
    def __init__(self,
                 img_size, patch_size,
                 dim, input_resolution,
                 depth, num_heads, window_size,
                 mlp_ratio, qkv_bias, qk_scale,
                 drop_rate, attn_drop_rate, drop_path_rate,
                 norm_layer, use_checkpoint,
                 compress_ratio, reduction_factor, conv_scale,
                 overlap_ratio,
                 resi_connection,
                 group_idx):
        super().__init__()

        self.basic_block = BasicBlock(
            img_size=img_size, patch_size=patch_size,
            dim=dim, input_resolution=input_resolution,
            depth=depth, num_heads=num_heads, window_size=window_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_layer=norm_layer, use_checkpoint=use_checkpoint,
            compress_ratio=compress_ratio, reduction_factor=reduction_factor, conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            group_idx=group_idx)

        self.patch_unembed = PatchUnEmbed(img_size=img_size,
                                          patch_size=patch_size,
                                          in_chans=0,
                                          embed_dim=dim)

        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=0,
                                      embed_dim=dim,
                                      norm_layer=None)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, kernel_size=3, stride=1, padding=1))
        elif resi_connection == 'identity':
            self.conv = nn.Identity()

    def forward(self, x, x_size, params, cdp=None):
        y = self.basic_block(x, x_size=x_size, params=params, cdp=cdp)
        y = self.patch_unembed(y, x_size)
        y = self.conv(y)
        y = self.patch_embed(y, x_size)
        x = x + y
        return x


@ARCH_REGISTRY.register()
class CDEHAT(nn.Module):
    def __init__(self,
                 img_size, patch_size, in_chans, out_chans, num_feat,
                 embed_dim, depths, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, ape=False, use_checkpoint=False,
                 compress_ratio=3, reduction_factor=30, conv_scale=0.01, overlap_ratio=0.5,
                 upscale=4,
                 resi_connection='1conv', downsampler="patchmerging", upsampler="pixelshuffle",
                 **kwargs):
        super().__init__()

        # Initialize the incoming parameters.
        self.img_size = img_size
        self.patch_size = patch_size = 1
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_feat = num_feat
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.patch_norm = patch_norm
        self.ape = ape
        self.use_checkpoint = use_checkpoint
        self.compress_ratio = compress_ratio
        self.reduction_factor = reduction_factor
        self.conv_scale = conv_scale
        self.overlap_ratio = overlap_ratio
        self.upscale = upscale
        self.resi_connection = resi_connection
        self.downsampler = downsampler
        self.upsampler = upsampler

        num_in_ch = in_chans
        num_out_ch = out_chans
        num_feat = 64

        mean = kwargs.get('mean', [0.0] * num_in_ch)
        std = kwargs.get('std', [1.0] * num_in_ch)
        self.mean = torch.Tensor(mean).reshape(1, num_in_ch, 1, 1)
        self.std = torch.Tensor(std).reshape(1, num_in_ch, 1, 1)

        # relative position index
        self.overlap_win_size = window_size + int(window_size * overlap_ratio)
        relative_position_index_swsa = WindowsHandling.get_relative_position_index(
            window_size=window_size)
        relative_position_index_sowsa = WindowsHandling.get_relative_position_index_for_overlap_window(
            window_size=window_size, overlap_win_size=self.overlap_win_size)
        self.register_buffer('relative_position_index_swsa', relative_position_index_swsa)
        self.register_buffer('relative_position_index_sowsa', relative_position_index_sowsa)

        self.use_cdp = kwargs.get('use_cdp', False)
        self.cdp_feat = kwargs.get('cdp_feat', 3)
        global use_cdp
        global cdp_feat
        use_cdp = self.use_cdp
        cdp_feat = self.cdp_feat

        # TODO: CDP.
        if use_cdp:
            self.encoder_gt = EncoderHR(in_chans_gt=num_in_ch, out_chans=cdp_feat, scale=upscale).cuda()
            self.encoder_lq = EncoderLR(in_chans_lq=num_in_ch, out_chans=cdp_feat).cuda()  # condition
            self.denoise_fn = Denoise(feats=cdp_feat, timesteps=10).cuda()  # denoise
            self.diffusion = GaussianDiffusion(denoise=self.denoise_fn, condition=self.encoder_lq, timesteps=10,
                                               feats=cdp_feat).cuda()

        # TODO: Shallow feature extraction.
        self.conv_first = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)

        # TODO: Deep feature extraction.
        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=in_chans,
                                      embed_dim=embed_dim,
                                      norm_layer=norm_layer if patch_norm is True else None)
        self.patch_resolution = self.patch_embed.patches_resolution
        self.num_patches = self.patch_embed.num_patches

        self.patch_unembed = PatchUnEmbed(img_size=img_size,
                                          patch_size=patch_size,
                                          in_chans=embed_dim,
                                          embed_dim=embed_dim)

        # Using absolute position embedding.
        if self.ape is True:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # Stochastic depth decay rule.

        # Build layers.
        self.num_layers = len(depths)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = DeepResidualAttentionGroup(
                img_size=img_size, patch_size=patch_size,
                dim=embed_dim, input_resolution=self.patch_resolution,
                depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                compress_ratio=compress_ratio, reduction_factor=reduction_factor, conv_scale=conv_scale,
                overlap_ratio=overlap_ratio,
                resi_connection='1conv',
                group_idx=i_layer)
            self.layers.append(layer)

        self.norm = norm_layer(self.embed_dim)

        # Build the last conv layer in deep feature extraction.
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=3, stride=1, padding=1))
        elif resi_connection == 'identity':
            self.conv_after_body = nn.Identity()

        # TODO: High quality image reconstruction.
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.01, inplace=True))
            self.upsample = PixelShuffle(upscale=upscale, num_feat=num_feat)
            self.conv_last = nn.Conv2d(num_feat, out_chans, kernel_size=3, stride=1, padding=1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = PixelShuffleDirect(upscale=upscale, num_feat=embed_dim, out_chans=out_chans)
        else:
            self.conv_last = nn.Conv2d(embed_dim, out_chans, kernel_size=3, stride=1, padding=1)

        # Initialize weights.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x, gt=None, is_diffusion=True):
        self.mean, self.std = self.mean.type_as(x), self.std.type_as(x)
        x = (x - self.mean) / self.std

        # TODO: CDP.
        cdp, cdp_diff = None, None
        if use_cdp:
            if self.training:
                if exists(gt):
                    gt = (gt - self.mean) / self.std
                if is_diffusion is False:  # Stage 1: training
                    cdp = self.encoder_gt(x, gt)
                    cdp_diff = cdp
                else:  # Stage 2: training
                    self.encoder_gt.eval()
                    with torch.no_grad():
                        cdp = self.encoder_gt(x, gt)
                    cdp_diff = self.diffusion(x, cdp)
            else:
                if exists(gt):
                    gt = (gt - self.mean) / self.std
                if is_diffusion is False:  # Stage 1: test
                    cdp = self.encoder_gt(x, gt)
                    cdp_diff = cdp
                else:  # Stage 2: test
                    cdp_diff = self.diffusion(x)

        # TODO: Shallow feature extraction.
        x = self.conv_first(x)  # b, embed_dim, h, w
        # TODO: Deep feature extraction.
        y = self.forward_features(x, cdp_diff)  # b, embed_dim, h, w
        y = self.conv_after_body(y)  # b, embed_dim, h, w
        x = x + y  # b, embed_dim, h, w
        # TODO: High quality image reconstruction.
        x = self.conv_before_upsample(x)
        x = self.upsample(x)
        x = self.conv_last(x)

        x = (x * self.std) + self.mean
        if self.training:
            return x, cdp, cdp_diff
        else:
            return x

    def forward_features(self, x, cdp=None):
        # The use of the parameter x_size allows the model to infer images of
        # different sizes than the initialization parameter self.img_size.
        x_size = x.shape[-2:]

        attn_mask_swsa = WindowsHandling.get_attention_mask(
            input_resolution=x_size,
            window_size=self.window_size,
            shift_size=self.window_size // 2
        ).to(x.device)
        attn_mask_sowsa_without_shift = WindowsHandling.get_attention_mask_for_overlap_window(
            input_resolution=x_size,
            window_size=self.window_size,
            overlap_win_size=self.overlap_win_size,
            shift_size=0  # without shift
        ).to(x.device)
        attn_mask_sowsa_with_shift = WindowsHandling.get_attention_mask_for_overlap_window(
            input_resolution=x_size,
            window_size=self.window_size,
            overlap_win_size=self.overlap_win_size,
            shift_size=self.window_size // 2  # with shift
        ).to(x.device)
        params = {'rpi_swsa': self.relative_position_index_swsa,
                  'attn_mask_swsa': attn_mask_swsa,
                  'rpi_sowsa': self.relative_position_index_sowsa,
                  'attn_mask_sowsa': [attn_mask_sowsa_without_shift, attn_mask_sowsa_with_shift]}

        x = self.patch_embed(x, x_size)  # b, h * w, embed_dim
        if self.ape:
            x = x + self.absolute_pos_embed  # b, h * w, embed_dim
        x = self.pos_drop(x)  # b, h * w, embed_dim
        for layer in self.layers:
            x = layer(x, x_size=x_size, params=params, cdp=cdp)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)  # b, embed_dim, h, w
        return x

