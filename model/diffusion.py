from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from denoising_diffusion_pytorch import GaussianDiffusion
from torch import nn
from tqdm import tqdm

from model.latent_encoder import LatentEncoder
from utils.config import exists, default


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class ClassConditionedGaussianDiffusion(GaussianDiffusion):
    def p_losses(self, x_start, t, class_embed=None, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, class_embed)

        if self.loss_type == "l1":
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == "l2":
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss


class AutoEncoderGaussianDiffusion(GaussianDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_encoder = LatentEncoder()

    def p_losses(self, x_start, t, class_embed=None, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_latent = self.latent_encoder(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, x_latent)

        if self.loss_type == "l1":
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == "l2":
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss


class PatchAutoEncoderGaussianDiffusion(GaussianDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_encoder = LatentEncoder()

    def p_losses(self, x_start, t, class_embed=None, noise=None):
        b, c, h, w = x_start.shape


        x_latent = self.latent_encoder(x_start)
        patch, position = self.patch_fn(x_start)

        noise = default(noise, lambda: torch.randn_like(patch))
        patch_noisy = self.q_sample(x_start=patch, t=t, noise=noise)
        patch_recon = self.denoise_fn(patch_noisy, t, x_latent)

        if self.loss_type == "l1":
            loss = (noise - patch_recon).abs().mean()
        elif self.loss_type == "l2":
            loss = F.mse_loss(noise, patch_recon)
        else:
            raise NotImplementedError()

        return loss
