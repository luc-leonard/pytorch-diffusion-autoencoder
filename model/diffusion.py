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
    def __init__(self, *args,n_classes=1, z_dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.class_embed = nn.Embedding(n_classes, z_dim)

    def p_mean_variance(self, x, t, clip_denoised: bool, class_id=None):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t, self.class_embed(class_id)))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, class_id=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, class_id=class_id)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, class_id):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), class_id=class_id)
        return img

    def p_losses(self, x_start, t, class_id=None, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, self.class_embed(class_id))

        if self.loss_type == "l1":
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == "l2":
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss


class AutoEncoderGaussianDiffusion(GaussianDiffusion):
    def __init__(self, *args, encoder_params, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_encoder = LatentEncoder(**encoder_params)

    def p_mean_variance(self, x, t, clip_denoised: bool, latent=None):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t, latent))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, latent=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, latent=latent)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, x):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        latent = self.latent_encoder(x.unsqueeze(0))
        print('sampling...')
        print('latent', latent.shape)
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), latent=latent)
        return img, latent

    def p_losses(self, x_start, t, class_id=None, noise=None):
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
