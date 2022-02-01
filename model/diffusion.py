import random

import albumentations
import numpy as np
import torch
import torch.nn.functional as F
from albumentations import get_random_crop_coords
from denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import noise_like
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from utils.config import default


class AutoEncoderGaussianDiffusion(GaussianDiffusion):
    def __init__(self, *args, latent_encoder, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_encoder = latent_encoder

    def convert_to_fp16(self):
        self.latent_encoder.convert_to_fp16()
        self.denoise_fn.convert_to_fp16()

    def p_mean_variance(self, x, t, clip_denoised: bool, latent=None):
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.denoise_fn(x, t, latent)
        )

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, latent=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, latent=latent
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, x):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        latent = self.latent_encoder(x)
        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), latent=latent
            )
        return img, latent

    @torch.no_grad()
    def encode(self, x):
        return self.latent_encoder(x), torch.randn_like(
            x
        )  # TODO: implement equation 8: stochastic encoder

    @torch.no_grad()
    def p_decode_loop(self, shape, latent, x_start=None, clip_denoised=True):
        device = self.betas.device
        b = shape[0]
        img = x_start if x_start is not None else torch.randn(shape, device=device)
        # img = self.q_sample(x_start=x_start,
        #               t=torch.full((1,), self.num_timesteps, device=self.betas.device, dtype=torch.long),
        #               noise=None)

        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), latent=latent
            )
        if clip_denoised:
            img = torch.clamp_(img, -1.0, 1.0)
        return img

    def p_losses(self, x_start, t, class_id=None, noise=None):
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
    def __init__(self, *args, latent_encoder, patch_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_encoder = latent_encoder
        self.patch_size = patch_size
        #self.cropper = albumentations.RandomCrop(patch_size, patch_size, True)

    def p_mean_variance(self, x, t, clip_denoised: bool, latent=None):
        b, c, h, w = x.shape
        coords = torch.tensor(np.arange(h * w).reshape(h, w)).to('cuda')[None, None] / (h*w)
        x_with_coords = torch.cat([x, coords], dim=1)

        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.denoise_fn(x_with_coords, t, latent)
        )

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, latent=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, latent=latent
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, x):
        device = self.betas.device
        b, c, h, w = x.shape

        img = torch.randn(shape, device=device)
        latent = self.latent_encoder(x)

        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), latent=latent
            )
        return img, latent

    @torch.no_grad()
    def encode(self, x):
        return self.latent_encoder(x)

    @torch.no_grad()
    def p_decode_loop(self, shape, latent, x_start=None):
        device = self.betas.device

        b, c, h, w = shape
        coords = np.arange(h * w).reshape(h, w)

        img = x_start if x_start is not None else torch.randn(shape, device=device)
        img = torch.stack([img, coords], dim=1)

        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), latent=latent
            )
        return img

    def get_random_crop_coords(self, x_start):
        b, c, h, w = x_start.shape
        w_start = random.random()
        h_start = random.random()

        crop_height = self.patch_size
        crop_width = self.patch_size
        y1 = int((h - crop_height) * h_start)
        y2 = y1 + crop_height

        x1 = int((w - crop_width) * w_start)
        x2 = x1 + crop_width

        return y1, y2, x1, x2

    def p_losses(self, x_start, t, class_id=None, noise=None):
        b, c, h, w = x_start.shape


        x_latent = self.latent_encoder(x_start)
        coords = torch.tensor(np.arange(h * w).reshape(h, w)).to(x_start.device)

        y1, y2, x1, x2 = self.get_random_crop_coords(x_start)
        x_cropped = x_start[:, :, y1:y2, x1:x2]

        coords_cropped = coords[y1:y2, x1:x2][None].repeat_interleave(b, dim=0).unsqueeze(1) / (h*w)
        noise = default(noise, lambda: torch.randn_like(x_cropped))

        x_cropped_noisy = self.q_sample(x_start=x_cropped, t=t, noise=noise)

        # coords are added as an additional channel
        x_cropped_noisy = torch.cat([x_cropped_noisy, coords_cropped], dim=1)

        x_cropped_predicted_noise = self.denoise_fn(x_cropped_noisy, t, x_latent)

        if self.loss_type == "l1":
            loss = (noise - x_cropped_predicted_noise).abs().mean()
        elif self.loss_type == "l2":
            loss = F.mse_loss(noise, x_cropped_predicted_noise, reduction="mean")
        else:
            raise NotImplementedError()

        return loss


class AutoEncoderWaveGaussianDiffusion(GaussianDiffusion):
    def __init__(self, *args, sample_rate, latent_encoder, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_encoder = latent_encoder
        self.sample_rate = sample_rate

    def p_mean_variance(self, x, t, clip_denoised: bool, latent=None):
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.denoise_fn(x, t, latent)
        )

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, latent=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, latent=latent
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, x):
        device = self.betas.device
        b, c, l = x.shape
        img = torch.randn(shape, device=device)
        latent = self.latent_encoder(x)

        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), latent=latent
            )
        return img, latent

    @torch.no_grad()
    def encode(self, x):
        return self.latent_encoder(x)

    @torch.no_grad()
    def p_decode_loop(self, shape, latent, x_start=None):
        device = self.betas.device

        b = shape[0]

        img = x_start if x_start is not None else torch.randn(shape, device=device)

        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), latent=latent
            )
        return img

    def p_losses(self, x_start, t, class_id=None, noise=None):
        b, c, l = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_latent = self.latent_encoder(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_cropped_predicted_noise = self.denoise_fn(x_noisy, t, x_latent)

        if self.loss_type == "l1":
            loss = (noise - x_cropped_predicted_noise).abs().mean()
        elif self.loss_type == "l2":
            loss = F.mse_loss(noise, x_cropped_predicted_noise, reduction="mean")
        else:
            raise NotImplementedError()

        return loss
