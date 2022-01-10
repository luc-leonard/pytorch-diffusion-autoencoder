# pytorch-diffusion-autoencoder
WIP: Unofficial implementation of diffusion autoencoders, using pytorch (https://diff-ae.github.io/)

Special thanks for https://github.com/lucidrains/denoising-diffusion-pytorch
# Usage

```
import torch
from model.diffusion import GaussianDiffusion
from model.unet import UNet

model = Unet(
        in_channels=3,
        out_channels=3,
        base_hidden_channels=128,
        n_layers=2,
        timestep_embed=16,
        chan_multiplier=[2, 2],
        inner_layers=[3, 3],
        attention_layers=[False, True ],
        z_dim=128,
)

diffusion = AutoEncoderGaussianDiffusion(
    model,
    image_size = 28,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)

training_images = torch.randn(8, 3, 28, 28)
loss = diffusion(training_images)
loss.backward()
# after a lot of training

image = torch.randn(3,28, 28) 
result, latent = diffusion.p_sample_loop((1, model.in_channels, *model.size), image)
result.shape # (3, 28, 28)
latent.shape # (1, 128)
```
