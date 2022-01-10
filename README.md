# pytorch-diffusion-autoencoder
WIP: Unofficial implementation of diffusion autoencoders, using pytorch (https://diff-ae.github.io/)

Special thanks for https://github.com/lucidrains/denoising-diffusion-pytorch

# Samples
![sample](./sample/sample_1.png)

![sample](./sample/interpolation.gif)

# Usage

```
import torch
from model.diffusion import GaussianDiffusion
from model.unet import UNet
from model.latent_encoder import LateEncoder

model = Unet(
        in_channels=1,
        out_channels=1,
        base_hidden_channels=32,
        n_layers=2,
        timestep_embed=16,
        chan_multiplier=[1, 2],
        inner_layers=[3, 3],
        attention_layers=[False, True ],
        z_dim=128,
)

# almost same as the model. z_dim MUST be equals.
encoder = LatentEncoder(
      in_channels = 1
      out_channels = 1
      base_hidden_channels = 32
      n_layers = 4
      chan_multiplier = [ 1,2,2, 4]
      inner_layers = [ 1,2,2, 4]
      attention_layers = [ False, False , True, True]
      z_dim = 128
    )

diffusion = AutoEncoderGaussianDiffusion(
    model,
    image_size = 28,
    timesteps = 1000,   # number of steps
    loss_type = 'l1',    # L1 or L2
    latent_encoder = encoder,
)

training_images = torch.randn(8, 1, 28, 28)
loss = diffusion(training_images)
loss.backward()
# after a lot of training

image = torch.randn(3,28, 28) 
result, latent = diffusion.p_sample_loop((1, model.in_channels, *model.size), image)
result.shape # (3, 28, 28)
latent.shape # (1, 128)
```
