import itertools

import click
import numpy as np
import omegaconf
import torch
import tqdm
from PIL.Image import LANCZOS
from torch import nn
from torchvision.transforms import ToPILImage

from torchvision.utils import make_grid

from scripts.train import DatasetWrapper
from utils.config import get_class_from_str
import imageio

@torch.no_grad()
def show_interpolation(diffusion, model, x_1, x_2, steps=100, i=0):
    print("Generating interpolation")
    x_1_latent, x1_noisy = diffusion.latent_encoder(x_1[None])
    x_2_latent, x2_noisy = diffusion.latent_encoder(x_2[None])


    interpolations = torch.stack([torch.lerp(x_1_latent, x_2_latent, t) for t in torch.linspace(0, 1, steps=steps).to(x_1.device)]).squeeze(1)

    noise = x1_noisy #torch.randn((steps, model.in_channels, *model.size)).to(x_1.device)
    y_s = diffusion.p_decode_loop((steps, model.in_channels, *model.size), interpolations.squeeze(1), x_start=noise)
    y_s = torch.clamp(y_s, 0, 1)
    video = imageio.get_writer(f'interpolation_{i}.gif', fps=25)
    for y in y_s:
        y = ToPILImage()(y.cpu())
        y = y.resize((64, 64), LANCZOS)
        video.append_data(np.array(y))
    for y in reversed(y_s):
        y = ToPILImage()(y.cpu())
        y = y.resize((64, 64), LANCZOS)
        video.append_data(np.array(y))
    video.close()




@click.command()
@click.option("--config-path", "-c", type=str)
@click.option("--device", "-d", type=str, default="cuda")
@click.option("--checkpoint-path", "-p", type=str)
@torch.no_grad()
def sample(config_path, checkpoint_path, device="cpu"):
    config = omegaconf.OmegaConf.load(config_path)

    model = get_class_from_str(config.model.target)(**config.model.params).to(device)
    encoder = get_class_from_str(config.encoder.target)(**config.encoder.params).to(device)
    diffusion = get_class_from_str(config.diffusion.target)(
        model, **config.diffusion.params, latent_encoder=encoder
    ).to(device)
    print(f"Resuming from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    diffusion.load_state_dict(checkpoint["ema_model_state_dict"], strict=True)

    model.eval()
    dataset = DatasetWrapper(
        get_class_from_str(config.data.target)(**config.data.params)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0
    )


    for i, (xs, _) in enumerate(tqdm.tqdm(dataloader)):
        x_1 = xs[0].to(device)
        x_2 = xs[1].to(device)
        show_interpolation(diffusion, model, x_1, x_2, 100, i)
        if i == 5:
            break




if __name__ == "__main__":
    sample()
