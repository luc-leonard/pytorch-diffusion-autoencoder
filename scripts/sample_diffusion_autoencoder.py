import itertools
from typing import List

import PIL.Image
import click
import numpy as np
import omegaconf
import torch
import torchvision.utils
import tqdm
from PIL.Image import LANCZOS
from torch import nn
from torchvision.transforms import ToPILImage, ToTensor

from torchvision.utils import make_grid

from scripts.train import DatasetWrapper
from utils.config import get_class_from_str
import imageio


def to_image(tensor):
    tensor = (tensor + 1) / 2
    return ToPILImage()(tensor.cpu())


def make_grid_from_pil(images: List[PIL.Image.Image], *args, **kwargs) -> PIL.Image:
    return make_grid([ToTensor()(image) for image in images], *args, **kwargs)

@torch.no_grad()
def show_interpolation(diffusion, model, x_1, x_2, output=None, steps=10, fps=2):
    print("Generating interpolation")

    x_1_latent = diffusion.latent_encoder(x_1[None])
    x_2_latent = diffusion.latent_encoder(x_2[None])

    interpolations = torch.stack(
        [
            torch.lerp(x_1_latent, x_2_latent, t)
            for t in torch.linspace(0, 1, steps=steps).to(x_1.device)
        ]
    ).squeeze(1)

    noise = torch.randn((1, model.in_channels, *model.size)).to(x_1.device).repeat_interleave(steps, dim=0)

    y_s = diffusion.p_decode_loop(
        (steps, model.in_channels, *model.size),
        interpolations.squeeze(1),
        x_start=noise,
    )

    x_1 = to_image(x_1.squeeze(0)).resize((128, 128), resample=LANCZOS)
    x_2 = to_image(x_2.squeeze(0)).resize((128, 128), resample=LANCZOS)
    y_0 = to_image(y_s[0]).resize((128, 128), resample=LANCZOS)
    y_1 = to_image(y_s[-1]).resize((128, 128), resample=LANCZOS)

    ToPILImage()(make_grid_from_pil([y_0, x_1, y_1, x_2], nrow=2)).save(output + '_grid.png')

    video = imageio.get_writer(output, fps=fps)
    for i in range(fps):
        y = to_image(y_s[0])
        y = y.resize((128, 128), LANCZOS)
        video.append_data(np.array(y))

    for y in y_s:
        y = to_image(y)
        y = y.resize((128, 128), LANCZOS)
        video.append_data(np.array(y))

    for i in range(fps):
        y = to_image(y_s[-1])
        y = y.resize((128, 128), LANCZOS)
        video.append_data(np.array(y))

    for y in reversed(y_s):
        y = to_image(y)
        y = y.resize((128, 128), LANCZOS)
        video.append_data(np.array(y))

    video.close()




@click.command()
@click.option("--config-path", "-c", type=str)
@click.option("--device", "-d", type=str, default="cuda")
@click.option("--checkpoint-path", "-p", type=str)
@click.option("--path", "-p", type=str)
@click.option("--path-2", "-p2", type=str)
@click.option("--output", "-o", type=str, default='output.gif')
@torch.no_grad()
def sample(config_path, checkpoint_path, device="cpu", path=None, path_2=None, output=None):
    config = omegaconf.OmegaConf.load(config_path)

    model = get_class_from_str(config.model.target)(**config.model.params).to(device)
    encoder = get_class_from_str(config.encoder.target)(**config.encoder.params).to(
        device
    )
    diffusion = get_class_from_str(config.diffusion.target)(
        model, **config.diffusion.params, latent_encoder=encoder
    ).to(device)
    print(f"Resuming from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    diffusion.load_state_dict(checkpoint["ema_model_state_dict"], strict=True)

    model.eval()
    dataset = DatasetWrapper(
        get_class_from_str(config.data.target)(**config.data.params)
    )


    if path:
        img = open_image(device, path)

        if not path_2:
            latent = diffusion.latent_encoder(img[None])

            noise = torch.randn((1, model.in_channels, *model.size)).to(img.device)
            y_s = diffusion.p_decode_loop(
                (1, model.in_channels, *model.size), latent, x_start=noise
            )[0]

            y_s = to_image((y_s + 1) / 2)
            img = to_image((img + 1) / 2)

            ToPILImage()(make_grid_from_pil([y_s, img])).save(output)
        else:
            img_2 = open_image(device, path_2)
            show_interpolation(diffusion, model, img, img_2, output=output, steps=50, fps=15)
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=0
        )

        for i, (xs, _) in enumerate(tqdm.tqdm(dataloader)):
            x_1 = xs[0].to(device)
            x_2 = xs[1].to(device)
            show_interpolation(diffusion, model, x_1, x_2, output=output, steps=50, fps=15)
            if i == 0:
                break


def open_image(device, path):
    img = PIL.Image.open(path).resize((64, 64)).convert("RGB")
    img = np.array(img).astype(np.uint8)
    img = (img / 127.5 - 1.0).astype(np.float32)
    img = torch.tensor(img).to(device)
    img = img.permute(2, 0, 1)
    return img


if __name__ == "__main__":
    sample()
