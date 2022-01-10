import click
import omegaconf
import torch
from torch import nn
from torchvision.transforms import ToPILImage

from model.diffusion import GaussianDiffusion
from model.unet import DiffusionModel
from torchvision.utils import make_grid

from utils.config import get_class_from_str


@click.command()
@click.option("--config-path", "-c", type=str)
@click.option("--device", "-d", type=str, default="cuda")
@click.option("--batch-size", "-b", type=int, default=1)
@click.option("--checkpoint-path", "-p", type=str)
def sample(config_path, checkpoint_path, batch_size=1, device="cuda"):
    config = omegaconf.OmegaConf.load(config_path)

    model = get_class_from_str(config.model.target)(**config.model.params).to(device)
    diffusion = get_class_from_str(config.diffusion.target)(
        model, **config.diffusion.params
    ).to(device)
    print(f"Resuming from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    diffusion.load_state_dict(checkpoint["model_state_dict"], strict=False)

    model.eval()
    # classes = torch.randint(0, 10, [batch_size]).to(device)
    classes = torch.tensor([1, 7, 1, 2, 1, 9, 8, 6]).to(device)
    print(classes)
    upsample = nn.Upsample(size=(model.size[0] * 3, model.size[1] * 3), mode="bilinear")

    for _ in range(2):
        result = diffusion.p_sample_loop(
            (batch_size, model.in_channel, *model.size), classes
        )
        ToPILImage()(make_grid(upsample(result.cpu()))).show()


if __name__ == "__main__":
    sample()
