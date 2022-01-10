import click
import omegaconf
import torch
from torch import nn
from torchvision.transforms import ToPILImage

from torchvision.utils import make_grid

from scripts.train import DatasetWrapper
from utils.config import get_class_from_str


@click.command()
@click.option("--config-path", "-c", type=str)
@click.option("--device", "-d", type=str, default="cuda")
@click.option("--batch-size", "-b", type=int, default=1)
@click.option("--checkpoint-path", "-p", type=str)
@torch.no_grad()
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
    dataset = DatasetWrapper(
        get_class_from_str(config.data.target)(**config.data.params)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=0
    )

    upsample = nn.Upsample(size=(model.size[0] * 3, model.size[1] * 3), mode="bilinear")

    for x, _ in dataloader:
        x = x.to(device)
        x = x[0]
        result, latent = diffusion.p_sample_loop((1, model.in_channels, *model.size), x)
        print(result.shape)
        print(x.shape)
        print(latent.shape)
        upsampled_x = upsample(x[None]).squeeze(0)
        upsampled_result = upsample(result).squeeze(0)
        ToPILImage()(upsampled_x.cpu()).show()
        ToPILImage()(upsampled_result.cpu()).show()
        break


if __name__ == "__main__":
    sample()
