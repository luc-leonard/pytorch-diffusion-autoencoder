from pathlib import Path

import click
import omegaconf
import torch
import torchvision.datasets
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

from utils.config import get_class_from_str

device = "cuda"


def train(config_path, name, epochs, resume_from):
    run_path = f"runs/{name}"
    config = omegaconf.OmegaConf.load(config_path)

    tb_writer = SummaryWriter(run_path)
    model = get_class_from_str(config.model.target)(**config.model.params).to(device)
    diffusion = get_class_from_str(config.diffusion.target)(
        model, **config.diffusion.params
    ).to(device)

    opt = torch.optim.AdamW(diffusion.parameters(), lr=config.training.learning_rate)
    step = 0
    if resume_from is not None:
        print(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from)
        diffusion.load_state_dict(checkpoint["model_state_dict"], strict=False)
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        step = checkpoint["step"]

    for g in opt.param_groups:
        g["lr"] = config.training.learning_rate

    print("creating dataset")
    full_dataset = get_class_from_str(config.data.target)(**config.data.params)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=4
    )
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=4)
    print("training")
    for epoch in range(epochs):
        tb_writer.add_scalar("epoch", epoch, step)
        new_step = do_epoch(
            train_dataloader, diffusion, epoch, model, opt, run_path, step, tb_writer
        )
        print(f"epoch {epoch} done")
        do_valid(
            valid_dataloader, diffusion, epoch, model, opt, run_path, step, tb_writer
        )
        step = new_step
        torch.save(
            {
                "model_state_dict": diffusion.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "step": step,
                "epoch": epoch,
            },
            Path(run_path) / f"diffusion_{step}.pt",
        )

@torch.no_grad()
def do_valid(dataloader, diffusion, model, step, tb_writer):
    model.eval()
    pbar = tqdm.tqdm(dataloader)
    for image, _ in pbar:
        image = image.to(device)
        with torch.no_grad():
            loss = diffusion(image)
        tb_writer.add_scalar("valid/loss", loss.item(), step)
        pbar.set_description(f"{step}: {loss.item():.4f}")
        step = step + 1

def do_epoch(dataloader, diffusion, epoch, model, opt, run_path, step, tb_writer):
    pbar = tqdm.tqdm(dataloader)
    for image, _ in pbar:
        image = image.to(device)

        opt.zero_grad()

        loss = diffusion(image)

        tb_writer.add_scalar("train/loss", loss.item(), step)
        loss.backward()
        opt.step()
        pbar.set_description(f"{step}: {loss.item():.4f}")
        if step % 500 == 0:
            sample(diffusion, model, step, image[0], tb_writer)
            tb_writer.add_image("train/real_image", (image[0] + 1) / 2, step)
            torch.save(
                {
                    "model_state_dict": diffusion.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "step": step,
                    "epoch": epoch,
                },
                Path(run_path) / f"diffusion_{step}.pt",
            )
        step = step + 1
    return step


def sample(diffusion, model, step, x, tb_writer):
    model.eval()
    generated, latent = diffusion.p_sample_loop((1, model.in_channels, *model.size), x)
    generated = (generated + 1) / 2
    model.train()
    tb_writer.add_image(
        "train/image", torchvision.utils.make_grid(generated, nrow=3), step
    )


# tb_writer.add_image("train/latent", latent[0], step)


@click.command()
@click.option("--config", "-c")
@click.option("--name", "-n")
@click.option("--epochs", "-e", default=500)
@click.option("--resume-from", "-r", default=None)
def main(config: str, name: str, resume_from: str, epochs: int):
    train(config, name, epochs, resume_from)


if __name__ == "__main__":
    main()
