from pathlib import Path

import click
import omegaconf
import torch
import torchvision.datasets
import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.config import get_class_from_str
from torch import nn

device = "cuda"

def number_of_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(config_path, name, epochs, resume_from):
    run_path = f"runs/{name}"
    config = omegaconf.OmegaConf.load(config_path)

    tb_writer = SummaryWriter(run_path)
    model = get_class_from_str(config.model.target)(**config.model.params).to(device)
    print(f'model has {number_of_params(model):,} trainable parameters')
    encoder = get_class_from_str(config.encoder.target)(**config.encoder.params).to(device)
    print(f'encoder has {number_of_params(encoder):,} trainable parameters')
    diffusion = get_class_from_str(config.diffusion.target)(
        model, **config.diffusion.params, latent_encoder=encoder
    ).to(device)

    opt = torch.optim.AdamW(diffusion.parameters(), lr=config.training.learning_rate)
    step = 0
    base_epoch = 0
    if resume_from is not None:
        print(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from)
        diffusion.load_state_dict(checkpoint["model_state_dict"], strict=False)
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        step = checkpoint["step"]
        base_epoch = checkpoint["epoch"]

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
        print(f"epoch {epoch + base_epoch} start")
        new_step = do_epoch(
            train_dataloader, diffusion, epoch + base_epoch, model, opt, run_path, step, tb_writer, config.training
        )
        print(f"epoch {epoch + base_epoch} done")
        do_valid(valid_dataloader, diffusion, model, step, tb_writer)
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
    base_step = step
    losses = []
    for image, _ in pbar:
        image = image.to(device)
        loss = diffusion(image)
        losses.append(loss)
        pbar.set_description(f"{step}: {loss.item():.4f}")
        step = step + 1
    for _step in range(base_step, step):
        tb_writer.add_scalar("valid/loss", torch.stack(losses).mean().item(), _step)
    sample(diffusion, model, step, image[0], 'valid', tb_writer)
    tb_writer.add_image("valid/real_image", (image[0] + 1) / 2, step)


def do_epoch(dataloader, diffusion, epoch, model, opt, run_path, step, tb_writer, training_config):
    pbar = tqdm.tqdm(dataloader)
    for image, _ in pbar:
        image = image.to(device)
        opt.zero_grad()

        loss = diffusion(image)
        tb_writer.add_scalar("train/loss", loss.item(), step)
        tb_writer.add_scalar("train/epoch", epoch, step)
        loss.backward()
        opt.step()
        pbar.set_description(f"{step}: {loss.item():.4f}")
        if step % training_config.sample_every == 0:
            sample(diffusion, model, step, image[0], 'train', tb_writer)
            tb_writer.add_image("train/real_image", (image[0] + 1) / 2, step)
        if step % training_config.save_every == 0:
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


def sample(diffusion, model, step, x, stage, tb_writer):
    model.eval()
    generated, latent = diffusion.p_sample_loop((1, model.in_channels, *model.size), x[None])
    generated = (generated + 1) / 2
    model.train()
    tb_writer.add_image(
        f"{stage}/image", torchvision.utils.make_grid(generated, nrow=3), step
    )


@click.command()
@click.option("--config", "-c")
@click.option("--name", "-n")
@click.option("--epochs", "-e", default=500)
@click.option("--resume-from", "-r", default=None)
def main(config: str, name: str, resume_from: str, epochs: int):
    train(config, name, epochs, resume_from)


if __name__ == "__main__":
    main()
