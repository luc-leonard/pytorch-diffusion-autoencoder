import copy
from pathlib import Path

import click
import omegaconf
import torch
import torchvision.datasets
import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.config import get_class_from_str, number_of_params
from torch import nn

device = "cuda"


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer(object):
    def __init__(
        self,
        config,
        checkpoint_path,
        name,
        nb_epochs_to_train,
    ):
        super().__init__()
        self.config = config
        self.run_path = f"runs/{name}"
        self.tb_writer = SummaryWriter(self.run_path)
        self._create_models(config)
        self._create_datasets(config)

        self.ema = EMA(config.training.ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = 500
        self.step_start_ema = 1000
        self.grandient_accumulation_steps = config.training.grandient_accumulation_steps
        self.opt = torch.optim.AdamW(self.diffusion.parameters(), lr=config.training.learning_rate)
        self.current_step = 0
        self.current_epoch = 0
        self.fp16 = config.training.fp16

        if checkpoint_path is not None:
            self.load(checkpoint_path)
        self.reset_parameters()
        self.nb_epochs_to_train = nb_epochs_to_train
        self.sample_every = config.training.sample_every
        self.save_every = config.training.save_every

    def _create_models(self, config):
        self.model = get_class_from_str(config.model.target)(**config.model.params).to(device)
        print(f'model has {number_of_params(self.model):,} trainable parameters')
        self.encoder = get_class_from_str(config.encoder.target)(**config.encoder.params).to(device)
        print(f'encoder has {number_of_params(self.encoder):,} trainable parameters')
        self.diffusion = get_class_from_str(config.diffusion.target)(
            self.model, **config.diffusion.params, latent_encoder=self.encoder
        ).to(device)

    def _create_datasets(self, config):
        full_dataset = get_class_from_str(config.data.target)(**config.data.params)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=4
        )
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.config.training.batch_size,
                                                       shuffle=True, num_workers=4)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, step):
        torch.save(
            {
                "model_state_dict": self.diffusion.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
                "step": self.current_step,
                "epoch": self.current_epoch,
            },
            Path(self.run_path) / f"diffusion_{step}.pt",
        )

    def load(self, checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        self.diffusion.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_step = checkpoint["step"]
        self.current_epoch = checkpoint["epoch"]


    def train(self):
        for epoch in range(self.current_epoch, self.current_epoch + self.nb_epochs_to_train):
            step = self._do_epoch(epoch)
            self._do_valid()
            self.current_step = step
            self.save(step)

    def _do_epoch(self, epoch):
        self.diffusion.train()
        step = self.current_step
        pbar = tqdm.tqdm(self.train_dataloader)
        scaler = None
        if self.fp16:
            scaler = torch.cuda.amp.GradScaler()
        for image, _ in pbar:
            image = image.to(device)
            self.opt.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                loss = self.diffusion(image)
            self.tb_writer.add_scalar("train/loss", loss.item(), step)
            self.tb_writer.add_scalar("train/epoch", epoch, step)

            pbar.set_description(f"{step}: {loss.item():.4f}")
            if self.fp16:
                assert scaler
                scaler.scale(loss).backward()
                scaler.step(self.opt)
                scaler.update()
            else:
                loss.backward()
                self.opt.step()

            if step % self.sample_every == 0:
                self.sample('train', image[0], step)
                self.tb_writer.add_image("train/real_image", (image[0] + 1) / 2, step)
            if step % self.save_every == 0:
                self.save(step)
            step = step + 1
        return step

    @torch.no_grad()
    def _do_valid(self):
        self.diffusion.eval()
        pbar = tqdm.tqdm(self.valid_dataloader)
        base_step = self.current_step
        step = base_step
        losses = []
        for image, _ in pbar:
            image = image.to(device)
            loss = self.diffusion(image)
            losses.append(loss)
            pbar.set_description(f"{step}: {loss.item():.4f}")
            step = step + 1
        for _step in range(base_step, step):
            self.tb_writer.add_scalar("valid/loss", torch.stack(losses).mean().item(), _step)
        self.sample('valid', image[0], step)
        self.tb_writer.add_image("valid/real_image", (image[0] + 1) / 2, step)

    @torch.no_grad()
    def sample(self, stage, x, step):
        self.diffusion.eval()
        generated, latent = self.diffusion.p_sample_loop((1, self.model.in_channels, *self.model.size), x[None])
        generated = (generated + 1) / 2
        self.diffusion.train()
        self.tb_writer.add_image(
            f"{stage}/image", torchvision.utils.make_grid(generated, nrow=3), self.current_step
        )


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
    #train(config, name, epochs, resume_from)
    _config = omegaconf.OmegaConf.load(config)
    Trainer(_config, resume_from, name, epochs).train()


if __name__ == "__main__":
    main()
