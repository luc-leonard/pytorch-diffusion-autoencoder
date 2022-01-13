import copy
import shutil
from pathlib import Path

import click
import omegaconf
import torch
import torchvision.datasets
import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.config import get_class_from_str, number_of_params
import logging

def init_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger

LOGGER = init_logger()
LOGGER.info("LOG SYSTEM: Started")

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
        self.grad_clip = config.training.grad_clip
        self.update_ema_every = 1000
        self.step_start_ema = 10000
        self.grandient_accumulation_steps = config.training.grandient_accumulation_steps
        self.opt = torch.optim.AdamW(self.diffusion.parameters(), lr=config.training.learning_rate)
        if config.training.scheduler != 'none':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, patience=config.training.scheduler.patience, verbose=True)
            print(f"Using scheduler {self.scheduler}")
        else:
            self.scheduler = None
        self.current_step = 0
        self.current_epoch = 0
        self.fp16 = config.training.fp16

        if checkpoint_path is not None:
            self.load(checkpoint_path)
        self.reset_parameters()
        self.nb_epochs_to_train = nb_epochs_to_train
        self.sample_every = config.training.sample_every
        self.save_every = config.training.save_every

        self.add_model_to_tensorboard()
        print(self.diffusion)

        for param_group in self.opt.param_groups:
            param_group['lr'] = config.training.learning_rate

    def add_model_to_tensorboard(self):
        ...
        # image, _ = next(iter(self.train_dataloader))
        # self.tb_writer.add_graph(self.diffusion, image.cuda())

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
            self.train_dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.config.training.batch_size,
                                                       shuffle=True, num_workers=4, pin_memory=True)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self, step):
        if step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, step):
        torch.save(
            {
                "model_state_dict": self.diffusion.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "step": step,
                "epoch": self.current_epoch,
            },
            str(Path(self.run_path) / f"diffusion_{step}.pt")),
        shutil.copy(Path(self.run_path) / f"diffusion_{step}.pt", Path(self.run_path) / "last.pt")

    def load(self, checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        self.diffusion.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_step = checkpoint["step"]
        self.current_epoch = checkpoint["epoch"]


    def train(self):
        base_epoch = self.current_epoch
        self.tb_writer.add_text("config", str(self.config))
        for epoch in range(base_epoch, base_epoch + self.nb_epochs_to_train):
            step = self._do_epoch(epoch)
            valid_loss = self._do_valid()
            self.current_epoch += 1
            print(f"Epoch {epoch} done, step {step}, valid loss {valid_loss}")
            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(valid_loss)
                print(f"Scheduler step {self.scheduler.num_bad_epochs}")
            self.current_step = step
            #self.save(step)

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
            self.tb_writer.add_scalar("train/lr", self.opt.param_groups[0]["lr"], step)

            pbar.set_description(f"{step}: {loss.item():.4f}")
            if self.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % self.grandient_accumulation_steps == 0:
                if self.fp16:
                    assert scaler
                    scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), self.grad_clip)
                    scaler.step(self.opt)
                    scaler.update()
                else:
                    self.opt.step()
            if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()

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
        mean_loss = torch.stack(losses).mean()
        for _step in range(base_step, step):
            self.tb_writer.add_scalar("valid/loss", mean_loss.item(), _step)
        self.sample('valid', image[0], step)
        self.tb_writer.add_image("valid/real_image", (image[0] + 1) / 2, step)
        return mean_loss

    @torch.no_grad()
    def sample(self, stage, x, step):
        self.diffusion.eval()
        generated, latent = self.diffusion.p_sample_loop((1, self.model.in_channels, *self.model.size), x[None])
        generated = (generated + 1) / 2
        self.diffusion.train()
        self.tb_writer.add_image(
            f"{stage}/image", torchvision.utils.make_grid(generated, nrow=3), step
        )


@click.command()
@click.option("--config", "-c")
@click.option("--name", "-n")
@click.option("--epochs", "-e", default=500)
@click.option("--resume-from", "-r", default=None)
def main(config: str, name: str, resume_from: str, epochs: int):
    _config = omegaconf.OmegaConf.load(config)
    Trainer(_config, resume_from, name, epochs).train()


if __name__ == "__main__":
    main()
