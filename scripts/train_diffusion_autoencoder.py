from pathlib import Path

import click
import omegaconf
import torch
import numpy as np
import torchvision.datasets
from torchvision.transforms import ToTensor

from data.image import MyImageFolderDataset
from model.diffusion import GaussianDiffusion
from model.unet import UNet, DiffusionModel
import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.config import get_class_from_str

device = 'cuda'


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x = self.dataset[index]
        image = x[0]
        if not isinstance(image, torch.Tensor):
            image = ToTensor()(image)
        return image, x[1]


def train(config_path, name, epochs, resume_from):
    run_path = f"runs/{name}"
    config = omegaconf.OmegaConf.load(config_path)

    tb_writer = SummaryWriter(run_path)
    model = get_class_from_str(config.model.target)(**config.model.params).to(device)
    diffusion = get_class_from_str(config.diffusion.target)(model, **config.diffusion.params).to(device)

    opt = torch.optim.AdamW(diffusion.parameters(), lr=config.training.learning_rate)
    step = 0
    if resume_from is not None:
        print(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from)
        diffusion.load_state_dict(checkpoint['model_state_dict'], strict=False)
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']

    for g in opt.param_groups:
        g['lr'] = config.training.learning_rate

    print('creating dataset')
    dataset = DatasetWrapper(get_class_from_str(config.data.target)(**config.data.params))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=4)
    print('training')
    for epoch in range(epochs):
        tb_writer.add_scalar('epoch', epoch, step)
        step = do_epoch(dataloader, diffusion, epoch, model, opt, run_path, step, tb_writer)

        torch.save({
            'model_state_dict': diffusion.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'step': step,
            'epoch': epoch
        }, Path(run_path) / f'diffusion_{step}.pt')


def do_epoch(dataloader, diffusion, epoch, model, opt, run_path, step, tb_writer):
    pbar = tqdm.tqdm(dataloader)
    for image, _ in pbar:
        image = image.to(device)

        opt.zero_grad()

        loss = diffusion(image)


        tb_writer.add_scalar("loss", loss.item(), step)
        loss.backward()
        opt.step()
        pbar.set_description(f"{step}: {loss.item():.4f}")
        if step % 500 == 0:
            sample(diffusion, model, step, image[0], tb_writer)
            tb_writer.add_image("real_image", (image[0] + 1) / 2, step)
            torch.save({
                'model_state_dict': diffusion.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'step': step,
                'epoch': epoch
            }, Path(run_path) / f'diffusion_{step}.pt')
        step = step + 1
    return step


def sample(diffusion, model, step, x, tb_writer):
    model.eval()
    generated = diffusion.p_sample_loop((1, model.in_channel, *model.size), x)
    generated = (generated + 1) / 2
    model.train()
    tb_writer.add_image("image", torchvision.utils.make_grid(generated, nrow=3), step)


@click.command()
@click.option('--config', '-c')
@click.option('--name', '-n')
@click.option('--epochs', '-e', default=10)
@click.option('--resume-from', '-r', default=None)
def main(config: str, name: str, resume_from: str, epochs: int):
    train(config, name, epochs, resume_from)


if __name__ == "__main__":
    main()
