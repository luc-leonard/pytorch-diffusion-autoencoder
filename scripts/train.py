from pathlib import Path

import click
import omegaconf
import torch
import numpy as np
import torchvision.datasets
from torchvision.transforms import ToTensor

from data.data import MyImageFolderDataset
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
    model = DiffusionModel(**config.model.params)
    model.to(device)
    diffusion = GaussianDiffusion(model, **config.diffusion.params).to(device)
    opt = torch.optim.Adam(diffusion.parameters(), lr=config.training.learning_rate)
    step = 0
    if resume_from is not None:
        print(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from)
        diffusion.load_state_dict(checkpoint['model_state_dict'], strict=False)
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
    print('creating dataset')
    dataset = DatasetWrapper(get_class_from_str(config.data.target)(**config.data.params))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=4)
    print('training')
    for epoch in range(epochs):
        tb_writer.add_scalar('epoch', epoch, step)
        pbar = tqdm.tqdm(dataloader)
        for image, class_id in pbar:
            image = image.to(device)
            class_id = class_id.to(device)
            opt.zero_grad()
            loss = diffusion(image, class_id)
            tb_writer.add_scalar("loss", loss.item(), step)
            loss.backward()
            opt.step()
            pbar.set_description(f"{step}: {loss.item():.4f}")
            if step % 1000 == 0:
                generated = diffusion.p_sample_loop((1, 1, 28, 28), torch.tensor([8]).to(device)).squeeze(0)
                tb_writer.add_image("image", generated, step)
                tb_writer.add_image("real_image", image[0], step)
                torch.save({
                    'model_state_dict': diffusion.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'step': step,
                    'epoch': epoch
                }, Path(run_path) / f'diffusion_{step}.pt')
            step = step + 1

        torch.save({
            'model_state_dict': diffusion.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'step': step,
            'epoch': epoch
        }, Path(run_path) / f'diffusion_{step}.pt')


@click.command()
@click.option('--config', '-c')
@click.option('--name', '-n')
@click.option('--epochs', '-e', default=10)
@click.option('--resume-from', '-r', default=None)
def main(config: str, name: str, resume_from: str, epochs: int):
    train(config, name, epochs, resume_from)


if __name__ == "__main__":
    main()
