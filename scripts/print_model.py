import click
import omegaconf
import torch

from scripts.train_diffusion_autoencoder import number_of_params
from utils.config import get_class_from_str


@click.command()
@click.option("--config-path", "-c", type=str)
@torch.no_grad()
def print_model(config_path):
    config = omegaconf.OmegaConf.load(config_path)

    model = get_class_from_str(config.model.target)(**config.model.params)
    encoder = get_class_from_str(config.encoder.target)(**config.encoder.params)
    print(model)
    print('--------------------------------------------------------------')
    print(encoder)

    print(f'model has {number_of_params(model):,} trainable parameters')
    print(f'encoder has {number_of_params(encoder):,} trainable parameters')



if __name__ == "__main__":
    print_model()
