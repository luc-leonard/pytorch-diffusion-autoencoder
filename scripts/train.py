from pathlib import Path

import torch
import numpy as np

from data.data import MyImageFolderDataset
from model.unet import UNet, DiffusionModel
import tqdm
from torch.utils.tensorboard import SummaryWriter


def train_step(model, noise_level, loss_fn, opt, x):
    N, C, X, Y = x.shape
    t = torch.randint(0, len(noise_level), [N], device=x.device)
    noise_scale = noise_level[t].unsqueeze(1).reshape(N, 1, 1, 1)
    noise_scale_sqrt = noise_scale ** 0.5
    noise = torch.randn_like(x)

    noisy_x = noise_scale_sqrt * x + (1.0 - noise_scale) ** 0.5 * noise
    opt.zero_grad()
    predicted = model(noisy_x, t)
    loss = loss_fn(noise, predicted.squeeze(1))
    loss.backward()
    opt.step()
    return loss


def train():
    run_path = "runs/1"
    Path(run_path).mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter("runs/1")
    model = DiffusionModel().to("cuda")
    opt = torch.optim.Adam(model.parameters(), lr=3e-5)
    loss_fn = torch.nn.L1Loss()
    steps = torch.linspace(1e-4, 0.05, 100).to("cuda")
    inference_steps = torch.linspace(1e-4, 0.05, 100).to("cuda")

    dataset = MyImageFolderDataset(
        data_dir="/media/lleonard/big_slow_disk/datasets/ffhq/images1024x1024/",
        resize=128,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True
    )
    noise_level = torch.cumprod(1 - steps, dim=0).to(torch.float32).to("cuda")

    pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (x, t) in pbar:
        x = x.to("cuda")
        loss = train_step(model, noise_level, loss_fn, opt, x)
        tb_writer.add_scalar("loss", loss, i)
        if i % 100 == 0:
            image = inference(model, steps)
            tb_writer.add_image("image", image, i)
        pbar.set_description(f"loss: {loss.item():.4f}")
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
        },
        run_path + "/model.pt",
    )


@torch.no_grad()
def inference(model, training_noise_schedule, inference_noise_schedule):
    model.eval()

    talpha = 1 - training_noise_schedule
    talpha_cum = torch.cumprod(talpha, dim=0)
    inference_noise_schedule = training_noise_schedule

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = torch.cumprod(alpha, dim=0)

    T = []
    for s in range(len(inference_noise_schedule)):
        for t in range(len(training_noise_schedule) - 1):
            if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                    talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5
                )
                T.append((t + twiddle).cpu().item())
                break
    T = np.array(T, dtype=np.float32)
    noisy_image = torch.randn(1, 3, 128, 128, device="cuda")
    for n in range(len(alpha) - 1, -1, -1):
        c1 = 1 / alpha[n] ** 0.5
        c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5
        noisy_image = c1 * (
            noisy_image
            - c2
            * model(
                noisy_image, torch.tensor([T[n]], device=noisy_image.device)
            ).squeeze(1)
        )
        if n > 0:
            noise = torch.randn_like(noisy_image)
            sigma = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]) ** 0.5
            noisy_image += sigma * noise
        noisy_image = torch.clamp(noisy_image, -1.0, 1.0)
    model.train()
    return noisy_image.squeeze(0)


def main():
    train()


if __name__ == "__main__":
    main()
