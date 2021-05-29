from utils import load_checkpoint, save_checkpoint
from config import (BATCH_SIZE, DEVICE, FEATURES_DISC, FEATURES_GEN, IMG_CHANNELS,
                    IMG_SIZE, LEARNING_RATE_DISC, LEARNING_RATE_GEN, LOAD_MODEL, NUM_EPOCH, NUM_WORKERS, Z_DIM)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, dataloader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights
from tqdm import tqdm

transforms = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(IMG_CHANNELS)],
            [0.5 for _ in range(IMG_CHANNELS)]),
    ]
)


def train_loop(gen, disc, loader, opt_gen, opt_disc, criterion, fixed_noise, writer_r, writer_f, step):
    loop = tqdm(loader)

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(DEVICE)

        ## Discriminator: max(log(D(x)) +  log(1 - D(G(z))))
        disc_real = disc(real).flatten()
        fake_d = gen(noise)
        disc_fake = disc(fake_d.detach()).flatten()

        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        loss_disc = (loss_disc_fake + loss_disc_real) / 2

        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        ## Generator: max(log(D(G(z))))
        output = disc(gen(noise)).flatten()
        loss_gen = criterion(output, torch.ones_like(output))

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        loop.set_postfix(
            {'Gen. Loss': loss_gen.item(), 'Disc. Loss': loss_disc.item()})

        with torch.no_grad():
            fake = gen(fixed_noise)

            img_grid_real = torchvision.utils.make_grid(
                real[:32], normalize=True)
            img_grid_fake = torchvision.utils.make_grid(
                fake[:32], normalize=True)

            writer_r.add_image("Real", img_grid_real, global_step=step)
            writer_f.add_image("Fake", img_grid_fake, global_step=step)

        step += 1


def train():
    dataset = datasets.MNIST(root="dataset/", train=True,
                             transform=transforms, download=True)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=NUM_WORKERS)

    gen = Generator(Z_DIM, IMG_CHANNELS, FEATURES_GEN).to(DEVICE)
    disc = Discriminator(IMG_CHANNELS, FEATURES_DISC).to(DEVICE)

    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(
        gen.parameters(), lr=LEARNING_RATE_GEN, betas=(0.5, 0.999))
    opt_disc = optim.Adam(
        disc.parameters(), lr=LEARNING_RATE_DISC, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    fixed_noise = torch.randn((32, Z_DIM, 1, 1)).to(DEVICE)
    writer_real = SummaryWriter("logs/real")
    writer_fake = SummaryWriter("logs/fake")
    step = 0

    if LOAD_MODEL:
        load_checkpoint(torch.load("gen.pth.tar"), gen)
        load_checkpoint(torch.load("disc.pth.tar"), disc)

    gen.train()
    disc.train()

    for epoch in range(NUM_EPOCH):
        train_loop(gen, disc, loader, opt_gen, opt_disc, criterion,
                   fixed_noise, writer_real, writer_fake, step)

        checkpoint_gen = {
            'gen_state_dict': gen.state_dict(),
            'optimizer': opt_gen.state_dict()
        }

        checkpoint_disc = {
            'disc_state_dict': disc.state_dict(),
            'optimizer': opt_disc.state_dict()
        }

        save_checkpoint(checkpoint_gen, "gen.pth.tar")
        save_checkpoint(checkpoint_disc, "disc.pth.tar")


if __name__ == '__main__':
    train()
