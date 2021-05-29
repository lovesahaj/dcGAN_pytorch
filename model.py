import torch
import torch.nn as nn
from torch.nn.modules.container import ModuleList


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.2)
            if isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.bias, 0)


class Discriminator(nn.Module):
    def __init__(self, img_channels, features_d):
        super(Discriminator, self).__init__()

        # Input Shape would be:  N * img_channels * H(64) * W(64)

        self.disc = nn.Sequential(
            *self.__block(img_channels, features_d,
                          4, 2, 1, batch_norm=False),  # 32 * 32
            *self.__block(features_d, features_d * 2, 4, 2, 1),  # 16 * 16
            *self.__block(features_d * 2, features_d * 4, 4, 2, 1),  # 8 * 8
            *self.__block(features_d * 4, features_d * 8, 4, 2, 1),  # 4 * 4
            *self.__block(features_d * 8, 1, 4, 2, 0,
                          batch_norm=False, act='sig')  # 1 * 1
        )

    def __block(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True, act='leakyrelu'):
        block = ModuleList()
        block.append(nn.Conv2d(in_channels, out_channels,
                     kernel_size, stride, padding, bias=False))

        if batch_norm:
            block.append(nn.BatchNorm2d(
                out_channels, track_running_stats=True),)

        if act == 'leakyrelu':
            block.append(nn.LeakyReLU(0.2))
        if act == 'sig':
            block.append(nn.Sigmoid())

        return block

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, features_g):
        super(Generator, self).__init__()

        self.block1 = nn.Sequential(
            *self.__block(z_dim, features_g * 16, 4, 1, 0),)
        self.block2 = nn.Sequential(
            *self.__block(features_g * 16, features_g * 8, 4, 2, 1),)
        self.block3 = nn.Sequential(
            *self.__block(features_g * 8, features_g * 4, 4, 2, 1),)
        self.block4 = nn.Sequential(
            *self.__block(features_g * 4, features_g * 2, 4, 2, 1),)
        self.block5 = nn.Sequential(
            *self.__block(features_g * 2, img_channels, 4, 2, 1,
                          batch_norm=False, act='tanh')
        )

    def __block(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True, act='relu'):
        block = ModuleList()
        block.append(nn.ConvTranspose2d(in_channels, out_channels,
                     kernel_size, stride, padding, bias=False))

        if batch_norm:
            block.append(nn.BatchNorm2d(
                out_channels, track_running_stats=True),)

        if act == 'relu':
            block.append(nn.ReLU())

        if act == 'tanh':
            block.append(nn.Tanh())

        return block

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        return out


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100

    x = torch.randn((N, in_channels, H, W))

    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)

    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)


if __name__ == "__main__":
    test()
