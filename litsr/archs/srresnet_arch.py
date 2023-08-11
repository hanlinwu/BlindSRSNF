from torch import nn
from math import log2
from litsr.utils.registry import ArchRegistry


@ArchRegistry.register()
class SRResNet(nn.Module):
    """
    PyTorch Module for SRGAN, https://arxiv.org/pdf/1609.04802.
    """

    def __init__(self, scale=4, ngf=64, n_blocks=16):
        super(SRResNet, self).__init__()

        self.head = nn.Sequential(
            nn.ReflectionPad2d(4), nn.Conv2d(3, ngf, kernel_size=9), nn.PReLU()
        )
        self.body = nn.Sequential(
            *[SRGANBlock(ngf) for _ in range(n_blocks)],
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf, kernel_size=3),
            nn.BatchNorm2d(ngf)
        )
        self.tail = nn.Sequential(
            UpscaleBlock(scale, ngf, act="prelu"),
            nn.ReflectionPad2d(4),
            nn.Conv2d(ngf, 3, kernel_size=9),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x) + x
        x = self.tail(x)
        return (x + 1) / 2


class SRGANBlock(nn.Module):
    """
    Building block of SRGAN.
    """

    def __init__(self, dim):
        super(SRGANBlock, self).__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.BatchNorm2d(dim),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        return x + self.net(x)


class UpscaleBlock(nn.Sequential):
    """
    Upscale block using sub-pixel convolutions.
    `scale_factor` can be selected from {2, 3, 4, 8}.
    """

    def __init__(self, scale_factor, dim, act=None):
        assert scale_factor in [2, 3, 4, 8]

        layers = []
        for _ in range(int(log2(scale_factor))):
            r = 2 if scale_factor % 2 == 0 else 3
            layers += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim * r * r, kernel_size=3),
                nn.PixelShuffle(r),
            ]

            if act == "relu":
                layers += [nn.ReLU(True)]
            elif act == "prelu":
                layers += [nn.PReLU()]

        super(UpscaleBlock, self).__init__(*layers)
