from torch import nn
import torch.nn.functional as F
from litsr.utils.registry import ArchRegistry


@ArchRegistry.register()
class SRCNN(nn.Module):
    def __init__(self, scale, in_channels, out_channels):
        """SRCNN

        Args:
            scale ([int | float]): scale factor 
            in_channels ([int]]): number of imput channels
            out_channels ([type]): number of output channels
        """
        super().__init__()

        self.scale = scale

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode="bicubic")
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
