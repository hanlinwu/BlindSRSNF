import torch
import torch.nn as nn
import torch.nn.functional as F
from litsr.utils.registry import ArchRegistry
from litsr.archs.rdn_arch import RDN


@ArchRegistry.register()
class OSM_RDN(RDN):
    def __init__(
        self,
        num_features,
        num_blocks,
        num_layers,
        rgb_range,
        in_channels,
        out_channels,
        rgb_mean=(0.4488, 0.4371, 0.4040),
        rgb_std=(1.0, 1.0, 1.0),
    ):
        super().__init__(
            scale=0,
            num_features=num_features,
            num_blocks=num_blocks,
            num_layers=num_layers,
            rgb_range=rgb_range,
            in_channels=in_channels,
            out_channels=out_channels,
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
        )
        # Redefine up-sampling net
        overscale = 5
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.OSMBlock = nn.Sequential(
            wn(nn.Conv2d(num_features, 1600, 3, padding=1)),
            nn.PixelShuffle(overscale),
            wn(nn.Conv2d(64, 3, 3, padding=1)),
        )

    def forward(self, x, out_size):
        h = self.sub_mean(x)
        f__1 = self.SFENet1(h)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))

        x = self.OSMBlock(x)
        x_1 = F.interpolate(x, out_size, mode="bicubic", align_corners=False)
        x_2 = F.interpolate(h, out_size, mode="bicubic", align_corners=False)

        out = x_1 + x_2
        out = self.add_mean(out)
        return out
