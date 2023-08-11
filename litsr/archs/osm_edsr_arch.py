import torch
import torch.nn as nn
import torch.nn.functional as F
from litsr.utils.registry import ArchRegistry
from litsr.archs.edsr_arch import EDSR, default_conv, ResBlock


@ArchRegistry.register()
class OSM_EDSR(EDSR):
    def __init__(self, n_resblocks, n_feats, in_channels, res_scale):
        super().__init__()

        conv = default_conv

        kernel_size = 3
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(in_channels, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.OSMBlock = nn.Sequential(
            wn(nn.Conv2d(n_feats, 1600, 3, padding=1)),
            nn.PixelShuffle(5),
            wn(nn.Conv2d(64, 3, 3, padding=1)),
        )

    def forward(self, x, out_size):
        h = self.head(x)
        res = self.body(h)

        res = self.OSMBlock(res)

        x_1 = F.interpolate(res, out_size, mode="bicubic", align_corners=False)
        x_2 = F.interpolate(x, out_size, mode="bicubic", align_corners=False)

        out = x_1 + x_2

        return out
