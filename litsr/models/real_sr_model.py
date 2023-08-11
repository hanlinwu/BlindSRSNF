import random

import numpy as np
import torch
from litsr.data import real_esrgan_degrade as deg
from litsr.data.srmd_degrade import SRMDPreprocessing
from litsr.metrics import calc_psnr_ssim
from litsr.models.sr_model import SRModel
from litsr.transforms import paired_random_crop, tensor2uint8
from litsr.utils import DiffJPEG, USMSharp, filter2D
from litsr.utils.registry import ModelRegistry
from torch.nn import functional as F


@ModelRegistry.register()
class RealSRModel(SRModel):
    """
    SR Model for real esrgan
    """

    def __init__(self, opt):
        """
        opt: in_channels, out_channels, num_features, num_blocks, num_layers
        """
        super(RealSRModel, self).__init__(opt)

        self.srmd_processing = SRMDPreprocessing(
            self.opt.valid.scale,
            kernel_size=self.opt.valid.blur_kernel,
            blur_type=self.opt.valid.blur_type,
            sig=self.opt.valid.get("sig"),
            lambda_1=self.opt.valid.get("lambda_1"),
            lambda_2=self.opt.valid.get("lambda_2"),
            theta=self.opt.valid.get("theta"),
            noise=self.opt.valid.noise,
        )

    def valid_degrade(self, hr):
        hr = hr * 255.0
        lr, _ = self.srmd_processing(hr, random=False)
        lr = lr / 255.0
        return lr

    def validation_step(self, batch, batch_idx):
        hr, name = batch
        lr = self.valid_degrade(hr)
        sr = self.forward(lr).detach()

        loss = self.loss_fn(sr, hr).item()
        sr_np, hr_np = tensor2uint8(
            [sr.cpu()[0], hr.cpu()[0]], self.hparams.data_module.args.rgb_range
        )

        lr_np = tensor2uint8(lr.cpu()[0], self.hparams.data_module.args.rgb_range)

        psnr, ssim = calc_psnr_ssim(
            sr_np, hr_np, crop_border=self.opt.valid.scale, test_Y=True
        )
        return {
            "val_loss": loss,
            "val_psnr": psnr,
            "val_ssim": ssim,
            "log_img_sr": sr_np,
            "log_img_lr": lr_np,
            "name": name[0],
        }

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
