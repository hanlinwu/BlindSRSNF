import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from litsr.archs import create_net
from litsr.metrics import calc_psnr_ssim
from litsr.transforms import tensor2uint8
from litsr.utils.logger import logger
from litsr.utils.registry import ModelRegistry
from torch.nn import functional as F
from thop import profile


@ModelRegistry.register()
class SRModel(pl.LightningModule):
    """
    Basic SR Model optimized by pixel-wise loss
    """

    def __init__(self, opt):
        """
        opt: in_channels, out_channels, num_features, num_blocks, num_layers
        """
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters(opt)
        self.opt = opt.lit_model.args

        # init super-resolution network
        self.sr_net = create_net(self.opt["network"])

        # define loss function (L1 loss)
        self.loss_fn = nn.L1Loss()

    def forward(self, lr):
        return self.sr_net(lr)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            lr, hr = batch["lr"], batch["hr"]
        else:
            lr, hr = batch
        sr = self.forward(lr)
        loss = self.loss_fn(sr, hr)
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([out["loss"] for out in outputs]).mean()
        self.log("train/loss", avg_loss, on_epoch=True)
        return

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        if isinstance(batch, dict):
            lr, hr, name = batch["lr"], batch["hr"], batch["file_name"]
        else:
            lr, hr, name = batch
        sr = self.forward(lr).detach()
        loss = self.loss_fn(sr, hr).item()
        sr_np, hr_np = tensor2uint8([sr.cpu()[0], hr.cpu()[0]], self.opt.rgb_range)

        if kwargs.get("no_crop_border", self.opt.valid.get("no_crop_border")):
            crop_border = 0
        else:
            crop_border = int(np.ceil(float(hr.shape[2]) / lr.shape[2]))

        test_Y = kwargs.get("test_Y", self.opt.valid.get("test_Y", True))

        if batch_idx == 0:
            logger.warning("Test_Y: {0}, crop_border: {1}".format(test_Y, crop_border))

        psnr, ssim = calc_psnr_ssim(
            sr_np, hr_np, crop_border=crop_border, test_Y=test_Y
        )

        return {
            "val_loss": loss,
            "val_psnr": psnr,
            "val_ssim": ssim,
            "log_img_sr": sr_np,
            "name": name[0],
        }

    def validation_epoch_end(self, outputs):
        tensorboard = self.logger.experiment
        log_img = outputs[0]["log_img_sr"]

        avg_loss = np.array([x["val_loss"] for x in outputs]).mean()
        avg_psnr = np.array([x["val_psnr"] for x in outputs]).mean()

        self.log("val/loss", avg_loss, on_epoch=True)
        self.log("val/psnr_x{0}".format(self.opt.valid.scale), avg_psnr, on_epoch=True)
        tensorboard.add_image(
            "images/{0}".format(self.opt.valid.scale),
            log_img,
            self.global_step,
            dataformats="HWC",
        )

        self.log("val/psnr", avg_psnr, on_epoch=True, prog_bar=True, logger=False)
        return

    def test_step(self, batch, batch_idx, *args, **kwargs):
        if len(batch) == 2:
            lr, name = batch
        else:
            lr, hr, name = batch
        torch.cuda.synchronize()
        start = time.time()
        sr = self.forward(lr).detach()
        torch.cuda.synchronize()
        end = time.time()
        if len(batch) == 2:
            sr_np = tensor2uint8(sr.cpu()[0], self.opt.rgb_range)
            return {"log_img_sr": sr_np, "name": name[0]}
        else:
            loss = self.loss_fn(sr, hr).item()
            sr_np, hr_np = tensor2uint8([sr.cpu()[0], hr.cpu()[0]], self.opt.rgb_range)

            if kwargs.get("no_crop_border", self.opt.valid.get("no_crop_border")):
                crop_border = 0
            else:
                crop_border = int(np.ceil(float(hr.shape[2]) / lr.shape[2]))

            test_Y = kwargs.get("test_Y", self.opt.valid.get("test_Y", True))

            if batch_idx == 0:
                logger.warning(
                    "Test_Y: {0}, crop_border: {1}".format(test_Y, crop_border)
                )

            psnr, ssim = calc_psnr_ssim(
                sr_np, hr_np, crop_border=crop_border, test_Y=test_Y
            )
            flops, params = profile(self.sr_net, inputs=(lr,), verbose=False)

            return {
                "val_loss": loss,
                "val_psnr": psnr,
                "val_ssim": ssim,
                "log_img": sr_np,
                "name": name[0],
                "time": end - start,
                "flops": flops,
                "params": params,
            }

    def test_step_bic(self, batch, batch_idx, *args, **kwargs):
        lr, hr, name = batch
        h, w = hr.shape[2:]
        torch.cuda.synchronize()
        start = time.time()

        sr = self.forward(lr).detach()
        sr = F.interpolate(sr, (h, w), mode="bicubic", align_corners=False)

        torch.cuda.synchronize()
        end = time.time()

        loss = self.loss_fn(sr, hr).item()
        sr_np, hr_np = tensor2uint8([sr.cpu()[0], hr.cpu()[0]], self.opt.rgb_range)

        if kwargs.get("no_crop_border", self.opt.valid.get("no_crop_border")):
            crop_border = 0
        else:
            crop_border = int(np.ceil(float(hr.shape[2]) / lr.shape[2]))

        test_Y = kwargs.get("test_Y", self.opt.valid.get("test_Y", True))

        if batch_idx == 0:
            logger.warning("Test_Y: {0}, crop_border: {1}".format(test_Y, crop_border))

        psnr, ssim = calc_psnr_ssim(
            sr_np, hr_np, crop_border=crop_border, test_Y=test_Y
        )
        return {
            "val_loss": loss,
            "val_psnr": psnr,
            "val_ssim": ssim,
            "log_img": sr_np,
            "name": name[0],
            "time": end - start,
        }

    def configure_optimizers(self):
        betas = self.opt.optimizer.get("betas") or (0.9, 0.999)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.opt.optimizer.lr, betas=betas
        )
        if self.opt.optimizer.get("lr_scheduler_step"):
            LR_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.opt.optimizer.lr_scheduler_step,
                gamma=self.opt.optimizer.lr_scheduler_gamma,
            )
        elif self.opt.optimizer.get("lr_scheduler_milestones"):
            LR_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.opt.optimizer.lr_scheduler_milestones,
                gamma=self.opt.optimizer.lr_scheduler_gamma,
            )
        else:
            raise Exception("No lr settings found! ")
        return [optimizer], [LR_scheduler]
