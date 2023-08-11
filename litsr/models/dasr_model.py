import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from litsr.archs import create_net
from litsr.metrics import calc_psnr_ssim
from litsr.transforms import tensor2uint8
from litsr.utils.registry import ModelRegistry
from litsr.data.srmd_degrade import SRMDPreprocessing


@ModelRegistry.register()
class DASRModel(pl.LightningModule):
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
        self.model = create_net(self.opt["network"])
        # self.E = self.model.E
        self.contrast_loss = torch.nn.CrossEntropyLoss()
        self.pretrain_epochs = self.hparams.trainer.pretrain_epochs

        self.degrade = SRMDPreprocessing(
            self.opt.scale,
            kernel_size=self.opt.blur_kernel,
            blur_type=self.opt.blur_type,
            sig_min=self.opt.get("sig_min"),
            sig_max=self.opt.get("sig_max"),
            lambda_min=self.opt.get("lambda_min"),
            lambda_max=self.opt.get("lambda_max"),
            noise=self.opt.noise,
        )
        self.valid_degrade = SRMDPreprocessing(
            self.opt.scale,
            kernel_size=self.opt.blur_kernel,
            blur_type=self.opt.blur_type,
            sig=self.opt.valid.get("sig"),
            lambda_1=self.opt.valid.get("lambda_1"),
            lambda_2=self.opt.valid.get("lambda_2"),
            theta=self.opt.valid.get("theta"),
            noise=self.opt.valid.noise,
        )

        # define loss function (L1 loss)
        self.loss_fn = nn.L1Loss()

    def forward(self, lr):
        return self.model(lr)

    def training_step(self, batch, batch_idx):
        hr = batch
        lr, _ = self.degrade(hr)

        if self.current_epoch < self.pretrain_epochs:
            _, output, target = self.model.E(im_q=lr[:, 0, ...], im_k=lr[:, 1, ...])
            loss_constrast = self.contrast_loss(output, target)
            loss = loss_constrast

            self.log("train/loss_contrast", loss_constrast)
        else:
            sr, output, target = self.model(lr)
            loss_SR = self.loss_fn(sr, hr[:, 0, ...])
            loss_constrast = self.contrast_loss(output, target)
            loss = loss_constrast + loss_SR

            self.log("train/loss_contrast", loss_constrast)
            self.log("train/loss_sr", loss_SR)

        return loss

    def validation_step(self, batch, *args):
        hr, name = batch

        hr, name = batch
        lr, _ = self.valid_degrade(hr, random=False)

        lr, hr = lr.squeeze(1), hr.squeeze(1)

        torch.cuda.synchronize()
        start = time.time()
        sr = self.forward(lr).detach()
        torch.cuda.synchronize()
        end = time.time()

        loss = self.loss_fn(sr, hr)
        sr_np, hr_np = tensor2uint8(
            [sr.cpu()[0], hr.cpu()[0]], self.hparams.data_module.args.rgb_range
        )
        psnr, ssim = calc_psnr_ssim(
            sr_np, hr_np, crop_border=self.opt.scale, test_Y=False
        )
        return {
            "val_loss": loss,
            "val_psnr": psnr,
            "val_ssim": ssim,
            "log_img_sr": sr_np,
            "name": name[0],
            "time": end - start,
        }

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_psnr = np.array([x["val_psnr"] for x in outputs]).mean()
        self.log("val/loss", avg_val_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/psnr", avg_val_psnr, on_epoch=True, prog_bar=True, logger=True)

        log_img_sr = outputs[0]["log_img_sr"]

        self.logger.experiment.add_image(
            "img_sr", log_img_sr, self.global_step, dataformats="HWC"
        )

        return

    def test_step(self, batch, *args):
        if len(batch) == 3:
            lr, hr, name = batch
        elif len(batch) == 2:
            hr, name = batch
            lr, _ = self.valid_degrade(hr.unsqueeze(1), random=False)

        lr = lr.squeeze(1)

        torch.cuda.synchronize()
        start = time.time()
        sr = self.forward(lr).detach()
        torch.cuda.synchronize()
        end = time.time()

        loss = self.loss_fn(sr, hr)
        sr_np, hr_np = tensor2uint8(
            [sr.cpu()[0], hr.cpu()[0]], self.hparams.data_module.args.rgb_range
        )
        lr_np = tensor2uint8(lr.cpu()[0], self.hparams.data_module.args.rgb_range)
        psnr, ssim = calc_psnr_ssim(
            sr_np, hr_np, crop_border=self.opt.scale, test_Y=True
        )
        return {
            "val_loss": loss,
            "val_psnr": psnr,
            "val_ssim": ssim,
            "log_img_sr": sr_np,
            "log_img_lr": lr_np,
            "name": name[0],
            "time": end - start,
        }

    def configure_optimizers(self):
        betas = self.opt.optimizer.get("betas") or (0.9, 0.999)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            betas=betas,
        )
        return optimizer

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        if epoch <= self.pretrain_epochs:
            lr = self.opt.optimizer.lr_encoder * (
                self.opt.optimizer.lr_encoder_gamma
                ** (epoch // self.opt.optimizer.lr_encoder_step)
            )
        else:
            lr = self.opt.optimizer.lr_sr * (
                self.opt.optimizer.lr_sr_gamma
                ** ((epoch - self.pretrain_epochs) // self.opt.optimizer.lr_sr_step)
            )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # update params
        optimizer.step(closure=optimizer_closure)
