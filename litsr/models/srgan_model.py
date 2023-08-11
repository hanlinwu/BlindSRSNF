import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from litsr.archs import load_or_create_net
from litsr.losses import GANLoss, VGGLoss
from litsr.metrics import calc_psnr_ssim
from litsr.transforms import tensor2uint8
from litsr.utils.logger import logger
from litsr.utils.registry import ModelRegistry
from thop import profile


@ModelRegistry.register()
class SRGANModel(pl.LightningModule):
    """
    GAN based SR model (Original SRGAN)
    """

    def __init__(self, opt):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters(opt)
        self.opt = opt.lit_model.args

        # init super-resolution network
        self.net_G = load_or_create_net(self.opt["net_G"])
        self.net_D = load_or_create_net(self.opt["net_D"])

        # define loss function (L1 loss)
        self.loss_Pix = {
            "MSE": nn.MSELoss(),
            "L1": nn.L1Loss(),
        }[self.opt.pix_loss_type]
        self.loss_VGG = VGGLoss(
            net_type=self.opt.vgg_loss_type, layer=self.opt.vgg_loss_layer
        )
        self.loss_GAN = GANLoss(gan_mode=self.opt.gan_loss_type)

        # import for GAN
        self.automatic_optimization = False

    def forward(self, lr):
        return self.net_G(lr)

    def training_step(self, batch, batch_idx):
        lr, hr = batch

        d_opt, g_opt = self.optimizers()

        ##################
        # train the discriminator
        ##################
        for p in self.net_D.parameters():
            p.requires_grad = True

        d_opt.zero_grad()

        sr = self.net_G(lr)

        # for real image
        d_out_real = self.net_D(hr)
        d_loss_real = self.loss_GAN(d_out_real, True)

        # for fake image
        d_out_fake = self.net_D(sr.detach())
        d_loss_fake = self.loss_GAN(d_out_fake, False)

        # combined discriminator loss
        d_loss = d_loss_real + d_loss_fake
        self.manual_backward(d_loss)
        d_opt.step()

        self.log("d_loss", d_loss, prog_bar=True)

        ##################
        # train generator
        ##################
        for p in self.net_D.parameters():
            p.requires_grad = False

        g_opt.zero_grad()
        # content loss
        pix_loss = self.loss_Pix(sr, hr)
        vgg_loss = self.loss_VGG(sr, hr)
        # adversarial loss
        adv_loss = self.loss_GAN(self.net_D(sr), True)

        # combined generator loss
        g_loss = (
            self.opt.vgg_loss_weight * vgg_loss
            + self.opt.pix_loss_weight * pix_loss
            + self.opt.gan_loss_weight * adv_loss
        )
        self.manual_backward(g_loss)
        g_opt.step()

        self.log("g_loss", g_loss, prog_bar=True)

        return

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        lr, hr, name = batch

        torch.cuda.synchronize()
        start = time.time()
        sr = self.forward(lr).detach()
        torch.cuda.synchronize()
        end = time.time()

        loss = self.loss_Pix(sr, hr).item()

        sr_np, hr_np = tensor2uint8([sr.cpu()[0], hr.cpu()[0]], self.opt.rgb_range)

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
            "time": end - start,
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
        lr, hr, name = batch

        torch.cuda.synchronize()
        start = time.time()
        sr = self.forward(lr).detach()
        torch.cuda.synchronize()
        end = time.time()

        loss = self.loss_Pix(sr, hr)

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
        # flops, params = profile(
        #     self.net_G,
        #     inputs=(lr),
        #     verbose=False,
        #     custom_ops={nn.ReLU: None},
        # )
        return {
            "val_loss": loss,
            "val_psnr": psnr,
            "val_ssim": ssim,
            "log_img_sr": sr_np,
            "name": name[0],
            "time": end - start,
            # "flops": flops,
            # "params": params,
        }

    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(
            self.net_G.parameters(), lr=self.opt.optimizer.lr_G
        )
        optimizer_D = torch.optim.Adam(
            self.net_D.parameters(), lr=self.opt.optimizer.lr_D
        )
        scheduler_G = torch.optim.lr_scheduler.StepLR(
            optimizer_G,
            step_size=self.opt.optimizer.scheduler_G_step,
            gamma=self.opt.optimizer.scheduler_G_gamma,
        )

        scheduler_D = torch.optim.lr_scheduler.StepLR(
            optimizer_D,
            step_size=self.opt.optimizer.scheduler_D_step,
            gamma=self.opt.optimizer.scheduler_D_gamma,
        )

        return [optimizer_D, optimizer_G], [scheduler_D, scheduler_G]
