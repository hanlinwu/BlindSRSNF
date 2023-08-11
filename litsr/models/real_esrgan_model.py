import random
import time

import numpy as np
import torch
from litsr.data.real_esrgan_degrade import (
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
)
from litsr.data.srmd_degrade import SRMDPreprocessing
from litsr.metrics import calc_psnr_ssim
from litsr.models.srgan_model import SRGANModel
from litsr.transforms import paired_random_crop, tensor2uint8
from litsr.utils import DiffJPEG, USMSharp, filter2D
from litsr.utils.logger import logger
from litsr.utils.registry import ModelRegistry
from torch.nn import functional as F


@ModelRegistry.register()
class RealESRGANModel(SRGANModel):
    """
    Enhanced GAN
    https://arxiv.org/abs/1809.00219
    Change the Discriminator to Relativistic Discriminator
    """

    def __init__(self, opt):
        super(RealESRGANModel, self).__init__(opt)
        self.jpeger = DiffJPEG(
            differentiable=False
        ).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get("queue_size", 180)
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

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.
        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, "queue_lr"):
            assert (
                self.queue_size % b == 0
            ), f"queue size {self.queue_size} should be divisible by batch size {b}"
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[
                self.queue_ptr : self.queue_ptr + b, :, :, :
            ] = self.lq.clone()
            self.queue_gt[
                self.queue_ptr : self.queue_ptr + b, :, :, :
            ] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data, is_train=True):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images."""
        if is_train:
            # training data synthesis
            self.gt = data["gt"].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data["kernel1"].to(self.device)
            self.kernel2 = data["kernel2"].to(self.device)
            self.sinc_kernel = data["sinc_kernel"].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt_usm, self.kernel1)
            # random resize
            updown_type = random.choices(
                ["up", "down", "keep"], self.opt["resize_prob"]
            )[0]
            if updown_type == "up":
                scale = np.random.uniform(1, self.opt["resize_range"][1])
            elif updown_type == "down":
                scale = np.random.uniform(self.opt["resize_range"][0], 1)
            else:
                scale = 1
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.opt["gray_noise_prob"]
            if np.random.uniform() < self.opt["gaussian_noise_prob"]:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.opt["noise_range"],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt["poisson_scale_range"],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False,
                )
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range"])
            out = torch.clamp(
                out, 0, 1
            )  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt["second_blur_prob"]:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(
                ["up", "down", "keep"], self.opt["resize_prob2"]
            )[0]
            if updown_type == "up":
                scale = np.random.uniform(1, self.opt["resize_range2"][1])
            elif updown_type == "down":
                scale = np.random.uniform(self.opt["resize_range2"][0], 1)
            else:
                scale = 1
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(
                out,
                size=(
                    int(ori_h / self.opt["scale"] * scale),
                    int(ori_w / self.opt["scale"] * scale),
                ),
                mode=mode,
            )
            # add noise
            gray_noise_prob = self.opt["gray_noise_prob2"]
            if np.random.uniform() < self.opt["gaussian_noise_prob2"]:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.opt["noise_range2"],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt["poisson_scale_range2"],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False,
                )

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(["area", "bilinear", "bicubic"])
                out = F.interpolate(
                    out,
                    size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]),
                    mode=mode,
                )
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range2"])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range2"])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(["area", "bilinear", "bicubic"])
                out = F.interpolate(
                    out,
                    size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]),
                    mode=mode,
                )
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

            # random crop
            gt_size = self.opt["gt_size"]
            (self.gt, self.gt_usm), self.lq = paired_random_crop(
                [self.gt, self.gt_usm], self.lq, gt_size, self.opt["scale"]
            )
            # training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = (
                self.lq.contiguous()
            )  # for the warning: grad and param do not obey the gradient layout contract
        else:
            # for paired training or validation
            self.lq = data["lq"].to(self.device)
            if "gt" in data:
                self.gt = data["gt"].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

    def training_step(self, batch, batch_idx):
        self.feed_data(batch, is_train=True)
        lr = self.lq
        l1_gt = self.gt_usm if self.opt["l1_gt_usm"] else self.gt
        percep_gt = self.gt_usm if self.opt["percep_gt_usm"] else self.gt
        gan_gt = self.gt_usm if self.opt["gan_gt_usm"] else self.gt

        d_opt, g_opt = self.optimizers()

        ##################
        # train generator
        ##################
        for p in self.net_D.parameters():
            p.requires_grad = False

        g_opt.zero_grad()
        sr = self.net_G(lr)

        # content loss
        pix_loss = self.loss_Pix(sr, l1_gt)
        vgg_loss = self.loss_VGG(sr, percep_gt)
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

        ##################
        # train the discriminator
        ##################
        for p in self.net_D.parameters():
            p.requires_grad = True

        d_opt.zero_grad()

        # for real image
        d_out_real = self.net_D(gan_gt)
        d_loss_real = self.loss_GAN(d_out_real, True)

        self.manual_backward(d_loss_real)

        # for fake image
        d_out_fake = self.net_D(sr.detach())
        d_loss_fake = self.loss_GAN(d_out_fake, False)
        self.manual_backward(d_loss_fake)
        d_opt.step()

        self.log("d_loss", d_loss_real.item() + d_loss_fake.item(), prog_bar=True)

        return

    def validation_step(self, batch, *args):
        hr, name = batch
        lr = self.valid_degrade(hr)
        sr = self.forward(lr).detach()

        loss = self.loss_Pix(sr, hr).item()
        sr_np, hr_np = tensor2uint8(
            [sr.cpu()[0], hr.cpu()[0]], self.hparams.data_module.args.rgb_range
        )

        psnr, ssim = calc_psnr_ssim(
            sr_np, hr_np, crop_border=self.opt.valid.scale, test_Y=True
        )
        return {
            "val_loss": loss,
            "val_psnr": psnr,
            "val_ssim": ssim,
            "log_img_sr": sr_np,
            "name": name[0],
        }

    def test_step(self, batch, batch_idx, *args, **kwargs):
        _, hr, name = batch
        lr = self.valid_degrade(hr)
        sr = self.forward(lr).detach()

        loss = self.loss_Pix(sr, hr)
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
