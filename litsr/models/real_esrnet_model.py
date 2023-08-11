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
class RealESRNetModel(SRModel):
    """
    SR Model for real esrgan
    """

    def __init__(self, opt):
        """
        opt: in_channels, out_channels, num_features, num_blocks, num_layers
        """
        super(RealESRNetModel, self).__init__(opt)

        self.usm_sharpener = USMSharp()
        self.jpeger = DiffJPEG(differentiable=False)
        self.queue_size = self.opt.get("queue_size", 180)
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
        # training pair pool
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, "queue_lr"):
            assert (
                self.queue_size % b == 0
            ), "queue size should be divisible by batch size"
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update
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
        if is_train:
            # training data synthesis
            self.gt = data["gt"]
            # USM the GT images
            if self.opt.get("gt_usm", True):
                self.gt = self.usm_sharpener(self.gt)

            self.kernel1 = data["kernel1"]
            self.kernel2 = data["kernel2"]
            self.sinc_kernel = data["sinc_kernel"]

            ori_h, ori_w = self.gt.size()[2:4]

            ###################################
            #### The first degradation process
            ###################################

            # blur
            out = filter2D(self.gt, self.kernel1)
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
            # noise
            gray_noise_prob = self.opt["gray_noise_prob"]
            if np.random.uniform() < self.opt["gaussian_noise_prob"]:
                out = deg.random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.opt["noise_range"],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                )
            else:
                out = deg.random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt["poisson_scale_range"],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False,
                )
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range"])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)

            ###################################
            #### The second degradation process
            ###################################

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
            # noise
            gray_noise_prob = self.opt["gray_noise_prob2"]
            if np.random.uniform() < self.opt["gaussian_noise_prob2"]:
                out = deg.random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.opt["noise_range2"],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                )
            else:
                out = deg.random_add_poisson_noise_pt(
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
                    size=(
                        ori_h // self.opt["scale"],
                        ori_w // self.opt["scale"],
                    ),
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
                    size=(
                        ori_h // self.opt["scale"],
                        ori_w // self.opt["scale"],
                    ),
                    mode=mode,
                )
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

            # random crop
            gt_size = self.opt["gt_size"]
            self.gt, self.lq = paired_random_crop(
                self.gt, self.lq, gt_size, self.opt["scale"]
            )

            # training pair pool
            self._dequeue_and_enqueue()
        else:
            self.lq = data["lq"].to(self.device)
            if "gt" in data:
                self.gt = data["gt"].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

    def training_step(self, batch, batch_idx):
        self.feed_data(batch)
        sr = self.forward(self.lq)
        loss = self.loss_fn(sr, self.gt)
        return loss

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
