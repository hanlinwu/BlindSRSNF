import functools
import time

import numpy as np
import pytorch_lightning as pl
import torch as th
from archs import ArchRegistry, create_net, load_or_create_net
from litsr.data.srmd_degrade import SRMDPreprocessing
from litsr.metrics import calc_psnr_ssim
from litsr.models.utils import accumulate
from litsr.transforms import denormalize, normalize, tensor2uint8
from litsr.utils.diffusion import Diffusion
from litsr.utils.moco_builder import MoCo
from torch.nn import functional as F
from torch.nn import init

from models import ModelRegistry

####################
# sample function
####################


def fast_samples_fn(
    model, diffusion, ctx, k_v, shape, device, skip=10, eta=1.0, include_x0_pred_freq=10
):
    samples, progressive_samples = diffusion.p_sample_loop_fast(
        model=model,
        ctx=ctx,
        k_v=k_v,
        shape=shape,
        device=device,
        noise_fn=th.randn,
        skip=skip,
        eta=eta,
        include_x0_pred_freq=include_x0_pred_freq,
    )
    return {
        "samples": samples,
        "progressive_samples": progressive_samples,
    }


def fastdpm_var_samples_fn(
    model, diffusion, ctx, k_v, shape, device, skip=10, eta=1.0, include_x0_pred_freq=10
):
    num_steps = diffusion.num_timesteps // skip
    samples = diffusion.p_sample_loop_fastdpm_var(
        model=model,
        ctx=ctx,
        k_v=k_v,
        shape=shape,
        device=device,
        noise_fn=th.randn,
        num_steps=num_steps,
        schedule="quadratic",
        eta=eta,
        include_x0_pred_freq=include_x0_pred_freq,
    )
    return {"samples": samples}


####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type="kaiming", scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print("Initialization method [{:s}]".format(init_type))
    if init_type == "normal":
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == "kaiming":
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == "orthogonal":
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            "initialization method [{:s}] not implemented".format(init_type)
        )


@ModelRegistry.register()
class DiffBlindSR3AtomJIFModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters(opt)
        self.opt = opt.lit_model.args

        self.pretrain_epochs = self.hparams.trainer.pretrain_epochs

        # init super-resolution network
        self.encoder = load_or_create_net(self.opt.encoder)
        self.dg_encoder = ArchRegistry.get(self.opt.dg_encoder)
        self.E = MoCo(base_encoder=self.dg_encoder)
        self.model = create_net(self.opt.diffusion_model)
        init_weights(self.model, init_type="orthogonal")

        if self.opt.get("ema", False):
            self.ema = create_net(self.opt.diffusion_model)
            init_weights(self.ema, init_type="orthogonal")
        else:
            self.ema = None

        # loss weight
        self.contrast_loss_weight = self.opt.get("contrast_loss_weight", 0.01)

        diff_conf = self.opt.diffusion

        assert diff_conf.t_encode_mode == "continuous"
        self.diffusion = Diffusion(**diff_conf)
        self.num_steps = len(self.diffusion.betas)

        self.mean, self.std = self.opt.mean, self.opt.std
        self.rgb_range = self.hparams.data_module.args.rgb_range
        self.normalize = lambda tensor: normalize(
            tensor, self.mean, self.std, inplace=True
        )
        self.denormalize = lambda tensor: denormalize(
            tensor, self.mean, self.std, inplace=True
        )

        self.degrade = SRMDPreprocessing(
            self.opt.scale,
            kernel_size=self.opt.blur_kernel,
            blur_type=self.opt.blur_type,
            sig_min=self.opt.sig_min,
            sig_max=self.opt.sig_max,
            lambda_min=self.opt.lambda_min,
            lambda_max=self.opt.lambda_max,
            noise=self.opt.noise,
        )
        self.valid_degrade = SRMDPreprocessing(
            self.opt.scale,
            kernel_size=self.opt.valid.blur_kernel,
            blur_type=self.opt.valid.blur_type,
            sig=self.opt.valid.get("sig"),
            lambda_1=self.opt.valid.get("lambda_1"),
            lambda_2=self.opt.valid.get("lambda_2"),
            theta=self.opt.valid.get("theta"),
            noise=self.opt.valid.get("noise"),
        )

        self.contrast_loss = th.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        # if "hr" in batch and "lr" in batch:
        #     hr, lr = batch["hr"], batch["lr"]
        # elif "hr" in batch:
        #     hr = batch["hr"]
        #     hr.mul_(255.0)
        #     lr, _ = self.degrade(hr)
        #     hr.div_(255.0)
        #     lr.div_(255.0)
        # else:
        #     hr = batch
        hr = batch
        hr.mul_(255.0)
        lr, _ = self.degrade(hr)
        hr.div_(255.0)
        lr.div_(255.0)

        # Normalize hr and lr
        self.normalize(hr)
        self.normalize(lr)

        im_q = lr[:, 0, ...]
        im_k = lr[:, 1, ...]
        hr = hr[:, 0, ...]

        if self.current_epoch < self.pretrain_epochs:
            _, output, target = self.E(im_q, im_k)
            loss_contrast = self.contrast_loss_weight * self.contrast_loss(
                output, target
            )
            hr_, ctx = self.encoder(im_q, return_features=True)
            loss_consis = F.l1_loss(hr_, hr)
            loss = loss_consis + loss_contrast
            loss_elbo = 0
        else:
            fea, output, target = self.E(im_q, im_k)
            loss_contrast = self.contrast_loss_weight * self.contrast_loss(
                output, target
            )

            hr_, ctx = self.encoder(im_q, return_features=True)
            loss_consis = F.l1_loss(hr_, hr)

            t = (th.rand(hr.shape[0], device=hr.device) * self.num_steps).type(th.int64)
            loss_elbo = self.diffusion.get_loss(self.model, hr, t, ctx=ctx, k_v=fea)
            loss = loss_consis + loss_elbo + loss_contrast

            if self.ema:
                accumulate(
                    self.ema,
                    self.model.module
                    if isinstance(self.model, th.nn.DataParallel)
                    else self.model,
                    self.opt.ema_rate,
                )

        self.log("train/loss_contrast", loss_contrast)
        self.log("train/loss_elbo", loss_elbo)
        self.log("train/loss_consis", loss_consis)

        return loss

    def validation_step(self, batch, *args, **kwargs):
        # choose use ema or not
        _model = self.ema if self.ema else self.model
        if "hr" in batch and "lr" in batch:
            hr, lr, name = batch["hr"], batch["lr"], batch["name"]
        elif "hr" in batch:
            if len(hr.shape) == 4:
                hr = hr.unsqueeze(1)
            hr.mul_(255.0)
            lr, _ = self.valid_degrade(hr, random=False)
            hr.div_(255.0)
            lr.div_(255.0)
        else:
            lr, hr, name = batch
            # raise NotImplementedError

        # Normalize hr and lr
        self.normalize(hr)
        self.normalize(lr)

        lr, hr = lr.squeeze(1), hr.squeeze(1)

        th.cuda.synchronize()
        tic = time.time()

        fea = self.E(lr, lr)
        _, ctx = self.encoder(lr, return_features=True)

        skip = self.opt.valid.skip
        eta = self.opt.valid.eta

        approxdiff = kwargs.get("approxdiff", "STEP")
        if approxdiff == "STEP":
            sample = fast_samples_fn(
                _model, self.diffusion, ctx, fea, hr.shape, self.device, skip, eta
            )
        elif approxdiff == "VAR":
            sample = fastdpm_var_samples_fn(
                _model, self.diffusion, ctx, fea, hr.shape, self.device, skip, eta
            )
        else:
            raise NotImplementedError

        th.cuda.synchronize()
        toc = time.time()

        sr = sample["samples"]

        self.denormalize(sr)
        self.denormalize(hr)
        self.denormalize(lr)

        crop_border = int(np.ceil(float(hr.shape[2]) / lr.shape[2]))
        sr_np, hr_np, lr_np = tensor2uint8(
            [sr.cpu()[0], hr.cpu()[0], lr.cpu()[0]], self.rgb_range
        )
        psnr, ssim = calc_psnr_ssim(
            sr_np, hr_np, crop_border=crop_border, test_Y=self.opt.valid.test_Y
        )
        return {
            "val_psnr": psnr,
            "val_ssim": ssim,
            "log_img_sr": sr_np,
            "log_img_lr": lr_np,
            "name": name[0],
            "time": toc - tic,
        }

    def validation_epoch_end(self, outputs):
        # avg_val_loss = th.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_psnr = np.array([x["val_psnr"] for x in outputs]).mean()
        # self.log("val/loss", avg_val_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/psnr", avg_val_psnr, on_epoch=True, prog_bar=True, logger=True)

        log_img_sr = outputs[0]["log_img_sr"]
        self.logger.experiment.add_image(
            "img_sr", log_img_sr, self.global_step, dataformats="HWC"
        )

        return

    def test_step(self, batch, *args, **kwargs):
        return self.validation_step(batch, *args, **kwargs)

    def test_step_lr_only(self, batch, *args):
        # choose use ema or not
        _model = self.ema if self.ema else self.model

        lr, name = batch
        hr_shape = [lr.shape[0], lr.shape[1], lr.shape[2] * 4, lr.shape[3] * 4]
        self.normalize(lr)

        th.cuda.synchronize()
        tic = time.time()

        fea = self.E(lr, lr)
        _, ctx = self.encoder(lr, return_features=True)

        skip = self.opt.valid.skip
        eta = self.opt.valid.eta
        sample = fast_samples_fn(
            _model, self.diffusion, ctx, fea, hr_shape, self.device, skip, eta
        )

        th.cuda.synchronize()
        toc = time.time()
        sr = sample["samples"]
        self.denormalize(sr)
        self.denormalize(lr)

        [sr_np, lr_np] = tensor2uint8([sr.cpu()[0], lr.cpu()[0]], self.rgb_range)

        return {
            "log_img_sr": sr_np,
            "log_img_lr": lr_np,
            "name": name[0],
            "time": toc - tic,
        }

    def test_step_lr_hr_paired(self, batch, *args, **kwargs):
        # choose use ema or not
        _model = self.ema if self.ema else self.model
        lr, hr, name = batch
        self.normalize(hr)
        self.normalize(lr)

        th.cuda.synchronize()
        tic = time.time()

        fea = self.E(lr, lr)
        _, ctx = self.encoder(lr, return_features=True)

        skip = self.opt.valid.skip
        eta = self.opt.valid.eta

        approxdiff = kwargs.get("approxdiff", "STEP")
        if approxdiff == "STEP":
            sample = fast_samples_fn(
                _model, self.diffusion, ctx, fea, hr.shape, self.device, skip, eta
            )
        elif approxdiff == "VAR":
            sample = fastdpm_var_samples_fn(
                _model, self.diffusion, ctx, fea, hr.shape, self.device, skip, eta
            )
        else:
            raise NotImplementedError

        th.cuda.synchronize()
        toc = time.time()

        sr = sample["samples"]

        self.denormalize(sr)
        self.denormalize(hr)
        self.denormalize(lr)

        crop_border = int(np.ceil(float(hr.shape[2]) / lr.shape[2]))
        sr_np, hr_np, lr_np = tensor2uint8(
            [sr.cpu()[0], hr.cpu()[0], lr.cpu()[0]], self.rgb_range
        )
        psnr, ssim = calc_psnr_ssim(
            sr_np, hr_np, crop_border=crop_border, test_Y=self.opt.valid.test_Y
        )
        return {
            "val_psnr": psnr,
            "val_ssim": ssim,
            "log_img_sr": sr_np,
            "log_img_lr": lr_np,
            "name": name[0],
            "time": toc - tic,
        }

    def configure_optimizers(self):
        betas = self.opt.optimizer.get("betas") or (0.9, 0.999)
        optimizer = th.optim.Adam(
            self.parameters(), lr=self.opt.optimizer.lr, betas=betas
        )
        if self.opt.optimizer.get("lr_scheduler_step"):
            LR_scheduler = th.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.opt.optimizer.lr_scheduler_step,
                gamma=self.opt.optimizer.lr_scheduler_gamma,
            )
        elif self.opt.optimizer.get("lr_scheduler_milestones"):
            LR_scheduler = th.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.opt.optimizer.lr_scheduler_milestones,
                gamma=self.opt.optimizer.lr_scheduler_gamma,
            )
        else:
            raise Exception("No lr settings found! ")
        return [optimizer], [LR_scheduler]
