import torch
from litsr.models.srgan_model import SRGANModel
from litsr.utils.registry import ModelRegistry


@ModelRegistry.register()
class ESRGANModel(SRGANModel):
    """
    Enhanced GAN
    https://arxiv.org/abs/1809.00219
    Change the Discriminator to Relativistic Discriminator
    """

    def training_step(self, batch, batch_idx):
        lr, hr = batch

        d_opt, g_opt = self.optimizers()

        sr = self.net_G(lr)

        ##################
        # train the discriminator
        ##################
        for p in self.net_D.parameters():
            p.requires_grad = True

        d_opt.zero_grad()

        # for real image
        d_out_fake = self.net_D(sr).detach()
        d_out_real = self.net_D(hr)
        d_loss_real = self.loss_GAN(d_out_real - torch.mean(d_out_fake), True) * 0.5
        self.manual_backward(d_loss_real)

        # for fake image
        d_out_fake = self.net_D(sr.detach())
        d_loss_fake = (
            self.loss_GAN(d_out_fake - torch.mean(d_out_real.detach()), False) * 0.5
        )
        self.manual_backward(d_loss_fake)

        d_opt.step()

        self.log("d_loss", d_loss_real.item() + d_loss_fake.item(), prog_bar=True)

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
