import torch
import torch.nn as nn
import torch.optim as optim

from models.discriminators import NLayerDiscriminator
from models.losses import GANLoss
from models.geo_loss import geometry_ensemble
from models.pseudo_model import Pseudo_Model

class Face_Model(Pseudo_Model):
    def __init__(self, device, cfg, use_ddp=False):
        super(Face_Model, self).__init__(device=device, cfg=cfg, use_ddp=use_ddp)
        self.use_esrgan = cfg.SR.MODEL == "ESRGAN"
        self.sr_warmup_iter = cfg.OPT_SR.WARMUP
        if self.use_esrgan:
            self.D_esrgan = NLayerDiscriminator(1, scale_factor=1, norm_layer=nn.InstanceNorm2d).to(device)
            self.opt_D_esrgan = optim.Adam(self.D_esrgan.parameters(), lr=cfg.OPT_SR.LR_D, betas=cfg.OPT_SR.BETAS_D)
            self.lr_D_esrgan = optim.lr_scheduler.MultiStepLR(self.opt_D_esrgan, milestones=cfg.OPT_SR.LR_MILESTONE, gamma=cfg.OPT_SR.LR_DECAY)
            self.nets["D_esrgan"] = self.D_esrgan
            self.optims["D_esrgan"] = self.opt_D_esrgan
            self.lr_decays["D_esrgan"] = self.lr_D_esrgan
            self.discs.append("D_esrgan")

            self.ragan_loss = GANLoss("vanilla")

    def warmup_checker(self):
        return self.n_iter <= self.sr_warmup_iter

    def lr_decay_step(self, shout=False):
        lrs = "\nLearning rates: "
        changed = False
        for i, n in enumerate(self.lr_decays):
            if self.warmup_checker() and n == "D_esrgan":
                continue
            lr_old = self.lr_decays[n].get_last_lr()[0]
            self.lr_decays[n].step()
            lr_new = self.lr_decays[n].get_last_lr()[0]
            if lr_old != lr_new:
                changed = True
                lrs += f", {n}={self.lr_decays[n].get_last_lr()[0]}" if i > 0 else f"{n}={self.lr_decays[n].get_last_lr()[0]}"
        if shout and changed: print(lrs)

    def train_step(self, Ys, Xs, Yds, Zs):
        '''
        Ys: high resolutions
        Xs: low resolutions
        Yds: down sampled HR
        Zs: noises
        '''
        self.n_iter += 1
        loss_dict = dict()

        # forward
        # print(Yds.size(), Zs.size())
        fake_Xs = self.G_yx(Yds, Zs)
        rec_Yds = self.G_xy(fake_Xs)
        fake_Yds = self.G_xy(Xs)
        rec_Xs = self.G_yx(Yds, Zs)
        loss_frequency = self.ffl(rec_Yds, Yds) + self.ffl(rec_Xs, Xs)
        loss_dict["F"] = loss_frequency.item()
        geo_Yds = geometry_ensemble(self.G_xy, Xs)
        idt_out = self.G_xy(Yds) if self.idt_input_clean else fake_Yds

        self.net_grad_toggle(["D_x", "D_y"], True)
        # D_x
        pred_fake_Xs = self.D_x(fake_Xs.detach())
        pred_real_Xs = self.D_x(Xs)
        loss_D_x = (self.gan_loss(pred_real_Xs, True, True) + self.gan_loss(pred_fake_Xs, False, True)) * 0.5
        self.opt_Dx.zero_grad()
        loss_D_x.backward()
        self.opt_Dx.step()
        loss_dict["D_x"] = loss_D_x.item()

        # D_y
        pred_fake_Yds = self.D_y(fake_Yds.detach())
        pred_real_Yds = self.D_y(Yds)
        loss_D_y = (self.gan_loss(pred_real_Yds, True, True) + self.gan_loss(pred_fake_Yds, False, True)) * 0.5
        self.opt_Dy.zero_grad()
        loss_D_y.backward()
        self.opt_Dy.step()
        loss_dict["D_y"] = loss_D_y.item()

        self.net_grad_toggle(["D_x", "D_y"], False)
        # G_yx
        self.opt_Gyx.zero_grad()
        self.opt_Gxy.zero_grad()
        pred_fake_Xs = self.D_x(fake_Xs)
        loss_gan_Gyx = self.gan_loss(pred_fake_Xs, True, False)
        loss_dict["G_yx_gan"] = loss_gan_Gyx.item()

        # G_xy
        pred_fake_Yds = self.D_y(fake_Yds)
        loss_gan_Gxy = self.gan_loss(pred_fake_Yds, True, False)
        loss_idt_Gxy = self.l1_loss(idt_out, Yds) if self.idt_input_clean else self.l1_loss(idt_out, Xs)
        loss_cycle = self.l1_loss(rec_Yds, Yds)
        loss_geo = self.l1_loss(fake_Yds, geo_Yds)
        loss_total_gen = loss_gan_Gyx + loss_gan_Gxy + self.cyc_weight * loss_cycle + self.idt_weight * loss_idt_Gxy + self.geo_weight * loss_geo + self.fre_weight * loss_frequency
        loss_dict["G_xy_gan"] = loss_gan_Gxy.item()
        loss_dict["G_xy_idt"] = loss_idt_Gxy.item()
        loss_dict["cyc_loss"] = loss_cycle.item()
        loss_dict["G_xy_geo"] = loss_geo.item()
        loss_dict["G_total"] = loss_total_gen.item()

        # gen loss backward and update
        loss_total_gen.backward()
        self.opt_Gyx.step()
        self.opt_Gxy.step()
        return loss_dict

if __name__ == "__main__":
    from yacs.config import CfgNode
    with open("../configs/MRI.yaml", "rb") as cf:
        CFG = CfgNode.load_cfg(cf)
        CFG.freeze()
    device = 0
    x = torch.randn(2, 1, 48, 48, dtype=torch.float32, device=device)
    y = torch.randn(2, 1, 96, 96, dtype=torch.float32, device=device)
    yd = torch.randn(2, 1, 48, 48, dtype=torch.float32, device=device)
    z = torch.randn(2, 1, 12, 12, dtype=torch.float32, device=device)
    model = Face_Model(device, CFG)
    losses = model.train_step(y, x, yd, z)
    file_name = model.net_save(".", True)
    model.net_load(file_name)
    for i in range(110000):
        model.lr_decay_step(True)
    info = f"  1/(1):"
    for i, itm in enumerate(losses.items()):
        info += f", {itm[0]}={itm[1]:.3f}" if i > 0 else f" {itm[0]}={itm[1]:.3f}"
    print(info)
    print("fin")
