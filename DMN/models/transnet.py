from models import common
import torch
import torch.nn as nn
from einops import rearrange,repeat
import torch.nn.functional as F
import numpy as np
from models.transformer import TransformerEncoder, TransformerDecoder

MIN_NUM_PATCHES = 12
def make_model(parent=False):
    return TransNet()

class BasicModule(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True,
                 bn=False, act=nn.ReLU(True)):
        super(BasicModule, self).__init__()

        m_body = []
        n_blocks = 5
        m_body = [
            common.ResBlock(conv, n_feat, kernel_size)
            for _ in range(n_blocks)
        ]
        self.body = nn.Sequential(*m_body)

    def forward(self, x):
        res = self.body(x)
        out = res
        return out

class TransNet(nn.Module):

    def __init__(self, conv=common.default_conv):
        super(TransNet, self).__init__()

        n_feats = 64
        kernel_size = 3
        act = nn.ReLU(True)

        # define head body
        m_head = [
            conv(in_channels=1, out_channels=n_feats, kernel_size=kernel_size),
        ]
        self.head = nn.Sequential(*m_head)

        # define main body
        self.feat_extrat_stage1 = BasicModule(conv, n_feats, kernel_size, act=act)
        self.feat_extrat_stage2 = BasicModule(conv, n_feats, kernel_size, act=act)
        self.feat_extrat_stage3 = BasicModule(conv, n_feats, kernel_size, act=act)
        self.feat_extrat_stage_noise = BasicModule(conv, n_feats, kernel_size, act=act)

        reduction = 4
        self.stage1_conv1x1 = conv(n_feats, n_feats // reduction, 1)
        self.stage2_conv1x1 = conv(n_feats, n_feats // reduction, 1)
        self.stage3_conv1x1 = conv(n_feats, n_feats // reduction, 1)
        self.stage_noise_conv1x1 = conv(n_feats, n_feats // reduction, 1)
        self.span_conv1x1 = conv(n_feats // reduction, n_feats, 1)

        # define tail body
        m_tail = [
            BasicModule(conv, n_feats, kernel_size, act=act),
            conv(n_feats, out_channels=1, kernel_size=kernel_size)
        ]
        self.tail = nn.Sequential(*m_tail)

        # define Transformer
        self.image_size = 48
        patch_size = 8
        dim = 512
        en_depth = 8
        de_depth = 2
        heads = 6
        mlp_dim = 512
        channels = n_feats // reduction
        dim_head = 32
        dropout = 0.0
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.patch_to_embedding_low1 = nn.Linear(patch_dim, dim)
        self.patch_to_embedding_low2 = nn.Linear(patch_dim, dim)
        self.patch_to_embedding_low3 = nn.Linear(patch_dim, dim)
        self.patch_to_embedding_low_noise = nn.Linear(patch_dim, dim)

        self.embedding_to_patch = nn.Linear(dim, patch_dim)

        self.encoder_stage1 = TransformerEncoder(dim, en_depth, heads, dim_head, mlp_dim, dropout)
        self.encoder_stage2 = TransformerEncoder(dim, en_depth, heads, dim_head, mlp_dim, dropout)
        self.encoder_stage3 = TransformerEncoder(dim, en_depth, heads, dim_head, mlp_dim, dropout)
        self.encoder_stage4 = TransformerEncoder(dim, en_depth, heads, dim_head, mlp_dim, dropout)

        self.decoder1 = TransformerDecoder(dim, de_depth, heads, dim_head, mlp_dim, dropout)
        self.decoder2 = TransformerDecoder(dim, de_depth, heads, dim_head, mlp_dim, dropout)
        self.decoder3 = TransformerDecoder(dim, en_depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):

        x = self.head(x)

        # feature extraction part
        feat_stage1 = self.feat_extrat_stage1(x)
        feat_stage2 = self.feat_extrat_stage2(x)
        feat_stage3 = self.feat_extrat_stage3(x)
        feat_stage4 = self.feat_extrat_stage_noise(x)

        feat_stage1 = self.stage1_conv1x1(feat_stage1)
        feat_stage2 = self.stage2_conv1x1(feat_stage2)
        feat_stage3 = self.stage3_conv1x1(feat_stage3)
        feat_stage4 = self.stage_noise_conv1x1(feat_stage4)

        # Transformer part
        p = self.patch_size
        feat_stage1 = rearrange(feat_stage1, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        feat_stage2 = rearrange(feat_stage2, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        feat_stage3 = rearrange(feat_stage3, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        feat_stage4 = rearrange(feat_stage4, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)

        feat_stage1 = self.patch_to_embedding_low1(feat_stage1)
        feat_stage2 = self.patch_to_embedding_low2(feat_stage2)
        feat_stage3 = self.patch_to_embedding_low3(feat_stage3)
        feat_stage4 = self.patch_to_embedding_low_noise(feat_stage4)

        # Encoder & Decoder
        feat_stage1 = self.encoder_stage1(feat_stage1)
        feat_stage2 = self.encoder_stage2(feat_stage2)
        feat_stage3 = self.encoder_stage3(feat_stage3)
        feat_stage4 = self.encoder_stage4(feat_stage4)
	
        feat_rec = self.decoder3(feat_stage1, feat_stage2)
        feat_rec = self.decoder2(feat_rec, feat_stage3)
        feat_rec = self.decoder1(feat_rec, feat_stage4)
        feat_rec = self.embedding_to_patch(feat_rec)
        feat_rec = rearrange(feat_rec, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=self.image_size//p, p1=p, p2=p)
        feat_rec = self.span_conv1x1(feat_rec)
        x = self.tail(feat_rec)
        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

class TransNoiseNET(TransNet):

    def __init__(self,n_feat=64, z_feat=8, leaky_neg=0.2):
        super(TransNoiseNET, self).__init__()
        leaky_neg = leaky_neg
        filter_size = 5
        z_channel = z_feat
        in_z = [nn.ConvTranspose2d(1, 2 * z_channel, 2, 2, 0, 0),  # 8 -> 16
                nn.LeakyReLU(leaky_neg),
                nn.ConvTranspose2d(2 * z_channel, 4 * z_channel, 2, 2, 0, 0),  # 16 -> 32
                nn.LeakyReLU(leaky_neg),
                nn.ConvTranspose2d(4 * z_channel, 8 * z_channel, 1, 1, 0, 0),  # 16 -> 32
                nn.LeakyReLU(leaky_neg)
                ]
        self.z_head = nn.Sequential(*in_z)
        self.merge = nn.Conv2d(n_feat, n_feat, 1, 1, 0)

    def forward(self, x, z=None):
        x = self.head(x)
        z = self.z_head(z)

        # feature extraction part
        feat_stage1 = self.feat_extrat_stage1(x)
        feat_stage2 = self.feat_extrat_stage2(x)
        feat_stage3 = self.feat_extrat_stage3(x)
        feat_stage_noise = self.feat_extrat_stage_noise(z)

        feat_stage1 = self.stage1_conv1x1(feat_stage1)
        feat_stage2 = self.stage2_conv1x1(feat_stage2)
        feat_stage3 = self.stage3_conv1x1(feat_stage3)
        feat_stage_noise = self.stage_noise_conv1x1(feat_stage_noise)

        # Transformer part
        p = self.patch_size
        feat_stage1 = rearrange(feat_stage1, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        feat_stage2 = rearrange(feat_stage2, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        feat_stage3 = rearrange(feat_stage3, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        feat_stage_noise = rearrange(feat_stage_noise, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)

        feat_stage1 = self.patch_to_embedding_low1(feat_stage1)
        feat_stage2 = self.patch_to_embedding_low2(feat_stage2)
        feat_stage3 = self.patch_to_embedding_low3(feat_stage3)
        feat_stage_noise = self.patch_to_embedding_low_noise(feat_stage_noise)

        # Encoder & Decoder
        feat_stage1 = self.encoder_stage1(feat_stage1)
        feat_stage2 = self.encoder_stage2(feat_stage2)
        feat_stage3 = self.encoder_stage3(feat_stage3)
        feat_stage_noise = self.encoder_stage4(feat_stage_noise)

        feat_rec = self.decoder3(feat_stage1, feat_stage_noise)
        feat_rec = self.decoder2(feat_rec, feat_stage2)
        feat_rec = self.decoder1(feat_rec, feat_stage3)
        feat_rec = self.embedding_to_patch(feat_rec)
        feat_rec = rearrange(feat_rec, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=self.image_size // p, p1=p, p2=p)
        feat_rec = self.span_conv1x1(feat_rec)
        x = self.tail(feat_rec)
        return x


if __name__ == "__main__":
    model = TransNoiseNET()
    model.eval()
    input = torch.rand(1, 1, 48, 48)
    Z = torch.randn(1, 1, 12, 12, dtype=torch.float32)
    model = model.cuda()
    input = input.cuda()
    Z = Z.cuda()
    sr = model(input, Z)
    print(sr.size())
