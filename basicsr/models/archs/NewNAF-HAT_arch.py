# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
NAFSSR: Stereo Image Super-Resolution Using NAFNet

@InProceedings{Chu2022NAFSSR,
  author    = {Xiaojie Chu and Liangyu Chen and Wenqing Yu},
  title     = {NAFSSR: Stereo Image Super-Resolution Using NAFNet},
  booktitle = {CVPRW},
  year      = {2022},
}
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.append("/home/thinkstation03/zjt/NAFNet-ALL/NAFNet-main/")

from basicsr.models.archs.NAFNet_arch import LayerNorm2d, NAFBlock
from basicsr.models.archs.arch_util import MySequential
from basicsr.models.archs.local_arch import Local_Base
from basicsr.utils.transformer import MLABlock
from basicsr.utils.tools import reverse_patches

from basicsr.models.archs.NAFNet_arch import FFCBlock, SimpleFFBBlock


class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM)
    '''
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r):
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3) # B, H, c, W (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  #B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l) #B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r

class DropPath(nn.Module):
    def __init__(self, drop_rate, module):
        super().__init__()
        self.drop_rate = drop_rate
        self.module = module

    def forward(self, *feats):
        if self.training and np.random.rand() < self.drop_rate:
            return feats

        new_feats = self.module(*feats)
        factor = 1. / (1 - self.drop_rate) if self.training else 1.

        if self.training and factor != 1.:
            new_feats = tuple([x+factor*(new_x-x) for x, new_x in zip(feats, new_feats)])
        return new_feats

class CrossStageAM(nn.Module):
    '''
    Cross Stage Attention Module (CrossStageAM)
    '''
    def __init__(self, c=1):
        super().__init__()
        self.Conv_F = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.Conv_img = nn.Conv2d(3, c, kernel_size=1, stride=1, padding=0)

    def forward(self, SRimg, feat):

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention =  torch.softmax(self.Conv_img(SRimg), dim=-1)
        x = feat*attention
        out = x + feat

        return out


    # def __init__(self, n_feat, kernel_size, bias):  # origin,paper giving
    #     super(SAM, self).__init__()
    #     self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
    #     self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
    #     self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    # def forward(self, x, x_img):
    #     # x1 = self.conv1(x)
    #     # img = self.conv2(x) + x_img
    #     # x2 = torch.sigmoid(self.conv3(img))
    #     # x1 = x1*x2
    #     # x1 = x1+x
    #     # return x1, img


class NAFBlockSR(nn.Module):
    '''
    ori
    NAFBlock for Super-Resolution
    '''
    def __init__(self, c, fusion=False, drop_out_rate=0.):
        super().__init__()
        self.blk = NAFBlock(c, drop_out_rate=drop_out_rate)
        self.fusion = SCAM(c) if fusion else None

    def forward(self, *feats):
        feats = tuple([self.blk(x) for x in feats])
        if self.fusion:
            feats = self.fusion(*feats)
        return feats

class FFCBlockSR(nn.Module):
    '''
    # 2023/10/31
    mint: 傅里叶卷积构成的块
    '''
    def __init__(self, c, fusion=False, drop_out_rate=0., enable_lfu=False):
        super().__init__()
        self.ffcblk = FFCBlock(c, drop_out_rate=drop_out_rate, enable_lfu=enable_lfu)
        self.fusion = SCAM(c) if fusion else None

    def forward(self, *feats):
        feats = tuple([self.ffcblk(x) for x in feats])
        if self.fusion:
            feats = self.fusion(*feats)
        return feats

class SimpleFFBBlockSR(nn.Module): 
    '''
    # 2023/11/01
    mint: 在simplegate后面接上简单的双分支，用SwinFFC的FFB实现
    '''
    def __init__(self, c, fusion=False, drop_out_rate=0., enable_lfu=False):
        super().__init__()
        self.blk = SimpleFFBBlock(c, drop_out_rate=drop_out_rate, enable_lfu=enable_lfu)
        self.fusion = SCAM(c) if fusion else None

    def forward(self, *feats):
        feats = tuple([self.blk(x) for x in feats])
        if self.fusion:
            feats = self.fusion(*feats)
        return feats


'''
# 上面是基础的（一个Block+一个PAM）　Ｂｌｏｃｋ
＃下面是组网络
'''

class NAF_FFCNetSR(nn.Module):
    '''
    NAFNet add FFCBlockSR for Super-Resolution
    NAFNet中把最后两层换成两个ffc构成的FFB
    '''
    def __init__(self, up_scale=4, width=48, num_blks=16, img_channel=3, drop_path_rate=0., drop_out_rate=0., fusion_from=-1, fusion_to=-1, dual=False):
        super().__init__()
        self.dual = dual    # dual input for stereo SR (left view, right view)
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        
        ffc_num = 2
        self.FFCSiBlock = [DropPath(
            drop_path_rate,
            FFCBlockSR(
                width,
                fusion = 1,
                drop_out_rate=drop_out_rate
            )) for i in range(ffc_num) ]

        bodylist = [DropPath(
                drop_path_rate, 
                NAFBlockSR(
                    width, 
                    fusion=(fusion_from <= i and i <= fusion_to), 
                    drop_out_rate=drop_out_rate
                )) for i in range(num_blks-ffc_num)]
        bodylist = bodylist + self.FFCSiBlock
        
        self.body = MySequential(
            *bodylist
        )

        # mint ET
        # self.attention = MLABlock(n_feat=48, dim=432)

        # mint ET

        self.up = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale**2, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.PixelShuffle(up_scale)
        )
        self.up_scale = up_scale

    def forward(self, inp):
        inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')
        if self.dual:
            inp = inp.chunk(2, dim=1)
        else:
            inp = (inp, )
        feats = [self.intro(x) for x in inp]
        feats = self.body(*feats)

        # mint add ET module
        # _, _, h, w = inp[0].size()
        # feats = [self.attention(feat) for feat in feats]
        # feats = [feat.permute(0, 2, 1) for feat in feats]
        # feats = [reverse_patches(feat, ( h, w), ( 3, 3), 1, 1) for feat in feats]
        # mint add ET module

        out = torch.cat([self.up(x) for x in feats], dim=1)
        out = out + inp_hr
        return out

class NAF_SimpleFFCNet_withHAT_SR(nn.Module):
    '''
    NAFNet add FFCBlockSR for Super-Resolution
    NAFNet中的基础模块，simplge gate换成简单的ffb，只有一层ffc
    '''
    def __init__(self, up_scale=4, width=48, num_blks=16, img_channel=3, drop_path_rate=0., drop_out_rate=0., fusion_from=-1, fusion_to=-1, dual=False):
        super().__init__()
        self.dual = dual    # dual input for stereo SR (left view, right view)
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        

        bodylist = [DropPath(
                drop_path_rate, 
                SimpleFFBBlockSR(
                    width, 
                    fusion=(fusion_from <= i and i <= fusion_to), 
                    drop_out_rate=drop_out_rate
                )) for i in range(num_blks)]
        
        self.body = MySequential(
            *bodylist
        )

        # mint ET
        # self.attention = MLABlock(n_feat=48, dim=432)
        # mint ET

        # self.up = nn.Sequential(
        #     nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale**2, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
        #     nn.PixelShuffle(up_scale)
        # )
        self.fakeup = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.up_scale = up_scale

    def forward(self, inp):
        # inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')
        inp_SR = inp
        if self.dual:
            inp = inp.chunk(2, dim=1)
        else:
            inp = (inp, )
        feats = [self.intro(x) for x in inp]
        feats = self.body(*feats)

        # mint add ET module
        # _, _, h, w = inp[0].size()
        # feats = [self.attention(feat) for feat in feats]
        # feats = [feat.permute(0, 2, 1) for feat in feats]
        # feats = [reverse_patches(feat, ( h, w), ( 3, 3), 1, 1) for feat in feats]
        # mint add ET module

        # out = torch.cat([self.up(x) for x in feats], dim=1)
        # out = out + inp_hr

        out = torch.cat([self.fakeup(x) for x in feats], dim=1)
        # print(out.shape, inp_SR.shape)
        out = out + inp_SR
        return out


class NAF_SimpleFFCNet_withHAT_downIN_SR(nn.Module):
    '''
    NAFNet add FFCBlockSR for Super-Resolution
    NAFNet中的基础模块，simplge gate换成简单的ffb，只有一层ffc
    '''
    def __init__(self, up_scale=4, width=48, num_blks=16, img_channel=3, drop_path_rate=0., drop_out_rate=0., fusion_from=-1, fusion_to=-1, dual=False):
        super().__init__()
        self.dual = dual    # dual input for stereo SR (left view, right view)

        self.pixelunshuffle = nn.PixelUnshuffle(downscale_factor=2)
        self.newintro = nn.Conv2d(in_channels=img_channel*2*2, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        

        bodylist = [DropPath(
                drop_path_rate, 
                SimpleFFBBlockSR(
                    width, 
                    fusion=(fusion_from <= i and i <= fusion_to), 
                    drop_out_rate=drop_out_rate
                )) for i in range(num_blks)]
        
        self.body = MySequential(
            *bodylist
        )

        # mint ET
        # self.attention = MLABlock(n_feat=48, dim=432)
        # mint ET

        # self.up = nn.Sequential(
        #     nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale**2, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
        #     nn.PixelShuffle(up_scale)
        # )

        self.beforepixelshuffle = nn.Conv2d(in_channels=width, out_channels=img_channel*2*2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)
        # self.fakeup = nn.Conv2d(in_channels=width*up_scale*up_scale, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.up_scale = up_scale

    def forward(self, inp):
        # 根据是否有特征图进行处理

        # inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')
        inp_SR = inp
        if self.dual:
            inp = inp.chunk(2, dim=1)
        else:
            inp = (inp, )
        # feats = [self.intro(x) for x in inp]

        _, c, h, w = inp[0].shape

        inp = [self.pixelunshuffle(x) for x in inp]
        feats = [self.newintro(x) for x in inp]

        feats = self.body(*feats)

        # mint add ET module
        # _, _, h, w = inp[0].size()
        # feats = [self.attention(feat) for feat in feats]
        # feats = [feat.permute(0, 2, 1) for feat in feats]
        # feats = [reverse_patches(feat, ( h, w), ( 3, 3), 1, 1) for feat in feats]
        # mint add ET module

        # out = torch.cat([self.up(x) for x in feats], dim=1)
        # out = out + inp_hr

        feats = [self.beforepixelshuffle(x) for x in feats]
        feats = [self.pixelshuffle(x) for x in feats]
        out = torch.cat(feats, dim=1)
        # out = torch.cat([self.fakeup(x) for x in feats], dim=1)

        # print(out.shape, inp_SR.shape)
        out = out + inp_SR
        return out


class NAF_SimpleFFCNet_withHAT_downIN_withCSAM_SR(nn.Module):
    '''
    NAFNet add FFCBlockSR for Super-Resolution
    NAFNet中的基础模块，simplge gate换成简单的ffb，只有一层ffc
    '''
    def __init__(self, up_scale=4, width=48, num_blks=16, img_channel=3, drop_path_rate=0., drop_out_rate=0., fusion_from=-1, fusion_to=-1, dual=False):
        super().__init__()
        self.dual = dual    # dual input for stereo SR (left view, right view)

        self.csam_channels = 3

        self.pixelunshuffle = nn.PixelUnshuffle(downscale_factor=2)
        self.newintro = nn.Conv2d(in_channels=img_channel*2*2, out_channels=width-self.csam_channels*4, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        
        self.csam = CrossStageAM(c=self.csam_channels)

        bodylist = [DropPath(
                drop_path_rate, 
                SimpleFFBBlockSR(
                    width, 
                    fusion=(fusion_from <= i and i <= fusion_to), 
                    drop_out_rate=drop_out_rate
                )) for i in range(num_blks)]
        
        self.body = MySequential(
            *bodylist
        )

        # mint ET
        # self.attention = MLABlock(n_feat=48, dim=432)
        # mint ET

        # self.up = nn.Sequential(
        #     nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale**2, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
        #     nn.PixelShuffle(up_scale)
        # )

        self.beforepixelshuffle = nn.Conv2d(in_channels=width, out_channels=img_channel*2*2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)
        # self.fakeup = nn.Conv2d(in_channels=width*up_scale*up_scale, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.up_scale = up_scale

    def forward(self, inp):
        # 根据是否有特征图进行处理

        # inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')
        # inp_SR = inp

        if self.dual:
            inp = inp.chunk(2, dim=1)
        else:
            inp = (inp, )
        # feats = [self.intro(x) for x in inp]

        _, c, h, w = inp[0].shape
        # print("c:", c)
        assert c==3+self.csam_channels, 'error C, in  inp.shape'
        out_L = inp[0]
        out_R = inp[1]
        img_L = out_L[:, :3, :, :]
        feat_L = out_L[:, 3:, :, :]
        img_R = out_R[:, :3, :, :]
        feat_R = out_R[:, 3:, :, :]

            
        csam_feat_L = self.csam(img_L, feat_L)
        csam_feat_R = self.csam(img_R, feat_R)
        out_feat = [csam_feat_L, csam_feat_R]

        inp = [img_L, img_R]
        inp_SR = torch.cat((img_L, img_R), dim=1)
        inp = [self.pixelunshuffle(x) for x in inp]
        feats = [self.newintro(x) for x in inp]

        out_feat = [self.pixelunshuffle(x) for x in out_feat]

        assert c==3+self.csam_channels,' error c, in shape'
        feats_L = torch.cat((feats[0], out_feat[0]), dim=1)
        feats_R = torch.cat((feats[1], out_feat[1]), dim=1)
        feats = [feats_L, feats_R]
            


        feats = self.body(*feats)

        # mint add ET module
        # _, _, h, w = inp[0].size()
        # feats = [self.attention(feat) for feat in feats]
        # feats = [feat.permute(0, 2, 1) for feat in feats]
        # feats = [reverse_patches(feat, ( h, w), ( 3, 3), 1, 1) for feat in feats]
        # mint add ET module

        # out = torch.cat([self.up(x) for x in feats], dim=1)
        # out = out + inp_hr

        feats = [self.beforepixelshuffle(x) for x in feats]
        feats = [self.pixelshuffle(x) for x in feats]
        out = torch.cat(feats, dim=1)
        # out = torch.cat([self.fakeup(x) for x in feats], dim=1)

        # print(out.shape, inp_SR.shape)
        out = out + inp_SR
        return out


class NAF_SimpleFFCNet_withHAT_downIN_abaCasmUsingConv_SR(nn.Module):
    '''
    NAFNet add FFCBlockSR for Super-Resolution
    NAFNet中的基础模块，simplge gate换成简单的ffb，只有一层ffc
    '''
    def __init__(self, up_scale=4, width=48, num_blks=16, img_channel=3, drop_path_rate=0., drop_out_rate=0., fusion_from=-1, fusion_to=-1, dual=False):
        super().__init__()
        self.dual = dual    # dual input for stereo SR (left view, right view)

        self.csam_channels = 3

        self.pixelunshuffle = nn.PixelUnshuffle(downscale_factor=2)
        self.newintro = nn.Conv2d(in_channels=img_channel*2*2, out_channels=width-self.csam_channels*4, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        
        self.csam = CrossStageAM(c=self.csam_channels)
        self.aba_conv = nn.Conv2d(in_channels=self.csam_channels, out_channels=self.csam_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        bodylist = [DropPath(
                drop_path_rate, 
                SimpleFFBBlockSR(
                    width, 
                    fusion=(fusion_from <= i and i <= fusion_to), 
                    drop_out_rate=drop_out_rate
                )) for i in range(num_blks)]
        
        self.body = MySequential(
            *bodylist
        )

        # mint ET
        # self.attention = MLABlock(n_feat=48, dim=432)
        # mint ET

        # self.up = nn.Sequential(
        #     nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale**2, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
        #     nn.PixelShuffle(up_scale)
        # )

        self.beforepixelshuffle = nn.Conv2d(in_channels=width, out_channels=img_channel*2*2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)
        # self.fakeup = nn.Conv2d(in_channels=width*up_scale*up_scale, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.up_scale = up_scale

    def forward(self, inp):
        # 根据是否有特征图进行处理

        # inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')
        # inp_SR = inp

        if self.dual:
            inp = inp.chunk(2, dim=1)
        else:
            inp = (inp, )
        # feats = [self.intro(x) for x in inp]

        _, c, h, w = inp[0].shape
        # print("c:", c)
        assert c==3+self.csam_channels, 'error C, in  inp.shape'
        out_L = inp[0]
        out_R = inp[1]
        img_L = out_L[:, :3, :, :]
        feat_L = out_L[:, 3:, :, :]
        img_R = out_R[:, :3, :, :]
        feat_R = out_R[:, 3:, :, :]

        ablation_CASM = True
        if ablation_CASM==True:            
            feat_L = self.aba_conv(feat_L)
            feat_R = self.aba_conv(feat_R)
            out_feat = [feat_L, feat_R]
        else:
            print('not ablation CASM, please checking')
            csam_feat_L = self.csam(img_L, feat_L)
            csam_feat_R = self.csam(img_R, feat_R)
            out_feat = [csam_feat_L, csam_feat_R]

        inp = [img_L, img_R]
        inp_SR = torch.cat((img_L, img_R), dim=1)
        inp = [self.pixelunshuffle(x) for x in inp]
        feats = [self.newintro(x) for x in inp]

        out_feat = [self.pixelunshuffle(x) for x in out_feat]

        assert c==3+self.csam_channels,' error c, in shape'
        feats_L = torch.cat((feats[0], out_feat[0]), dim=1)
        feats_R = torch.cat((feats[1], out_feat[1]), dim=1)
        feats = [feats_L, feats_R]
            


        feats = self.body(*feats)

        # mint add ET module
        # _, _, h, w = inp[0].size()
        # feats = [self.attention(feat) for feat in feats]
        # feats = [feat.permute(0, 2, 1) for feat in feats]
        # feats = [reverse_patches(feat, ( h, w), ( 3, 3), 1, 1) for feat in feats]
        # mint add ET module

        # out = torch.cat([self.up(x) for x in feats], dim=1)
        # out = out + inp_hr

        feats = [self.beforepixelshuffle(x) for x in feats]
        feats = [self.pixelshuffle(x) for x in feats]
        out = torch.cat(feats, dim=1)
        # out = torch.cat([self.fakeup(x) for x in feats], dim=1)

        # print(out.shape, inp_SR.shape)
        out = out + inp_SR
        return out

'''
# 下面是Local_base 有关的关系，将构建好的网络加上local base
'''

class NewSSR(Local_Base, NAF_FFCNetSR):  # 把最后的几层换掉
    def __init__(self, *args, train_size=(1, 6, 30, 90), fast_imp=False, fusion_from=-1, fusion_to=1000, **kwargs):
        Local_Base.__init__(self)
        NAF_FFCNetSR.__init__(self, *args, img_channel=3, fusion_from=fusion_from, fusion_to=fusion_to, dual=True, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


class NewsimpleFFCwithHATSSR(Local_Base, NAF_SimpleFFCNet_withHAT_SR):  # NAFNet中的基础模块，simplge gate换成简单的ffb，只有一层ffc
    '''
    NAFNet add FFCBlockSR for Super-Resolution
    NAFNet中的基础模块，simplge gate换成简单的ffb，只有一层ffc
    '''
    def __init__(self, *args, train_size=(1, 6, 30, 90), fast_imp=False, fusion_from=-1, fusion_to=1000, **kwargs):
        Local_Base.__init__(self)
        NAF_SimpleFFCNet_withHAT_SR.__init__(self, *args, img_channel=3, fusion_from=fusion_from, fusion_to=fusion_to, dual=True, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


class NewsimpleFFCwithHAT_down_SSR(Local_Base, NAF_SimpleFFCNet_withHAT_downIN_SR):
    '''
    NAFNet add FFCBlockSR for Super-Resolution
    NAFNet中的基础模块，simplge gate换成简单的ffb，只有一层ffc
    输出到body前，用pixel shuffle减小图片的尺寸，最后up回原来的尺寸
    '''
    def __init__(self, *args, train_size=(1, 6, 30, 90), fast_imp=False, fusion_from=-1, fusion_to=1000, **kwargs):
        Local_Base.__init__(self)
        NAF_SimpleFFCNet_withHAT_downIN_SR.__init__(self, *args, img_channel=3, fusion_from=fusion_from, fusion_to=fusion_to, dual=True, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

class NewsimpleFFCwithHAT_down_withCSAM_SSR(Local_Base, NAF_SimpleFFCNet_withHAT_downIN_withCSAM_SR):
    '''
    NAFNet add FFCBlockSR for Super-Resolution
    NAFNet中的基础模块，simplge gate换成简单的ffb，只有一层ffc
    输出到body前，用pixel shuffle减小图片的尺寸，最后up回原来的尺寸
    '''
    def __init__(self, *args, train_size=(1, 12, 30, 90), fast_imp=False, fusion_from=-1, fusion_to=1000, **kwargs):
        Local_Base.__init__(self)
        NAF_SimpleFFCNet_withHAT_downIN_withCSAM_SR.__init__(self, *args, img_channel=3, fusion_from=fusion_from, fusion_to=fusion_to, dual=True, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

class NewsimpleFFCwithHAT_down_abaCsamUsingConv_SSR(Local_Base, NAF_SimpleFFCNet_withHAT_downIN_abaCasmUsingConv_SR):
    '''
    NAFNet add FFCBlockSR for Super-Resolution
    NAFNet中的基础模块，simplge gate换成简单的ffb，只有一层ffc
    输出到body前，用pixel shuffle减小图片的尺寸，最后up回原来的尺寸
    '''
    def __init__(self, *args, train_size=(1, 12, 30, 90), fast_imp=False, fusion_from=-1, fusion_to=1000, **kwargs):
        Local_Base.__init__(self)
        NAF_SimpleFFCNet_withHAT_downIN_abaCasmUsingConv_SR.__init__(self, *args, img_channel=3, fusion_from=fusion_from, fusion_to=fusion_to, dual=True, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

if __name__ == '__main__':
    num_blks = 28 # 28
    width = 64 #  64
    droppath= 0.1
    train_size = (1, 12, 60, 180) 

    # net = NewSSR(up_scale=2, train_size=train_size, fast_imp=True, width=width, num_blks=num_blks, drop_path_rate=droppath)
    # net_choose = NewSSR, NewsimpleFFCwithHATSSR, NewsimpleFFCwithHAT_down_SSR
    net = NewsimpleFFCwithHAT_down_abaCsamUsingConv_SSR(up_scale=2, train_size=train_size, fast_imp=True, width=width, num_blks=num_blks, drop_path_rate=droppath)

    inp_shape = (12, 60, 180)

    from ptflops import get_model_complexity_info
    FLOPS = 0
    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    # params = float(params[:-4])

    macs = float(macs[:-4]) + FLOPS / 10 ** 9

    print('mac', macs, '   params', params)


    import time

    img =  torch.randn(1 , 12, 60, 180)

    # 记录开始时间
    start_time = time.time()

    out = net(img)
    print('output.size:', out.size()[-2:])

    # 记录结束时间
    end_time = time.time()

    # 计算执行时间
    execution_time = end_time - start_time
    # 打印执行时间
    print(f"代码执行时间：{execution_time} 秒")

    # from basicsr.models.archs.arch_util import measure_inference_speed
    # net = net.cuda()
    # data = torch.randn((1, 6, 128, 128)).cuda()
    # measure_inference_speed(net, (data,))




