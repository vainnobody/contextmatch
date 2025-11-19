import model.backbone.resnet as resnet
from model.backbone.xception import xception

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3Plus, self).__init__()

        # if 'resnet' in cfg['backbone']:
        self.backbone = resnet.__dict__[cfg['backbone']](pretrained=True,
                                                             replace_stride_with_dilation=cfg['replace_stride_with_dilation'])
        # else:
        #     assert cfg['backbone'] == 'xception'
        #     self.backbone = xception(pretrained=True)

        low_channels = 256
        high_channels = 2048

        self.head = ASPPModule(high_channels, cfg['dilations'])

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))

        scale_in_ch = 2 * 256

        self.scale_attn = nn.Sequential(
            nn.Conv2d(scale_in_ch, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, bias=False),
            nn.Sigmoid())

        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)

    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)

        out = self.classifier(feature)

        return out, feature


    def _fwd(self, x, need_fp=False, feature_scale=None):
        h, w = x.shape[-2:]
        feats = self.backbone.base_forward(x)

        c1, c4 = feats[0], feats[-1]

        if need_fp:
            # if random.random() > 0.5:
            if feature_scale == 1.0:
                c1_fp, c4_fp = c1, c4
            else:
                c1_fp = F.interpolate(c1, scale_factor=feature_scale, mode="bilinear", align_corners=True)
                c4_fp = F.interpolate(c4, scale_factor=feature_scale, mode="bilinear", align_corners=True)
            outs, features = self._decode(torch.cat((c1_fp, nn.Dropout2d(0.5)(c1_fp))), torch.cat((c4_fp, nn.Dropout2d(0.5)(c4_fp))))
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)
            feature, _ = features.chunk(2)
            return out, feature, out_fp

        out, feature = self._decode(c1, c4)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        return out, feature

    def nscale_forward(self, inputs, scales, mode):
        """
        Hierarchical attention, primarily used for getting best inference
        results.

        We use attention at multiple scales, giving priority to the lower
        resolutions. For example, if we have 4 scales {0.5, 1.0, 1.5, 2.0},
        then evaluation is done as follows:

              p_joint = attn_1.5 * p_1.5 + (1 - attn_1.5) * down(p_2.0)
              p_joint = attn_1.0 * p_1.0 + (1 - attn_1.0) * down(p_joint)
              p_joint = up(attn_0.5 * p_0.5) * (1 - up(attn_0.5)) * p_joint

        The target scale is always 1.0, and 1.0 is expected to be part of the
        list of scales. When predictions are done at greater than 1.0 scale,
        the predictions are downsampled before combining with the next lower
        scale.

        Inputs:
          scales - a list of scales to evaluate
          inputs - dict containing 'images', the input, and 'gts', the ground
                   truth mask

        Output:
          If training, return loss, else return prediction + attention
        """
        x_1x = inputs
        B,C,H,W = x_1x.shape
        pre_add = torch.zeros(B, 21, H, W).cuda()

        assert 1.0 in scales, 'expected 1.0 to be the target scale'
        # Lower resolution provides attention for higher rez predictions,
        # so we evaluate in order: high to low
        scales = sorted(scales, reverse=True)
        pred = None
        last_feats = None

        if mode== 'atten_fusion' :
            for idx, s in enumerate(scales):
                x = ResizeX(x_1x, s)
                # p, feats = self._fwd(x)
                p, feats, _ = self._fwd(x, x_1x)

                # Generate attention prediction
                if idx > 0:
                    assert last_feats is not None
                    # downscale feats
                    last_feats = scale_as(last_feats, feats)
                    cat_feats = torch.cat([feats, last_feats], 1)
                    attn = self.scale_attn(cat_feats)
                    attn = scale_as(attn, p)

                if pred is None:
                    # This is the top scale prediction
                    pred = p
                elif s >= 1.0:
                    # downscale previous
                    pred = scale_as(pred, p)
                    pred = attn * p + (1 - attn) * pred
                else:
                    # upscale current
                    p = attn * p
                    p = scale_as(p, pred)
                    attn = scale_as(attn, pred)
                    pred = p + (1 - attn) * pred

                last_feats = feats
            return pred
        else:
            for idx, s in enumerate(scales):
                x = ResizeX(x_1x, s)
                # p, feats = self._fwd(x)
                p, feats, _ = self._fwd(x, x_1x)
                if s != 1.0:
                    p = scale_as(p, x_1x)
                pre_add += p
            return pre_add

        # if self.training:
        #     assert 'gts' in inputs
        #     gts = inputs['gts']
        #     loss = self.criterion(pred, gts)
        #     return loss
        # else:
            # FIXME: should add multi-scale values for pred and attn
            # return {'pred': pred,
            #         'attn_10x': attn}
            # return pred

    def two_scale_forward(self, inputs, scale_factor, feature_scale):
        x_1x = inputs
        B, C, H, W = x_1x.shape
        if scale_factor == None:
            out, _ = self._fwd(x_1x)
            return out

        if scale_factor > 1.0:
            x_lo = x_1x
            x_hi = ResizeX(x_1x, scale_factor)
            # x_hi[(B//2):], _ = self.random_masking(x_hi[(B//2):] , 32, 0.5)

            p_lo_ori, feats_lo, out_fp = self._fwd(x_lo, need_fp=True, feature_scale=feature_scale)
            p_hi, feats_hi = self._fwd(x_hi)
            p_hi = scale_as(p_hi, x_1x)
            feats_hi = scale_as(feats_hi, feats_lo)
            cat_feats = torch.cat([feats_lo, feats_hi], 1)
            logit_attn = self.scale_attn(cat_feats)
            logit_attn = scale_as(logit_attn, p_lo_ori)
            p_lo = logit_attn * p_lo_ori
            p_lo_up = scale_as(p_lo, p_hi)
            logit_attn = scale_as(logit_attn, p_hi)
            joint_pred = p_lo_up + (1 - logit_attn) * p_hi
            joint_pred = scale_as(joint_pred, p_lo)


            # return {'pred_joint': joint_pred,
            #         'pred': p_hi}
            return {
                    'pred_joint': joint_pred,
                    'pred_ori': p_lo_ori,
                    'pred_fp': out_fp,
                    'pred_size': p_hi}


        else:
            x_lo = ResizeX(x_1x, scale_factor)
            x_hi = x_1x
            p_lo, feats_lo = self._fwd(x_lo)
            p_hi, feats_hi, out_fp = self._fwd(x_hi, need_fp=True, feature_scale=feature_scale)
            p_lo_ori = scale_as(p_lo, p_hi)
            feats_hi = scale_as(feats_hi, feats_lo)
            cat_feats = torch.cat([feats_lo, feats_hi], 1)
            logit_attn = self.scale_attn(cat_feats)
            logit_attn = scale_as(logit_attn, p_lo)
            p_lo_att = logit_attn * p_lo
            p_lo_att = scale_as(p_lo_att, p_hi)
            logit_attn = scale_as(logit_attn, p_hi)
            joint_pred = p_lo_att + (1 - logit_attn) * p_hi
  
            return {
                    'pred_joint': joint_pred,
                    'pred_ori': p_hi,
                    'pred_fp': out_fp,
                    'pred_size': p_lo_ori}


    def forward(self, inputs, scale_factor=None, feature_scale=1.0, scales=None, eval_mode='atten_fusion'):
        # cfg.MODEL.N_SCALES 多尺度列表
        # if scales and not self.training:
        if scales:
            return self.nscale_forward(inputs, scales, eval_mode)

        return self.two_scale_forward(inputs, scale_factor, feature_scale)






def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block

def ResizeX(x, scale_factor):
    '''
    scale x by some factor
    '''
    # if cfg.OPTIONS.TORCH_VERSION >= 1.5:
    x_scaled = torch.nn.functional.interpolate(
        x, scale_factor=scale_factor, mode='bilinear',
        align_corners=True)
        # align_corners=False, recompute_scale_factor=True)
    # else:
    #     x_scaled = torch.nn.functional.interpolate(
    #         x, scale_factor=scale_factor, mode='bilinear',
    #         align_corners=False)
    return x_scaled

def scale_as(x, y):
    '''
    scale x to the same size as y
    '''
    y_size = y.size(2), y.size(3)

    # if cfg.OPTIONS.TORCH_VERSION >= 1.5:
    x_scaled = torch.nn.functional.interpolate(
        x, size=y_size, mode='bilinear',
        align_corners=True)
        # align_corners=False)
    # else:
    #     x_scaled = torch.nn.functional.interpolate(
    #         x, size=y_size, mode='bilinear',
    #         align_corners=align_corners)
    return x_scaled

def make_attn_head(in_ch, out_ch):
    bot_ch = 256

    od = OrderedDict([('conv0', nn.Conv2d(in_ch, bot_ch, kernel_size=3,
                                          padding=1, bias=False)),
                      ('bn0', nn.BatchNorm2d(bot_ch)),
                      ('re0', nn.ReLU(inplace=True))])

    od['conv1'] = nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1,
                            bias=False)
    od['bn1'] = nn.BatchNorm2d(bot_ch)
    od['re1'] = nn.ReLU(inplace=True)
    od['conv2'] = nn.Conv2d(bot_ch, out_ch, kernel_size=1, bias=False)
    od['sig'] = nn.Sigmoid()
    attn_head = nn.Sequential(od)
    # init_attn(attn_head)
    return attn_head

class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)
