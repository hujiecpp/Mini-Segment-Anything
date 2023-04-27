# ImageEncoderYOSO for SAM-mini
# from https://github.com/hujiecpp/YOSO/blob/main/projects/YOSO/yoso/neck.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
import fvcore.nn.weight_init as weight_init
from detectron2.modeling import build_backbone
from detectron2.layers import DeformConv, ModulatedDeformConv

class DeformLayer(nn.Module):
    def __init__(self, in_planes, out_planes, deconv_kernel=4, deconv_stride=2, deconv_pad=1, deconv_out_pad=0, modulate_deform=True, num_groups=1, deform_num_groups=1, dilation=1):
        super(DeformLayer, self).__init__()
        self.deform_modulated = modulate_deform
        if modulate_deform:
            deform_conv_op = ModulatedDeformConv
            offset_channels = 27
        else:
            deform_conv_op = DeformConv
            offset_channels = 18
        
        self.dcn_offset = nn.Conv2d(in_planes, offset_channels * deform_num_groups, kernel_size=3, stride=1, padding=1*dilation, dilation=dilation)
        self.dcn = deform_conv_op(in_planes, out_planes, kernel_size=3, stride=1, padding=1*dilation, bias=False, groups=num_groups, dilation=dilation, deformable_groups=deform_num_groups)
        for layer in [self.dcn]:
            weight_init.c2_msra_fill(layer)

        nn.init.constant_(self.dcn_offset.weight, 0)
        nn.init.constant_(self.dcn_offset.bias, 0)
        
        self.dcn_bn = nn.SyncBatchNorm(out_planes)
        self.up_sample = nn.ConvTranspose2d(in_channels=out_planes, out_channels=out_planes, kernel_size=deconv_kernel, stride=deconv_stride, padding=deconv_pad, output_padding=deconv_out_pad, bias=False)
        self._deconv_init()
        self.up_bn = nn.SyncBatchNorm(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x
        if self.deform_modulated:
            offset_mask = self.dcn_offset(out)
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            out = self.dcn(out, offset, mask)
        else:
            offset = self.dcn_offset(out)
            out = self.dcn(out, offset)
        x = out
        
        x = self.dcn_bn(x)
        x = self.relu(x)
        x = self.up_sample(x)
        x = self.up_bn(x)
        x = self.relu(x)
        return x

    def _deconv_init(self):
        w = self.up_sample.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]


class LiteDeformConv(nn.Module):
    def __init__(self, cfg, backbone_shape):
        super(LiteDeformConv, self).__init__()
        in_features = cfg.MODEL.YOSO.IN_FEATURES
        in_channels = []
        out_channels = [cfg.MODEL.YOSO.AGG_DIM]
        for feat in in_features:
            tmp = backbone_shape[feat].channels
            in_channels.append(tmp)
            out_channels.append(tmp//2)
        
        self.lateral_conv0 = nn.Conv2d(in_channels=in_channels[-1], out_channels=out_channels[-1], kernel_size=1, stride=1, padding=0)
        self.deform_conv1 = DeformLayer(in_planes=out_channels[-1], out_planes=out_channels[-2])
        self.lateral_conv1 = nn.Conv2d(in_channels=in_channels[-2], out_channels=out_channels[-2], kernel_size=1, stride=1, padding=0)
        self.deform_conv2 = DeformLayer(in_planes=out_channels[-2], out_planes=out_channels[-3])
        self.lateral_conv2 = nn.Conv2d(in_channels=in_channels[-3], out_channels=out_channels[-3], kernel_size=1, stride=1, padding=0)
        self.deform_conv3 = DeformLayer(in_planes=out_channels[-3], out_planes=out_channels[-4])
        self.lateral_conv3 = nn.Conv2d(in_channels=in_channels[-4], out_channels=out_channels[-4], kernel_size=1, stride=1, padding=0)
        self.output_conv = nn.Conv2d(in_channels=out_channels[-5], out_channels=out_channels[-5], kernel_size=3, stride=1, padding=1)
        self.bias = nn.Parameter(torch.FloatTensor(1,out_channels[-5],1,1), requires_grad=True)
        self.bias.data.fill_(0.0)
        
        self.conv_a5 = nn.Conv2d(in_channels=out_channels[-1], out_channels=out_channels[-5], kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_a4 = nn.Conv2d(in_channels=out_channels[-2], out_channels=out_channels[-5], kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_a3 = nn.Conv2d(in_channels=out_channels[-3], out_channels=out_channels[-5], kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_a2 = nn.Conv2d(in_channels=out_channels[-4], out_channels=out_channels[-5], kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, features_list):
        p5 = self.lateral_conv0(features_list[-1])
        x5 = p5
        x = self.deform_conv1(x5)

        p4 = self.lateral_conv1(features_list[-2])
        x4 = p4 + x
        x = self.deform_conv2(x4)

        p3 = self.lateral_conv2(features_list[-3])
        x3 = p3 + x
        x = self.deform_conv3(x3)

        p2 = self.lateral_conv3(features_list[-4])
        x2 = p2 + x
        
        # CFA
        x5 = F.interpolate(self.conv_a5(x5), scale_factor=8, align_corners=False, mode='bilinear')
        x4 = F.interpolate(self.conv_a4(x4), scale_factor=4, align_corners=False, mode='bilinear')
        x3 = F.interpolate(self.conv_a3(x3), scale_factor=2, align_corners=False, mode='bilinear')
        x2 = self.conv_a2(x2)
        x = x5 + x4 + x3 + x2 + self.bias
        
        x = self.output_conv(x)

        return x


class YOSONeck(nn.Module):
    def __init__(self, cfg, backbone_shape):
        super().__init__()

        self.deconv = LiteDeformConv(cfg=cfg, backbone_shape=backbone_shape)
        self.loc_conv = nn.Conv2d(in_channels=128+2, out_channels=cfg.MODEL.YOSO.HIDDEN_DIM, kernel_size=1, stride=1)

        self.conv_1 = nn.Conv2d(in_channels=cfg.MODEL.YOSO.HIDDEN_DIM, out_channels=cfg.MODEL.YOSO.HIDDEN_DIM, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=cfg.MODEL.YOSO.HIDDEN_DIM, out_channels=cfg.MODEL.YOSO.HIDDEN_DIM, kernel_size=4, stride=2, padding=1)

    def generate_coord(self, input_feat):
        x_range = torch.linspace(-1, 1, input_feat.shape[-1], device=input_feat.device)
        y_range = torch.linspace(-1, 1, input_feat.shape[-2], device=input_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([input_feat.shape[0], 1, -1, -1])
        x = x.expand([input_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        return coord_feat

    def forward(self, features_list):
        features = self.deconv(features_list)
        coord_feat = self.generate_coord(features)
        features = torch.cat([features, coord_feat], 1)
        features = self.loc_conv(features)

        # features = F.interpolate(features, scale_factor=0.25, mode="bilinear", align_corners=False)
        features = self.relu(features)
        features = self.relu(self.conv_1(features))
        features = self.conv_2(features)

        return features

class ImageEncoderYOSO(nn.Module):
    def __init__(self, img_size: int = 1024):
        super().__init__()
        self.img_size = img_size

        cfg = get_cfg()
        cfg.MODEL.YOSO = CN()
        cfg.MODEL.YOSO.IN_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.YOSO.HIDDEN_DIM = 256
        cfg.MODEL.YOSO.AGG_DIM = 128
        cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
        cfg.MODEL.BACKBONE.FREEZE_AT = 0
        cfg.MODEL.RESNETS.DEPTH = 50
        cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
        
        self.in_features = cfg.MODEL.YOSO.IN_FEATURES
        self.backbone = build_backbone(cfg)
        self.yoso_neck = YOSONeck(cfg=cfg, backbone_shape=self.backbone.output_shape())
        # self.to('cuda')

    def forward(self, images):
        backbone_feats = self.backbone(images)
        # print(backbone_feats)
        features = list()
        for f in self.in_features:
            features.append(backbone_feats[f])

        neck_feats = self.yoso_neck(features)
        return neck_feats

# build SAM_mini
from segment_anything.modeling import MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

def build_sam_yoso_r50(checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderYOSO(),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict, strict=True)
    return sam