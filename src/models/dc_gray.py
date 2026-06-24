import time

import torch
from torch import nn
import numpy as np
from .common_model import CompressionModel
from .video_net import  ResBlock, UNet, bilinearupsacling, bilineardownsacling, get_hyper_enc_dec_models, flow_warp, ResBlock_PConv, Gray_Spynet
from ..layers.layers import subpel_conv3x3, subpel_conv1x1, DepthConvBlock, \
    ResidualBlockWithStride, ResidualBlockUpsample
from ..utils.stream_helper import get_downsampled_shape, encode_p, decode_p, filesize, \
    get_state_dict
from functools import partial
g_ch_1x = 48
g_ch_2x = 64
g_ch_4x = 96
g_ch_8x = 96
g_ch_16x = 128


class OffsetDiversity(nn.Module):
    def __init__(self, in_channel=g_ch_1x, aux_feature_num=g_ch_1x+1+2,
                 offset_num=2, group_num=16, max_residue_magnitude=40, inplace=False):
        super().__init__()
        self.in_channel = in_channel
        self.offset_num = offset_num
        self.group_num = group_num
        self.max_residue_magnitude = max_residue_magnitude
        self.conv_offset = nn.Sequential(
            nn.Conv2d(aux_feature_num, g_ch_2x, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=inplace),
            nn.Conv2d(g_ch_2x, g_ch_2x, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=inplace),
            nn.Conv2d(g_ch_2x, 3 * group_num * offset_num, 3, 1, 1),
        )
        self.fusion = nn.Conv2d(in_channel * offset_num, in_channel, 1, 1, groups=group_num)

    def forward(self, x, aux_feature, flow):
        B, C, H, W = x.shape
        out = self.conv_offset(aux_feature)
        out = bilinearupsacling(out)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        mask = torch.sigmoid(mask)
        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.repeat(1, self.group_num * self.offset_num, 1, 1)

        # warp
        offset = offset.view(B * self.group_num * self.offset_num, 2, H, W)
        mask = mask.view(B * self.group_num * self.offset_num, 1, H, W)
        x = x.view(B * self.group_num, C // self.group_num, H, W)
        x = x.repeat(self.offset_num, 1, 1, 1)
        x = flow_warp(x, offset)
        x = x * mask
        x = x.view(B, C * self.offset_num, H, W)
        x = self.fusion(x)

        return x


class FeatureExtractor(nn.Module):
    def __init__(self, inplace=False, resblock_type='normal', n_div=4):
        super().__init__()
        if resblock_type == 'normal':
            _resblock = partial(ResBlock, inplace=inplace)
        elif resblock_type == 'pconv':
            _resblock = partial(ResBlock_PConv, inplace=inplace, n_div=n_div)
            
        self.conv1 = nn.Conv2d(g_ch_1x, g_ch_1x, 3, stride=1, padding=1)
        self.res_block1 = _resblock(g_ch_1x)
        self.conv2 = nn.Conv2d(g_ch_1x, g_ch_2x, 3, stride=2, padding=1)
        self.res_block2 = _resblock(g_ch_2x)
        self.conv3 = nn.Conv2d(g_ch_2x, g_ch_4x, 3, stride=2, padding=1)
        self.res_block3 = _resblock(g_ch_4x)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)

        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.res_block3(layer3)

        return layer1, layer2, layer3


class MultiScaleContextFusion(nn.Module):
    def __init__(self, inplace=False, resblock_type='normal', n_div=4):
        super().__init__()
        if resblock_type == 'normal':
            _resblock = partial(ResBlock, inplace=inplace)
        elif resblock_type == 'pconv':
            _resblock = partial(ResBlock_PConv, inplace=inplace, n_div=n_div)
        
        self.conv3_up = subpel_conv3x3(g_ch_4x, g_ch_2x, 2)
        self.res_block3_up = _resblock(g_ch_2x)
        self.conv3_out = nn.Conv2d(g_ch_4x, g_ch_4x, 3, padding=1)
        self.res_block3_out = _resblock(g_ch_4x)
        self.conv2_up = subpel_conv3x3(g_ch_2x * 2, g_ch_1x, 2)
        self.res_block2_up = _resblock(g_ch_1x)
        self.conv2_out = nn.Conv2d(g_ch_2x * 2, g_ch_2x, 3, padding=1)
        self.res_block2_out = _resblock(g_ch_2x)
        self.conv1_out = nn.Conv2d(g_ch_1x * 2, g_ch_1x, 3, padding=1)
        self.res_block1_out = _resblock(g_ch_1x)

    def forward(self, context1, context2, context3):
        context3_up = self.conv3_up(context3)
        context3_up = self.res_block3_up(context3_up)
        context3_out = self.conv3_out(context3)
        context3_out = self.res_block3_out(context3_out)
        context2_up = self.conv2_up(torch.cat((context3_up, context2), dim=1))
        context2_up = self.res_block2_up(context2_up)
        context2_out = self.conv2_out(torch.cat((context3_up, context2), dim=1))
        context2_out = self.res_block2_out(context2_out)
        context1_out = self.conv1_out(torch.cat((context2_up, context1), dim=1))
        context1_out = self.res_block1_out(context1_out)
        context1 = context1 + context1_out
        context2 = context2 + context2_out
        context3 = context3 + context3_out

        return context1, context2, context3


class MvEnc(nn.Module):
    def __init__(self, input_channel, channel, inplace=False):
        super().__init__()
        self.enc_1 = nn.Sequential(
            ResidualBlockWithStride(input_channel, channel, stride=2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
        )
        self.enc_2 = ResidualBlockWithStride(channel, channel, stride=2, inplace=inplace)

        self.adaptor_0 = DepthConvBlock(channel, channel, inplace=inplace)
        self.adaptor_1 = DepthConvBlock(channel * 2, channel, inplace=inplace)
        self.enc_3 = nn.Sequential(
            ResidualBlockWithStride(channel, channel, stride=2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
            nn.Conv2d(channel, channel, 3, stride=2, padding=1),
        )

    def forward(self, x, context, quant_step):
        out = self.enc_1(x)
        out = out * quant_step
        out = self.enc_2(out)
        if context is None:
            out = self.adaptor_0(out)
        else:
            out = self.adaptor_1(torch.cat((out, context), dim=1))
        return self.enc_3(out)


class MvDec(nn.Module):
    def __init__(self, output_channel, channel, inplace=False):
        super().__init__()
        self.dec_1 = nn.Sequential(
            DepthConvBlock(channel, channel, inplace=inplace),
            ResidualBlockUpsample(channel, channel, 2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
            ResidualBlockUpsample(channel, channel, 2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace)
        )
        self.dec_2 = ResidualBlockUpsample(channel, channel, 2, inplace=inplace)
        self.dec_3 = nn.Sequential(
            DepthConvBlock(channel, channel, inplace=inplace),
            subpel_conv1x1(channel, output_channel, 2),
        )

    def forward(self, x, quant_step):
        feature = self.dec_1(x)
        out = self.dec_2(feature)
        out = out * quant_step
        mv = self.dec_3(out)
        return mv, feature


class ContextualEncoder(nn.Module):
    def __init__(self, inplace=False, resblock_type='normal', n_div=4):
        super().__init__()
        if resblock_type == 'normal':
            _resblock = partial(ResBlock, bottleneck=True, slope=0.1, end_with_relu=True, inplace=inplace)
        elif resblock_type == 'pconv':
            _resblock = partial(ResBlock_PConv, bottleneck=True, slope=0.1, end_with_relu=True, inplace=inplace, n_div=n_div)
        
        self.conv1 = nn.Conv2d(g_ch_1x + 1, g_ch_2x, 3, stride=2, padding=1)
        self.res1 = _resblock(g_ch_2x * 2)
        self.conv2 = nn.Conv2d(g_ch_2x * 2, g_ch_4x, 3, stride=2, padding=1)
        self.res2 = _resblock(g_ch_4x * 2)
        self.conv3 = nn.Conv2d(g_ch_4x * 2, g_ch_8x, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(g_ch_8x, g_ch_16x, 3, stride=2, padding=1)

    def forward(self, x, context1, context2, context3, quant_step):
        feature = self.conv1(torch.cat([x, context1], dim=1))
        feature = self.res1(torch.cat([feature, context2], dim=1))
        feature = feature * quant_step
        feature = self.conv2(feature)
        feature = self.res2(torch.cat([feature, context3], dim=1))
        feature = self.conv3(feature)
        feature = self.conv4(feature)
        return feature


class ContextualDecoder(nn.Module):
    def __init__(self, inplace=False, resblock_type='normal', n_div=4):
        super().__init__()
        if resblock_type == 'normal':
            _resblock = partial(ResBlock, bottleneck=True, slope=0.1, end_with_relu=True, inplace=inplace)
        elif resblock_type == 'pconv':
            _resblock = partial(ResBlock_PConv, bottleneck=True, slope=0.1, end_with_relu=True, inplace=inplace, n_div=n_div)
        
        self.up1 = subpel_conv3x3(g_ch_16x, g_ch_8x, 2)
        self.up2 = subpel_conv3x3(g_ch_8x, g_ch_4x, 2)
        self.res1 = _resblock(g_ch_4x * 2)
        self.up3 = subpel_conv3x3(g_ch_4x * 2, g_ch_2x, 2)
        self.res2 = _resblock(g_ch_2x * 2)
        self.up4 = subpel_conv3x3(g_ch_2x * 2, 32, 2)

    def forward(self, x, context2, context3, quant_step):
        feature = self.up1(x)
        feature = self.up2(feature)
        feature = self.res1(torch.cat([feature, context3], dim=1))
        feature = self.up3(feature)
        feature = feature * quant_step
        feature = self.res2(torch.cat([feature, context2], dim=1))
        feature = self.up4(feature)
        return feature


class ReconGeneration(nn.Module):
    def __init__(self, ctx_channel=g_ch_1x, res_channel=32, inplace=False):
        super().__init__()
        self.first_conv = nn.Conv2d(ctx_channel + res_channel, g_ch_1x, 3, stride=1, padding=1)
        self.unet_1 = UNet(g_ch_1x, g_ch_1x, inplace=inplace)
        self.unet_2 = UNet(g_ch_1x, g_ch_1x, inplace=inplace)
        self.recon_conv = nn.Conv2d(g_ch_1x, 1, 3, stride=1, padding=1)

    def forward(self, ctx, res):
        feature = self.first_conv(torch.cat((ctx, res), dim=1))
        feature = self.unet_1(feature)
        feature = self.unet_2(feature)
        recon = self.recon_conv(feature)
        return feature, recon


class DMC(CompressionModel):
    def __init__(self, me_pretrain_path=None, spy_type='dc',
            anchor_num=4, ec_thread=False, stream_part=1, inplace=False, mse_type='rgb_mse', resblock_type='normal', n_div=4):
        super().__init__(y_distribution='laplace', z_channel=g_ch_16x, mv_z_channel=64,
                         ec_thread=ec_thread, stream_part=stream_part)

        channel_mv = 64
        channel_N = 64
        self.optic_flow = Gray_Spynet()
        self.align = OffsetDiversity(inplace=inplace)

        self.mv_encoder = MvEnc(2, channel_mv)
        self.mv_hyper_prior_encoder, self.mv_hyper_prior_decoder = \
            get_hyper_enc_dec_models(channel_mv, channel_N, inplace=inplace)

        self.mv_y_prior_fusion_adaptor_0 = DepthConvBlock(channel_mv * 1, channel_mv * 2,
                                                          inplace=inplace)
        self.mv_y_prior_fusion_adaptor_1 = DepthConvBlock(channel_mv * 2, channel_mv * 2,
                                                          inplace=inplace)

        self.mv_y_prior_fusion = nn.Sequential(
            DepthConvBlock(channel_mv * 2, channel_mv * 3, inplace=inplace),
            DepthConvBlock(channel_mv * 3, channel_mv * 3, inplace=inplace),
        )

        self.mv_y_spatial_prior_adaptor_1 = nn.Conv2d(channel_mv * 4, channel_mv * 3, 1)
        self.mv_y_spatial_prior_adaptor_2 = nn.Conv2d(channel_mv * 4, channel_mv * 3, 1)
        self.mv_y_spatial_prior_adaptor_3 = nn.Conv2d(channel_mv * 4, channel_mv * 3, 1)

        self.mv_y_spatial_prior = nn.Sequential(
            DepthConvBlock(channel_mv * 3, channel_mv * 3, inplace=inplace),
            DepthConvBlock(channel_mv * 3, channel_mv * 3, inplace=inplace),
            DepthConvBlock(channel_mv * 3, channel_mv * 2, inplace=inplace),
        )

        self.mv_decoder = MvDec(2, channel_mv, inplace=inplace)

        self.feature_adaptor_I = nn.Conv2d(1, g_ch_1x, 3, stride=1, padding=1)
        self.feature_adaptor = nn.ModuleList([nn.Conv2d(g_ch_1x, g_ch_1x, 1) for _ in range(3)])
        self.feature_extractor = FeatureExtractor(inplace=inplace, resblock_type=resblock_type, n_div=n_div)
        self.context_fusion_net = MultiScaleContextFusion(inplace=inplace, resblock_type=resblock_type, n_div=n_div)

        self.contextual_encoder = ContextualEncoder(inplace=inplace, resblock_type=resblock_type, n_div=n_div)

        self.contextual_hyper_prior_encoder, self.contextual_hyper_prior_decoder = \
            get_hyper_enc_dec_models(g_ch_16x, g_ch_16x, True, inplace=inplace)

        self.temporal_prior_encoder = nn.Sequential(
            nn.Conv2d(g_ch_4x, g_ch_8x, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=inplace),
            nn.Conv2d(g_ch_8x, g_ch_16x, 3, stride=2, padding=1),
        )

        self.y_prior_fusion_adaptor_0 = DepthConvBlock(g_ch_16x * 2, g_ch_16x * 3,
                                                       inplace=inplace)
        self.y_prior_fusion_adaptor_1 = DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3,
                                                       inplace=inplace)

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
        )

        self.y_spatial_prior_adaptor_1 = nn.Conv2d(g_ch_16x * 4, g_ch_16x * 3, 1)
        self.y_spatial_prior_adaptor_2 = nn.Conv2d(g_ch_16x * 4, g_ch_16x * 3, 1)
        self.y_spatial_prior_adaptor_3 = nn.Conv2d(g_ch_16x * 4, g_ch_16x * 3, 1)

        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 2, inplace=inplace),
        )

        self.contextual_decoder = ContextualDecoder(inplace=inplace, resblock_type=resblock_type, n_div=n_div)
        self.recon_generation_net = ReconGeneration(inplace=inplace)

        self.mv_y_q_basic_enc = nn.Parameter(torch.ones((1, channel_mv, 1, 1)))
        self.mv_y_q_scale_enc = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.mv_y_q_scale_enc_fine = None
        self.mv_y_q_basic_dec = nn.Parameter(torch.ones((1, channel_mv, 1, 1)))
        self.mv_y_q_scale_dec = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.mv_y_q_scale_dec_fine = None

        self.y_q_basic_enc = nn.Parameter(torch.ones((1, g_ch_2x * 2, 1, 1)))
        self.y_q_scale_enc = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.y_q_scale_enc_fine = None
        self.y_q_basic_dec = nn.Parameter(torch.ones((1, g_ch_2x, 1, 1)))
        self.y_q_scale_dec = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.y_q_scale_dec_fine = None
        self.anchor_num = int(anchor_num)

        self._initialize_weights()

        # if mv_enc_dec_pretrain_path is not None:
        #     self.load_mv_enc_dec_pretrain(mv_enc_dec_pretrain_path)
        #     print(f"loaded mv_enc_dec weights from: {mv_enc_dec_pretrain_path}")
        if me_pretrain_path is not None:
            self.load_me_pretrain(me_pretrain_path)
            print(f"loaded optic flow weights from: {me_pretrain_path}")
        


    def load_me_pretrain(self, path):
        ckpt = torch.load(path, map_location=torch.device('cpu'))
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        s = {}
        for k in ckpt:
            if k.startswith('optic_flow.'):
                s[k[len('optic_flow.'):]] = ckpt[k]
            else:
                s[k] = ckpt[k]
        self.optic_flow.load_state_dict(s)

    def multi_scale_feature_extractor(self, dpb, index):
        if dpb["ref_feature"] is None:
            feature = self.feature_adaptor_I(dpb["ref_frame"])
        else:
            index = index % 4
            index_map = [0, 1, 0, 2]
            index = index_map[index]
            feature = self.feature_adaptor[index](dpb["ref_feature"])
        return self.feature_extractor(feature)

    def motion_compensation(self, dpb, mv, index):
        warpframe = flow_warp(dpb["ref_frame"], mv)
        mv2 = bilineardownsacling(mv) / 2
        mv3 = bilineardownsacling(mv2) / 2
        ref_feature1, ref_feature2, ref_feature3 = self.multi_scale_feature_extractor(dpb, index)
        context1_init = flow_warp(ref_feature1, mv)
        context1 = self.align(ref_feature1, torch.cat(
            (context1_init, warpframe, mv), dim=1), mv)
        context2 = flow_warp(ref_feature2, mv2)
        context3 = flow_warp(ref_feature3, mv3)
        context1, context2, context3 = self.context_fusion_net(context1, context2, context3)
        return context1, context2, context3, warpframe

    @staticmethod
    def get_q_scales_from_ckpt(ckpt_path):
        ckpt = get_state_dict(ckpt_path)
        y_q_scale_enc = ckpt["y_q_scale_enc"].reshape(-1)
        y_q_scale_dec = ckpt["y_q_scale_dec"].reshape(-1)
        y_q_basic_enc = ckpt["y_q_basic_enc"].reshape(-1)
        y_q_basic_dec = ckpt["y_q_basic_dec"].reshape(-1)
        mv_y_q_scale_enc = ckpt["mv_y_q_scale_enc"].reshape(-1)
        mv_y_q_scale_dec = ckpt["mv_y_q_scale_dec"].reshape(-1)
        mv_y_q_basic_enc = ckpt["mv_y_q_basic_enc"].reshape(-1)
        mv_y_q_basic_dec = ckpt["mv_y_q_basic_dec"].reshape(-1)
        return y_q_basic_enc, y_q_basic_dec, y_q_scale_enc, y_q_scale_dec, mv_y_q_basic_enc, mv_y_q_basic_dec, mv_y_q_scale_enc, mv_y_q_scale_dec

    def mv_prior_param_decoder(self, mv_z_hat, dpb, slice_shape=None):
        mv_params = self.mv_hyper_prior_decoder(mv_z_hat)
        mv_params = self.slice_to_y(mv_params, slice_shape)
        ref_mv_y = dpb["ref_mv_y"]
        if ref_mv_y is None:
            mv_params = self.mv_y_prior_fusion_adaptor_0(mv_params)
        else:
            mv_params = torch.cat((mv_params, ref_mv_y), dim=1)
            mv_params = self.mv_y_prior_fusion_adaptor_1(mv_params)
        mv_params = self.mv_y_prior_fusion(mv_params)
        return mv_params

    def res_prior_param_decoder(self, z_hat, dpb, context3, slice_shape=None):
        hierarchical_params = self.contextual_hyper_prior_decoder(z_hat)
        hierarchical_params = self.slice_to_y(hierarchical_params, slice_shape)
        temporal_params = self.temporal_prior_encoder(context3)
        ref_y = dpb["ref_y"]
        if ref_y is None:
            params = torch.cat((temporal_params, hierarchical_params), dim=1)
            params = self.y_prior_fusion_adaptor_0(params)
        else:
            params = torch.cat((temporal_params, hierarchical_params, ref_y), dim=1)
            params = self.y_prior_fusion_adaptor_1(params)
        params = self.y_prior_fusion(params)
        return params

    def get_recon_and_feature(self, y_hat, context1, context2, context3, y_q_dec):
        recon_image_feature = self.contextual_decoder(y_hat, context2, context3, y_q_dec)
        feature, x_hat = self.recon_generation_net(recon_image_feature, context1)
        # x_hat = x_hat.clamp_(0, 1)
        return x_hat, feature

    def motion_estimation_and_mv_encoding(self, x, dpb, mv_y_q_enc):
        est_mv = self.optic_flow(x, dpb["ref_frame"])
        ref_mv_feature = dpb["ref_mv_feature"]
        mv_y = self.mv_encoder(est_mv, ref_mv_feature, mv_y_q_enc)
        return mv_y

    def get_q_for_inference(self, q_in_ckpt, q_index):
        mv_y_q_scale_enc = self.mv_y_q_scale_enc if q_in_ckpt else self.mv_y_q_scale_enc_fine
        mv_y_q_enc = self.get_curr_q(mv_y_q_scale_enc, self.mv_y_q_basic_enc, q_index=q_index)
        mv_y_q_scale_dec = self.mv_y_q_scale_dec if q_in_ckpt else self.mv_y_q_scale_dec_fine
        mv_y_q_dec = self.get_curr_q(mv_y_q_scale_dec, self.mv_y_q_basic_dec, q_index=q_index)

        y_q_scale_enc = self.y_q_scale_enc if q_in_ckpt else self.y_q_scale_enc_fine
        y_q_enc = self.get_curr_q(y_q_scale_enc, self.y_q_basic_enc, q_index=q_index)
        y_q_scale_dec = self.y_q_scale_dec if q_in_ckpt else self.y_q_scale_dec_fine
        y_q_dec = self.get_curr_q(y_q_scale_dec, self.y_q_basic_dec, q_index=q_index)
        return mv_y_q_enc, mv_y_q_dec, y_q_enc, y_q_dec

    def compress(self, x, dpb, q_in_ckpt, q_index, frame_idx, reset_entropy_coder=True, return_bit_stream=False):
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x
        mv_y_q_enc, mv_y_q_dec, y_q_enc, y_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)
        mv_y = self.motion_estimation_and_mv_encoding(x, dpb, mv_y_q_enc)
        mv_y_pad, slice_shape = self.pad_for_y(mv_y)
        mv_z = self.mv_hyper_prior_encoder(mv_y_pad)
        mv_z_hat = torch.round(mv_z)
        mv_params = self.mv_prior_param_decoder(mv_z_hat, dpb, slice_shape)
        mv_y_q_w_0, mv_y_q_w_1, mv_y_q_w_2, mv_y_q_w_3, \
            mv_scales_w_0, mv_scales_w_1, mv_scales_w_2, mv_scales_w_3, mv_y_hat = \
            self.compress_four_part_prior(
                mv_y, mv_params,
                self.mv_y_spatial_prior_adaptor_1, self.mv_y_spatial_prior_adaptor_2,
                self.mv_y_spatial_prior_adaptor_3, self.mv_y_spatial_prior)

        mv_hat, mv_feature = self.mv_decoder(mv_y_hat, mv_y_q_dec)
        context1, context2, context3, _ = self.motion_compensation(dpb, mv_hat, frame_idx)

        y = self.contextual_encoder(x, context1, context2, context3, y_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.contextual_hyper_prior_encoder(y_pad)
        z_hat = torch.round(z)
        params = self.res_prior_param_decoder(z_hat, dpb, context3, slice_shape)
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, \
            scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat = \
            self.compress_four_part_prior(
                y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
                self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        if reset_entropy_coder:
            self.entropy_coder.reset()
        self.bit_estimator_z_mv.encode(mv_z_hat)
        self.bit_estimator_z.encode(z_hat)
        self.gaussian_encoder.encode(mv_y_q_w_0, mv_scales_w_0)
        self.gaussian_encoder.encode(mv_y_q_w_1, mv_scales_w_1)
        self.gaussian_encoder.encode(mv_y_q_w_2, mv_scales_w_2)
        self.gaussian_encoder.encode(mv_y_q_w_3, mv_scales_w_3)
        self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        self.gaussian_encoder.encode(y_q_w_2, scales_w_2)
        self.gaussian_encoder.encode(y_q_w_3, scales_w_3)
        if reset_entropy_coder:
            self.entropy_coder.flush()

        x_hat, feature = self.get_recon_and_feature(y_hat, context1, context2, context3, y_q_dec)
        bit_stream = self.entropy_coder.get_encoded_stream() if return_bit_stream else None

        result = {
            "dpb": {
                "ref_frame": x_hat,
                "ref_feature": feature,
                "ref_mv_feature": mv_feature,
                "ref_y": y_hat,
                "ref_mv_y": mv_y_hat,
                "mv_hat": mv_hat, 
            },
        }
        if bit_stream is not None:
            result["bit_stream"] = bit_stream
        return result


    def decompress(self, dpb, height, width, q_in_ckpt, q_index, frame_idx, string=None):
        _, mv_y_q_dec, _, y_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        if string is not None:
            self.entropy_coder.set_stream(string)
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        z_size = get_downsampled_shape(height, width, 64)
        y_height, y_width = get_downsampled_shape(height, width, 16)
        slice_shape = self.get_to_y_slice_shape(y_height, y_width)
        mv_z_hat = self.bit_estimator_z_mv.decode_stream(z_size, dtype, device)
        z_hat = self.bit_estimator_z.decode_stream(z_size, dtype, device)
        mv_params = self.mv_prior_param_decoder(mv_z_hat, dpb, slice_shape)
        mv_y_hat = self.decompress_four_part_prior(mv_params,
                                                   self.mv_y_spatial_prior_adaptor_1,
                                                   self.mv_y_spatial_prior_adaptor_2,
                                                   self.mv_y_spatial_prior_adaptor_3,
                                                   self.mv_y_spatial_prior)

        mv_hat, mv_feature = self.mv_decoder(mv_y_hat, mv_y_q_dec)
        context1, context2, context3, _ = self.motion_compensation(dpb, mv_hat, frame_idx)

        params = self.res_prior_param_decoder(z_hat, dpb, context3, slice_shape)
        y_hat = self.decompress_four_part_prior(params,
                                                self.y_spatial_prior_adaptor_1,
                                                self.y_spatial_prior_adaptor_2,
                                                self.y_spatial_prior_adaptor_3,
                                                self.y_spatial_prior)
        x_hat, feature = self.get_recon_and_feature(y_hat, context1, context2, context3, y_q_dec)

        return {
            "dpb": {
                "ref_frame": x_hat,
                "ref_feature": feature,
                "ref_mv_feature": mv_feature,
                "ref_y": y_hat,
                "ref_mv_y": mv_y_hat,
                "mv_hat": mv_hat, 
            },
        }

    def encode_decode(self, x, dpb, q_in_ckpt, q_index, output_path=None,
                      pic_width=None, pic_height=None, frame_idx=None):
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x
        if output_path is not None:
            device = x.device
            torch.cuda.synchronize(device=device)
            t0 = time.time()
            encoded = self.compress(x, dpb, q_in_ckpt, q_index, frame_idx, reset_entropy_coder=True, return_bit_stream=True)
            encode_p(encoded['bit_stream'], q_in_ckpt, q_index, output_path)
            bits = filesize(output_path) * 8
            torch.cuda.synchronize(device=device)
            t1 = time.time()
            q_in_ckpt, q_index, string = decode_p(output_path)

            decoded = self.decompress(dpb, pic_height, pic_width,
                                      q_in_ckpt, q_index, frame_idx, string=string)
            torch.cuda.synchronize(device=device)
            t2 = time.time()
            result = {
                "dpb": decoded["dpb"],
                "bit": bits,
                "encoding_time": t1 - t0,
                "decoding_time": t2 - t1,
            }
            return result

        encoded = self.forward_one_frame(x, dpb, q_in_ckpt=q_in_ckpt, q_index=q_index, frame_idx=frame_idx)
        result = {
            "dpb": encoded['dpb'],
            "bit": encoded['bit'].item(),
            "encoding_time": 0,
            "decoding_time": 0,
        }
        return result


    def forward_one_frame(self, x, dpb, q_in_ckpt=False, q_index=None, frame_idx=None):
        mv_y_q_enc, mv_y_q_dec, y_q_enc, y_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)
        est_mv = self.optic_flow(x, dpb["ref_frame"])
        mv_y = self.mv_encoder(est_mv, dpb["ref_mv_feature"], mv_y_q_enc)
        mv_y_pad, slice_shape = self.pad_for_y(mv_y)
        mv_z = self.mv_hyper_prior_encoder(mv_y_pad)
        mv_z_hat = self.quant(mv_z)
        mv_params = self.mv_prior_param_decoder(mv_z_hat, dpb, slice_shape)
        mv_y_res, mv_y_q, mv_y_hat, mv_scales_hat = self.forward_four_part_prior(
            mv_y, mv_params, self.mv_y_spatial_prior_adaptor_1, self.mv_y_spatial_prior_adaptor_2,
            self.mv_y_spatial_prior_adaptor_3, self.mv_y_spatial_prior)

        mv_hat, mv_feature = self.mv_decoder(mv_y_hat, mv_y_q_dec)
        context1, context2, context3, warp_frame = self.motion_compensation(dpb, mv_hat, frame_idx)

        y = self.contextual_encoder(x, context1, context2, context3, y_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.contextual_hyper_prior_encoder(y_pad)
        z_hat = self.quant(z)
        params = self.res_prior_param_decoder(z_hat, dpb, context3, slice_shape)

        y_res, y_q, y_hat, scales_hat = self.forward_four_part_prior(
            y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior)
        recon_image_feature = self.contextual_decoder(y_hat, context2, context3, y_q_dec)
        feature, x_hat = self.recon_generation_net(recon_image_feature, context1)

        B, C, H, W = x.size()
        pixel_num = H * W


        y_for_bit = y_q
        mv_y_for_bit = mv_y_q
        z_for_bit = z_hat
        mv_z_for_bit = mv_z_hat
        
        bits_y = self.get_y_laplace_bits(y_for_bit, scales_hat)
        bits_mv_y = self.get_y_laplace_bits(mv_y_for_bit, mv_scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        bits_mv_z = self.get_z_bits(mv_z_for_bit, self.bit_estimator_z_mv)

        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        bpp_mv_y = torch.sum(bits_mv_y, dim=(1, 2, 3)) / pixel_num
        bpp_mv_z = torch.sum(bits_mv_z, dim=(1, 2, 3)) / pixel_num

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z
        bit = torch.sum(bpp) * pixel_num
        bit_y = torch.sum(bpp_y) * pixel_num
        bit_z = torch.sum(bpp_z) * pixel_num
        bit_mv_y = torch.sum(bpp_mv_y) * pixel_num
        bit_mv_z = torch.sum(bpp_mv_z) * pixel_num

        return {"bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                # "ssim": ssim,
                "dpb": {
                    "ref_frame": x_hat,
                    "ref_feature": feature,
                    "ref_mv_feature": mv_feature,
                    "ref_y": y_hat,
                    "ref_mv_y": mv_y_hat,
                    "mv_hat": mv_hat, 
                },
                "bit": bit,
                "bit_y": bit_y,
                "bit_z": bit_z,
                "bit_mv_y": bit_mv_y,
                "bit_mv_z": bit_mv_z,
                }


        

    def load_state_dict(self, state_dict):
        result_dict = {}
        message = ""
        for key, weight in state_dict.items():
            if key[-len("attn_mask"):] == "attn_mask":
                message += f", {key}"
                continue
            result_dict[key] = weight
        super().load_state_dict(result_dict)
