import time
import torch
from torch import nn
import numpy as np
from .common_model import CompressionModel
from .video_net import ResBlock, UNet, bilinearupsacling, bilineardownsacling, \
    get_hyper_enc_dec_models, flow_warp, ResBlock_PConv
from ..layers.layers import subpel_conv3x3, DepthConvBlock
from ..utils.stream_helper import get_downsampled_shape, encode_p, decode_p, filesize, \
    get_state_dict
from functools import partial
g_ch_1x = 48
g_ch_2x = 64
g_ch_4x = 96
g_ch_8x = 96
g_ch_16x = 128


class OffsetDiversity(nn.Module):
    def __init__(self, in_channel=g_ch_1x, aux_feature_num=g_ch_1x+2+2, # 输入为色度信息，2个通道：g_ch_1x+3+2->g_ch_1x+2+2
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


class ContextualEncoder(nn.Module):
    def __init__(self, inplace=False, resblock_type='normal', n_div=4):
        super().__init__()
        if resblock_type == 'normal':
            _resblock = partial(ResBlock, bottleneck=True, slope=0.1, end_with_relu=True, inplace=inplace)
        elif resblock_type == 'pconv':
            _resblock = partial(ResBlock_PConv, bottleneck=True, slope=0.1, end_with_relu=True, inplace=inplace, n_div=n_div)
        
        self.conv1 = nn.Conv2d(g_ch_1x + 2, g_ch_2x, 3, stride=2, padding=1) # 输入为色度信息，2个通道
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
    def __init__(self, ctx_channel=g_ch_1x, res_channel=32, inplace=False, update_ctx=False):
        super().__init__()
        self.first_conv = nn.Conv2d(ctx_channel + res_channel, g_ch_1x, 3, stride=1, padding=1)
        self.unet_1 = UNet(g_ch_1x, g_ch_1x, inplace=inplace)
        self.unet_2 = UNet(g_ch_1x, g_ch_1x, inplace=inplace)
        self.recon_conv = nn.Conv2d(g_ch_1x, 2, 3, stride=1, padding=1) # 输出为色度信息，2个通道
        self.update_ctx = update_ctx
        if update_ctx:
        # 用作update_ctx
            self.downsample = nn.Conv2d(g_ch_1x, g_ch_1x, 3, stride=2, padding=1)
            self.tune_color_conv = nn.Conv2d(g_ch_1x, g_ch_1x, 3, stride=1, padding=1)
            self.tune_gray_conv = nn.Conv2d(g_ch_1x, g_ch_1x, 3, stride=1, padding=1)
            self.merge_conv = nn.Conv2d(g_ch_1x * 2, g_ch_1x, 1, stride=1, padding=0)
        

    def update_ctx_with_gray(self, ctx, gray_ctx):
        gray_ctx = self.downsample(gray_ctx)
        color_ctx = self.tune_color_conv(ctx)
        gray_ctx = self.tune_gray_conv(gray_ctx)
        weight_ctx = self.merge_conv(torch.cat((color_ctx, gray_ctx), dim=1))
        mask = torch.sigmoid(weight_ctx)
        ctx = ctx * mask
        return ctx
        
    
    def forward(self, ctx, res, gray_ctx=None):
        feature = self.first_conv(torch.cat((ctx, res), dim=1))
        feature = self.unet_1(feature)
        feature = self.unet_2(feature)
        if gray_ctx is not None and self.update_ctx:
            # 进行后处理
            feature = self.update_ctx_with_gray(feature, gray_ctx)
        recon = self.recon_conv(feature)
        return feature, recon


class DMC(CompressionModel):
    def __init__(self, anchor_num=4, ec_thread=False, stream_part=1, inplace=False, update_ctx=True, down_mv_type='bilinear', resblock_type='normal', n_div=4):
        super().__init__(y_distribution='laplace', z_channel=g_ch_16x, mv_z_channel=None,
                         ec_thread=ec_thread, stream_part=stream_part)
            
        self.down_mv_type = down_mv_type
        if self.down_mv_type == 'conv3':
            self.down_mv_conv = nn.Conv2d(2, 2, 3, stride=2, padding=1)

        self.align = OffsetDiversity(inplace=inplace)


        self.feature_adaptor_I = nn.Conv2d(2, g_ch_1x, 3, stride=1, padding=1) # 输入为色度信息，2个通道
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
        self.recon_generation_net = ReconGeneration(inplace=inplace, update_ctx=update_ctx)


        self.y_q_basic_enc = nn.Parameter(torch.ones((1, g_ch_2x * 2, 1, 1)))
        self.y_q_scale_enc = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.y_q_scale_enc_fine = None
        self.y_q_basic_dec = nn.Parameter(torch.ones((1, g_ch_2x, 1, 1)))
        self.y_q_scale_dec = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.y_q_scale_dec_fine = None
        self.anchor_num = int(anchor_num)

        self.noise_level = 0.4

        self._initialize_weights()


    def downsample_mv(self, mv):
        if self.down_mv_type == 'bilinear':
            return bilineardownsacling(mv) / 2
        elif self.down_mv_type == 'conv3':
            return self.down_mv_conv(mv) / 2
        else:
            return mv
        

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

    def get_recon_and_feature(self, y_hat, context1, context2, context3, y_q_dec, gray_ctx=None):
        recon_image_feature = self.contextual_decoder(y_hat, context2, context3, y_q_dec)
        feature, x_hat = self.recon_generation_net(recon_image_feature, context1, gray_ctx)
        x_hat = x_hat.clamp_(0, 1)
        return x_hat, feature


    def get_q_for_inference(self, q_in_ckpt, q_index):
        y_q_scale_enc = self.y_q_scale_enc if q_in_ckpt else self.y_q_scale_enc_fine
        y_q_enc = self.get_curr_q(y_q_scale_enc, self.y_q_basic_enc, q_index=q_index)
        y_q_scale_dec = self.y_q_scale_dec if q_in_ckpt else self.y_q_scale_dec_fine
        y_q_dec = self.get_curr_q(y_q_scale_dec, self.y_q_basic_dec, q_index=q_index)
        return y_q_enc, y_q_dec

    def compress(self, x, est_mv, gray_ctx, dpb, q_in_ckpt, q_index, frame_idx, reset_entropy_coder=True, return_bit_stream=False):
        est_mv = self.downsample_mv(est_mv)
        y_q_enc, y_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        context1, context2, context3, _ = self.motion_compensation(dpb, est_mv, frame_idx)

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
        self.bit_estimator_z.encode(z_hat)
        self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        self.gaussian_encoder.encode(y_q_w_2, scales_w_2)
        self.gaussian_encoder.encode(y_q_w_3, scales_w_3)
        if reset_entropy_coder:
            self.entropy_coder.flush()

        x_hat, feature = self.get_recon_and_feature(y_hat, context1, context2, context3, y_q_dec, gray_ctx)
        bit_stream = self.entropy_coder.get_encoded_stream() if return_bit_stream else None

        result = {
            "dpb": {
                "ref_frame": x_hat,
                "ref_feature": feature,
                "ref_y": y_hat,
            },
        }
        if bit_stream is not None:
            result["bit_stream"] = bit_stream
        return result


    def decompress(self, est_mv, gray_ctx, dpb, height, width, q_in_ckpt, q_index, frame_idx, string=None):
        est_mv = self.downsample_mv(est_mv)
        
        _, y_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        if string is not None:
            self.entropy_coder.set_stream(string)
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        z_size = get_downsampled_shape(height, width, 64)
        y_height, y_width = get_downsampled_shape(height, width, 16)
        slice_shape = self.get_to_y_slice_shape(y_height, y_width)

        z_hat = self.bit_estimator_z.decode_stream(z_size, dtype, device)


        context1, context2, context3, _ = self.motion_compensation(dpb, est_mv, frame_idx)

        params = self.res_prior_param_decoder(z_hat, dpb, context3, slice_shape)
        y_hat = self.decompress_four_part_prior(params,
                                                self.y_spatial_prior_adaptor_1,
                                                self.y_spatial_prior_adaptor_2,
                                                self.y_spatial_prior_adaptor_3,
                                                self.y_spatial_prior)
        x_hat, feature = self.get_recon_and_feature(y_hat, context1, context2, context3, y_q_dec, gray_ctx)

        return {
            "dpb": {
                "ref_frame": x_hat,
                "ref_feature": feature,
                "ref_y": y_hat,
            },
        }

    def encode_decode(self, x, est_mv, gray_ctx, dpb, q_in_ckpt, q_index, output_path=None,
                      pic_width=None, pic_height=None, frame_idx=None):
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x
        if output_path is not None:
            device = x.device
            torch.cuda.synchronize(device=device)
            t0 = time.time()
            encoded = self.compress(x, est_mv, gray_ctx, dpb, q_in_ckpt, q_index, frame_idx, reset_entropy_coder=True, return_bit_stream=True)
            encode_p(encoded['bit_stream'], q_in_ckpt, q_index, output_path)
            bits = filesize(output_path) * 8
            torch.cuda.synchronize(device=device)
            t1 = time.time()
            q_in_ckpt, q_index, string = decode_p(output_path)

            decoded = self.decompress(est_mv, gray_ctx, dpb, pic_height, pic_width,
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

        encoded = self.forward_one_frame(x, est_mv, gray_ctx, dpb, q_in_ckpt=q_in_ckpt, q_index=q_index, frame_idx=frame_idx)
        result = {
            "dpb": encoded['dpb'],
            "bit": encoded['bit'].item(),
            "encoding_time": 0,
            "decoding_time": 0,
        }
        return result

    def forward_one_frame(self, x, est_mv, gray_ctx, dpb, q_in_ckpt=False, q_index=None, frame_idx=None):
        est_mv = self.downsample_mv(est_mv)
        y_q_enc, y_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)
        context1, context2, context3, warp_frame = self.motion_compensation(dpb, est_mv, frame_idx)

        y = self.contextual_encoder(x, context1, context2, context3, y_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.contextual_hyper_prior_encoder(y_pad)
        z_hat = self.quant(z)
        params = self.res_prior_param_decoder(z_hat, dpb, context3, slice_shape)

        y_res, y_q, y_hat, scales_hat = self.forward_four_part_prior(
            y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior)
        x_hat, feature = self.get_recon_and_feature(y_hat, context1, context2, context3, y_q_dec, gray_ctx)

        B, _, H, W = x.size()
        pixel_num = H * W

        y_for_bit = y_q
        z_for_bit = z_hat
        bits_y = self.get_y_laplace_bits(y_for_bit, scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)

        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        bpp = bpp_y + bpp_z

        bit = torch.sum(bpp) * pixel_num
        bit_y = torch.sum(bpp_y) * pixel_num
        bit_z = torch.sum(bpp_z) * pixel_num

        return {"bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "dpb": {
                    "ref_frame": x_hat,
                    "ref_feature": feature,
                    "ref_y": y_hat,
                },
                "bit": bit,
                "bit_y": bit_y,
                "bit_z": bit_z,
                }



    def load_state_dict(self, state_dict):
        result_dict = {}
        message = ""
        for key, weight in state_dict.items():
            if key[-len("attn_mask"):] == "attn_mask":
                message += f", {key}"
                continue
            result_dict[key] = weight

        # print(f"**parameter   {message}    are ignored when loading model**")
        super().load_state_dict(result_dict)
        
