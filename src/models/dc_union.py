import time

import torch
from torch import nn
from ..utils.stream_helper import encode_p, decode_p, filesize, get_state_dict
from src.models.dc_gray import DMC as GRAY_DC
from src.models.dc_color import DMC as COLOR_DC

import torch.nn.functional as F


class DMC(nn.Module):
    def __init__(self, gray_net: GRAY_DC, color_net: COLOR_DC):
        super().__init__()
        self.gray_net = gray_net
        self.color_net = color_net

        
        
    def build_dpb(self, ref_y=None, ref_uv=None):
        y_dpb = {
            "ref_frame": ref_y,
            "ref_feature": None,
            "ref_mv_feature": None,
            "ref_y": None,
            "ref_mv_y": None,
        }
        uv_dpb = {
            "ref_frame": ref_uv,
            "ref_feature": None,
            "ref_y": None,
        }
        return y_dpb, uv_dpb
        
    def get_q_scales_from_ckpt(ckpt_path):
        ckpt = get_state_dict(ckpt_path)
        y_info = {
            "y_q_scale_enc": ckpt["gray_net.y_q_scale_enc"].reshape(-1),
            "y_q_scale_dec": ckpt["gray_net.y_q_scale_dec"].reshape(-1),
            "mv_y_q_scale_enc": ckpt["gray_net.mv_y_q_scale_enc"].reshape(-1),
            "mv_y_q_scale_dec": ckpt["gray_net.mv_y_q_scale_dec"].reshape(-1),
        }
        uv_info = {
            "y_q_scale_enc": ckpt["color_net.y_q_scale_enc"].reshape(-1),
            "y_q_scale_dec": ckpt["color_net.y_q_scale_dec"].reshape(-1),
        }

        return y_info, uv_info
        

    def update(self, force=False):
        # ! step1: 初始gray_net的熵编码器
        self.gray_net.update(force=force)
        # ! step2: union_net的熵编码指向gray_net的熵编码器
        self.entropy_coder = self.gray_net.entropy_coder
        # ! step3: color_net的gaussian_encoder指向，bit_estimator_z和熵编码器绑定
        self.color_net.gaussian_encoder = self.gray_net.gaussian_encoder
        self.color_net.bit_estimator_z.update(force=force, entropy_coder=self.entropy_coder)

    def compress(self, cur_y, cur_uv, y_dpb, uv_dpb, q_in_ckpt, q_index, frame_idx):
        self.entropy_coder.reset()
        y_result = self.gray_net.compress(
            cur_y, y_dpb,
            q_in_ckpt=q_in_ckpt,
            q_index=q_index,
            frame_idx=frame_idx,
            reset_entropy_coder=False,
            return_bit_stream=False,
        )
        est_mv, y_ctx = y_result["dpb"]["mv_hat"], y_result["dpb"]["ref_feature"]
        uv_result = self.color_net.compress(
            cur_uv, est_mv, y_ctx, uv_dpb,
            q_in_ckpt=q_in_ckpt,
            q_index=q_index,
            frame_idx=frame_idx,
            reset_entropy_coder=False,
            return_bit_stream=False,
        )
        self.entropy_coder.flush()
        bit_stream = self.entropy_coder.get_encoded_stream()

        return y_result, uv_result, bit_stream


    
    def decompress(self, y_dpb, uv_dpb, string, pic_height, pic_width, q_in_ckpt, q_index, frame_idx):
        self.entropy_coder.set_stream(string)
        y_result = self.gray_net.decompress(y_dpb, pic_height, pic_width, q_in_ckpt, q_index, frame_idx)
        est_mv, y_ctx = y_result["dpb"]["mv_hat"], y_result["dpb"]["ref_feature"]
        uv_result = self.color_net.decompress(est_mv, y_ctx, uv_dpb, pic_height // 2, pic_width // 2, q_in_ckpt, q_index, frame_idx)
        return y_result, uv_result

        

    def encode_decode(self, cur_y, cur_uv, y_dpb, uv_dpb, q_in_ckpt, q_index, output_path=None,
                    pic_width=None, pic_height=None, frame_idx=None):
        if output_path is not None:
            device = cur_y.device
            torch.cuda.synchronize(device=device)
            t0 = time.time()
            _, _, bit_stream = self.compress(cur_y, cur_uv, y_dpb, uv_dpb, q_in_ckpt, q_index, frame_idx)
            encode_p(bit_stream, q_in_ckpt, q_index, output_path)
            bits = filesize(output_path) * 8
            torch.cuda.synchronize(device=device)
            t1 = time.time()
            q_in_ckpt, q_index, string = decode_p(output_path)
            decoded_y, decoded_uv = self.decompress(y_dpb, uv_dpb, string, pic_height, pic_width, q_in_ckpt, q_index, frame_idx)
            torch.cuda.synchronize(device=device)
            t2 = time.time()
            union_result = {
                "bit": bits,
                "encoding_time": t1 - t0,
                "decoding_time": t2 - t1,
            }
            return decoded_y, decoded_uv, union_result

        y_result, uv_result, _ = self.forward_one_frame(cur_y, cur_uv, y_dpb, uv_dpb, q_in_ckpt, q_index, frame_idx)
        y_result["bit"] = y_result["bit"].item()
        uv_result["bit"] = uv_result["bit"].item()
        union_result = {
            "bit": y_result["bit"] + uv_result["bit"],
            "bit_y": y_result["bit"],
            "bit_uv": uv_result["bit"],
            "encoding_time": 0,
            "decoding_time": 0,
        }
        
        return y_result, uv_result, union_result


    def forward_one_frame(self, gray_x, color_x, gray_dpb, color_dpb, q_in_ckpt=False, q_index=None, frame_idx=None):
        gray_result = self.gray_net.forward_one_frame(gray_x, gray_dpb, q_in_ckpt, q_index, frame_idx)
        est_mv, gray_ctx = gray_result["dpb"]["mv_hat"], gray_result["dpb"]["ref_feature"]

        color_result = self.color_net.forward_one_frame(color_x, est_mv, gray_ctx, color_dpb, q_in_ckpt, q_index, frame_idx)

        union_result = {
                        "bit": gray_result["bit"] + color_result["bit"],
                        "bit_y": gray_result["bit_y"] + color_result["bit_y"],
                        "bit_z": gray_result["bit_z"] + color_result["bit_z"],
                        "bit_mv_y": gray_result["bit_y"],
                        "bit_mv_z": gray_result["bit_mv_z"],
                    }
        return gray_result, color_result, union_result
    


    def load_state_dict(self, state_dict):
        result_dict = {}
        message = ""
        for key, weight in state_dict.items():
            if key[-len("attn_mask"):] == "attn_mask":
                message += f", {key}"
                continue
            result_dict[key] = weight
        super().load_state_dict(result_dict)