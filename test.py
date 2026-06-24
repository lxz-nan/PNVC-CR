# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import time
from collections import defaultdict

os.environ['TORCH_HOME'] = '/model/2263836119/basic_model'
os.environ['TRANSFORMERS_CACHE'] = '/model/2263836119/basic_model'
cache_dir = '/model/2263836119/basic_model'

import numpy as np
import torch
import torch.nn.functional as F
from DISTS_pytorch import DISTS
from lpips import LPIPS
from tqdm import tqdm

from src.models.dc_color import DMC as DMC_DC_COLOR
from src.models.dc_gray import DMC as DMC_DC_GRAY
from src.models.dc_union import DMC as DMC_DC_UNION
from src.models.image_model import IntraNoAR
from src.transforms.functional import ycbcr420_to_444, yuv_420_to_444, ycbcr2rgb, yuv_444_to_420
from src.utils.metrics import calc_psnr, calc_msssim, calc_msssim_rgb
from src.utils.common import create_folder, dump_json, generate_log_json_dic, print_scales, str2bool
from src.utils.stream_helper import get_padding_size, get_state_dict
from src.utils.video_reader import YUVReader
from src.utils.video_writer import YUVWriter


def cvt_frame_type_split(frame):
    gray_frame, u_frame, v_frame = yuv_444_to_420(frame)
    color_frame = torch.cat((u_frame, v_frame), dim=1)
    return gray_frame, color_frame

def parse_args():
    parser = argparse.ArgumentParser(description='Example testing script')

    parser.add_argument('--ec_thread', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--stream_part_i', type=int, default=1)
    parser.add_argument('--stream_part_p', type=int, default=1)
    parser.add_argument('--i_frame_model_path', type=str)
    parser.add_argument('--p_frame_model_path', type=str, default=None)
    parser.add_argument('--rate_num', type=int, default=4)
    parser.add_argument('--i_frame_q_indexes', type=int, nargs='+')
    parser.add_argument('--p_frame_q_indexes', type=int, nargs='+')
    parser.add_argument('--force_intra', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--force_frame_num', type=int, default=-1)
    parser.add_argument('--force_intra_period', type=int, default=-1)
    parser.add_argument('--test_config', type=str)
    parser.add_argument('--yuv420', type=str2bool, default=True)
    parser.add_argument('--worker', '-w', type=int, default=1, help='worker number')
    parser.add_argument('--cuda', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--cuda_device', default='0')
    
    parser.add_argument('--calc_ssim', type=str2bool, default=False, required=False)
    parser.add_argument('--write_stream', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--stream_path', type=str, default='out_bin')
    parser.add_argument('--save_decoded_frame', type=str2bool, default=False)
    parser.add_argument('--decoded_frame_path', type=str, default='decoded_frames')
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--is_debug', type=str2bool, default=True)
    parser.add_argument('--pad_size', type=int, default=32)
    parser.add_argument('--update_ctx', type=str2bool, default=True)

    parser.add_argument('--gray_resblock_type', type=str, default='pconv')
    parser.add_argument('--gray_n_div', type=float, default=1.5)
    parser.add_argument('--color_resblock_type', type=str, default='pconv')
    parser.add_argument('--color_n_div', type=float, default=1.5)
    
    parser.add_argument('--reset_interval', type=int, default=32)

    parser.set_defaults(i_frame_model_path='model_pth/cvpr2023_image_yuv420_psnr.pth.tar')
    parser.set_defaults(test_config='jsons/dataset_config_example_yuv420.json')
    parser.set_defaults(output_path='output/test.json')
    args = parser.parse_args()

    if args.is_debug:
        args.p_frame_model_path = 'model_pth/PNVC-CR.pth.tar'
        args.test_config = 'jsons/debug.json'
        args.verbose = 2
        args.worker = 4
        args.force_intra_period = 1000000
        args.force_frame_num = 96
        args.output_path = 'output/CR_F96_IP32.json'
        # args.calc_ssim = True
        # args.write_stream = True
        # args.stream_path = 'output/stream/CR_F96_IP32'
        # args.save_decoded_frame = True
        # args.decoded_frame_path = 'output/decoded_frame/CR_F96_IP32'


    print(json.dumps(vars(args), indent=4))
    return args


def np_image_to_tensor(img):
    image = torch.from_numpy(img).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    return image


def get_distortion(x_rec, x, lpips_net, dists_net, calc_ssim=False):
    y_rec, u_rec, v_rec = [t.squeeze(0).squeeze(0).cpu().numpy() for t in x_rec]
    y, u, v = [t.squeeze(0).squeeze(0).cpu().numpy() for t in x]
    
    psnr_y = calc_psnr(y, y_rec, data_range=1)
    psnr_u = calc_psnr(u, u_rec, data_range=1)
    psnr_v = calc_psnr(v, v_rec, data_range=1)
    psnr = (6 * psnr_y + psnr_u + psnr_v) / 8

    if calc_ssim:
        ssim_y = calc_msssim(y, y_rec, data_range=1)
        ssim_u = calc_msssim(u, u_rec, data_range=1)
        ssim_v = calc_msssim(v, v_rec, data_range=1)
    else:
        ssim_y, ssim_u, ssim_v = 0., 0., 0.
        
    ssim = (6 * ssim_y + ssim_u + ssim_v) / 8

    curr_psnr = [psnr, psnr_y, psnr_u, psnr_v]
    curr_ssim = [ssim, ssim_y, ssim_u, ssim_v]

    rgb_rec = ycbcr2rgb(yuv_420_to_444(x_rec))      
    rgb_ori = ycbcr2rgb(yuv_420_to_444(x))
    lpips_val = lpips_net(rgb_rec, rgb_ori).item()
    dists_val = dists_net(rgb_rec, rgb_ori).item()
    
    return curr_psnr, curr_ssim, lpips_val, dists_val


def run_test(union_net, i_frame_net, lpips_alex_net, dists_vgg_net, args):
    frame_num = args['frame_num']
    gop_size = args['gop_size']
    write_stream = 'write_stream' in args and args['write_stream']
    save_decoded_frame = 'save_decoded_frame' in args and args['save_decoded_frame']
    verbose = args.get('verbose', 0)
    device = next(i_frame_net.parameters()).device
    reset_interval = args['reset_interval']

    src_reader = YUVReader(args['src_path'], args['src_width'], args['src_height'])

    if save_decoded_frame:
        recon_writer = YUVWriter(args['recon_path'], args['src_width'], args['src_height'])

    frame_types = []
    metric_dic = defaultdict(list)
    frame_pixel_num = 0
    start_time = time.time()
    p_frame_number = 0
    overall_p_encoding_time = overall_p_decoding_time = 0
    
    calc_ssim = args.get('calc_ssim', False)

    with torch.no_grad():
        for frame_idx in range(frame_num):
            frame_start_time = time.time()
            
            y, uv = src_reader.read_one_frame(dst_format="420")
            yuv = ycbcr420_to_444(y, uv, order=0)
            
            x = np_image_to_tensor(yuv).to(device)
            y_tensor = np_image_to_tensor(y).to(device)
            uv_tensor = np_image_to_tensor(uv).to(device)
            u_tensor, v_tensor = uv_tensor[:, 0:1, :, :], uv_tensor[:, 1:2, :, :]

            pic_height, pic_width = x.shape[2], x.shape[3]
            if frame_pixel_num == 0:
                frame_pixel_num = pic_height * pic_width
            else:
                assert frame_pixel_num == pic_height * pic_width
            
            padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width, args['pad_size'])
            x_padded = F.pad(x, (padding_l, padding_r, padding_t, padding_b), mode="replicate")
            
            pad_height, pad_width = x_padded.shape[2], x_padded.shape[3]

            bin_path = os.path.join(args['bin_folder'], f"{frame_idx}.bin") if write_stream else None
                
            if frame_idx % gop_size == 0:
                result = i_frame_net.encode_decode(x_padded, args['q_in_ckpt'], args['i_frame_q_index'], bin_path, pic_height=pad_height, pic_width=pad_width)
                recon_gray_frame, recon_color_frame = cvt_frame_type_split(result["x_hat"])
                gray_dpb, color_dpb = union_net.build_dpb(recon_gray_frame, recon_color_frame)
                
                frame_types.append(0)
                gray_bpp, color_bpp = result["bit"], result["bit"]
                bpp = result["bit"]
                
            else:
                if reset_interval > 0 and frame_idx % reset_interval == 1:
                    gray_dpb, color_dpb = union_net.build_dpb(gray_dpb["ref_frame"], color_dpb["ref_frame"])
                
                gray_x, color_x = cvt_frame_type_split(x_padded)
                gray_result, color_result, union_result = union_net.encode_decode(
                    gray_x, color_x, gray_dpb, color_dpb, args['q_in_ckpt'],
                    args['p_frame_q_index'], bin_path, pic_height=pad_height, pic_width=pad_width, frame_idx=frame_idx % 4)
                
                gray_dpb = gray_result["dpb"]
                color_dpb = color_result["dpb"]
                gray_bpp = union_result.get("bit_y", 0.0)
                color_bpp = union_result.get("bit_uv", 0.0)
                bpp = union_result["bit"]
                
                recon_gray_frame = gray_dpb["ref_frame"]                                       
                recon_color_frame = color_dpb["ref_frame"]
                frame_types.append(1)
                p_frame_number += 1
                overall_p_encoding_time += union_result['encoding_time']
                overall_p_decoding_time += union_result['decoding_time']
            
            recon_gray_frame = F.pad(recon_gray_frame, (-padding_l, -padding_r, -padding_t, -padding_b)).clamp(0, 1)
            recon_color_frame = F.pad(recon_color_frame, (-padding_l // 2, -padding_r // 2, -padding_t // 2, -padding_b // 2)).clamp(0, 1)
            
            y_hat = recon_gray_frame
            u_hat = recon_color_frame[:, 0:1, :, :]
            v_hat = recon_color_frame[:, 1:2, :, :]
            
            curr_psnr, curr_ssim, lpips_val, dists_val = get_distortion(
                x_rec=(y_hat, u_hat, v_hat),
                x=(y_tensor, u_tensor, v_tensor),
                lpips_net=lpips_alex_net,
                dists_net=dists_vgg_net,
                calc_ssim=calc_ssim
            )

            psnr, psnr_y, psnr_u, psnr_v = curr_psnr
            ssim, ssim_y, ssim_u, ssim_v = curr_ssim
            
            metric_dic['psnr'].append(psnr)
            metric_dic['psnr_y'].append(psnr_y)
            metric_dic['psnr_u'].append(psnr_u)
            metric_dic['psnr_v'].append(psnr_v)
            metric_dic['psnr_yuv420'].append(psnr)
            if calc_ssim:
                metric_dic['ssim_y'].append(ssim_y)
                metric_dic['ssim_u'].append(ssim_u)
                metric_dic['ssim_v'].append(ssim_v)
                metric_dic['ssim_yuv420'].append(ssim)
            
            metric_dic['bpp'].append(bpp)
            if write_stream is None:
                metric_dic['bpp_y'].append(gray_bpp)
                metric_dic['bpp_uv'].append(color_bpp)
                metric_dic['color_rate_ratio'].append(color_bpp / bpp)
            metric_dic['lpips'].append(lpips_val)
            metric_dic['dists'].append(dists_val)

            frame_end_time = time.time()

            if verbose >= 2:
                print(f"frame_idx:{frame_idx}, {frame_end_time - frame_start_time:.3f} s, bpp: {bpp:.1f}, psnr_yuv420: {psnr:.3f}")
                
            if save_decoded_frame:
                y_hat = y_hat.squeeze(0).cpu().numpy()
                uv_hat = recon_color_frame.squeeze(0).cpu().numpy()
                recon_writer.write_one_frame(y=y_hat, uv=uv_hat, src_format='420')
                
    src_reader.close()
    if save_decoded_frame:
        recon_writer.close()
        
    test_time = time.time() - start_time
    ave_enc_time = overall_p_encoding_time/p_frame_number * 1000
    ave_dec_time = overall_p_decoding_time/p_frame_number * 1000
    if verbose >= 1 and p_frame_number > 0:
        print(f"encoding/decoding {p_frame_number} P frames, "
              f"average encoding time {ave_enc_time:.0f} ms, "
              f"average decoding time {ave_dec_time:.0f} ms.")

    log_result = generate_log_json_dic(frame_num, frame_pixel_num, test_time, ave_enc_time, ave_dec_time, frame_types, metric_dic)
    return log_result


i_frame_net = None  # the model is initialized after each process is spawn, thus OK for multiprocess
union_net = None
lpips_alex_net = None
dists_vgg_net = None

def encode_one(args):
    global i_frame_net
    global union_net
    global lpips_alex_net
    global dists_vgg_net

    sub_dir_name = args['video_path']
    bin_folder = os.path.join(args['stream_path'], sub_dir_name, str(args['rate_idx']))
    if args['write_stream']:
        create_folder(bin_folder, True)

    if args['save_decoded_frame']:
        recon_path = f"{args['decoded_frame_path']}/{args['rate_idx']}/{args['ds_name']}/{args['video_path']}"
        os.makedirs(os.path.dirname(recon_path), exist_ok=True)
    else:
        recon_path = None

    args['src_path'] = os.path.join(args['dataset_path'], sub_dir_name)
    args['bin_folder'] = bin_folder
    args['recon_path'] = recon_path
    try:
        result = run_test(union_net, i_frame_net, lpips_alex_net, dists_vgg_net, args)
    except Exception as e:
        print(f"Error processing {args['video_path']}: {e}")
        result = {}  
    result['ds_name'] = args['ds_name']
    result['video_path'] = args['video_path']
    result['rate_idx'] = args['rate_idx']

    return result


def worker(args):
    return encode_one(args)


def init_func(args):
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)
    gpu_num = 0
    if args.cuda:
        gpu_num = torch.cuda.device_count()

    process_name = multiprocessing.current_process().name
    process_idx = int(process_name[process_name.rfind('-') + 1:])
    gpu_id = -1
    if gpu_num > 0:
        gpu_id = process_idx % gpu_num
    if gpu_id >= 0:
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"

    global i_frame_net
    i_state_dict = get_state_dict(args.i_frame_model_path)
    i_frame_net = IntraNoAR(ec_thread=args.ec_thread, stream_part=args.stream_part_i,
                            inplace=True)
    i_frame_net.load_state_dict(i_state_dict)
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()

    global union_net
    global lpips_alex_net
    global dists_vgg_net
    if not args.force_intra:
        gray_video_net = DMC_DC_GRAY(ec_thread=args.ec_thread, stream_part=args.stream_part_p,
                        inplace=True, resblock_type=args.gray_resblock_type, n_div=args.gray_n_div)
        color_video_net = DMC_DC_COLOR(ec_thread=args.ec_thread, stream_part=args.stream_part_p,
                        inplace=True, update_ctx=args.update_ctx, resblock_type=args.color_resblock_type, n_div=args.color_n_div)
        
        union_net = DMC_DC_UNION(gray_net=gray_video_net, color_net=color_video_net)
        if args.p_frame_model_path is not None:
            union_state_dict = get_state_dict(args.p_frame_model_path)
            union_net.load_state_dict(union_state_dict)

        union_net = union_net.to(device)
        union_net.eval()
        
    lpips_alex_net = LPIPS(net='alex').to(device).eval()
    lpips_alex_net.requires_grad_(False)
    dists_vgg_net = DISTS().to(device).eval()

    if args.write_stream:
        if union_net is not None:
            union_net.update(force=True)
        i_frame_net.update(force=True)


def main(args):
    begin_time = time.time()

    torch.backends.cudnn.enabled = True


    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    worker_num = args.worker
    assert worker_num >= 1

    with open(args.test_config) as f:
        config = json.load(f)

    multiprocessing.set_start_method("spawn")
    threadpool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_num, initializer=init_func, initargs=(args,))
    objs = []

    count_frames = 0
    count_sequences = 0

    rate_num = args.rate_num
    i_frame_q_scale_enc, i_frame_q_scale_dec = \
        IntraNoAR.get_q_scales_from_ckpt(args.i_frame_model_path)

    print_scales({"q_scale_enc in intra ckpt": i_frame_q_scale_enc, "q_scale_dec in intra ckpt": i_frame_q_scale_dec})
    i_frame_q_indexes = []
    q_in_ckpt = False
    if args.i_frame_q_indexes is not None:
        assert len(args.i_frame_q_indexes) == rate_num
        i_frame_q_indexes = args.i_frame_q_indexes
    elif len(i_frame_q_scale_enc) == rate_num:
        assert rate_num == 4
        q_in_ckpt = True
        i_frame_q_indexes = [0, 1, 2, 3]
    else:
        assert rate_num >= 2 and rate_num <= 64
        for i in np.linspace(0, 63, num=rate_num):
            i_frame_q_indexes.append(int(i+0.5))


    if not args.force_intra:
        p_frame_q_indexes = []
        p_frame_q_indexes = i_frame_q_indexes

    print(f"testing {rate_num} rates, using q_indexes: ", end='')
    for q in i_frame_q_indexes:
        print(f"{q}, ", end='')

    root_path = config['root_path']
    config = config['test_classes']
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        for seq_name in config[ds_name]['sequences']:
            count_sequences += 1
            for rate_idx in range(rate_num):
                cur_args = {}
                cur_args['rate_idx'] = rate_idx
                cur_args['q_in_ckpt'] = q_in_ckpt
                cur_args['i_frame_q_index'] = i_frame_q_indexes[rate_idx]
                if not args.force_intra:
                    cur_args['p_frame_q_index'] = p_frame_q_indexes[rate_idx]
                cur_args['force_intra'] = args.force_intra
                cur_args['video_path'] = seq_name
                cur_args['dist_in_yuv420'] = args.yuv420
                cur_args['src_type'] = config[ds_name]['src_type']
                cur_args['src_height'] = config[ds_name]['sequences'][seq_name]['height']
                cur_args['src_width'] = config[ds_name]['sequences'][seq_name]['width']
                cur_args['gop_size'] = config[ds_name]['sequences'][seq_name]['intra_period']
                if args.force_intra:
                    cur_args['gop_size'] = 1
                if args.force_intra_period > 0:
                    cur_args['gop_size'] = args.force_intra_period
                cur_args['frame_num'] = config[ds_name]['sequences'][seq_name]['frames']
                if args.force_frame_num > 0:
                    cur_args['frame_num'] = args.force_frame_num
                cur_args['calc_ssim'] = args.calc_ssim
                if "dataset_path" in config[ds_name]:
                    cur_args['dataset_path'] = config[ds_name]['dataset_path']
                else:
                    cur_args['dataset_path'] = os.path.join(root_path, config[ds_name]['base_path'])
                cur_args['reset_interval'] = args.reset_interval
                cur_args['write_stream'] = args.write_stream
                cur_args['stream_path'] = args.stream_path
                cur_args['save_decoded_frame'] = args.save_decoded_frame
                cur_args['decoded_frame_path'] = f'{args.decoded_frame_path}'
                cur_args['ds_name'] = ds_name
                cur_args['verbose'] = args.verbose
                cur_args['pad_size'] = args.pad_size
                cur_args['update_ctx'] = args.update_ctx
                count_frames += cur_args['frame_num']

                obj = threadpool_executor.submit(worker, cur_args)
                objs.append(obj)

    results = []
    for obj in tqdm(objs):
        result = obj.result()
        results.append(result)

    log_result = {}
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        log_result[ds_name] = {}
        for seq in config[ds_name]['sequences']:
            if '.yuv' == seq[-4:]: seq=seq[:-4]
            log_result[ds_name][seq] = {}
            for rate in range(rate_num):
                for res in results:
                    if res['rate_idx'] == rate and ds_name == res['ds_name'] \
                            and seq in res['video_path']:
                        log_result[ds_name][seq][f"{rate:03d}"] = res

    out_json_dir = os.path.dirname(args.output_path)
    if len(out_json_dir) > 0:
        create_folder(out_json_dir, True)
    with open(args.output_path, 'w') as fp:
        dump_json(log_result, fp, float_digits=6, indent=2)
        print(f"write log to {args.output_path}")

    total_minutes = (time.time() - begin_time) / 60
    print('Test finished')
    print(f'Tested {count_frames} frames from {count_sequences} sequences')
    print(f'Total elapsed time: {total_minutes:.1f} min')


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        main(args)
