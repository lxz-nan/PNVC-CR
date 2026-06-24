#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

# 1) IP=-1, F=96
python test.py --is_debug False -w 8 \
    --i_frame_model_path model_pth/cvpr2023_image_yuv420_psnr.pth.tar \
    --p_frame_model_path model_pth/PNVC-CR.pth.tar \
    --test_config jsons/dataset_config_example_yuv420.json \
    --reset_interval 32 \
    --force_intra_period 100000 \
    --force_frame_num 96 \
    --pad_size 32 \
    --calc_ssim True \
    --output_path output/test_CR_ip-1_f96.json

# 2) IP=32, F=96
python test.py --is_debug False -w 8 \
    --i_frame_model_path model_pth/cvpr2023_image_yuv420_psnr.pth.tar \
    --p_frame_model_path model_pth/PNVC-CR.pth.tar \
    --test_config jsons/dataset_config_example_yuv420.json \
    --reset_interval 32 \
    --force_intra_period 32 \
    --force_frame_num 96 \
    --pad_size 32 \
    --calc_ssim True \
    --output_path output/test_CR_ip32_f96.json

# 3) IP=-1, F=-1
python test.py --is_debug False -w 8 \
    --i_frame_model_path model_pth/cvpr2023_image_yuv420_psnr.pth.tar \
    --p_frame_model_path model_pth/PNVC-CR.pth.tar \
    --test_config jsons/dataset_config_example_yuv420.json \
    --reset_interval 32 \
    --force_intra_period 100000 \
    --pad_size 32 \
    --calc_ssim True \
    --output_path output/test_CR_ip-1_fall.json
