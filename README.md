# PNVC-CR

Official testing code for PNVC-CR. 

## 1. Environment

The code follows the DCVC-DC environment and was tested with Python 3.8, PyTorch 1.11, and CUDA 11.3.

```bash
conda create -n pnvc-cr python=3.8
conda activate pnvc-cr

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## 2. Prepare Models And Data

Put the pretrained checkpoints under `model_pth/`:

```text
model_pth/
  cvpr2023_image_yuv420_psnr.pth.tar
  PNVC-CR.pth.tar
```

Put YUV test sequences under the dataset root used by the json config. By default, the configs use:

```text
datasets/Common_test_YUV/
```

For example:

```text
datasets/Common_test_YUV/
  HEVC_D/
    BasketballPass_416x240_50.yuv
    ...
  HEVC_C/
    BQMall_832x480_60.yuv
    ...
```

You can also edit `root_path` in `jsons/dataset_config_example_yuv420.json` or `jsons/debug.json` to point to your local dataset directory.

## 3. Compile Entropy Coder

This step is only needed if you want to write real bitstreams with `--write_stream True`.

```bash
bash shs/cpp.sh
```

## 4. Test PNVC-CR

Run the provided script:

```bash
bash shs/test_CR.sh
```

It evaluates PNVC-CR with three settings:

```text
IP=-1, F=96
IP=32, F=96
IP=-1, F=all
```

The results are saved under `output/`:

```text
output/test_CR_ip-1_f96.json
output/test_CR_ip32_f96.json
output/test_CR_ip-1_fall.json
```

To write bitstreams, compile first and add:

```bash
--write_stream True --stream_path output/stream/PNVC-CR
```

To save decoded YUV frames, add:

```bash
--save_decoded_frame True --decoded_frame_path output/decoded_frame/PNVC-CR
```
