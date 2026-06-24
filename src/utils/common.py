import json
import os
from unittest.mock import patch


def str2bool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1")


def create_folder(path, print_if_create=False):
    if not os.path.exists(path):
        os.makedirs(path)
        if print_if_create:
            print(f"created folder: {path}")


@patch('json.encoder.c_make_encoder', None)
def dump_json(obj, fid, float_digits=-1, **kwargs):
    of = json.encoder._make_iterencode  # pylint: disable=W0212

    def inner(*args, **kwargs):
        args = list(args)
        # fifth argument is float formater which we will replace
        args[4] = lambda o: format(o, '.%df' % float_digits)
        return of(*args, **kwargs)

    with patch('json.encoder._make_iterencode', wraps=inner):
        json.dump(obj, fid, **kwargs)


def generate_log_json_dic(frame_num, frame_pixel_num, test_time, ave_enc_time,
                          ave_dec_time, frame_types, metric_dic, verbose=False):
    i_num = frame_types.count(0)
    p_num = frame_num - i_num

    log_result = {
        'frame_pixel_num': frame_pixel_num,
        'i_frame_num': i_num,
        'p_frame_num': p_num,
        'test_time': test_time,
        'ave_enc_time_ms': ave_enc_time,
        'ave_dec_time_ms': ave_dec_time
    }

    def calculate_metrics(metrics):
        i_metric = sum(metrics[idx] for idx in range(frame_num) if frame_types[idx] == 0)
        p_metric = sum(metrics[idx] for idx in range(frame_num) if frame_types[idx] != 0)
        all_metric = i_metric + p_metric
        return i_metric, p_metric, all_metric

    for key, values in metric_dic.items():
        i_metric, p_metric, all_metric = calculate_metrics(values)
        if 'bpp' in key:
            i_metric = i_metric / frame_pixel_num
            p_metric = p_metric / frame_pixel_num
            all_metric = all_metric / frame_pixel_num
        log_result[f'ave_i_frame_{key}'] = i_metric / i_num
        log_result[f'ave_p_frame_{key}'] = p_metric / p_num if p_num > 0 else 0
        log_result[f'ave_all_frame_{key}'] = all_metric / frame_num

    if verbose:
        log_result['frame_type'] = frame_types

    return log_result


def print_scales(dic):
    for k, v in dic.items():
        print(f"{k}: ", end='')
        for q in v:
            print(f"{q:.3f}, ", end='')
        print()
