import os
import numpy as np
import scipy.ndimage as ndimage


def parse_info_cfg(info_path):
    data = {}
    if not os.path.exists(info_path):
        return data
    with open(info_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, val = line.split(':', 1)
            data[key.strip()] = val.strip()
    return data


def preprocess_slice_exact(image_slice, mask_slice=None, current_spacing=(1.5, 1.5),
                           target_spacing=(1.25, 1.25), target_shape=(216, 256)):
    mean, std = np.mean(image_slice), np.std(image_slice)
    norm_img = (image_slice - mean) / (std + 1e-8)

    resize_factor = np.array(current_spacing) / np.array(target_spacing)
    new_shape = np.round(norm_img.shape * resize_factor).astype(int)
    real_resize_factor = new_shape / norm_img.shape

    resampled_img = ndimage.zoom(norm_img, real_resize_factor, order=3)
    resampled_mask = ndimage.zoom(mask_slice, real_resize_factor, order=0) if mask_slice is not None else None

    final_img = np.zeros(target_shape, dtype=np.float32)
    final_mask = np.zeros(target_shape, dtype=np.uint8) if mask_slice is not None else None

    h, w = resampled_img.shape
    th, tw = target_shape

    crop_h, pad_h = max(0, h - th), max(0, th - h)
    crop_w, pad_w = max(0, w - tw), max(0, tw - w)

    src_h_start = crop_h // 2
    src_w_start = crop_w // 2
    dst_h_start = pad_h // 2
    dst_w_start = pad_w // 2

    h_len = min(h, th)
    w_len = min(w, tw)

    final_img[dst_h_start:dst_h_start + h_len, dst_w_start:dst_w_start + w_len] = \
        resampled_img[src_h_start:src_h_start + h_len, src_w_start:src_w_start + w_len]

    if mask_slice is not None:
        final_mask[dst_h_start:dst_h_start + h_len, dst_w_start:dst_w_start + w_len] = \
            resampled_mask[src_h_start:src_h_start + h_len, src_w_start:src_w_start + w_len]

    return final_img, final_mask
