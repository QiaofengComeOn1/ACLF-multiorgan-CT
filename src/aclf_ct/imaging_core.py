"""Five-slice tensor preparation and embedding extraction."""

import os
import numpy as np
import torch
import torch.nn.functional as F
import timm


def np_to_torch_5ch(img_hw5):
    x = np.transpose(img_hw5, (2, 0, 1)).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.from_numpy(x)[None, ...]
    return x


def resize_tensor(x, size):
    return F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)


def standardize_batch(x):
    B = x.shape[0]
    x_ = x.view(B, -1)
    mean = x_.mean(dim=1, keepdim=True)
    std = x_.std(dim=1, keepdim=True).clamp_min(1e-06)
    x_ = (x_ - mean) / std
    return x_.view_as(x)


def get_bbox_from_mask(msk, margin=20):
    pos = np.where(msk > 0)
    if len(pos[0]) == 0:
        return (0, msk.shape[0], 0, msk.shape[1])
    y_min = max(int(np.min(pos[0])) - margin, 0)
    y_max = min(int(np.max(pos[0])) + margin + 1, msk.shape[0])
    x_min = max(int(np.min(pos[1])) - margin, 0)
    x_max = min(int(np.max(pos[1])) + margin + 1, msk.shape[1])
    return (y_min, y_max, x_min, x_max)


def crop_img_and_mask(img, msk, margin=20):
    y1, y2, x1, x2 = get_bbox_from_mask(msk, margin=margin)
    img_crop = img[y1:y2, x1:x2, :]
    msk_crop = msk[y1:y2, x1:x2, :]
    return (img_crop, msk_crop, (y1, y2, x1, x2))


def apply_mask(img, msk):
    return img * (msk > 0)


def preprocess_input(img_crop, msk_crop, use_mask_for_model=True):
    if use_mask_for_model:
        return apply_mask(img_crop, msk_crop)
    return img_crop.copy()


def adapt_first_conv_in_state_dict(sd: dict, target_in_chans: int=5, method: str='mean'):
    selected_key = None
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim == 4 and v.shape[1] == 3:
            selected_key = k
            break
    if selected_key is None:
        return (sd, None)
    w = sd[selected_key]
    if method == 'mean':
        w_mean = w.mean(dim=1, keepdim=True)
        w_new = w_mean.repeat(1, target_in_chans, 1, 1)
    elif method == 'replicate':
        reps = math.ceil(target_in_chans / 3)
        w_rep = w.repeat(1, reps, 1, 1)
        w_new = w_rep[:, :target_in_chans, :, :]
    else:
        raise ValueError(f'未知 ADAPT_METHOD: {method}')
    sd[selected_key] = w_new
    return (sd, selected_key)


def unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ['state_dict', 'model', 'net', 'model_state_dict']:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if isinstance(ckpt, dict):
        new_sd = {}
        for k, v in ckpt.items():
            nk = k.replace('module.', '')
            nk = nk.replace('_orig_mod.', '')
            new_sd[nk] = v
        return new_sd
    return ckpt


def filter_state_dict_for_model(sd, model):
    model_sd = model.state_dict()
    keep_sd = {}
    skipped = []
    for k, v in sd.items():
        if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape):
            keep_sd[k] = v
        else:
            skipped.append(k)
    return (keep_sd, skipped)


@torch.inference_mode()
def extract_embedding(model, x):
    try:
        y = model(x)
    except Exception:
        feat = model.forward_features(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        if feat.ndim == 4:
            feat = F.adaptive_avg_pool2d(feat, 1)
            feat = torch.flatten(feat, 1)
            return feat
        if feat.ndim == 3:
            if feat.shape[1] > 1:
                feat = feat.mean(dim=1)
            return feat
        return torch.flatten(feat, 1)
    if isinstance(y, (list, tuple)):
        y = y[0]
    if y.ndim == 4:
        y = F.adaptive_avg_pool2d(y, 1)
        y = torch.flatten(y, 1)
    elif y.ndim > 2:
        if y.ndim == 3 and y.shape[1] > 1:
            y = y.mean(dim=1)
        else:
            y = torch.flatten(y, 1)
    return y

import math
