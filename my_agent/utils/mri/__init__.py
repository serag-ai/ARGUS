#!/usr/bin/env python
# coding: utf-8

import os

from my_agent.utils.mri.cache import get_cache, segment_cache_key
from my_agent.utils.mri.constants import DEFAULT_LABEL_NAMES
from my_agent.utils.mri.inference import load_quicknat_model, segment_single_volume
from my_agent.utils.mri.io_save import save_segmentation, save_label_masks


def run_quicknat_with_cache(image_path, model_path, orientation, device="cpu", batch_size=20) -> dict:
    cache = get_cache()
    key = segment_cache_key(image_path, model_path, orientation, device, batch_size)
    if key in cache:
        return cache[key]

    # load model
    try:
        model, device = load_quicknat_model(model_path, device)
    except Exception as e:
        out = {"ok": False, "reason": f"Failed to load model: {e}",
               "vol_pred": None, "header": None, "affine": None,
               "model_path": model_path, "orientation": orientation,
               "segmentation_path": "", "label_mask_paths": {}}
        cache[key] = out
        return out

    # inference
    try:
        vol_pred_xyz, header, affine = segment_single_volume(image_path, model, orientation, batch_size, device)
    except Exception as e:
        out = {"ok": False, "reason": f"Segmentation inference failed: {e}",
               "vol_pred": None, "header": None, "affine": None,
               "model_path": model_path, "orientation": orientation,
               "segmentation_path": "", "label_mask_paths": {}}
        cache[key] = out
        return out

    # save
    try:
        segmentation_path, regions_dir, base = save_segmentation(vol_pred_xyz, affine, header, image_path, orientation)
        label_mask_paths = save_label_masks(vol_pred_xyz, affine, header, regions_dir, base, orientation, DEFAULT_LABEL_NAMES)
        ok, reason = True, None
    except Exception as e:
        segmentation_path, label_mask_paths = "", {}
        ok, reason = False, f"Failed to save segmentation outputs: {e}"

    out = {"ok": ok, "reason": reason,
           "vol_pred": vol_pred_xyz, "header": header, "affine": affine,
           "model_path": model_path, "orientation": orientation,
           "segmentation_path": segmentation_path, "label_mask_paths": label_mask_paths}

    cache[key] = out
    return out
