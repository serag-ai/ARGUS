import os
import re
import numpy as np
import nibabel as nib

def sanitize_filename(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-]+", "", s)

def save_label_masks(vol_pred_xyz, affine, header, out_dir, base, orientation, label_names, skip_empty=True) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    mask_header = header.copy()
    mask_header.set_data_dtype(np.uint8)

    paths = {}
    for label_id in range(1, len(label_names)):
        label_name = label_names[label_id]
        mask = (vol_pred_xyz == label_id).astype(np.uint8)
        if skip_empty and int(mask.sum()) == 0:
            continue

        safe_label = sanitize_filename(label_name)
        mask_filename = f"{base}_{safe_label}_{orientation}_mask.nii.gz"
        mask_path = os.path.join(out_dir, mask_filename)

        nib.save(nib.Nifti1Image(mask, affine, mask_header), mask_path)
        paths[label_name] = mask_path

    return paths

def save_segmentation(vol_pred_xyz, affine, header, image_path, orientation):
    seg_header = header.copy()
    seg_header.set_data_dtype(np.int16)

    seg_dir = os.path.join(os.path.dirname(image_path), "segmentations")
    os.makedirs(seg_dir, exist_ok=True)

    base = os.path.basename(image_path).replace(".nii.gz", "").replace(".nii", "")
    segmentation_path = os.path.join(seg_dir, f"{base}_seg_{orientation}.nii.gz")

    nib.save(nib.Nifti1Image(vol_pred_xyz, affine, seg_header), segmentation_path)
    if not os.path.exists(segmentation_path):
        raise RuntimeError("Segmentation file was not created on disk.")

    regions_dir = os.path.join(seg_dir, "regions")
    return segmentation_path, regions_dir, base
