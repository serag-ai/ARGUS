import os
import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation, ornt_transform, apply_orientation

from my_agent.utils.mri.orientation import to_slice_first_xyz

def load_and_preprocess_eval(image_path: str, orientation: str):
    parent = os.path.basename(os.path.dirname(image_path))
    fname = os.path.basename(image_path)
    if parent == "segmentations" or "_seg_" in fname:
        raise ValueError(
            f"load_and_preprocess_eval got a segmentation file as input:\n  {image_path}\n"
            "Pass the original MRI, not a *_seg_*.nii.gz file."
        )

    orig_img = nib.load(image_path)
    can_img = nib.as_closest_canonical(orig_img)

    orig_ornt = io_orientation(orig_img.affine)
    can_ornt = io_orientation(can_img.affine)

    vol_xyz_can = can_img.get_fdata(dtype=np.float32)

    vmin = float(np.min(vol_xyz_can))
    vmax = float(np.max(vol_xyz_can))
    vol_xyz_can = (vol_xyz_can - vmin) / (vmax - vmin + 1e-8)

    vol_shw_can = to_slice_first_xyz(vol_xyz_can, orientation)
    return vol_shw_can, can_img, orig_img, orig_ornt, can_ornt

def reorient_from_canonical_to_original(pred_xyz_can: np.ndarray, orig_ornt, can_ornt) -> np.ndarray:
    transform = ornt_transform(can_ornt, orig_ornt)
    return apply_orientation(pred_xyz_can, transform)
