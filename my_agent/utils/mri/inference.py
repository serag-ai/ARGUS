import sys
import numpy as np
import torch

from my_agent.utils.mri import quicknat as quicknat_module
import my_agent.nn_common_modules as nn_common_modules_module
import my_agent.squeeze_and_excitation as squeeze_and_excitation_module

from my_agent.utils.mri.constants import (
    DEFAULT_LABEL_NAMES, MODEL_TO_CANONICAL_ID_MAP, ENFORCE_MODEL_N_CLASSES
)
from my_agent.utils.mri.preprocess import load_and_preprocess_eval, reorient_from_canonical_to_original
from my_agent.utils.mri.orientation import from_slice_first_to_xyz
from my_agent.utils.mri.labels import remap_label_ids, validate_label_space


def load_quicknat_model(model_path: str, device: str):
    # Ensure checkpoint can resolve old module paths
    sys.modules.setdefault("quicknat", quicknat_module)
    sys.modules.setdefault("nn_common_modules", nn_common_modules_module)
    sys.modules.setdefault("squeeze_and_excitation", squeeze_and_excitation_module)

    cuda_available = torch.cuda.is_available() and device != "cpu"
    if cuda_available:
        model = torch.load(model_path, weights_only=False)
        model.to(device)
    else:
        model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
        device = "cpu"
    model.eval()
    return model, device


def segment_single_volume(image_path, model, orientation, batch_size, device):
    vol_shw_can, _can_img, orig_img, orig_ornt, can_ornt = load_and_preprocess_eval(image_path, orientation)

    if vol_shw_can.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol_shw_can.shape}")

    vol_nchw = vol_shw_can[:, np.newaxis, :, :]
    x = torch.from_numpy(vol_nchw).float()
    if device != "cpu":
        x = x.to(device)

    preds = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            out = model(x[i:i + batch_size])

            if out.ndim != 4:
                raise ValueError(f"Expected model output [B,C,H,W], got shape={tuple(out.shape)}")

            if ENFORCE_MODEL_N_CLASSES:
                n_classes = int(out.shape[1])
                expected = len(DEFAULT_LABEL_NAMES)
                if n_classes != expected:
                    raise ValueError(
                        f"Model outputs n_classes={n_classes}, but DEFAULT_LABEL_NAMES has {expected} entries.\n"
                        "Fix label list for this checkpoint OR define MODEL_TO_CANONICAL_ID_MAP."
                    )

            _, lab = torch.max(out, dim=1)
            preds.append(lab)

    pred_shw = torch.cat(preds, dim=0).cpu().numpy().astype(np.int16)
    pred_xyz_can = from_slice_first_to_xyz(pred_shw, orientation).astype(np.int16)
    pred_xyz_orig = reorient_from_canonical_to_original(pred_xyz_can, orig_ornt, can_ornt)

    if MODEL_TO_CANONICAL_ID_MAP is not None:
        pred_xyz_orig = remap_label_ids(pred_xyz_orig, MODEL_TO_CANONICAL_ID_MAP, keep_unmapped_as_bg=True)

    pred_xyz_orig = pred_xyz_orig.astype(np.int16, copy=False)

    orig_shape = orig_img.shape[:3]
    if pred_xyz_orig.shape != orig_shape:
        raise ValueError(
            f"Reoriented prediction shape does not match original image.\n"
            f"  pred_xyz_orig.shape = {pred_xyz_orig.shape}\n"
            f"  orig_img.shape      = {orig_shape}\n"
            "This indicates an orientation/axis transform mismatch."
        )

    val = validate_label_space(pred_xyz_orig, DEFAULT_LABEL_NAMES, allow_extra=False)
    if not val["ok"]:
        raise ValueError(
            "Segmentation label IDs do not match DEFAULT_LABEL_NAMES.\n"
            f"Details: {val}\n"
            "Set MODEL_TO_CANONICAL_ID_MAP to fix class ID mapping."
        )

    return pred_xyz_orig, orig_img.header, orig_img.affine
