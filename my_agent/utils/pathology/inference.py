

# interface.py
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass
from skimage.feature import peak_local_max
from scipy.ndimage import maximum_filter
import pandas as pd
from torch.utils.data import DataLoader
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import matplotlib.pyplot as plt

from my_agent.utils.pathology.model import ResNetUNetAttention
from my_agent.utils.pathology.cache import (
    pathology_cache_key, cache_key_str, run_id_from_key,
    cache_set, cache_get
)


def _safe_stem(name: str) -> str:
    base = os.path.basename(str(name))
    stem, _ = os.path.splitext(base)
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in stem)[:200]

def extract_centroids_from_heatmap(heatmap, threshold=0.5, min_distance=10):
    heatmap = heatmap.squeeze()
    peaks = peak_local_max(heatmap, min_distance=min_distance, threshold_abs=threshold)
    return peaks

def evaluate_model(
    model_path,
    test_dataset,
    images_folder: str,
    out_root: str,
    threshold=0.5,
    batch_size=8,
    device=None,
    img_size=(224, 224),
):
    """
    Disk-backed predictions-only segmentation.

    Returns:
      heatmap_paths: list[str]  .npy float32 per image
      mask_paths:    list[str]  .npy uint8 per image
      test_filenames:list[str]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    # stable key
    key_dict = pathology_cache_key(
        images_folder=images_folder,
        model_path=model_path,
        device=str(torch_device),
        batch_size=batch_size,
        threshold=threshold,
        img_size=img_size,
    )
    key_str = cache_key_str(key_dict)

    
    cached = cache_get(key_str)
    if cached is not None:
        return (
            cached["heatmap_paths"],
            cached["mask_paths"],
            cached["test_filenames"],
        )

    
    run_id = run_id_from_key(key_dict)
    run_dir = os.path.join(os.path.abspath(out_root), run_id)
    heat_dir = os.path.join(run_dir, "heatmaps")
    mask_dir = os.path.join(run_dir, "masks")

    os.makedirs(heat_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    model = ResNetUNetAttention().to(torch_device)
    model.load_state_dict(torch.load(model_path, map_location=torch_device))
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=int(batch_size), shuffle=False)

    heatmap_paths = []
    mask_paths = []
    test_filenames = []

    print("Running predictions (no GT) ...")
    with torch.no_grad():
        global_idx = 0
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    imgs, _unused_heatmaps, fnames = batch
                elif len(batch) == 4:
                    imgs, fnames, _ow, _oh = batch
                elif len(batch) == 2:
                    imgs, fnames = batch
                else:
                    imgs, fnames = batch[0], batch[-1]
            else:
                raise ValueError("DataLoader batch must be a tuple/list.")

            imgs = imgs.to(torch_device)  # [B,3,H,W]
            logits = model(imgs)

            if logits.ndim == 4 and logits.shape[2:] != imgs.shape[2:]:
                logits = F.interpolate(
                    logits, size=imgs.shape[2:], mode="bilinear", align_corners=False
                )

            prob = torch.sigmoid(logits)
            if prob.ndim == 4:
                prob_2d = prob[:, 0]  # [B,H,W]
            elif prob.ndim == 3:
                prob_2d = prob
            else:
                raise ValueError(f"Unexpected prob shape: {tuple(prob.shape)}")

            prob_np = prob_2d.detach().cpu().numpy().astype(np.float32)  # [B,H,W]
            mask_np = (prob_np > float(threshold)).astype(np.uint8)      # [B,H,W]

            # normalize filenames list
            if isinstance(fnames, (list, tuple)):
                fn_list = list(fnames)
            else:
                try:
                    fn_list = list(fnames)
                except Exception:
                    fn_list = [fnames] * prob_np.shape[0]

            for i in range(prob_np.shape[0]):
                fname_i = fn_list[i] if i < len(fn_list) else fn_list[0]
                if isinstance(fname_i, (list, tuple)) and len(fname_i) == 1:
                    fname_i = fname_i[0]
                fname_i = str(fname_i)

                stem = _safe_stem(fname_i)
                suffix = f"{global_idx:06d}"
                hp = os.path.join(heat_dir, f"{stem}__{suffix}.npy")
                mp = os.path.join(mask_dir, f"{stem}__{suffix}.npy")

                np.save(hp, prob_np[i].astype(np.float32))
                np.save(mp, mask_np[i].astype(np.uint8))

                heatmap_paths.append(hp)
                mask_paths.append(mp)
                test_filenames.append(fname_i)

                global_idx += 1

    
    cache_set(key_str, {
        "run_dir": run_dir,
        "heatmap_paths": heatmap_paths,
        "mask_paths": mask_paths,
        "test_filenames": test_filenames,
        "cache_key": key_str,
        "key_dict": key_dict,
    })

    return heatmap_paths, mask_paths, test_filenames

def extract_and_compare_centroids_from_paths(
    heatmap_paths,
    test_filenames,
    threshold=0.6,
    min_distance=10,
):
    """
    Loads each .npy heatmap, extracts peak coordinates using
    extract_centroids_from_heatmap. 
    """
    results = []

    for hp, fname in zip(heatmap_paths, test_filenames):
        heat_np = np.load(hp).astype(np.float32)



        peaks = extract_centroids_from_heatmap(
            heat_np,
            threshold=float(threshold),
            min_distance=int(min_distance),
        )
        # print("peaks =", peaks)
        # print("num peaks =", len(peaks) if len(peaks) else 0)

        peaks = np.asarray(peaks)
        pred_count = int(len(peaks)) if peaks.size else 0
        coords = peaks.tolist() if peaks.size else []

        results.append({
            "filename": str(fname),
            "pred_count": pred_count,
            "heatmap_path": str(hp),
            "coords": coords,
        })

    return {
        "ok": True,
        "results": results,
    }


def overlay_cell_heatmap(
    image_path: str,
    heatmap_path: str,
    alpha: float = 0.45,
    out_path: str = "cell_overlay.png",
    threshold: float = 0.6,
    min_distance: int = 10,
):

    # Load image
    img = Image.open(image_path).convert("RGB")
    img_np = np.asarray(img).astype(np.float32)

    vmin, vmax = np.percentile(img_np, [1, 99])
    img_np = np.clip((img_np - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # Load heatmap
    heat = np.load(heatmap_path).astype(np.float32)

    # Extract peaks from original heatmap space
    peaks = extract_centroids_from_heatmap(
        heat,
        threshold=float(threshold),
        min_distance=int(min_distance),
    )
    # print("peaks =", peaks)
    # print("num peaks =", len(peaks) if len(peaks) else 0)

    peaks = np.asarray(peaks)

    # 4) Map coords to image size if needed
    H, W = img_np.shape[:2]
    hH, hW = heat.shape[:2]

    dir_name = os.path.dirname(out_path)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(img_np, interpolation="nearest")

    if peaks.size:
        for peak_y, peak_x in peaks:
            y_img = peak_y * H / hH
            x_img = peak_x * W / hW

            plt.scatter(
                x_img,
                y_img,
                s=180,
                facecolors="none",
                edgecolors="red",
                linewidths=2
            )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"[overlay_cell_heatmap] Saved marked image to: {out_path}")
    print(f"[overlay_cell_heatmap] Number of detected peaks: {len(peaks) if peaks.size else 0}")
    return out_path

