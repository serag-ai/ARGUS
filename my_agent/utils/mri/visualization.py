# my_agent/utils/visualization.py
import numpy as np
import os

# Try to import matplotlib, but do NOT crash if it fails
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception as e:
    print(f"[visualization] matplotlib import failed ({e}); overlays disabled.")
    plt = None
    _HAS_MPL = False

# New high-contrast palette.
# Labels follow your original ordering.
PALETTE = np.array([
    [0, 0, 0],           #  0: Background (black)

    [230, 230, 230],     #  1: Left WM (light grey)
    [255, 178, 102],     #  2: Left Cortex (orange)
    [0, 191, 255],       #  3: Left Lateral Ventricle (deep sky blue)
    [135, 206, 250],     #  4: Left Inf Lat Ventricle (light blue)

    [255, 255, 153],     #  5: Left Cerebellum WM (pale yellow)
    [255, 215, 0],       #  6: Left Cerebellum Cortex (gold)
    [50, 205, 50],       #  7: Left Thalamus (lime green)
    [0, 128, 0],         #  8: Left Caudate (green)
    [186, 85, 211],      #  9: Left Putamen (orchid)
    [221, 160, 221],     # 10: Left Pallidum (plum)

    [0, 255, 255],       # 11: 3rd Ventricle (cyan)
    [72, 209, 204],      # 12: 4th Ventricle (turquoise)
    [255, 99, 71],       # 13: Brain Stem (tomato)
    [220, 20, 60],       # 14: Left Hippocampus (crimson)
    [255, 105, 180],     # 15: Left Amygdala (hot pink)

    [176, 196, 222],     # 16: CSF (cranial, blue-grey)
    [0, 250, 154],       # 17: Left Accumbens (spring green)
    [199, 21, 133],      # 18: Left Ventral DC (violet red)

    [200, 200, 200],     # 19: Right WM (darker grey)
    [205, 133, 63],      # 20: Right Cortex (peru)
    [0, 0, 139],         # 21: Right Lateral Ventricle (dark blue)
    [65, 105, 225],      # 22: Right Inf Lat Ventricle (royal blue)

    [238, 232, 170],     # 23: Right Cerebellum WM (pale goldenrod)
    [218, 165, 32],      # 24: Right Cerebellum Cortex (goldenrod)
    [34, 139, 34],       # 25: Right Thalamus (forest green)
    [0, 100, 0],         # 26: Right Caudate (dark green)
    [148, 0, 211],       # 27: Right Putamen (dark violet)
    [139, 0, 139],       # 28: Right Pallidum (dark magenta)

    [178, 34, 34],       # 29: Right Hippocampus (firebrick)
    [219, 112, 147],     # 30: Right Amygdala (pale violet red)
    [46, 139, 87],       # 31: Right Accumbens (sea green)
    [123, 104, 238],     # 32: Right Ventral DC (medium slate blue)
], dtype=np.uint8)


def overlay_slice(
    image: np.ndarray,
    mask: np.ndarray,
    slice_idx: int,
    axis: int = 2,
    label_colors: dict | None = None,
    alpha: float = 0.8,           # more opaque by default
    out_path: str = "overlay.png",
):
    """
    Save an overlay image: segmentation mask drawn on top of the original MRI slice.

    image: 3D array (H, W, D)
    mask:  3D array (H, W, D) with integer labels
    slice_idx: which slice along 'axis' to visualize
    axis: 0, 1, or 2
    label_colors: optional mapping {label_id: [R, G, B]} with values in [0,255] or [0,1]
                  If None, uses the fixed PALETTE defined above.
    alpha: opacity of the segmentation colors (1.0 = fully solid).
    """

    if not _HAS_MPL or plt is None:
        print("[overlay_slice] matplotlib not available; skipping overlay save.")
        return

    # 1) Extract slice
    if axis == 0:
        img_slice = image[slice_idx, :, :]
        msk_slice = mask[slice_idx, :, :]
    elif axis == 1:
        img_slice = image[:, slice_idx, :]
        msk_slice = mask[:, slice_idx, :]
    else:
        img_slice = image[:, :, slice_idx]
        msk_slice = mask[:, :, slice_idx]

    # 2) Normalize image to [0, 1]
    img_slice = img_slice.astype(np.float32)
    vmin, vmax = np.percentile(img_slice, [1, 99])
    img_slice = np.clip((img_slice - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # 3) Grayscale -> RGB
    img_rgb = np.stack([img_slice] * 3, axis=-1)  # (H, W, 3)

    # 4) Build color mask
    msk_slice = msk_slice.astype(int)
    H, W = msk_slice.shape
    color_mask = np.zeros((H, W, 3), dtype=np.float32)

    if label_colors is not None:
        # Use colors coming from dataset_config.json (or elsewhere)
        for label_id, color in label_colors.items():
            color = np.array(color, dtype=np.float32)
            if color.max() > 1.0:
                color = color / 255.0
            color_mask[msk_slice == label_id] = color
    else:
        # Use fixed PALETTE by label index
        idx = np.clip(msk_slice, 0, PALETTE.shape[0] - 1)
        color_mask = PALETTE[idx].astype(np.float32) / 255.0

    # 5) Blend only where mask > 0
    blended = img_rgb.copy()
    mask_any = msk_slice > 0
    blended[mask_any] = (
        (1.0 - alpha) * img_rgb[mask_any] + alpha * color_mask[mask_any]
    )

    # 6) Save
    dir_name = os.path.dirname(out_path)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(blended, interpolation="nearest")  # no smoothing/shrinking
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"[overlay_slice] Saved overlay to: {out_path}")
