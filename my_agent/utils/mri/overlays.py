import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def _mid_slices_xyz(vol_xyz: np.ndarray):
    return vol_xyz.shape[0] // 2, vol_xyz.shape[1] // 2, vol_xyz.shape[2] // 2

def _normalize_0_1(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    return (x - mn) / (mx - mn + eps)

def render_overlay_pngs(image_path: str, segmentation_path: str, out_dir: str, alpha: float = 0.35) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    img = nib.load(image_path)
    seg = nib.load(segmentation_path)

    img_can = nib.as_closest_canonical(img)
    seg_can = nib.as_closest_canonical(seg)

    I = _normalize_0_1(img_can.get_fdata(dtype=np.float32))
    L = np.rint(seg_can.get_fdata(dtype=np.float32)).astype(np.int32)

    xmid, ymid, zmid = _mid_slices_xyz(I)

    views = {
        "SAG": (I[xmid, :, :], L[xmid, :, :]),
        "COR": (I[:, ymid, :], L[:, ymid, :]),
        "AXI": (I[:, :, zmid], L[:, :, zmid]),
    }

    cmap = plt.get_cmap("tab20")
    out_paths = {}

    for k, (im2d, lab2d) in views.items():
        im_show = np.rot90(im2d)
        lab_show = np.rot90(lab2d)

        class_ids = [int(c) for c in np.unique(lab_show) if int(c) != 0]
        H, W = lab_show.shape
        overlay_rgb = np.zeros((H, W, 3), dtype=np.float32)

        for cid in class_ids:
            color = np.array(cmap(cid % cmap.N)[:3], dtype=np.float32)
            overlay_rgb[lab_show == cid] = color

        fig = plt.figure(figsize=(5.5, 5.5), dpi=180)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.imshow(im_show, cmap="gray", interpolation="nearest")
        if class_ids:
            ax.imshow(overlay_rgb, alpha=alpha, interpolation="nearest")

        png_path = os.path.join(out_dir, f"mid_{k}.png")
        fig.savefig(png_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        out_paths[k] = png_path

    return out_paths
