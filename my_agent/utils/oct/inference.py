import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from my_agent.utils.oct.constants import OCT_INPUT_SHAPE
from my_agent.utils.oct.preprocess import load_and_preprocess_image
from my_agent.utils.oct.model_registry import get_oct_model
from my_agent.utils.oct.postprocess import softmax_to_labelmap

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def run_oct_segmentation(
    image_path: str,
    model_path: str,
    out_dir: str,
    save_prob_npy: bool = True,
    save_label_png: bool = True,
) -> dict:
    """
    Loads OCT U-Net model (.keras) and predicts a softmax mask.
    Outputs:
      out_dir/prob/<stem>.npy   float32 [H,W,C] (optional)
      out_dir/label/<stem>.png  uint8 label map (optional)
    """
    _ensure_dir(out_dir)
    prob_dir = os.path.join(out_dir, "prob")
    lab_dir  = os.path.join(out_dir, "label")
    if save_prob_npy: _ensure_dir(prob_dir)
    if save_label_png: _ensure_dir(lab_dir)

    model = get_oct_model(model_path)

    # preprocess
    x = load_and_preprocess_image(image_path, target_size=(OCT_INPUT_SHAPE[0], OCT_INPUT_SHAPE[1]))  # [1,H,W,3]

    # predict [1,H,W,C]
    prob = model.predict(x)
    if isinstance(prob, (list, tuple)):
        prob = prob[0]
    prob = np.asarray(prob)
    if prob.ndim == 4:
        prob_hw_c = prob[0]
    else:
        raise ValueError(f"Unexpected model output shape: {prob.shape}")

    labelmap = softmax_to_labelmap(prob_hw_c)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    prob_path = ""
    label_path = ""

    if save_prob_npy:
        prob_path = os.path.join(prob_dir, f"{stem}.npy")
        np.save(prob_path, prob_hw_c.astype(np.float32, copy=False))

    if save_label_png:
        
        label_path = os.path.join(lab_dir, f"{stem}.png")
        Image.fromarray(labelmap.astype(np.uint8, copy=False)).save(label_path)

    return {
        "ok": True,
        "image_path": image_path,
        "model_path": model_path,
        "out_dir": out_dir,
        "prob_path": prob_path,
        "label_path": label_path,
        "label_shape": list(labelmap.shape),
        "num_classes": int(prob_hw_c.shape[-1]),
    }
PALETTE = np.array([
    [0,   0,   0],      # 0  Background
    [255, 0,   0],      # 1  Layer 1  (Red)
    [0,   255, 0],      # 2  Layer 2  (Green)
    [0,   0,   255],    # 3  Layer 3  (Blue)
    [255, 255, 0],      # 4  Layer 4  (Yellow)
    [255, 0,   255],    # 5  Layer 5  (Magenta)
    [0,   255, 255],    # 6  Layer 6  (Cyan)
    [255, 128, 0],      # 7  Layer 7  (Orange)
    [128, 0,   255],    # 8  Layer 8  (Purple)
    [0,   128, 255],    # 9  Layer 9  (Sky Blue)
    [128, 255, 0],      # 10 Layer 10 (Lime)
    [255, 0,   128],    # 11 Layer 11 (Pink)
], dtype=np.uint8)

def overlay_oct_upper_boundaries(
    image_path: str,
    prob_path: str,
    n_classes: int = 11,
    label_colors: dict | None = None,
    alpha: float = 0.95,
    thickness: int = 2,
    out_path: str = "oct_upperlines_overlay.png",
) -> str:

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"image_path not found: {image_path}")

    if not os.path.exists(prob_path):
        raise FileNotFoundError(f"prob_path not found: {prob_path}")

 
    # Load image 
    
    img = Image.open(image_path).convert("RGB")
    img_rgb = np.asarray(img).astype(np.float32)

    vmin, vmax = np.percentile(img_rgb, [1, 99])
    img_rgb = np.clip((img_rgb - vmin) / (vmax - vmin + 1e-8), 0, 1)

    H_img, W_img = img_rgb.shape[:2]

    
    # Load probabilities/logits
  
    arr = np.load(prob_path)

    # Convert arr label mask at model resolution
    if arr.ndim == 2:
        # already labels
        mask_small = arr.astype(np.int32)

    elif arr.ndim == 3:
        # find channel axis by n_classes
        ch_axes = [ax for ax in range(3) if arr.shape[ax] == n_classes]
        if len(ch_axes) != 1:
            raise ValueError(
                f"Expected exactly one channel axis with size n_classes={n_classes}. "
                f"Got arr.shape={arr.shape} and channel_axes={ch_axes}"
            )
        ch_ax = ch_axes[0]

        # move channels to last (H,W,C)
        arr_hw_c = np.moveaxis(arr, ch_ax, -1)

        # argmax over channels (H,W)
        mask_small = np.argmax(arr_hw_c, axis=-1).astype(np.int32)

    else:
        raise ValueError(f"Unsupported prob array shape: {arr.shape}")

    if mask_small.ndim != 2:
        raise ValueError(f"mask_small must be 2D after conversion; got {mask_small.shape}")

    
    # Resize label mask to image size 
   
    if mask_small.shape != (H_img, W_img):
        mask_img = Image.fromarray(mask_small.astype(np.uint8), mode="L")
        mask_img = mask_img.resize((W_img, H_img), resample=Image.NEAREST)
        mask = np.asarray(mask_img).astype(np.int32)
    else:
        mask = mask_small

    H, W = mask.shape  # (H_img, W_img)

   
    labels = np.unique(mask)
    labels = labels[labels != 0]  # ignore background

    boundary_rgb = np.zeros((H, W, 3), dtype=np.float32)
    boundary_any = np.zeros((H, W), dtype=bool)

    
    for label_id in labels.tolist():
        if label_colors is not None and label_id in label_colors:
            color = np.array(label_colors[label_id], dtype=np.float32)
            if color.max() > 1.0:
                color = color / 255.0
        else:
            idx = int(np.clip(label_id, 0, PALETTE.shape[0] - 1))
            color = PALETTE[idx].astype(np.float32) / 255.0

        for x in range(W):
            ys = np.where(mask[:, x] == label_id)[0]
            if ys.size == 0:
                continue

            y0 = int(ys.min())  # upper-most pixel
            y_start = max(0, y0 - (thickness // 2))
            y_end = min(H, y_start + thickness)

            boundary_rgb[y_start:y_end, x, :] = color
            boundary_any[y_start:y_end, x] = True


    blended = img_rgb.copy()
    blended[boundary_any] = (1.0 - alpha) * img_rgb[boundary_any] + alpha * boundary_rgb[boundary_any]

    dir_name = os.path.dirname(out_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.imshow(blended, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()

    print(f"[overlay_oct_upper_boundaries] Saved to: {out_path}")
    return out_path