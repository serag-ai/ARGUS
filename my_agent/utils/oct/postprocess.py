import numpy as np
from PIL import Image

def softmax_to_labelmap(prob: np.ndarray) -> np.ndarray:
    """
    prob: [H,W,C] float
    returns: [H,W] uint8 label ids
    """
    return np.argmax(prob, axis=-1).astype(np.uint8)

def labelmap_to_overlay(labelmap: np.ndarray, base_rgb: np.ndarray | None = None, alpha: float = 0.35) -> Image.Image:
    """
    Simple overlay with a fixed palette (repeatable).
    base_rgb: [H,W,3] uint8 optional; if None, uses grayscale background.
    """
    H, W = labelmap.shape
    if base_rgb is None:
        base_rgb = np.stack([labelmap*0, labelmap*0, labelmap*0], axis=-1).astype(np.uint8)

    # deterministic palette
    rng = np.random.default_rng(123)
    palette = rng.integers(0, 255, size=(256, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)

    color = palette[labelmap]  # [H,W,3]
    out = (base_rgb.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha).clip(0,255).astype(np.uint8)
    return Image.fromarray(out)