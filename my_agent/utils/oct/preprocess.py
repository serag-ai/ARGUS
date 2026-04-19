import numpy as np
from PIL import Image

def normalize_patch(img_arr: np.ndarray) -> np.ndarray:
    """
    Notebook logic:
      - convert to array
      - scale so max becomes 255 (if max>0)
      - return float32
    """
    x = img_arr.astype("float32", copy=False)
    max_val = float(np.max(x)) if x.size else 0.0
    if max_val > 0:
        x *= 255.0 / max_val
    return x.astype("float32", copy=False)

def load_and_preprocess_image(image_path: str, target_size=(384,384)) -> np.ndarray:
    """
    Returns a single image batch [1,H,W,3] float32.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_size[1], target_size[0]))  # PIL uses (W,H)
    arr = np.array(img)  # uint8
    arr = normalize_patch(arr)
    arr = np.expand_dims(arr, axis=0)  # [1,H,W,3]
    return arr