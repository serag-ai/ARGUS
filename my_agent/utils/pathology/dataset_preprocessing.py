import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms, models


def scale_coords(coords, orig_size, target_size):
    """
    Scale coordinates (x,y) from original size to target size.
    Args:
        coords: list of (x,y)
        orig_size: (orig_w, orig_h)
        target_size: (new_w, new_h)
    """
    orig_w, orig_h = orig_size
    new_w, new_h = target_size
    scaled = []
    for (x,y) in coords:
        sx = int(x * new_w / orig_w)
        sy = int(y * new_h / orig_h)
        scaled.append((sx, sy))
    return scaled

def generate_gaussian_heatmap(coords, img_shape=(224,224), sigma=3):  
    """
    Generate a heatmap from centroid coordinates.
    Args:
        coords: list of (x,y) tuples
        img_shape: (H, W)
        sigma: radius for Gaussian (circle size)
    Returns:
        heatmap: numpy array [H, W]
    """
    heatmap = np.zeros(img_shape, dtype=np.float32)
    for (x, y) in coords:
        if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
            cv2.circle(heatmap, (x, y), sigma, 1, -1)  
    return heatmap



def _is_image_file(p: str) -> bool:
    return str(p).lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


class CentroidHeatmapDataset(Dataset):
    """
    Single-image dataset:
      - Takes ONE image path (not a folder)
      - Loads matching .xyc file from xyc_folder using the image basename
      - Returns (img_tensor, heatmap_tensor, filename)

    Assumes you already have:
      - scale_coords(coords, (orig_w, orig_h), (new_w, new_h))
      - generate_gaussian_heatmap(coords, img_shape=(H,W), sigma=int)
    """
    def __init__(self, image_path: str, xyc_folder: str, img_size=(224, 224), sigma=3, transform=None):
        """
        Args:
            image_path (str): path to a single image (.png/.jpg/.jpeg)
            xyc_folder (str): folder containing .xyc centroid files
            img_size (tuple): resize target (H, W)
            sigma (int): gaussian blob sigma
            transform: optional torchvision transform applied to PIL image (after resize)
        """
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError("image_path must be a non-empty string.")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not image_path.lower().endswith((".png", ".jpg", ".jpeg")):
            raise ValueError(f"Unsupported image extension: {image_path}")

        if not isinstance(xyc_folder, str) or not xyc_folder.strip():
            raise ValueError("xyc_folder must be a non-empty string.")
        if not os.path.isdir(xyc_folder):
            raise FileNotFoundError(f"xyc_folder not found: {xyc_folder}")

        self.image_path = image_path
        self.xyc_folder = xyc_folder
        self.img_size = tuple(img_size)  # (H, W)
        self.sigma = int(sigma)

        self.fname = os.path.basename(image_path)
        self.stem = os.path.splitext(self.fname)[0]

        self.resize = transforms.Resize(self.img_size)  # expects (H, W)
        self.to_tensor = transforms.ToTensor()

        self.transform = transform

    def __len__(self):
        return 1

    def _load_xyc(self, xyc_path: str):
        coords = []
        if not os.path.exists(xyc_path):
            return coords

        with open(xyc_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                
                if len(parts) >= 2:
                    try:
                        x, y = int(parts[0]), int(parts[1])
                        coords.append((x, y))
                    except ValueError:
                        continue
        return coords

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("This dataset contains exactly one item (idx must be 0).")

        # Load image
        img_pil = Image.open(self.image_path).convert("RGB")
        orig_w, orig_h = img_pil.size

        img_pil = self.resize(img_pil)
        if self.transform is not None:
            img = self.transform(img_pil)
            # enforce tensor output
            if not torch.is_tensor(img):
                raise TypeError("transform must return a torch.Tensor.")
        else:
            img = self.to_tensor(img_pil)

       
        xyc_path = os.path.join(self.xyc_folder, self.stem + ".xyc")
        coords = self._load_xyc(xyc_path)

        new_h, new_w = self.img_size[0], self.img_size[1]
        coords = scale_coords(coords, (orig_w, orig_h), (new_w, new_h))

        heatmap_np = generate_gaussian_heatmap(coords, img_shape=self.img_size, sigma=self.sigma)
        heatmap = torch.as_tensor(heatmap_np, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        return img, heatmap, self.fname

    
    
class PathologyInferenceDataset(Dataset):

    def __init__(self, input_path: str, img_size=(224, 224), transform=None):
        self.input_path = str(input_path)
        self.img_size = tuple(img_size)  # (H, W)
        self.transform = transform

        # same style as CentroidHeatmapDataset
        self.resize = transforms.Resize(self.img_size)   # (H, W)
        self.to_tensor = transforms.ToTensor()

        if os.path.isdir(self.input_path):
            self.image_paths = sorted([
                os.path.join(self.input_path, f)
                for f in os.listdir(self.input_path)
                if _is_image_file(f)
            ])
        else:
            if not os.path.exists(self.input_path):
                raise FileNotFoundError(f"Input path does not exist: {self.input_path}")
            if not _is_image_file(self.input_path):
                raise ValueError(f"Input path is not a supported image file: {self.input_path}")
            self.image_paths = [self.input_path]

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in: {self.input_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        fname = os.path.basename(img_path)

        # Load image
        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size

        # Resize
        img_pil = self.resize(img_pil)

        # Transform/ToTensor
        if self.transform is not None:
            img_t = self.transform(img_pil)
            if not torch.is_tensor(img_t):
                raise TypeError("transform must return a torch.Tensor.")
        else:
            img_t = self.to_tensor(img_pil)  # (3,H,W), float32 [0,1]

        return img_t, fname, orig_w, orig_h