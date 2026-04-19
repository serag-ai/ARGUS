import numpy as np

def to_slice_first_xyz(vol_xyz: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "SAG":
        return vol_xyz
    if orientation == "COR":
        return vol_xyz.transpose((2, 0, 1))
    if orientation == "AXI":
        return vol_xyz.transpose((1, 2, 0))
    raise ValueError(f"Unknown orientation: {orientation}")

def from_slice_first_to_xyz(pred_shw: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "SAG":
        return pred_shw
    if orientation == "COR":
        return pred_shw.transpose((1, 2, 0))
    if orientation == "AXI":
        return pred_shw.transpose((2, 0, 1))
    raise ValueError(f"Unknown orientation: {orientation}")
