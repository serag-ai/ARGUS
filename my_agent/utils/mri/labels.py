import numpy as np

def remap_label_ids(vol: np.ndarray, id_map: dict[int, int], keep_unmapped_as_bg: bool = True) -> np.ndarray:
    vol = np.asarray(vol)
    if vol.dtype.kind == "f":
        vol = np.rint(vol).astype(np.int32)
    else:
        vol = vol.astype(np.int32, copy=False)

    out = np.zeros_like(vol, dtype=np.int32) if keep_unmapped_as_bg else vol.copy()
    for src, dst in id_map.items():
        out[vol == int(src)] = int(dst)
    return out

def validate_label_space(vol: np.ndarray, label_names: list[str], allow_extra: bool = False) -> dict:
    uniq = np.unique(vol.astype(np.int32, copy=False))
    mx = int(uniq.max()) if uniq.size else 0
    expected_max = len(label_names) - 1

    ok = True
    problems = []
    if mx > expected_max and not allow_extra:
        ok = False
        problems.append(f"max_label={mx} exceeds expected_max={expected_max} (len(label_names)={len(label_names)})")

    return {
        "ok": ok,
        "unique_labels": uniq.tolist(),
        "max_label": mx,
        "expected_max": expected_max,
        "problems": problems,
    }
