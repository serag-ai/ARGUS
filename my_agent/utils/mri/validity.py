import numpy as np
from scipy import ndimage as ndi

def check_validity(
    vol,
    segmentation_path: str,
    min_voxels: int = 10,
    class_ids: list[int] | None = None,
    class_names: dict[int, str] | None = None,
    require_all_classes: bool = False,
    min_voxels_per_class: int = 10,
    connectivity: int = 26,
    max_components_per_class: int | None = 200,
    min_largest_cc_voxels_per_class: int = 10,
    min_largest_cc_fraction_per_class: float = 0.25,
) -> dict:
    if vol is None:
        return {"segmentation_valid": False, "reason": "Missing segmentation volume",
                "total_segmented_voxels": 0, "segmentation_path": segmentation_path,
                "per_class": None, "failed_classes": None}

    vol = np.asarray(vol)
    if vol.ndim != 3:
        return {"segmentation_valid": False, "reason": f"Expected 3D label map, got ndim={vol.ndim}",
                "total_segmented_voxels": int(np.count_nonzero(vol != 0)),
                "segmentation_path": segmentation_path, "per_class": None, "failed_classes": None}

    if np.issubdtype(vol.dtype, np.floating):
        vol = np.rint(vol).astype(np.int32)
    else:
        vol = vol.astype(np.int32, copy=False)

    total_voxels = int(np.count_nonzero(vol != 0))
    if total_voxels < min_voxels:
        return {"segmentation_valid": False, "reason": "Segmentation is fully black or too small",
                "total_segmented_voxels": total_voxels, "segmentation_path": segmentation_path,
                "per_class": {}, "failed_classes": []}

    if class_ids is None:
        uniq = np.unique(vol)
        class_ids = [int(c) for c in uniq if int(c) != 0]

    if connectivity not in (6, 18, 26):
        return {"segmentation_valid": False, "reason": f"Invalid connectivity={connectivity}; must be 6, 18, or 26",
                "total_segmented_voxels": total_voxels, "segmentation_path": segmentation_path,
                "per_class": None, "failed_classes": None}

    conn_map = {6: 1, 18: 2, 26: 3}
    structure = ndi.generate_binary_structure(rank=3, connectivity=conn_map[connectivity])

    per_class = {}
    failed = []

    for cid in class_ids:
        name = class_names.get(cid, str(cid)) if class_names else str(cid)
        class_mask = (vol == cid)
        class_vox = int(np.count_nonzero(class_mask))

        if class_vox < min_voxels_per_class:
            per_class[name] = {"valid": False, "reason": "Class missing or too small",
                               "voxels": class_vox, "n_components": 0,
                               "largest_cc_voxels": 0, "largest_cc_fraction": 0.0}
            if require_all_classes or class_vox > 0:
                failed.append(name)
            continue

        labeled, n_cc = ndi.label(class_mask, structure=structure)
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        largest_cc = int(sizes.max()) if sizes.size else 0
        largest_frac = float(largest_cc) / float(class_vox) if class_vox > 0 else 0.0

        reasons = []
        if max_components_per_class is not None and int(n_cc) > max_components_per_class:
            reasons.append(f"Too many components (n={int(n_cc)})")
        if largest_cc < min_largest_cc_voxels_per_class:
            reasons.append(f"Largest component too small (largest={largest_cc})")
        if largest_frac < min_largest_cc_fraction_per_class:
            reasons.append(f"Fragmented (largest_frac={largest_frac:.3f})")

        ok = (len(reasons) == 0)
        per_class[name] = {"valid": ok, "reason": None if ok else "; ".join(reasons),
                           "voxels": class_vox, "n_components": int(n_cc),
                           "largest_cc_voxels": largest_cc, "largest_cc_fraction": largest_frac}

        if not ok:
            failed.append(name)

    overall_ok = (len(failed) == 0) if not require_all_classes else (len(failed) == 0)
    return {"segmentation_valid": bool(overall_ok),
            "reason": None if overall_ok else f"{len(failed)} class(es) failed validity checks",
            "total_segmented_voxels": total_voxels,
            "segmentation_path": segmentation_path,
            "per_class": per_class,
            "failed_classes": failed}
