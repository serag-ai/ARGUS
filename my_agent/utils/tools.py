import os
import ast
import numpy as np
import torch 
import nibabel as nib 
from PIL import Image
from scipy.ndimage import label as cc_label
from langchain_core.tools import tool
import json
from langchain_tavily import TavilySearch
from typing import Dict, Any, Tuple,Optional, List


#MRI
from my_agent.utils.mri import run_quicknat_with_cache
from my_agent.utils.mri.cache import get_cache, segment_cache_key
from my_agent.utils.mri.constants import DEFAULT_LABEL_NAMES
from my_agent.utils.mri.volumes import compute_voxel_counts, voxel_mm3_from_header
from my_agent.utils.mri.validity import check_validity
#Pathology
from my_agent.utils.pathology.inference import evaluate_model 
from my_agent.utils.pathology.dataset_preprocessing import PathologyInferenceDataset
from my_agent.utils.pathology.inference import extract_and_compare_centroids_from_paths
#OCT
from my_agent.utils.oct.inference import run_oct_segmentation

#pdf
from my_agent.utils.mri.pdf_report import create_mri_pdf_report, create_oct_pdf_report, create_pathology_pdf_report

#Helpers
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _is_npy(p: str) -> bool:
    return isinstance(p, str) and p.lower().endswith(".npy")



# MRI TOOLS 

@tool(return_direct=True)
def segment_mri_quicknat(image_path: str, model_path: str, orientation: str, device: str = "cpu", batch_size: int = 20) -> dict:
    """
    Run QuickNAT segmentation and save outputs to disk.
    """
    seg = run_quicknat_with_cache(image_path, model_path, orientation, device, batch_size)

    if not seg.get("ok", False):
        return {"ok": False, "reason": seg.get("reason", "Unknown segmentation failure"),
                "model_path": model_path, "orientation": orientation,
                "total_segmented_voxels": 0.0,
                "segmentation_path": "", "label_mask_paths": {}}

    total_voxels = float(np.count_nonzero(seg["vol_pred"]))
    if total_voxels < 10:
        return {"ok": False, "reason": f"The Segmentation is empty, Total segmentation pixels found: {total_voxels}",
                "model_path": seg["model_path"], "orientation": seg["orientation"],
                "total_segmented_voxels": total_voxels,
                "segmentation_path": seg["segmentation_path"],
                "label_mask_paths": seg.get("label_mask_paths", {})}

    return {"ok": True, "reason": None,
            "model_path": seg["model_path"], "orientation": seg["orientation"],
            "total_segmented_voxels": total_voxels,
            "segmentation_path": seg["segmentation_path"],
            "label_mask_paths": seg.get("label_mask_paths", {})}


@tool(return_direct=True)
def compute_volumes_from_segmentation(
    image_path: str,
    model_path: str,
    brain_region: str | None = None,
    orientation: str = "COR",
    device: str = "cpu",
    batch_size: int = 20,
    run_if_missing_cache: bool = True,
) -> dict:
    """
    Compute per-region voxel counts and physical volumes (mm³) from a cached segmentation.
    """
    cache = get_cache()
    key = segment_cache_key(image_path, model_path, orientation, device, batch_size)

    if key in cache:
        seg = cache[key]
    else:
        if not run_if_missing_cache:
            return {"ok": False, "has_volumes": False, "reason": "No cached segmentation found", "segmentation_path": ""}
        seg = run_quicknat_with_cache(image_path, model_path, orientation, device, batch_size)

    if not seg.get("ok", True):
        return {"ok": False, "has_volumes": False,
                "reason": f"Segmentation not available: {seg.get('reason', 'unknown')}",
                "segmentation_path": seg.get("segmentation_path", "")}

    vol_pred = seg.get("vol_pred")
    header = seg.get("header")
    if vol_pred is None or header is None:
        return {"ok": False, "has_volumes": False, "reason": "Missing vol_pred or header", "segmentation_path": seg.get("segmentation_path", "")}

    counts = compute_voxel_counts(vol_pred, DEFAULT_LABEL_NAMES)
    voxel_mm3 = voxel_mm3_from_header(header)
    vols_mm3 = {name: float(cnt) * voxel_mm3 for name, cnt in counts.items()}

    out = {
        "ok": True,
        "has_volumes": True,
        "reason": None,
        "model_path": seg.get("model_path", model_path),
        "orientation": seg.get("orientation", orientation),
        "voxel_volume_mm3": float(voxel_mm3),
        "voxels_per_label": counts,
        "volumes_mm3_per_label": vols_mm3,
        "brain_region": brain_region,
        "total_segmented_voxels": float(sum(counts.values())),
        "segmentation_path": seg.get("segmentation_path", ""),
        "label_mask_paths": seg.get("label_mask_paths", {}),
    }

    if brain_region:
        if brain_region in counts:
            out["brain_region_voxels"] = float(counts[brain_region])
            out["brain_region_volume_mm3"] = float(vols_mm3[brain_region])
        else:
            out["ok"] = False
            out["has_volumes"] = False
            out["reason"] = f"Region '{brain_region}' not found. Available: {list(counts.keys())}"

    return out



@tool(return_direct=True)
def run_quicknat_segmentation_and_volume(
    image_path: str,
    model_path: str,
    brain_region: str | None = None,
    orientation: str = "COR",
    device: str = "cpu",
    batch_size: int = 20
) -> dict:
    """
    Convenience tool: compute volumes (mm³) for all classes, with region.
    """
    _ = segment_mri_quicknat(image_path, model_path, orientation, device, batch_size)
    return compute_volumes_from_segmentation(image_path, model_path, brain_region, orientation, device, batch_size, run_if_missing_cache=True)


@tool(return_direct=True)
def generate_mri_report(patient_id: str, brain_region: str, volume_mm3: float, model_used: str, fallback_used: bool) -> str:
    """
    Generate a minimal narrative line for a single region volume.
    """
    return (f"Patient {patient_id}: volume of {brain_region} is {volume_mm3:.2f} mm³. "
            f"Model: {model_used}, Fallback: {fallback_used}.")




def _safe_get(d: dict, key: str, default=None):
    try:
        return d.get(key, default)
    except Exception:
        return default

@tool
def get_case_data_from_cache(
    cache_key: str,
    requested_fields: Optional[List[str]] = None,
) -> dict:
    """
    Returns structured case data.
    """
    # cache = get_cache()
    # payload = cache.get(cache_key) or {}
    payload = {}

    if not isinstance(payload, dict):
        payload = {}

    req = [str(x).strip() for x in (requested_fields or []) if str(x).strip()]
    if not req:
        return {"ok": True, "cache_key": cache_key, "data": payload}

    
    def flatten(d: Dict[str, Any], parent="") -> Dict[str, Any]:
        out = {}
        for k, v in (d or {}).items():
            kk = f"{parent}.{k}" if parent else str(k)
            if isinstance(v, dict):
                out.update(flatten(v, kk))
            else:
                out[kk] = v
        return out

    flat = flatten(payload)
    selected: Dict[str, Any] = {}
    missing: List[str] = []

    for f in req:
        if f in payload:
            selected[f] = payload[f]
        elif f in flat:
            selected[f] = flat[f]
        else:
            # case-insensitive top-level match
            match = None
            for k in payload.keys():
                if str(k).lower() == f.lower():
                    match = k
                    break
            if match is not None:
                selected[str(match)] = payload.get(match)
            else:
                missing.append(f)

    return {
        "ok": True,
        "cache_key": cache_key,
        "requested_fields": req,
        "missing_fields": missing,
        "data": selected,
    }

@tool
def merge_case_payloads(
    patient_id: str,
    modality: str,
    model_used: str,
    fallback_used: bool,
    payloads: List[Dict[str, Any]],
) -> dict:
    """
    Merge multiple tool outputs.
    """
    merged: Dict[str, Any] = {
        "patient_id": patient_id,
        "modality": modality,
        "model_used": model_used,
        "fallback_used": fallback_used,
        "data": {},
    }

    merged_list = []
    for p in payloads or []:
        if isinstance(p, dict):
            merged_list.append(p)
    merged["data"]["payloads"] = merged_list

    return {"ok": True, "merged": merged}


@tool(return_direct=True)
def tavily_web_search(query: str, max_results: int = 3) -> dict:
    """
    Run a Tavily web search for external references.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"ok": False, "error": "TAVILY_API_KEY not set.", "query": query, "results": [], "results_count": 0}

    try:
        searcher = TavilySearch(max_results=max_results)
        resp = searcher.invoke(query)
        results = resp.get("results", []) if isinstance(resp, dict) else []
        cleaned = [{"title": r.get("title"), "url": r.get("url"), "content": r.get("content")}
                   for r in results if isinstance(r, dict)]
        return {"ok": True, "error": None, "query": query, "results": cleaned, "results_count": len(cleaned)}
    except Exception as e:
        return {"ok": False, "error": str(e), "query": query, "results": [], "results_count": 0}


### Pathology ###
def _stable_cache_key(d: dict) -> str:
    """
    Stable cache key string: JSON with sorted keys, compact separators.
    """
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@tool
def pathology_segment(
    images_folder: str,
    model_path: str,
    out_root: str,
    device: str = "cpu",
    batch_size: int = 8,
    threshold: float = 0.5,
) -> dict:
    """
    Segment Pathology and return back segmentation and proababilty map
    
    """

    test_dataset = PathologyInferenceDataset(input_path=images_folder, img_size=(224, 224))

    heatmap_paths, mask_paths, test_filenames = evaluate_model(
        model_path=model_path,
        test_dataset=test_dataset,
        images_folder=images_folder,
        out_root=out_root,
        threshold=threshold,
        batch_size=batch_size,
        device=device,
        img_size=(224, 224),
    )

    # run_dir derivation
    run_dir = None
    if heatmap_paths:
        hp0 = str(heatmap_paths[0])
        if "/heatmaps/" in hp0:
            run_dir = hp0.split("/heatmaps/")[0]

   
    key_dict = {
        "images_folder": images_folder,
        "model_path": model_path,
        "heatmap_path":heatmap_paths,
        "mask_paths":mask_paths,
        "device": device,
        "batch_size": int(batch_size),
        "threshold": float(threshold),
        "img_size": [224, 224],
    }
    cache_key = _stable_cache_key(key_dict)

  
    out = {
        "ok": True,
        "artifacts": {
            "run_dir": run_dir,
            "heatmaps_dir": (os.path.join(run_dir, "heatmaps") if run_dir else None),
            "masks_dir": (os.path.join(run_dir, "masks") if run_dir else None),
            "test_filenames": [str(x) for x in (test_filenames or [])],
            "heatmap_paths": [str(x) for x in (heatmap_paths or [])],
            "mask_paths": [str(x) for x in (mask_paths or [])],
            "cache_key": cache_key,
            "key_dict": key_dict,
        },
    }

    cache = get_cache()
    cache[cache_key] = out

    return out

@tool
def pathology_count_cells(
    cache_key: str,
) -> dict:
    """
    Count cells from cached pathology segmentation artifacts.
    Returns JSON.
    """
    cache = get_cache()
    entry = cache.get(cache_key)

    if not entry:
        return {"ok": False, "reason": "cache_key not found", "cache_key": str(cache_key)}

    artifacts = entry.get("artifacts", {})

    heatmap_paths = artifacts.get("heatmap_paths", []) or []
    test_filenames = artifacts.get("test_filenames", []) or []

    if not heatmap_paths or not test_filenames:
        return {
            "ok": False,
            "reason": "Cached entry missing heatmap_paths/test_filenames",
            "cache_key": str(cache_key),
            "available_keys": sorted(list(artifacts.keys())) if isinstance(artifacts, dict) else [],
        }

    out = extract_and_compare_centroids_from_paths(
        heatmap_paths=heatmap_paths,
        test_filenames=test_filenames,
        threshold=0.6,
        min_distance=10,
    )

    results = out.get("results", []) or []
    total_cells = int(sum(int(r.get("pred_count", 0)) for r in results))

    return {
        "ok": True,
        "cache_key": str(cache_key),
        "num_cells_total": total_cells,
        "num_images": int(len(results)),
        "results": results,
    }

##### OCT ######
@tool
def oct_segment(
    image_path: str,
    model_path: str,
    out_dir: str,
) -> dict:
    """
    OCT segmentation tool (pretrained .keras only).
    Runs OCT U-Net and saves outputs to out_dir.
    """
    return run_oct_segmentation(
        image_path=image_path,
        model_path=model_path,
        out_dir=out_dir,
        save_prob_npy=True,
        save_label_png=True,
    )



AXIAL_UM_PER_PX = 8.0   # FIXED (from 7.94 mm / 992 px) based on the paper of the dataset


OCT_CLASSES = [
    "background", "RNFL", "GCL", "IPL", "INL", "OPL",
    "ONL", "ISOS", "RPE", "CHOROID", "nerves-vessels-OPTICDISC"
]

NAME2ID = {n: i for i, n in enumerate(OCT_CLASSES)}


def summarize_vector(vec: np.ndarray) -> dict:
    valid = np.isfinite(vec)
    n_valid = int(valid.sum())

    if n_valid == 0:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
        }

    vals = vec[valid]
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "std": float(np.std(vals)),
    }


def summarize_sector(vec: np.ndarray, idx: np.ndarray) -> dict:
    if idx.size == 0:
        return summarize_vector(np.array([]))
    return summarize_vector(vec[idx])


def build_sector_summaries(vec: np.ndarray, width: int) -> dict:
    thirds = np.array_split(np.arange(width), 3)
    return {
        "left": summarize_sector(vec, thirds[0]),
        "center": summarize_sector(vec, thirds[1]),
        "right": summarize_sector(vec, thirds[2]),
    }


@tool
def compute_oct_retinal_thickness_from_prob(prob_path: str) -> dict:
    """
    Compute OCT retinal thickness in MICROMETERS (µm) only.
    Scale is fixed using dataset calibration.
    """

    if not os.path.exists(prob_path):
        return {"ok": False, "reason": "prob_path not found"}

    try:
        prob = np.load(prob_path).astype(np.float32)
    except Exception as e:
        return {"ok": False, "reason": str(e)}

    if prob.ndim != 3 or prob.shape[-1] != len(OCT_CLASSES):
        return {
            "ok": False,
            "reason": f"Expected shape (H,W,{len(OCT_CLASSES)})"
        }

    label_map = np.argmax(prob, axis=-1).astype(np.int16)
    H, W = label_map.shape

    # Layers
    layer_names = ["RNFL", "GCL", "IPL", "INL", "OPL", "ONL", "ISOS", "RPE"]
    layer_ids = {name: NAME2ID[name] for name in layer_names}

    retina_ids = np.array(list(layer_ids.values()))
    rpe_id = layer_ids["RPE"]

    # Initialize in µm directly
    full_retinal_thickness_um = np.full(W, np.nan, dtype=np.float32)
    per_layer_thickness_um = {
        name: np.full(W, np.nan, dtype=np.float32) for name in layer_names
    }

    rpe_present = np.zeros(W, dtype=bool)

    for x in range(W):
        col = label_map[:, x]

        # Full retina thickness
        retina_rows = np.flatnonzero(np.isin(col, retina_ids))
        if retina_rows.size > 0:
            y_top = int(retina_rows.min())

            rpe_rows = np.flatnonzero(col == rpe_id)
            if rpe_rows.size > 0:
                rpe_present[x] = True
                y_bottom = int(rpe_rows.max())

                if y_bottom >= y_top:
                    thickness_px = (y_bottom - y_top + 1)
                    full_retinal_thickness_um[x] = thickness_px * AXIAL_UM_PER_PX

        # Per-layer thickness
        for name, lid in layer_ids.items():
            rows = np.flatnonzero(col == lid)
            if rows.size == 0:
                continue

            y_top = int(rows.min())
            y_bottom = int(rows.max())

            thickness_px = (y_bottom - y_top + 1)
            per_layer_thickness_um[name][x] = thickness_px * AXIAL_UM_PER_PX

    # Summaries
    full_summary = summarize_vector(full_retinal_thickness_um)

    layer_summaries = {
        name: summarize_vector(per_layer_thickness_um[name])
        for name in layer_names
    }

    sector_summaries = {
        "full_retinal_thickness": build_sector_summaries(full_retinal_thickness_um, W)
    }

    for name in layer_names:
        sector_summaries[name] = build_sector_summaries(
            per_layer_thickness_um[name], W
        )

    rpe_continuity = {
        "presence_ratio": float(rpe_present.mean()) if W > 0 else 0.0,
        "missing_ratio": float((~rpe_present).mean()) if W > 0 else 0.0,
    }

    return {
        "ok": True,
        "unit": "micrometers",
        "axial_um_per_px": AXIAL_UM_PER_PX,

        "full_retinal_thickness": full_summary,
        "layers": layer_summaries,
        "sector_summaries": sector_summaries,
        "rpe_continuity": rpe_continuity,
    }

##### Validation tools

CLASS_CONFIG = {
    "oct": {
        "class_names": OCT_CLASSES,
        "background_class": 0,
        "default_min_component_size": 20,
    },
    "mri": {
        "class_names": DEFAULT_LABEL_NAMES,
        "background_class": 0,
        "default_min_component_size": 20,
    },
}


def _strip_known_ext(filename: str) -> str:
    filename_lower = filename.lower()

    if filename_lower.endswith(".nii.gz"):
        return filename[:-7]
    if filename_lower.endswith(".nii"):
        return filename[:-4]
    if filename_lower.endswith(".npz"):
        return filename[:-4]
    if filename_lower.endswith(".npy"):
        return filename[:-4]
    if filename_lower.endswith(".png"):
        return filename[:-4]

    return os.path.splitext(filename)[0]


def _resolve_mri_seg_path(seg_path: str) -> str:
    """
    Accepts:
      - direct segmentation path
      - original MRI image path

    Returns the first existing candidate if found.
    """
    if not seg_path:
        return seg_path

    seg_path = os.path.abspath(seg_path.strip())

    if os.path.exists(seg_path):
        return seg_path

    base_dir = os.path.dirname(seg_path)
    fname = os.path.basename(seg_path)
    stem = _strip_known_ext(fname)

    candidate_paths = [
        os.path.join(base_dir, "segmentations", fname),
    ]

    if not stem.endswith(("_seg_COR", "_seg_AXI", "_seg_SAG")):
        candidate_paths.extend([
            os.path.join(base_dir, "segmentations", f"{stem}_seg_COR.nii.gz"),
            os.path.join(base_dir, "segmentations", f"{stem}_seg_AXI.nii.gz"),
            os.path.join(base_dir, "segmentations", f"{stem}_seg_SAG.nii.gz"),
            os.path.join(base_dir, "segmentations", f"{stem}_seg_COR.nii"),
            os.path.join(base_dir, "segmentations", f"{stem}_seg_AXI.nii"),
            os.path.join(base_dir, "segmentations", f"{stem}_seg_SAG.nii"),
        ])

    parent = os.path.dirname(base_dir)
    if os.path.basename(base_dir).lower() == "segmentations":
        candidate_paths.extend([
            os.path.join(base_dir, f"{stem}.nii.gz"),
            os.path.join(base_dir, f"{stem}.nii"),
        ])
    else:
        candidate_paths.extend([
            os.path.join(parent, "segmentations", f"{stem}.nii.gz"),
            os.path.join(parent, "segmentations", f"{stem}.nii"),
        ])

    seen = set()
    unique_candidates = []
    for p in candidate_paths:
        if p not in seen:
            unique_candidates.append(p)
            seen.add(p)

    for cand in unique_candidates:
        if os.path.exists(cand):
            return cand

    return seg_path


def _resolve_oct_seg_path(seg_path: str) -> str:
    """
    OCT can use:
      - .png label masks
      - .npy masks or probability maps
      - .npz masks or probability maps
    """
    if not seg_path:
        return seg_path

    seg_path = os.path.abspath(seg_path.strip())

    if os.path.exists(seg_path):
        return seg_path

    return seg_path


def _resolve_seg_path(seg_path: str, modality: str) -> str:
    modality = modality.lower().strip()

    if modality == "mri":
        return _resolve_mri_seg_path(seg_path)
    if modality == "oct":
        return _resolve_oct_seg_path(seg_path)

    return os.path.abspath(seg_path.strip()) if seg_path else seg_path


def load_segmentation(seg_path: str) -> np.ndarray:
    """
    Supported:
      OCT:
        - .png
        - .npy
        - .npz
      MRI:
        - .nii
        - .nii.gz
    """
    seg_path_lower = seg_path.lower()

    if seg_path_lower.endswith(".png"):
        img = Image.open(seg_path)
        return np.array(img)

    if seg_path_lower.endswith(".npy"):
        try:
            return np.load(seg_path, allow_pickle=False)
        except ValueError as e:
            if "allow_pickle=False" in str(e):
                return np.load(seg_path, allow_pickle=True)
            raise

    if seg_path_lower.endswith(".npz"):
        data = np.load(seg_path, allow_pickle=False)
        if len(data.files) == 0:
            raise ValueError(f"No arrays found inside npz file: {seg_path}")
        return data[data.files[0]]

    if seg_path_lower.endswith(".nii") or seg_path_lower.endswith(".nii.gz"):
        return nib.load(seg_path).get_fdata()

    raise ValueError(f"Unsupported segmentation file format: {seg_path}")


def _remap_oct_mask_if_needed(mask: np.ndarray) -> np.ndarray:
    """
    Remap grayscale-coded OCT masks like [0, 26, 51, ...] into class ids [0..N-1]
    only when needed.
    """
    mask = np.asarray(mask)

    if mask.ndim != 2:
        return mask.astype(np.int32)

    unique_vals = np.unique(mask)
    num_classes = len(OCT_CLASSES)

    if unique_vals.size == 0:
        return mask.astype(np.int32)

    if unique_vals.min() >= 0 and unique_vals.max() <= num_classes - 1:
        return mask.astype(np.int32)

    if unique_vals.size <= num_classes:
        sorted_vals = sorted(unique_vals.tolist())
        mapping = {val: idx for idx, val in enumerate(sorted_vals)}
        remapped = np.vectorize(mapping.get)(mask)
        return remapped.astype(np.int32)

    return mask.astype(np.int32)


def _convert_to_label_mask(arr: np.ndarray, modality: str) -> np.ndarray:
    """
    Convert input array into integer label mask.
    """
    modality = modality.lower().strip()
    arr = np.asarray(arr)

    if modality == "oct":
        if arr.ndim == 2:
            return _remap_oct_mask_if_needed(arr)

        if arr.ndim == 3:
            num_classes = len(OCT_CLASSES)

            if arr.shape[-1] == num_classes:
                return np.argmax(arr, axis=-1).astype(np.int32)

            if arr.shape[0] == num_classes:
                return np.argmax(arr, axis=0).astype(np.int32)

            if arr.shape[-1] in (3, 4):
                rgb = arr[..., :3]
                if np.all(rgb[..., 0] == rgb[..., 1]) and np.all(rgb[..., 1] == rgb[..., 2]):
                    return _remap_oct_mask_if_needed(rgb[..., 0])

            possible_axes = [i for i, s in enumerate(arr.shape) if s == num_classes]
            if possible_axes:
                return np.argmax(arr, axis=possible_axes[0]).astype(np.int32)

            raise ValueError(f"Unsupported OCT array shape: {arr.shape}")

        if arr.ndim == 4:
            num_classes = len(OCT_CLASSES)
            possible_axes = [i for i, s in enumerate(arr.shape) if s == num_classes]
            if possible_axes:
                return np.argmax(arr, axis=possible_axes[0]).astype(np.int32)

            smallest_axis = int(np.argmin(arr.shape))
            return np.argmax(arr, axis=smallest_axis).astype(np.int32)

        raise ValueError(f"Unsupported OCT array shape: {arr.shape}")

    if modality == "mri":
        if arr.ndim == 3:
            return np.rint(arr).astype(np.int32)

        if arr.ndim == 4:
            num_classes = len(DEFAULT_LABEL_NAMES)
            possible_axes = [i for i, s in enumerate(arr.shape) if s == num_classes]

            if possible_axes:
                return np.argmax(arr, axis=possible_axes[0]).astype(np.int32)

            smallest_axis = int(np.argmin(arr.shape))
            return np.argmax(arr, axis=smallest_axis).astype(np.int32)

        raise ValueError(f"Unsupported MRI array shape: {arr.shape}")

    raise ValueError(f"Unsupported modality: {modality}")


def _id_to_name(cid: int, class_names: List[str]) -> str:
    if 0 <= cid < len(class_names):
        return class_names[cid]
    return f"UnknownClass_{cid}"


def _check_class_has_valid_component(
    mask: np.ndarray,
    class_id: int,
    min_component_size: int,
) -> Dict[str, Any]:
    """
    A class is valid if:
      - it exists in the mask
      - it has at least one connected component >= min_component_size
    """
    binary = (mask == class_id)

    if not np.any(binary):
        return {
            "present": False,
            "valid_component": False,
            "reason": "class_not_present",
        }

    labeled, num = cc_label(binary)

    if num == 0:
        return {
            "present": False,
            "valid_component": False,
            "reason": "class_not_present",
        }

    component_sizes = np.bincount(labeled.ravel())
    if component_sizes.size <= 1:
        return {
            "present": True,
            "valid_component": False,
            "reason": "no_foreground_component",
        }

    component_sizes[0] = 0
    largest_component = int(component_sizes.max())

    if largest_component < int(min_component_size):
        return {
            "present": True,
            "valid_component": False,
            "reason": f"largest_component_too_small(<{min_component_size})",
        }

    return {
        "present": True,
        "valid_component": True,
        "reason": "ok",
    }


@tool
def check_segmentation_validity(
    seg_path: str,
    modality: str,
    required_class_ids: Optional[list[int]] = None,
    min_component_size: Optional[int] = None,
) -> dict:
    """
    Validate OCT or MRI segmentation.

    Main rules:
      1. file must exist
      2. segmentation must load successfully
      3. segmentation must not be empty / all background
      4. connectivity check runs
      5. if required_class_ids is given, validate those classes
         otherwise validate all present foreground classes
      6. if any checked class is invalid then STOP
    """
    modality = modality.lower().strip()

    if modality not in CLASS_CONFIG:
        return {
            "valid": False,
            "reason": f"Unsupported modality: {modality}",
            "modality": modality,
            "seg_path_received": seg_path,
            "seg_path_used": None,
            "present_class_ids": [],
            "present_class_names": [],
            "checked_class_ids": [],
            "checked_class_names": [],
            "missing_class_ids": [],
            "missing_class_names": [],
            "invalid_class_ids": [],
            "invalid_class_names": [],
            "class_validation_details": {},
        }

    class_names = CLASS_CONFIG[modality]["class_names"]
    background_class = CLASS_CONFIG[modality]["background_class"]

    if min_component_size is None:
        min_component_size = CLASS_CONFIG[modality]["default_min_component_size"]

    seg_path_received = seg_path
    seg_path = _resolve_seg_path(seg_path, modality)

    if not seg_path or not os.path.exists(seg_path):
        return {
            "valid": False,
            "reason": f"seg_path not found: {seg_path_received}",
            "modality": modality,
            "seg_path_received": seg_path_received,
            "seg_path_used": seg_path,
            "present_class_ids": [],
            "present_class_names": [],
            "checked_class_ids": [],
            "checked_class_names": [],
            "missing_class_ids": [],
            "missing_class_names": [],
            "invalid_class_ids": [],
            "invalid_class_names": [],
            "class_validation_details": {},
        }

    try:
        arr = load_segmentation(seg_path)
    except Exception as e:
        return {
            "valid": False,
            "reason": f"Failed to load segmentation: {type(e).__name__}: {e}",
            "modality": modality,
            "seg_path_received": seg_path_received,
            "seg_path_used": seg_path,
            "present_class_ids": [],
            "present_class_names": [],
            "checked_class_ids": [],
            "checked_class_names": [],
            "missing_class_ids": [],
            "missing_class_names": [],
            "invalid_class_ids": [],
            "invalid_class_names": [],
            "class_validation_details": {},
        }

    try:
        mask = _convert_to_label_mask(arr, modality=modality)
    except Exception as e:
        return {
            "valid": False,
            "reason": f"Failed to convert segmentation to label mask: {type(e).__name__}: {e}",
            "modality": modality,
            "seg_path_received": seg_path_received,
            "seg_path_used": seg_path,
            "present_class_ids": [],
            "present_class_names": [],
            "checked_class_ids": [],
            "checked_class_names": [],
            "missing_class_ids": [],
            "missing_class_names": [],
            "invalid_class_ids": [],
            "invalid_class_names": [],
            "class_validation_details": {},
        }

    if mask.size == 0:
        return {
            "valid": False,
            "reason": "Segmentation array is empty.",
            "modality": modality,
            "seg_path_received": seg_path_received,
            "seg_path_used": seg_path,
            "present_class_ids": [],
            "present_class_names": [],
            "checked_class_ids": [],
            "checked_class_names": [],
            "missing_class_ids": [],
            "missing_class_names": [],
            "invalid_class_ids": [],
            "invalid_class_names": [],
            "class_validation_details": {},
        }

    if not np.any(mask != background_class):
        return {
            "valid": False,
            "reason": "Segmentation is fully background.",
            "modality": modality,
            "seg_path_received": seg_path_received,
            "seg_path_used": seg_path,
            "present_class_ids": [],
            "present_class_names": [],
            "checked_class_ids": [],
            "checked_class_names": [],
            "missing_class_ids": [],
            "missing_class_names": [],
            "invalid_class_ids": [],
            "invalid_class_names": [],
            "class_validation_details": {},
        }

    present_class_ids = sorted(int(x) for x in np.unique(mask).tolist() if int(x) != background_class)
    present_class_names = [_id_to_name(cid, class_names) for cid in present_class_ids]

    #Connectivity check
    if required_class_ids is None:
        checked_class_ids = present_class_ids.copy()
    else:
        checked_class_ids = [int(cid) for cid in required_class_ids]

    checked_class_names = [_id_to_name(cid, class_names) for cid in checked_class_ids]

    missing_class_ids = []
    invalid_class_ids = []
    class_validation_details = {}

    for cid in checked_class_ids:
        detail = _check_class_has_valid_component(
            mask=mask,
            class_id=cid,
            min_component_size=min_component_size,
        )

        class_validation_details[str(cid)] = {
            "class_name": _id_to_name(cid, class_names),
            **detail,
        }

        if not detail["present"]:
            missing_class_ids.append(cid)
        elif not detail["valid_component"]:
            invalid_class_ids.append(cid)

    missing_class_names = [_id_to_name(cid, class_names) for cid in missing_class_ids]
    invalid_class_names = [_id_to_name(cid, class_names) for cid in invalid_class_ids]

    # Always enforce connectivity result
    if missing_class_ids or invalid_class_ids:
        reasons = []
        if missing_class_names:
            reasons.append(f"Missing classes: {missing_class_names}")
        if invalid_class_names:
            reasons.append(f"Invalid connected components: {invalid_class_names}")

        return {
            "valid": False,
            "reason": " | ".join(reasons),
            "modality": modality,
            "seg_path_received": seg_path_received,
            "seg_path_used": seg_path,
            "present_class_ids": present_class_ids,
            "present_class_names": present_class_names,
            "checked_class_ids": checked_class_ids,
            "checked_class_names": checked_class_names,
            "missing_class_ids": missing_class_ids,
            "missing_class_names": missing_class_names,
            "invalid_class_ids": invalid_class_ids,
            "invalid_class_names": invalid_class_names,
            "class_validation_details": class_validation_details,
        }

    return {
        "valid": True,
        "reason": "Segmentation is valid.",
        "modality": modality,
        "seg_path_received": seg_path_received,
        "seg_path_used": seg_path,
        "present_class_ids": present_class_ids,
        "present_class_names": present_class_names,
        "checked_class_ids": checked_class_ids,
        "checked_class_names": checked_class_names,
        "missing_class_ids": [],
        "missing_class_names": [],
        "invalid_class_ids": [],
        "invalid_class_names": [],
        "class_validation_details": class_validation_details,
    }



#PDF Tool
@tool
def create_pdf_final_report(
    modality: str,                          # "MRI" | "OCT" | "PATHOLOGY"
    patient_id: str,
    image_path: str,
    model_path: str,
    report_text: str,
    # MRI
    segmentation_path: str | None = None,
    orientation: str | None = None,
    # OCT
    prob_path: str | None = None,
    # Pathology
    heatmap_path: str | None = None,
    out_pdf_path: str | None = None,
    web_search_agent: dict | None = None,
) -> dict:
    """
    Create a final PDF report for MRI, OCT, or Pathology.

    MRI requires:
      segmentation_path, orientation

    OCT requires:
      prob_path

    Pathology requires:
      heatmap_path
    """

    modality_clean = (modality or "").strip().upper()

    if modality_clean == "MRI":
        if not segmentation_path:
            return {"ok": False, "error": "segmentation_path required for MRI"}
        if not orientation:
            return {"ok": False, "error": "orientation required for MRI"}

        return create_mri_pdf_report(
            patient_id=patient_id,
            image_path=image_path,
            segmentation_path=segmentation_path,
            orientation=orientation,
            model_path=model_path,
            report_text=report_text,
            out_pdf_path=out_pdf_path,
        )

    if modality_clean == "OCT":
        if not prob_path:
            return {"ok": False, "error": "prob_path required for OCT"}

        return create_oct_pdf_report(
            patient_id=patient_id,
            image_path=image_path,
            prob_path=prob_path,
            model_path=model_path,
            report_text=report_text,
            out_pdf_path=out_pdf_path,
        )

    if modality_clean == "PATHOLOGY":
        if not heatmap_path:
            return {"ok": False, "error": "heatmap_path required for PATHOLOGY"}

        return create_pathology_pdf_report(
            patient_id=patient_id,
            image_path=image_path,
            heatmap_path=heatmap_path,
            model_path=model_path,
            report_text=report_text,
            out_pdf_path=out_pdf_path,
        )

    return {"ok": False, "error": f"Unsupported modality: {modality}. Expected MRI, OCT, or PATHOLOGY."}