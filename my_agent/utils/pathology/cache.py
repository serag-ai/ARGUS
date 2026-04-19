# cache.py
import os
import json
import hashlib


_PATHOLOGY_PRED_CACHE = {}


def _key_to_str(key_dict: dict) -> str:
    """
    Stable stringify (safe to pass between tools).
    """
    return json.dumps(key_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def pathology_cache_key(
    images_folder: str,
    model_path: str,
    device: str,
    batch_size: int,
    threshold: float,
    img_size=(224, 224),
) -> dict:
    """
    Stable, supervisor-safe key (NO id(test_dataset)).
    """
    return {
        "images_folder": os.path.abspath(str(images_folder)),
        "model_path": os.path.abspath(str(model_path)),
        "device": str(device),
        "batch_size": int(batch_size),
        "threshold": float(threshold),
        "img_size": [int(img_size[0]), int(img_size[1])],
    }


def cache_key_str(key_dict: dict) -> str:
    return _key_to_str(key_dict)


def run_id_from_key(key_dict: dict) -> str:
    """
    Optional deterministic short id derived from key (useful for naming run dirs elsewhere).
    Cache layer itself does not write to disk.
    """
    s = cache_key_str(key_dict).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]


def cache_set(key_str: str, value: dict):
    """
    Store a JSON-serializable dict in memory under key_str.
    """
    _PATHOLOGY_PRED_CACHE[str(key_str)] = value


def cache_get(key_str: str):
    return _PATHOLOGY_PRED_CACHE.get(str(key_str), None)


def cache_has(key_str: str) -> bool:
    return str(key_str) in _PATHOLOGY_PRED_CACHE


def cache_delete(key_str: str):
    _PATHOLOGY_PRED_CACHE.pop(str(key_str), None)


def cache_clear():
    _PATHOLOGY_PRED_CACHE.clear()


def cache_keys():
    return list(_PATHOLOGY_PRED_CACHE.keys())