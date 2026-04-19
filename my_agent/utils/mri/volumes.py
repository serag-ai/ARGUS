import numpy as np

def compute_voxel_counts(prediction_map, label_names):
    counts = {}
    for label_id in range(1, len(label_names)):
        counts[label_names[label_id]] = float(np.sum(prediction_map == label_id))
    return counts

def voxel_mm3_from_header(header):
    zooms = header.get_zooms()
    if zooms is None or len(zooms) < 3:
        raise ValueError(f"Invalid header zooms: {zooms}")
    return float(zooms[0] * zooms[1] * zooms[2])
