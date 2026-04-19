#!/usr/bin/env python
# coding: utf-8

from typing import Any, Dict, Tuple, List
from typing_extensions import TypedDict


class MRIState(TypedDict, total=False):
    # Inputs
    user_request: str
    target_structure: str              
    input_pkl_path: str
    input_nii_path: str 

    # Planning
    plan: List[Dict[str, Any]]          
    step_index: int                     
    execution_trace: List[Dict[str, Any]]

    # Raw data
    loaded_mri: Any                     
    reference_mask: Any                 
    voxel_spacing: Tuple[float, float, float]

    # Segmentation
    segmentation_method: str            
    segmentation_mask: Any              
    segmentation_model_name: str
    segmentation_source: str            
    confidence_score: float
    target_label_value: int

    # Volume metrics
    voxel_count: int
    volume_mm3: float
    volume_ml: float

    # Quality metrics
    dice_score: float
    iou_score: float
    segmentation_quality_ok: bool
    fallback_used: bool

    # Output
    summary_text: str                  
    full_report_text: str               
    visualization_info: Dict[str, Any]

