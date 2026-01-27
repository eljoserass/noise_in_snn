"""
Utility modules for object detection.
"""

from .boxes import (
    encode_boxes,
    decode_boxes,
    xywh_to_xyxy,
    xyxy_to_xywh,
    compute_iou,
    nms,
    match_anchors_to_targets
)

from .anchors import (
    generate_anchors,
    generate_anchors_for_model,
    calculate_feature_map_sizes,
    get_num_anchors_per_cell,
    DEFAULT_ANCHOR_CONFIG,
    DEFAULT_FEAT_SIZES,
    DEFAULT_FEAT_SIZES_RGB,
    DEFAULT_FEAT_SIZES_EB
)

from .losses import SSDLoss

from .detection import detect

__all__ = [
    # Box utilities
    'encode_boxes',
    'decode_boxes',
    'xywh_to_xyxy',
    'xyxy_to_xywh',
    'compute_iou',
    'nms',
    'match_anchors_to_targets',
    # Anchor utilities
    'generate_anchors',
    'generate_anchors_for_model',
    'calculate_feature_map_sizes',
    'get_num_anchors_per_cell',
    'DEFAULT_ANCHOR_CONFIG',
    'DEFAULT_FEAT_SIZES',
    'DEFAULT_FEAT_SIZES_RGB',
    'DEFAULT_FEAT_SIZES_EB',
    # Loss
    'SSDLoss',
    # Detection
    'detect',
]
