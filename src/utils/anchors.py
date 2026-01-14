"""
Anchor generation utilities for SSD.
Shared between ANN and SNN models.
"""

import torch
import math
from typing import List, Tuple


# Default anchor configurations per feature map
# (aspect_ratios, scales)
DEFAULT_ANCHOR_CONFIG = [
    ([1, 2, 0.5], 0.1),           # feat1: 40x30
    ([1, 2, 0.5, 3, 1/3], 0.2),   # feat2: 20x15
    ([1, 2, 0.5, 3, 1/3], 0.35),  # feat3: 10x7
    ([1, 2, 0.5, 3, 1/3], 0.5),   # feat4: 5x4
    ([1, 2, 0.5], 0.65),          # feat5: 3x2
    ([1, 2, 0.5], 0.8),           # feat6: 1x1
]

# Feature map sizes for 640x480 input
DEFAULT_FEAT_SIZES = [
    (40, 30), (20, 15), (10, 7), (5, 4), (3, 2), (1, 1)
]


def generate_anchors(anchor_config: List[Tuple] = None, 
                     feat_sizes: List[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Generate default anchor boxes for all feature maps.
    
    Args:
        anchor_config: List of (aspect_ratios, scale) tuples for each feature map
        feat_sizes: List of (width, height) tuples for each feature map
        
    Returns:
        Tensor of shape (num_total_anchors, 4) with (cx, cy, w, h) in normalized coordinates.
    """
    if anchor_config is None:
        anchor_config = DEFAULT_ANCHOR_CONFIG
    if feat_sizes is None:
        feat_sizes = DEFAULT_FEAT_SIZES
    
    anchors = []
    
    for idx, ((aspect_ratios, scale), (fw, fh)) in enumerate(zip(anchor_config, feat_sizes)):
        # Add next scale for computing additional anchor
        next_scale = anchor_config[idx + 1][1] if idx < len(anchor_config) - 1 else 1.0
        
        for i in range(fh):
            for j in range(fw):
                # Center of anchor (normalized)
                cx = (j + 0.5) / fw
                cy = (i + 0.5) / fh
                
                # For aspect ratio 1, add an extra anchor with geometric mean of scales
                anchors.append([cx, cy, scale, scale])
                anchors.append([cx, cy, math.sqrt(scale * next_scale), math.sqrt(scale * next_scale)])
                
                # Other aspect ratios
                for ar in aspect_ratios:
                    if ar != 1:
                        anchors.append([cx, cy, scale * math.sqrt(ar), scale / math.sqrt(ar)])
    
    return torch.tensor(anchors, dtype=torch.float32)


def get_num_anchors_per_cell(anchor_config: List[Tuple] = None) -> List[int]:
    """
    Calculate number of anchors per cell for each feature map.
    
    Args:
        anchor_config: List of (aspect_ratios, scale) tuples for each feature map
        
    Returns:
        List of number of anchors per feature map
    """
    if anchor_config is None:
        anchor_config = DEFAULT_ANCHOR_CONFIG
    
    return [len(cfg[0]) + 1 for cfg in anchor_config]  # +1 for additional scale
