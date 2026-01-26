"""
Anchor generation utilities for SSD.
Shared between ANN and SNN models.
"""

import torch
import math
from typing import List, Tuple


# Default anchor configurations per feature map
# (aspect_ratios, scales)
# For TUMTraf dataset: cars (wide), pedestrians (tall), trucks/buses (large)
DEFAULT_ANCHOR_CONFIG = [
    ([1, 2, 0.5], 0.1),           # feat1: small objects
    ([1, 2, 0.5, 3, 1/3], 0.2),   # feat2: medium objects  
    ([1, 2, 0.5, 3, 1/3], 0.375), # feat3: larger objects
    ([1, 2, 0.5, 3, 1/3], 0.55),  # feat4: large vehicles
    ([1, 2, 0.5], 0.725),         # feat5: very large objects
]

# Feature map sizes for 482x442 input (actual TUMTraf image size)
# After successive /2 pooling: 241x221 → 120x110 → 60x55 → 30x27 → 15x13 → 7x6
DEFAULT_FEAT_SIZES = [
    (60, 55),   # feat1: block3 output
    (30, 27),   # feat2: block4 output
    (15, 13),   # feat3: block5 output
    (15, 13),   # feat4: extra1 (same size, no pool)
    (7, 6),     # feat5: extra2 (stride 2)
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
