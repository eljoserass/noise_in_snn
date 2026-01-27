"""
Anchor generation utilities for SSD.
Shared between ANN and SNN models.
"""

import torch
import torch.nn as nn
import math
from typing import List, Tuple, Optional


# Default anchor configurations per feature map
# (aspect_ratios, scales)
# TUNED for TUMTraf dataset based on object size analysis:
# - 90% of objects < 0.120 scale
# - Bicycles avg: 0.028×0.056, Pedestrians: 0.018×0.059
# - Cars avg: 0.056×0.063, Trucks: 0.099×0.089
# Balanced configuration optimized for small object detection
DEFAULT_ANCHOR_CONFIG = [
    ([1, 2, 0.5], 0.06),          # feat1: very small objects (bicycles, distant cars)
    ([1, 2, 0.5, 3, 1/3], 0.10),  # feat2: small objects (pedestrians, close bicycles)
    ([1, 2, 0.5, 3, 1/3], 0.18),  # feat3: medium objects (cars, close pedestrians)
    ([1, 2, 0.5, 3, 1/3], 0.30),  # feat4: larger vehicles (trucks, buses)
    ([1, 2, 0.5], 0.50),          # feat5: very large objects (close buses/trucks)
]

# Feature map sizes - DEPRECATED: Now calculated dynamically from model
# These are kept as fallback for legacy code only
# Event-based (442x482): [(60, 55), (30, 27), (15, 13), (15, 13), (7, 6)] → 20,568 anchors
# RGB (480x640): [(60, 80), (30, 40), (15, 20), (15, 20), (7, 10)] → 30,280 anchors
DEFAULT_FEAT_SIZES_EB = [
    (60, 55),   # feat1: block3 output (for 442x482)
    (30, 27),   # feat2: block4 output
    (15, 13),   # feat3: block5 output
    (15, 13),   # feat4: extra1 (same size, no pool)
    (7, 6),     # feat5: extra2 (stride 2)
]

DEFAULT_FEAT_SIZES_RGB = [
    (60, 80),   # feat1: block3 output (for 480x640)
    (30, 40),   # feat2: block4 output
    (15, 20),   # feat3: block5 output
    (15, 20),   # feat4: extra1 (same size, no pool)
    (7, 10),    # feat5: extra2 (stride 2)
]

# For backward compatibility
DEFAULT_FEAT_SIZES = DEFAULT_FEAT_SIZES_EB


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


def calculate_feature_map_sizes(model: nn.Module, input_size: Tuple[int, int, int], 
                                device: torch.device) -> List[Tuple[int, int]]:
    """
    Calculate actual feature map sizes by running a forward pass.
    This ensures anchors match the model's actual output dimensions.
    
    Args:
        model: The SSD model (ANN or SNN)
        input_size: (channels, height, width) of input images
        device: Device to run the model on
        
    Returns:
        List of (width, height) tuples for each feature map
        
    Example:
        >>> model = VGG11_SSD_ANN(num_classes=7)
        >>> feat_sizes = calculate_feature_map_sizes(model, (3, 480, 640), device)
        >>> # Returns [(80, 60), (40, 30), (20, 15), (20, 15), (10, 8)]
    """
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_size, device=device)
        
        # Get feature maps by intercepting the model
        x = dummy_input
        
        # Pass through initial layers
        if hasattr(model, 'block1'):  # ANN
            x = model.block1(x)
            x = model.block2(x)
            feat1 = model.block3(x)
            feat2 = model.block4(feat1)
            feat3 = model.block5(feat2)
            feat4 = model.extra1(feat3)
            feat5 = model.extra2(feat4)
        else:  # SNN - similar structure but different layer names
            # Block 1
            cur = model.conv1(x)
            spk = model.lif1(cur)
            spk = model.pool1(spk)
            # Block 2
            cur = model.conv2(spk)
            spk = model.lif2(cur)
            spk = model.pool2(spk)
            # Block 3
            cur = model.conv3_1(spk)
            spk = model.lif3_1(cur)
            cur = model.conv3_2(spk)
            spk = model.lif3_2(cur)
            feat1 = model.pool3(spk)
            # Block 4
            cur = model.conv4_1(feat1)
            spk = model.lif4_1(cur)
            cur = model.conv4_2(spk)
            spk = model.lif4_2(cur)
            feat2 = model.pool4(spk)
            # Block 5
            cur = model.conv5_1(feat2)
            spk = model.lif5_1(cur)
            cur = model.conv5_2(spk)
            spk = model.lif5_2(cur)
            feat3 = model.pool5(spk)
            # Extra layers
            cur = model.conv6(feat3)
            spk = model.lif6(cur)
            cur = model.conv7(spk)
            feat4 = model.lif7(cur)
            
            cur = model.conv8_1(feat4)
            spk = model.lif8_1(cur)
            cur = model.conv8_2(spk)
            feat5 = model.lif8_2(cur)
        
        features = [feat1, feat2, feat3, feat4, feat5]
        
        # Extract (width, height) from each feature map
        feat_sizes = [(f.shape[3], f.shape[2]) for f in features]  # (W, H)
        
    return feat_sizes


def generate_anchors_for_model(model: nn.Module, input_size: Tuple[int, int, int],
                               device: torch.device, 
                               anchor_config: List[Tuple] = None) -> torch.Tensor:
    """
    Generate anchors dynamically based on model's actual feature map sizes.
    This is the RECOMMENDED way to generate anchors.
    
    Args:
        model: The SSD model (ANN or SNN)
        input_size: (channels, height, width) of input images
        device: Device to run the model on
        anchor_config: Optional custom anchor configuration (aspect ratios/scales)
                      Defaults to TUMTraf-optimized config
        
    Returns:
        Anchors tensor of shape (num_total_anchors, 4) on the specified device
        
    Example:
        >>> model = VGG11_SSD_ANN(num_classes=7).to(device)
        >>> anchors = generate_anchors_for_model(model, (3, 480, 640), device)
        >>> print(f"Generated {anchors.shape[0]} anchors")
    """
    feat_sizes = calculate_feature_map_sizes(model, input_size, device)
    anchors = generate_anchors(anchor_config=anchor_config, feat_sizes=feat_sizes)
    return anchors.to(device)
