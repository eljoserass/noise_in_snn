"""
Detection utilities for object detection.
Shared between ANN and SNN models.
"""

import torch
import torch.nn.functional as F
from typing import List

from .boxes import decode_boxes, xywh_to_xyxy, nms


def detect(model, image: torch.Tensor, anchors: torch.Tensor,
           conf_threshold: float = 0.5, nms_threshold: float = 0.45,
           device: torch.device = None) -> List[dict]:
    """
    Perform detection on a single image.
    
    Args:
        model: Detection model (SSD_VGG9 or similar)
        image: (1, C, H, W) input image tensor
        anchors: (num_anchors, 4) anchor boxes
        conf_threshold: confidence threshold
        nms_threshold: NMS IoU threshold
        device: device to use
        
    Returns:
        List of detections, each with keys: 'box', 'score', 'class'
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        cls_preds, loc_preds = model(image.to(device))
    
    # Process predictions
    cls_preds = F.softmax(cls_preds[0], dim=1)  # (num_anchors, num_classes)
    loc_preds = loc_preds[0]  # (num_anchors, 4)
    
    # Decode boxes
    boxes = decode_boxes(loc_preds, anchors.to(device))
    boxes = xywh_to_xyxy(boxes)
    
    # Clip to [0, 1]
    boxes = boxes.clamp(min=0, max=1)
    
    detections = []
    num_classes = cls_preds.size(1)
    
    # Process each class (skip background at index 0)
    for cls_idx in range(1, num_classes):
        scores = cls_preds[:, cls_idx]
        
        # Filter by confidence
        mask = scores > conf_threshold
        if not mask.any():
            continue
            
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        
        # NMS
        keep = nms(filtered_boxes, filtered_scores, nms_threshold)
        
        for idx in keep:
            detections.append({
                'box': filtered_boxes[idx].cpu().numpy(),
                'score': filtered_scores[idx].item(),
                'class': cls_idx
            })
    
    return detections
