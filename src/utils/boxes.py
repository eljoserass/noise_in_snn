"""
Bounding box utilities for object detection.
Shared between ANN and SNN models.
"""

import torch
from typing import List, Tuple


def encode_boxes(matched_boxes: torch.Tensor, anchors: torch.Tensor, 
                 variances: List[float] = [0.1, 0.2]) -> torch.Tensor:
    """
    Encode matched ground truth boxes relative to anchors.
    
    Args:
        matched_boxes: (num_anchors, 4) ground truth boxes in (cx, cy, w, h) format
        anchors: (num_anchors, 4) anchor boxes in (cx, cy, w, h) format
        variances: scaling factors for center and size offsets
        
    Returns:
        Encoded offsets (num_anchors, 4)
    """
    # Center offsets
    g_cx = (matched_boxes[:, 0] - anchors[:, 0]) / (anchors[:, 2] * variances[0])
    g_cy = (matched_boxes[:, 1] - anchors[:, 1]) / (anchors[:, 3] * variances[0])
    
    # Size offsets
    g_w = torch.log(matched_boxes[:, 2] / anchors[:, 2].clamp(min=1e-6)) / variances[1]
    g_h = torch.log(matched_boxes[:, 3] / anchors[:, 3].clamp(min=1e-6)) / variances[1]
    
    return torch.stack([g_cx, g_cy, g_w, g_h], dim=1)


def decode_boxes(loc_preds: torch.Tensor, anchors: torch.Tensor, 
                 variances: List[float] = [0.1, 0.2]) -> torch.Tensor:
    """
    Decode predicted box offsets to actual boxes.
    
    Args:
        loc_preds: (num_anchors, 4) predicted offsets
        anchors: (num_anchors, 4) anchor boxes in (cx, cy, w, h) format
        variances: scaling factors
        
    Returns:
        Decoded boxes (num_anchors, 4) in (cx, cy, w, h) format
    """
    boxes = torch.zeros_like(loc_preds)
    
    boxes[:, 0] = loc_preds[:, 0] * variances[0] * anchors[:, 2] + anchors[:, 0]
    boxes[:, 1] = loc_preds[:, 1] * variances[0] * anchors[:, 3] + anchors[:, 1]
    boxes[:, 2] = torch.exp(loc_preds[:, 2] * variances[1]) * anchors[:, 2]
    boxes[:, 3] = torch.exp(loc_preds[:, 3] * variances[1]) * anchors[:, 3]
    
    return boxes


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format."""
    return torch.stack([
        boxes[:, 0] - boxes[:, 2] / 2,
        boxes[:, 1] - boxes[:, 3] / 2,
        boxes[:, 0] + boxes[:, 2] / 2,
        boxes[:, 1] + boxes[:, 3] / 2,
    ], dim=1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format."""
    return torch.stack([
        (boxes[:, 0] + boxes[:, 2]) / 2,
        (boxes[:, 1] + boxes[:, 3]) / 2,
        boxes[:, 2] - boxes[:, 0],
        boxes[:, 3] - boxes[:, 1],
    ], dim=1)


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: (N, 4) boxes in (x1, y1, x2, y2) format
        boxes2: (M, 4) boxes in (x1, y1, x2, y2) format
        
    Returns:
        IoU matrix of shape (N, M)
    """
    N = boxes1.size(0)
    M = boxes2.size(0)
    
    # Expand to compute pairwise
    boxes1 = boxes1.unsqueeze(1).expand(N, M, 4)
    boxes2 = boxes2.unsqueeze(0).expand(N, M, 4)
    
    # Intersection
    inter_x1 = torch.max(boxes1[:, :, 0], boxes2[:, :, 0])
    inter_y1 = torch.max(boxes1[:, :, 1], boxes2[:, :, 1])
    inter_x2 = torch.min(boxes1[:, :, 2], boxes2[:, :, 2])
    inter_y2 = torch.min(boxes1[:, :, 3], boxes2[:, :, 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    
    # Union
    area1 = (boxes1[:, :, 2] - boxes1[:, :, 0]) * (boxes1[:, :, 3] - boxes1[:, :, 1])
    area2 = (boxes2[:, :, 2] - boxes2[:, :, 0]) * (boxes2[:, :, 3] - boxes2[:, :, 1])
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area.clamp(min=1e-6)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Non-maximum suppression.
    
    Args:
        boxes: (N, 4) boxes in (x1, y1, x2, y2) format
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Indices of kept boxes
    """
    if boxes.size(0) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    
    # Sort by score
    _, order = scores.sort(descending=True)
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
            
        i = order[0].item()
        keep.append(i)
        
        # Compute IoU with remaining boxes
        ious = compute_iou(boxes[i:i+1], boxes[order[1:]])[0]
        
        # Keep boxes with IoU below threshold
        mask = ious <= iou_threshold
        order = order[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def match_anchors_to_targets(anchors: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor,
                              iou_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Match anchors to ground truth boxes.
    
    Args:
        anchors: (num_anchors, 4) in (cx, cy, w, h) format (normalized)
        gt_boxes: (num_gt, 4) in (cx, cy, w, h) format (normalized)
        gt_labels: (num_gt,) class labels (1-indexed, 0 is background)
        iou_threshold: IoU threshold for positive match
        
    Returns:
        cls_targets: (num_anchors,) class labels for each anchor
        loc_targets: (num_anchors, 4) encoded box offsets
    """
    num_anchors = anchors.size(0)
    device = anchors.device
    
    if gt_boxes.size(0) == 0:
        # No ground truth boxes
        return torch.zeros(num_anchors, dtype=torch.long, device=device), \
               torch.zeros(num_anchors, 4, device=device)
    
    # Convert to xyxy for IoU computation
    anchors_xyxy = xywh_to_xyxy(anchors)
    gt_boxes_xyxy = xywh_to_xyxy(gt_boxes)
    
    # Compute IoU
    ious = compute_iou(anchors_xyxy, gt_boxes_xyxy)  # (num_anchors, num_gt)
    
    # For each anchor, find best matching ground truth
    best_gt_iou, best_gt_idx = ious.max(dim=1)
    
    # For each ground truth, find best matching anchor (to ensure all GT are matched)
    best_anchor_iou, best_anchor_idx = ious.max(dim=0)
    
    # Assign best anchor to each GT (force match)
    for gt_idx, anchor_idx in enumerate(best_anchor_idx):
        best_gt_idx[anchor_idx] = gt_idx
        best_gt_iou[anchor_idx] = 2.0  # Ensure this is above threshold
    
    # Assign labels
    cls_targets = gt_labels[best_gt_idx]
    cls_targets[best_gt_iou < iou_threshold] = 0  # Background
    
    # Encode matched boxes
    matched_boxes = gt_boxes[best_gt_idx]
    loc_targets = encode_boxes(matched_boxes, anchors)
    
    return cls_targets, loc_targets
