"""
Loss functions for object detection.
Shared between ANN and SNN models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SSDLoss(nn.Module):
    """
    Multi-task loss for SSD: classification loss + localization loss.
    Uses hard negative mining for class imbalance.
    """
    
    def __init__(
        self,
        num_classes: int,
        neg_pos_ratio: float = 3.0,
        alpha: float = 1.0,
        class_weights: torch.Tensor | None = None,
    ):
        super(SSDLoss, self).__init__()
        self.num_classes = num_classes
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if class_weights.numel() != num_classes:
                raise ValueError(
                    f"class_weights length {class_weights.numel()} != num_classes {num_classes}"
                )
        self.register_buffer("class_weights", class_weights)
        
    def forward(self, cls_preds: torch.Tensor, loc_preds: torch.Tensor,
                cls_targets: torch.Tensor, loc_targets: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute SSD loss.
        
        Args:
            cls_preds: (batch, num_anchors, num_classes)
            loc_preds: (batch, num_anchors, 4)
            cls_targets: (batch, num_anchors) - class labels (0 = background)
            loc_targets: (batch, num_anchors, 4) - target box offsets
            
        Returns:
            total_loss, (cls_loss, loc_loss)
        """
        batch_size = cls_preds.size(0)
        num_anchors = cls_preds.size(1)
        
        # Positive mask (non-background)
        pos_mask = cls_targets > 0  # (batch, num_anchors)
        num_pos = pos_mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        # Localization loss (only for positive anchors)
        if pos_mask.any():
            loc_loss = F.smooth_l1_loss(
                loc_preds[pos_mask], 
                loc_targets[pos_mask], 
                reduction='sum'
            )
        else:
            loc_loss = torch.tensor(0.0, device=cls_preds.device)
        
        # Classification loss with hard negative mining
        cls_loss_all = F.cross_entropy(
            cls_preds.view(-1, self.num_classes),
            cls_targets.view(-1),
            weight=self.class_weights,
            reduction='none'
        ).view(batch_size, num_anchors)
        
        # Hard negative mining
        cls_loss_pos = cls_loss_all[pos_mask].sum()
        
        # For negatives, sort by loss and take top-k
        cls_loss_neg = cls_loss_all.clone()
        cls_loss_neg[pos_mask] = 0  # Zero out positive samples
        
        _, idx = cls_loss_neg.sort(dim=1, descending=True)
        _, rank = idx.sort(dim=1)
        
        num_neg = (self.neg_pos_ratio * num_pos).clamp(max=num_anchors - num_pos.sum(dim=1, keepdim=True))
        neg_mask = rank < num_neg.expand_as(rank)
        
        cls_loss_neg = cls_loss_neg[neg_mask].sum()
        
        # Total losses
        total_num_pos = num_pos.sum().float().clamp(min=1)
        cls_loss = (cls_loss_pos + cls_loss_neg) / total_num_pos
        loc_loss = self.alpha * loc_loss / total_num_pos
        
        return cls_loss + loc_loss, (cls_loss, loc_loss)
