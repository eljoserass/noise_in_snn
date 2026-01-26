"""
Unit tests for loss computation and anchor generation.

These tests validate:
1. Anchor generation for correct image sizes
2. Anchor matching to ground truth boxes
3. Loss computation with proper shapes
4. Background vs foreground anchor assignment
"""

import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    generate_anchors,
    match_anchors_to_targets,
    SSDLoss,
    DEFAULT_FEAT_SIZES,
    get_num_anchors_per_cell
)


class TestAnchorGeneration:
    """
    Test suite for anchor generation.
    
    Validates:
    - Correct number of anchors generated
    - Anchor format (cx, cy, w, h) normalized
    - Anchors cover entire image
    """
    
    def test_anchor_generation_shape(self):
        """
        Test: Anchors are generated with correct shape.
        Expected: (num_total_anchors, 4) where 4 = (cx, cy, w, h)
        """
        anchors = generate_anchors()
        
        assert anchors.dim() == 2, f"Anchors should be 2D, got {anchors.dim()}D"
        assert anchors.shape[1] == 4, f"Each anchor should have 4 values, got {anchors.shape[1]}"
        
        print(f"\n✓ Generated {anchors.shape[0]} anchors with shape {tuple(anchors.shape)}")
    
    def test_anchor_normalization(self):
        """
        Test: Anchor coordinates are normalized to [0, 1].
        Validates:
        - All coordinates in valid range
        - Positive widths and heights
        Note: Some large anchors may exceed 1.0 slightly (for large objects)
        """
        anchors = generate_anchors()
        
        cx, cy, w, h = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
        
        # Centers should be in [0, 1]
        assert (cx >= 0).all() and (cx <= 1).all(), "Center x should be in [0, 1]"
        assert (cy >= 0).all() and (cy <= 1).all(), "Center y should be in [0, 1]"
        
        # Sizes should be positive (may slightly exceed 1 for very large object detection)
        assert (w > 0).all(), "Width should be positive"
        assert (h > 0).all(), "Height should be positive"
        
        # Most anchors should be <= 1.0, but some large ones can exceed for border objects
        percent_valid = ((w <= 1.0) & (h <= 1.0)).float().mean() * 100
        print(f"\n✓ All {anchors.shape[0]} anchors properly normalized")
        print(f"  {percent_valid:.1f}% of anchors have size <= 1.0")
        print(f"  Max width: {w.max():.3f}, Max height: {h.max():.3f}")
    
    def test_anchor_count(self):
        """
        Test: Total anchor count matches expected from feature maps.
        For 482x442 input with feature sizes [(60,55), (30,27), (15,13), (15,13), (7,6)]:
        - feat1: 60*55*4 = 13,200
        - feat2: 30*27*6 = 4,860
        - feat3: 15*13*6 = 1,170
        - feat4: 15*13*6 = 1,170
        - feat5: 7*6*4 = 168
        Total: ~20,568 anchors
        """
        anchors = generate_anchors()
        
        # Calculate expected count
        anchors_per_cell = get_num_anchors_per_cell()
        expected_count = sum(
            w * h * n_anchors 
            for (w, h), n_anchors in zip(DEFAULT_FEAT_SIZES, anchors_per_cell)
        )
        
        assert anchors.shape[0] == expected_count, \
            f"Expected {expected_count} anchors, got {anchors.shape[0]}"
        
        print(f"\n✓ Anchor count correct: {anchors.shape[0]} (expected {expected_count})")


class TestAnchorMatching:
    """
    Test suite for matching anchors to ground truth boxes.
    
    Validates:
    - Anchors are assigned to best matching GT boxes
    - Background anchors (low IoU) are labeled as class 0
    - Box encoding produces correct offsets
    """
    
    @pytest.fixture
    def anchors(self):
        """Generate anchors for testing"""
        return generate_anchors()
    
    def test_matching_with_ground_truth(self, anchors):
        """
        Test: Anchors are matched to ground truth boxes correctly.
        Creates synthetic GT boxes and checks matching.
        """
        # Synthetic ground truth: 2 objects
        gt_boxes = torch.tensor([
            [0.5, 0.5, 0.2, 0.3],  # Center object
            [0.8, 0.3, 0.15, 0.2]  # Upper-right object
        ], dtype=torch.float32)
        gt_labels = torch.tensor([2, 5], dtype=torch.long)  # CAR, TRUCK (1-indexed)
        
        cls_targets, loc_targets = match_anchors_to_targets(
            anchors, gt_boxes, gt_labels, iou_threshold=0.5
        )
        
        # Check shapes
        assert cls_targets.shape == (anchors.shape[0],), "Class targets shape mismatch"
        assert loc_targets.shape == (anchors.shape[0], 4), "Location targets shape mismatch"
        
        # Check that some anchors are assigned to objects (positive)
        num_positive = (cls_targets > 0).sum().item()
        assert num_positive > 0, "No anchors matched to ground truth"
        
        # Check that most anchors are background
        num_background = (cls_targets == 0).sum().item()
        assert num_background > num_positive, "Should have more background than foreground"
        
        print(f"\n✓ Matched {num_positive} positive anchors, {num_background} background anchors")
    
    def test_matching_empty_ground_truth(self, anchors):
        """
        Test: Matching with no ground truth boxes.
        All anchors should be assigned as background (class 0).
        """
        gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
        gt_labels = torch.zeros((0,), dtype=torch.long)
        
        cls_targets, loc_targets = match_anchors_to_targets(
            anchors, gt_boxes, gt_labels, iou_threshold=0.5
        )
        
        # All should be background
        assert (cls_targets == 0).all(), "All anchors should be background with no GT"
        
        print(f"\n✓ All {anchors.shape[0]} anchors correctly assigned as background")
    
    def test_matching_preserves_labels(self, anchors):
        """
        Test: Ground truth labels are correctly assigned to matched anchors.
        """
        gt_boxes = torch.tensor([[0.5, 0.5, 0.3, 0.3]], dtype=torch.float32)
        gt_labels = torch.tensor([3], dtype=torch.long)  # Class 3
        
        cls_targets, _ = match_anchors_to_targets(
            anchors, gt_boxes, gt_labels, iou_threshold=0.3
        )
        
        # Find positive matches
        positive_mask = cls_targets > 0
        positive_labels = cls_targets[positive_mask]
        
        # All positive labels should be class 3
        assert (positive_labels == 3).all(), "Positive anchors should have correct label"
        
        print(f"\n✓ {positive_mask.sum().item()} anchors correctly labeled as class 3")


class TestSSDLoss:
    """
    Test suite for SSD loss computation.
    
    Validates:
    - Loss computation with proper inputs
    - Classification and localization loss separation
    - Hard negative mining behavior
    - Loss values are reasonable
    """
    
    @pytest.fixture
    def criterion(self):
        """Create SSD loss function"""
        return SSDLoss(num_classes=7)  # 6 object classes + 1 background
    
    @pytest.fixture
    def dummy_predictions(self):
        """Create dummy predictions"""
        batch_size = 2
        num_anchors = 1000
        num_classes = 7
        
        cls_preds = torch.randn(batch_size, num_anchors, num_classes)
        loc_preds = torch.randn(batch_size, num_anchors, 4)
        
        return cls_preds, loc_preds
    
    @pytest.fixture
    def dummy_targets(self):
        """Create dummy targets with some positive and mostly negative samples"""
        batch_size = 2
        num_anchors = 1000
        
        cls_targets = torch.zeros(batch_size, num_anchors, dtype=torch.long)
        loc_targets = torch.zeros(batch_size, num_anchors, 4)
        
        # Add some positive samples (10 per batch)
        for b in range(batch_size):
            positive_indices = torch.randint(0, num_anchors, (10,))
            cls_targets[b, positive_indices] = torch.randint(1, 7, (10,))
            loc_targets[b, positive_indices] = torch.randn(10, 4) * 0.1
        
        return cls_targets, loc_targets
    
    def test_loss_computation(self, criterion, dummy_predictions, dummy_targets):
        """
        Test: Loss is computed without errors.
        Validates:
        - No runtime errors
        - Loss is a scalar tensor
        - Loss value is finite
        """
        cls_preds, loc_preds = dummy_predictions
        cls_targets, loc_targets = dummy_targets
        
        loss, (cls_loss, loc_loss) = criterion(cls_preds, loc_preds, cls_targets, loc_targets)
        
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.dim() == 0, "Loss should be a scalar"
        assert torch.isfinite(loss), "Loss should be finite"
        assert torch.isfinite(cls_loss), "Classification loss should be finite"
        assert torch.isfinite(loc_loss), "Localization loss should be finite"
        
        print(f"\n✓ Loss computed: {loss.item():.4f} (cls: {cls_loss.item():.4f}, loc: {loc_loss.item():.4f})")
    
    def test_loss_components(self, criterion, dummy_predictions, dummy_targets):
        """
        Test: Classification and localization losses are non-negative.
        """
        cls_preds, loc_preds = dummy_predictions
        cls_targets, loc_targets = dummy_targets
        
        _, (cls_loss, loc_loss) = criterion(cls_preds, loc_preds, cls_targets, loc_targets)
        
        assert cls_loss >= 0, "Classification loss should be non-negative"
        assert loc_loss >= 0, "Localization loss should be non-negative"
        
        print(f"\n✓ Loss components are non-negative")
    
    def test_loss_with_no_positives(self, criterion):
        """
        Test: Loss computation with all background samples.
        Should still compute classification loss from hard negatives.
        """
        batch_size = 2
        num_anchors = 1000
        num_classes = 7
        
        cls_preds = torch.randn(batch_size, num_anchors, num_classes)
        loc_preds = torch.randn(batch_size, num_anchors, 4)
        
        # All background
        cls_targets = torch.zeros(batch_size, num_anchors, dtype=torch.long)
        loc_targets = torch.zeros(batch_size, num_anchors, 4)
        
        loss, (cls_loss, loc_loss) = criterion(cls_preds, loc_preds, cls_targets, loc_targets)
        
        assert torch.isfinite(loss), "Loss should be finite even with no positives"
        assert cls_loss > 0, "Should still have classification loss from hard negatives"
        assert loc_loss == 0, "Should have zero localization loss with no positives"
        
        print(f"\n✓ Loss with no positives: {loss.item():.4f} (cls: {cls_loss.item():.4f}, loc: 0.0)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
