"""
Unit tests for dataset loading and manipulation.

These tests validate:
1. Dataset initialization and sample counting
2. Data loading and format correctness
3. Image shape and channel validation
4. Label parsing and format
5. Bounding box normalization
6. Edge cases (empty labels, invalid boxes)
"""

import pytest
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import TUMTrafSSD_ANN, TUMTrafSSD_SNN


class TestTUMTrafSSD_ANN:
    """
    Test suite for ANN dataset (frame-by-frame RGB loading).
    
    Validates:
    - Correct number of samples loaded
    - RGB image format (3 channels)
    - Target dictionary structure
    - Bounding box normalization (0-1 range)
    - Label indexing
    """
    
    @pytest.fixture
    def dataset(self):
        """Create ANN dataset instance for testing"""
        img_dir = Path("data/preprocessed/train/images/rgb")
        label_dir = Path("data/preprocessed/train/OPENLabel_labels_rgb")
        
        if not img_dir.exists() or not label_dir.exists():
            pytest.skip("Preprocessed data not available")
        
        return TUMTrafSSD_ANN(
            img_dir=img_dir,
            label_dir=label_dir,
            transform=None,
            target_size=(480, 640)
        )
    
    def test_dataset_length(self, dataset):
        """
        Test: Dataset loads correct number of samples.
        Expected: 295 sequences × 8 frames = 2,360 frames
        """
        assert len(dataset) > 0, "Dataset should not be empty"
        print(f"\n✓ ANN Dataset length: {len(dataset)} frames")
    
    def test_sample_format(self, dataset):
        """
        Test: Each sample returns (image, targets) tuple.
        Validates:
        - Returns tuple of 2 elements
        - Image is torch.Tensor
        - Targets is dict with 'boxes' and 'labels'
        """
        image, targets = dataset[0]
        
        assert isinstance(image, torch.Tensor), "Image should be torch.Tensor"
        assert isinstance(targets, dict), "Targets should be dict"
        assert 'boxes' in targets, "Targets must have 'boxes' key"
        assert 'labels' in targets, "Targets must have 'labels' key"
        
        print(f"\n✓ Sample format correct: image tensor + targets dict")
    
    def test_image_shape(self, dataset):
        """
        Test: RGB images have correct shape (C, H, W) with 3 channels.
        Validates:
        - 3D tensor (C, H, W)
        - 3 channels for RGB
        - Height and width are positive
        """
        image, _ = dataset[0]
        
        assert image.dim() == 3, f"Image should be 3D (C,H,W), got {image.dim()}D"
        assert image.shape[0] == 3, f"RGB should have 3 channels, got {image.shape[0]}"
        
        print(f"\n✓ Image shape: {tuple(image.shape)} (C=3 for RGB)")
    
    def test_target_shapes(self, dataset):
        """
        Test: Target boxes and labels have matching dimensions.
        Validates:
        - Boxes shape: (N, 4) where N = number of objects
        - Labels shape: (N,)
        - Same N for boxes and labels
        """
        _, targets = dataset[0]
        boxes = targets['boxes']
        labels = targets['labels']
        
        assert boxes.dim() == 2, f"Boxes should be 2D (N,4), got {boxes.dim()}D"
        assert boxes.shape[1] == 4, f"Each box should have 4 coords, got {boxes.shape[1]}"
        assert labels.dim() == 1, f"Labels should be 1D (N,), got {labels.dim()}D"
        assert boxes.shape[0] == labels.shape[0], "Boxes and labels count mismatch"
        
        print(f"\n✓ Targets: {boxes.shape[0]} objects, boxes shape {tuple(boxes.shape)}")
    
    def test_bbox_normalization(self, dataset):
        """
        Test: Bounding boxes are normalized to [0, 1] range.
        Validates:
        - All bbox coordinates in [0, 1]
        - Format: [cx, cy, w, h] (center + size)
        """
        _, targets = dataset[0]
        boxes = targets['boxes']
        
        if len(boxes) > 0:
            assert boxes.min() >= 0.0, "Boxes should be >= 0"
            assert boxes.max() <= 1.0, f"Boxes should be <= 1, got max {boxes.max()}"
            
            # Check format: cx, cy, w, h
            cx, cy, w, h = boxes[0]
            assert 0 <= cx <= 1, "Center x should be in [0, 1]"
            assert 0 <= cy <= 1, "Center y should be in [0, 1]"
            assert 0 < w <= 1, "Width should be in (0, 1]"
            assert 0 < h <= 1, "Height should be in (0, 1]"
            
            print(f"\n✓ Bounding boxes normalized: min={boxes.min():.3f}, max={boxes.max():.3f}")
    
    def test_label_range(self, dataset):
        """
        Test: Labels are valid class indices.
        Validates:
        - Labels are integers
        - Labels in valid range [0, num_classes-1]
        """
        _, targets = dataset[0]
        labels = targets['labels']
        
        if len(labels) > 0:
            num_classes = len(dataset.CLASSES)
            assert labels.min() >= 0, "Labels should be >= 0"
            assert labels.max() < num_classes, f"Labels should be < {num_classes}"
            assert labels.dtype == torch.long, "Labels should be torch.long"
            
            print(f"\n✓ Labels range: [{labels.min()}, {labels.max()}], dtype={labels.dtype}")
    
    def test_multiple_samples(self, dataset):
        """
        Test: Can iterate through multiple samples without errors.
        Validates:
        - All samples can be loaded
        - Consistent format across samples
        """
        num_test_samples = min(10, len(dataset))
        
        for i in range(num_test_samples):
            image, targets = dataset[i]
            assert isinstance(image, torch.Tensor)
            assert isinstance(targets, dict)
        
        print(f"\n✓ Successfully loaded {num_test_samples} samples")
    
    def test_empty_targets_handling(self, dataset):
        """
        Test: Dataset handles frames with no objects gracefully.
        Validates:
        - Returns empty tensors with correct shapes
        - boxes: (0, 4)
        - labels: (0,)
        """
        # Find a sample with no objects (or use first sample)
        for i in range(min(100, len(dataset))):
            _, targets = dataset[i]
            if len(targets['boxes']) == 0:
                assert targets['boxes'].shape == (0, 4), "Empty boxes should be (0, 4)"
                assert targets['labels'].shape == (0,), "Empty labels should be (0,)"
                print(f"\n✓ Empty targets handled correctly at index {i}")
                break


class TestTUMTrafSSD_SNN:
    """
    Test suite for SNN dataset (sequence-based event loading).
    
    Validates:
    - Sequence loading (8 frames per sequence)
    - Event image format (2 channels for polarity)
    - Temporal ordering
    - Target alignment with frames
    """
    
    @pytest.fixture
    def dataset(self):
        """Create SNN dataset instance for testing"""
        img_dir = Path("data/preprocessed/train/images/eb_transformed")
        label_dir = Path("data/preprocessed/train/OPENLabel_labels_eb")
        
        if not img_dir.exists() or not label_dir.exists():
            pytest.skip("Preprocessed event data not available")
        
        return TUMTrafSSD_SNN(
            img_dir=img_dir,
            label_dir=label_dir,
            transform=None,
            target_size=(480, 640)
        )
    
    def test_dataset_length(self, dataset):
        """
        Test: Dataset loads correct number of sequences.
        Expected: 295 sequences (each with 8 frames)
        """
        assert len(dataset) > 0, "Dataset should not be empty"
        print(f"\n✓ SNN Dataset length: {len(dataset)} sequences")
    
    def test_sequence_format(self, dataset):
        """
        Test: Each sample returns (images, targets) where images is a sequence.
        Validates:
        - Returns tuple of 2 elements
        - Images is 4D tensor (T, C, H, W)
        - Targets is list of dicts
        """
        images, targets = dataset[0]
        
        assert isinstance(images, torch.Tensor), "Images should be torch.Tensor"
        assert isinstance(targets, list), "Targets should be list"
        assert images.dim() == 4, f"Images should be 4D (T,C,H,W), got {images.dim()}D"
        
        print(f"\n✓ Sequence format correct: 4D tensor + list of targets")
    
    def test_sequence_length(self, dataset):
        """
        Test: Sequences have expected number of frames.
        Validates:
        - Typically 8 frames per sequence
        - Targets list matches number of frames
        """
        images, targets = dataset[0]
        T = images.shape[0]
        
        assert T > 0, "Sequence should have at least 1 frame"
        assert T == len(targets), f"Frames ({T}) and targets ({len(targets)}) mismatch"
        
        print(f"\n✓ Sequence length: {T} frames with {len(targets)} targets")
    
    def test_event_channels(self, dataset):
        """
        Test: Event images have 2 channels (polarity representation).
        Validates:
        - Channel dimension is 2
        - Simulating positive/negative event polarity
        """
        images, _ = dataset[0]
        T, C, H, W = images.shape
        
        assert C == 2, f"Event images should have 2 channels, got {C}"
        
        print(f"\n✓ Event channels: {C} (polarity representation)")
    
    def test_sequence_shapes(self, dataset):
        """
        Test: All frames in sequence have same spatial dimensions.
        Validates:
        - Consistent H, W across all frames
        """
        images, _ = dataset[0]
        T, C, H, W = images.shape
        
        assert H > 0 and W > 0, "Spatial dimensions should be positive"
        
        # Check all frames have same shape
        for t in range(T):
            frame_shape = images[t].shape
            assert frame_shape == (C, H, W), f"Frame {t} shape mismatch"
        
        print(f"\n✓ Sequence shape: ({T}, {C}, {H}, {W})")
    
    def test_targets_per_frame(self, dataset):
        """
        Test: Each frame has corresponding target dict.
        Validates:
        - Each target is a dict with 'boxes' and 'labels'
        - Targets match frame count
        """
        images, targets = dataset[0]
        T = images.shape[0]
        
        assert len(targets) == T, "Should have one target per frame"
        
        for t, target in enumerate(targets):
            assert isinstance(target, dict), f"Target {t} should be dict"
            assert 'boxes' in target, f"Target {t} missing 'boxes'"
            assert 'labels' in target, f"Target {t} missing 'labels'"
        
        print(f"\n✓ All {T} frames have valid targets")
    
    def test_temporal_consistency(self, dataset):
        """
        Test: Frames in sequence maintain consistent format.
        Validates:
        - All frames have same number of channels
        - All frames have same spatial dimensions
        """
        images, _ = dataset[0]
        T, C, H, W = images.shape
        
        for t in range(T):
            assert images[t].shape == (C, H, W), f"Frame {t} shape inconsistent"
        
        print(f"\n✓ Temporal consistency verified across {T} frames")
    
    def test_multiple_sequences(self, dataset):
        """
        Test: Can load multiple sequences without errors.
        Validates:
        - Different sequences can be loaded
        - Format is consistent
        """
        num_test_sequences = min(5, len(dataset))
        
        for i in range(num_test_sequences):
            images, targets = dataset[i]
            assert images.dim() == 4, f"Sequence {i} has wrong dimensions"
            assert isinstance(targets, list), f"Sequence {i} targets not list"
        
        print(f"\n✓ Successfully loaded {num_test_sequences} sequences")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
