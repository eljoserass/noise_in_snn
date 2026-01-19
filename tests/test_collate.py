"""
Unit tests for collate functions used in DataLoaders.

These tests validate:
1. ANN collate function (batch frames)
2. SNN collate function (batch sequences)
3. Tensor stacking and format
4. Handling variable-sized batches
"""

import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import TUMTrafSSD_ANN, TUMTrafSSD_SNN
from scripts.train_ann_rgb import collate_fn
from scripts.train_snn_eb import collate_fn_snn


class TestCollateANN:
    """
    Test suite for ANN collate function.
    
    Validates:
    - Batching multiple frames together
    - Stacking images into (B, C, H, W)
    - Preserving target dictionaries
    """
    
    @pytest.fixture
    def sample_batch(self):
        """Create mock batch of ANN samples"""
        # Simulate 3 samples in a batch
        batch = []
        for i in range(3):
            image = torch.randn(3, 480, 640)  # RGB image
            targets = {
                'boxes': torch.rand(2, 4),  # 2 objects, normalized boxes
                'labels': torch.randint(0, 6, (2,))
            }
            batch.append((image, targets))
        return batch
    
    def test_collate_stacks_images(self, sample_batch):
        """
        Test: Collate stacks images into batch dimension.
        Validates:
        - Output images shape: (B, C, H, W)
        - B = batch size
        """
        images, targets = collate_fn(sample_batch)
        
        assert isinstance(images, torch.Tensor), "Images should be stacked tensor"
        assert images.dim() == 4, f"Batched images should be 4D, got {images.dim()}D"
        assert images.shape[0] == len(sample_batch), "Batch size mismatch"
        assert images.shape[1] == 3, "Should have 3 RGB channels"
        
        print(f"\n✓ ANN collate: stacked {images.shape[0]} images → {tuple(images.shape)}")
    
    def test_collate_preserves_targets(self, sample_batch):
        """
        Test: Collate preserves targets as list of dicts.
        Validates:
        - Targets is list
        - Length matches batch size
        - Each element is dict with 'boxes' and 'labels'
        """
        _, targets = collate_fn(sample_batch)
        
        assert isinstance(targets, list), "Targets should be list"
        assert len(targets) == len(sample_batch), "Targets count mismatch"
        
        for i, target in enumerate(targets):
            assert isinstance(target, dict), f"Target {i} should be dict"
            assert 'boxes' in target, f"Target {i} missing 'boxes'"
            assert 'labels' in target, f"Target {i} missing 'labels'"
        
        print(f"\n✓ ANN collate: preserved {len(targets)} target dicts")
    
    def test_collate_variable_objects(self):
        """
        Test: Collate handles frames with different number of objects.
        Validates:
        - Some frames have many objects, some have few/none
        - All handled correctly
        """
        batch = [
            (torch.randn(3, 480, 640), {'boxes': torch.rand(5, 4), 'labels': torch.randint(0, 6, (5,))}),
            (torch.randn(3, 480, 640), {'boxes': torch.rand(1, 4), 'labels': torch.randint(0, 6, (1,))}),
            (torch.randn(3, 480, 640), {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.long)}),
        ]
        
        images, targets = collate_fn(batch)
        
        assert images.shape[0] == 3, "Should handle 3 samples"
        assert len(targets) == 3, "Should have 3 target dicts"
        assert targets[0]['boxes'].shape[0] == 5, "First sample should have 5 objects"
        assert targets[1]['boxes'].shape[0] == 1, "Second sample should have 1 object"
        assert targets[2]['boxes'].shape[0] == 0, "Third sample should have 0 objects"
        
        print(f"\n✓ ANN collate: handled variable objects (5, 1, 0)")


class TestCollateSNN:
    """
    Test suite for SNN collate function.
    
    Validates:
    - Batching sequences (typically batch_size=1)
    - Preserving temporal dimension
    - Handling sequence format
    """
    
    @pytest.fixture
    def sample_sequence_batch(self):
        """Create mock batch of SNN sequences"""
        # Simulate 1 sequence (typical for SNN)
        T = 8  # frames per sequence
        sequence_images = torch.randn(T, 2, 480, 640)  # Event sequence
        
        sequence_targets = []
        for t in range(T):
            targets = {
                'boxes': torch.rand(2, 4),
                'labels': torch.randint(0, 6, (2,))
            }
            sequence_targets.append(targets)
        
        batch = [(sequence_images, sequence_targets)]
        return batch
    
    def test_collate_preserves_sequences(self, sample_sequence_batch):
        """
        Test: SNN collate preserves sequence structure.
        Validates:
        - Returns lists (not stacked tensors)
        - Each sequence maintains (T, C, H, W) shape
        """
        images_list, targets_list = collate_fn_snn(sample_sequence_batch)
        
        assert isinstance(images_list, list), "Should return list of sequences"
        assert isinstance(targets_list, list), "Should return list of target sequences"
        assert len(images_list) == len(sample_sequence_batch), "Sequence count mismatch"
        
        # Check first sequence
        seq_images = images_list[0]
        assert seq_images.dim() == 4, f"Sequence should be 4D (T,C,H,W), got {seq_images.dim()}D"
        assert seq_images.shape[1] == 2, "Event images should have 2 channels"
        
        print(f"\n✓ SNN collate: preserved sequence shape {tuple(seq_images.shape)}")
    
    def test_collate_targets_structure(self, sample_sequence_batch):
        """
        Test: SNN collate preserves targets as list of lists.
        Validates:
        - Outer list: sequences in batch
        - Inner list: targets per frame in sequence
        """
        _, targets_list = collate_fn_snn(sample_sequence_batch)
        
        assert len(targets_list) == 1, "Should have 1 sequence"
        
        seq_targets = targets_list[0]
        assert isinstance(seq_targets, list), "Sequence targets should be list"
        
        T = len(seq_targets)
        for t in range(T):
            assert isinstance(seq_targets[t], dict), f"Frame {t} target should be dict"
            assert 'boxes' in seq_targets[t], f"Frame {t} missing 'boxes'"
            assert 'labels' in seq_targets[t], f"Frame {t} missing 'labels'"
        
        print(f"\n✓ SNN collate: preserved {T} frame targets in sequence")
    
    def test_collate_multiple_sequences(self):
        """
        Test: SNN collate can handle multiple sequences in batch.
        Validates:
        - Multiple sequences preserved separately
        - Each maintains correct format
        """
        batch = []
        for _ in range(2):  # 2 sequences
            T = 8
            seq_images = torch.randn(T, 2, 480, 640)
            seq_targets = [{'boxes': torch.rand(1, 4), 'labels': torch.randint(0, 6, (1,))} for _ in range(T)]
            batch.append((seq_images, seq_targets))
        
        images_list, targets_list = collate_fn_snn(batch)
        
        assert len(images_list) == 2, "Should have 2 sequences"
        assert len(targets_list) == 2, "Should have 2 target sequences"
        
        for i in range(2):
            assert images_list[i].shape[0] == 8, f"Sequence {i} should have 8 frames"
            assert len(targets_list[i]) == 8, f"Sequence {i} should have 8 target dicts"
        
        print(f"\n✓ SNN collate: handled {len(images_list)} sequences")
    
    def test_collate_temporal_order(self, sample_sequence_batch):
        """
        Test: SNN collate maintains temporal ordering of frames.
        Validates:
        - Frame order is preserved
        - Target alignment with frames is maintained
        """
        images_list, targets_list = collate_fn_snn(sample_sequence_batch)
        
        seq_images = images_list[0]
        seq_targets = targets_list[0]
        
        T = seq_images.shape[0]
        assert T == len(seq_targets), "Frames and targets count must match"
        
        # Verify temporal dimension is first
        assert seq_images.shape == (T, 2, 480, 640), "Temporal dimension should be first"
        
        print(f"\n✓ SNN collate: temporal order preserved ({T} frames)")


class TestDataLoaderIntegration:
    """
    Integration tests for DataLoader with collate functions.
    
    Validates:
    - DataLoader can iterate with custom collate
    - Batching works correctly
    - No errors during iteration
    """
    
    def test_ann_dataloader(self):
        """
        Test: ANN DataLoader works with collate_fn.
        """
        from torch.utils.data import DataLoader
        
        img_dir = Path("data/preprocessed/train/images/rgb")
        label_dir = Path("data/preprocessed/train/OPENLabel_labels_rgb")
        
        if not img_dir.exists():
            pytest.skip("Data not available")
        
        dataset = TUMTrafSSD_ANN(img_dir, label_dir, transform=None)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)
        
        # Try to get one batch
        images, targets = next(iter(dataloader))
        
        assert images.shape[0] == 2, "Batch size should be 2"
        assert len(targets) == 2, "Should have 2 target dicts"
        
        print(f"\n✓ ANN DataLoader: batch shape {tuple(images.shape)}")
    
    def test_snn_dataloader(self):
        """
        Test: SNN DataLoader works with collate_fn_snn.
        """
        from torch.utils.data import DataLoader
        
        img_dir = Path("data/preprocessed/train/images/eb_transformed")
        label_dir = Path("data/preprocessed/train/OPENLabel_labels_eb")
        
        if not img_dir.exists():
            pytest.skip("Data not available")
        
        dataset = TUMTrafSSD_SNN(img_dir, label_dir, transform=None)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_snn, shuffle=False)
        
        # Try to get one batch (one sequence)
        sequences, targets_sequences = next(iter(dataloader))
        
        assert len(sequences) == 1, "Batch should contain 1 sequence"
        assert sequences[0].dim() == 4, "Sequence should be 4D"
        
        print(f"\n✓ SNN DataLoader: sequence shape {tuple(sequences[0].shape)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
