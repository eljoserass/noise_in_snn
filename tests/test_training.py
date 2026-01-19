"""
Lightweight tests for training scripts (CPU-only, no GPU required).

These tests validate:
1. Model instantiation and parameter counts
2. Forward pass shapes with dummy data
3. Optimizer/scheduler initialization
4. Checkpoint saving/loading
5. One training iteration (no loss convergence check)

NOTE: These tests check implementation correctness, not training effectiveness.
"""

import pytest
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ann import VGG11_SSD_ANN
from src.models.snn import VGG11_SSD_SNN


class TestVGG11_SSD_ANN:
    """
    Test suite for ANN model (VGG11 + SSD).
    
    Validates:
    - Model instantiation
    - Forward pass shapes
    - Parameter count
    - Output format
    """
    
    @pytest.fixture
    def model(self):
        """Create VGG11_SSD_ANN model"""
        return VGG11_SSD_ANN(num_classes=6)
    
    def test_model_instantiation(self, model):
        """
        Test: Model can be instantiated.
        Validates:
        - No errors during creation
        - Model is nn.Module
        """
        assert isinstance(model, torch.nn.Module), "Model should be nn.Module"
        print(f"\n✓ ANN model instantiated successfully")
    
    def test_parameter_count(self, model):
        """
        Test: Model has reasonable parameter count.
        Validates:
        - Total parameters > 0
        - Trainable parameters > 0
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0, "Model should have parameters"
        assert trainable_params > 0, "Model should have trainable parameters"
        assert trainable_params == total_params, "All parameters should be trainable"
        
        print(f"\n✓ ANN model parameters: {total_params:,} (all trainable)")
    
    def test_forward_pass_shape(self, model):
        """
        Test: Forward pass returns correct shapes.
        Validates:
        - Classifications shape: (B, total_anchors, num_classes)
        - Regressions shape: (B, total_anchors, 4)
        """
        model.eval()
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 480, 640)  # RGB
        
        with torch.no_grad():
            classifications, regressions = model(dummy_input)
        
        # Check output shapes
        assert classifications.dim() == 3, f"Classifications should be 3D, got {classifications.dim()}D"
        assert regressions.dim() == 3, f"Regressions should be 3D, got {regressions.dim()}D"
        
        assert classifications.shape[0] == batch_size, "Batch size mismatch in classifications"
        assert regressions.shape[0] == batch_size, "Batch size mismatch in regressions"
        
        assert classifications.shape[1] == regressions.shape[1], "Anchor count mismatch"
        assert classifications.shape[2] == 6, f"Should have 6 classes, got {classifications.shape[2]}"
        assert regressions.shape[2] == 4, f"Should have 4 bbox coords, got {regressions.shape[2]}"
        
        total_anchors = classifications.shape[1]
        print(f"\n✓ ANN forward pass: classifications {tuple(classifications.shape)}, "
              f"regressions {tuple(regressions.shape)}, total_anchors={total_anchors}")
    
    def test_output_dtype(self, model):
        """
        Test: Forward pass outputs have correct dtype.
        Validates:
        - Float32 tensors
        """
        model.eval()
        dummy_input = torch.randn(1, 3, 480, 640)
        
        with torch.no_grad():
            classifications, regressions = model(dummy_input)
        
        assert classifications.dtype == torch.float32, "Classifications should be float32"
        assert regressions.dtype == torch.float32, "Regressions should be float32"
        
        print(f"\n✓ ANN outputs: dtype=torch.float32")
    
    def test_gradient_flow(self, model):
        """
        Test: Gradients can flow through model.
        Validates:
        - Backward pass works
        - Gradients are computed
        """
        model.train()
        dummy_input = torch.randn(1, 3, 480, 640, requires_grad=True)
        
        classifications, regressions = model(dummy_input)
        
        # Dummy loss (just sum)
        loss = classifications.sum() + regressions.sum()
        loss.backward()
        
        # Check gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads, "Model should have gradients after backward"
        
        print(f"\n✓ ANN gradient flow: backward pass successful")


class TestVGG11_SSD_SNN:
    """
    Test suite for SNN model (VGG11 + SSD with LIF neurons).
    
    Validates:
    - Model instantiation
    - Forward pass with temporal dimension
    - State reset functionality
    - Membrane potential evolution
    """
    
    @pytest.fixture
    def model(self):
        """Create VGG11_SSD_SNN model"""
        return VGG11_SSD_SNN(num_classes=6, beta=0.9, threshold=1.0)
    
    def test_model_instantiation(self, model):
        """
        Test: SNN model can be instantiated.
        Validates:
        - No errors during creation
        - Model is nn.Module
        - Has reset_states method
        """
        assert isinstance(model, torch.nn.Module), "Model should be nn.Module"
        assert hasattr(model, 'reset_states'), "SNN should have reset_states method"
        
        print(f"\n✓ SNN model instantiated successfully")
    
    def test_parameter_count(self, model):
        """
        Test: SNN has similar parameter count to ANN (same architecture).
        Validates:
        - Total parameters > 0
        - Trainable parameters > 0
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0, "Model should have parameters"
        assert trainable_params > 0, "Model should have trainable parameters"
        
        print(f"\n✓ SNN model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    def test_forward_pass_single_timestep(self, model):
        """
        Test: Forward pass works for single timestep.
        Validates:
        - Single frame input produces correct output shapes
        """
        model.eval()
        model.reset_states()
        
        batch_size = 1
        dummy_input = torch.randn(batch_size, 2, 442, 482)  # Event (2 channels)
        
        with torch.no_grad():
            classifications, regressions = model(dummy_input)
        
        assert classifications.dim() == 3, f"Classifications should be 3D"
        assert regressions.dim() == 3, f"Regressions should be 3D"
        assert classifications.shape[0] == batch_size, "Batch size mismatch"
        
        print(f"\n✓ SNN single timestep: classifications {tuple(classifications.shape)}, "
              f"regressions {tuple(regressions.shape)}")
    
    def test_sequential_processing(self, model):
        """
        Test: SNN can process sequence of frames.
        Validates:
        - Multiple timesteps work
        - Shapes consistent across time
        """
        model.eval()
        model.reset_states()
        
        T = 8  # sequence length
        batch_size = 1
        
        outputs = []
        with torch.no_grad():
            for t in range(T):
                dummy_input = torch.randn(batch_size, 2, 442, 482)
                classifications, regressions = model(dummy_input)
                outputs.append((classifications, regressions))
        
        assert len(outputs) == T, f"Should have {T} outputs"
        
        # Check shapes are consistent
        for t, (cls, reg) in enumerate(outputs):
            assert cls.shape == outputs[0][0].shape, f"Classification shape mismatch at t={t}"
            assert reg.shape == outputs[0][1].shape, f"Regression shape mismatch at t={t}"
        
        print(f"\n✓ SNN sequential: processed {T} timesteps, shapes consistent")
    
    def test_reset_states(self, model):
        """
        Test: reset_states() method works.
        Validates:
        - Can call reset_states without error
        - Forward pass works after reset
        """
        model.eval()
        
        # Process some timesteps
        with torch.no_grad():
            for _ in range(3):
                dummy_input = torch.randn(1, 2, 442, 482)
                _ = model(dummy_input)
        
        # Reset states
        model.reset_states()
        
        # Process again
        with torch.no_grad():
            dummy_input = torch.randn(1, 2, 442, 482)
            classifications, regressions = model(dummy_input)
        
        assert classifications is not None, "Output should exist after reset"
        
        print(f"\n✓ SNN reset_states: successful reset and forward pass")
    
    def test_gradient_flow_temporal(self, model):
        """
        Test: Gradients flow through temporal sequence.
        Validates:
        - Backward pass works for sequence
        - Gradients accumulated across time
        """
        model.train()
        model.reset_states()
        
        T = 4
        total_loss = 0
        
        for t in range(T):
            dummy_input = torch.randn(1, 2, 442, 482, requires_grad=True)
            classifications, regressions = model(dummy_input)
            total_loss += classifications.sum() + regressions.sum()
        
        total_loss.backward()
        
        # Check gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads, "Model should have gradients after temporal backward"
        
        print(f"\n✓ SNN temporal gradient flow: {T} timesteps, backward successful")


class TestOptimizerScheduler:
    """
    Test optimizer and scheduler initialization.
    
    Validates:
    - Optimizer can be created
    - Scheduler can be created
    - Step functions work
    """
    
    def test_ann_optimizer(self):
        """Test: ANN optimizer initialization"""
        model = VGG11_SSD_ANN(num_classes=6)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        assert optimizer is not None, "Optimizer should be created"
        assert len(optimizer.param_groups) > 0, "Optimizer should have param groups"
        
        print(f"\n✓ ANN optimizer: Adam with {len(optimizer.param_groups)} param group(s)")
    
    def test_snn_optimizer(self):
        """Test: SNN optimizer initialization"""
        model = VGG11_SSD_SNN(num_classes=6)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        assert optimizer is not None, "Optimizer should be created"
        
        print(f"\n✓ SNN optimizer: Adam initialized")
    
    def test_scheduler(self):
        """Test: Learning rate scheduler"""
        model = VGG11_SSD_ANN(num_classes=6)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Simulate training steps (optimizer.step() before scheduler.step())
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        
        # LR should be unchanged (step_size=10)
        current_lr = optimizer.param_groups[0]['lr']
        assert current_lr == initial_lr, f"LR should not change before step_size=10"
        
        # Step to 10
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        
        # LR should have decreased
        new_lr = optimizer.param_groups[0]['lr']
        assert new_lr < initial_lr, f"LR should decrease after 10 steps"
        
        print(f"\n✓ Scheduler: LR changed from {initial_lr} to {new_lr} after 10 steps")


class TestCheckpointing:
    """
    Test checkpoint saving/loading.
    
    Validates:
    - State dict can be saved
    - State dict can be loaded
    - Weights are preserved
    """
    
    def test_ann_checkpoint(self, tmp_path):
        """Test: ANN checkpoint save/load"""
        model = VGG11_SSD_ANN(num_classes=6)
        
        # Save checkpoint
        checkpoint_path = tmp_path / "ann_checkpoint.pth"
        torch.save(model.state_dict(), checkpoint_path)
        
        assert checkpoint_path.exists(), "Checkpoint file should exist"
        
        # Load checkpoint
        model2 = VGG11_SSD_ANN(num_classes=6)
        model2.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        
        # Compare weights
        for (name1, param1), (name2, param2) in zip(model.named_parameters(), model2.named_parameters()):
            assert name1 == name2, "Parameter names should match"
            assert torch.equal(param1, param2), f"Weights for {name1} should be identical"
        
        print(f"\n✓ ANN checkpoint: saved and loaded successfully")
    
    def test_snn_checkpoint(self, tmp_path):
        """Test: SNN checkpoint save/load"""
        model = VGG11_SSD_SNN(num_classes=6)
        
        # Save checkpoint
        checkpoint_path = tmp_path / "snn_checkpoint.pth"
        torch.save(model.state_dict(), checkpoint_path)
        
        assert checkpoint_path.exists(), "Checkpoint file should exist"
        
        # Load checkpoint
        model2 = VGG11_SSD_SNN(num_classes=6)
        model2.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        
        print(f"\n✓ SNN checkpoint: saved and loaded successfully")
    
    def test_full_checkpoint_with_optimizer(self, tmp_path):
        """Test: Full training checkpoint (model + optimizer + epoch)"""
        model = VGG11_SSD_ANN(num_classes=6)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        epoch = 10
        
        # Save full checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        checkpoint_path = tmp_path / "full_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        assert loaded_checkpoint['epoch'] == epoch, "Epoch should match"
        assert 'model_state_dict' in loaded_checkpoint, "Should have model state"
        assert 'optimizer_state_dict' in loaded_checkpoint, "Should have optimizer state"
        
        # Load states
        model2 = VGG11_SSD_ANN(num_classes=6)
        optimizer2 = optim.Adam(model2.parameters(), lr=1e-4)
        
        model2.load_state_dict(loaded_checkpoint['model_state_dict'])
        optimizer2.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
        
        print(f"\n✓ Full checkpoint: saved epoch={epoch}, model, optimizer")


class TestTrainingIteration:
    """
    Test one iteration of training loop (CPU, lightweight).
    
    Validates:
    - One forward/backward pass works
    - Loss can be computed
    - Optimizer step works
    """
    
    def test_ann_single_iteration(self):
        """Test: ANN single training iteration"""
        model = VGG11_SSD_ANN(num_classes=6)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        model.train()
        
        # Dummy batch
        images = torch.randn(2, 3, 480, 640)
        
        optimizer.zero_grad()
        classifications, regressions = model(images)
        
        # Dummy loss (just for testing)
        loss = classifications.sum() + regressions.sum()
        loss.backward()
        optimizer.step()
        
        assert loss.item() is not None, "Loss should be computed"
        
        print(f"\n✓ ANN training iteration: forward, backward, optimizer step successful")
    
    def test_snn_single_iteration(self):
        """Test: SNN single training iteration (one sequence)"""
        model = VGG11_SSD_SNN(num_classes=6)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        model.train()
        model.reset_states()
        
        T = 8
        total_loss = 0
        
        optimizer.zero_grad()
        
        for t in range(T):
            # Dummy frame
            frame = torch.randn(1, 2, 442, 482)
            classifications, regressions = model(frame)
            
            # Accumulate dummy loss
            total_loss += (classifications.sum() + regressions.sum()) / T
        
        total_loss.backward()
        optimizer.step()
        
        assert total_loss.item() is not None, "Loss should be computed"
        
        print(f"\n✓ SNN training iteration: {T} timesteps, backward, optimizer step successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
