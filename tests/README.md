# Test Suite Documentation

This directory contains comprehensive tests for the VGG11-SSD implementation for both ANN (RGB) and SNN (Event-based) models.

## Test Overview

**Total Tests**: 44 tests across 3 test files
**Status**: ✅ All passing
**Run Time**: ~24 seconds (CPU only)

## Test Files

### 1. `test_datasets.py` (16 tests)
Tests for data loading and preprocessing pipeline.

**TUMTrafSSD_ANN Tests (8 tests)**:
- ✅ Dataset length (2,358 RGB frames)
- ✅ Sample format (image tensor + targets dict)
- ✅ Image shape (3, 480, 640) for RGB
- ✅ Target shapes (boxes: Nx4, labels: N)
- ✅ Bounding box normalization (0-1 range)
- ✅ Label range validation (0-5 for 6 classes)
- ✅ Multiple sample loading
- ✅ Empty targets handling

**TUMTrafSSD_SNN Tests (8 tests)**:
- ✅ Dataset length (295 sequences)
- ✅ Sequence format (T,C,H,W tensor + list of T targets)
- ✅ Sequence length (8 frames per sequence)
- ✅ Event channels (2 channels for polarity)
- ✅ Sequence shapes (8, 2, 442, 482)
- ✅ Targets per frame (all 8 frames have valid targets)
- ✅ Temporal consistency across frames
- ✅ Multiple sequence loading

### 2. `test_collate.py` (9 tests)
Tests for DataLoader collate functions.

**ANN Collate Tests (3 tests)**:
- ✅ Image stacking to (B, C, H, W)
- ✅ Target preservation as list of dicts
- ✅ Variable object counts per frame (5, 1, 0)

**SNN Collate Tests (4 tests)**:
- ✅ Sequence preservation (T, C, H, W)
- ✅ Target structure (list of lists)
- ✅ Multiple sequences in batch
- ✅ Temporal ordering maintained

**DataLoader Integration (2 tests)**:
- ✅ ANN DataLoader with batch_size=2
- ✅ SNN DataLoader with batch_size=1

### 3. `test_training.py` (19 tests)
Lightweight tests for training components (CPU only, no GPU required).

**VGG11_SSD_ANN Tests (5 tests)**:
- ✅ Model instantiation
- ✅ Parameter count (17,814,916 parameters)
- ✅ Forward pass shapes (classifications: [B, 30320, 6], regressions: [B, 30320, 4])
- ✅ Output dtype (float32)
- ✅ Gradient flow (backward pass)

**VGG11_SSD_SNN Tests (6 tests)**:
- ✅ Model instantiation (with reset_states method)
- ✅ Parameter count (17,814,340 parameters)
- ✅ Single timestep forward pass (classifications: [1, 20624, 6])
- ✅ Sequential processing (8 timesteps)
- ✅ State reset functionality
- ✅ Temporal gradient flow (4 timesteps)

**Optimizer & Scheduler Tests (3 tests)**:
- ✅ ANN optimizer (Adam)
- ✅ SNN optimizer (Adam)
- ✅ Learning rate scheduler (StepLR)

**Checkpointing Tests (3 tests)**:
- ✅ ANN checkpoint save/load
- ✅ SNN checkpoint save/load
- ✅ Full checkpoint (model + optimizer + epoch)

**Training Iteration Tests (2 tests)**:
- ✅ ANN single training iteration
- ✅ SNN single training iteration (8 timesteps)

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_datasets.py -v
pytest tests/test_collate.py -v
pytest tests/test_training.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_datasets.py::TestTUMTrafSSD_ANN -v
pytest tests/test_training.py::TestVGG11_SSD_SNN -v
```

### Run with Verbose Output
```bash
pytest tests/ -v -s
```

### Run with Coverage (optional)
```bash
pip install pytest-cov
pytest tests/ --cov=src --cov-report=html
```

## Key Validation Points

### Data Pipeline ✅
- **ANN**: 2,358 RGB frames loaded correctly
- **SNN**: 295 sequences (8 frames each) loaded correctly
- **Shapes**: RGB (3, 480, 640), Event (2, 442, 482)
- **Normalization**: Bounding boxes in [0, 1] range
- **Labels**: Integer labels in [0, 5] range

### Model Architecture ✅
- **ANN**: 17.8M parameters, 30,320 anchors
- **SNN**: 17.8M parameters, 20,624 anchors
- **Forward Pass**: Correct output shapes
- **Gradients**: Flow correctly through models

### Training Components ✅
- **Optimizers**: Adam with weight decay
- **Schedulers**: StepLR functional
- **Checkpointing**: Save/load works correctly
- **Training Loop**: Single iteration works

## Test Assumptions Validated

1. ✅ **Event images are grayscale**, not 2-channel (converted in dataset)
2. ✅ **Sequences contain 8 frames** each
3. ✅ **295 training sequences** available
4. ✅ **2,358 RGB frames** for ANN training
5. ✅ **Collate functions** handle batching correctly
6. ✅ **SNN maintains temporal state** across frames
7. ✅ **reset_states()** works for sequence initialization
8. ✅ **Gradient flow** works for both ANN and SNN
9. ✅ **Different anchor counts** due to image size differences

## Notes

- **No GPU Required**: All tests run on CPU
- **Lightweight**: Tests check implementation correctness, not training convergence
- **Fast**: Complete test suite runs in ~24 seconds
- **Warning**: One harmless warning about scheduler step order (doesn't affect tests)

## Next Steps

After tests pass, you can proceed with:
1. Implementing SSD loss functions (focal loss + smooth L1)
2. Implementing anchor matching strategy
3. Running actual training on GPU
4. Evaluating on test sets (day, night_with_light_off, night_with_light_on)