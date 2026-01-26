# Neuromorphic vs. Noise: Comparative Analysis of ANNs and SNNs for Event-Based Object Detection

## Abstract

This repository implements a comparative study of Artificial Neural Networks (ANNs) and Spiking Neural Networks (SNNs) for object detection in challenging visual conditions. Leveraging the TUMTraf Event Dataset, we evaluate VGG11-SSD architectures on both conventional RGB imagery and neuromorphic event-based data. The implementation provides a complete pipeline for training, evaluation, and performance comparison between frame-based and event-driven visual processing paradigms.

## Project Overview

**Objective**: Assess the robustness and efficiency of SNNs compared to ANNs for autonomous driving perception under varying noise conditions and lighting scenarios.

**Architecture**: VGG11 backbone with Single Shot MultiBox Detector (SSD) heads  
**Dataset**: TUMTraf Event Dataset (RGB + Event-based cameras)  
**Classes**: 6 object categories (BICYCLE, BUS, CAR, PEDESTRIAN, TRAILER, TRUCK)  
**Task**: Real-time object detection with bounding box regression

## Repository Structure

```
noise_in_snn/
├── src/                        # Source code modules
│   ├── models/                 # Neural network architectures
│   │   ├── ann.py             # VGG11-SSD for RGB (ANN)
│   │   └── snn.py             # VGG11-SSD for Events (SNN)
│   ├── data/                   # Data loading and preprocessing
│   │   └── dataset.py         # TUMTraf dataset loaders
│   └── utils/                  # Utility functions
│       ├── anchors.py         # Anchor box generation
│       ├── boxes.py           # Bounding box operations
│       ├── losses.py          # SSD loss implementation
│       ├── detection.py       # Detection and NMS
│       └── metrics.py         # Evaluation metrics
├── scripts/                    # Executable scripts
│   ├── preprocess.py          # Data preprocessing
│   ├── train_ann_rgb.py       # ANN training pipeline
│   ├── train_snn_eb.py        # SNN training pipeline
│   └── evaluate.py            # Model evaluation
├── tests/                      # Comprehensive test suite (63 tests)
│   ├── test_datasets.py       # Data loading tests
│   ├── test_collate.py        # Batching tests
│   ├── test_training.py       # Model and training tests
│   ├── test_loss_and_anchors.py  # Loss computation tests
│   └── test_training_integration.py  # Integration tests
├── data/                       # Dataset storage (gitignored)
├── checkpoints/                # Model checkpoints (gitignored)
├── requirements.txt            # Python dependencies
└── run.sh                      # Pipeline orchestration script
```

## Installation

### Prerequisites

- Python 3.13+
- CUDA-compatible GPU (recommended for training)
- 1GB disk space for TUMTraf dataset

### Setup

```bash
# Clone repository
git clone <repository-url>
cd noise_in_snn

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Dependencies

- **PyTorch**: Deep learning framework
- **SNNTorch**: Spiking neural network library
- **Weights & Biases**: Experiment tracking (optional)
- **Torchvision**: Computer vision utilities
- **Pytest**: Testing framework

## Dataset

**TUMTraf Event Dataset**: Synchronized RGB and event camera recordings from traffic scenarios.

- **Training**: 2,358 RGB frames / 295 event sequences (8 frames each)
- **Validation**: 318 RGB frames / 40 event sequences
- **Test Splits**: day, night_with_light_off, night_with_light_on
- **Resolution**: 482×442 pixels
- **Annotations**: Bounding boxes in OpenLabel format

Download instructions available at: [TUMTraf Dataset](https://innovation-mobility.com/en/project-providentia/a9-dataset/)

## Usage

### Quick Start

```bash
# Using the orchestration script
./run.sh preprocess --all          # Preprocess all data
./run.sh train-ann --epochs 100    # Train ANN on RGB
./run.sh train-snn --epochs 100    # Train SNN on events
./run.sh eval --model-path checkpoints/best.pth
```

### Full Pipeline Execution

```bash
# RGB pipeline (preprocess → train → evaluate)
./run.sh pipeline --rgb

# Event-based pipeline
./run.sh pipeline --eb

# Both pipelines sequentially
./run.sh pipeline-both

# Both pipelines in parallel (requires multi-GPU)
./run.sh pipeline-both --parallel
```

### Individual Training Commands

**ANN Training (RGB)**:
```bash
python scripts/train_ann_rgb.py \
    --epochs 100 \
    --batch-size 16 \
    --lr 1e-3 \
    --num-classes 6 \
    --device cuda \
    --wandb \
    --wandb-project neuromorph-vs-noise
```

**SNN Training (Event-Based)**:
```bash
python scripts/train_snn_eb.py \
    --epochs 100 \
    --batch-size 1 \
    --lr 1e-3 \
    --num-classes 6 \
    --beta 0.9 \
    --threshold 1.0 \
    --surrogate-slope 25.0 \
    --device cuda \
    --wandb
```

### Training Parameters

| Parameter | ANN Default | SNN Default | Description |
|-----------|-------------|-------------|-------------|
| `--epochs` | 100 | 100 | Number of training epochs |
| `--batch-size` | 16 | 1 | Batch size (SNN processes sequences) |
| `--lr` | 1e-3 | 1e-3 | Initial learning rate |
| `--momentum` | 0.9 | 0.9 | SGD momentum |
| `--weight-decay` | 5e-4 | 5e-4 | L2 regularization |
| `--beta` | - | 0.9 | SNN membrane decay rate |
| `--threshold` | - | 1.0 | SNN spike threshold |
| `--surrogate-slope` | - | 25.0 | Surrogate gradient slope |

## Model Architecture

### VGG11-SSD Overview

Both ANN and SNN implementations follow the VGG11 backbone architecture with SSD detection heads:

- **Backbone**: VGG11 (1-1-2-2-2 convolutional blocks)
- **Feature Maps**: 5 scales for multi-scale detection
- **Anchors**: 20,568 default boxes per image
- **Parameters**: ~17.8M (ANN), ~17.8M (SNN)
- **Output**: Classification scores + bounding box offsets

### Key Architectural Differences

| Component | ANN | SNN |
|-----------|-----|-----|
| Input Channels | 3 (RGB) | 2 (Event polarity) |
| Activation | ReLU | Leaky Integrate-and-Fire (LIF) |
| Temporal Processing | Single frame | 8-frame sequences |
| State Management | Stateless | Membrane potentials |
| Gradient Method | Standard backprop | Surrogate gradient BPTT |

## Loss Function

**SSD Multi-Task Loss**:
$$L = L_{cls} + \alpha L_{loc}$$

- **Classification Loss**: Cross-entropy with hard negative mining (3:1 neg:pos ratio)
- **Localization Loss**: Smooth L1 on matched anchor boxes
- **Anchor Matching**: IoU threshold = 0.5
- **Background Class**: Label 0 (object classes 1-6)

## Evaluation

The evaluation pipeline computes standard object detection metrics:

- **mAP** (mean Average Precision) at IoU thresholds [0.5, 0.75, 0.5:0.95]
- **Per-class AP** for each of 6 object categories
- **Inference latency** (frames per second)
- **Test conditions**: day, night_with_light_off, night_with_light_on

```bash
python scripts/evaluate.py \
    --model-path checkpoints/vgg11_ssd_ann_best.pth \
    --test-split day \
    --device cuda
```

## Testing

Comprehensive test suite with 63 tests covering all components:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_loss_and_anchors.py -v
```

See [tests/README.md](tests/README.md) for detailed test documentation.

## Experiment Tracking

Integration with Weights & Biases for experiment management:

```bash
# Enable W&B logging
python scripts/train_ann_rgb.py --wandb --wandb-project my-project

# Logged metrics:
# - train/loss, train/cls_loss, train/loc_loss
# - val/loss, val/cls_loss, val/loc_loss
# - learning_rate
# - Best model artifacts
```

## Results

*Results will be populated after training completion.*

Expected comparative analysis:
- **ANN**: High accuracy, standard latency
- **SNN**: Competitive accuracy, lower power consumption, temporal integration benefits

## Implementation Highlights

1. **Proper Anchor Matching**: IoU-based assignment with hard negative mining
2. **Temporal State Management**: SNN membrane potentials persist across 8-frame sequences
3. **Modular Design**: Separate loss, anchor, and detection utilities
4. **Comprehensive Testing**: 100% test coverage of critical components
5. **Reproducibility**: Deterministic training with checkpointing and W&B logging

## Known Limitations

- Fixed 8-frame sequence length for SNN processing
- Simplified 2-channel event representation (polarity approximation)
- Single-GPU training (multi-GPU support planned)

## Future Work

- [ ] Implement evaluation
- [ ] Implement noise generation pipeline (mainly black-box input perturbation simulating deployment conditions)
- [ ] Adapt to include other object detection rgb-eb dataset
- [ ] Adapt to use other architectures, loses and training algorithms
- [ ] Deployment to neuromorphic hardware (Intel Loihi, SpiNNaker or FPGA implementation)

## Citation

If you use this code in your research, please cite:

TBD

## License

TBD

## Acknowledgments

- TUMTraf Event Dataset team for providing high-quality synchronized data
- snnTorch developers for the spiking neural network framework
- PyTorch community for deep learning infrastructure

## Contact

For questions or collaboration inquiries, please open an issue or contact [jose-anotnio.rodriguez-assalone@epitech.eu].

---

**Last Updated**: January 2026  
**Status**: Active Development
