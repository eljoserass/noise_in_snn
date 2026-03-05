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

### DSEC Data + Simulation Utilities

For the DSEC-to-events workflow (R2 sync, grayscale RGB, imagecorruptions, v2e sweeps):

```bash
# Sync from Cloudflare R2 -> local
python scripts/r2_sync.py download \
  --env-file ../dsec_data_managing/.env \
  --remote-prefix dsec/train/zurich_city_02_b \
  --local-dir data/dsec_sample/zurich_city_02_b

# Build grayscale/corruptions and generate v2e events
python scripts/dsec_rgb_event_pipeline.py \
  --sequence-dir data/dsec_sample/zurich_city_02_b \
  --validate-rectification \
  --run-imagecorruptions \
  --run-v2e \
  --v2e-on-corruptions \
  --skip-existing

# Probe v2e timestamp scaling behavior
python scripts/v2e_timestamp_probe.py \
  --frames-dir outputs/v2e_timestamp_experiment/input_frames \
  --fps-list 2,10

# Batch over many sequences/splits
python scripts/dsec_batch_pipeline.py \
  --dsec-root data/dsec \
  --splits train,val,test \
  --jobs 4 \
  --extra-args "--run-imagecorruptions --run-v2e --v2e-on-corruptions --skip-existing"

# Night runs (3-machine scripts + nohup helper)
bash scripts/night_runs/machine_a_download.sh
bash scripts/night_runs/machine_b_train.sh
bash scripts/night_runs/machine_c_convert_loop.sh
bash scripts/night_runs/launch_nohup.sh scripts/night_runs/machine_a_download.sh logs/machine_a.log
```

### DSEC Training/Eval (No Timestep Repetition)

Dedicated DSEC training/evaluation entrypoints:

```bash
# ANN baseline (RGB; grayscale by default for fair comparison)
python scripts/train_ann_dsec.py \
  --dsec-root data/dsec \
  --train-split train \
  --val-split val \
  --class-ids 0,1,2,3,4,5,6,7 \
  --wandb

# SNN baseline (event sequences; one forward pass per frame, no frame repetition)
python scripts/train_snn_dsec.py \
  --dsec-root data/dsec \
  --train-split train \
  --val-split val \
  --event-source real \
  --event-relpath events/left/events.h5 \
  --sequence-length 8 \
  --sequence-stride 8 \
  --class-ids 0,1,2,3,4,5,6,7 \
  --wandb

# Evaluate checkpoints on DSEC test split(s)
python scripts/evaluate_dsec.py \
  --model-path checkpoints/vgg11_ssd_snn_dsec_best.pth \
  --model-type snn \
  --dsec-root data/dsec \
  --test-splits test
```

Legacy TUMTraf scripts are preserved under explicit names:

```bash
scripts/train_ann_tumtraf.py
scripts/train_snn_tumtraf.py
scripts/evaluate_tumtraf.py
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

The evaluation pipeline implements comprehensive object detection metrics with **grouped evaluation** for fair ANN vs SNN comparison:

### Metrics Computed

- **mAP** (mean Average Precision) at configurable IoU thresholds
  - Default: [0.5, 0.75]
  - Supports COCO-style mAP@[0.5:0.95:0.05]
- **Per-class AP** for each of 6 object categories with TP/FP counts
- **Inference Latency**: mean/std, frames per second
- **Energy Consumption** (framework available): FLOPs (ANN), spike counts (SNN)

### Grouped Evaluation Strategy

To ensure fair comparison between stateless ANN and temporal SNN:

- **ANN**: Processes each frame independently within 8-frame groups
- **SNN**: Processes 8-frame sequences with temporal membrane state integration
- **Both**: Evaluated on identical frames from the same sequences
- **mAP**: Computed across all individual frames from all groups

This approach ensures both models see the same visual data while respecting their different processing paradigms.

### Usage Examples

**Evaluate ANN on day test set:**
```bash
python scripts/evaluate.py \
    --model-path checkpoints/vgg11_ssd_ann_best.pth \
    --model-type ann \
    --test-split test/day \
    --conf-threshold 0.5 \
    --nms-threshold 0.5 \
    --iou-thresholds 0.5 0.75 \
    --device cuda
```

**Evaluate SNN on night test set:**
```bash
python scripts/evaluate.py \
    --model-path checkpoints/vgg11_ssd_snn_best.pth \
    --model-type snn \
    --test-split test/night_with_light_on \
    --beta 0.9 \
    --threshold 1.0 \
    --surrogate-slope 25.0 \
    --conf-threshold 0.5 \
    --nms-threshold 0.5 \
    --iou-thresholds 0.5 0.75 \
    --device cuda
```

**Batch evaluation (all test splits):**
```bash
# Using the orchestration script
./run.sh eval-all --model-path checkpoints/best.pth --model-type ann
```

### Evaluation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-path` | Required | Path to model checkpoint (.pth) |
| `--model-type` | Required | Model type: `ann` or `snn` |
| `--test-split` | `test/day` | Test split: `test/day`, `test/night_with_light_off`, `test/night_with_light_on` |
| `--conf-threshold` | 0.5 | Confidence threshold for detections |
| `--nms-threshold` | 0.5 | NMS IoU threshold |
| `--iou-thresholds` | [0.5, 0.75] | IoU thresholds for mAP computation |
| `--batch-size` | 1 | Batch size (recommend 1 for fair comparison) |
| `--output-dir` | `results/` | Directory to save evaluation JSON |
| `--measure-energy` | False | Enable energy consumption measurement |

### Output Format

**Console Output:**
```
==============================================================
EVALUATION RESULTS - ANN
==============================================================

Dataset: test/day
Model: checkpoints/vgg11_ssd_ann_best.pth

Images evaluated: 2358
Total predictions: 15240
Total ground truths: 18456

===================mAP Results===================
mAP (average): 0.4523
mAP@0.5: 0.5234
mAP@0.75: 0.3812

================Per-Class AP================

At IoU=0.5:
  BICYCLE     : AP=0.3421  (GT=1234, Pred=1456, TP= 845, FP= 611)
  BUS         : AP=0.5678  (GT= 234, Pred= 267, TP= 189, FP=  78)
  CAR         : AP=0.6234  (GT=8945, Pred=9123, TP=5678, FP=3445)
  ...

================Latency================
Mean: 12.34 ms
Std:  1.23 ms
FPS:  81.03
==============================================================

✓ Results saved to: results/eval_ann_test_day.json
```

**JSON Output** (`results/eval_ann_test_day.json`):
```json
{
  "mAP_avg": 0.4523,
  "mAP@0.5": 0.5234,
  "mAP@0.75": 0.3812,
  "per_class_AP@0.5": {
    "BICYCLE": {"ap": 0.3421, "num_gt": 1234, "num_pred": 1456, "tp": 845, "fp": 611},
    ...
  },
  "latency": {
    "mean_ms": 12.34,
    "std_ms": 1.23,
    "fps": 81.03
  },
  "num_images": 2358,
  "num_predictions": 15240,
  "num_ground_truths": 18456,
  "config": {...}
}
```

### Energy Consumption (Future Implementation)

Framework available for energy estimation:

- **ANN**: FLOPs counting via `ptflops` or `fvcore` libraries
- **SNN**: Spike operation counting with hooks in LIF neurons
- **Comparison**: Energy efficiency ratio (SNN spikes / ANN FLOPs)

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
- Dependant of the preprocessing of the dataset, processed directly in frames not event format. Temporal resolution is very low, which could hinder SNN advantages
- Labels on the dataset are not evenly distributed per classes accross both modalities

## Future Work

- [x] Implement evaluation with grouped comparison strategy
- [ ] Implement energy consumption measurement (FLOPs/spikes)
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
