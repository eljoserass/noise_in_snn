# SNN Timestep Accumulation - Quick Start

## What Changed?

Added **frame-level timestep repetition** to improve SNN temporal integration. Each frame is now presented multiple times (T timesteps) to allow membrane potential to accumulate.

**Before**: Each frame shown once → only 8 timesteps total
**Now**: Each frame shown T times → 8 × T timesteps total (default: 80 timesteps)

## Quick Usage

### Training

```bash
# Default (10 timesteps per frame - balanced)
./run.sh train-snn --eb --epochs 50

# Higher quality (25 timesteps - literature standard)
./run.sh train-snn --eb --timesteps-per-frame 25 --beta 0.95 --epochs 50

# Fast experiments (5 timesteps)
./run.sh train-snn --eb --timesteps-per-frame 5 --beta 0.90 --epochs 50
```

### Evaluation

**IMPORTANT**: Match the timesteps-per-frame used during training!

```bash
python scripts/evaluate.py \
    --model-path checkpoints/vgg11_ssd_snn_best.pth \
    --model-type snn \
    --timesteps-per-frame 10  # Must match training!
```

### Full Pipeline

```bash
# Complete pipeline with 20 timesteps/frame
./run.sh pipeline --eb \
    --timesteps-per-frame 20 \
    --beta 0.93 \
    --epochs 100
```

## Recommended Starting Configurations

### For Quick Experiments
```bash
--timesteps-per-frame 5 --beta 0.90
# Training time: ~baseline
# Expected improvement: moderate
```

### Balanced (Recommended)
```bash
--timesteps-per-frame 10 --beta 0.92
# Training time: ~2× baseline
# Expected improvement: good (+2-3% mAP)
```

### High Performance
```bash
--timesteps-per-frame 25 --beta 0.95
# Training time: ~5× baseline
# Expected improvement: best (+3-5% mAP)
```

## What to Expect

### Performance
- Better temporal integration → improved detection
- More stable membrane dynamics
- Typical improvement: +2-5% mAP when going from T=5 to T=25
- Most benefit for: small objects, motion tracking, challenging conditions

### Training Time
- Linear scaling: T=10 takes ~2× longer than T=5
- T=25 takes ~5× longer than T=5
- Memory usage: ~1.5-2× higher

### When to Use Higher T
- You have GPU memory available (≥16GB for T=25)
- Current performance plateaus
- Working on challenging test sets (night scenes)
- Final training runs (after hyperparameter tuning)

### When to Use Lower T
- Quick prototyping/debugging
- Limited compute resources
- Initial experiments to tune other hyperparameters
- Sufficient performance with lower values

## Parameter Tuning

### Beta (Membrane Decay) Selection

Match beta to total timesteps (sequence_length × T):

| Your T Value | Total Timesteps | Recommended Beta |
|--------------|-----------------|------------------|
| 5            | 40              | 0.90 |
| 10           | 80              | 0.92 |
| 15           | 120             | 0.93 |
| 20           | 160             | 0.94 |
| 25           | 200             | 0.95 |
| 50           | 400             | 0.96 |

**Rule of thumb**: Higher T needs higher beta for longer memory

## Troubleshooting

**Problem**: Training too slow
→ Reduce `--timesteps-per-frame` to 5 or 7

**Problem**: Out of memory
→ Reduce `--timesteps-per-frame`
→ Ensure `--batch-size 1`

**Problem**: No improvement with higher T
→ Increase beta to match (see table above)
→ Train for more epochs (longer sequences need more training)
→ Check learning rate isn't too high

**Problem**: Evaluation results differ from training
→ Verify `--timesteps-per-frame` matches between train/eval
→ Check beta parameter matches

## Technical Details

See [`notes/docs/TIMESTEP_ACCUMULATION.md`](notes/docs/TIMESTEP_ACCUMULATION.md) for:
- Literature references
- Implementation details
- In-depth parameter guidelines
- Future enhancement ideas

## Example Training Run

```bash
# 1. Start with default to test
./run.sh train-snn --eb --epochs 10

# 2. If working well, increase for better performance
./run.sh train-snn --eb --timesteps-per-frame 20 --beta 0.94 --epochs 100

# 3. Evaluate with matching parameters
python scripts/evaluate.py \
    --model-path checkpoints/vgg11_ssd_snn_best.pth \
    --model-type snn \
    --timesteps-per-frame 20 \
    --beta 0.94
```
