# MeanFlow IK Training - Version 1

## Overview

This module implements the training pipeline for MeanFlow-based inverse kinematics models. The training uses a novel loss function based on the MeanFlow identity, computed efficiently using Jacobian-Vector Products (JVP).

## MeanFlow Identity

The network learns to satisfy the MeanFlow condition:

```
∂u/∂t + v_t · ∇_q u = 0
```

where:
- `u(q, r, t)` is the learned velocity field
- `v_t = (q₁ - q₀) / (t₁ - t₀)` is the flow velocity along the linear interpolation path
- `∂u/∂t` is the partial time derivative
- `v_t · ∇_q u` is the advection term (flow velocity times spatial gradient)

### Why MeanFlow?

Traditional Flow Matching requires ODE integration (10-100 steps) at inference time. MeanFlow allows **single-step inference** by learning the mean velocity field directly:

```
q_solution = q_reference + u(q_reference, 0, 1)
```

## Training Data Format

Training data should contain the following tensors:

```python
{
    "q": torch.Tensor,          # Current joint angles [N, DOF]
    "target_pose": torch.Tensor, # Target pose [N, 7] (x,y,z,qw,qx,qy,qz)
    "q_target": torch.Tensor,    # Target joint angles [N, DOF]
    "r": torch.Tensor,           # Reference time [N, 1], sampled from [0, 1]
    "t": torch.Tensor,           # Evaluation time [N, 1], sampled from [0, 1]
}
```

The data represents trajectories following linear interpolation:
```
q(t) = q₀ + (q_target - q₀) * (t - r) / (1 - r)
```

## Loss Computation

### 1. JVP-Based Loss

The loss is computed using the MeanFlow residual:

```python
loss = || ∂u/∂t + v_t · ∇_q u ||²
```

We use `torch.func.jvp` (Jacobian-Vector Product) to efficiently compute the total derivative:

```python
d/dt u = v_t · ∇_q u + ∂u/∂t
```

This requires only a single forward pass plus JVP computation (~16% overhead compared to standard backprop).

### 2. Adaptive Weighting

To handle the varying difficulty of samples, we use adaptive loss weighting:

```
w = 1 / (||q_target - q||² + c)^p
```

Parameters:
- `c = 1e-4`: Regularization constant (prevents division by zero)
- `p = 0.5`: Weight power (default)

Intuition: Samples closer to the target (smaller `||q_target - q||`) get higher weight, encouraging accurate convergence.

## Configuration

### Default Hyperparameters

```python
from mfik.train.v1 import TrainConfig

config = TrainConfig()
# Optimization
config.learning_rate = 1e-4
config.weight_decay = 1e-5
config.betas = (0.9, 0.999)

# Scheduling
config.warmup_steps = 1000
config.scheduler_type = "cosine"
config.min_lr = 1e-6

# Training
config.batch_size = 256
config.num_epochs = 100
config.gradient_clip = 1.0

# Loss
config.loss_weight_type = "adaptive"
config.adaptive_weight_c = 1e-4
config.adaptive_weight_p = 0.5
```

### Robot-Specific Configs

```python
# Panda (7-DOF)
config = TrainConfig.panda_default()

# UR10 (6-DOF)
config = TrainConfig.ur10_default()

# Quick test
config = TrainConfig.quick_test()
```

## Usage

### Training from Script

```bash
# Train Panda model
python scripts/train.py \
    --robot panda \
    --data_path data/panda_train.pt \
    --output_dir outputs/panda

# Train UR10 model
python scripts/train.py \
    --robot ur10 \
    --data_path data/ur10_train.pt \
    --output_dir outputs/ur10

# Quick test with dummy data
python scripts/train.py --quick_test
```

### Training from Code

```python
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from mfik.model.v1 import MeanFlowNet, ModelConfig
from mfik.train.v1 import TrainConfig, MeanFlowTrainer

# Create model
model_config = ModelConfig.panda_default()
model = MeanFlowNet(model_config)

# Create trainer
train_config = TrainConfig.panda_default()
trainer = MeanFlowTrainer(model=model, config=train_config)

# Load data
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)

# Training loop
for epoch in range(train_config.num_epochs):
    for batch in train_loader:
        metrics = trainer.train_step(batch)
        trainer.global_step += 1

        # Logging, validation, checkpointing...
```

## Training Pipeline

### 1. Data Preparation

Generate training data using the data module:

```bash
python scripts/generate_data.py \
    --robot panda \
    --num_samples 1000000 \
    --output data/panda_train.pt
```

### 2. Model Training

```bash
python scripts/train.py \
    --robot panda \
    --data_path data/panda_train.pt \
    --batch_size 256 \
    --num_epochs 100 \
    --output_dir outputs/panda
```

### 3. Monitoring

Training metrics are logged to TensorBoard:

```bash
tensorboard --logdir outputs/panda/logs
```

Key metrics:
- `train/loss`: MeanFlow loss
- `train/meanflow_residual`: Residual norm
- `train/velocity_norm`: Predicted velocity magnitude
- `train/grad_norm`: Gradient norm
- `train/lr`: Learning rate
- `val/val_loss`: Validation loss

### 4. Checkpoint Management

Checkpoints are saved to `outputs/{robot}/checkpoints/`:
- `best_model.pth`: Best validation loss
- `checkpoint_epoch_{n}.pth`: End of each epoch
- `checkpoint_step_{n}.pth`: Every N steps
- `final_model.pth`: Final trained model

## Learning Rate Schedule

### Cosine Annealing with Warmup (Default)

```
       base_lr ─┐
                │ ╱╲
                │╱  ╲
                │     ╲___
warmup   decay  │          ╲___
    ├──────┼────┤               ╲___
0   │      │    │                    ╲___ min_lr
    └──────┴────┴────────────────────────────────→ steps
         warmup_steps          total_steps
```

- Linear warmup from 0 to `base_lr` over `warmup_steps`
- Cosine decay from `base_lr` to `min_lr` over remaining steps

### Linear Decay with Warmup

```
       base_lr ─┐
                │ ╱╲
                │╱  ╲
                │     ╲
warmup   decay  │      ╲
    ├──────┼────┤        ╲
0   │      │    │          ╲
    └──────┴────┴────────────╲_____ min_lr
         warmup_steps           total_steps
```

## Optimization Details

### AdamW Optimizer

- **Why AdamW**: Decoupled weight decay provides better regularization than L2 penalty
- **Learning rate**: `1e-4` (typical for transformer-like architectures)
- **Weight decay**: `1e-5` (light regularization)
- **Betas**: `(0.9, 0.999)` (standard Adam parameters)

### Gradient Clipping

```python
gradient_clip = 1.0  # Clip gradients to max norm of 1.0
```

Prevents gradient explosion during early training or when encountering difficult samples.

### Mixed Precision Training

```python
use_amp = True  # Automatic Mixed Precision
```

Speeds up training by ~2x with minimal accuracy loss. Uses FP16 for forward/backward pass, FP32 for parameter updates.

## Advanced Features

### Resume Training

```bash
python scripts/train.py \
    --robot panda \
    --data_path data/panda_train.pt \
    --resume_from outputs/panda/checkpoints/checkpoint_epoch_50.pth
```

### Custom Configuration

```python
config = TrainConfig()
config.batch_size = 512  # Larger batch
config.learning_rate = 2e-4  # Higher LR
config.gradient_clip = 2.0  # Less aggressive clipping
config.loss_weight_type = "uniform"  # Disable adaptive weighting
```

### Multiple GPUs (Future)

```python
# Wrap model with DataParallel or DistributedDataParallel
model = torch.nn.DataParallel(model)
```

## Troubleshooting

### Loss Not Decreasing

1. **Check data**: Verify data format and ranges
2. **Reduce learning rate**: Try `1e-5` instead of `1e-4`
3. **Increase warmup**: More gradual warmup helps stability
4. **Disable AMP**: Mixed precision can cause numerical issues

### NaN Loss

1. **Gradient clipping**: Ensure `gradient_clip > 0`
2. **Check data**: Look for NaN/Inf in training data
3. **Reduce learning rate**: Start with `1e-5`
4. **Increase regularization**: Higher `weight_decay`

### Slow Training

1. **Increase batch size**: Use largest batch that fits in memory
2. **Enable AMP**: `use_amp = True` for ~2x speedup
3. **More workers**: Increase `num_workers` for data loading
4. **Profile code**: Use PyTorch profiler to find bottlenecks

## Performance Expectations

### Training Time

| Robot | Dataset Size | Batch Size | GPU       | Time/Epoch | Total Time |
|-------|--------------|------------|-----------|------------|------------|
| Panda | 1M samples   | 256        | RTX 4090  | ~5 min     | ~8 hours   |
| UR10  | 1M samples   | 256        | RTX 4090  | ~5 min     | ~8 hours   |

### Convergence

- **Initial loss**: ~1-10 (depends on data scale)
- **Converged loss**: ~0.001-0.01
- **Epochs to converge**: 50-100

### Memory Usage

- **Model**: ~70 MB (17.9M parameters in FP32)
- **Optimizer**: ~140 MB (Adam states)
- **Batch (256)**: ~200 MB
- **Total**: ~500 MB (easily fits on 8GB GPUs)

## Version History

- **v1.0.0** (2024-12): Initial release
  - MeanFlow loss with JVP computation
  - AdamW optimizer with cosine scheduling
  - Adaptive loss weighting
  - TensorBoard logging
  - Checkpoint management

## Future Improvements (v2+)

- Distributed training (multi-GPU)
- On-the-fly data augmentation
- Curriculum learning (easy → hard samples)
- Uncertainty quantification
- Online hard example mining
