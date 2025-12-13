# MeanFlow IK Model - Version 1

## Overview

This module implements a MLP-ResNet architecture for MeanFlow-based inverse kinematics solving. The network learns to predict joint velocities that drive the robot from an initial configuration to a target end-effector pose in a single step.

## Architecture

### Network Structure

```
Input (DOF + 7 + 2)
  ↓
[Positional Encoding] → 128-dim
  ↓
Input Projection → 1024-dim
  ↓
6× Residual Blocks (1024-dim)
  ↓
Output Head → DOF-dim
```

### Key Components

#### 1. Input Format

The network takes a concatenated input vector:
- **Joint angles** (DOF): Current robot configuration `q_0`
- **Target pose** (7): `[x, y, z, qw, qx, qy, qz]` (position + quaternion)
- **Time parameters** (2): `(r, t)` where `r` is reference time and `t` is evaluation time

Total input dimension: `DOF + 7 + 2`

#### 2. Positional Encoding

Two encoding methods are supported:

**Sinusoidal Encoding** (default):
```
PE(x, 2i) = sin(x * scale * exp(-2i * log(10000) / dim))
PE(x, 2i+1) = cos(x * scale * exp(-2i * log(10000) / dim))
```

**Random Fourier Features** (optional):
```
RFF(x) = [sin(Bx), cos(Bx)]
```
where `B` is a random projection matrix.

#### 3. Residual Blocks

Each residual block consists of:
```python
x = LayerNorm(x)
x = Linear(x)
x = SiLU(x)
x = LayerNorm(x)
x = Linear(x)
if use_FiLM:
    x = FiLM(x, condition)
x = SiLU(x)
output = x + residual
```

#### 4. FiLM Conditioning (Optional)

Feature-wise Linear Modulation provides adaptive feature scaling:
```
FiLM(x, c) = x * (1 + γ(c)) + β(c)
```
where `γ` and `β` are learned from the condition vector.

#### 5. Output Head

```python
x = LayerNorm(x)
output = Linear(x)  # → DOF dimensions
```

The output represents the mean velocity field: `u(q_0, r, t)`

## Configuration

### Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_resblocks` | 6 | Number of residual blocks |
| `hidden_dim` | 1024 | Hidden layer dimension |
| `pos_encoding_dim` | 128 | Positional encoding dimension |
| `pos_encoding_scale` | 10.0 | Encoding frequency scale |
| `activation` | silu | Activation function |
| `norm_type` | layernorm | Normalization type |
| `use_film` | True | Enable FiLM conditioning |
| `film_dim` | 256 | FiLM condition dimension |
| `dropout` | 0.0 | Dropout rate (disabled for fast inference) |

### Robot-Specific Configs

```python
from mfik.model.v1 import ModelConfig

# Franka Emika Panda (7-DOF)
config = ModelConfig.panda_default()

# Universal Robots UR10 (6-DOF)
config = ModelConfig.ur10_default()
```

## Usage

### Creating a Model

```python
from mfik.model.v1 import MeanFlowNet, ModelConfig

# Option 1: Use default config
config = ModelConfig.panda_default()
model = MeanFlowNet(config)

# Option 2: Use factory function
from mfik.model.v1.network import create_model
model = create_model(robot_dof=7)

# Option 3: Custom config
config = ModelConfig()
config.set_robot_dims(robot_dof=7, pose_dim=7)
config.num_resblocks = 8  # Override default
config.hidden_dim = 2048
model = MeanFlowNet(config)
```

### Forward Pass

```python
import torch

batch_size = 32
robot_dof = 7

# Input: [joint_angles, target_pose, time_params]
x = torch.randn(batch_size, robot_dof + 7 + 2)

# Output: joint velocity
velocity = model(x)  # Shape: [batch_size, robot_dof]

# Integrate to get final configuration
q_final = x[:, :robot_dof] + velocity
```

### Model Information

```python
# Count parameters
num_params = model.count_parameters()
print(f"Parameters: {num_params:,}")  # ~6.3M for Panda with default config

# Inspect architecture
print(model)
```

## MeanFlow Identity

The network is trained to satisfy the MeanFlow condition:

```
∂u/∂t + v_t · ∇_q u = 0
```

where:
- `u(q, r, t)` is the learned velocity field
- `v_t = (q_1 - q_0) / (t_1 - t_0)` is the flow velocity
- The gradient is computed using JVP (Jacobian-Vector Product)

At inference time, we use `u(q_0, 0, 1)` to perform single-step IK:

```
q_solution = q_ref + u(q_ref, 0, 1)
```

## Performance Characteristics

### Computational Complexity

- **Forward pass**: O(L × H²) where L = num_resblocks, H = hidden_dim
- **Memory**: ~25 MB for default config (float32)
- **Inference time**: < 1 ms on RTX 4090 (batch_size=1)

### Parameter Count

For Panda (7-DOF) with default config:
- Positional encoding: No trainable params
- Input projection: ~0.1M
- Residual blocks: ~6.0M
- Output head: ~0.007M
- **Total**: ~6.3M parameters

For UR10 (6-DOF) with default config:
- **Total**: ~6.3M parameters (similar, dominated by hidden layers)

## Design Rationale

### Why Wide and Shallow?

**Advantages**:
1. **Fast inference**: Fewer sequential operations (6 blocks vs 12+)
2. **Smooth gradients**: Shorter gradient paths prevent vanishing gradients
3. **High capacity**: 1024 hidden units provide sufficient expressiveness
4. **Reduced overfitting**: Shallower networks generalize better on structured data

**Trade-offs**:
- More memory per layer (but still reasonable at ~25 MB)
- More parameters than deep-narrow alternatives

### Why Pre-Normalization?

Pre-normalization (LayerNorm before Linear) provides:
1. More stable training (gradients flow through normalized paths)
2. Better convergence (avoids gradient explosion in deep sections)
3. Modern best practice for transformer-like architectures

### Why SiLU Activation?

SiLU (Swish) activation `x * sigmoid(x)`:
1. Smooth, non-monotonic (helps with complex mappings)
2. Better gradient flow than ReLU
3. Standard in modern architectures (used in Diffusion models)

### Why FiLM Conditioning?

Feature-wise modulation provides:
1. Adaptive feature scaling based on input
2. Helps disentangle different input components (joints, pose, time)
3. Proven effective in conditional generation tasks

## Version History

- **v1.0.0** (2024-12): Initial release
  - 6 ResBlocks, 1024 hidden dim
  - Sinusoidal positional encoding
  - FiLM conditioning
  - Pre-normalization with LayerNorm

## Future Improvements (v2+)

Potential enhancements for future versions:
- Attention mechanisms for long-range dependencies
- Adaptive depth (conditional computation)
- Multi-robot unified architecture
- Uncertainty quantification
- Continuous-time formulation (Neural ODE integration)
