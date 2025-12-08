# Change: Optimize Numerical IK Solver

## Why
The current numerical IK solver uses finite differences for Jacobian calculation, which is slow and inaccurate (fails to reach <1mm error).
The user requires high precision (1e-3mm) and high throughput (1M+ calls) for dataset generation.

## What Changes
- Replace finite difference Jacobian with PyTorch Autograd (exact and vectorized).
- Vectorize LMA and Jacobian update steps (remove per-batch loops).
- Tune convergence parameters (damping, step size) for higher accuracy.

## Impact
- Affected specs: `ik-solver` (added numerical solver requirements)
- Affected code: `mfik/robot/inverse_kinematics.py`

