"""
Evaluation Metrics

Provides functions for computing IK solution quality metrics:
- Position and orientation errors
- Success rate statistics
- Inference latency measurement
"""

import time
from typing import Tuple, Optional, Callable

import torch
import numpy as np

from ..robot.forward_kinematics import ForwardKinematics


def compute_pose_error(
    q_pred: torch.Tensor,
    target_pose: torch.Tensor,
    fk: ForwardKinematics,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute position and orientation errors between predicted joints and target pose.
    
    Args:
        q_pred: Predicted joint angles [batch, n_joints] or [n_joints]
        target_pose: Target end-effector pose [batch, 7] or [7] (pos + quat)
        fk: Forward kinematics calculator
        
    Returns:
        Position errors [batch] in meters
        Orientation errors [batch] in radians
    """
    # Ensure batch dimension
    if q_pred.dim() == 1:
        q_pred = q_pred.unsqueeze(0)
        target_pose = target_pose.unsqueeze(0)
    
    batch_size = q_pred.shape[0]
    
    # Compute FK
    pred_positions, pred_quaternions = fk.compute(q_pred)
    pred_pose = torch.cat([pred_positions, pred_quaternions], dim=-1)  # [batch, 7]
    
    # Position error (Euclidean distance)
    pos_pred = pred_pose[:, :3]
    pos_target = target_pose[:, :3]
    pos_errors = torch.norm(pos_pred - pos_target, dim=1)
    
    # Orientation error (quaternion distance)
    quat_pred = pred_pose[:, 3:]
    quat_target = target_pose[:, 3:]
    rot_errors = quaternion_distance(quat_pred, quat_target)
    
    return pos_errors, rot_errors


def quaternion_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Compute angular distance between two quaternions.
    
    Args:
        q1: First quaternion [batch, 4] (w, x, y, z)
        q2: Second quaternion [batch, 4]
        
    Returns:
        Angular distance in radians [batch]
    """
    # Normalize quaternions
    q1 = q1 / torch.norm(q1, dim=1, keepdim=True)
    q2 = q2 / torch.norm(q2, dim=1, keepdim=True)
    
    # Compute inner product
    dot = torch.abs(torch.sum(q1 * q2, dim=1))
    
    # Clamp to avoid numerical issues with acos
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # Angular distance: theta = 2 * arccos(|q1 · q2|)
    angle = 2.0 * torch.acos(dot)
    
    return angle


def compute_success_rate(
    pos_errors: torch.Tensor,
    rot_errors: torch.Tensor,
    pos_threshold: float = 0.005,  # 5mm
    rot_threshold: float = 0.0873,  # 5 degrees
) -> Tuple[float, float, float]:
    """
    Compute success rates based on error thresholds.
    
    Args:
        pos_errors: Position errors [N] in meters
        rot_errors: Rotation errors [N] in radians
        pos_threshold: Position error threshold in meters
        rot_threshold: Rotation error threshold in radians
        
    Returns:
        Position success rate (0-1)
        Rotation success rate (0-1)
        Combined success rate (both satisfied)
    """
    pos_success = (pos_errors <= pos_threshold).float().mean().item()
    rot_success = (rot_errors <= rot_threshold).float().mean().item()
    
    # Combined: both conditions must be satisfied
    combined_success = (
        (pos_errors <= pos_threshold) & (rot_errors <= rot_threshold)
    ).float().mean().item()
    
    return pos_success, rot_success, combined_success


def measure_latency(
    inference_fn: Callable,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = "cpu",
) -> dict:
    """
    Measure inference latency statistics.
    
    Args:
        inference_fn: Function to measure (should return output)
        num_iterations: Number of timing iterations
        warmup_iterations: Number of warmup iterations
        device: Device for synchronization ('cpu' or 'cuda')
        
    Returns:
        Dictionary with latency statistics (mean, median, std, min, max) in ms
    """
    # Warmup
    for _ in range(warmup_iterations):
        _ = inference_fn()
    
    # Synchronize if using GPU
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    
    # Measure
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = inference_fn()
        
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }


def compute_joint_statistics(
    q_solutions: torch.Tensor,
    joint_limits: torch.Tensor,
) -> dict:
    """
    Compute statistics about joint configurations.
    
    Args:
        q_solutions: Joint angles [batch, n_joints]
        joint_limits: Joint limits [n_joints, 2] (min, max)
        
    Returns:
        Dictionary with joint statistics
    """
    n_joints = q_solutions.shape[1]
    batch_size = q_solutions.shape[0]
    
    # Check limit violations
    violations = (
        (q_solutions < joint_limits[:, 0]) | 
        (q_solutions > joint_limits[:, 1])
    )
    violation_rate = violations.float().mean(dim=0)  # Per joint
    total_violation_rate = violations.any(dim=1).float().mean().item()
    
    # Joint range utilization
    joint_ranges = joint_limits[:, 1] - joint_limits[:, 0]
    utilized_ranges = (q_solutions.max(dim=0)[0] - q_solutions.min(dim=0)[0])
    range_utilization = (utilized_ranges / joint_ranges).mean().item()
    
    return {
        "violation_rate_per_joint": violation_rate.tolist(),
        "total_violation_rate": total_violation_rate,
        "range_utilization": range_utilization,
        "mean_per_joint": q_solutions.mean(dim=0).tolist(),
        "std_per_joint": q_solutions.std(dim=0).tolist(),
    }


def compute_trajectory_smoothness(
    trajectory: torch.Tensor,
    dt: float = 1.0,
) -> dict:
    """
    Compute smoothness metrics for a joint trajectory.
    
    Args:
        trajectory: Joint trajectory [T, n_joints]
        dt: Time step between frames
        
    Returns:
        Dictionary with smoothness metrics
    """
    T, n_joints = trajectory.shape
    
    if T < 2:
        raise ValueError("Trajectory must have at least 2 frames")
    
    # Velocity (first derivative)
    velocity = (trajectory[1:] - trajectory[:-1]) / dt  # [T-1, n_joints]
    
    # Acceleration (second derivative)
    if T > 2:
        acceleration = (velocity[1:] - velocity[:-1]) / dt  # [T-2, n_joints]
    else:
        acceleration = torch.zeros(1, n_joints)
    
    # Jerk (third derivative)
    if T > 3:
        jerk = (acceleration[1:] - acceleration[:-1]) / dt  # [T-3, n_joints]
    else:
        jerk = torch.zeros(1, n_joints)
    
    # Compute statistics
    return {
        "max_velocity": velocity.abs().max().item(),
        "mean_velocity": velocity.abs().mean().item(),
        "max_acceleration": acceleration.abs().max().item(),
        "mean_acceleration": acceleration.abs().mean().item(),
        "max_jerk": jerk.abs().max().item(),
        "mean_jerk": jerk.abs().mean().item(),
        "velocity_per_joint": velocity.abs().max(dim=0)[0].tolist(),
        "acceleration_per_joint": acceleration.abs().max(dim=0)[0].tolist(),
    }


def compute_workspace_coverage(
    positions: torch.Tensor,
    grid_resolution: float = 0.05,
) -> dict:
    """
    Compute workspace coverage statistics.
    
    Args:
        positions: End-effector positions [N, 3]
        grid_resolution: Grid cell size in meters
        
    Returns:
        Dictionary with coverage statistics
    """
    # Compute workspace bounds
    min_pos = positions.min(dim=0)[0]
    max_pos = positions.max(dim=0)[0]
    workspace_volume = torch.prod(max_pos - min_pos).item()
    
    # Discretize positions into grid
    grid_coords = ((positions - min_pos) / grid_resolution).long()
    
    # Count unique cells
    unique_cells = torch.unique(grid_coords, dim=0).shape[0]
    
    # Theoretical grid size
    grid_dims = ((max_pos - min_pos) / grid_resolution).ceil().long()
    total_cells = torch.prod(grid_dims).item()
    
    coverage_ratio = unique_cells / max(total_cells, 1)
    
    return {
        "workspace_volume_m3": workspace_volume,
        "min_position": min_pos.tolist(),
        "max_position": max_pos.tolist(),
        "unique_cells": unique_cells,
        "total_cells": int(total_cells),
        "coverage_ratio": coverage_ratio,
    }
