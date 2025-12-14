"""
Trajectory Tracking Test

Evaluates IK solver on continuous trajectories (line, circle, spiral).
"""

import argparse
from pathlib import Path

import torch
import numpy as np

from mfik.eval.inference import IKSolver
from mfik.eval.metrics import (
    compute_pose_error,
    compute_trajectory_smoothness,
)
from mfik.eval.visualization import (
    plot_trajectory_tracking,
    plot_error_distribution,
)


def generate_line_trajectory(start, end, n_points=100):
    """Generate linear trajectory in Cartesian space."""
    t = torch.linspace(0, 1, n_points)
    positions = start[:3].unsqueeze(0) + t.unsqueeze(1) * (end[:3] - start[:3]).unsqueeze(0)
    
    # Keep orientation constant (use start orientation)
    orientations = start[3:].unsqueeze(0).expand(n_points, -1)
    
    return torch.cat([positions, orientations], dim=1)


def generate_circle_trajectory(center, radius, normal, n_points=100):
    """Generate circular trajectory."""
    theta = torch.linspace(0, 2 * np.pi, n_points)
    
    # Generate circle in XY plane then rotate
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    z = torch.zeros_like(x)
    
    positions = torch.stack([x, y, z], dim=1) + center[:3]
    
    # Fixed orientation
    orientations = torch.tensor([1, 0, 0, 0]).unsqueeze(0).expand(n_points, -1)
    
    return torch.cat([positions, orientations], dim=1)


def generate_spiral_trajectory(center, radius, height, n_turns=2, n_points=200):
    """Generate spiral trajectory."""
    theta = torch.linspace(0, n_turns * 2 * np.pi, n_points)
    
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    z = torch.linspace(0, height, n_points)
    
    positions = torch.stack([x, y, z], dim=1) + center[:3]
    
    # Fixed orientation
    orientations = torch.tensor([1, 0, 0, 0]).unsqueeze(0).expand(n_points, -1)
    
    return torch.cat([positions, orientations], dim=1)


def evaluate_trajectory(
    solver: IKSolver,
    trajectory: torch.Tensor,
    trajectory_name: str,
    output_dir: Path,
):
    """Evaluate IK solver on a trajectory."""
    print(f"\nEvaluating {trajectory_name} trajectory ({len(trajectory)} points)...")
    
    # Solve trajectory
    q_trajectory, metadata = solver.solve_trajectory(
        trajectory,
        check_smoothness=True,
        smoothness_threshold=0.1,
    )
    
    # Compute errors
    pos_errors, rot_errors = compute_pose_error(
        q_trajectory, trajectory, solver.fk
    )
    
    # Compute smoothness
    smoothness = compute_trajectory_smoothness(q_trajectory, dt=0.01)
    
    # Print results
    print(f"  Position Error: {pos_errors.mean().item() * 1000:.2f} mm "
          f"(max: {pos_errors.max().item() * 1000:.2f} mm)")
    print(f"  Rotation Error: {rot_errors.mean().item() * 180 / np.pi:.2f}° "
          f"(max: {rot_errors.max().item() * 180 / np.pi:.2f}°)")
    print(f"  Avg Time per Frame: {metadata['avg_time_per_frame'] * 1000:.2f} ms")
    print(f"  Smoothness Violations: {len(metadata['smoothness_violations'])}")
    print(f"  Max Velocity: {smoothness['max_velocity']:.3f} rad/s")
    print(f"  Max Acceleration: {smoothness['max_acceleration']:.3f} rad/s²")
    
    # Visualize
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_trajectory_tracking(
        None,  # No ground truth
        q_trajectory,
        joint_names=[f"Joint {i+1}" for i in range(q_trajectory.shape[1])],
        save_path=str(output_dir / f"{trajectory_name}_joints.png")
    )
    
    plot_error_distribution(
        pos_errors, rot_errors,
        save_path=str(output_dir / f"{trajectory_name}_errors.png")
    )
    
    return {
        "pos_errors": pos_errors,
        "rot_errors": rot_errors,
        "smoothness": smoothness,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Trajectory tracking test")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--urdf", type=str, required=True,
                       help="Path to robot URDF")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device (cpu or cuda)")
    parser.add_argument("--output-dir", type=str, default="results/trajectory_test",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Load solver
    print(f"Loading solver from {args.checkpoint}...")
    solver = IKSolver.from_checkpoint(
        args.checkpoint,
        args.urdf,
        device=args.device,
    )
    
    output_dir = Path(args.output_dir)
    
    # Test 1: Linear trajectory
    print("\n" + "=" * 60)
    print("TEST 1: Linear Trajectory")
    print("=" * 60)
    
    start_pose = torch.tensor([0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], device=args.device)
    end_pose = torch.tensor([0.3, 0.3, 0.5, 1.0, 0.0, 0.0, 0.0], device=args.device)
    line_traj = generate_line_trajectory(start_pose, end_pose, n_points=100)
    
    evaluate_trajectory(solver, line_traj, "linear", output_dir / "linear")
    
    # Test 2: Circular trajectory
    print("\n" + "=" * 60)
    print("TEST 2: Circular Trajectory")
    print("=" * 60)
    
    center = torch.tensor([0.4, 0.0, 0.5], device=args.device)
    circle_traj = generate_circle_trajectory(
        torch.cat([center, torch.tensor([1, 0, 0, 0], device=args.device)]),
        radius=0.1,
        normal=torch.tensor([0, 0, 1], device=args.device),
        n_points=100,
    )
    
    evaluate_trajectory(solver, circle_traj, "circular", output_dir / "circular")
    
    # Test 3: Spiral trajectory
    print("\n" + "=" * 60)
    print("TEST 3: Spiral Trajectory")
    print("=" * 60)
    
    spiral_traj = generate_spiral_trajectory(
        torch.cat([center, torch.tensor([1, 0, 0, 0], device=args.device)]),
        radius=0.1,
        height=0.2,
        n_turns=2,
        n_points=200,
    )
    
    evaluate_trajectory(solver, spiral_traj, "spiral", output_dir / "spiral")
    
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
