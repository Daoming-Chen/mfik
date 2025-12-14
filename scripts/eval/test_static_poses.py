"""
Static Random Pose Test

Evaluates IK solver on random reachable poses.
"""

import argparse
from pathlib import Path

import torch
import numpy as np

from mfik.eval.inference import IKSolver
from mfik.eval.metrics import (
    compute_pose_error,
    compute_success_rate,
    compute_joint_statistics,
)
from mfik.eval.visualization import (
    plot_error_distribution,
    plot_workspace_heatmap,
    generate_performance_report,
)
from mfik.data.sampling import sample_random_configurations
from mfik.robot.forward_kinematics import ForwardKinematics
from mfik.robot.urdf import get_kinematic_chain


def generate_test_poses(urdf_path: str, n_samples: int = 10000, device: str = "cpu"):
    """Generate random reachable target poses."""
    print(f"Generating {n_samples} random test poses...")
    
    chain = get_kinematic_chain(urdf_path)
    fk = ForwardKinematics(chain, device=device)
    
    # Sample random valid joint configurations
    q_samples = sample_random_configurations(chain, n_samples)
    q_samples = torch.tensor(q_samples, dtype=torch.float32, device=device)
    
    # Compute FK to get target poses
    positions, quaternions = fk.compute(q_samples)
    target_poses = torch.cat([positions, quaternions], dim=-1)  # [N, 7]
    
    return target_poses, q_samples


def evaluate_solver(
    solver: IKSolver,
    test_poses: torch.Tensor,
    q_ground_truth: torch.Tensor,
    output_dir: Path,
):
    """Run comprehensive evaluation."""
    print(f"\nEvaluating {len(test_poses)} test poses...")
    
    # Random reference joints (not ground truth)
    q_refs = torch.rand_like(q_ground_truth) * 2 - 1  # Random in [-1, 1]
    
    # Solve IK
    print("Running inference...")
    q_pred = solver.solve(test_poses, q_refs)
    
    # Compute metrics
    print("Computing metrics...")
    pos_errors, rot_errors = compute_pose_error(q_pred, test_poses, solver.fk)
    
    pos_success, rot_success, combined_success = compute_success_rate(
        pos_errors, rot_errors
    )
    
    joint_stats = compute_joint_statistics(q_pred, solver.joint_limits)
    
    # Benchmark latency
    print("Benchmarking latency...")
    benchmark_results = solver.benchmark(
        test_poses[:100],
        q_refs[:100],
        warmup_steps=10,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Position Error: {pos_errors.mean().item() * 1000:.2f} mm "
          f"(median: {pos_errors.median().item() * 1000:.2f} mm, "
          f"95%: {torch.quantile(pos_errors, 0.95).item() * 1000:.2f} mm)")
    print(f"Rotation Error: {rot_errors.mean().item() * 180 / np.pi:.2f}° "
          f"(median: {rot_errors.median().item() * 180 / np.pi:.2f}°, "
          f"95%: {torch.quantile(rot_errors, 0.95).item() * 180 / np.pi:.2f}°)")
    print(f"\nSuccess Rate:")
    print(f"  Position: {pos_success * 100:.1f}%")
    print(f"  Rotation: {rot_success * 100:.1f}%")
    print(f"  Combined: {combined_success * 100:.1f}%")
    print(f"\nLatency: {benchmark_results['avg_latency'] * 1000:.3f} ms "
          f"(95%: {benchmark_results.get('pos_error_95', 0) * 1000:.2f} mm)")
    print(f"Throughput: {benchmark_results['throughput']:.1f} inferences/sec")
    print(f"\nJoint Violations: {joint_stats['total_violation_rate'] * 100:.2f}%")
    print("=" * 60 + "\n")
    
    # Generate plots
    print("Generating visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_error_distribution(
        pos_errors, rot_errors,
        save_path=str(output_dir / "error_distribution.png")
    )
    
    # Get positions for heatmap
    pred_positions, pred_quaternions = solver.fk.compute(q_pred)
    pred_poses = torch.cat([pred_positions, pred_quaternions], dim=-1)
    plot_workspace_heatmap(
        pred_poses[:, :3], pos_errors,
        plane="xy",
        save_path=str(output_dir / "workspace_heatmap_xy.png")
    )
    plot_workspace_heatmap(
        pred_poses[:, :3], pos_errors,
        plane="xz",
        save_path=str(output_dir / "workspace_heatmap_xz.png")
    )
    
    # Generate report
    results = {
        "pos_error_mean": pos_errors.mean().item(),
        "pos_error_median": pos_errors.median().item(),
        "pos_error_95": torch.quantile(pos_errors, 0.95).item(),
        "rot_error_mean": rot_errors.mean().item(),
        "rot_error_median": rot_errors.median().item(),
        "rot_error_95": torch.quantile(rot_errors, 0.95).item(),
        "success_rate": (pos_success, rot_success, combined_success),
        "avg_latency": benchmark_results['avg_latency'],
        "throughput": benchmark_results['throughput'],
        "joint_stats": joint_stats,
    }
    
    generate_performance_report(
        results,
        save_path=str(output_dir / "performance_report.md")
    )
    
    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Static random pose test")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--urdf", type=str, required=True,
                       help="Path to robot URDF")
    parser.add_argument("--n-samples", type=int, default=10000,
                       help="Number of test samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device (cpu or cuda)")
    parser.add_argument("--output-dir", type=str, default="results/static_test",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Load solver
    print(f"Loading solver from {args.checkpoint}...")
    solver = IKSolver.from_checkpoint(
        args.checkpoint,
        args.urdf,
        device=args.device,
    )
    
    # Generate test data
    test_poses, q_ground_truth = generate_test_poses(
        args.urdf,
        args.n_samples,
        device=args.device,
    )
    
    # Evaluate
    output_dir = Path(args.output_dir)
    evaluate_solver(solver, test_poses, q_ground_truth, output_dir)


if __name__ == "__main__":
    main()
