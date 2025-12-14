"""
Numerical IK Comparison

Compares MeanFlow IK solver against numerical iterative methods.
"""

import argparse
from pathlib import Path
import time

import torch
import numpy as np

from mfik.eval.inference import IKSolver
from mfik.eval.metrics import compute_pose_error, compute_success_rate
from mfik.eval.visualization import plot_comparison
from mfik.robot.inverse_kinematics import InverseKinematics
from mfik.robot.urdf import get_kinematic_chain
from mfik.robot.forward_kinematics import ForwardKinematics
from mfik.data.sampling import sample_random_configurations


def evaluate_numerical_ik(
    ik_solver: InverseKinematics,
    test_poses: torch.Tensor,
    q_init: torch.Tensor,
    method: str = "jacobian_pseudoinverse",
    max_iterations: int = 100,
):
    """Evaluate numerical IK method."""
    print(f"Evaluating numerical IK ({method})...")
    
    n_samples = len(test_poses)
    q_solutions = torch.zeros(n_samples, ik_solver.fk.n_joints, device=test_poses.device)
    latencies = []
    success_count = 0
    
    for i in range(n_samples):
        start_time = time.perf_counter()
        
        try:
            q_sol, converged = ik_solver.solve(
                test_poses[i],
                q_init[i],
                method=method,
                max_iterations=max_iterations,
                tolerance=1e-4,
            )
            
            q_solutions[i] = q_sol
            if converged:
                success_count += 1
        except Exception as e:
            # If IK fails, keep zero solution
            pass
        
        elapsed = time.perf_counter() - start_time
        latencies.append(elapsed * 1000)  # Convert to ms
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_samples} samples...")
    
    latencies = np.array(latencies)
    
    return q_solutions, latencies, success_count / n_samples


def main():
    parser = argparse.ArgumentParser(description="Compare with numerical IK")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to MeanFlow model checkpoint")
    parser.add_argument("--urdf", type=str, required=True,
                       help="Path to robot URDF")
    parser.add_argument("--n-samples", type=int, default=1000,
                       help="Number of test samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device (cpu or cuda)")
    parser.add_argument("--output-dir", type=str, default="results/comparison",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Load MeanFlow solver
    print(f"Loading MeanFlow solver from {args.checkpoint}...")
    meanflow_solver = IKSolver.from_checkpoint(
        args.checkpoint,
        args.urdf,
        device=args.device,
    )
    
    # Setup numerical IK
    chain = get_kinematic_chain(args.urdf)
    fk = ForwardKinematics(chain, device=args.device)
    numerical_ik = InverseKinematics(fk)
    
    # Generate test poses
    print(f"\nGenerating {args.n_samples} random test poses...")
    q_samples = sample_random_configurations(chain, args.n_samples)
    q_samples = torch.tensor(q_samples, dtype=torch.float32, device=args.device)
    positions, quaternions = fk.compute(q_samples)
    test_poses = torch.cat([positions, quaternions], dim=-1)  # [N, 7]
    
    # Random initial guesses
    q_init = torch.rand_like(q_samples) * 2 - 1
    
    # Evaluate MeanFlow
    print("\n" + "=" * 60)
    print("EVALUATING MEANFLOW IK")
    print("=" * 60)
    
    start_time = time.time()
    q_meanflow = meanflow_solver.solve(test_poses, q_init)
    meanflow_time = time.time() - start_time
    
    pos_errors_mf, rot_errors_mf = compute_pose_error(q_meanflow, test_poses, fk)
    pos_success_mf, rot_success_mf, combined_success_mf = compute_success_rate(
        pos_errors_mf, rot_errors_mf
    )
    
    print(f"Position Error: {pos_errors_mf.mean().item() * 1000:.2f} mm")
    print(f"Rotation Error: {rot_errors_mf.mean().item() * 180 / np.pi:.2f}°")
    print(f"Success Rate: {combined_success_mf * 100:.1f}%")
    print(f"Total Time: {meanflow_time:.2f} s")
    print(f"Avg Latency: {meanflow_time / args.n_samples * 1000:.3f} ms")
    
    # Evaluate Jacobian Pseudoinverse
    print("\n" + "=" * 60)
    print("EVALUATING JACOBIAN PSEUDOINVERSE")
    print("=" * 60)
    
    q_jacobian, latencies_jac, success_jac = evaluate_numerical_ik(
        numerical_ik, test_poses, q_init,
        method="jacobian_pseudoinverse",
        max_iterations=100,
    )
    
    pos_errors_jac, rot_errors_jac = compute_pose_error(q_jacobian, test_poses, fk)
    pos_success_jac, rot_success_jac, combined_success_jac = compute_success_rate(
        pos_errors_jac, rot_errors_jac
    )
    
    print(f"Position Error: {pos_errors_jac.mean().item() * 1000:.2f} mm")
    print(f"Rotation Error: {rot_errors_jac.mean().item() * 180 / np.pi:.2f}°")
    print(f"Success Rate: {combined_success_jac * 100:.1f}%")
    print(f"Avg Latency: {np.mean(latencies_jac):.3f} ms")
    
    # Evaluate Damped Least Squares
    print("\n" + "=" * 60)
    print("EVALUATING DAMPED LEAST SQUARES")
    print("=" * 60)
    
    q_dls, latencies_dls, success_dls = evaluate_numerical_ik(
        numerical_ik, test_poses, q_init,
        method="damped_least_squares",
        max_iterations=100,
    )
    
    pos_errors_dls, rot_errors_dls = compute_pose_error(q_dls, test_poses, fk)
    pos_success_dls, rot_success_dls, combined_success_dls = compute_success_rate(
        pos_errors_dls, rot_errors_dls
    )
    
    print(f"Position Error: {pos_errors_dls.mean().item() * 1000:.2f} mm")
    print(f"Rotation Error: {rot_errors_dls.mean().item() * 180 / np.pi:.2f}°")
    print(f"Success Rate: {combined_success_dls * 100:.1f}%")
    print(f"Avg Latency: {np.mean(latencies_dls):.3f} ms")
    
    # Generate comparison plots
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        "MeanFlow": {
            "pos_errors": pos_errors_mf,
            "rot_errors": rot_errors_mf,
        },
        "Jacobian PI": {
            "pos_errors": pos_errors_jac,
            "rot_errors": rot_errors_jac,
        },
        "Damped LS": {
            "pos_errors": pos_errors_dls,
            "rot_errors": rot_errors_dls,
        },
    }
    
    plot_comparison(
        results_dict,
        metric_name="pos_error",
        save_path=str(output_dir / "position_error_comparison.png")
    )
    
    plot_comparison(
        results_dict,
        metric_name="rot_error",
        save_path=str(output_dir / "rotation_error_comparison.png")
    )
    
    # Summary table
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Method':<20} {'Pos Error (mm)':<15} {'Rot Error (°)':<15} {'Success Rate':<15} {'Latency (ms)':<15}")
    print("-" * 80)
    print(f"{'MeanFlow':<20} {pos_errors_mf.mean().item() * 1000:<15.2f} "
          f"{rot_errors_mf.mean().item() * 180 / np.pi:<15.2f} "
          f"{combined_success_mf * 100:<15.1f} "
          f"{meanflow_time / args.n_samples * 1000:<15.3f}")
    print(f"{'Jacobian PI':<20} {pos_errors_jac.mean().item() * 1000:<15.2f} "
          f"{rot_errors_jac.mean().item() * 180 / np.pi:<15.2f} "
          f"{combined_success_jac * 100:<15.1f} "
          f"{np.mean(latencies_jac):<15.3f}")
    print(f"{'Damped LS':<20} {pos_errors_dls.mean().item() * 1000:<15.2f} "
          f"{rot_errors_dls.mean().item() * 180 / np.pi:<15.2f} "
          f"{combined_success_dls * 100:<15.1f} "
          f"{np.mean(latencies_dls):<15.3f}")
    print("=" * 60)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
