"""
Unit tests for eval module
"""

import unittest
from pathlib import Path
import tempfile

import torch
import numpy as np

from mfik.eval.metrics import (
    compute_pose_error,
    quaternion_distance,
    compute_success_rate,
    measure_latency,
    compute_joint_statistics,
    compute_trajectory_smoothness,
    compute_workspace_coverage,
)
from mfik.robot.urdf import get_kinematic_chain
from mfik.robot.forward_kinematics import ForwardKinematics


class TestMetrics(unittest.TestCase):
    """Test evaluation metrics functions."""
    
    @classmethod
    def setUpClass(cls):
        """Setup test fixtures."""
        # Use Panda robot
        cls.urdf_path = "robots/panda_arm.urdf"
        cls.chain = get_kinematic_chain(cls.urdf_path)
        cls.fk = ForwardKinematics(cls.chain, device="cpu")
        cls.device = "cpu"
    
    def test_quaternion_distance(self):
        """Test quaternion distance calculation."""
        # Identity quaternions
        q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        q2 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        dist = quaternion_distance(q1, q2)
        self.assertAlmostEqual(dist.item(), 0.0, places=5)
        
        # 90 degree rotation around z-axis
        q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        q2 = torch.tensor([[0.707, 0.0, 0.0, 0.707]])  # 90 deg around z
        dist = quaternion_distance(q1, q2)
        expected = np.pi / 2  # 90 degrees
        self.assertAlmostEqual(dist.item(), expected, places=3)
        
        # Opposite quaternions (same rotation)
        q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        q2 = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])
        dist = quaternion_distance(q1, q2)
        self.assertAlmostEqual(dist.item(), 0.0, places=5)
    
    def test_compute_pose_error(self):
        """Test pose error computation."""
        # Generate random joint configuration
        q = torch.zeros(1, self.fk.n_joints)
        target_pose = self.fk.forward(q)
        
        # Same configuration should have zero error
        pos_errors, rot_errors = compute_pose_error(q, target_pose, self.fk)
        self.assertLess(pos_errors.item(), 1e-5)
        self.assertLess(rot_errors.item(), 1e-5)
        
        # Different configuration should have non-zero error
        q_perturbed = q + torch.randn_like(q) * 0.1
        pos_errors, rot_errors = compute_pose_error(q_perturbed, target_pose, self.fk)
        self.assertGreater(pos_errors.item(), 0)
    
    def test_compute_success_rate(self):
        """Test success rate computation."""
        # All successful
        pos_errors = torch.tensor([0.001, 0.002, 0.003])  # < 5mm
        rot_errors = torch.tensor([0.01, 0.02, 0.03])  # < 5 degrees
        
        pos_success, rot_success, combined = compute_success_rate(
            pos_errors, rot_errors
        )
        self.assertEqual(pos_success, 1.0)
        self.assertEqual(rot_success, 1.0)
        self.assertEqual(combined, 1.0)
        
        # Mixed success
        pos_errors = torch.tensor([0.001, 0.002, 0.01])  # Last one fails
        rot_errors = torch.tensor([0.01, 0.02, 0.03])
        
        pos_success, rot_success, combined = compute_success_rate(
            pos_errors, rot_errors, pos_threshold=0.005
        )
        self.assertAlmostEqual(pos_success, 2/3, places=2)
        self.assertEqual(rot_success, 1.0)
        self.assertAlmostEqual(combined, 2/3, places=2)
    
    def test_measure_latency(self):
        """Test latency measurement."""
        def dummy_inference():
            # Simulate some computation
            x = torch.randn(100, 100)
            return torch.matmul(x, x)
        
        stats = measure_latency(
            dummy_inference,
            num_iterations=10,
            warmup_iterations=2,
            device="cpu",
        )
        
        self.assertIn("mean_ms", stats)
        self.assertIn("median_ms", stats)
        self.assertIn("std_ms", stats)
        self.assertGreater(stats["mean_ms"], 0)
    
    def test_compute_joint_statistics(self):
        """Test joint statistics computation."""
        # Create test data
        q_solutions = torch.tensor([
            [0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.6, -0.4, 0.1, 0.1, 0.1, 0.1],
            [3.5, 0.7, -0.3, 0.2, 0.2, 0.2, 0.2],  # First joint violates
        ])
        
        joint_limits = torch.tensor([
            [-2.8, 2.8],
            [-1.7, 1.7],
            [-2.8, 2.8],
            [-3.0, 0.0],
            [-2.8, 2.8],
            [-0.0, 3.7],
            [-2.8, 2.8],
        ])
        
        stats = compute_joint_statistics(q_solutions, joint_limits)
        
        self.assertIn("violation_rate_per_joint", stats)
        self.assertIn("total_violation_rate", stats)
        self.assertGreater(stats["total_violation_rate"], 0)
        self.assertEqual(len(stats["violation_rate_per_joint"]), 7)
    
    def test_compute_trajectory_smoothness(self):
        """Test trajectory smoothness metrics."""
        # Create smooth trajectory
        t = torch.linspace(0, 2 * np.pi, 100)
        trajectory = torch.stack([
            torch.sin(t),
            torch.cos(t),
            t / 10,
            torch.zeros_like(t),
            torch.zeros_like(t),
            torch.zeros_like(t),
            torch.zeros_like(t),
        ], dim=1)
        
        stats = compute_trajectory_smoothness(trajectory, dt=0.01)
        
        self.assertIn("max_velocity", stats)
        self.assertIn("max_acceleration", stats)
        self.assertIn("max_jerk", stats)
        self.assertGreater(stats["max_velocity"], 0)
    
    def test_compute_workspace_coverage(self):
        """Test workspace coverage computation."""
        # Create random positions
        positions = torch.randn(1000, 3) * 0.5
        
        stats = compute_workspace_coverage(positions, grid_resolution=0.05)
        
        self.assertIn("workspace_volume_m3", stats)
        self.assertIn("unique_cells", stats)
        self.assertIn("coverage_ratio", stats)
        self.assertGreater(stats["workspace_volume_m3"], 0)
        self.assertGreater(stats["coverage_ratio"], 0)
        self.assertLessEqual(stats["coverage_ratio"], 1.0)


class TestInference(unittest.TestCase):
    """Test IK inference interface."""
    
    def setUp(self):
        """Setup for each test."""
        from mfik.model.v1.network import MeanFlowNet
        from mfik.model.v1.config import ModelConfig
        from mfik.eval.inference import IKSolver
        
        # Create minimal model
        self.urdf_path = "robots/panda_arm.urdf"
        config = ModelConfig(
            n_joints=7,
            pose_dim=7,
            hidden_dim=64,
            num_blocks=2,
        )
        model = MeanFlowNet(config)
        
        self.solver = IKSolver(
            urdf_path=self.urdf_path,
            model=model,
            config=config,
            device="cpu",
        )
    
    def test_solve_single(self):
        """Test single IK solve."""
        # Generate random target
        target_pose = torch.randn(7)
        target_pose[3:] = target_pose[3:] / torch.norm(target_pose[3:])  # Normalize quat
        
        # Solve
        q_pred = self.solver.solve(target_pose)
        
        self.assertEqual(q_pred.shape, (7,))
        self.assertTrue(torch.all(torch.isfinite(q_pred)))
    
    def test_solve_batch(self):
        """Test batch IK solve."""
        batch_size = 10
        target_poses = torch.randn(batch_size, 7)
        target_poses[:, 3:] = target_poses[:, 3:] / torch.norm(target_poses[:, 3:], dim=1, keepdim=True)
        
        q_pred = self.solver.solve(target_poses)
        
        self.assertEqual(q_pred.shape, (batch_size, 7))
        self.assertTrue(torch.all(torch.isfinite(q_pred)))
    
    def test_trajectory_tracking(self):
        """Test trajectory solving."""
        # Generate simple trajectory
        T = 20
        target_poses = torch.randn(T, 7)
        target_poses[:, 3:] = target_poses[:, 3:] / torch.norm(target_poses[:, 3:], dim=1, keepdim=True)
        
        trajectory, metadata = self.solver.solve_trajectory(target_poses)
        
        self.assertEqual(trajectory.shape, (T, 7))
        self.assertIn("total_time", metadata)
        self.assertIn("smoothness_violations", metadata)
    
    def test_reset_tracking(self):
        """Test tracking state reset."""
        self.solver.last_solution = torch.randn(7)
        self.solver.reset_tracking()
        self.assertIsNone(self.solver.last_solution)


class TestVisualization(unittest.TestCase):
    """Test visualization functions."""
    
    def test_plot_error_distribution(self):
        """Test error distribution plotting."""
        from mfik.eval.visualization import plot_error_distribution
        
        pos_errors = torch.rand(100) * 0.01
        rot_errors = torch.rand(100) * 0.1
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "test_plot.png")
            fig = plot_error_distribution(pos_errors, rot_errors, save_path=save_path)
            
            self.assertTrue(Path(save_path).exists())
            self.assertIsNotNone(fig)
    
    def test_plot_workspace_heatmap(self):
        """Test workspace heatmap plotting."""
        from mfik.eval.visualization import plot_workspace_heatmap
        
        positions = torch.randn(100, 3)
        errors = torch.rand(100) * 0.01
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "test_heatmap.png")
            fig = plot_workspace_heatmap(positions, errors, save_path=save_path)
            
            self.assertTrue(Path(save_path).exists())
            self.assertIsNotNone(fig)
    
    def test_generate_performance_report(self):
        """Test performance report generation."""
        from mfik.eval.visualization import generate_performance_report
        
        results = {
            "pos_error_mean": 0.002,
            "pos_error_median": 0.0015,
            "pos_error_95": 0.004,
            "rot_error_mean": 0.05,
            "rot_error_median": 0.04,
            "rot_error_95": 0.08,
            "success_rate": (0.95, 0.93, 0.92),
            "avg_latency": 0.001,
            "throughput": 1000,
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "report.md")
            generate_performance_report(results, save_path)
            
            self.assertTrue(Path(save_path).exists())
            
            # Check content
            with open(save_path) as f:
                content = f.read()
                self.assertIn("Performance Report", content)
                self.assertIn("Accuracy Metrics", content)


if __name__ == "__main__":
    unittest.main()
