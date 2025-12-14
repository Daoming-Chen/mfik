"""
IK Inference Interface

Provides version-agnostic single-step IK inference with support for:
- Single and batch inference
- Dynamic model loading
- Trajectory tracking mode
- Joint limit post-processing
"""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Literal

import torch
import numpy as np

from ..robot.urdf import parse_urdf, get_kinematic_chain
from ..robot.forward_kinematics import ForwardKinematics
from ..model.v1.network import MeanFlowNet
from ..model.v1.checkpoint import CheckpointManager
from ..model.v1.config import ModelConfig


class UnreachableTargetError(Exception):
    """Raised when target pose is outside robot workspace."""
    pass


class IKSolver:
    """
    Version-agnostic IK solver with single-step inference.
    
    Supports batch inference, GPU acceleration, and trajectory tracking mode.
    """
    
    def __init__(
        self,
        urdf_path: str,
        model: torch.nn.Module,
        config: ModelConfig,
        device: str = "cpu",
        joint_limits: Optional[torch.Tensor] = None,
    ):
        """
        Initialize IK solver.
        
        Args:
            urdf_path: Path to URDF file
            model: Trained neural network
            config: Model configuration
            device: Computation device ('cpu' or 'cuda')
            joint_limits: Optional joint limits [n_joints, 2] (min, max)
        """
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        
        # Parse URDF and setup FK
        self.urdf_path = urdf_path
        self.chain = get_kinematic_chain(urdf_path)
        self.fk = ForwardKinematics(self.chain, device=device)
        self.n_joints = self.chain.n_joints
        
        # Joint limits
        if joint_limits is None:
            joint_limits = self._extract_joint_limits()
        self.joint_limits = joint_limits.to(device)
        
        # Trajectory tracking state
        self.last_solution: Optional[torch.Tensor] = None
        
    def _extract_joint_limits(self) -> torch.Tensor:
        """Extract joint limits from URDF."""
        limits = torch.zeros(self.n_joints, 2)
        for i, joint in enumerate(self.chain.joints):
            if joint.limit is not None:
                limits[i, 0] = joint.limit.lower
                limits[i, 1] = joint.limit.upper
            else:
                # Default to [-2π, 2π] for unlimited joints
                limits[i, 0] = -2 * np.pi
                limits[i, 1] = 2 * np.pi
        return limits
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        urdf_path: str,
        device: str = "cpu",
    ) -> "IKSolver":
        """
        Load solver from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            urdf_path: Path to robot URDF
            device: Computation device
            
        Returns:
            Initialized IKSolver instance
        """
        # Load checkpoint (CheckpointManager returns already loaded model)
        checkpoint_data = CheckpointManager.load_checkpoint(
            checkpoint_path, device=device
        )
        
        model = checkpoint_data["model"]
        config = checkpoint_data["config"]
        
        return cls(
            urdf_path=urdf_path,
            model=model,
            config=config,
            device=device,
        )
    
    def solve(
        self,
        target_pose: Union[np.ndarray, torch.Tensor],
        q_ref: Optional[Union[np.ndarray, torch.Tensor]] = None,
        strategy: Literal["closest", "min_motion", "avoid_limits"] = "closest",
        clip_joints: bool = True,
        check_singularity: bool = False,
        return_confidence: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        """
        Single-step IK inference.
        
        Args:
            target_pose: Target end-effector pose [7] (pos + quat) or [batch, 7]
            q_ref: Reference joint angles [n_joints] or [batch, n_joints]
                   If None, uses zero configuration or last solution (in tracking mode)
            strategy: Solution selection strategy (all equivalent to 'closest' in current design)
            clip_joints: Whether to clip joints to limits
            check_singularity: Whether to check for singularities
            return_confidence: Whether to return confidence score
            
        Returns:
            Joint solution [n_joints] or [batch, n_joints]
            Optional confidence score (0-1) if return_confidence=True
        """
        # Convert inputs to tensors
        target_pose = self._to_tensor(target_pose)
        is_batch = target_pose.dim() == 2
        
        if not is_batch:
            target_pose = target_pose.unsqueeze(0)
        
        batch_size = target_pose.shape[0]
        
        # Handle reference joints
        if q_ref is None:
            if self.last_solution is not None:
                # Trajectory tracking mode: use last solution
                q_ref = self.last_solution.expand(batch_size, -1)
            else:
                # Use zero configuration
                q_ref = torch.zeros(batch_size, self.n_joints, device=self.device)
        else:
            q_ref = self._to_tensor(q_ref)
            if q_ref.dim() == 1:
                q_ref = q_ref.unsqueeze(0).expand(batch_size, -1)
        
        # Inference
        with torch.no_grad():
            # Prepare input: concatenate [q_ref, target_pose, r, t]
            # Format: [joint_angles (DOF), target_pose (7), time_params (2)]
            # Note: Trainer uses [q, target_pose, r, t]
            # We want to query velocity at t=0 (start) to reach t=1 (target)
            # r is the reference start time (0)
            t = torch.zeros(batch_size, 1, device=self.device)
            r = torch.zeros(batch_size, 1, device=self.device)
            
            # Concatenate all inputs in correct order: q, target, r, t
            model_input = torch.cat([q_ref, target_pose, r, t], dim=-1)
            
            # Forward pass: predict velocity
            velocity = self.model(model_input)
            
            # Single-step update: q_pred = q_ref + velocity
            q_pred = q_ref + velocity
        
        # Post-processing
        if clip_joints:
            q_pred = self._clip_to_limits(q_pred)
        
        if check_singularity:
            self._check_singularity(q_pred)
        
        # Update tracking state
        if batch_size == 1:
            self.last_solution = q_pred[0]
        
        # Remove batch dimension if input was single
        if not is_batch:
            q_pred = q_pred.squeeze(0)
        
        if return_confidence:
            # Compute confidence based on FK error
            confidence = self._compute_confidence(q_pred, target_pose)
            return q_pred, confidence
        
        return q_pred
    
    def solve_trajectory(
        self,
        target_poses: Union[np.ndarray, torch.Tensor],
        q_init: Optional[Union[np.ndarray, torch.Tensor]] = None,
        check_smoothness: bool = True,
        smoothness_threshold: float = 0.1,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Solve IK for a trajectory (sequence of poses).
        
        Args:
            target_poses: Sequence of target poses [T, 7]
            q_init: Initial joint configuration [n_joints]. If None, uses zero config
            check_smoothness: Whether to check trajectory smoothness
            smoothness_threshold: Maximum allowed joint change between frames (radians)
            
        Returns:
            Joint trajectory [T, n_joints]
            Metadata dict with smoothness violations and timing
        """
        target_poses = self._to_tensor(target_poses)
        if target_poses.dim() != 2:
            raise ValueError(f"Expected target_poses shape [T, 7], got {target_poses.shape}")
        
        T = target_poses.shape[0]
        
        # Initialize trajectory
        if q_init is None:
            q_init = torch.zeros(self.n_joints, device=self.device)
        else:
            q_init = self._to_tensor(q_init)
        
        # Reset tracking state
        self.last_solution = q_init
        
        # Solve each frame
        trajectory = torch.zeros(T, self.n_joints, device=self.device)
        smoothness_violations = []
        
        start_time = time.time()
        
        for t in range(T):
            q_t = self.solve(target_poses[t], q_ref=None)  # Uses last_solution
            trajectory[t] = q_t
            
            # Check smoothness
            if check_smoothness and t > 0:
                delta = torch.abs(trajectory[t] - trajectory[t-1]).max().item()
                if delta > smoothness_threshold:
                    smoothness_violations.append({
                        "frame": t,
                        "max_delta": delta,
                    })
        
        elapsed = time.time() - start_time
        
        metadata = {
            "total_time": elapsed,
            "avg_time_per_frame": elapsed / T,
            "smoothness_violations": smoothness_violations,
        }
        
        return trajectory, metadata
    
    def benchmark(
        self,
        test_poses: Union[np.ndarray, torch.Tensor],
        q_refs: Optional[Union[np.ndarray, torch.Tensor]] = None,
        warmup_steps: int = 10,
    ) -> Dict:
        """
        Benchmark inference performance.
        
        Args:
            test_poses: Test target poses [N, 7]
            q_refs: Optional reference joints [N, n_joints]
            warmup_steps: Number of warmup iterations
            
        Returns:
            Performance metrics dictionary
        """
        test_poses = self._to_tensor(test_poses)
        N = test_poses.shape[0]
        
        if q_refs is None:
            q_refs = torch.zeros(N, self.n_joints, device=self.device)
        else:
            q_refs = self._to_tensor(q_refs)
        
        # Warmup
        for _ in range(warmup_steps):
            _ = self.solve(test_poses[0], q_refs[0])
        
        # Benchmark
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        
        start_time = time.time()
        solutions = self.solve(test_poses, q_refs)
        
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        
        # Compute accuracy
        from .metrics import compute_pose_error
        pos_errors, rot_errors = compute_pose_error(
            solutions, test_poses, self.fk
        )
        
        return {
            "total_time": elapsed,
            "avg_latency": elapsed / N,
            "throughput": N / elapsed,
            "pos_error_mean": pos_errors.mean().item(),
            "pos_error_median": pos_errors.median().item(),
            "pos_error_95": torch.quantile(pos_errors, 0.95).item(),
            "rot_error_mean": rot_errors.mean().item(),
            "rot_error_median": rot_errors.median().item(),
            "rot_error_95": torch.quantile(rot_errors, 0.95).item(),
        }
    
    def reset_tracking(self):
        """Reset trajectory tracking state."""
        self.last_solution = None
    
    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert numpy array or tensor to device tensor."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return x.to(self.device)
    
    def _clip_to_limits(self, q: torch.Tensor) -> torch.Tensor:
        """Clip joint angles to limits."""
        return torch.clamp(
            q,
            min=self.joint_limits[:, 0],
            max=self.joint_limits[:, 1],
        )
    
    def _check_singularity(self, q: torch.Tensor):
        """Check for singularities (Jacobian rank deficiency)."""
        # Compute Jacobian condition number
        # This is a placeholder - actual implementation would require
        # computing the geometric Jacobian
        pass
    
    def _compute_confidence(
        self,
        q_pred: torch.Tensor,
        target_pose: torch.Tensor,
    ) -> float:
        """
        Compute confidence score based on FK error.
        
        Returns value in [0, 1], where 1 means perfect solution.
        """
        # Compute FK
        pred_positions, pred_quaternions = self.fk.compute(q_pred)
        pred_pose = torch.cat([pred_positions, pred_quaternions], dim=-1)
        
        # Compute errors
        from .metrics import compute_pose_error
        pos_errors, rot_errors = compute_pose_error(
            q_pred, target_pose, self.fk
        )
        
        # Normalize errors to confidence (lower error = higher confidence)
        # Using exponential decay: conf = exp(-error / threshold)
        pos_threshold = 0.005  # 5mm
        rot_threshold = 0.0873  # 5 degrees in radians
        
        pos_conf = torch.exp(-pos_errors / pos_threshold)
        rot_conf = torch.exp(-rot_errors / rot_threshold)
        
        # Geometric mean
        confidence = torch.sqrt(pos_conf * rot_conf).mean().item()
        
        return confidence
