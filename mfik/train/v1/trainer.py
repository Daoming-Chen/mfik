"""
MeanFlow Trainer with JVP-based loss computation.

The trainer implements the MeanFlow condition:
    ∂u/∂t + v_t · ∇_q u = 0

where u(q, r, t) is the learned velocity field.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Callable

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from .config import TrainConfig
from .optimizer import create_optimizer, create_scheduler, clip_gradients, get_gradient_norm


class MeanFlowTrainer:
    """
    MeanFlow trainer with JVP-based loss computation.

    The loss is computed using the MeanFlow identity:
        loss = || ∂u/∂t + v_t · ∇_q u ||^2
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        robot_fk: Optional[Callable] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: MeanFlow network
            config: Training configuration
            robot_fk: Optional forward kinematics function for validation
        """
        self.model = model
        self.config = config
        self.robot_fk = robot_fk
        self.device = torch.device(config.device)

        # Move model to device
        self.model.to(self.device)

        # Optimizer
        self.optimizer = create_optimizer(
            model=self.model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
        )

        # Scheduler (will be initialized when training starts)
        self.scheduler = None

        # Mixed precision
        self.scaler = GradScaler('cuda') if config.use_amp else None

        # Tracking
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float("inf")

        # TensorBoard
        self.writer = None

    def compute_meanflow_loss(
        self,
        q: torch.Tensor,
        target_pose: torch.Tensor,
        q_target: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        v_t_gt: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute MeanFlow loss using JVP.

        Args:
            q: Current joint angles [B, DOF]
            target_pose: Target end-effector pose [B, 7]
            q_target: Target joint angles [B, DOF]
            r: Reference time parameter [B, 1]
            t: Evaluation time parameter [B, 1]
            v_t_gt: Ground truth velocity [B, DOF] (optional)

        Returns:
            Dictionary with loss and metrics
        """
        batch_size = q.shape[0]
        dof = q.shape[1]

        # Prepare input: [q, target_pose, r, t]
        x = torch.cat([q, target_pose, r, t], dim=-1)  # [B, DOF + 7 + 2]

        # Compute velocity field u(q, r, t)
        u = self.model(x)  # [B, DOF]

        # Compute flow velocity v_t
        if v_t_gt is not None:
            v_t = v_t_gt
        else:
            # Fallback: Compute flow velocity v_t = (q_target - q) / (1 - t)
            # Note: This is unstable when t -> 1
            dt = torch.clamp(1.0 - t, min=1e-8)  # Avoid division by zero
            v_t = (q_target - q) / dt  # [B, DOF]

        # Stop gradient on v_t (treat as constant for JVP)
        v_t = v_t.detach()

        # Conditional Flow Matching Loss: || u - v_t ||^2
        # We want the learned velocity field u to match the conditional vector field v_t
        diff = u - v_t
        loss_per_sample = torch.sum(diff ** 2, dim=-1, keepdim=True)  # [B, 1]

        # Compute adaptive weights if configured
        if self.config.loss_weight_type == "adaptive":
            # w = 1 / (||q_target - q||^2 + c)^p
            delta_norm_sq = torch.sum((q_target - q) ** 2, dim=-1, keepdim=True)  # [B, 1]
            weights = 1.0 / (delta_norm_sq + self.config.adaptive_weight_c) ** self.config.adaptive_weight_p
            weights = weights.detach()  # Stop gradient
        else:
            weights = torch.ones(batch_size, 1, device=self.device)

        # Weighted loss
        weighted_loss = weights * loss_per_sample  # [B, 1]
        loss = weighted_loss.mean()

        # Metrics
        metrics = {
            "loss": loss,
            "velocity_error": diff.norm(dim=-1).mean(),
            "velocity_norm": u.norm(dim=-1).mean(),
            "flow_velocity_norm": v_t.norm(dim=-1).mean(),
        }

        return metrics

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Batch of training data with keys:
                - q: Current joint angles [B, DOF]
                - target_pose: Target pose [B, 7]
                - q_target: Target joint angles [B, DOF]
                - r: Reference time [B, 1]
                - t: Evaluation time [B, 1]

        Returns:
            Dictionary of metrics
        """
        self.model.train()

        # Move batch to device
        q = batch["q"].to(self.device)
        target_pose = batch["target_pose"].to(self.device)
        q_target = batch["q_target"].to(self.device)
        r = batch["r"].to(self.device)
        t = batch["t"].to(self.device)

        # Forward pass with mixed precision
        with autocast(device_type='cuda', enabled=self.config.use_amp):
            v_t_gt = batch["v_t"].to(self.device) if "v_t" in batch else None
            metrics = self.compute_meanflow_loss(q, target_pose, q_target, r, t, v_t_gt)
            loss = metrics["loss"]

        # Backward pass
        if self.config.use_amp:
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = clip_gradients(self.model, self.config.gradient_clip)
            else:
                grad_norm = get_gradient_norm(self.model)

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                grad_norm = clip_gradients(self.model, self.config.gradient_clip)
            else:
                grad_norm = get_gradient_norm(self.model)

            # Optimizer step
            self.optimizer.step()

        # Scheduler step (must be after optimizer.step())
        if self.scheduler is not None:
            self.scheduler.step()

        self.optimizer.zero_grad()

        # Convert metrics to float
        metrics_float = {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}
        metrics_float["grad_norm"] = grad_norm
        if self.scheduler is not None:
            metrics_float["lr"] = self.scheduler.get_last_lr()[0]
        else:
            metrics_float["lr"] = self.config.learning_rate

        return metrics_float

    def validate(
        self,
        val_loader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """
        Validation loop.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_residual = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                q = batch["q"].to(self.device)
                target_pose = batch["target_pose"].to(self.device)
                q_target = batch["q_target"].to(self.device)
                r = batch["r"].to(self.device)
                t = batch["t"].to(self.device)

                with autocast(device_type='cuda', enabled=self.config.use_amp):
                    metrics = self.compute_meanflow_loss(q, target_pose, q_target, r, t)

                total_loss += metrics["loss"].item()
                total_residual += metrics["velocity_error"].item()
                num_batches += 1

        val_metrics = {
            "val_loss": total_loss / num_batches,
            "val_velocity_error": total_residual / num_batches,
        }

        return val_metrics

    def save_checkpoint(self, checkpoint_path: str, metadata: Optional[Dict] = None):
        """Save training checkpoint."""
        from mfik.model.v1 import CheckpointManager

        checkpoint_metadata = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
        }
        if metadata:
            checkpoint_metadata.update(metadata)

        CheckpointManager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            config=self.model.config,
            checkpoint_path=checkpoint_path,
            metadata=checkpoint_metadata,
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        from mfik.model.v1 import CheckpointManager

        result = CheckpointManager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            device=self.config.device,
            load_optimizer=True,
        )

        self.model.load_state_dict(result["model"].state_dict())
        if "optimizer_state_dict" in result:
            self.optimizer.load_state_dict(result["optimizer_state_dict"])

        metadata = result.get("metadata", {})
        self.current_epoch = metadata.get("epoch", 0)
        self.global_step = metadata.get("global_step", 0)
        self.best_loss = metadata.get("best_loss", float("inf"))

        print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
