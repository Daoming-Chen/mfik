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
from torch.amp import autocast
from torch.cuda.amp import GradScaler
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
        self.scaler = GradScaler() if config.use_amp else None

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
    ) -> Dict[str, torch.Tensor]:
        """
        Compute MeanFlow loss using JVP.

        Args:
            q: Current joint angles [B, DOF]
            target_pose: Target end-effector pose [B, 7]
            q_target: Target joint angles [B, DOF]
            r: Reference time parameter [B, 1]
            t: Evaluation time parameter [B, 1]

        Returns:
            Dictionary with loss and metrics
        """
        batch_size = q.shape[0]
        dof = q.shape[1]

        # Prepare input: [q, target_pose, r, t]
        x = torch.cat([q, target_pose, r, t], dim=-1)  # [B, DOF + 7 + 2]

        # Compute velocity field u(q, r, t)
        u = self.model(x)  # [B, DOF]

        # Compute flow velocity v_t = (q_target - q) / (1 - r)
        # Note: We use linear interpolation q_t = q + v_t * (t - r)
        dt = torch.clamp(1.0 - r, min=1e-8)  # Avoid division by zero
        v_t = (q_target - q) / dt  # [B, DOF]

        # Stop gradient on v_t (treat as constant for JVP)
        v_t = v_t.detach()

        # Compute time derivative ∂u/∂t using JVP
        # We need to compute: v_t · ∇_q u + ∂u/∂t

        # Create tangent vectors for JVP
        # For ∂u/∂q: tangent is v_t (flow velocity)
        # For ∂u/∂t: tangent is 1.0
        tangent_q = v_t  # [B, DOF]
        tangent_t = torch.ones_like(t)  # [B, 1]
        tangent_r = torch.zeros_like(r)  # [B, 1]
        tangent_pose = torch.zeros_like(target_pose)  # [B, 7]

        tangent_x = torch.cat([tangent_q, tangent_pose, tangent_r, tangent_t], dim=-1)

        # Compute JVP: d/dt u = v_t · ∇_q u + ∂u/∂t
        # Using torch.func.jvp (Jacobian-Vector Product) with vmap for batching
        def model_fn(x_input):
            return self.model(x_input)

        # Batch JVP computation using vmap
        def single_jvp(x_i, tangent_i):
            _, jvp_i = torch.func.jvp(lambda x: model_fn(x.unsqueeze(0)), (x_i,), (tangent_i,))
            return jvp_i.squeeze(0)
        
        # Apply vmap to batch the JVP computation
        jvp_output = torch.vmap(single_jvp)(x, tangent_x)  # [B, DOF]

        # MeanFlow loss: || v_t · ∇_q u + ∂u/∂t ||^2
        # Note: jvp_output = v_t · ∇_q u + ∂u/∂t
        meanflow_residual = jvp_output  # [B, DOF]

        # Compute adaptive weights if configured
        if self.config.loss_weight_type == "adaptive":
            # w = 1 / (||q_target - q||^2 + c)^p
            delta_norm_sq = torch.sum((q_target - q) ** 2, dim=-1, keepdim=True)  # [B, 1]
            weights = 1.0 / (delta_norm_sq + self.config.adaptive_weight_c) ** self.config.adaptive_weight_p
            weights = weights.detach()  # Stop gradient
        else:
            weights = torch.ones(batch_size, 1, device=self.device)

        # Weighted loss
        loss_per_sample = torch.sum(meanflow_residual ** 2, dim=-1, keepdim=True)  # [B, 1]
        weighted_loss = weights * loss_per_sample  # [B, 1]
        loss = weighted_loss.mean()

        # Metrics
        metrics = {
            "loss": loss,
            "meanflow_residual": meanflow_residual.abs().mean(),
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
            metrics = self.compute_meanflow_loss(q, target_pose, q_target, r, t)
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
                total_residual += metrics["meanflow_residual"].item()
                num_batches += 1

        val_metrics = {
            "val_loss": total_loss / num_batches,
            "val_residual": total_residual / num_batches,
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
