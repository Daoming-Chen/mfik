"""
Optimizer and learning rate scheduler utilities.
"""

import math
from typing import Optional

import torch
import torch.optim as optim


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
) -> optim.Optimizer:
    """
    Create AdamW optimizer.

    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        betas: Adam betas
        eps: Adam epsilon

    Returns:
        AdamW optimizer
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )
    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = "cosine",
    num_training_steps: Optional[int] = None,
    warmup_steps: int = 1000,
    min_lr: float = 1e-6,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler with warmup.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ("cosine" or "linear")
        num_training_steps: Total number of training steps
        warmup_steps: Number of warmup steps
        min_lr: Minimum learning rate

    Returns:
        Learning rate scheduler
    """
    if num_training_steps is None:
        return None

    if scheduler_type == "cosine":
        return CosineAnnealingWarmupLR(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=num_training_steps,
            min_lr=min_lr,
        )
    elif scheduler_type == "linear":
        return LinearWarmupLR(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=num_training_steps,
            min_lr=min_lr,
        )
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")


class CosineAnnealingWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing learning rate scheduler with linear warmup.

    Learning rate schedule:
    - Linear warmup from 0 to base_lr over warmup_steps
    - Cosine annealing from base_lr to min_lr over remaining steps
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


class LinearWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear learning rate scheduler with warmup.

    Learning rate schedule:
    - Linear warmup from 0 to base_lr over warmup_steps
    - Linear decay from base_lr to min_lr over remaining steps
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return [
                base_lr - (base_lr - self.min_lr) * progress
                for base_lr in self.base_lrs
            ]


def clip_gradients(
    model: torch.nn.Module,
    max_norm: float,
) -> float:
    """
    Clip gradients by global norm.

    Args:
        model: Model with gradients
        max_norm: Maximum gradient norm

    Returns:
        Total gradient norm before clipping
    """
    if max_norm <= 0:
        return 0.0

    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
    )
    return total_norm.item()


def get_gradient_norm(model: torch.nn.Module) -> float:
    """
    Compute total gradient norm.

    Args:
        model: Model with gradients

    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
