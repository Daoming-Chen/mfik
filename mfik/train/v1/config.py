"""
Training Configuration for MeanFlow IK Solver v1

Default hyperparameters for training process.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """Configuration for MeanFlow training."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # "cosine" or "linear"
    warmup_steps: int = 1000
    min_lr: float = 1e-6

    # Training
    batch_size: int = 256
    num_epochs: int = 100
    gradient_clip: float = 1.0
    accumulation_steps: int = 1  # Gradient accumulation

    # MeanFlow loss
    loss_weight_type: str = "adaptive"  # "adaptive" or "uniform"
    adaptive_weight_c: float = 1e-4  # Constant in adaptive weight
    adaptive_weight_p: float = 0.5  # Power in adaptive weight

    # Logging
    log_interval: int = 100  # Log every N steps
    eval_interval: int = 1000  # Evaluate every N steps
    checkpoint_interval: int = 5000  # Save checkpoint every N steps

    # Checkpoint
    checkpoint_dir: str = "checkpoints"
    save_top_k: int = 3  # Keep top k checkpoints
    resume_from: Optional[str] = None  # Path to checkpoint to resume from

    # Device
    device: str = "cuda"  # "cuda" or "cpu"
    num_workers: int = 4  # DataLoader workers

    # Mixed precision
    use_amp: bool = True  # Automatic mixed precision

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        """Validate configuration."""
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert 0 <= self.gradient_clip, "gradient_clip must be non-negative"
        assert self.loss_weight_type in ["adaptive", "uniform"], \
            "loss_weight_type must be 'adaptive' or 'uniform'"
        assert self.scheduler_type in ["cosine", "linear"], \
            "scheduler_type must be 'cosine' or 'linear'"

    @classmethod
    def quick_test(cls) -> "TrainConfig":
        """Configuration for quick testing (small dataset, few epochs)."""
        config = cls()
        config.batch_size = 32
        config.num_epochs = 5
        config.log_interval = 10
        config.eval_interval = 50
        config.checkpoint_interval = 100
        config.use_amp = False  # Disable for debugging
        return config

    @classmethod
    def panda_default(cls) -> "TrainConfig":
        """Default configuration for Franka Emika Panda training."""
        config = cls()
        config.learning_rate = 1e-4
        config.batch_size = 256
        config.num_epochs = 100
        config.warmup_steps = 1000
        return config

    @classmethod
    def ur10_default(cls) -> "TrainConfig":
        """Default configuration for Universal Robots UR10 training."""
        config = cls()
        config.learning_rate = 1e-4
        config.batch_size = 256
        config.num_epochs = 100
        config.warmup_steps = 1000
        return config
