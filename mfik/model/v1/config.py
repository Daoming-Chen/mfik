"""
Model Configuration for MeanFlow IK Solver v1

Default hyperparameters for network architecture and training.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    """Configuration for MeanFlow ResNet architecture."""

    # Architecture
    num_resblocks: int = 6
    hidden_dim: int = 1024
    input_dim: int = None  # Will be set based on robot DOF + pose dim
    output_dim: int = None  # Will be set based on robot DOF

    # Positional encoding
    use_positional_encoding: bool = True
    pos_encoding_type: Literal["sinusoidal", "rff"] = "sinusoidal"
    pos_encoding_scale: float = 10.0
    pos_encoding_dim: int = 128

    # FiLM conditioning
    use_film: bool = True
    film_dim: int = 256

    # Activation
    activation: str = "silu"  # SiLU (Swish) activation

    # Normalization
    use_prenorm: bool = True
    norm_type: Literal["layernorm", "batchnorm"] = "layernorm"

    # Dropout (default: no dropout for fast inference)
    dropout: float = 0.0

    # Residual connections
    residual_scale: float = 1.0

    def __post_init__(self):
        """Validate configuration."""
        assert self.num_resblocks > 0, "num_resblocks must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert 0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)"

    def set_robot_dims(self, robot_dof: int, pose_dim: int = 7):
        """
        Set input/output dimensions based on robot configuration.

        Args:
            robot_dof: Number of robot joints
            pose_dim: Target pose dimension (default: 7 for [x,y,z,qw,qx,qy,qz])
        """
        # Input: joint angles (DOF) + target pose (pose_dim) + time params (2)
        self.input_dim = robot_dof + pose_dim + 2
        # Output: joint velocity (DOF)
        self.output_dim = robot_dof

    @classmethod
    def panda_default(cls) -> "ModelConfig":
        """Default configuration for Franka Emika Panda (7-DOF)."""
        config = cls()
        config.set_robot_dims(robot_dof=7, pose_dim=7)
        return config

    @classmethod
    def ur10_default(cls) -> "ModelConfig":
        """Default configuration for Universal Robots UR10 (6-DOF)."""
        config = cls()
        config.set_robot_dims(robot_dof=6, pose_dim=7)
        return config
