"""
MeanFlow ResNet Architecture for IK Solving

Network components:
- Positional encoding (sinusoidal or RFF)
- Input projection
- Residual blocks with pre-normalization
- FiLM conditioning
- Output head
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for continuous inputs."""

    def __init__(self, dim: int, scale: float = 10.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., D]
        Returns:
            Encoded tensor [..., D * dim * 2]
        """
        device = x.device
        half_dim = self.dim // 2

        # Compute frequencies
        freqs = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32)
            * -(math.log(10000.0) / (half_dim - 1))
        )
        freqs = freqs * self.scale

        # Apply encoding to each input dimension
        x_expanded = x[..., :, None]  # [..., D, 1]
        args = x_expanded * freqs  # [..., D, half_dim]

        # Concatenate sin and cos
        encoding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [..., D, dim]
        return encoding.flatten(-2)  # [..., D * dim]


class RandomFourierFeatures(nn.Module):
    """Random Fourier Features for positional encoding."""

    def __init__(self, input_dim: int, feature_dim: int, scale: float = 10.0):
        super().__init__()
        self.feature_dim = feature_dim
        # Random projection matrix (fixed, not trainable)
        self.register_buffer(
            "B", torch.randn(input_dim, feature_dim // 2) * scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., input_dim]
        Returns:
            Encoded tensor [..., feature_dim]
        """
        x_proj = x @ self.B  # [..., feature_dim // 2]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation layer."""

    def __init__(self, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.scale_shift = nn.Linear(condition_dim, hidden_dim * 2)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor [..., hidden_dim]
            condition: Conditioning tensor [..., condition_dim]
        Returns:
            Modulated tensor [..., hidden_dim]
        """
        scale_shift = self.scale_shift(condition)
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)
        return x * (1 + scale) + shift


class ResidualBlock(nn.Module):
    """Residual block with pre-normalization and optional FiLM conditioning."""

    def __init__(
        self,
        hidden_dim: int,
        condition_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "silu",
        norm_type: str = "layernorm",
    ):
        super().__init__()

        # Normalization
        if norm_type == "layernorm":
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
        elif norm_type == "batchnorm":
            self.norm1 = nn.BatchNorm1d(hidden_dim)
            self.norm2 = nn.BatchNorm1d(hidden_dim)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # Activation
        if activation == "silu":
            self.act = nn.SiLU()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Layers
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Optional FiLM conditioning
        self.film = FiLM(hidden_dim, condition_dim) if condition_dim else None

    def forward(
        self, x: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., hidden_dim]
            condition: Optional conditioning tensor [..., condition_dim]
        Returns:
            Output tensor [..., hidden_dim]
        """
        residual = x

        # First layer
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)

        # Second layer
        x = self.norm2(x)
        x = self.linear2(x)

        # FiLM conditioning
        if self.film is not None and condition is not None:
            x = self.film(x, condition)

        x = self.act(x)
        x = self.dropout(x)

        # Residual connection
        return x + residual


class MeanFlowNet(nn.Module):
    """
    MeanFlow ResNet for inverse kinematics.

    Architecture:
        Input -> [Positional Encoding] -> Projection -> ResBlocks -> Output Head
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        assert config.input_dim is not None, "input_dim must be set"
        assert config.output_dim is not None, "output_dim must be set"

        # Positional encoding
        if config.use_positional_encoding:
            if config.pos_encoding_type == "sinusoidal":
                self.pos_encoder = SinusoidalPositionalEncoding(
                    dim=config.pos_encoding_dim, scale=config.pos_encoding_scale
                )
                encoded_dim = config.input_dim * config.pos_encoding_dim
            elif config.pos_encoding_type == "rff":
                self.pos_encoder = RandomFourierFeatures(
                    input_dim=config.input_dim,
                    feature_dim=config.pos_encoding_dim,
                    scale=config.pos_encoding_scale,
                )
                encoded_dim = config.pos_encoding_dim
            else:
                raise ValueError(f"Unknown pos_encoding_type: {config.pos_encoding_type}")
        else:
            self.pos_encoder = None
            encoded_dim = config.input_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(encoded_dim, config.hidden_dim), nn.SiLU()
        )

        # Residual blocks
        condition_dim = config.film_dim if config.use_film else None
        self.resblocks = nn.ModuleList(
            [
                ResidualBlock(
                    hidden_dim=config.hidden_dim,
                    condition_dim=condition_dim,
                    dropout=config.dropout,
                    activation=config.activation,
                    norm_type=config.norm_type,
                )
                for _ in range(config.num_resblocks)
            ]
        )

        # Condition projection (if using FiLM)
        if config.use_film:
            self.condition_proj = nn.Sequential(
                nn.Linear(config.input_dim, config.film_dim), nn.SiLU()
            )
        else:
            self.condition_proj = None

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.output_dim)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [..., input_dim]
               Format: [joint_angles (DOF), target_pose (7), time_params (2)]

        Returns:
            Joint velocity tensor [..., output_dim]
        """
        # Positional encoding
        if self.pos_encoder is not None:
            x_encoded = self.pos_encoder(x)
        else:
            x_encoded = x

        # Input projection
        h = self.input_proj(x_encoded)

        # Condition projection (for FiLM)
        if self.condition_proj is not None:
            condition = self.condition_proj(x)
        else:
            condition = None

        # Residual blocks
        for block in self.resblocks:
            h = block(h, condition)

        # Output head
        output = self.output_head(h)

        return output

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(robot_dof: int, config_override: Optional[dict] = None):
    """
    Factory function to create MeanFlowNet.

    Args:
        robot_dof: Number of robot joints
        config_override: Optional config overrides

    Returns:
        Initialized MeanFlowNet
    """
    from .config import ModelConfig

    config = ModelConfig()
    config.set_robot_dims(robot_dof=robot_dof, pose_dim=7)

    if config_override:
        for key, value in config_override.items():
            setattr(config, key, value)

    return MeanFlowNet(config)
