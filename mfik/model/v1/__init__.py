"""
MeanFlow IK Solver - Version 1

Architecture: MLP-ResNet with FiLM conditioning
- 6 Residual Blocks
- 1024 hidden units
- Positional encoding
- Pre-normalization
"""

from .config import ModelConfig
from .network import MeanFlowNet
from .checkpoint import CheckpointManager

__all__ = ["ModelConfig", "MeanFlowNet", "CheckpointManager"]
__version__ = "1.0.0"
