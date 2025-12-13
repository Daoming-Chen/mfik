"""
MeanFlow IK Training - Version 1

Training pipeline with MeanFlow loss and JVP computation.
"""

from .config import TrainConfig
from .trainer import MeanFlowTrainer
from .optimizer import create_optimizer, create_scheduler

__all__ = ["TrainConfig", "MeanFlowTrainer", "create_optimizer", "create_scheduler"]
__version__ = "1.0.0"
