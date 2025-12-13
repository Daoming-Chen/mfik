"""
Model checkpoint management with version metadata.
"""

from pathlib import Path
from typing import Dict, Optional, Any

import torch


class CheckpointManager:
    """Manages model checkpoints with versioning metadata."""

    @staticmethod
    def save_checkpoint(
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        config: Any,
        checkpoint_path: str,
        metadata: Optional[Dict] = None,
    ):
        """
        Save model checkpoint with version metadata.

        Args:
            model: PyTorch model
            optimizer: Optional optimizer state
            config: Model configuration
            checkpoint_path: Path to save checkpoint
            metadata: Optional additional metadata (epoch, metrics, etc.)
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Build checkpoint dict
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "version": "v1",
            "model_type": "MeanFlowNet",
        }

        # Add optimizer state if provided
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        # Add metadata
        if metadata:
            checkpoint["metadata"] = metadata

        # Save
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        device: str = "cpu",
        load_optimizer: bool = False,
    ) -> Dict:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model to
            load_optimizer: Whether to load optimizer state

        Returns:
            Dictionary containing model, config, and optional optimizer
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint (weights_only=False for compatibility with config objects)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Verify version
        if "version" not in checkpoint:
            raise ValueError("Checkpoint missing version information")

        version = checkpoint["version"]
        print(f"Loading checkpoint from {version}")

        # Load model
        if version == "v1":
            from mfik.model.v1 import MeanFlowNet

            config = checkpoint["config"]
            model = MeanFlowNet(config)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
        else:
            raise ValueError(f"Unsupported checkpoint version: {version}")

        result = {
            "model": model,
            "config": config,
            "version": version,
            "metadata": checkpoint.get("metadata", {}),
        }

        # Load optimizer if requested
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            result["optimizer_state_dict"] = checkpoint["optimizer_state_dict"]

        return result

    @staticmethod
    def save_pretrained(
        model: torch.nn.Module,
        config: Any,
        save_dir: str,
        model_name: str = "model.pth",
    ):
        """
        Save pretrained model in a standardized format.

        Args:
            model: Trained model
            config: Model configuration
            save_dir: Directory to save model
            model_name: Model filename
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = save_dir / model_name
        CheckpointManager.save_checkpoint(
            model=model,
            optimizer=None,
            config=config,
            checkpoint_path=str(checkpoint_path),
            metadata={"pretrained": True},
        )

        # Save config separately for easy inspection
        import json

        config_dict = {
            k: v for k, v in vars(config).items() if not k.startswith("_")
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        print(f"Pretrained model saved to {save_dir}")

    @staticmethod
    def load_pretrained(
        model_path: str,
        device: str = "cpu",
    ) -> torch.nn.Module:
        """
        Load pretrained model.

        Args:
            model_path: Path to model checkpoint or directory
            device: Device to load model to

        Returns:
            Loaded model
        """
        model_path = Path(model_path)

        # If directory provided, look for model.pth
        if model_path.is_dir():
            model_path = model_path / "model.pth"

        result = CheckpointManager.load_checkpoint(str(model_path), device=device)
        model = result["model"]
        model.eval()  # Set to evaluation mode

        return model
