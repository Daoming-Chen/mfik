"""
Model utilities for version discovery and dynamic loading.
"""

import importlib
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch


def list_versions() -> List[str]:
    """
    List all available model versions.

    Returns:
        List of version strings (e.g., ['v1', 'v2'])
    """
    model_dir = Path(__file__).parent
    versions = []

    for item in model_dir.iterdir():
        if item.is_dir() and item.name.startswith("v"):
            # Check if it has __init__.py
            if (item / "__init__.py").exists():
                versions.append(item.name)

    return sorted(versions)


def get_version_info(version: str) -> Dict[str, Any]:
    """
    Get information about a specific model version.

    Args:
        version: Version string (e.g., 'v1')

    Returns:
        Dictionary with version metadata
    """
    try:
        module = importlib.import_module(f"mfik.model.{version}")
        return {
            "version": version,
            "module": module.__name__,
            "version_number": getattr(module, "__version__", "unknown"),
            "available": True,
        }
    except ImportError:
        return {
            "version": version,
            "available": False,
            "error": f"Version {version} not found",
        }


def load_model(
    version: str,
    robot_dof: int,
    config_override: Optional[Dict] = None,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Dynamically load a model by version.

    Args:
        version: Model version (e.g., 'v1')
        robot_dof: Number of robot joints
        config_override: Optional config overrides
        checkpoint_path: Optional path to pretrained checkpoint
        device: Device to load model to

    Returns:
        Initialized model

    Example:
        >>> model = load_model('v1', robot_dof=7, device='cuda')
        >>> model = load_model('v1', robot_dof=7, checkpoint_path='panda_v1.pth')
    """
    if version not in list_versions():
        raise ValueError(
            f"Version {version} not found. Available: {list_versions()}"
        )

    # Import version-specific modules
    if version == "v1":
        from mfik.model.v1 import MeanFlowNet, ModelConfig
        from mfik.model.v1.checkpoint import CheckpointManager

        # Create config
        config = ModelConfig()
        config.set_robot_dims(robot_dof=robot_dof, pose_dim=7)

        if config_override:
            for key, value in config_override.items():
                setattr(config, key, value)

        # Load from checkpoint if provided
        if checkpoint_path:
            result = CheckpointManager.load_checkpoint(checkpoint_path, device=device)
            model = result["model"]
        else:
            model = MeanFlowNet(config)
            model.to(device)

    else:
        raise NotImplementedError(f"Version {version} not yet implemented")

    return model


def create_model_from_config(config_path: str, checkpoint_path: Optional[str] = None) -> torch.nn.Module:
    """
    Create model from a configuration file.

    Args:
        config_path: Path to JSON config file
        checkpoint_path: Optional checkpoint to load weights

    Returns:
        Initialized model
    """
    import json

    with open(config_path) as f:
        config_dict = json.load(f)

    version = config_dict.get("version", "v1")
    robot_dof = config_dict.get("robot_dof")

    if robot_dof is None:
        # Try to infer from output_dim
        robot_dof = config_dict.get("output_dim")

    if robot_dof is None:
        raise ValueError("Config must specify either 'robot_dof' or 'output_dim'")

    # Remove special keys
    config_override = {
        k: v for k, v in config_dict.items() if k not in ["version", "robot_dof"]
    }

    return load_model(
        version=version,
        robot_dof=robot_dof,
        config_override=config_override,
        checkpoint_path=checkpoint_path,
    )


def compare_models(version1: str, version2: str, robot_dof: int = 7) -> Dict:
    """
    Compare two model versions.

    Args:
        version1: First version
        version2: Second version
        robot_dof: Robot DOF for comparison

    Returns:
        Comparison dictionary
    """
    model1 = load_model(version1, robot_dof=robot_dof)
    model2 = load_model(version2, robot_dof=robot_dof)

    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    comparison = {
        version1: {
            "parameters": count_params(model1),
            "config": getattr(model1, "config", None),
        },
        version2: {
            "parameters": count_params(model2),
            "config": getattr(model2, "config", None),
        },
    }

    return comparison


def print_model_info(version: str = "v1", robot_dof: int = 7):
    """
    Print detailed information about a model version.

    Args:
        version: Model version
        robot_dof: Robot DOF
    """
    print(f"\n{'='*60}")
    print(f"Model Version: {version}")
    print(f"{'='*60}\n")

    # Version info
    info = get_version_info(version)
    print(f"Module: {info.get('module', 'N/A')}")
    print(f"Version Number: {info.get('version_number', 'N/A')}")
    print(f"Available: {info.get('available', False)}\n")

    if info.get("available"):
        # Create model
        model = load_model(version, robot_dof=robot_dof)

        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Robot DOF: {robot_dof}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")

        # Config
        if hasattr(model, "config"):
            print(f"\nConfiguration:")
            config = model.config
            for key, value in vars(config).items():
                if not key.startswith("_"):
                    print(f"  {key}: {value}")

        print(f"\n{'='*60}\n")
