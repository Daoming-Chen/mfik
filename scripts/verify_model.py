"""
Verification script for MeanFlow model architecture.

Tests:
- Input/output dimensions
- Parameter count
- Forward pass
- Checkpoint save/load
"""

import torch
from mfik.model.v1 import MeanFlowNet, ModelConfig, CheckpointManager
from mfik.model.utils import list_versions, print_model_info


def test_model_dimensions():
    """Test that model dimensions are correct."""
    print("\n" + "=" * 60)
    print("Testing Model Dimensions")
    print("=" * 60)

    # Test Panda (7-DOF)
    config = ModelConfig.panda_default()
    model = MeanFlowNet(config)

    batch_size = 32
    input_dim = config.input_dim  # 7 + 7 + 2 = 16
    output_dim = config.output_dim  # 7

    print(f"\nPanda (7-DOF):")
    print(f"  Input dim: {input_dim} (expected: 16)")
    print(f"  Output dim: {output_dim} (expected: 7)")

    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    y = model(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")

    assert x.shape == (batch_size, input_dim), "Input shape mismatch"
    assert y.shape == (batch_size, output_dim), "Output shape mismatch"
    print("  ✓ Dimensions correct")

    # Test UR10 (6-DOF)
    config = ModelConfig.ur10_default()
    model = MeanFlowNet(config)

    input_dim = config.input_dim  # 6 + 7 + 2 = 15
    output_dim = config.output_dim  # 6

    print(f"\nUR10 (6-DOF):")
    print(f"  Input dim: {input_dim} (expected: 15)")
    print(f"  Output dim: {output_dim} (expected: 6)")

    x = torch.randn(batch_size, input_dim)
    y = model(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")

    assert x.shape == (batch_size, input_dim), "Input shape mismatch"
    assert y.shape == (batch_size, output_dim), "Output shape mismatch"
    print("  ✓ Dimensions correct")


def test_parameter_count():
    """Test parameter count."""
    print("\n" + "=" * 60)
    print("Testing Parameter Count")
    print("=" * 60)

    config = ModelConfig.panda_default()
    model = MeanFlowNet(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = model.count_parameters()

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Expected: ~10-20M parameters (with positional encoding)
    assert 10_000_000 < total_params < 30_000_000, "Parameter count out of expected range"
    print("  ✓ Parameter count in expected range (10M-30M)")

    # Print breakdown
    print("\nParameter breakdown:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.numel():,}")


def test_forward_pass():
    """Test forward pass with different batch sizes."""
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)

    config = ModelConfig.panda_default()
    model = MeanFlowNet(config)
    model.eval()

    # Test different batch sizes
    for batch_size in [1, 16, 128]:
        x = torch.randn(batch_size, config.input_dim)

        with torch.no_grad():
            y = model(x)

        print(f"\nBatch size {batch_size}:")
        print(f"  Input: {x.shape}")
        print(f"  Output: {y.shape}")
        print(f"  Output range: [{y.min():.3f}, {y.max():.3f}]")

        assert y.shape == (batch_size, config.output_dim), "Output shape mismatch"
        assert not torch.isnan(y).any(), "NaN in output"
        assert not torch.isinf(y).any(), "Inf in output"

    print("\n  ✓ Forward pass successful for all batch sizes")


def test_checkpoint():
    """Test checkpoint save/load."""
    print("\n" + "=" * 60)
    print("Testing Checkpoint Save/Load")
    print("=" * 60)

    import tempfile
    import os

    config = ModelConfig.panda_default()
    model = MeanFlowNet(config)

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        checkpoint_path = f.name

    try:
        # Save checkpoint
        print(f"\nSaving checkpoint to {checkpoint_path}")
        CheckpointManager.save_checkpoint(
            model=model,
            optimizer=None,
            config=config,
            checkpoint_path=checkpoint_path,
            metadata={"test": True, "epoch": 0},
        )
        print("  ✓ Checkpoint saved")

        # Load checkpoint
        print(f"\nLoading checkpoint from {checkpoint_path}")
        result = CheckpointManager.load_checkpoint(checkpoint_path)

        loaded_model = result["model"]
        loaded_config = result["config"]
        metadata = result["metadata"]

        print(f"  Version: {result['version']}")
        print(f"  Metadata: {metadata}")
        print("  ✓ Checkpoint loaded")

        # Verify model equivalence
        x = torch.randn(1, config.input_dim)
        with torch.no_grad():
            y1 = model(x)
            y2 = loaded_model(x)

        diff = (y1 - y2).abs().max().item()
        print(f"\nMax difference in outputs: {diff}")
        assert diff < 1e-6, "Loaded model produces different outputs"
        print("  ✓ Loaded model matches original")

    finally:
        # Clean up
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


def test_device_compatibility():
    """Test CPU/GPU compatibility."""
    print("\n" + "=" * 60)
    print("Testing Device Compatibility")
    print("=" * 60)

    config = ModelConfig.panda_default()
    model = MeanFlowNet(config)

    # CPU
    print("\nTesting on CPU:")
    x_cpu = torch.randn(8, config.input_dim)
    y_cpu = model(x_cpu)
    print(f"  Input device: {x_cpu.device}")
    print(f"  Output device: {y_cpu.device}")
    print("  ✓ CPU execution successful")

    # GPU (if available)
    if torch.cuda.is_available():
        print("\nTesting on CUDA:")
        model_cuda = model.cuda()
        x_cuda = x_cpu.cuda()
        y_cuda = model_cuda(x_cuda)
        print(f"  Input device: {x_cuda.device}")
        print(f"  Output device: {y_cuda.device}")
        assert y_cuda.device.type == "cuda", "Output not on CUDA"
        print("  ✓ CUDA execution successful")
    else:
        print("\n  CUDA not available, skipping GPU test")


def test_version_utilities():
    """Test version utilities."""
    print("\n" + "=" * 60)
    print("Testing Version Utilities")
    print("=" * 60)

    # List versions
    versions = list_versions()
    print(f"\nAvailable versions: {versions}")
    assert "v1" in versions, "v1 not found"
    assert "v2" in versions, "v2 directory not found"
    print("  ✓ Version discovery working")

    # Print model info
    print_model_info(version="v1", robot_dof=7)


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MeanFlow Model Verification")
    print("=" * 60)

    try:
        test_model_dimensions()
        test_parameter_count()
        test_forward_pass()
        test_checkpoint()
        test_device_compatibility()
        test_version_utilities()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
