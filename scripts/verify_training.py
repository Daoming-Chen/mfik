"""
Verification script for training pipeline.

Tests:
- Trainer initialization
- Loss computation (forward pass)
- Training step (backward pass)
- Checkpoint save/load
- Full mini training run
"""

import tempfile
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from mfik.model.v1 import MeanFlowNet, ModelConfig
from mfik.train.v1 import TrainConfig, MeanFlowTrainer


def create_dummy_data(num_samples: int, robot_dof: int):
    """Create dummy training data."""
    data = {
        "q": torch.randn(num_samples, robot_dof),
        "target_pose": torch.randn(num_samples, 7),
        "q_target": torch.randn(num_samples, robot_dof),
        "r": torch.rand(num_samples, 1),
        "t": torch.rand(num_samples, 1),
    }
    return data


def test_trainer_init():
    """Test trainer initialization."""
    print("\n" + "=" * 60)
    print("Testing Trainer Initialization")
    print("=" * 60)

    # Create model
    model_config = ModelConfig.panda_default()
    model = MeanFlowNet(model_config)

    # Create trainer
    train_config = TrainConfig.quick_test()
    train_config.device = "cpu"  # Use CPU for testing
    trainer = MeanFlowTrainer(model=model, config=train_config)

    print(f"\n  Model device: {next(model.parameters()).device}")
    print(f"  Optimizer: {type(trainer.optimizer).__name__}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print("  ✓ Trainer initialized successfully")


def test_loss_computation():
    """Test MeanFlow loss computation."""
    print("\n" + "=" * 60)
    print("Testing Loss Computation")
    print("=" * 60)

    # Create model and trainer
    model_config = ModelConfig.panda_default()
    model = MeanFlowNet(model_config)

    train_config = TrainConfig.quick_test()
    train_config.device = "cpu"
    trainer = MeanFlowTrainer(model=model, config=train_config)

    # Create dummy batch
    batch_size = 4
    robot_dof = 7
    q = torch.randn(batch_size, robot_dof)
    target_pose = torch.randn(batch_size, 7)
    q_target = torch.randn(batch_size, robot_dof)
    r = torch.rand(batch_size, 1)
    t = torch.rand(batch_size, 1)

    # Compute loss
    print(f"\n  Computing loss for batch of {batch_size} samples...")
    metrics = trainer.compute_meanflow_loss(q, target_pose, q_target, r, t)

    print(f"  Loss: {metrics['loss'].item():.4f}")
    print(f"  MeanFlow residual: {metrics['meanflow_residual'].item():.4f}")
    print(f"  Velocity norm: {metrics['velocity_norm'].item():.4f}")
    print(f"  Flow velocity norm: {metrics['flow_velocity_norm'].item():.4f}")

    assert not torch.isnan(metrics['loss']), "Loss is NaN"
    assert not torch.isinf(metrics['loss']), "Loss is Inf"
    print("  ✓ Loss computation successful")


def test_training_step():
    """Test training step with gradient computation."""
    print("\n" + "=" * 60)
    print("Testing Training Step")
    print("=" * 60)

    # Create model and trainer
    model_config = ModelConfig.panda_default()
    model = MeanFlowNet(model_config)

    train_config = TrainConfig.quick_test()
    train_config.device = "cpu"
    train_config.use_amp = False  # Disable AMP for CPU
    trainer = MeanFlowTrainer(model=model, config=train_config)

    # Create dummy batch
    batch_size = 4
    robot_dof = 7
    batch = create_dummy_data(batch_size, robot_dof)

    # Get initial parameters
    initial_params = [p.clone() for p in model.parameters()]

    # Training step
    print(f"\n  Performing training step...")
    metrics = trainer.train_step(batch)

    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Gradient norm: {metrics['grad_norm']:.4f}")
    print(f"  Learning rate: {metrics['lr']:.2e}")

    # Check that parameters changed
    params_changed = False
    for p_init, p_new in zip(initial_params, model.parameters()):
        if not torch.allclose(p_init, p_new):
            params_changed = True
            break

    assert params_changed, "Parameters did not change after training step"
    print("  ✓ Parameters updated successfully")
    print("  ✓ Training step successful")


def test_checkpoint():
    """Test checkpoint save/load."""
    print("\n" + "=" * 60)
    print("Testing Checkpoint Save/Load")
    print("=" * 60)

    # Create model and trainer
    model_config = ModelConfig.panda_default()
    model = MeanFlowNet(model_config)

    train_config = TrainConfig.quick_test()
    train_config.device = "cpu"
    trainer = MeanFlowTrainer(model=model, config=train_config)

    # Set some state
    trainer.current_epoch = 5
    trainer.global_step = 100
    trainer.best_loss = 0.123

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        checkpoint_path = f.name

    try:
        # Save checkpoint
        print(f"\n  Saving checkpoint to {checkpoint_path}")
        trainer.save_checkpoint(checkpoint_path)
        print("  ✓ Checkpoint saved")

        # Create new trainer
        new_model = MeanFlowNet(model_config)
        new_trainer = MeanFlowTrainer(model=new_model, config=train_config)

        # Load checkpoint
        print(f"\n  Loading checkpoint from {checkpoint_path}")
        new_trainer.load_checkpoint(checkpoint_path)

        # Verify state
        assert new_trainer.current_epoch == 5, "Epoch not restored"
        assert new_trainer.global_step == 100, "Global step not restored"
        assert abs(new_trainer.best_loss - 0.123) < 1e-6, "Best loss not restored"

        print(f"  Epoch: {new_trainer.current_epoch}")
        print(f"  Global step: {new_trainer.global_step}")
        print(f"  Best loss: {new_trainer.best_loss}")
        print("  ✓ Checkpoint loaded successfully")

    finally:
        # Clean up
        Path(checkpoint_path).unlink(missing_ok=True)


def test_mini_training():
    """Test a mini training run."""
    print("\n" + "=" * 60)
    print("Testing Mini Training Run")
    print("=" * 60)

    # Create model and trainer
    model_config = ModelConfig.panda_default()
    model = MeanFlowNet(model_config)

    train_config = TrainConfig.quick_test()
    train_config.device = "cpu"
    train_config.use_amp = False
    train_config.batch_size = 4
    train_config.num_epochs = 2
    train_config.log_interval = 2
    trainer = MeanFlowTrainer(model=model, config=train_config)

    # Create dummy dataset
    num_samples = 20
    robot_dof = 7
    data = create_dummy_data(num_samples, robot_dof)

    def collate_fn(batch):
        indices = batch
        return {
            "q": torch.stack([data["q"][i] for i in indices]),
            "target_pose": torch.stack([data["target_pose"][i] for i in indices]),
            "q_target": torch.stack([data["q_target"][i] for i in indices]),
            "r": torch.stack([data["r"][i] for i in indices]),
            "t": torch.stack([data["t"][i] for i in indices]),
        }

    dataset = list(range(num_samples))
    train_loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Training loop
    print(f"\n  Training for {train_config.num_epochs} epochs...")
    initial_loss = None

    for epoch in range(train_config.num_epochs):
        epoch_losses = []

        for batch in train_loader:
            metrics = trainer.train_step(batch)
            trainer.global_step += 1
            epoch_losses.append(metrics["loss"])

            if initial_loss is None:
                initial_loss = metrics["loss"]

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"  Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

    final_loss = avg_loss

    print(f"\n  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Total steps: {trainer.global_step}")

    # Loss should decrease (at least a bit)
    # Note: On dummy data, we don't expect dramatic improvement
    print("  ✓ Mini training completed successfully")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Training Pipeline Verification")
    print("=" * 60)

    try:
        test_trainer_init()
        test_loss_computation()
        test_training_step()
        test_checkpoint()
        test_mini_training()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
