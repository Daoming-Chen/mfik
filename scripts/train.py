"""
Training script for MeanFlow IK models.

Usage:
    python scripts/train.py --robot panda --data_path data/panda_train.pt
"""

import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mfik.model.v1 import MeanFlowNet, ModelConfig
from mfik.train.v1 import TrainConfig, MeanFlowTrainer, create_scheduler


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    output_dir: Path,
):
    """
    Main training loop.

    Args:
        model: MeanFlow model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        output_dir: Output directory for checkpoints and logs
    """
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / config.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    trainer = MeanFlowTrainer(model=model, config=config)

    # Resume from checkpoint if specified
    if config.resume_from:
        trainer.load_checkpoint(config.resume_from)

    # Initialize scheduler
    num_training_steps = len(train_loader) * config.num_epochs
    if config.use_scheduler:
        trainer.scheduler = create_scheduler(
            optimizer=trainer.optimizer,
            scheduler_type=config.scheduler_type,
            num_training_steps=num_training_steps,
            warmup_steps=config.warmup_steps,
            min_lr=config.min_lr,
        )

    # TensorBoard writer
    log_dir = output_dir / "logs"
    trainer.writer = SummaryWriter(log_dir=str(log_dir))

    print(f"Training for {config.num_epochs} epochs")
    print(f"Total steps: {num_training_steps}")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print(f"Logs will be saved to: {log_dir}")

    # Training loop
    for epoch in range(trainer.current_epoch, config.num_epochs):
        trainer.current_epoch = epoch
        epoch_metrics = []

        # Training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            # Training step
            metrics = trainer.train_step(batch)
            trainer.global_step += 1
            epoch_metrics.append(metrics)

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "lr": f"{metrics['lr']:.2e}",
            })

            # Logging
            if trainer.global_step % config.log_interval == 0:
                for key, value in metrics.items():
                    trainer.writer.add_scalar(f"train/{key}", value, trainer.global_step)

            # Validation
            if trainer.global_step % config.eval_interval == 0:
                val_metrics = trainer.validate(val_loader)
                for key, value in val_metrics.items():
                    trainer.writer.add_scalar(f"val/{key}", value, trainer.global_step)

                print(f"\nStep {trainer.global_step} - Val Loss: {val_metrics['val_loss']:.4f}")

                # Save best model
                if val_metrics["val_loss"] < trainer.best_loss:
                    trainer.best_loss = val_metrics["val_loss"]
                    best_path = checkpoint_dir / "best_model.pth"
                    trainer.save_checkpoint(
                        str(best_path),
                        metadata={"val_loss": val_metrics["val_loss"]},
                    )
                    print(f"Saved best model to {best_path}")

            # Checkpoint saving
            if trainer.global_step % config.checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_step_{trainer.global_step}.pth"
                trainer.save_checkpoint(str(checkpoint_path))
                print(f"\nSaved checkpoint to {checkpoint_path}")

        # Epoch summary
        avg_loss = np.mean([m["loss"] for m in epoch_metrics])
        print(f"\nEpoch {epoch+1} - Average Loss: {avg_loss:.4f}")

        # Save epoch checkpoint
        epoch_checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        trainer.save_checkpoint(str(epoch_checkpoint_path))

    # Final validation
    print("\nFinal validation...")
    val_metrics = trainer.validate(val_loader)
    print(f"Final Val Loss: {val_metrics['val_loss']:.4f}")

    # Save final model
    final_path = checkpoint_dir / "final_model.pth"
    trainer.save_checkpoint(str(final_path))
    print(f"\nTraining complete! Final model saved to {final_path}")

    trainer.writer.close()


def create_dummy_dataset(num_samples: int, robot_dof: int):
    """Create dummy dataset for testing."""
    data = {
        "q": torch.randn(num_samples, robot_dof),
        "target_pose": torch.randn(num_samples, 7),
        "q_target": torch.randn(num_samples, robot_dof),
        "r": torch.rand(num_samples, 1),
        "t": torch.rand(num_samples, 1),
    }
    return torch.utils.data.TensorDataset(
        data["q"], data["target_pose"], data["q_target"], data["r"], data["t"]
    )


class MeanFlowDataset(torch.utils.data.Dataset):
    """Dataset wrapper for MeanFlow training data."""

    def __init__(self, data_path: str):
        """
        Load dataset from file.

        Args:
            data_path: Path to .pt file containing dataset
        """
        self.data = torch.load(data_path, weights_only=False)

        # Validate data format
        required_keys = ["q", "target_pose", "q_target", "r", "t"]
        for key in required_keys:
            if key not in self.data:
                raise ValueError(f"Dataset missing required key: {key}")

    def __len__(self):
        return len(self.data["q"])

    def __getitem__(self, idx):
        return {
            "q": self.data["q"][idx],
            "target_pose": self.data["target_pose"][idx],
            "q_target": self.data["q_target"][idx],
            "r": self.data["r"][idx],
            "t": self.data["t"][idx],
        }


def main():
    parser = argparse.ArgumentParser(description="Train MeanFlow IK model")
    parser.add_argument("--robot", type=str, default="panda", choices=["panda", "ur10"],
                        help="Robot type")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to training data (.pt file)")
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Path to validation data (.pt file)")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (overrides config)")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate (overrides config)")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Number of epochs (overrides config)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--quick_test", action="store_true",
                        help="Run quick test with dummy data")
    args = parser.parse_args()

    # Set seed
    set_seed(42)

    # Create model config
    if args.robot == "panda":
        model_config = ModelConfig.panda_default()
        robot_dof = 7
    elif args.robot == "ur10":
        model_config = ModelConfig.ur10_default()
        robot_dof = 6
    else:
        raise ValueError(f"Unknown robot: {args.robot}")

    # Create model
    model = MeanFlowNet(model_config)
    print(f"\nModel: {args.robot}")
    print(f"Parameters: {model.count_parameters():,}")

    # Create training config
    if args.quick_test:
        train_config = TrainConfig.quick_test()
    elif args.robot == "panda":
        train_config = TrainConfig.panda_default()
    else:
        train_config = TrainConfig.ur10_default()

    # Override config
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.learning_rate is not None:
        train_config.learning_rate = args.learning_rate
    if args.num_epochs is not None:
        train_config.num_epochs = args.num_epochs
    if args.resume_from is not None:
        train_config.resume_from = args.resume_from

    # Create datasets
    if args.quick_test:
        print("\nCreating dummy dataset for quick test...")
        train_dataset = create_dummy_dataset(1000, robot_dof)
        val_dataset = create_dummy_dataset(200, robot_dof)

        def collate_fn(batch):
            return {
                "q": torch.stack([b[0] for b in batch]),
                "target_pose": torch.stack([b[1] for b in batch]),
                "q_target": torch.stack([b[2] for b in batch]),
                "r": torch.stack([b[3] for b in batch]),
                "t": torch.stack([b[4] for b in batch]),
            }
    else:
        if args.data_path is None:
            raise ValueError("--data_path required when not using --quick_test")

        print(f"\nLoading training data from {args.data_path}")
        train_dataset = MeanFlowDataset(args.data_path)

        if args.val_data_path:
            print(f"Loading validation data from {args.val_data_path}")
            val_dataset = MeanFlowDataset(args.val_data_path)
        else:
            # Split training data
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )

        collate_fn = None

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        collate_fn=collate_fn,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Train
    output_dir = Path(args.output_dir)
    train(model, train_loader, val_loader, train_config, output_dir)


if __name__ == "__main__":
    main()
