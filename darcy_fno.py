from __future__ import annotations

import argparse
from pathlib import Path

from sciml.darcy import DarcyConfig, DarcyFNO, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Fourier Neural Operator for 2D Darcy flow.")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("darcy_data") / "Darcy_2D_data_train.mat",
        help="Path to the Darcy training dataset.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("darcy_data") / "Darcy_2D_data_test.mat",
        help="Path to the Darcy test dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "darcy_fno",
        help="Directory where plots and checkpoints will be stored.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=20, help="Mini-batch size.")
    parser.add_argument("--width", type=int, default=32, help="Channel width for the FNO.")
    parser.add_argument("--modes", type=int, default=12, help="Number of Fourier modes kept per spatial dimension.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--history-interval", type=int, default=5, help="How often to flush CSV training history.")
    parser.add_argument(
        "--checkpoint-interval", type=int, default=25, help="How often to save a resumable training checkpoint."
    )
    parser.add_argument("--resume-from", type=Path, default=None, help="Optional path to a checkpoint to resume from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DarcyConfig(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        history_interval=args.history_interval,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume_from,
    )
    artifacts = run_training(config, DarcyFNO(modes1=args.modes, modes2=args.modes, width=args.width))
    print("Saved artifacts:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
