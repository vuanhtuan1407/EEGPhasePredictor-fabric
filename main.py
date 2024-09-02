import argparse

from src.eegpp import params
from src.eegpp.trainer import EEGKFoldTrainer
from src.eegpp.visualization import visualize_results


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="fft2c")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--n_splits", type=int, default=2)
    parser.add_argument("--n_workers", type=int, default=0)
    parser.add_argument('--auto_visualize', type=bool, default=True)
    parser.add_argument("--early_stopping", type=int, default=None)
    parser.add_argument("--export_torchscript", type=bool, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    trainer = EEGKFoldTrainer(
        model_type=args.model_type,
        lr=args.lr,
        batch_size=args.batch_size,
        n_splits=args.n_splits,
        n_epochs=args.n_epochs,
        n_workers=args.n_workers,
        accelerator=params.ACCELERATOR,
        devices=params.DEVICES,
        early_stopping=None,  # Force not apply early stopping because of KFold training process
        export_torchscript=args.export_torchscript,
    )
    trainer.fit()
    trainer.test()

    if args.auto_visualize:
        visualize_results()
