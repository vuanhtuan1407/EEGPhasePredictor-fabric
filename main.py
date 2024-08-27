import argparse

import torch

from src.eegpp import params
from src.eegpp.trainer import EEGKFoldTrainer
from src.eegpp.visualization import visualize_results

torch.set_float32_matmul_precision('medium')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="fft2c")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--n_splits", type=int, default=2)
    parser.add_argument("--n_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--auto_visualize', type=bool, default=True)
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
    )
    trainer.fit()
    trainer.test()

    if args.auto_visualize:
        visualize_results()
