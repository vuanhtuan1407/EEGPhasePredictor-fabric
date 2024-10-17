import argparse

from src.eegpp2 import params
from src.eegpp2.inference import infer
from src.eegpp2.trainer import EEGKFoldTrainer
from src.eegpp2.visualization import visualize_results


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='infer', choices=['infer', 'train'])
    parser.add_argument("--model_type", type=str, default="stftcnn1dnc")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--n_splits", type=int, default=2)
    parser.add_argument("--n_workers", type=int, default=0)
    parser.add_argument("--resume_checkpoint", type=bool, default=False)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument('--auto_visualize', type=bool, default=True)
    parser.add_argument("--early_stopping", type=int, default=None)
    parser.add_argument("--export_torchscript", type=bool, default=False)
    parser.add_argument('--data_file', type=str, default='default',
                        help='Need to specify data file path when in infer mode')
    parser.add_argument("--infer_path", type=str, default=None)
    parser.add_argument("--remove_tmp", type=bool, default=True)
    return parser.parse_args()


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='default')
    parser.add_argument("--infer_path", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="stftcnn1dnc")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--n_workers", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    return parser.parse_args()


def run():
    args = parse_arguments()
    if args.mode == 'train':
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
            resume_checkpoint=args.resume_checkpoint,
            checkpoint_path=args.checkpoint_path,
        )
        trainer.fit()
        trainer.test()

        if args.auto_visualize:
            visualize_results(args.model_type)
    else:
        # opts = parse_options()
        opts = args  # dummy code
        if args.data_file == 'default':
            raise ValueError('--data_file must be specified in infer mode')
        else:
            infer(
                data_path=opts.data_file,
                infer_path=opts.infer_path,
                model_type=opts.model_type,
                batch_size=opts.batch_size,
                n_workers=opts.n_workers,
                checkpoint_path=opts.checkpoint_path,
                remove_tmp=opts.remove_tmp,
            )


if __name__ == "__main__":
    run()

    # import torch
    #
    # t = torch.rand(2, 5120)
    # window = torch.hamming_window(2048)
    # rf = torch.fft.rfft(t)
    # rs = torch.stft(t, n_fft=2048, win_length=2048, hop_length=512, normalized=True, return_complex=True, window=window)
    # rt = torch.sqrt(rs.real ** 2 + rs.imag ** 2)
    # print(rt, rt.shape)
