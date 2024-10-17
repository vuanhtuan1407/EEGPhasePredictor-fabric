import argparse
import os

import dropbox
from pathlib import Path

import torch
import numpy as np
from lightning import Fabric
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.eegpp import CACHE_DIR, params
from src.eegpp.dataset import EEGDataset
from src.eegpp.utils.common_utils import get_path_slash
from src.eegpp.utils.data_utils import dump_seq_with_no_labels, LABEL_DICT
from src.eegpp.utils.model_utils import get_model, check_using_ft, freeze_parameters

torch.set_float32_matmul_precision('medium')


def download_storage(file, remote_type):
    token_path = input('Enter path to token file: ')
    if not os.path.exists(token_path):
        raise ValueError('Token file does not exist!')
    else:
        app_key, app_secret, refresh_token = np.loadtxt(token_path, dtype=str)

    dbx = dropbox.Dropbox(
        app_key=app_key,
        app_secret=app_secret,
        oauth2_refresh_token=refresh_token
    )

    print(f'Downloading {remote_type}...')
    dbx.files_download_to_file(
        download_path=str(Path(CACHE_DIR) / file),
        path=f'/{remote_type}/{file}',
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_token', type=str, default='')
    parser.add_argument("--model_type", type=str, default="wtresnet501dnc")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--n_workers", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument('--auto_visualize', type=bool, default=True)
    return parser.parse_args()


def get_checkpoint(model_type, torchscript=False):
    """
    check cache folder
    - if checkpoint exists, load checkpoint directly
    - else download checkpoint then load checkpoint
    :return:
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    if torchscript:
        best_checkpoint = os.path.join(CACHE_DIR, f'{model_type}_best_scripted.pt')
        if not os.path.isfile(best_checkpoint):
            download_storage(
                remote_type='checkpoints',
                file=f'{model_type}_best_scripted.pt'
            )
        model = torch.jit.load(best_checkpoint)
        return model

    else:
        model = get_model(model_type)
        best_checkpoint = os.path.join(CACHE_DIR, f'{model_type}_best.pkl')
        if not os.path.exists(best_checkpoint):
            download_storage(
                remote_type='checkpoints',
                file=f'{model_type}_best.pkl'
            )
        state_dict = torch.load(best_checkpoint, weights_only=True)
        model.load_state_dict(state_dict['model_state_dict'])
        return model


def infer(data_path, infer_path=None, model_type='stftcnn1dnc', batch_size=10, n_workers=0, checkpoint_path=None):
    parent_dir = (get_path_slash()).join(data_path.split(get_path_slash())[:-1])
    minmax_normalized = True
    if check_using_ft(model_type):
        minmax_normalized = False
    if data_path.endswith('.pkl'):
        dump_path = data_path
    else:
        t = 1
        while os.path.exists(str(Path(parent_dir) / f'dump_eeg_{t}_infer.pkl')):
            t += 1
        dump_seq_with_no_labels(
            seq_files=[data_path],
            step_ms=4000,
            save_files=[str(Path(CACHE_DIR) / f'dump_eeg_{t}_infer.pkl')],
        )
        dump_path = str(Path(CACHE_DIR) / f'dump_eeg_{t}_infer.pkl')

    dataset = EEGDataset(
        dump_path=dump_path,
        w_out=params.W_OUT,
        minmax_normalized=minmax_normalized,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    model = get_checkpoint(model_type, torchscript=False)

    fabric = Fabric(
        accelerator='auto',
        devices='auto'
    )
    model = fabric.setup_module(model)
    dataloader = fabric.setup_dataloaders(dataloader)
    softmax = torch.nn.Softmax(dim=-1)

    test_pred = []
    test_pred_binary = []

    model.eval()
    with torch.no_grad():
        # freeze_parameters(model)
        infer_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Inference")
        for batch_idx, batch in infer_bar:
            x, _, _ = batch
            pred, pred_binary = model(x)
            # ONLY TEST ON THE MAIN SEGMENT
            test_pred.append(pred[:, params.POS_IDX, :-1])  # ignore the last class
            test_pred_binary.append(pred_binary[:, params.POS_IDX, :])

        test_pred = torch.concat(test_pred, dim=0)
        test_pred_binary = torch.concat(test_pred_binary, dim=0)

        test_pred = softmax(test_pred).detach().cpu().numpy()
        test_pred_binary = softmax(test_pred_binary).detach().cpu().numpy()

        test_pred = np.argmax(test_pred, axis=-1)
        test_pred_binary = np.argmax(test_pred_binary)

    start_datetime = dataset.start_datetime

    max_len = len(start_datetime) if len(start_datetime) < len(test_pred) else len(test_pred)
    infos = []
    note = '''
        This beta file which is generated by AI tool is for labeling sleep stage.
        **The values of EpochNo are always begin at 1.   
    '''
    header = f'{"EpochNo": <7}\t{"Stage": <5}\tTime'
    infos.append(note)
    infos.append(header)
    for i in range(max_len):
        info = f'{i + 1: <7}\t{LABEL_DICT[test_pred[i]]: <5}\t{start_datetime[i]}'
        infos.append(info)

    if infer_path is None:
        infer_path = f'{parent_dir}/inference_result.txt'

    print("Saving inference file...")
    np.savetxt(infer_path, infos, fmt='%s')
    # os.remove(dump_path)


if __name__ == '__main__':
    infer(data_path='cache/dump_eeg_1_infer.pkl', infer_path='../../inference_result.txt')
