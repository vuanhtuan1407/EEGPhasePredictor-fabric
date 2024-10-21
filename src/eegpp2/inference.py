import os

import dropbox
from pathlib import Path

import torch
import numpy as np
from lightning import Fabric
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import CACHE_DIR, params
from .out import OUT_DIR
from .dataset import EEGDataset
from .utils.common_utils import get_path_slash
from .utils.data_utils import dump_seq_with_no_labels, LABEL_DICT
from .utils.model_utils import get_model, check_using_ft, freeze_parameters

torch.set_float32_matmul_precision('medium')
def ensureDir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def get_dataset(opts):
    import yaml
    from pathlib import Path
    import os
    model_type = opts.model_type

    config = yaml.safe_load(open(opts.yaml_config_path))
    DATA_DIR = config["datasets"]["data_dir"]
    TMP_DIR = config["datasets"]["tmp_dir"]
    OUT_DIR = config["datasets"]["out_dir"]
    ensureDir(TMP_DIR)
    ensureDir(OUT_DIR)
    for i in range(len(config["datasets"]["seq_files"])):
        full_dump_path = Path(os.path.join(TMP_DIR, config["datasets"]["seq_files"][i])).with_suffix(".pkl")
        full_seq_path = os.path.join(DATA_DIR, config["datasets"]["seq_files"][i])

        # datasets.append([dataset, dataset.idx_2lb])

        minmax_normalized = True
        use_tmp = False
        if check_using_ft(model_type):
            minmax_normalized = False
        if not os.path.exists(full_dump_path):
            print("Generating pkl file from seq file {}...".format(full_seq_path))
            dump_seq_with_no_labels(
                seq_files=[full_seq_path],
                step_ms=4000,
                save_files=[full_dump_path],
            )
            use_tmp = True

        dataset = EEGDataset(
            dump_path=full_dump_path,
            w_out=params.W_OUT,
            minmax_normalized=minmax_normalized,
        )
        yield dataset


def download_storage(file, remote_type, default=True):
    if default:
        from .params import CDIR
        token_path = "%s/TOKEN.txt" % CDIR
    else:
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

    print(f'Downloading model {remote_type}...')
    dbx.files_download_to_file(
        download_path=str(os.path.join(OUT_DIR, 'checkpoints', file)),
        path=f'/{remote_type}/{file}',
    )


def get_checkpoint(model_type, torchscript=False, map_to_device=torch.device('cpu')):
    """
    check cache folder
    - if checkpoint exists, load checkpoint directly
    - else download checkpoint then load checkpoint
    :return:
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    if torchscript:
        # best_checkpoint = os.path.join(OUT_DIR, 'checkpoints', f'{model_type}_best_scripted.pt')
        # if not os.path.isfile(best_checkpoint):
        #     download_storage(
        #         remote_type='checkpoints',
        #         file=f'{model_type}_best_scripted.pt'
        #     )
        # model = torch.jit.load(best_checkpoint)
        # return model
        raise NotImplemented("Not Implemented for torchscript converter")
    else:
        model = get_model(model_type)
        best_checkpoint = os.path.join(OUT_DIR, 'checkpoints', f'{model_type}_best.pkl')
        if not os.path.exists(best_checkpoint):
            download_storage(
                remote_type='checkpoints',
                file=f'{model_type}_best.pkl'
            )
        state_dict = torch.load(best_checkpoint, weights_only=True, map_location=map_to_device)
        model.load_state_dict(state_dict['model_state_dict'])
        return model

def infer2(opts=None):
    import yaml
    config = yaml.safe_load(open(opts.yaml_config_path))
    OUT_DIR = config["datasets"]["out_dir"]
    DATA_DIR = config["datasets"]["data_dir"]
    model_type = opts.model_type
    SEPERATOR = "\t"
    if config["datasets"]["out_seperator"] == " ":
        SEPERATOR = " "
    TEMPLATES = None
    if "template_files" in config["datasets"] and len(config["datasets"]["template_files"]) > 0:
        TEMPLATES = config["datasets"]["template_files"]
    print(TEMPLATES)
    # model_dir = get_model_dirname()
    # utils.ensureDir(model_dir)

    batch_size  = opts.batch_size


    if torch.cuda.is_available() or torch.backends.mps.is_available():
        accelerator = 'auto'
    else:
        accelerator = 'cpu'

    fabric = Fabric(
        accelerator=accelerator,
        devices='auto'
    )


    model = get_checkpoint(model_type, torchscript=False, map_to_device=fabric.device)
    fabric.print(f"Using: {fabric.device}")
    model = fabric.setup_module(model)
    #
    # model = get_model(n_class)
    # model.load_state_dict(torch.load(params.MODEL_PATH, map_location=device))
    # model.to(device)

    datasets = get_dataset(opts)
    torch.no_grad()
    for ki, ds in enumerate(datasets):
        print("Interring...")
        ki = ki + 1
        infer_ds = ds

        BASE_NAME = Path(infer_ds.inp_path).stem
        if BASE_NAME.startswith("raw_"):
            BASE_NAME = BASE_NAME[4:]


        dataloader = DataLoader(infer_ds, batch_size=batch_size, shuffle=False, num_workers=0,drop_last=True)

        dataloader = fabric.setup_dataloaders(dataloader)
        softmax = torch.nn.Softmax(dim=-1)

        test_pred = []
        test_pred_binary = []
        model.eval()
        predicted_lbs = []
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




        for lb_id in test_pred:
            predicted_lbs.append(LABEL_DICT[lb_id])
        fout = open("%s/%s_TMP_LBTEXT.txt" % (OUT_DIR, BASE_NAME), "w")
        for lbname in predicted_lbs:
            fout.write("%s\n" % lbname)
        fout.close()

        if TEMPLATES is not None:
            from .overwrite_template import write_file2
            template_path = "%s/%s" % (DATA_DIR, TEMPLATES[ki-1])

            out_path = Path(template_path)
            out_path = "%s/%s_final%s" % (OUT_DIR, out_path.stem, out_path.suffix)
            print("Use template: ", template_path)
            write_file2(predicted_lbs,template_path, out_path)
            print("Output: ", out_path)

def infer(data_path, infer_path=None, model_type='stftcnn1dnc', batch_size=10, n_workers=0, checkpoint_path=None,
          remove_tmp=False):
    parent_dir = (get_path_slash()).join(data_path.split(get_path_slash())[:-1])
    minmax_normalized = True
    use_tmp = False
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
        use_tmp = True

    dataset = EEGDataset(
        dump_path=dump_path,
        w_out=params.W_OUT,
        minmax_normalized=minmax_normalized,
    )

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        accelerator = 'auto'
    else:
        accelerator = 'cpu'

    fabric = Fabric(
        accelerator=accelerator,
        devices='auto'
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    model = get_checkpoint(model_type, torchscript=False, map_to_device=fabric.device)

    fabric.print(f"Using: {fabric.device}")

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
    print("Inference file saved successfully.")
    if use_tmp and remove_tmp:
        os.remove(dump_path)


if __name__ == '__main__':
    infer(data_path='cache/dump_eeg_1_infer.pkl', infer_path='../../inference_result.txt')
