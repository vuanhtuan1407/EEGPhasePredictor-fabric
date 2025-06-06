import os

import dropbox
import joblib
import numpy as np
from tqdm import tqdm

from ..data import SEQ_FILES, LABEL_FILES, DUMP_DATA_FILES
from .. import params
from ..utils.common_utils import get_path_slash, convert_ms2datetime, convert_datetime2ms

LABEL_DICT = {0: "W", 1: "W*", 2: "NR", 3: "NR*", 4: "R", 5: "R*", -1: "others"}

PATH_SLASH = get_path_slash()

INF_V = 1e6

SEP = None
def get_lb_idx(lb_text):
    lb_idx = -1
    for k, v in LABEL_DICT.items():
        if v == lb_text:
            lb_idx = k
            break
    return lb_idx


def dump_seq_with_labels(seq_files=SEQ_FILES, lb_files=LABEL_FILES, save_files=DUMP_DATA_FILES["train"]):
    try:
        all_start_ms, all_eeg, all_emg, all_mot, all_lbs, all_mxs = load_seq_with_labels(seq_files, lb_files)
        for i, (start_ms, eeg, emg, mot, lbs, mxs) in enumerate(
                zip(all_start_ms, all_eeg, all_emg, all_mot, all_lbs, all_mxs)):
            print(f'Dumping data in file {save_files[i]}')
            start_datetime = [convert_ms2datetime(ms) for ms in start_ms]
            joblib.dump((start_datetime, eeg, emg, mot, lbs, mxs), save_files[i])
    except Exception as e:
        raise e


def dump_seq_with_no_labels(seq_files=SEQ_FILES, step_ms=4000, save_files=DUMP_DATA_FILES["infer"]):
    try:
        all_start_ms, all_eeg, all_emg, all_mot, all_mxs = load_seq_only(seq_files, step_ms)
        for i, (start_ms, eeg, emg, mot, mxs) in enumerate(zip(all_start_ms, all_eeg, all_emg, all_mot, all_mxs)):
            print(f'Dumping data in file {save_files[i]}')
            start_datetime = [convert_ms2datetime(ms) for ms in start_ms]
            joblib.dump((start_datetime, eeg, emg, mot, mxs), save_files[i])
    except Exception as e:
        raise e


def load_seq_with_labels(seq_files=SEQ_FILES, lb_files=LABEL_FILES):
    all_start_ms, all_lbs = load_lbs(lb_files)
    print('Processing sequences...')
    all_eeg, all_emg, all_mot, all_mxs = [[], [], [], []]
    global SEP
    for i, seq_file in enumerate(seq_files):
        print(seq_file)
        start_ms = all_start_ms[i]
        lbs = all_lbs[i]
        eeg, emg, mot = [[], [], []]
        mxs = [-INF_V, -INF_V, -INF_V]
        tmp_idx = 0
        with open(seq_file, 'r', encoding='utf-8', errors='replace') as f:
            data = f.readlines()
            start_line = 0
            while not data[start_line].__contains__('Time') and start_line < len(data):
                start_line = start_line + 1
            tmp_eeg, tmp_emg, tmp_mot = [[], [], []]
            s = 0
            for line in tqdm(data[start_line + 1:], total=len(data[start_line + 1:]),
                             desc=seq_file.split(PATH_SLASH)[-1]):
                s += 1
                if SEP is None or s==1:
                    if line.__contains__(','):
                        SEP = ","
                    else:
                        SEP = "\t"
                info = line.split(SEP)

                if len(info) == 2:  # Final line in raw_S1_EEG1_23 hr.txt
                    dt, values = info[0], (info[1], 0.0, 0.0)  # Fill missing value
                else:
                    dt, values = info[0], (info[1], info[2], info[3])

                for j, value in enumerate(values):
                    value = float(value)
                    if abs(value) > mxs[j]:
                        mxs[j] = abs(float(value))

                ms = convert_datetime2ms(dt)
                if tmp_idx < len(start_ms) - 1 and ms == start_ms[tmp_idx + 1]:
                    if len(tmp_eeg) >= params.MAX_SEQ_SIZE and len(tmp_emg) >= params.MAX_SEQ_SIZE and len(
                            tmp_mot) == params.MAX_SEQ_SIZE:
                        eeg.append(tmp_eeg[:params.MAX_SEQ_SIZE])
                        emg.append(tmp_emg[:params.MAX_SEQ_SIZE])
                        mot.append(tmp_mot[:params.MAX_SEQ_SIZE])
                    else:
                        print("Sequence size less than default. Ignore this segment")

                    tmp_idx += 1
                    tmp_eeg, tmp_emg, tmp_mot = [[], [], []]

                tmp_eeg.append(float(values[0]))
                tmp_emg.append(float(values[1]))
                tmp_mot.append(float(values[2]))

            # update if num lbs > num segments and
            if len(tmp_eeg) >= params.MAX_SEQ_SIZE and len(tmp_emg) >= params.MAX_SEQ_SIZE and len(
                    tmp_mot) == params.MAX_SEQ_SIZE:
                eeg.append(tmp_eeg)
                emg.append(tmp_emg)
                mot.append(tmp_mot)
                tmp_idx += 1
            else:
                print("Sequence size less than default. Ignore this segment")

            all_start_ms[i] = start_ms[: tmp_idx]
            all_lbs[i] = lbs[: tmp_idx]

            all_eeg.append(eeg)
            all_emg.append(emg)
            all_mot.append(mot)
            all_mxs.append(mxs)

    return all_start_ms, all_eeg, all_emg, all_mot, all_lbs, all_mxs


def load_seq_only(data_files=SEQ_FILES, step_ms=None):
    if step_ms is None:
        step_ms = 4000
    print('Processing sequences...')
    print(data_files)
    global SEP
    all_start_ms, all_eeg, all_emg, all_mot, all_mxs = [[], [], [], [], []]
    for data_file in data_files:
        start_ms, eeg, emg, mot = [[], [], [], []]
        mxs = [-INF_V, -INF_V, -INF_V]
        tmp_ms = 0
        with open(data_file, 'r', encoding='utf-8', errors='replace') as f:
            data = f.readlines()
            start_line = 0
            while not data[start_line].__contains__('Time') and start_line < len(data):
                start_line = start_line + 1
            tmp_eeg, tmp_emg, tmp_mot = [[], [], []]
            s = 0
            for line in tqdm(data[start_line + 1:], total=len(data[start_line + 1:]),
                             desc=data_file.split(PATH_SLASH)[-1]):
                s += 1
                if SEP is None or s == 1:
                    if line.__contains__(','):
                        SEP = ","
                    else:
                        SEP = "\t"
                info = line.split(SEP)
                if len(info) == 2:  # Final line in raw_S1_EEG1_23 hr.txt
                    dt, values = info[0], (info[1], 0.0, 0.0)  # Fill missing value
                else:
                    dt, values = info[0], (info[1], info[2], info[3])

                for j, value in enumerate(values):
                    value = float(value)
                    if abs(float(value)) > mxs[j]:
                        mxs[j] = abs(float(value))

                ms = convert_datetime2ms(dt)
                if tmp_ms == 0:
                    tmp_ms = ms
                if ms - tmp_ms >= step_ms:
                    start_ms.append(tmp_ms)
                    eeg.append(tmp_eeg)
                    emg.append(tmp_emg)
                    mot.append(tmp_mot)
                    tmp_ms = ms
                    tmp_eeg, tmp_emg, tmp_mot = [[], [], []]

                tmp_eeg.append(float(values[0]))
                tmp_emg.append(float(values[1]))
                tmp_mot.append(float(values[2]))

            start_ms.append(tmp_ms)
            eeg.append(tmp_eeg)
            emg.append(tmp_emg)
            mot.append(tmp_mot)

        all_start_ms.append(start_ms)
        all_eeg.append(eeg)
        all_emg.append(emg)
        all_mot.append(mot)
        all_mxs.append(mxs)

    return all_start_ms, all_eeg, all_emg, all_mot, all_mxs


def load_lbs(data_files=LABEL_FILES):
    print("Processing labels...")
    all_start_ms, all_lbs = [[], []]
    for data_file in data_files:
        with open(data_file, 'r', errors='replace', encoding='utf-8') as f:
            data = f.readlines()
            start_line = 0
            while not data[start_line].__contains__('Time') and start_line < len(data):
                start_line = start_line + 1
            tmp_ms, tmp_lbs = [[], []]
            for line in tqdm(data[start_line + 1:-1], total=len(data[start_line + 1:-1]),
                             desc=data_file.split(PATH_SLASH)[-1]):
                _, lb_text, dt = line.split('\t')[:3]
                ms = convert_datetime2ms(dt)
                lb = get_lb_idx(lb_text)
                tmp_ms.append(ms)
                tmp_lbs.append(lb)
            all_start_ms.append(tmp_ms)
            all_lbs.append(tmp_lbs)

    return all_start_ms, all_lbs


def get_dataset_train(remote_type='dump'):
    # remote_type in ['raw', 'dump']
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

    for i in range(3):
        print(f'Downloading {DUMP_DATA_FILES["train"][i].split(get_path_slash())[-1]}...')
        dbx.files_download_to_file(
            download_path=DUMP_DATA_FILES['train'][i],
            path=f'/data/{remote_type}/{DUMP_DATA_FILES["train"][i].split(get_path_slash())[-1]}',
        )


def create_new_dataset_objdet():
    all_start_ms, all_lbs = load_lbs()
    all_transfer_ms = []
    for j, lbs in enumerate(all_lbs):
        start_ms = all_start_ms[j]
        transfer_ms = []
        for i in range(len(lbs) - 1):
            lb_i = lbs[i]
            lb_ii = lbs[i + 1]
            # c1 = lb_i % 2 == 0 and lb_ii % 2 == 0 and lb_i != lb_ii and lb_i != -1 and lb_ii != -1
            # c2 = lb_i != lb_ii and lb_i != -1 and lb_ii != -1
            c3 = lb_i != -1 and lb_ii != -1 and abs(lb_i - lb_ii) >= 1 and (lb_i % 2 == 0 or lb_ii % 2 == 0)
            if c3:
                transfer_ms.append(start_ms[i + 1])
        print(np.array(transfer_ms).shape)
        all_transfer_ms.append(transfer_ms)

    # print(all_transfer_ms)
    return all_transfer_ms


if __name__ == '__main__':
    # os.makedirs(DUMP_DATA_DIR, exist_ok=True)
    dump_seq_with_labels()
    # load_seq_only(step_ms=4000)
    # load_seq_with_labels()
    # create_new_dataset_objdet()
