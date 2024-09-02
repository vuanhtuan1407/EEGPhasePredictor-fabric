# import joblib

import joblib
import torch
from torch.utils.data import random_split
from tqdm import tqdm

from data import SEQ_FILES, LABEL_FILES, DUMP_DATA_FILES
from src.eegpp import params
from src.eegpp.utils.general_utils import get_path_slash, convert_ms2datetime, convert_datetime2ms

LABEL_DICT = {0: "W", 1: "W*", 2: "NR", 3: "NR*", 4: "R", 5: "R*", -1: "others"}

PATH_SLASH = get_path_slash()

INF_V = 1e6


def get_lb_idx(lb_text):
    lb_idx = -1
    for k, v in LABEL_DICT.items():
        if v == lb_text:
            lb_idx = k
            break
    return lb_idx


def dump_seq_with_labels(seq_files=SEQ_FILES, lb_files=LABEL_FILES):
    try:
        all_start_ms, all_eeg, all_emg, all_mot, all_lbs, all_mxs = load_seq_with_labels(seq_files, lb_files)
        for i, (start_ms, eeg, emg, mot, lbs, mxs) in enumerate(
                zip(all_start_ms, all_eeg, all_emg, all_mot, all_lbs, all_mxs)):
            print(f'Dumping data in file {DUMP_DATA_FILES["train"][i]}')
            start_datetime = [convert_ms2datetime(ms) for ms in start_ms]
            joblib.dump((start_datetime, eeg, emg, mot, lbs, mxs), DUMP_DATA_FILES['train'][i])
    except Exception as e:
        raise e


def dump_seq_with_no_labels(seq_files=SEQ_FILES, step_ms=4000):
    try:
        all_start_ms, all_eeg, all_emg, all_mot, all_mxs = load_seq_only(seq_files, step_ms)
        for i, (start_ms, eeg, emg, mot, mxs) in enumerate(zip(all_start_ms, all_eeg, all_emg, all_mot, all_mxs)):
            print(f'Dumping data in file {DUMP_DATA_FILES["infer"][i]}')
            start_datetime = [convert_ms2datetime(ms) for ms in start_ms]
            joblib.dump((start_datetime, eeg, emg, mot, mxs), DUMP_DATA_FILES['infer'][i])
    except Exception as e:
        raise e


def load_seq_with_labels(seq_files=SEQ_FILES, lb_files=LABEL_FILES):
    all_start_ms, all_lbs = load_lbs(lb_files)
    print('Processing sequences...')
    all_eeg, all_emg, all_mot, all_mxs = [[], [], [], []]
    for i, seq_file in enumerate(seq_files):
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
            for line in tqdm(data[start_line + 1:], total=len(data[start_line + 1:]),
                             desc=seq_file.split(PATH_SLASH)[-1]):
                info = line.split('\t')
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
            for line in tqdm(data[start_line + 1:], total=len(data[start_line + 1:]),
                             desc=data_file.split(PATH_SLASH)[-1]):
                info = line.split('\t')
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


def split_dataset(dataset, train_val_test_rate: list[int] = [0.7, 0.1, 0.2], generator=None):
    if not generator:
        generator = torch.Generator().manual_seed(0)
    if sum(train_val_test_rate) != 1.0:
        raise ValueError('train + val + test must be == 1')
    else:
        train_set, val_set, test_set = random_split(dataset, train_val_test_rate, generator=generator)
        return train_set, val_set, test_set


if __name__ == '__main__':
    # os.makedirs(DUMP_DATA_DIR, exist_ok=True)
    dump_seq_with_labels()
    # load_seq_only(step_ms=4000)
    # load_seq_with_labels()
