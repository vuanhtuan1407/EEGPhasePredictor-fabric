import os
from pathlib import Path

DATA_DIR = os.path.abspath(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
DUMP_DATA_DIR = os.path.join(DATA_DIR, 'dump')

SEQ_FILES = [
    str(Path(RAW_DATA_DIR, "raw_K3_EEG3_11h.txt")),
    str(Path(RAW_DATA_DIR, "raw_RS2_EEG1_23 hr.txt")),
    str(Path(RAW_DATA_DIR, "raw_S1_EEG1_23 hr.txt")),
]

LABEL_FILES = [
    str(Path(RAW_DATA_DIR, "K3_EEG3_11h.txt")),
    str(Path(RAW_DATA_DIR, "RS2_EEG1_23 hr.txt")),
    str(Path(RAW_DATA_DIR, "S1_EEG1_23 hr.txt")),
]

DUMP_DATA_FILES = {
    "train": [
        str(Path(DUMP_DATA_DIR, "dump_eeg_1.pkl")),
        str(Path(DUMP_DATA_DIR, "dump_eeg_2.pkl")),
        str(Path(DUMP_DATA_DIR, "dump_eeg_3.pkl")),
    ],
    "infer": [
        str(Path(DUMP_DATA_DIR, "dump_eeg_1_infer.pkl")),
        str(Path(DUMP_DATA_DIR, "dump_eeg_2_infer.pkl")),
        str(Path(DUMP_DATA_DIR, "dump_eeg_3_infer.pkl")),
    ]
}
