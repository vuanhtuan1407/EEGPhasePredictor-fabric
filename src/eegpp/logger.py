import os
from pathlib import Path
from typing import Optional

import pandas as pd

from out import OUT_DIR


class MyLogger(object):
    def __init__(self, log_dir=None):
        if log_dir is None:
            log_dir = f'{OUT_DIR}/logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        self.fold_state_dict = {'epoch': [], 'fold': []}
        self.global_state_dict = {'epoch': []}
        self.glb_flag = True

    def update_epoch(self, epoch, fold: Optional[int] = None):
        if fold is not None:
            self.glb_flag = False
            # update for both training and validation
            self.fold_state_dict['epoch'].append(epoch)
            self.fold_state_dict['fold'].append(fold)
        else:
            self.glb_flag = True
            self.global_state_dict['epoch'].append(epoch)

    def log_dict(self, state: dict):
        if self.glb_flag is False:
            for k, v in state.items():
                if k not in self.fold_state_dict.keys():
                    self.fold_state_dict[k] = []
                    self.fold_state_dict[k].append(v)
                else:
                    self.fold_state_dict[k].append(v)
        else:
            for k, v in state.items():
                if k not in self.global_state_dict.keys():
                    self.global_state_dict[k] = []
                    self.global_state_dict[k].append(v)
                else:
                    self.global_state_dict[k].append(v)

    def save_to_csv(self):
        global_state_df = pd.DataFrame.from_dict(self.global_state_dict)
        fold_state_df = pd.DataFrame.from_dict(self.fold_state_dict)
        global_state_df.to_csv(str(Path(self.log_dir, 'my_global_logs.csv')), index=False,
                               header=True, float_format="%.4f")
        fold_state_df.to_csv(str(Path(self.log_dir, "my_fold_logs.csv")), index=False,
                             header=True, float_format="%.4f")
