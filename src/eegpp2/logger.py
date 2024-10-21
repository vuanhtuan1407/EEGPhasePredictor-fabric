import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .out import OUT_DIR


class MyLogger(object):
    def __init__(self, log_dir=None):
        if log_dir is None:
            log_dir = f'{OUT_DIR}/logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        self.fit_state_dict = {'fold': [], 'epoch': []}
        self.test_state_dict = {}
        self.flag = None

    def update_flag(self, flag, epoch: Optional[int] = None, fold: Optional[int] = None):
        if flag == 'fit':
            if epoch is None or fold is None:
                raise ValueError('Epoch and fold must be set!')
            else:
                self.fit_state_dict['fold'].append(fold)
                self.fit_state_dict['epoch'].append(epoch)

        self.flag = flag

    def log_model_summary(self, model_type, model_summary):
        with open(os.path.join(self.log_dir, f'{model_type}_summary.log'), 'w', encoding='utf-8') as f:
            f.write(model_summary)

    def log_error(self, error):
        with open(os.path.join(self.log_dir, 'error.log'), 'w', encoding='utf-8') as f:
            f.write(error)

    def log_dict(self, state: dict):
        if self.flag == 'fit':
            for k, v in state.items():
                if k not in self.fit_state_dict.keys():
                    self.fit_state_dict[k] = []
                    self.fit_state_dict[k].append(v)
                else:
                    self.fit_state_dict[k].append(v)
        elif self.flag == 'test':
            for k, v in state.items():
                if k not in self.test_state_dict.keys():
                    self.test_state_dict[k] = []
                    self.test_state_dict[k].append(v)
                else:
                    self.test_state_dict[k].append(v)
        else:
            raise ValueError(f'Flag {self.flag} not recognized!')

    def save_to_csv(self):
        if self.flag == 'fit':
            fit_state_df = pd.DataFrame.from_dict(self.fit_state_dict)
            fit_state_df.to_csv(str(Path(self.log_dir, "fit_logs.csv")), index=False,
                                header=True, float_format="%.4f")
        elif self.flag == 'test':
            test_metrics_state_df = pd.DataFrame.from_dict(self.test_state_dict)
            test_metrics_state_df.to_csv(str(Path(self.log_dir, "test_logs.csv")), index=False,
                                         header=True, float_format="%.4f")
        else:
            raise ValueError(f'Flag {self.flag} not recognized!')
