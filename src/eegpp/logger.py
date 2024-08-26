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
        self.fit_state_dict = {'epoch': [], 'fold': []}
        self.val_metrics_state_dict = {'epoch': []}
        self.test_metrics_state_dict = {}
        self.flag = None

    def update_flag(self, flag, epoch: Optional[int] = None, fold: Optional[int] = None):
        if flag == 'fit':
            if epoch is None or fold is None:
                raise ValueError('Epoch or fold must be set')
            else:
                self.fit_state_dict['epoch'].append(epoch)
                self.fit_state_dict['fold'].append(fold)
        elif flag == 'val_metrics':
            if epoch is None:
                raise ValueError('Epoch must be set')
            else:
                self.val_metrics_state_dict['epoch'].append(epoch)

        self.flag = flag

    def log_model_summary(self, model_summary):
        with open(os.path.join(self.log_dir, 'model_summary.log'), 'w') as f:
            f.write(model_summary)

    def log_dict(self, state: dict):
        if self.flag == 'fit':
            for k, v in state.items():
                if k not in self.fit_state_dict.keys():
                    self.fit_state_dict[k] = []
                    self.fit_state_dict[k].append(v)
                else:
                    self.fit_state_dict[k].append(v)
        elif self.flag == 'val_metrics':
            for k, v in state.items():
                if k not in self.val_metrics_state_dict.keys():
                    self.val_metrics_state_dict[k] = []
                    self.val_metrics_state_dict[k].append(v)
                else:
                    self.val_metrics_state_dict[k].append(v)
        elif self.flag == 'test_metrics':
            for k, v in state.items():
                if k not in self.test_metrics_state_dict.keys():
                    self.test_metrics_state_dict[k] = []
                    self.test_metrics_state_dict[k].append(v)
                else:
                    self.test_metrics_state_dict[k].append(v)

    def save_to_csv(self):
        val_metrics_state_df = pd.DataFrame.from_dict(self.val_metrics_state_dict)
        fit_state_df = pd.DataFrame.from_dict(self.fit_state_dict)
        test_metrics_state_df = pd.DataFrame.from_dict(self.test_metrics_state_dict)
        val_metrics_state_df.to_csv(str(Path(self.log_dir, 'my_val_metrics_logs.csv')), index=False,
                                    header=True, float_format="%.4f")
        fit_state_df.to_csv(str(Path(self.log_dir, "my_fit_logs.csv")), index=False,
                            header=True, float_format="%.4f")
        test_metrics_state_df.to_csv(str(Path(self.log_dir, "my_test_metrics_logs.csv")), index=False,
                                     header=True, float_format="%.4f")
