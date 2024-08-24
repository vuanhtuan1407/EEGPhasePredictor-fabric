from typing import Literal

import joblib
import torch
from torch.utils.data import Dataset

from src.eegpp import params
from src.eegpp.utils.data_utils import LABEL_DICT


class EEGDataset(Dataset):
    def __init__(self, dump_path: str, w_out=3, contain_side: Literal['left', 'right', 'both', 'none'] = 'both',
                 is_infer=False):
        """
        EEG Dataset
        :param dump_path:
        :param w_out: must be an odd if contain == 'both'
        :param contain_side:
        """
        # data = (start_datetime, eeg, emg, mot, [lbs], mxs)
        self.is_infer = is_infer
        self.w_out = w_out
        self.contain_side = contain_side
        if not is_infer:
            self.start_datetime, self.eeg, self.emg, self.mot, self.lbs, self.mxs = joblib.load(dump_path)
        else:
            self.start_datetime, self.eeg, self.emg, self.mot, self.mxs = joblib.load(dump_path)
            self.lbs = []
        self.segment_length = params.MAX_SEQ_SIZE

    def __len__(self):
        return len(self.start_datetime)

    def __getitem__(self, idx):
        if self.contain_side == 'none':
            return self._getseq_idx(idx), self._getlb_idx(idx)
        else:
            seqs, lbs = [[], []]
            if self.contain_side == 'right':
                for i in range(idx, idx + self.w_out):
                    seqs.append(self._getseq_idx(i))
                    lbs.append(self._getlb_idx(i))
            elif self.contain_side == 'left':
                for i in range(idx - self.w_out + 1, idx + 1):
                    seqs.append(self._getseq_idx(i))
                    lbs.append(self._getlb_idx(i))
            elif self.contain_side == 'both':
                pos_shift = int((self.w_out - 1) / 2)
                for i in range(idx - pos_shift, idx + pos_shift + 1):
                    seqs.append(self._getseq_idx(i))
                    lbs.append(self._getlb_idx(i))
            seqs = torch.concat(seqs, dim=-1)
            lbs = torch.stack(lbs)
        return seqs, lbs

    def _getseq_idx(self, idx):
        if idx < 0 or idx >= self.__len__():
            eeg = torch.zeros(self.segment_length, dtype=torch.float32)
            emg = torch.zeros(self.segment_length, dtype=torch.float32)
            mot = torch.zeros(self.segment_length, dtype=torch.float32)
        else:
            eeg = torch.tensor(self.eeg[idx], dtype=torch.float32) / self.mxs[0]
            emg = torch.tensor(self.emg[idx], dtype=torch.float32) / self.mxs[1]
            mot = torch.tensor(self.mot[idx], dtype=torch.float32) / self.mxs[2]

        return torch.stack([eeg, emg, mot])


    def _getlb_idx(self, idx):
        lb = torch.zeros(len(LABEL_DICT), dtype=torch.float32)
        if not self.is_infer:
            if idx < 0 or idx >= self.__len__():
                lb_idx = -1
            else:
                lb_idx = self.lbs[idx]
            lb[lb_idx] = 1.0
            return lb
        else:
            # torch.fill(lb, -1)
            for v in lb:
                v = -1
            return lb
