import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Subset, ConcatDataset, DataLoader, random_split

from data import DUMP_DATA_FILES
from src.eegpp import params
from src.eegpp.dataset import EEGDataset

"""
    Can not use train/val/test in KFold. Just use train/val or train/test
"""


class EEGKFoldDataLoader:
    def __init__(
            self,
            dataset_files='all',
            n_splits=5,
            n_workers=0,
            batch_size=4,
    ):
        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.datasets = None
        self.all_splits = None
        self.train_val_datasets = None
        self.fold = 0

        if isinstance(dataset_files, list):
            self.dataset_files = dataset_files
        else:
            self.dataset_files = range(len(DUMP_DATA_FILES['train']))

        self.n_splits = n_splits
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(params.RD_SEED)
        self.setup()

    def setup(self):
        print("Loading data...")
        datasets = []
        train_val_dts, test_dts = [], []
        for i in self.dataset_files:
            dump_file = DUMP_DATA_FILES['train'][i]
            print("Loading dump file {}".format(dump_file))
            i_dataset = EEGDataset(dump_file, w_out=params.W_OUT)
            datasets.append(i_dataset)
            train_val_dt, test_dt = random_split(i_dataset, [0.9, 0.1], generator=self.generator)
            train_val_dts.append(train_val_dt)
            test_dts.append(test_dt)

        self.datasets = datasets
        self.train_val_datasets = train_val_dts
        self.test_dataset = ConcatDataset(test_dts)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=params.RD_SEED)
        all_splits = []
        for i, _ in enumerate(self.datasets):
            splits = [subset for subset in kf.split(train_val_dts[i])]
            all_splits.append(splits)
        self.all_splits = all_splits

    def set_fold(self, k):
        if k is None or k < 0 or k >= self.n_splits:
            print("Fold value is invalid. Set to default value: 0")
            self.fold = 0
        else:
            self.fold = k

        train_dts, val_dts = [], []
        for i, (dataset, splits) in enumerate(zip(self.datasets, self.all_splits)):
            train_ids, val_ids = splits[k]
            train_subset_ids = np.array(self.train_val_datasets[i].indices)[train_ids]
            val_subset_ids = np.array(self.train_val_datasets[i].indices)[val_ids]
            train_dt = Subset(dataset, train_subset_ids)
            val_dt = Subset(dataset, val_subset_ids)
            train_dts.append(train_dt)
            val_dts.append(val_dt)

        self.train_dataset = ConcatDataset(train_dts)
        self.val_dataset = ConcatDataset(val_dts)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            drop_last=True
        )


if __name__ == '__main__':
    eeg_data_loader = EEGKFoldDataLoader()
    eeg_data_loader.set_fold(0)
    # dataloader = eeg_data_loader.train_dataloader()
    # for batch in dataloader:
    #     x, _, _ = batch
    #     print(x.shape)
    # pass
