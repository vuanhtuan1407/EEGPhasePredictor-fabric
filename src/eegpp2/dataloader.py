import os

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Subset, ConcatDataset, DataLoader, random_split, Sampler, RandomSampler

from .data import DUMP_DATA_FILES, DUMP_DATA_DIR
from . import params
from .dataset import EEGDataset
from .utils.data_utils import get_dataset_train

LABEL_MARKER = "EpochNo"

def fread_header_labels(inp):
    global SEP_CHECKED, SEPERATOR
    fin = open(inp, errors='ignore')

    headers = []

    while True:
        line = fin.readline()
        if line == "":
            break
        headers.append(line)
        if line.startswith(LABEL_MARKER):
            break

    return "".join(headers), fin

class EEGKFoldSampler(Sampler):
    def __init__(self, dataset, k_fold):
        super().__init__()

    def __iter__(self):
        pass

    def __len__(self):
        pass


class EEGKFoldDataLoader:
    def __init__(
            self,
            dataset_files='default',
            n_splits=5,
            n_workers=0,
            batch_size=4,
            minmax_normalized=True,
    ):
        """
        :param dataset_files: must be list of Path to dump file or "default"
        :param n_splits:
        :param n_workers:
        :param batch_size:
        :param minmax_normalized:
        """

        self.minmax_normalized = minmax_normalized

        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.datasets = None
        self.all_splits = None
        self.train_val_datasets = None
        self.current_fold = None
        self.current_epoch = None

        if isinstance(dataset_files, list):
            self.dataset_files = dataset_files
        else:
            self.prepare_default_data()
            self.dataset_files = DUMP_DATA_FILES['train']

        self.n_splits = n_splits
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.split_generator = torch.Generator().manual_seed(params.RD_SEED)
        self.dataloader_generator = torch.Generator().manual_seed(0)
        self.setup()

    @staticmethod
    def prepare_default_data():
        os.makedirs(DUMP_DATA_DIR, exist_ok=True)
        for _, _, files in os.walk(DUMP_DATA_DIR):
            if len(files) != len(DUMP_DATA_FILES['train']):
                get_dataset_train(remote_type='dump')

    def setup(self):
        print("Loading data...")
        datasets = []
        train_val_dts, test_dts = [], []
        for i, _ in enumerate(self.dataset_files):
            # print(type(DUMP_DATA_FILES), DUMP_DATA_FILES)
            dump_file = DUMP_DATA_FILES['train'][i]
            print("Loading dump file {}".format(dump_file))
            i_dataset = EEGDataset(dump_file, w_out=params.W_OUT, minmax_normalized=self.minmax_normalized)
            datasets.append(i_dataset)
            train_val_dt, test_dt = random_split(i_dataset, [0.9, 0.1], generator=self.split_generator)
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
            self.current_fold = 0
        else:
            self.current_fold = k

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

    def set_epoch(self, epoch):
        if self.current_fold is None:
            print("Fold attribute is None. Set to default value: 0")
            self.current_fold = 0
        self.current_epoch = epoch
        dataloader_generator_seed = (self.current_epoch + 1) * (self.current_fold + 1)
        self.dataloader_generator = self.dataloader_generator.manual_seed(dataloader_generator_seed)

    def train_dataloader(self, epoch):
        self.set_epoch(epoch)
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=RandomSampler(self.train_dataset, generator=self.dataloader_generator),
            num_workers=self.n_workers,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            drop_last=True,
        )


if __name__ == '__main__':
    dataloader = EEGKFoldDataLoader()
    dataloader.set_fold(0)
    train_dataloader = dataloader.train_dataloader(epoch=0)
    for i, data in enumerate(train_dataloader):
        print(data)
