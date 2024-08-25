from sklearn.model_selection import KFold
from torch.utils.data import Subset, ConcatDataset, DataLoader

from data import DUMP_DATA_FILES
from src.eegpp import params
from src.eegpp.dataset import EEGDataset


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
        self.all_splits = None
        self.datasets = None
        self.fold = None

        if isinstance(dataset_files, list):
            self.dataset_files = dataset_files
        else:
            self.dataset_files = [0, 1, 2]

        self.n_splits = n_splits
        self.n_workers = n_workers
        self.batch_size = batch_size

        self.setup()

    def setup(self):
        datasets = []
        for idx in self.dataset_files:
            dump_file = DUMP_DATA_FILES['train'][idx]
            print("Loading dump file {}".format(dump_file))
            i_dataset = EEGDataset(dump_file, w_out=params.W_OUT)
            datasets.append(i_dataset)

        self.datasets = datasets
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=params.RD_SEED)
        all_splits = []
        for dataset in datasets:
            splits = [subset for subset in kf.split(dataset)]
            all_splits.append(splits)
        self.all_splits = all_splits

    def set_fold(self, k):
        if k is None or k < 0 or k >= self.n_splits:
            print("Fold value is invalid. Set to default value of 0")
            self.fold = 0
        else:
            self.fold = k

        train_dts, val_dts = [[], []]
        for i, splits in enumerate(self.all_splits):
            train_ids, val_ids = splits[k]
            train_dt = Subset(self.datasets[i], train_ids)
            val_dt = Subset(self.datasets[i], val_ids)
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
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        pass
