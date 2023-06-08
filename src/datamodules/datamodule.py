import logging
from typing import Tuple, Optional

import numpy as np
import random

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)

class DataModule(pl.LightningDataModule):
    """Class implementing LightningDataModule."""

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset = None,
        batch_size: int = 16, 
        num_workers: int = 2, 
        seed: int = 123,
    ) -> None:
        super().__init__()

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage: Optional[str] = None) -> Tuple[Dataset, Dataset]:
        """Load experiments' dataset."""
        
        if stage is None: 
            if self.test_dataset is None: 
                self.train_dataset, self.test_dataset = self._split_data_(dataset=self.train_dataset,
                                                                          test_size=0.3, shuffle=True)
            return self.train_dataset, self.test_dataset
        else:
            pass


    def train_dataloader(self):
        """Function returning train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Function returning validation dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Function returning test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _split_data_(self, dataset: Dataset, test_size: float = 0.3, 
                     shuffle: bool = True) -> Tuple[Dataset, Dataset]:
        """Split dataset on train and test.

        :param dataset: dataset for splitting
        :param test_size: size of test data
        :param shuffle: if True, shuffle data
        :return: tuple of
            - train dataset
            - test dataset
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        len_dataset = len(dataset)
        idx = np.arange(len_dataset)

        if shuffle:
            train_idx = random.sample(list(idx), int((1 - test_size) * len_dataset))
        else:
            train_idx = idx[: -int(test_size * len_dataset)]
        test_idx = np.setdiff1d(idx, train_idx)

        train_set = Subset(dataset, train_idx)
        test_set = Subset(dataset, test_idx)
        return train_set, test_set
