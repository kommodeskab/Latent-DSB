import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def split_dataset(
    train_dataset: Dataset, val_dataset: Optional[Dataset], train_val_split: Optional[float] = None
) -> tuple[Dataset, Dataset]:
    if train_val_split is not None:
        trainsize = int(train_val_split * len(train_dataset))
        train_dataset, val_dataset = random_split(train_dataset, [trainsize, len(train_dataset) - trainsize])
    return train_dataset, val_dataset


class BaseDM(pl.LightningDataModule):
    """
    A base data module for datasets.
    If no validation dataset is provided, it splits the training dataset into
    training and validation sets based on the provided 'train_val_split' ratio.
    The validation and test dataloaders are created without shuffling and drop_last.

    Args:
        trainset (Dataset): The training dataset.
        valset (Optional[Dataset], optional): The validation dataset. Defaults to None.
        testset (Optional[Dataset], optional): The test dataset. Defaults to None.
        train_val_split (Optional[float], optional): The ratio to split the training dataset for validation if 'valset' is None. Defaults to None.
        **kwargs: Additional keyword arguments for the DataLoader.
    """

    def __init__(
        self,
        trainset: Dataset,
        valset: Optional[Dataset] = None,
        testset: Optional[Dataset] = None,
        train_val_split: Optional[float] = None,
        **kwargs,
    ):
        """
        A base data module for datasets.
        It takes a dataset and splits into train and validation (if val_dataset is None).
        """
        super().__init__()
        if valset is None:
            assert train_val_split is not None, "If no valset is provided, train_val_split must be specified."

        self.trainset, self.valset = split_dataset(trainset, valset, train_val_split)
        self.testset = testset
        self.kwargs = kwargs

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            **self.kwargs,
        )

    def val_dataloader(self):
        if self.valset is None:
            return None

        kwargs = self.kwargs.copy()
        kwargs.pop("shuffle", None)
        kwargs.pop("drop_last", None)

        return DataLoader(
            dataset=self.valset,
            shuffle=False,
            drop_last=False,
            **kwargs,
        )

    def test_dataloader(self):
        if self.testset is None:
            return None

        kwargs = self.kwargs.copy()
        kwargs.pop("shuffle", None)
        kwargs.pop("drop_last", None)

        return DataLoader(
            dataset=self.testset,
            shuffle=False,
            drop_last=False,
            **kwargs,
        )
