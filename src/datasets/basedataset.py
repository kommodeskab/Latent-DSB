from torch.utils.data import Dataset
import hashlib
from src import Sample
import os
from dotenv import load_dotenv

load_dotenv()


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    @property
    def unique_identifier(self):
        """
        A unique identifier of this class.
        This is based on the unique properties of the class, i.e. the attributes defined in __init__.

        Returns:
            str: A unique identifier string.
        """
        attr_str = str(vars(self))
        return hashlib.md5(attr_str.encode()).hexdigest()

    @property
    def data_path(self) -> str:
        """
        Returns the `DATA_PATH` variable defined in the `.env` file.
        This is used to define where datasets are stored.
        Usually, `DATA_PATH="/data"`.

        Returns:
            str: The data path.
        """
        return os.getenv("DATA_PATH")

    def __len__(self) -> int:
        raise NotImplementedError("Length method not implemented")

    def __getitem__(self, index: int) -> Sample:
        raise NotImplementedError("Get item method not implemented")
