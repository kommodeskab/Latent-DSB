from torch.utils.data import Dataset
import hashlib
from torchvision import transforms
from .utils import get_data_path
from torch import Tensor

class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    @property
    def unique_identifier(self):
        attr_str = str(vars(self))
        return hashlib.md5(attr_str.encode()).hexdigest()
    
    @property
    def data_path(self):
        return get_data_path()
    
class ImageDataset(BaseDataset):
    def __init__(
        self,
        dataset : Dataset[Tensor],
        img_size : int,
    ):
        """
        The base image dataset class.
        Enables data augmentation and resizing.
        """
        super().__init__()
        self.dataset = dataset        
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx) -> Tensor:
        img = self.dataset[idx]
        img = self.transform(img).clamp(-1, 1)
        return img