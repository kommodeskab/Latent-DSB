from torchvision.transforms import ToTensor
import torch
from .basedataset import ImageDataset, BaseDataset
import torchvision

class CelebA(BaseDataset):
    def __init__(self, attr : int | None = None, on_or_off : bool | None = None):
        """
        A special version of the CelebA dataset that only returns images with a specific attribute
        Can be used to create a dataset with only smiling faces, for example.
        """
        super().__init__()
            
        self.celeba = torchvision.datasets.CelebA(
            root=self.data_path,
            split="all",
            download=True,
            transform=ToTensor(),
            target_type="attr",
        )
        # if attr is None, return all images
        self.indices = torch.arange(len(self.celeba))
        if attr is not None:
            mask = self.attr[:, attr] == on_or_off
            self.indices = self.indices[mask]
    
    @property
    def attr(self):
        return self.celeba.attr
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        idx = self.indices[idx]
        img, _ = self.celeba[idx]
        return img
    
class CelebADataset(ImageDataset):
    def __init__(
        self,
        img_size : int = 64, 
        attr : int | None = None,
        on_or_off : bool | None = None,
        augment : bool = False,
    ):
        dataset = CelebA(attr = attr, on_or_off = on_or_off)
        super().__init__(dataset, img_size, augment)
        