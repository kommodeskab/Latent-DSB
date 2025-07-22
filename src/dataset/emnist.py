from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch import Tensor
from .utils import get_data_path

class EMNIST(Dataset):
    def __init__(self, split : str, img_size : int = 32):
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((img_size, img_size)),
                ])
        self.data_path = get_data_path()
        self.emnist_dataset = datasets.EMNIST(
            root=self.data_path,
            split=split,
            download=True,
            transform=transform,
        )
        super().__init__()

    @property
    def targets(self):
        return self.emnist_dataset.targets
        
    def __len__(self):
        return len(self.emnist_dataset)
    
    def __getitem__(self, idx):
        image, label = self.emnist_dataset[idx]
        image = transforms.functional.rotate(image, 90)
        image = transforms.functional.vflip(image)
        return image * 2 - 1, label
    
class EMNISTNoLabel(EMNIST):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __getitem__(self, idx) -> Tensor:
        image, _ = super().__getitem__(idx)
        return image