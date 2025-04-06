from torchvision import datasets, transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from .basedataset import BaseDataset
from torch import Tensor

class EMNIST(BaseDataset):
    def __init__(self, split : str, img_size : int = 32):
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((img_size, img_size)),
                ])
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
    
class FilteredMNIST(EMNISTNoLabel):
    def __init__(self, digit : int, img_size : int = 32):
        super().__init__(split="digits", img_size=img_size)
        self.indices = [i for i, label in enumerate(self.targets) if label == digit]

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        idx = self.indices[idx]
        return super().__getitem__(idx)
     
if __name__ == "__main__":
    dataset = FilteredMNIST(digit=2)
    fig, axs = plt.subplots(1, 5)
    for i, ax in enumerate(axs):
        ax.imshow(dataset[i][0].squeeze() * 0.5 + 0.5, cmap="gray")
    plt.savefig("data/samples/filtered_mnist.png")