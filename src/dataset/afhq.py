import os
from torchvision import transforms
from PIL import Image
import torch
from src.dataset.basedataset import ImageDataset, BaseDataset
from typing import Literal

class AFHQ(BaseDataset):
    """
    Crawls the AFHQ dataset
    The dataset is located in the following structure:
    - data
        - afhq
            - train
                - cat
                - dog
                - wild
            - val
                - cat
                - dog
                - wild
    """
    def __init__(self, split : Literal['dog', 'cat'], train : bool = True):
        super().__init__()
        train = "train" if train else "val"
        root = os.path.join(f"{self.data_path}/afhq/{train}", split)
        self.files = [os.path.join(root, file) for file in os.listdir(root)]
        self.to_tensor = transforms.ToTensor()
            
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx) -> torch.Tensor:
        img = Image.open(self.files[idx])
        img = self.to_tensor(img)
        return img
    
class AFHQDataset(ImageDataset):
    def __init__(
        self,
        split : str,
        train : bool,
        img_size : int,
    ):
        dataset = AFHQ(split = split, train = train)
        super().__init__(dataset, img_size)
    
if __name__ == "__main__":
    dataset = AFHQDataset(split = "cat", train = True)
    print(len(dataset))