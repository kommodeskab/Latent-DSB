from torch.utils.data import Dataset
from .utils import get_data_path
import os
import json

class Ears(Dataset):
    def __init__(self):
        super().__init__()
        data_path = get_data_path()
        data_path = os.path.join(data_path, 'ears')
        stats = json.load(open(os.path.join(data_path, 'speaker_statistics.json')))
        print(stats)