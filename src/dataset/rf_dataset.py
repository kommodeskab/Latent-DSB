from torch.utils.data import Dataset, DataLoader
from src.utils import get_ckpt_path, get_project_from_id
import wandb
from hydra.utils import instantiate
from src.lightning_modules import FM
from .utils import get_data_path
import os
from tqdm import tqdm
import torch

def make_reflow_dataset(
    experiment_id : str, 
    new_dataset_name : str,
    dataset : Dataset,
    batch_size : int = 128, 
    dataset_size : int | None = None,
    making_x0 : bool = True
    ) ->  None:
    project_name = get_project_from_id(experiment_id)
    api = wandb.Api()
    run = api.run(f"kommodeskab-danmarks-tekniske-universitet-dtu/{project_name}/{experiment_id}")
    config = run.config
    
    model_config = config['model']['model']
    encoder_config = config['model']['encoder_decoder']
    
    model = instantiate(model_config)
    encoder = instantiate(encoder_config)
    
    dataset_size = dataset_size or len(dataset)
    dataset.__len__ = lambda: dataset_size
    print(f'Dataset size: {len(dataset)}')
    
    ckpt_path = get_ckpt_path(experiment_id, last=False)
    try:
        fm = FM.load_from_checkpoint(ckpt_path, model=model, encoder_decoder=encoder, strict=True)
    except Exception as e:
        print("Could load with strict=True. Trying with strict=False")
        fm = FM.load_from_checkpoint(ckpt_path, model=model, encoder_decoder=encoder, strict=False)
    
    backward_str = 'making_x0' if making_x0 else 'making_x1'
    data_path = get_data_path(dataset)
    data_path = os.path.join(data_path, f'reflow_datasets/{new_dataset_name}_{backward_str}')
    os.makedirs(data_path, exist_ok=True)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True, drop_last=True)

    for i, batch in tqdm(enumerate(dataloader), desc='Generating dataset'):
        x0, x1 = batch
        x_start = x1 if making_x0 else x0
        x_start_encoded = fm.encode(x_start)
        x_end_encoded = fm.sample(x_start_encoded, return_trajectory=False)
        x_end = fm.decode(x_end_encoded)
        
        if making_x0:
            x0, x1 = x_end, x_start
        else:
            x0, x1 = x_start, x_end
        
        save_path = os.path.join(data_path, f'{i}.pt')
        torch.save({'x0': x0, 'x1': x1}, save_path)
        
class RFDataset(Dataset):
    def __init__(self, dataset_name : str, backward : bool = False):
        data_path = get_data_path()
        backward_str = 'backward' if backward else 'forward'
        data_path = os.path.join(data_path, f'reflow_datasets/{dataset_name}_{backward_str}')
        file_names = os.listdir(data_path)
        self.file_names = [os.path.join(data_path, file_name) for file_name in file_names]
        first_file = torch.load(os.path.join(data_path, self.file_names[0]))
        self.batch_size = first_file['x0'].size(0)
        
    def __len__(self):
        return len(self.file_names) * self.batch_size
    
    def load_batch(self, idx : int):
        file_name = self.file_names[idx]
        batch = torch.load(file_name)
        return batch
    