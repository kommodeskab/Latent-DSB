import torch
from src.utils import get_ckpt_path
from hydra.utils import instantiate
import wandb
from torch.nn import Module
from pytorch_lightning import LightningModule

class PretrainedModel(torch.nn.Module):
    def __init__(
        self,
        project_name : str,
        experiment_id : str,
        model_keyword : str,
    ):
        super().__init__()
        api = wandb.Api()
        run = api.run(f"kommodeskab-danmarks-tekniske-universitet-dtu/{project_name}/{experiment_id}")
        config = run.config
        model_config = config['model'][model_keyword]
        
        dummy_model : Module = instantiate(model_config)
        print("Loading pretrained model of type", type(dummy_model))
        self.__dict__ = dummy_model.__dict__.copy()
        ckpt_path = get_ckpt_path(project_name, experiment_id)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        state_dict = ckpt['state_dict']
        state_dict = {k.replace(f'{model_keyword}.', ''): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict)
        
        self.forward = dummy_model.forward
        