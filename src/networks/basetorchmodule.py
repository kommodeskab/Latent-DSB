import torch
from src.utils import get_ckpt_path

class BaseTorchModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def _load_weights_from_experiment(self, experiment_id, old_model_keyword):
        ckpt_path = get_ckpt_path(experiment_id)
        ckpt = torch.load(ckpt_path)
        weights = {k: v for k, v in ckpt["state_dict"].items() if k.startswith(f"{old_model_keyword}.")}
        weights = {k.split(".", 1)[1]: v for k, v in weights.items()}
        self.load_state_dict(weights, strict=True)
        
class PretrainedModel(BaseTorchModule):
    def __init__(
        self, 
        module : BaseTorchModule, 
        experiment_id : str, 
        old_model_keyword : str
        ):
        super().__init__()
        self.module = module
        self.module._load_weights_from_experiment(experiment_id, old_model_keyword)
        
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)