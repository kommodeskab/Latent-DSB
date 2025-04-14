import torch
from src.utils import get_ckpt_path, get_project_from_id
from hydra.utils import instantiate
import wandb
from torch.nn import Module
from src.utils import filter_dict_by_prefix

def config_from_id(experiment_id : str) -> dict:
    project_name = get_project_from_id(experiment_id)
    api = wandb.Api()
    # TODO: make this more dynamical for other users/projects
    possible_names = [
        "kommodeskab-danmarks-tekniske-universitet-dtu",
        "bjornsandjensen-dtu",
    ]
    for name in possible_names:
        try:
            run = api.run(f"{name}/{project_name}/{experiment_id}")
            print(f"Found experiment {experiment_id} in {name}.")
            return run.config
        except wandb.errors.CommError:
            print(f"Failed to get config from wandb for {experiment_id} in {name}. Trying next.")
        
    raise ValueError(f"Could not find experiment {experiment_id} in any of the projects: {possible_names}.")

def model_config_from_id(experiment_id : str, model_keyword : str) -> dict:
    config = config_from_id(experiment_id)
    if 'PretrainedModel' in config['model'][model_keyword]['_target_']:
        new_id = config['model'][model_keyword]['experiment_id']
        return model_config_from_id(new_id, model_keyword)
    return config['model'][model_keyword]

class PretrainedModel:
    def __new__(
        cls,
        experiment_id : str,
        model_keyword : str,
        ckpt_filename : str | None = None,
    ) -> Module:
        model_config = model_config_from_id(experiment_id, model_keyword)
        dummy_model : Module = instantiate(model_config)
        ckpt_path = get_ckpt_path(experiment_id, last=False, filename=ckpt_filename)
        print("Loading pretrained model of type", type(dummy_model))
        print("Loading checkpoint from", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict : dict[str, Module] = ckpt['state_dict']
        state_dict = filter_dict_by_prefix(state_dict, [f'{model_keyword}.'], remove_prefix=True)
        dummy_model.load_state_dict(state_dict)
        return dummy_model