import torch
from src.utils import get_ckpt_path, model_config_from_id
from hydra.utils import instantiate
from torch.nn import Module
from src.utils import filter_dict_by_prefix

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