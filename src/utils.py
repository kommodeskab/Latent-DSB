import os
import hydra
from omegaconf import DictConfig
from datetime import datetime
import glob

def get_current_time() -> str:
    now = datetime.now()
    return now.strftime("%d%m%y%H%M%S")

def instantiate_callbacks(callback_cfg : DictConfig | None) -> list:
    callbacks = []
    
    if callback_cfg is None:
        return callbacks
    
    for _, callback_params in callback_cfg.items():
        callback = hydra.utils.instantiate(callback_params)
        callbacks.append(callback)
        
    return callbacks

def get_ckpt_path(project_name : str, experiment_id : str):
    folder_to_ckpt_path = f"logs/{project_name}/{experiment_id}/checkpoints"
    ckpt_paths = glob.glob(f"{folder_to_ckpt_path}/*.ckpt")
    
    if len(ckpt_paths) == 0:
        raise FileNotFoundError("No checkpoint found")
    
    # return the latest checkpoint
    return max(ckpt_paths, key=os.path.getctime)