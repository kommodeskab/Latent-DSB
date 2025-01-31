import os
import hydra
from omegaconf import DictConfig
from datetime import datetime
import glob
from typing import Any
import wandb

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

def filter_dict_by_prefix(d : dict[str, Any], prefixs : list[str], remove_prefix : bool = False) -> dict:
    """
    Only keep the key-value pairs in the dictionary if the key starts with any of the strings in prefix list.
    If remove_prefix is True, the prefix will be removed from the key.
    """
    new_dict = {}
    for k, v in d.items():
        for prefix in prefixs:
            if k.startswith(prefix):
                if remove_prefix:
                    new_dict[k[len(prefix):]] = v
                else:
                    new_dict[k] = v
                break
    return new_dict

def what_logs_to_delete():
    project_names = wandb.Api().projects()
    project_names = [project.name for project in project_names]
    print("It is safe to delete the following logs:")
    for project_name in project_names:
        if not os.path.exists(f"logs/{project_name}"):
            continue
        
        runs = wandb.Api().runs(project_name)
        run_ids = [run.id for run in runs]
        local_run_ids = os.listdir(f"logs/{project_name}")
        local_run_ids.sort(reverse=True)
        
        for local_run_id in local_run_ids:
            if local_run_id not in run_ids:
                print(f"logs/{project_name}/{local_run_id}")
                
if __name__ == "__main__":
    what_logs_to_delete()