from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
from src.utils import instantiate_callbacks, get_current_time, get_ckpt_path
from src.networks.basetorchmodule import model_config_from_id
import pytorch_lightning as pl
import os, hydra, torch
from pytorch_lightning import LightningDataModule, LightningModule, Callback
import wandb
import yaml

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["USE_FLASH_ATTENTION"] = "1"

def update_dict(d : dict | list):
    if isinstance(d, dict):
        if d.get('_target_', None) == "src.networks.PretrainedModel":
            model_keyword = d['model_keyword']
            experiment_id = d['experiment_id']
            model_config = model_config_from_id(experiment_id, model_keyword)
            d.clear()
            d.update(model_config)
        for k, v in d.items():
            update_dict(v)
    elif isinstance(d, list):
        for v in d:
            update_dict(v)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(cfg : DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed)

    project_name, task_name, id = cfg.project_name, cfg.task_name, cfg.continue_from_id
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    update_dict(config)

    print("Config:", yaml.dump(config, default_flow_style=False, sort_keys=False))
    
    logger = WandbLogger(
        **cfg.logger,
        project = project_name, 
        name = task_name, 
        id = get_current_time() if not id else id, 
        config = config
        )
    
    if id:
        print(f"Continuing from id: {id}")
        ckpt_path = get_ckpt_path(id, last=True)
    else:
        ckpt_path = None
    
    print("Instantiating callbacks..")
    callbacks : list[Callback] = instantiate_callbacks(cfg.get("callbacks", None))

    print("Setting up trainer..")
    trainer = Trainer(
        **cfg.trainer, 
        logger = logger, 
        callbacks = callbacks
        )
    
    print("Instantiating model and datamodule..")
    datamodule : LightningDataModule = hydra.utils.instantiate(cfg.data)
    model : LightningModule = hydra.utils.instantiate(cfg.model)

    if cfg.compile:
        print("Compiling model..")
        torch.compile(model)
        
    print("Beginning training..")
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)
    wandb.finish()

if __name__ == "__main__":
    my_app()