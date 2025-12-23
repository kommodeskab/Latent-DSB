from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
from src.utils import instantiate_callbacks, get_current_time, get_ckpt_path, model_config_from_id
import pytorch_lightning as pl
import os, hydra, torch
from pytorch_lightning import LightningDataModule, LightningModule, Callback
import wandb
import yaml
import logging

os.environ["HYDRA_FULL_ERROR"] = "1"

@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(cfg : DictConfig) -> None:
    logger = logging.getLogger(__name__)

    pl.seed_everything(cfg.seed)

    project_name, task_name, id = cfg.project_name, cfg.task_name, cfg.continue_from_id
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    print("Config:", yaml.dump(config, default_flow_style=False, sort_keys=False))
    
    wandblogger = WandbLogger(
        **cfg.logger,
        project = project_name, 
        name = task_name, 
        id = get_current_time() if not id else str(id), 
        config = config
        )
    
    if id:
        logger.info(f"Continuing from id: {id}")
        ckpt_path = get_ckpt_path(id, last=True)
    else:
        ckpt_path = None
    
    logger.info("Instantiating callbacks..")
    callbacks : list[Callback] = instantiate_callbacks(cfg.get("callbacks", None))

    logger.info("Setting up trainer..")
    trainer = Trainer(
        **cfg.trainer, 
        logger = wandblogger, 
        callbacks = callbacks
        )
    
    logger.info("Instantiating model and datamodule..")
    datamodule : LightningDataModule = hydra.utils.instantiate(cfg.data)
    model : LightningModule = hydra.utils.instantiate(cfg.model)

    if cfg.compile:
        logger.info("Compiling model..")
        model = torch.compile(model, fullgraph=False)
        
    logger.info("Beginning training..")
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)
    wandb.finish()

if __name__ == "__main__":
    my_app()