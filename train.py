from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
from src.utils import instantiate_callbacks, get_current_time, get_ckpt_path
import pytorch_lightning as pl
import os, hydra, torch
from pytorch_lightning import LightningDataModule, LightningModule, Callback
import wandb

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["USE_FLASH_ATTENTION"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(cfg : DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed)

    project_name, task_name = cfg.project_name, cfg.task_name
    
    if id := cfg.continue_from_id:
        print(f"Continuing from id: {id}")
        ckpt_path = get_ckpt_path(project_name, id)
    else:
        ckpt_path = None
    
    print("Setting up logger..")
    logger = WandbLogger(
        **cfg.logger,
        project = project_name, 
        name = task_name, 
        id = get_current_time() if not id else id, 
        config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )
    
    print("Instantiating model and datamodule..")
    datamodule : LightningDataModule = hydra.utils.instantiate(cfg.data)
    model : LightningModule = hydra.utils.instantiate(cfg.model)

    if cfg.compile:
        print("Compiling model..")
        torch.compile(model)
    
    print("Instantiating callbacks..")
    callbacks : list[Callback] = instantiate_callbacks(cfg.get("callbacks", None))

    print("Setting up trainer..")
    trainer = Trainer(
        **cfg.trainer, 
        logger = logger, 
        callbacks = callbacks
        )
        
    print("Beginning training..")
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)
    wandb.finish()

if __name__ == "__main__":
    my_app()