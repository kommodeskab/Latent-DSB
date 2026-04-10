import os
import hydra
from omegaconf import DictConfig
from datetime import datetime
import wandb
import random
import numpy as np
import torch
from hydra.utils import instantiate
import torch.nn as nn
from pytorch_lightning.callbacks import Callback
import tempfile
import shutil
import logging
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from src import Batch
from contextlib import contextmanager, nullcontext, AbstractContextManager
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@contextmanager
def temporary_seed(seed: int):
    """
    Context manager for setting a temporary random seed. Sets `random`, `numpy`, `torch` and `torch.cuda` (if available) seeds.
    Code executed inside the context manager will have the specified random seed.
    What is it good for? For reproducing results, e.g. during validation or testing.
    Example usage:
    >>> torch.randn(3) # random tensor
    >>> with temporary_seed(42):
    >>>     torch.randn(3) # tensor with seed 42

    Args:
        seed (int): The temporary random seed to set.
    """
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()

    # CUDA RNG state is unsafe to access from forked DataLoader workers.
    in_worker = torch.utils.data.get_worker_info() is not None
    use_cuda_state = torch.cuda.is_available() and torch.cuda.is_initialized() and not in_worker
    cuda_state = torch.cuda.get_rng_state() if use_cuda_state else None

    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if use_cuda_state:
            torch.cuda.manual_seed_all(seed)
        yield

    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)


def get_context(seed: int, deterministic: bool) -> AbstractContextManager:
    """
    Returns a context manager for setting a temporary random seed if `deterministic` is `True`, otherwise returns a null context manager.

    Args:
        seed (int): The temporary random seed to set if `deterministic` is `True`.
        deterministic (bool): Whether to set the temporary random seed.

    Returns:
        contextlib.AbstractContextManager: A context manager for setting a temporary random seed if `deterministic` is `True`, otherwise a null context manager.
    """
    return temporary_seed(seed) if deterministic else nullcontext()


def get_current_time() -> str:
    """Returns the current time as a string in the format 'ddmmyyHHMMSS'.

    Returns:
        str: The current time as a string.
    """
    now = datetime.now()
    return now.strftime("%d%m%y%H%M%S")


def instantiate_callbacks(callback_cfg: DictConfig | None) -> list[Callback]:
    """
    Function for instantiating callbacks given a `DictConfig`.
    If `callback_cfg` is `None`, an empty list is returned.
    This function is useful for hydra-based configuration of PyTorch Lightning callbacks.

    Args:
        callback_cfg (DictConfig | None): A `DictConfig` containing the callback configurations. If `None`, no callbacks are instantiated.

    Returns:
        list[Callback]: A list of instantiated callbacks.
    """
    callbacks: list[Callback] = []

    if callback_cfg is None:
        return callbacks

    for _, callback_params in callback_cfg.items():
        callback = hydra.utils.instantiate(callback_params)
        callbacks.append(callback)

    return callbacks


def project_from_id(id: str) -> str:
    """
    Returns the project name for a specific WandB run ID.

    Args:
        id (str): The run ID
    Returns:
        str: The project name for the specified run ID.
    """
    project_names = wandb.Api().projects()
    project_names = [project.name for project in project_names]
    for project_name in project_names:
        runs = wandb.Api().runs(project_name)
        run_ids = [run.id for run in runs]
        if id in run_ids:
            return project_name

    raise ValueError(f"Could not find project for experiment id '{id}'.")


def wandb_entity() -> str:
    """
    Returns the current WandB entity (user or team).

    Returns:
        str: The current WandB entity.
    """
    return os.environ.get("WANDB_ENTITY") or wandb.api.viewer()["entity"]


def run_from_id(id: str) -> wandb.Run:
    """
    Returns the WandB run for a specific run ID.
    Args:
        id (str): The run ID
    Returns:
        wandb.Run: The WandB run for the specified run ID.
    """
    name = wandb_entity()
    project = project_from_id(id)
    path = f"{name}/{project}/{id}"
    try:
        run = wandb.Api().run(path)
        logger.info(f"Found experiment {path}.")
        return run
    except wandb.errors.CommError:
        pass

    raise ValueError(f"Could not find experiment {path}.")


def get_ckpt_path(
    id: str,
    filename: str = "last",
):
    """
    Returns the path to a specific WandB checkpoint.
    If the checkpoint does not exist locally, it attempts to download it from WandB.
    Example usage:
    >>> ckpt_path = get_ckpt_path("12345678", "best")
    >>> ckpt = torch.load(ckpt_path)
    >>> module = DummyModule.load_from_checkpoint(ckpt_path)

    Args:
        id (str): The WandB run ID.
        filename (str, optional): The checkpoint filename. If the checkpoint is remote, then specify the name of the artifact containing the checkpoint. Defaults to "last".

    Returns:
        str: Path to the checkpoint file.
    """

    project = project_from_id(id)
    ckpt_path = f"logs/{project}/{id}/checkpoints/{filename}.ckpt"

    if not os.path.exists(ckpt_path):
        try:
            ckpt_path = download_checkpoint(
                id=id,
                filename=filename,
            )
            logger.info(f"Checkpoint downloaded from WandB to {ckpt_path}.")
        except Exception as e:
            raise ValueError(
                f"Could not find or download checkpoint with filename '{filename}' for experiment id '{id}' in project '{project}'."
            ) from e

    return ckpt_path


def what_logs_to_delete():
    """
    Prints out the WandB logs that can be safely deleted.
    By "safely deleted", we mean logs that exist locally but not on WandB.
    Here, we assume that if a run ID does not exist on WandB, then it has become obsolete and can be deleted.
    Generally, it is safe to delete most local logs since WandB keeps track of all experiments, including model checkpoints (if they are logged using a `ModelCheckpoint` callback).
    """
    api = wandb.Api()
    project_names = api.projects()
    project_names = [project.name for project in project_names]
    print("It is safe to delete the following folders:")
    for project_name in project_names:
        if not os.path.exists(f"logs/{project_name}"):
            continue

        runs = api.runs(project_name)
        run_ids = [run.id for run in runs]
        local_run_ids = os.listdir(f"logs/{project_name}")
        local_run_ids.sort(reverse=True)

        for local_run_id in local_run_ids:
            if local_run_id not in run_ids:
                # delete the folder
                print(f"logs/{project_name}/{local_run_id}")

    print("Done")


def config_from_id(id: str) -> dict:
    """
    Returns the config for a specific run.

    Args:
        id (str): The run ID

    Raises:
        ValueError: If the experiment with the given project and ID could not be found.

    Returns:
        dict: The configuration dictionary for the specified run.
    """
    try:
        run = run_from_id(id)
        logger.info(f"Found experiment {id}.")
        return run.config
    except wandb.errors.CommError:
        pass

    raise ValueError(f"Could not find experiment {id} on WandB.")


def model_config_from_id(id: str, model_keyword: str) -> dict:
    """
    Returns the model config for a specific run.

    Args:
        id (str): The run ID.
        model_keyword (str): The keyword in the config corresponding to the model.

    Returns:
        dict: The model configuration dictionary for the specified run.
    """
    config = config_from_id(id)
    return config["model"][model_keyword]


def module_from_id(
    id: str,
    ckpt_filename: str = "last",
) -> pl.LightningModule:
    """
    Loads a PyTorch Lightning module from a specific WandB run ID and checkpoint.

    Args:
        id (str): The run ID
        ckpt_filename (str, optional): The checkpoint filename. Defaults to "last".

    Returns:
        pl.LightningModule: The loaded PyTorch Lightning module.
    """

    config = config_from_id(id)
    model_config = config["model"]
    module: pl.LightningModule = instantiate(model_config)

    ckpt_path = get_ckpt_path(id, ckpt_filename)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    module.load_state_dict(ckpt["state_dict"])
    logger.info(f"Loaded module from experiment id {id} at checkpoint {ckpt_path}.")

    return module


def model_from_id(
    id: str,
    model_keyword: str,
    ckpt_filename: str = "last",
) -> nn.Module:
    """
    Loads a model from a specific WandB run ID and checkpoint.
    This is NOT a PyTorch Lightning module, but the underlying model inside the module.

    Args:
        id (str): The run ID.
        model_keyword (str): The keyword in the config corresponding to the model.
        ckpt_filename (str, optional): The checkpoint filename. Defaults to "last".

    Returns:
        nn.Module: The loaded model.
    """
    module = module_from_id(
        id=id,
        ckpt_filename=ckpt_filename,
    )

    model = getattr(module, model_keyword)

    return model


def get_root() -> str:
    """
    Returns:
        str: The root directory of this project based on git.
    """
    return os.popen("git rev-parse --show-toplevel").read().strip()


def get_artifact(
    id: str,
    filename: str,
):
    """Downloads a given artifact from WandB

    Args:
        id (str): The run ID
        filename (str): The name of the artifact

    Returns:
        wandb.Artifact: The requested artifact
    """
    project = project_from_id(id)
    return wandb.Api().artifact(f"{project}/{filename}")


def download_checkpoint(
    id: str,
    filename: str,
) -> str:
    """
    Downloads a model checkpoint from WandB and saves it locally.
    Return the path to the downloaded checkpoint.

    Args:
        id (str): The run ID
        filename (str): The name of the artifact

    Returns:
        str: The path to the downloaded checkpoint
    """

    # specify where to save the checkpoint
    root = get_root()
    project = project_from_id(id)
    savedir = f"{root}/logs/{project}/{id}/checkpoints"
    os.makedirs(savedir, exist_ok=True)

    # Define the final destination file path
    final_path = os.path.join(savedir, f"{filename}.ckpt")

    # Use a temporary directory to download first
    # to not overwrite existing files, since it will download as 'model.ckpt' as default (wandb logic)
    with tempfile.TemporaryDirectory() as temp_dir:
        # download the artifact to a temporary directory
        artifact = get_artifact(id, filename)
        artifact.download(root=temp_dir)

        # The file inside the artifact is named 'model.ckpt' (see modelcheckpoint.py)
        source_path = os.path.join(temp_dir, "model.ckpt")

        assert not os.path.exists(final_path), f"Checkpoint already exists at {final_path}."

        # Move and rename to the final destination
        shutil.move(source_path, final_path)
        logger.info(f"Downloaded checkpoint to {final_path}.")

    return final_path


def get_batch_from_dataset(dataset: Dataset, batch_size: int, shuffle: bool = False) -> Batch:
    """
    Returns a single batch of a given size from the dataset.
    This is useful for quickly getting a batch, e.g. for testing purposes.

    Args:
        dataset (Dataset): The dataset to get the batch from.
        batch_size (int): The size of the batch.
        shuffle (bool, optional): Whether to shuffle the dataset before getting the batch. Defaults to False.

    Returns:
        Batch: A batch of data from the dataset.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return next(iter(dataloader))


if __name__ == "__main__":
    projectname = project_from_id("220126133817")
    print(projectname)
