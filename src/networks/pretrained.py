from torch.nn import Module
from src.utils import model_from_id


class PretrainedModel:
    """
    Factory class that returns a pretrained nn.Module loaded from Weights & Biases (WandB)
    given a project name, run ID, and model keyword.

    Args:
        id (str): The WandB run ID.
        model_keyword (str): The keyword to identify the model in the module.
        ckpt_filename (str, optional): The checkpoint filename to load. Defaults to "last".
    """

    def __new__(
        cls,
        id: str,
        model_keyword: str,
        ckpt_filename: str = "last",
    ) -> Module:
        return model_from_id(
            id,
            model_keyword,
            ckpt_filename,
        )
