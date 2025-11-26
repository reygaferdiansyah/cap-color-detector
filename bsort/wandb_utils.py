import os
from typing import Any, Dict

import wandb


def init_wandb(cfg: Dict[str, Any]) -> Any:
    """Initializes a Weights & Biases run for experiment tracking.

    This function performs the login authentication and starts a new W&B run
    logged under the specific project, naming the run based on the model size
    and experiment name defined in the configuration.

    Args:
        cfg (Dict[str, Any]): A dictionary containing training configurations.
            It must contain keys for 'training' which include 'model_size'
            and 'name'.

    Returns:
        Any: The active W&B Run object used for logging metrics during training.
    """
    wandb.login()
    return wandb.init(
        project="cap-color-detector",
        name=f"{cfg['training']['model_size']}_{cfg['training']['name']}",
        config=cfg,
    )
