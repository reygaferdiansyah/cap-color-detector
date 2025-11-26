from typing import Any, Dict

import torch
import yaml
from ultralytics import YOLO

from .utils import download_dataset
from .wandb_utils import init_wandb


def train_model(config_path: str) -> None:
    """Trains a YOLO model using parameters from a YAML configuration file.

    This function handles the full training pipeline: loading configuration,
    downloading the dataset, initializing Weights & Biases (W&B) logging,
    executing the training loop, and performing validation.

    Args:
        config_path (str): The file path to the YAML configuration document
            containing dataset paths and training hyperparameters.

    Returns:
        None: This function does not return a value but saves the trained model
        artifacts to the specified project directory.
    """
    with open(config_path) as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")

    download_dataset()
    wandb_run = init_wandb(cfg)

    # Initialize model based on the size specified in config (e.g., yolov8n.pt)
    model = YOLO(f"{cfg['training']['model_size']}.pt")

    # Start training using parameters unpacked from the config dictionary
    model.train(
        data=cfg["dataset"]["path"],
        epochs=cfg["training"]["epochs"],
        batch=cfg["training"]["batch"],
        imgsz=cfg["training"]["imgsz"],
        workers=cfg["training"]["workers"],
        device=0,
        amp=cfg["training"]["amp"],
        optimizer=cfg["training"]["optimizer"],
        lr0=cfg["training"]["lr0"],
        lrf=cfg["training"]["lrf"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_epochs=cfg["training"]["warmup_epochs"],
        hsv_h=cfg["training"]["hsv_h"],
        hsv_s=cfg["training"]["hsv_s"],
        hsv_v=cfg["training"]["hsv_v"],
        degrees=cfg["training"]["degrees"],
        translate=cfg["training"]["translate"],
        scale=cfg["training"]["scale"],
        shear=cfg["training"]["shear"],
        flipud=cfg["training"]["flipud"],
        fliplr=cfg["training"]["fliplr"],
        mosaic=cfg["training"]["mosaic"],
        mixup=cfg["training"]["mixup"],
        copy_paste=cfg["training"]["copy_paste"],
        close_mosaic=cfg["training"]["close_mosaic"],
        box=cfg["training"]["box"],
        cls=cfg["training"]["cls"],
        dfl=cfg["training"]["dfl"],
        label_smoothing=cfg["training"]["label_smoothing"],
        patience=cfg["training"]["patience"],
        pretrained=True,
        project=cfg["training"]["project"],
        name=cfg["training"]["name"],
        exist_ok=cfg["training"]["exist_ok"],
        plots=cfg["training"]["plots"],
        save=cfg["training"]["save"],
        save_period=cfg["training"]["save_period"],
    )

    # Validate the model using the validation set
    model.val(
        data=cfg["dataset"]["path"],
        conf=cfg["training"]["val_conf"],
        iou=cfg["training"]["val_iou"],
        plots=True,
        save_json=True,
    )

    wandb_run.finish()
