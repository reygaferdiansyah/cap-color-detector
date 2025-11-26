import click

from .inference import run_inference, run_inference_all_models
from .trainer import train_model


@click.group()
def cli() -> None:
    """Cap Color Detector using YOLOv12 and YOLOv11 Pipeline.

    This acts as the main entry point for the command line interface.
    """
    pass


@cli.command()
@click.option("--config", required=True, help="Path to config YAML")
def train(config: str) -> None:
    """Trains a model using the provided configuration file.

    Args:
        config (str): Path to the YAML configuration file.

    Returns:
        None
    """
    train_model(config)


@cli.command()
@click.argument("source")
@click.option(
    "--model",
    "-m",
    default="yolov12m",
    type=click.Choice(
        ["yolov12n", "yolov12s", "yolov12m", "yolov11n", "yolov11s", "yolov11m"]
    ),
    help="Select specific model",
)
@click.option(
    "--all", "run_all", is_flag=True, help="Run ALL 6 models simultaneously (Flex Mode)"
)
@click.option("--conf", default=0.35, type=float, help="Confidence threshold")
@click.option("--show", is_flag=True, help="Display results in a window")
@click.option("--crop", is_flag=True, help="Save cropped objects")
def infer(
    source: str, model: str, run_all: bool, conf: float, show: bool, crop: bool
) -> None:
    """Runs inference on the cap detector.

    Examples:
        bsort infer test/cap.jpg            (Uses yolov12m by default)
        bsort infer 0 --show                (Webcam real-time)
        bsort infer video.mp4 -m yolov12n   (Fast inference)
        bsort infer image.jpg --all         (FLEX MODE: All 6 models)

    Args:
        source (str): The input source (image path, video path, or '0' for webcam).
        model (str): The specific model architecture to use.
        run_all (bool): If True, triggers execution of all available models.
        conf (float): The confidence threshold for detections.
        show (bool): If True, displays the inference results in a GUI window.
        crop (bool): If True, saves crops of detected objects to disk.

    Returns:
        None
    """
    if run_all:
        print("FLEX MODE: RUNNING ALL 6 MODELS SIMULTANEOUSLY")
        run_inference_all_models(
            source=source,
            conf=conf,
            show=show,
            save_crop=crop,
        )
    else:
        print(f"Running single model: {model.upper()}")
        run_inference(
            source=source,
            model_name=model,
            conf=conf,
            show=show,
            save_crop=crop,
        )


if __name__ == "__main__":
    cli()
