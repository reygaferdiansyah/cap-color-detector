from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict

from ultralytics import YOLO
from ultralytics.engine.results import Results


# Dictionary containing paths to all trained models
AVAILABLE_MODELS = {
    "yolov12n": "runs/yolov12/nano/final_v12_nano/weights/best.pt",
    "yolov12s": "runs/yolov12/small/final_v12_small/weights/best.pt",
    "yolov12m": "runs/yolov12/medium/final_v12_medium/weights/best.pt",
    "yolov11n": "runs/yolov11/nano/final_v11_nano/weights/best.pt",
    "yolov11s": "runs/yolov11/small/final_v11_small/weights/best.pt",
    "yolov11m": "runs/yolov11/medium/final_v11_medium/weights/best.pt",
}


def run_inference(
    source: str,
    model_name: str = "yolov12m",
    output_dir: str = "runs/inference",
    conf: float = 0.35,
    iou: float = 0.6,
    imgsz: int = 640,
    save: bool = True,
    show: bool = False,
    save_crop: bool = False,
    device: Optional[str] = None,
) -> List[Results]:
    """Runs object detection inference using a specific YOLO model.

    This function loads the specified model weights and performs inference on
    the given source (image, video, or webcam).

    Args:
        source (str): Path to input image/video or '0' for webcam.
        model_name (str): Key name of the model to use from AVAILABLE_MODELS.
            Defaults to "yolov12m".
        output_dir (str): Directory where results will be saved.
            Defaults to "runs/inference".
        conf (float): Confidence threshold for detections. Defaults to 0.35.
        iou (float): IOU threshold for NMS. Defaults to 0.6.
        imgsz (int): Input image size for the model. Defaults to 640.
        save (bool): If True, saves the inference results. Defaults to True.
        show (bool): If True, displays the results in a window. Defaults to False.
        save_crop (bool): If True, saves cropped images of detected objects.
            Defaults to False.
        device (Optional[str]): Device to run inference on (e.g., 'cpu', '0').
            Defaults to None (auto-detect).

    Returns:
        List[Results]: A list of Ultralytics Result objects containing boxes,
        masks, and probabilities.

    Raises:
        ValueError: If the provided model_name is not found in AVAILABLE_MODELS.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not found!")

    model_path = AVAILABLE_MODELS[model_name]
    model = YOLO(model_path)

    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    results: List[Results] = model(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        save=save,
        show=show,
        save_crop=save_crop,
        project=str(output_path.parent),
        name=output_path.name,
        exist_ok=True,
        device=device,
        verbose=False,
    )

    total = sum(len(r.boxes) for r in results if r.boxes is not None)
    print(f"\n[{model_name.upper()}] -> {total} caps detected")
    print(f"Results saved to: {output_path.resolve()}\n")
    return results


def run_inference_all_models(
    source: str,
    output_dir: str = "runs/inference_all",
    conf: float = 0.35,
    iou: float = 0.6,
    imgsz: int = 640,
    save: bool = True,
    show: bool = False,
    save_crop: bool = False,
    device: Optional[str] = None,
) -> Dict[str, List[Results]]:
    """Runs inference sequentially using all available models.

    This function iterates through the AVAILABLE_MODELS dictionary and runs
    inference for each model on the same source data.

    Args:
        source (str): Path to input image/video.
        output_dir (str): Base directory for saving combined results.
            Defaults to "runs/inference_all".
        conf (float): Confidence threshold. Defaults to 0.35.
        iou (float): IOU threshold. Defaults to 0.6.
        imgsz (int): Image size. Defaults to 640.
        save (bool): If True, saves results. Defaults to True.
        show (bool): If True, shows visualization. Defaults to False.
        save_crop (bool): If True, saves crops. Defaults to False.
        device (Optional[str]): Computation device. Defaults to None.

    Returns:
        Dict[str, List[Results]]: A dictionary where keys are model names and
        values are the list of Result objects from that model.
    """
    print("TESTING 6 MODELS\n")
    print(f"Source: {source}\n")

    all_results: Dict[str, List[Results]] = {}
    base_output = Path(output_dir)
    base_output.mkdir(parents=True, exist_ok=True)

    for model_name, model_path in AVAILABLE_MODELS.items():
        print(f"Running -> {model_name.upper()}")

        results = run_inference(
            source=source,
            model_name=model_name,
            output_dir=output_dir,
            conf=conf,
            show=show,
            save_crop=save_crop,
            device=device,
        )
        all_results[model_name] = results

    print("ALL MODELS COMPLETED!")
    print(f"Results available at -> {base_output.resolve()}\n")
    return all_results