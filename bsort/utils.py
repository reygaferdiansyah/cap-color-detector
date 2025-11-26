import os

from roboflow import Roboflow


def download_dataset() -> str:
    """Downloads the object detection dataset from Roboflow.

    This function checks if the dataset directory already exists locally.
    If not, it initializes the Roboflow client and downloads specific
    version of the project in YOLOv8 format.

    Returns:
        str: The local file path to the directory containing the dataset.
    """
    dataset_dir = "datasets/Object_detection_Caps-1"

    if os.path.exists(dataset_dir):
        print(f"Dataset already exists: {dataset_dir}")
        return dataset_dir

    print("Downloading dataset from Roboflow...")

    rf = Roboflow(api_key="OeMGg7rPtyz3TMN3JWtR")
    project = rf.workspace("stikom-cki-mkng8").project("object_detection_caps-y8odj")
    version = project.version(1)

    # Download the dataset to the 'datasets' folder
    dataset = version.download("yolov8", location="datasets")

    print(f"Dataset successfully downloaded: {dataset.location}")
    return dataset.location
