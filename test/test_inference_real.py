# test/test_inference_real.py

from pathlib import Path
from typing import List

import pytest

from bsort.inference import (AVAILABLE_MODELS, run_inference,
                             run_inference_all_models)

# Test images folder relative to this test file location
TEST_IMAGES_DIR = (
    Path(__file__).parent.parent / "datasets" / "Object_Detection_Caps-1" / "test"
)

# Allowed image extensions
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def get_image_files(folder: Path) -> List[Path]:
    """Retrieves all image files from a specific directory.

    Args:
        folder (Path): The directory path to search for images.

    Returns:
        List[Path]: A list of paths pointing to valid image files.
    """
    return [f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS]


@pytest.mark.skipif(
    not TEST_IMAGES_DIR.exists(), reason="Test images folder does not exist"
)
def test_run_inference_real() -> None:
    """Tests inference on a single real image using the first available file.

    This test checks if the inference pipeline runs successfully on real data
    without errors and returns a valid list of results.
    """
    image_files = get_image_files(TEST_IMAGES_DIR)
    if not image_files:
        pytest.skip("No image files found in test directory")

    img_path = image_files[0]
    results = run_inference(source=str(img_path), show=False, save=False)

    # Ensure output is a list
    assert isinstance(results, list)
    assert all(hasattr(r, "boxes") for r in results)


@pytest.mark.skipif(
    not TEST_IMAGES_DIR.exists(), reason="Test images folder does not exist"
)
def test_run_inference_all_models_real() -> None:
    """Tests inference for all models using a single real image.

    This ensures that the multi-model inference wrapper correctly iterates
    over all defined models and returns valid results for each.
    """
    image_files = get_image_files(TEST_IMAGES_DIR)
    if not image_files:
        pytest.skip("No image files found in test directory")

    img_path = image_files[0]
    all_results = run_inference_all_models(source=str(img_path), show=False, save=False)

    # Ensure all models were executed
    assert set(all_results.keys()) == set(AVAILABLE_MODELS.keys())

    for model_name, results in all_results.items():
        assert isinstance(results, list)
        assert all(hasattr(r, "boxes") for r in results)
