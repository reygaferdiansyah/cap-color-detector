# test/test_inference.py
from unittest.mock import MagicMock, patch

import pytest

from bsort.inference import (AVAILABLE_MODELS, run_inference,
                             run_inference_all_models)


def test_run_inference_single_model_mocked() -> None:
    """Tests single model inference using a mocked YOLO instance.

    This test verifies that the inference function correctly initializes
    the YOLO class with the expected weights and returns the processed results.
    """
    dummy_source = "dummy.jpg"

    with patch("bsort.inference.YOLO") as mock_yolo_class:
        # Mock YOLO instance
        mock_model_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = [1, 2, 3]  # Simulate 3 detections
        mock_model_instance.return_value = [mock_result]  # YOLO()() returns a list
        mock_yolo_class.return_value = mock_model_instance

        results = run_inference(source=dummy_source)
        assert len(results) == 1
        mock_yolo_class.assert_called_once_with(AVAILABLE_MODELS["yolov12m"])


def test_run_inference_invalid_model() -> None:
    """Tests that a ValueError is raised for an invalid model name.

    This ensures that requesting a model not listed in AVAILABLE_MODELS
    triggers the appropriate exception.
    """
    # Note: match string must correspond to the English error message in inference.py
    with pytest.raises(ValueError, match="not found"):
        run_inference(source="fake.jpg", model_name="yolov99x")


def test_run_inference_all_models_mocked() -> None:
    """Tests that the batch inference function iterates through all available models.

    This verifies that run_inference_all_models calls the single inference
    function exactly once for every model in the registry.
    """
    dummy_source = "dummy.jpg"

    with patch("bsort.inference.run_inference") as mock_run:
        mock_run.return_value = ["dummy_result"]
        all_results = run_inference_all_models(source=dummy_source)

        assert len(all_results) == len(AVAILABLE_MODELS)
        for key in AVAILABLE_MODELS.keys():
            assert key in all_results
