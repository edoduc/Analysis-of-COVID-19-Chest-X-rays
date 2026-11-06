from io import BytesIO
from unittest.mock import Mock

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import api.app as api_module


client = TestClient(api_module.app)


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("L", (16, 16), color=128).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_reports_available_model():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_family"] == "xgboost"


def test_predict_rejects_non_image_upload():
    response = client.post(
        "/predict",
        files={"image": ("image.png", b"not an image", "image/png")},
    )

    assert response.status_code == 422


def test_predict_routes_low_confidence_to_review(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "predict",
        lambda _: {
            "class_name": "COVID",
            "class_index": 0,
            "probabilities": np.array([0.4, 0.3, 0.2, 0.1]),
            "confidence": 0.4,
        },
    )

    response = client.post(
        "/predict",
        files={
            "image": ("image.png", _png_bytes(), "image/png"),
            "mask": ("mask.png", _png_bytes(), "image/png"),
        },
    )

    assert response.status_code == 200
    assert response.json()["decision"] == "review_required"
    assert response.json()["decision_reason"] == "confidence_below_threshold"
    assert response.json()["model_family"] == "xgboost"
    assert response.json()["class_name"] is None
    assert response.json()["candidate_class"] == "COVID"


def test_predict_requires_mask_for_automatic_prediction(monkeypatch):
    predict_mock = Mock(side_effect=AssertionError("Inference should not run without a mask."))
    monkeypatch.setattr(api_module, "predict", predict_mock)

    response = client.post(
        "/predict",
        files={"image": ("image.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["decision"] == "review_required"
    assert response.json()["decision_reason"] == "lung_mask_missing"
    assert response.json()["model_family"] == "xgboost"
    assert response.json()["class_name"] is None
    assert response.json()["confidence"] is None
    predict_mock.assert_not_called()