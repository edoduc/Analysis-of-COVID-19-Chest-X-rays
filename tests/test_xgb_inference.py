from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

import api.app as api_module
from src.xgb_inference import FEATURE_NAMES, extract_xgb_features


def test_xgb_features_match_data_science_schema():
    image = np.zeros((299, 299), dtype=np.uint8)

    features = extract_xgb_features(image)

    assert features.shape == (1, 617)
    assert tuple(features.columns) == FEATURE_NAMES
    assert features.columns[:15].tolist() == [
        "mean",
        "std",
        "min",
        "max",
        "median",
        "p5",
        "p25",
        "p75",
        "p95",
        "skewness",
        "kurtosis",
        "entropy",
        "laplacian_variance",
        "gradient_magnitude_mean",
        "gradient_magnitude_std",
    ]
    assert features.columns[15] == "lbp_0"
    assert features.columns[40] == "lbp_25"
    assert features.columns[41] == "hog_0"
    assert features.columns[-1] == "hog_575"


def test_saved_xgb_artifact_predicts_on_cpu():
    from src.xgb_inference import load_xgb_model

    model = load_xgb_model()
    features = extract_xgb_features(np.zeros((299, 299), dtype=np.uint8))
    model_features = features if hasattr(model, "feature_names_in_") else features.to_numpy()
    probabilities = model.predict_proba(model_features)[0]

    assert len(probabilities) == 4
    assert np.isclose(probabilities.sum(), 1.0)


def test_api_predicts_with_saved_xgb_artifact():
    project_root = Path(__file__).resolve().parents[1]
    image_path = project_root / "data" / "demo_samples" / "COVID" / "COVID-1439.png"
    mask_path = project_root / "data" / "demo_samples" / "COVID" / "COVID-1439_mask.png"
    client = TestClient(api_module.app)

    response = client.post(
        "/predict",
        files={
            "image": (image_path.name, image_path.read_bytes(), "image/png"),
            "mask": (mask_path.name, mask_path.read_bytes(), "image/png"),
        },
    )

    assert response.status_code == 200
    assert response.json()["model_family"] == "xgboost"
    assert response.json()["mask_used"] is True
    assert len(response.json()["probabilities"]) == 4