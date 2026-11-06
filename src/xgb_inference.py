import os
from functools import lru_cache
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from src.constants import CLASS_NAMES, PREPROCESS_SIZE
from src.feature_extraction_ml import (
    extract_hog_features,
    extract_lbp_features,
    extract_stat_features,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "xgboost" / "xgb_best.pkl"
STAT_FEATURE_NAMES = (
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
)
FEATURE_NAMES = (
    *STAT_FEATURE_NAMES,
    *(f"lbp_{index}" for index in range(26)),
    *(f"hog_{index}" for index in range(576)),
)


def get_xgb_model_path() -> Path:
    configured_path = os.environ.get("XGBOOST_MODEL_PATH")
    if not configured_path:
        return DEFAULT_MODEL_PATH

    model_path = Path(configured_path).expanduser()
    return model_path if model_path.is_absolute() else ROOT_DIR / model_path


@lru_cache(maxsize=2)
def _load_xgb_model(model_path: str):
    model = joblib.load(model_path)
    if getattr(model, "n_features_in_", None) != len(FEATURE_NAMES):
        raise RuntimeError(
            f"Le modèle XGBoost doit attendre {len(FEATURE_NAMES)} features."
        )

    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None and tuple(model_feature_names) != FEATURE_NAMES:
        raise RuntimeError("L'ordre des features ne correspond pas au modèle XGBoost.")

    model_classes = np.asarray(getattr(model, "classes_", []))
    if not np.array_equal(model_classes, np.arange(len(CLASS_NAMES))):
        raise RuntimeError("Le mapping de classes du modèle XGBoost est inattendu.")

    if "device" in model.get_params(deep=False):
        model.set_params(device="cpu")
    return model


def load_xgb_model(model_path: str | Path | None = None):
    resolved_path = Path(model_path) if model_path is not None else get_xgb_model_path()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Modèle XGBoost introuvable : {resolved_path}")
    return _load_xgb_model(str(resolved_path.resolve()))


def extract_xgb_features(image_array: np.ndarray) -> pd.DataFrame:
    image = np.asarray(image_array)
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[:, :, 0]
        else:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if image.ndim != 2:
        raise ValueError("L'image doit être en niveaux de gris ou RGB.")

    image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape != (PREPROCESS_SIZE, PREPROCESS_SIZE):
        image = cv2.resize(
            image,
            (PREPROCESS_SIZE, PREPROCESS_SIZE),
            interpolation=cv2.INTER_AREA,
        )

    feature_values = np.concatenate(
        [
            extract_stat_features(image),
            extract_lbp_features(image, radius=3, method="uniform"),
            extract_hog_features(image),
        ]
    )
    if feature_values.size != len(FEATURE_NAMES):
        raise RuntimeError(
            f"{feature_values.size} features extraites, {len(FEATURE_NAMES)} attendues."
        )

    return pd.DataFrame([feature_values], columns=FEATURE_NAMES)


def predict(image_array: np.ndarray) -> dict[str, object]:
    features = extract_xgb_features(image_array)
    model = load_xgb_model()
    model_features = (
        features
        if getattr(model, "feature_names_in_", None) is not None
        else features.to_numpy()
    )
    probabilities = np.asarray(model.predict_proba(model_features)[0], dtype=np.float64)
    probability_index = int(probabilities.argmax())
    class_index = int(model.classes_[probability_index])

    return {
        "class_name": CLASS_NAMES[class_index],
        "class_index": class_index,
        "probabilities": probabilities,
        "confidence": float(probabilities[probability_index]),
    }