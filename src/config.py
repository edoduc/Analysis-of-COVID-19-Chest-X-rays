"""
Centralized Configuration and Path Management.
Provides single-source-of-truth constants and configuration loading.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Project Root Directory (always relative to this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Standard Directory Structure
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "COVID-19_Radiography_Dataset"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_DIR = DATA_DIR / "train"
TEST_DATA_DIR = DATA_DIR / "test"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Standard Default Dataset Classes
DEFAULT_CLASSES = [
    "COVID",
    "Lung_Opacity",
    "Normal",
    "Viral Pneumonia",
]

DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config.json"


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """
    Safely load project configuration from JSON with fallback defaults.

    :param config_path: Optional path to configuration file. Defaults to PROJECT_ROOT / 'config.json'.
    :return: Configuration dictionary.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_FILE

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        return {
            "image_size": 299,
            "model_image_size": 224,
            "paths": {
                "raw_data_dir": "data/raw/COVID-19_Radiography_Dataset",
            },
            "feature_extraction": {
                "lbp": {"radius": 3, "method": "uniform"},
                "hog": {
                    "hog_image_size": [160, 160],
                    "orientations": 9,
                    "pixels_per_cell": [32, 32],
                    "cells_per_block": [2, 2],
                    "block_norm": "L2-Hys",
                },
            },
            "models": {
                "seed": 42,
                "test_size": 0.2,
                "svm": {"kernel": "rbf", "class_weight": "balanced", "random_state": 42},
                "rf": {"n_estimators": 100, "class_weight": "balanced", "random_state": 42},
                "xgb": {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 4, "random_state": 42},
            },
        }

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
