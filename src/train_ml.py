"""
Training pipeline for classical machine learning models with MLflow tracking and feature scaling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import FEATURES_DIR, MODELS_DIR, PROJECT_ROOT, load_config
from src.logger import get_logger

logger = get_logger(__name__)


def load_and_combine_features(
    features_dir: Path, split_name: str, feature_list: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dynamically loads and horizontally stacks requested feature arrays (.npy).
    """
    feature_arrays = []

    for feat in feature_list:
        file_path = features_dir / f"X_{split_name}_{feat}.npy"
        if not file_path.exists():
            raise FileNotFoundError(
                f"Could not find {file_path}. Did you run feature extraction for '{feat}'?"
            )

        logger.info("Loading '%s' features for %s...", feat, split_name)
        data = np.load(file_path)
        feature_arrays.append(data)

    x_combined = np.hstack(feature_arrays)
    y_path = features_dir / f"y_{split_name}.npy"
    if not y_path.exists():
        raise FileNotFoundError(f"Missing label file: {y_path}")
    y = np.load(y_path)

    return x_combined, y


def build_estimator(model_name: str, models_config: dict[str, Any]) -> Any:
    """
    Instantiates the machine learning classifier based on config.
    """
    model_name = model_name.lower()
    if model_name == "svm":
        cfg = models_config.get("svm", {})
        return SVC(
            kernel=cfg.get("kernel", "rbf"),
            class_weight=cfg.get("class_weight", "balanced"),
            probability=True,
            random_state=cfg.get("random_state", 42),
        )
    elif model_name == "rf":
        cfg = models_config.get("rf", {})
        return RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 100),
            class_weight=cfg.get("class_weight", "balanced"),
            random_state=cfg.get("random_state", 42),
            n_jobs=-1,
        )
    elif model_name == "xgb":
        cfg = models_config.get("xgb", {})
        return XGBClassifier(
            n_estimators=cfg.get("n_estimators", 150),
            learning_rate=cfg.get("learning_rate", 0.1),
            max_depth=cfg.get("max_depth", 4),
            random_state=cfg.get("random_state", 42),
            n_jobs=-1,
        )
    else:
        raise ValueError(
            f"Model '{model_name}' not supported. Choose from 'svm', 'rf', 'xgb'."
        )


def build_pipeline(
    model_name: str, models_config: dict[str, Any], use_scaler: bool = True
) -> Pipeline:
    """
    Wraps estimator in a Scikit-Learn Pipeline with optional StandardScaler.
    Production best practice: pipeline encapsulates feature scaling parameters.
    """
    estimator = build_estimator(model_name, models_config)
    steps = []
    # SVMs strictly benefit from scaling. Trees are scale-invariant, but scaling doesn't hurt them.
    if use_scaler or model_name == "svm":
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def train(
    model_name: str = "rf",
    features: Sequence[str] = ("stats", "hog"),
    exp_name: str = "XRay_Classical_ML",
    use_scaler: bool = True,
    config_path: Path | str | None = None,
) -> None:
    """
    Full training workflow: data loading -> pipeline fitting -> validation -> artifact logging.
    """
    config = load_config(config_path)
    models_config = config.get("models", {})

    models_dir = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    features_dir = FEATURES_DIR

    logger.info("Preparing to train %s with features: %s", model_name.upper(), features)
    x_train, y_train = load_and_combine_features(features_dir, "train", features)
    x_test, y_test = load_and_combine_features(features_dir, "test", features)

    logger.info("Training matrix shape: %s", x_train.shape)
    logger.info("Testing matrix shape: %s", x_test.shape)

    mlflow.set_experiment(exp_name)
    run_name = f"{model_name}_{'_'.join(features)}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("features_used", list(features))
        mlflow.log_param("feature_vector_size", x_train.shape[1])
        mlflow.log_param("use_scaler", use_scaler)

        spec_config = models_config.get(model_name, {})
        for param_name, param_val in spec_config.items():
            mlflow.log_param(f"hyperparam_{param_name}", param_val)

        logger.info("Building model pipeline (scaler=%s)...", use_scaler)
        pipeline = build_pipeline(model_name, models_config, use_scaler=use_scaler)

        logger.info("Training pipeline...")
        pipeline.fit(x_train, y_train)
        logger.info("Training completed.")

        logger.info("Predicting on test set...")
        y_pred = pipeline.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1_macro", f1_macro)
        mlflow.log_metric("test_f1_weighted", f1_weighted)

        logger.info("Test Accuracy: %.4f", acc)
        logger.info("Test F1 (Weighted): %.4f", f1_weighted)
        logger.info("Test F1 (Macro): %.4f", f1_macro)

        # Save pipeline model locally and to MLflow
        model_filename = f"{model_name}_{'_'.join(features)}.joblib"
        model_path = models_dir / model_filename
        joblib.dump(pipeline, model_path)
        try:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="pipeline_model",
                serialization_format="cloudpickle",
            )
        except Exception as e:
            logger.warning("MLflow model logging with cloudpickle: %s", e)
            mlflow.log_artifact(str(model_path), artifact_path="model_joblib")

        logger.info("Model pipeline saved locally to %s", model_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ML models on extracted X-Ray features."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["svm", "rf", "xgb"],
        default="rf",
        help="Model architecture to train",
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=["stats", "hog"],
        help="List of features to combine (e.g., stats hog lbp)",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="XRay_Classical_ML",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.json",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Disable StandardScaler in pipeline",
    )
    args = parser.parse_args()

    train(
        model_name=args.model,
        features=args.features,
        exp_name=args.exp_name,
        use_scaler=not args.no_scale,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
