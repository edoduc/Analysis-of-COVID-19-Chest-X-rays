"""
Evaluation and benchmarking pipeline for Classical ML and PyTorch Deep Learning models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

import torch
import torch.nn as nn

from src.config import DEFAULT_CLASSES, FEATURES_DIR, MODELS_DIR, PROJECT_ROOT, REPORTS_DIR
from src.data_loader import DatasetLoader
from src.feature_extraction_ml import assert_features_match
from src.logger import get_logger
from src.models import build_model

logger = get_logger(__name__)

# Same MLflow dataset name for every model (ML and DL): they are all evaluated
# on the manifest's test split, so MLflow shows one shared dataset digest.
DATASET_NAME = "covid_xray_test"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Test metrics logged identically for every model family."""
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_precision_macro": float(p),
        "test_recall_macro": float(r),
        "test_f1_macro": float(f1),
        "test_f1_weighted": float(f1_w),
    }


def log_test_dataset(loader: DatasetLoader, fingerprint: dict[str, str]) -> None:
    """
    Logs the frozen test split to the active MLflow run: lineage tags plus an
    `mlflow.data` input built from the manifest rows only (no predictions), so
    the dataset digest is identical across all models.
    """
    import mlflow
    import mlflow.data

    mlflow.set_tags(fingerprint)
    test_rows = loader.split_frame("test")[["class_name", "image_path"]]
    dataset = mlflow.data.from_pandas(test_rows, name=DATASET_NAME, targets="class_name")
    mlflow.log_input(dataset, context="evaluation")


def evaluate_ml_model(
    model_path: Path, features_dir: Path, feature_list: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Loads a Scikit-Learn model or Pipeline and evaluates it on test feature arrays."""
    assert_features_match(features_dir, DatasetLoader(), ["test"])
    logger.info("Loading ML model artifact: %s", model_path.name)
    model = joblib.load(model_path)

    feature_arrays = [np.load(features_dir / f"X_test_{feat}.npy") for feat in feature_list]
    x_test = np.hstack(feature_arrays)
    y_test = np.load(features_dir / "y_test.npy")

    logger.info("Running inference on test feature matrix shape: %s", x_test.shape)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None

    return y_test, y_pred, y_prob


def load_pytorch_checkpoint(model: nn.Module, model_path: Path, device: torch.device) -> nn.Module:
    """
    Safely loads weights into model, prioritizing weights_only=True.
    Loads to CPU first to avoid GPU deserialization OOM spikes, then transfers to device.
    """
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception as e:
        logger.warning(
            "weights_only=True failed (%s). Falling back to weights_only=False (ensure model source is trusted).", e
        )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    try:
        return model.to(device).eval()
    except (torch.AcceleratorError, RuntimeError) as e:
        logger.warning("Transfer to %s failed (%s). Falling back to CPU.", device, e)
        return model.to("cpu").eval()


def eval_transform_for(arch: str, img_size: tuple[int, int] = (224, 224)):
    """Evaluation transform matching how each architecture was trained."""
    from torchvision import transforms

    steps = [
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
    if arch.lower() != "custom_cnn":
        steps.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(steps)


def evaluate_deep_model(
    model_path: Path,
    arch: str = "resnet50",
    img_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    device_name: str | None = None,
    loader: DatasetLoader | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates a PyTorch model on the frozen test split of the manifest.
    There is deliberately no fallback: if the manifest is missing, this fails.
    """
    from torch.utils.data import DataLoader

    logger.info("Evaluating Deep Learning model (%s): %s", arch, model_path.name)
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Requested device: %s", device)

    model = build_model(arch, num_classes=len(DEFAULT_CLASSES))
    model = load_pytorch_checkpoint(model, model_path, device)
    actual_device = next(model.parameters()).device
    logger.info("Active model compute device: %s", actual_device)

    loader = loader or DatasetLoader()
    eval_dataset = loader.torch_dataset("test", transform=eval_transform_for(arch, img_size))

    val_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[np.ndarray] = []

    logger.info("Running inference on %d test samples...", len(eval_dataset))
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(actual_device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(targets.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str], save_path: Path
) -> None:
    """Generates and saves a clean Confusion Matrix plot using Scikit-Learn and Matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        display_labels=class_names,
        cmap="Blues",
        ax=ax,
        colorbar=True,
    )
    disp.ax_.set_title("Confusion Matrix")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    logger.info("Confusion Matrix saved to: %s", save_path)


def resolve_model_path(filename: str) -> Path:
    """Searches for model checkpoint at standard paths."""
    candidate = Path(filename)
    if candidate.exists():
        return candidate

    standard_dirs = [
        MODELS_DIR,
        PROJECT_ROOT / "models" / "resnet50",
        PROJECT_ROOT / "models" / "custom_cnn",
        PROJECT_ROOT / "models" / "saved_models",
        PROJECT_ROOT / "models",
    ]

    for d in standard_dirs:
        p = d / filename
        if p.exists():
            return p

    raise FileNotFoundError(f"Model file '{filename}' not found in any standard directory: {standard_dirs}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and compare local framework models."
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["ml", "deep"],
        required=True,
        help="Type of model to evaluate ('ml' or 'deep')",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["resnet50", "custom_cnn", "mobilenet"],
        default="resnet50",
        help="Architecture of deep model",
    )
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Filename or path of model weights",
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="*",
        default=["stats", "hog"],
        help="Features used (only for ML models)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Target compute device ('cuda' or 'cpu')",
    )
    args = parser.parse_args()

    model_path = resolve_model_path(args.filename)
    class_names = list(DEFAULT_CLASSES)

    if args.type == "ml":
        y_test, y_pred, _ = evaluate_ml_model(model_path, FEATURES_DIR, args.features)
    else:
        y_test, y_pred, _ = evaluate_deep_model(
            model_path, arch=args.arch, device_name=args.device
        )

    acc = accuracy_score(y_test, y_pred)
    logger.info("Evaluation Results for: %s", args.filename)
    logger.info("Overall Accuracy: %.4f", acc)
    print("\n" + "=" * 50)
    print(f"RESULTS FOR: {args.filename} ({args.arch if args.type == 'deep' else 'ML'})")
    print(f"Overall Accuracy: {acc:.4f}")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm_filename = f"cm_{Path(args.filename).stem}.png"
    plot_confusion_matrix(y_test, y_pred, class_names, REPORTS_DIR / cm_filename)


if __name__ == "__main__":
    main()
