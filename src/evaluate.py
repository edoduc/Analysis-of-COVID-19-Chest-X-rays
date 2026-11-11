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
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn

from src.config import DEFAULT_CLASSES, FEATURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, REPORTS_DIR, TEST_DATA_DIR
from src.logger import get_logger
from src.models import CustomCNN, build_model

logger = get_logger(__name__)


def evaluate_ml_model(
    model_path: Path, features_dir: Path, feature_list: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Loads a Scikit-Learn model or Pipeline and evaluates it on test feature arrays."""
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


def evaluate_deep_model(
    model_path: Path,
    data_dir: Path | None = None,
    arch: str = "resnet50",
    img_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    device_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates a PyTorch model on the frozen test dataset (or fallback validation split).
    """
    from torch.utils.data import DataLoader, random_split
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    logger.info("Evaluating Deep Learning model (%s): %s", arch, model_path.name)
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Requested device: %s", device)

    # 1. Instantiate and load weights
    model = build_model(arch, num_classes=len(DEFAULT_CLASSES))
    model = load_pytorch_checkpoint(model, model_path, device)
    actual_device = next(model.parameters()).device
    logger.info("Active model compute device: %s", actual_device)

    # 2. Setup transforms
    if arch.lower() == "custom_cnn":
        eval_transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )
    else:
        eval_transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # 3. Determine evaluation dataset source
    test_path = TEST_DATA_DIR if data_dir is None else Path(data_dir)

    if test_path.exists() and any(test_path.iterdir()):
        logger.info("Evaluating directly on frozen test dataset: %s", test_path)
        eval_dataset = ImageFolder(root=str(test_path), transform=eval_transform)
    else:
        fallback_dir = PROCESSED_DATA_DIR
        logger.warning(
            "Test directory '%s' not found. Falling back to 20%% split on '%s'.",
            test_path,
            fallback_dir,
        )
        full_dataset = ImageFolder(root=str(fallback_dir), transform=eval_transform)
        generator = torch.Generator().manual_seed(42)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        _, eval_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

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
        "--data-dir",
        type=str,
        default=None,
        help="Custom test dataset directory (for deep models)",
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
    classes_file = FEATURES_DIR / "classes.npy"
    if classes_file.exists():
        class_names = list(np.load(classes_file))
    else:
        class_names = DEFAULT_CLASSES

    if args.type == "ml":
        y_test, y_pred, _ = evaluate_ml_model(model_path, FEATURES_DIR, args.features)
    else:
        data_dir = Path(args.data_dir) if args.data_dir else None
        y_test, y_pred, _ = evaluate_deep_model(
            model_path, data_dir=data_dir, arch=args.arch, device_name=args.device
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
