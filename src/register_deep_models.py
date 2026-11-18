"""
Post-hoc registration of externally trained Deep Learning models (Google Colab)
into MLflow, evaluated on the same frozen test set as the classical ML models.

For each model, ONE run contains:
  - params + tags (git commit, DVC hash of the test data, preprocessing, ...)
  - the evaluation dataset (mlflow.log_input) -> "Datasets used" in the UI
  - test_* metrics (sklearn), classification report and confusion matrix
  - the PyTorch model, registered in the Model Registry

Predictions are cached on disk, so a failure after inference never costs
another full pass over the test set.

Usage:
    uv run src/register_deep_models.py
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import mlflow
import mlflow.data
import mlflow.pytorch
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import torch
from torchvision.datasets import ImageFolder
import yaml

from src.config import DEFAULT_CLASSES, PROJECT_ROOT, REPORTS_DIR, TEST_DATA_DIR
from src.evaluate import evaluate_deep_model, plot_confusion_matrix
from src.logger import get_logger
from src.models.architectures import build_model

load_dotenv()
logger = get_logger(__name__)

IMG_SIZE = 224
EXPERIMENT_NAME = "Deep_Model_Registry"
DATASET_NAME = "covid_xray_test"  # use the SAME name for ML and DL models
CACHE_DIR = REPORTS_DIR / "cache"

# None = use the GPU if available, otherwise CPU. Set to "cpu" to force CPU.
INFERENCE_DEVICE: str | None = None

DEEP_MODELS: list[dict[str, Any]] = [
    {
        "name": "resnet50_xray",
        "arch": "resnet50",
        "checkpoint": PROJECT_ROOT / "models" / "resnet50" / "resnet50_best.pth",
        "state_dict_key": None,
        "preprocessing": "grayscale3+imagenet_norm",
        "description": "ResNet50 fine-tuned for COVID-19 Chest X-Ray classification (4 classes).",
    },
    {
        "name": "custom_cnn_xray",
        "arch": "custom_cnn",
        "checkpoint": PROJECT_ROOT / "models" / "custom_cnn" / "best_checkpoint.pt",
        "state_dict_key": "model_state_dict",
        "preprocessing": "grayscale3+[0,1]",
        "description": "Custom 5-block CNN trained for COVID-19 Chest X-Ray classification (4 classes).",
    },
]

# Keys that may exist in a Colab checkpoint dict (logged as colab_*, NOT as test_*)
COLAB_METRIC_KEYS = ("val_acc", "val_loss", "train_acc", "train_loss", "epoch")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def get_dvc_hash(data_path: Path) -> str:
    """
    Read the DVC md5 of the test data from '<data_path>.dvc' (or from dvc.lock).
    Returns "unknown" if the data is not tracked that way.
    """
    dvc_file = data_path.with_name(data_path.name + ".dvc")
    try:
        if dvc_file.exists():
            return yaml.safe_load(dvc_file.read_text())["outs"][0]["md5"]
        lock = PROJECT_ROOT / "dvc.lock"
        if lock.exists():
            rel = data_path.relative_to(PROJECT_ROOT).as_posix()
            for stage in yaml.safe_load(lock.read_text()).get("stages", {}).values():
                for out in stage.get("outs", []):
                    if out.get("path") == rel:
                        return out["md5"]
    except Exception:
        logger.exception("Could not read DVC hash")
    return "unknown"


def check_test_set() -> int:
    """Fail loudly instead of silently evaluating on a different dataset."""
    if not TEST_DATA_DIR.exists() or not any(TEST_DATA_DIR.iterdir()):
        raise FileNotFoundError(
            f"Frozen test set not found at {TEST_DATA_DIR}. Run `dvc pull` first."
        )
    ds = ImageFolder(root=str(TEST_DATA_DIR))
    if ds.classes != list(DEFAULT_CLASSES):
        raise ValueError(
            f"Class order mismatch: ImageFolder={ds.classes} vs DEFAULT_CLASSES={list(DEFAULT_CLASSES)}"
        )
    return len(ds)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_precision_macro": float(p),
        "test_recall_macro": float(r),
        "test_f1_macro": float(f1),
        "test_f1_weighted": float(f1_w),
    }


def load_model_instance(
    arch: str, checkpoint_path: Path, state_dict_key: str | None
) -> tuple[torch.nn.Module, dict[str, float]]:
    """Load weights on CPU; also return any numeric training metrics in the checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model = build_model(arch, num_classes=len(DEFAULT_CLASSES))
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = raw[state_dict_key] if state_dict_key else raw
    # Colab DataParallel checkpoints prefix keys with "module."
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    colab_metrics: dict[str, float] = {}
    if state_dict_key and isinstance(raw, dict):
        for k in COLAB_METRIC_KEYS:
            if k in raw:
                try:
                    colab_metrics[f"colab_{k}"] = float(raw[k])
                except (TypeError, ValueError):
                    pass

    logger.info("Loaded '%s' weights from %s", arch, checkpoint_path.name)
    return model, colab_metrics


def already_registered(client: mlflow.MlflowClient, model_name: str, ckpt_sha: str) -> bool:
    """Idempotency: skip if a version with the same checkpoint hash exists."""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception:
        return False
    return any(v.tags.get("checkpoint_sha256") == ckpt_sha for v in versions)


def get_predictions(
    model_cfg: dict[str, Any], ckpt_sha: str, dvc_hash: str
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on the frozen test set, or reload cached predictions."""
    cache = CACHE_DIR / f"preds_{model_cfg['name']}_{ckpt_sha[:12]}_{dvc_hash[:8]}.npz"
    if cache.exists():
        d = np.load(cache)
        logger.info("Loaded cached predictions from %s", cache.name)
        return d["y_true"], d["y_pred"]

    y_true, y_pred, _ = evaluate_deep_model(
        model_cfg["checkpoint"],
        data_dir=TEST_DATA_DIR,
        arch=model_cfg["arch"],
        device_name=INFERENCE_DEVICE,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, y_true=y_true, y_pred=y_pred)
    logger.info("Predictions cached to %s", cache.name)
    return y_true, y_pred


# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #
def register_model_to_mlflow(
    model_cfg: dict[str, Any],
    client: mlflow.MlflowClient,
    git_commit: str,
    dvc_hash: str,
) -> None:
    model_name = model_cfg["name"]
    ckpt: Path = model_cfg["checkpoint"]
    ckpt_sha = sha256_of(ckpt)

    if already_registered(client, model_name, ckpt_sha):
        logger.info("%s: this checkpoint is already registered, skipping.", model_name)
        return

    logger.info("Registering model: %s", model_name)

    # 1. Predictions on the frozen test set (cached)
    y_true, y_pred = get_predictions(model_cfg, ckpt_sha, dvc_hash)

    # 2. Model + signature
    model, colab_metrics = load_model_instance(
        model_cfg["arch"], ckpt, model_cfg["state_dict_key"]
    )
    input_example = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    with torch.no_grad():
        output_example = model(torch.from_numpy(input_example)).numpy()
    signature = infer_signature(input_example, output_example)

    # 3. MLflow dataset (metadata + hash only; images stay in DVC)
    eval_df = pd.DataFrame({"label": y_true, "prediction": y_pred})
    dataset = mlflow.data.from_pandas(
        eval_df, name=DATASET_NAME, targets="label", predictions="prediction"
    )

    with mlflow.start_run(run_name=f"register_{model_name}") as run:
        logger.info("Started run %s", run.info.run_id)
        mlflow.set_tags(
            {
                "source": "google_colab",
                "framework": "pytorch",
                "registered_post_hoc": "true",
                "metrics_source": "recomputed_on_frozen_test_set",
                "git_commit": git_commit,
                "dvc_test_data_md5": dvc_hash,
                "preprocessing": model_cfg["preprocessing"],
                "checkpoint_sha256": ckpt_sha,
            }
        )
        mlflow.log_params(
            {
                "arch": model_cfg["arch"],
                "num_classes": len(DEFAULT_CLASSES),
                "classes": ",".join(DEFAULT_CLASSES),
                "img_size": IMG_SIZE,
                "checkpoint_file": ckpt.name,
            }
        )
        if colab_metrics:
            mlflow.log_metrics(colab_metrics)  # training-time numbers, kept separate

        logger.info("Logging evaluation dataset...")
        mlflow.log_input(dataset, context="evaluation")

        logger.info("Logging test metrics...")
        metrics = compute_metrics(y_true, y_pred)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(
            classification_report(
                y_true, y_pred, target_names=list(DEFAULT_CLASSES), output_dict=True, zero_division=0
            ),
            "evaluation/classification_report.json",
        )
        logger.info("Test metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

        logger.info("Logging confusion matrix...")
        cm_path = REPORTS_DIR / f"cm_{model_name}.png"
        plot_confusion_matrix(y_true, y_pred, list(DEFAULT_CLASSES), cm_path)
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation")

        logger.info("Uploading and registering model (this can take a minute)...")
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",  # artifact_path="model" on MLflow 2.x
            registered_model_name=model_name,
            signature=signature,
            input_example=input_example,
        )

    version = model_info.registered_model_version
    client.update_registered_model(model_name, description=model_cfg["description"])
    client.update_model_version(
        model_name, version, description=f"Imported from {ckpt.name}"
    )
    client.set_model_version_tag(model_name, version, "checkpoint_sha256", ckpt_sha)
    client.set_model_version_tag(model_name, version, "dvc_test_data_md5", dvc_hash)
    client.set_registered_model_alias(model_name, "candidate", version)
    logger.info("Registered %s version %s (alias 'candidate')", model_name, version)


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is not set. Check your .env file.")

    mlflow.set_tracking_uri(tracking_uri)
    logger.info("Connected to MLflow Tracking URI: %s", tracking_uri)

    n_test = check_test_set()
    git_commit = get_git_commit()
    dvc_hash = get_dvc_hash(TEST_DATA_DIR)
    logger.info(
        "Test set: %d images | git=%s | dvc test data md5=%s", n_test, git_commit[:8], dvc_hash
    )

    client = mlflow.MlflowClient()
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        client.create_experiment(EXPERIMENT_NAME, artifact_location="mlflow-artifacts:/")
    mlflow.set_experiment(EXPERIMENT_NAME)

    for model_cfg in DEEP_MODELS:
        try:
            register_model_to_mlflow(model_cfg, client, git_commit, dvc_hash)
        except Exception:
            logger.exception("Failed to register %s", model_cfg["name"])

    logger.info("Registration script finished.")


if __name__ == "__main__":
    main()