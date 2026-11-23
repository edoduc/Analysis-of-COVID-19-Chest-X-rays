"""
Post-hoc registration of externally trained Deep Learning models (Google Colab)
into MLflow, evaluated on the SAME frozen test split as the classical ML models:
the test rows of the DVC-tracked manifest data/splits.csv, read via DatasetLoader.

For each model, ONE run contains:
  - params + tags (git commit, split manifest / processed data DVC md5, preprocessing, ...)
  - the evaluation dataset (mlflow.log_input) -> same "covid_xray_test" digest as ML runs
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
import mlflow.pytorch
from mlflow.models import infer_signature
import numpy as np
from sklearn.metrics import classification_report
import torch

from src.config import DEFAULT_CLASSES, PROJECT_ROOT, REPORTS_DIR
from src.data_loader import DatasetLoader
from src.evaluate import compute_metrics, evaluate_deep_model, log_test_dataset, plot_confusion_matrix
from src.logger import get_logger
from src.models.architectures import build_model

load_dotenv()
logger = get_logger(__name__)

IMG_SIZE = 224
EXPERIMENT_NAME = "Deep_Model_Registry"
CACHE_DIR = REPORTS_DIR / "cache"
# Part of the idempotency key: bump it to force re-registration in a new format.
SERIALIZATION_FORMAT = "pickle"

# None = use the GPU if available, otherwise CPU. Set to "cpu" to force CPU.
INFERENCE_DEVICE: str | None = None

# Only models trained on the train/validation splits of data/splits.csv belong here.
# The custom CNN is not listed: it was trained on an older split, so its test
# metrics would be leaked. Add it back once retrained on splits.csv.
DEEP_MODELS: list[dict[str, Any]] = [
    {
        "name": "resnet50_xray",
        "arch": "resnet50",
        "checkpoint": PROJECT_ROOT / "models" / "resnet50" / "resnet50_best.pth",
        "state_dict_key": None,
        "preprocessing": "grayscale3+imagenet_norm",
        "description": "ResNet50 fine-tuned for COVID-19 Chest X-Ray classification (4 classes).",
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


def already_registered(
    client: mlflow.MlflowClient, model_name: str, ckpt_sha: str, split_md5: str
) -> bool:
    """
    Idempotency: skip if a version exists for the same checkpoint, evaluated on
    the same split manifest, saved in the same serialization format.
    These are MODEL VERSION tags (not run tags), set atomically at registration.
    """
    versions = client.search_model_versions(f"name='{model_name}'")
    return any(
        v.tags.get("checkpoint_sha256") == ckpt_sha
        and v.tags.get("split_manifest_md5") == split_md5
        and v.tags.get("serialization_format") == SERIALIZATION_FORMAT
        for v in versions
    )


def get_predictions(
    model_cfg: dict[str, Any], ckpt_sha: str, split_md5: str, loader: DatasetLoader
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on the frozen test split, or reload cached predictions."""
    cache = CACHE_DIR / f"preds_{model_cfg['name']}_{ckpt_sha[:12]}_{split_md5[:12]}.npz"
    if cache.exists():
        d = np.load(cache)
        logger.info("Loaded cached predictions from %s", cache.name)
        return d["y_true"], d["y_pred"]

    y_true, y_pred, _ = evaluate_deep_model(
        model_cfg["checkpoint"],
        arch=model_cfg["arch"],
        device_name=INFERENCE_DEVICE,
        loader=loader,
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
    loader: DatasetLoader,
    fingerprint: dict[str, str],
) -> None:
    model_name = model_cfg["name"]
    ckpt: Path = model_cfg["checkpoint"]
    ckpt_sha = sha256_of(ckpt)
    split_md5 = fingerprint["split_manifest_md5"]

    if already_registered(client, model_name, ckpt_sha, split_md5):
        logger.info("%s: this checkpoint is already registered, skipping.", model_name)
        return

    logger.info("Registering model: %s", model_name)

    # 1. Predictions on the frozen test split (cached)
    y_true, y_pred = get_predictions(model_cfg, ckpt_sha, split_md5, loader)

    # 2. Model + signature
    model, colab_metrics = load_model_instance(
        model_cfg["arch"], ckpt, model_cfg["state_dict_key"]
    )
    input_example = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    with torch.no_grad():
        output_example = model(torch.from_numpy(input_example)).numpy()
    signature = infer_signature(input_example, output_example)

    # Tags shared by the run and the model version. The version tags are the
    # idempotency key read by already_registered().
    lineage_tags = {
        "checkpoint_sha256": ckpt_sha,
        "serialization_format": SERIALIZATION_FORMAT,
        **fingerprint,
    }

    with mlflow.start_run(run_name=f"register_{model_name}") as run:
        logger.info("Started run %s", run.info.run_id)
        mlflow.set_tags(
            {
                "source": "google_colab",
                "framework": "pytorch",
                "registered_post_hoc": "true",
                "metrics_source": "recomputed_on_frozen_test_set",
                "git_commit": git_commit,
                "preprocessing": model_cfg["preprocessing"],
                **lineage_tags,
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
        log_test_dataset(loader, fingerprint)

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

        logger.info("Uploading model (this can take a minute)...")
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",  # artifact_path="model" on MLflow 2.x
            signature=signature,
            input_example=input_example,
            serialization_format=SERIALIZATION_FORMAT,
        )

    # Register separately so the version is created WITH its tags in one call:
    # a crash can no longer leave an untagged version that defeats idempotency.
    logger.info("Registering model version...")
    model_version = mlflow.register_model(
        model_info.model_uri, model_name, tags=lineage_tags
    )
    version = model_version.version
    client.update_registered_model(model_name, description=model_cfg["description"])
    client.update_model_version(
        model_name, version, description=f"Imported from {ckpt.name}"
    )
    client.set_registered_model_alias(model_name, "candidate", version)
    logger.info("Registered %s version %s (alias 'candidate')", model_name, version)


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is not set. Check your .env file.")

    mlflow.set_tracking_uri(tracking_uri)
    logger.info("Connected to MLflow Tracking URI: %s", tracking_uri)

    loader = DatasetLoader()
    fingerprint = loader.fingerprint()  # raises if splits.csv != its committed .dvc
    git_commit = get_git_commit()
    logger.info(
        "Test split: %d images | git=%s | %s",
        len(loader.split_frame("test")), git_commit[:8], fingerprint,
    )

    client = mlflow.MlflowClient()
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        client.create_experiment(EXPERIMENT_NAME, artifact_location="mlflow-artifacts:/")
    mlflow.set_experiment(EXPERIMENT_NAME)

    for model_cfg in DEEP_MODELS:
        try:
            register_model_to_mlflow(model_cfg, client, git_commit, loader, fingerprint)
        except Exception:
            logger.exception("Failed to register %s", model_cfg["name"])

    logger.info("Registration script finished.")


if __name__ == "__main__":
    main()
