from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from prefect import flow, task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AVAILABLE_MODELS = ("rf", "svm", "xgb")
AVAILABLE_FEATURES = ("stats", "lbp", "hog")
DVC_TARGETS = (
    "data/raw.dvc",
    "data/processed.dvc",
    "data/splits.csv.dvc",
    "data/features.dvc",
)


@task(name="Run pipeline step", log_prints=True)
def run_command(step_name: str, command: list[str], dry_run: bool) -> None:
    if dry_run:
        print(f"Would run {step_name}: {command!r}")
        return
    print(f"Starting: {step_name}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def module_command(module: str, *arguments: str) -> list[str]:
    return [sys.executable, "-m", module, *arguments]


def dvc_command(*arguments: str) -> list[str]:
    executable = shutil.which("dvc")
    if executable is None:
        raise RuntimeError(
            "DVC is not available; install the mlops dependency group.")
    return [executable, *arguments]


@flow(name="covid-xray-mlops", log_prints=True)
def run_mlop_pipeline(
    model: str = "xgb",
    features: tuple[str, ...] = ("stats", "lbp", "hog"),
    push_data: bool = False,
    dry_run: bool = False,
) -> None:
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Unsupported model: {model}")
    if not features or any(feature not in AVAILABLE_FEATURES for feature in features):
        raise ValueError(
            f"Features must be selected from {AVAILABLE_FEATURES}.")
    if len(set(features)) != len(features):
        raise ValueError("Features must not contain duplicates.")

    run_command("Preprocess images", module_command(
        "src.preprocessing"), dry_run)
    run_command(
        "Create train/validation/test splits",
        module_command("src.create_split_dataset"),
        dry_run,
    )
    run_command("Extract image features", module_command(
        "src.feature_extraction_ml"), dry_run)

    for data_path in (
        "data/raw",
        "data/processed",
        "data/splits.csv",
        "data/features",
    ):
        run_command(
            f"Track {data_path} with DVC",
            dvc_command("add", data_path),
            dry_run,
        )

    if push_data:
        run_command(
            "Push data to DagsHub",
            dvc_command("push", "-r", "dagshub", *DVC_TARGETS),
            dry_run,
        )

    feature_arguments = (*features,)
    run_command(
        f"Train {model.upper()}",
        module_command(
            "src.train_ml",
            "--model",
            model,
            "--features",
            *feature_arguments,
        ),
        dry_run,
    )

    model_filename = f"{model}_{'_'.join(features)}.joblib"
    run_command(
        "Evaluate candidate on the held-out test split",
        module_command(
            "src.evaluate",
            "--type",
            "ml",
            "--filename",
            model_filename,
            "--features",
            *feature_arguments,
        ),
        dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the COVID X-ray MLOps pipeline with Prefect.")
    parser.add_argument("--model", choices=AVAILABLE_MODELS, default="xgb")
    parser.add_argument(
        "--features",
        nargs="+",
        choices=AVAILABLE_FEATURES,
        default=["stats", "lbp", "hog"],
    )
    parser.add_argument(
        "--push-data",
        action="store_true",
        help="Push DVC-tracked data to the configured DagsHub remote.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show pipeline commands without running them.",
    )
    args = parser.parse_args()
    run_mlop_pipeline(
        model=args.model,
        features=tuple(args.features),
        push_data=args.push_data,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
