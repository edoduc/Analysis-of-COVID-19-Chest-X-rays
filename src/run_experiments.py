"""
Ablation experiment runner for classical machine learning models.
Executes systematic feature set combinations with secure process management.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from typing import Sequence

from src.config import PROJECT_ROOT
from src.logger import get_logger

logger = get_logger(__name__)


def get_all_combinations(feature_list: Sequence[str]) -> list[tuple[str, ...]]:
    """
    Generates all non-empty subsets of the provided features (solos, pairs, etc.).
    """
    all_combinations: list[tuple[str, ...]] = []
    for r in range(1, len(feature_list) + 1):
        all_combinations.extend(itertools.combinations(feature_list, r))
    return all_combinations


def run_ablation_study(
    models: Sequence[str] = ("rf", "svm"),
    features: Sequence[str] = ("stats", "lbp", "hog"),
    exp_name: str = "XRay_Ablation_Study",
) -> None:
    """
    Runs automated grid of training runs using secure subprocess execution.
    """
    train_script = PROJECT_ROOT / "src" / "train_ml.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    combinations = get_all_combinations(features)
    total_runs = len(combinations) * len(models)
    logger.info("Starting ablation study: %d runs (%d models x %d feature sets)", total_runs, len(models), len(combinations))

    current_run = 1
    for model in models:
        for combo in combinations:
            combo_name = "_".join(combo)
            logger.info(
                "[%d/%d] Launching %s with features: %s",
                current_run,
                total_runs,
                model.upper(),
                combo_name,
            )

            # Secure execution: shell=False, sys.executable ensures same virtualenv
            cmd = [
                sys.executable,
                str(train_script),
                "--model",
                model,
                "--features",
                *combo,
                "--exp_name",
                exp_name,
            ]

            start_time = time.time()
            try:
                subprocess.run(cmd, check=True)
                elapsed = time.time() - start_time
                logger.info(
                    "Finished %s with %s in %.1f seconds.",
                    model.upper(),
                    combo_name,
                    elapsed,
                )
            except subprocess.CalledProcessError as e:
                logger.error("Run %d (%s with %s) failed with return code %s", current_run, model, combo_name, e.returncode)

            current_run += 1

    logger.info("All ablation experiments completed. Inspect results in MLflow.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run systematic feature ablation experiments.")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["rf", "svm"],
        help="Models to test (choices: rf, svm, xgb)",
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=["stats", "lbp", "hog"],
        help="Available feature sets to combine",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="XRay_Ablation_Study",
        help="MLflow experiment name",
    )
    args = parser.parse_args()

    run_ablation_study(
        models=args.models,
        features=args.features,
        exp_name=args.exp_name,
    )


if __name__ == "__main__":
    main()
