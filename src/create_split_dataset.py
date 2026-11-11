"""
One-time stratified train/test dataset partition script.
Splits processed dataset into frozen 'train' and 'test' directories.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DATA_DIR, PROCESSED_DATA_DIR, load_config
from src.data_loader import DatasetLoader
from src.logger import get_logger

logger = get_logger(__name__)


def create_frozen_split(
    processed_dir: Path | str,
    output_dir: Path | str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Performs stratified split and copies images to frozen train/test directories.
    """
    processed_path = Path(processed_dir)
    output_path = Path(output_dir)

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_path}")

    logger.info("Initializing DatasetLoader for split from: %s", processed_path)
    loader = DatasetLoader(data_dir=processed_path)
    loader.split_and_save_dataset(
        output_base_dir=output_path, test_size=test_size, random_state=random_state
    )
    logger.info("Frozen dataset partition created successfully under: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Perform stratified split and save to data/train and data/test"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config file"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Path to processed images folder"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Base destination folder"
    )
    parser.add_argument(
        "--test-size", type=float, default=None, help="Test set fraction (e.g. 0.2)"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random state seed"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    models_config = config.get("models", {})

    test_size = args.test_size if args.test_size is not None else models_config.get("test_size", 0.2)
    random_state = args.seed if args.seed is not None else models_config.get("seed", 42)
    processed_dir = Path(args.data_dir) if args.data_dir else PROCESSED_DATA_DIR
    output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR

    create_frozen_split(
        processed_dir=processed_dir,
        output_dir=output_dir,
        test_size=test_size,
        random_state=random_state,
    )


if __name__ == "__main__":
    main()
