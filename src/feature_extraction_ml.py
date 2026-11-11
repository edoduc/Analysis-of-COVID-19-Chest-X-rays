"""
Feature extraction pipeline for classical machine learning models.
Extracts statistical ROI features, Local Binary Patterns (LBP), and Histogram of Oriented Gradients (HOG).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.stats import kurtosis, skew
from skimage.feature import hog, local_binary_pattern
from skimage.measure import shannon_entropy
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.config import DATA_DIR, FEATURES_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, TEST_DATA_DIR, TRAIN_DATA_DIR, load_config
from src.data_loader import DatasetLoader
from src.logger import get_logger

logger = get_logger(__name__)


def extract_stat_features(image: np.ndarray) -> np.ndarray:
    """
    Statistical feature extraction on non-zero region of interest (ROI).
    Returns a 15-dimensional vector.
    """
    pixels = image[image > 0]

    if len(pixels) == 0:
        return np.zeros(15, dtype=np.float32)

    grad_x = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_pixels = grad_mag[image > 0]

    features = np.array(
        [
            pixels.mean(),
            pixels.std(),
            pixels.min(),
            pixels.max(),
            np.median(pixels),
            np.percentile(pixels, 5),
            np.percentile(pixels, 25),
            np.percentile(pixels, 75),
            np.percentile(pixels, 95),
            skew(pixels),
            kurtosis(pixels),
            shannon_entropy(pixels),
            cv2.Laplacian(image.astype(np.float32), cv2.CV_32F)[image > 0].var(),
            grad_pixels.mean(),
            grad_pixels.std(),
        ],
        dtype=np.float32,
    )

    return features


def extract_lbp_features(
    image: np.ndarray, radius: int = 3, method: str = "uniform"
) -> np.ndarray:
    """
    Local Binary Pattern feature extraction on non-zero ROI.
    Returns a (8*radius + 2)-dimensional normalized histogram vector.
    """
    n_points = 8 * radius
    lbp = local_binary_pattern(image, P=n_points, R=radius, method=method)

    mask = image > 0
    lbp_pixels = lbp[mask]
    n_bins = n_points + 2

    if len(lbp_pixels) == 0:
        return np.zeros(n_bins, dtype=np.float32)

    hist, _ = np.histogram(lbp_pixels, bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def extract_hog_features(
    image: np.ndarray,
    hog_image_size: tuple[int, int] = (160, 160),
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (32, 32),
    cells_per_block: tuple[int, int] = (2, 2),
    block_norm: str = "L2-Hys",
) -> np.ndarray:
    """
    Histogram of Oriented Gradients (HOG) feature extraction.
    """
    if image.shape[:2] != hog_image_size:
        image = cv2.resize(image, hog_image_size, interpolation=cv2.INTER_LINEAR)

    features = hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        visualize=False,
        feature_vector=True,
    )
    return features.astype(np.float32)


def extract_features_from_images(
    images: Sequence[np.ndarray],
    split_name: str,
    lbp_config: dict[str, Any],
    hog_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts stats, LBP, and HOG arrays from a sequence of images.
    """
    all_stats: list[np.ndarray] = []
    all_lbp: list[np.ndarray] = []
    all_hog: list[np.ndarray] = []

    radius = lbp_config.get("radius", 3)
    method = lbp_config.get("method", "uniform")

    hog_image_size = tuple(hog_config.get("hog_image_size", [160, 160]))
    orientations = hog_config.get("orientations", 9)
    pixels_per_cell = tuple(hog_config.get("pixels_per_cell", [32, 32]))
    cells_per_block = tuple(hog_config.get("cells_per_block", [2, 2]))
    block_norm = hog_config.get("block_norm", "L2-Hys")

    for img in tqdm(images, desc=f"Extracting {split_name} features"):
        all_stats.append(extract_stat_features(img))
        all_lbp.append(extract_lbp_features(img, radius=radius, method=method))
        all_hog.append(
            extract_hog_features(
                img,
                hog_image_size=hog_image_size,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm=block_norm,
            )
        )

    return np.array(all_stats), np.array(all_lbp), np.array(all_hog)


def save_features(
    output_dir: Path,
    split_name: str,
    stats: np.ndarray,
    lbp: np.ndarray,
    hog_feats: np.ndarray,
    y: np.ndarray,
) -> None:
    """Saves extracted feature matrices into .npy format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"X_{split_name}_stats.npy", stats)
    np.save(output_dir / f"X_{split_name}_lbp.npy", lbp)
    np.save(output_dir / f"X_{split_name}_hog.npy", hog_feats)
    np.save(output_dir / f"y_{split_name}.npy", y)
    logger.info("Saved %s feature matrices to %s", split_name, output_dir)


def run_feature_extraction(
    train_dir: Path | str | None = None,
    test_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    config_path: Path | str | None = None,
) -> None:
    """
    Main execution workflow for feature extraction.
    Prioritizes pre-split frozen directories (train/test), falling back to processed dataset.
    """
    config = load_config(config_path)
    fe_config = config.get("feature_extraction", {})
    lbp_config = fe_config.get("lbp", {})
    hog_config = fe_config.get("hog", {})

    out_path = Path(output_dir) if output_dir else FEATURES_DIR
    out_path.mkdir(parents=True, exist_ok=True)

    train_path = Path(train_dir) if train_dir else TRAIN_DATA_DIR
    test_path = Path(test_dir) if test_dir else TEST_DATA_DIR
    proc_path = Path(processed_dir) if processed_dir else PROCESSED_DATA_DIR

    label_encoder = LabelEncoder()

    # Case A: Frozen train and test sets exist on disk (preferred for reproducibility)
    if train_path.exists() and test_path.exists():
        logger.info("Detected frozen dataset splits at: %s and %s", train_path, test_path)
        train_loader = DatasetLoader(data_dir=train_path)
        test_loader = DatasetLoader(data_dir=test_path)

        x_train, y_train_raw = train_loader.load_dataset()
        x_test, y_test_raw = test_loader.load_dataset()

        # Fit label encoder on combined classes
        all_labels = np.concatenate([y_train_raw, y_test_raw])
        label_encoder.fit(all_labels)

        y_train = label_encoder.transform(y_train_raw)
        y_test = label_encoder.transform(y_test_raw)
    else:
        # Case B: Dynamic split from processed directory
        logger.info("Frozen splits not found. Loading from processed directory: %s", proc_path)
        loader = DatasetLoader(data_dir=proc_path)
        x_train, x_test, y_train, y_test, label_encoder = loader.get_train_test_split()

    # Save classes mapping
    np.save(out_path / "classes.npy", label_encoder.classes_)
    logger.info("Saved class mapping: %s", list(label_encoder.classes_))

    # Extract & Save Train Features
    train_stats, train_lbp, train_hog = extract_features_from_images(
        x_train, "train", lbp_config, hog_config
    )
    save_features(out_path, "train", train_stats, train_lbp, train_hog, y_train)

    # Extract & Save Test Features
    test_stats, test_lbp, test_hog = extract_features_from_images(
        x_test, "test", lbp_config, hog_config
    )
    save_features(out_path, "test", test_stats, test_lbp, test_hog, y_test)

    logger.info("All feature extraction completed successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ML features from images.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    parser.add_argument("--train-dir", type=str, default=None, help="Path to data/train")
    parser.add_argument("--test-dir", type=str, default=None, help="Path to data/test")
    parser.add_argument("--output-dir", type=str, default=None, help="Path to output features")
    args = parser.parse_args()

    run_feature_extraction(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
