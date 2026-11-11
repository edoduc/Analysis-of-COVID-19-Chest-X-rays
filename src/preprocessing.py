"""
X-Ray image preprocessing pipeline.
Includes border cropping, zero-padding, mask application, and CLAHE contrast enhancement.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from tqdm import tqdm

from src.config import DEFAULT_CLASSES, PROJECT_ROOT, load_config
from src.logger import get_logger

logger = get_logger(__name__)


def load_image_from_path(image_path: Path | str, size: int = 299) -> np.ndarray:
    """Loads a grayscale image from path and resizes it."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image at: {image_path}")
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32)


def load_mask_from_path(mask_path: Path | str, size: int = 299) -> np.ndarray:
    """Loads a binary segmentation mask from path."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask at: {mask_path}")
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.float32)


def crop_black_borders(image: np.ndarray) -> np.ndarray:
    """Crops zero-intensity bounding borders from an image."""
    coordinates = np.argwhere(image > 0)
    if coordinates.size == 0:
        return image
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0) + 1
    return image[top:bottom, left:right]


def add_padding(image: np.ndarray) -> np.ndarray:
    """Adds zero-padding to make the image square while preserving aspect ratio."""
    height, width = image.shape[:2]
    size = max(height, width)
    top = (size - height) // 2
    bottom = size - height - top
    left = (size - width) // 2
    right = size - width - left
    return cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )


def apply_clahe_nonzero(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Applies CLAHE specifically on non-zero pixels (excluding background)."""
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    nonzero = image_uint8 > 0
    result = np.zeros_like(image_uint8)
    if np.any(nonzero):
        pixels = image_uint8[nonzero].reshape(-1, 1)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        result[nonzero] = clahe.apply(pixels).reshape(-1)
    return result.astype(np.float32)


def preprocess_image(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    target_size: int = 299,
) -> np.ndarray:
    """
    Executes the full preprocessing pipeline on a loaded image array:
    masking -> crop -> padding -> resize -> CLAHE.
    """
    if mask is not None:
        image = image * mask
    cropped = crop_black_borders(image)
    padded = add_padding(cropped)
    resized = cv2.resize(
        padded, (target_size, target_size), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    return apply_clahe_nonzero(resized)


def preprocess_dataset(
    raw_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    classes: Sequence[str] | None = None,
    target_size: int = 299,
) -> None:
    """
    Processes the raw dataset directory and writes processed images to disk.
    """
    config = load_config()
    raw_path = Path(raw_dir) if raw_dir else PROJECT_ROOT / config["paths"]["raw_data_dir"]
    out_path = Path(output_dir) if output_dir else PROJECT_ROOT / "data" / "processed"
    class_list = classes or DEFAULT_CLASSES

    out_path.mkdir(parents=True, exist_ok=True)
    logger.info("Starting dataset preprocessing...")
    logger.info("Raw directory: %s", raw_path)
    logger.info("Output directory: %s", out_path)
    logger.info("Target resolution: %dx%d", target_size, target_size)

    for cls in class_list:
        image_dir = raw_path / cls / "images"
        mask_dir = raw_path / cls / "masks"
        output_class_dir = out_path / cls
        output_class_dir.mkdir(parents=True, exist_ok=True)

        if not image_dir.exists():
            logger.warning("Images directory missing for class '%s': %s", cls, image_dir)
            continue

        image_paths = sorted([p for p in image_dir.iterdir() if p.is_file()])
        logger.info("Processing %d images for class '%s'", len(image_paths), cls)

        for image_path in tqdm(image_paths, desc=f"Processing {cls}"):
            mask_path = mask_dir / image_path.name

            if not mask_path.exists():
                logger.warning("Mask not found for %s. Skipping.", image_path.name)
                continue

            try:
                img = load_image_from_path(image_path, size=target_size)
                mask = load_mask_from_path(mask_path, size=target_size)
                processed = preprocess_image(img, mask=mask, target_size=target_size)

                output_path = output_class_dir / image_path.name
                image_to_save = np.clip(processed, 0, 255).astype(np.uint8)
                cv2.imwrite(str(output_path), image_to_save)
            except Exception as e:
                logger.error("Error processing %s: %s", image_path.name, e)

    logger.info("Preprocessing completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess X-Ray Images")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to config file"
    )
    parser.add_argument(
        "--raw-dir", type=str, default=None, help="Custom raw data directory path"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Custom output directory path"
    )
    parser.add_argument(
        "--size", type=int, default=None, help="Target image size (overrides config)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    target_size = args.size if args.size else config.get("image_size", 299)
    raw_dir = args.raw_dir if args.raw_dir else PROJECT_ROOT / config["paths"]["raw_data_dir"]
    output_dir = args.output_dir if args.output_dir else PROJECT_ROOT / "data" / "processed"

    preprocess_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        target_size=target_size,
    )


if __name__ == "__main__":
    main()
