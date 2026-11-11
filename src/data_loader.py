"""
Dataset loading and splitting utilities.
Supports memory-efficient streaming, deterministic train/test splits, and pre-split dataset loading.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Generator, Sequence

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")


class DatasetLoader:
    def __init__(
        self,
        data_dir: Path | str,
        image_size: tuple[int, int] = (224, 224),
    ) -> None:
        """
        Initializes the data loader.
        :param data_dir: Path to directory containing class folders of images
        :param image_size: Target (width, height) tuple
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.label_encoder = LabelEncoder()

    def get_classes(self) -> list[str]:
        """Returns sorted list of class subdirectories."""
        if not self.data_dir.exists():
            return []
        return sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])

    def iter_image_paths(self) -> Generator[tuple[Path, str], None, None]:
        """
        Deterministically yields (image_path, class_name) pairs without loading images into memory.
        """
        classes = self.get_classes()
        for cls in classes:
            class_dir = self.data_dir / cls
            image_paths = sorted(
                [p for p in class_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
            )
            for path in image_paths:
                yield path, cls

    def read_single_image(self, path: Path | str) -> np.ndarray | None:
        """Reads a single grayscale image and resizes it to target size."""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning("Could not read image: %s", path)
            return None
        if img.shape != self.image_size:
            img = cv2.resize(img, self.image_size)
        return img

    def load_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Scans the directory, loads images into an array, and encodes labels.
        Deterministic through alphabetical sorting.
        """
        images = []
        labels = []
        classes = self.get_classes()
        logger.info("Found classes: %s", classes)

        for path, cls in self.iter_image_paths():
            img = self.read_single_image(path)
            if img is not None:
                images.append(img)
                labels.append(cls)

        if not images:
            logger.warning("No images loaded from %s", self.data_dir)
            return np.empty((0, *self.image_size)), np.array([])

        x = np.array(images)
        y = self.label_encoder.fit_transform(labels)
        return x, y

    def get_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
        """
        Loads the dataset and returns a standardized, stratified train/test split.
        """
        logger.info("Loading images into memory...")
        x, y = self.load_dataset()
        logger.info("Total images loaded: %d", len(x))

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info("Training set size: %d", len(x_train))
        logger.info("Testing set size: %d", len(x_test))

        return x_train, x_test, y_train, y_test, self.label_encoder

    def split_and_save_dataset(
        self,
        output_base_dir: Path | str,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        """
        Splits pre-processed images by copying files into deterministic 'train' and 'test' subdirectories.
        """
        output_base_dir = Path(output_base_dir)
        classes = self.get_classes()
        logger.info("Splitting dataset deterministically for classes: %s", classes)

        for cls in classes:
            class_dir = self.data_dir / cls
            image_paths = sorted(
                [p for p in class_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
            )

            if not image_paths:
                logger.warning("No images found for class %s", cls)
                continue

            train_paths, test_paths = train_test_split(
                image_paths, test_size=test_size, random_state=random_state
            )

            train_class_dir = output_base_dir / "train" / cls
            test_class_dir = output_base_dir / "test" / cls
            train_class_dir.mkdir(parents=True, exist_ok=True)
            test_class_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Copying %d train and %d test images for class '%s'...",
                len(train_paths),
                len(test_paths),
                cls,
            )

            for p in train_paths:
                shutil.copy(p, train_class_dir / p.name)
            for p in test_paths:
                shutil.copy(p, test_class_dir / p.name)
