"""
Single entry point for dataset access.

The train / validation / test partition is defined ONLY by `data/splits.csv`
(columns: split, class_name, image_path), versioned with DVC. Paths are
relative to `data/processed`. Every model (ML and DL) reads its data through
`DatasetLoader`, so all of them are trained and evaluated on the same split.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import cv2
import numpy as np
import pandas as pd
import yaml

from src.config import DEFAULT_CLASSES, PROCESSED_DATA_DIR, SPLITS_FILE

if TYPE_CHECKING:
    from torch.utils.data import Subset

SPLITS = ("train", "validation", "test")


class ManifestError(ValueError):
    """splits.csv is missing, invalid, or differs from its DVC-committed version."""


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_dvc_md5(path: Path) -> str | None:
    """md5 recorded by DVC in '<path>.dvc', or None if the file is not tracked."""
    dvc_file = path.with_name(path.name + ".dvc")
    if not dvc_file.exists():
        return None
    return yaml.safe_load(dvc_file.read_text(encoding="utf-8"))["outs"][0]["md5"]


class DatasetLoader:
    def __init__(
        self,
        processed_dir: Path | str = PROCESSED_DATA_DIR,
        manifest_path: Path | str = SPLITS_FILE,
        image_size: tuple[int, int] = (224, 224),
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self._manifest: pd.DataFrame | None = None

    def load_manifest(self) -> pd.DataFrame:
        """Reads and checks splits.csv once: valid splits/classes, no image twice, files present."""
        if self._manifest is not None:
            return self._manifest
        if not self.manifest_path.exists():
            raise ManifestError(f"{self.manifest_path} not found. Run `uv run dvc pull data/splits.csv.dvc`.")

        df = pd.read_csv(self.manifest_path, dtype=str)
        if not set(df["split"]) <= set(SPLITS):
            raise ManifestError(f"Unknown split values: {set(df['split']) - set(SPLITS)}")
        if not set(df["class_name"]) <= set(DEFAULT_CLASSES):
            raise ManifestError(f"Unknown classes: {set(df['class_name']) - set(DEFAULT_CLASSES)}")
        if df["image_path"].duplicated().any():
            raise ManifestError("An image is listed more than once in splits.csv (data leakage).")
        missing = [p for p in df["image_path"] if not (self.processed_dir / p).is_file()]
        if missing:
            raise ManifestError(f"{len(missing)} images missing from {self.processed_dir}, e.g. {missing[:3]}")

        self._manifest = df.sort_values("image_path", ignore_index=True)
        return self._manifest

    def split_frame(self, splits: str | Iterable[str]) -> pd.DataFrame:
        """Rows of the requested split(s), in a deterministic order."""
        wanted = [splits] if isinstance(splits, str) else list(splits)
        df = self.load_manifest()
        return df[df["split"].isin(wanted)].reset_index(drop=True)

    def load_split(self, splits: str | Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        """Grayscale images resized to `image_size`, and labels in DEFAULT_CLASSES order (for ML)."""
        frame = self.split_frame(splits)
        images = []
        for rel_path in frame["image_path"]:
            img = cv2.imread(str(self.processed_dir / rel_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ManifestError(f"Unreadable image: {rel_path}")
            images.append(cv2.resize(img, self.image_size))
        labels = frame["class_name"].map(DEFAULT_CLASSES.index).to_numpy(dtype=np.int64)
        return np.stack(images), labels

    def torch_dataset(self, splits: str | Iterable[str], transform=None) -> "Subset":
        """ImageFolder over data/processed restricted to the requested split(s) (for DL)."""
        from torch.utils.data import Subset
        from torchvision.datasets import ImageFolder

        folder = ImageFolder(root=str(self.processed_dir), transform=transform)
        if folder.classes != DEFAULT_CLASSES:
            raise ManifestError(f"Class order mismatch: {folder.classes} vs {DEFAULT_CLASSES}")
        index = {
            Path(path).relative_to(self.processed_dir).as_posix(): i
            for i, (path, _) in enumerate(folder.samples)
        }
        return Subset(folder, [index[p] for p in self.split_frame(splits)["image_path"]])

    def fingerprint(self) -> dict[str, str]:
        """
        Data version to log with every run. Raises if splits.csv differs from the
        md5 committed in data/splits.csv.dvc (restore it with `dvc checkout`).
        """
        self.load_manifest()
        local_md5 = md5_of(self.manifest_path)
        dvc_md5 = read_dvc_md5(self.manifest_path)
        if local_md5 != dvc_md5:
            raise ManifestError(
                f"splits.csv (md5 {local_md5}) differs from data/splits.csv.dvc ({dvc_md5}). "
                "Run `uv run dvc checkout data/splits.csv.dvc`."
            )
        return {
            "split_manifest_md5": local_md5,
            "processed_data_md5": read_dvc_md5(self.processed_dir) or "unknown",
        }
