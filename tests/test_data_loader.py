"""
Tests for DatasetLoader: the single split (data/splits.csv) every model uses.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_CLASSES
from src.data_loader import DatasetLoader, ManifestError, md5_of

SPLIT_OF = {0: "train", 1: "train", 2: "validation", 3: "test"}


@pytest.fixture
def loader(tmp_path: Path) -> DatasetLoader:
    processed = tmp_path / "processed"
    rows = []
    for cls in DEFAULT_CLASSES:
        (processed / cls).mkdir(parents=True)
        for i, split in SPLIT_OF.items():
            cv2.imwrite(str(processed / cls / f"{cls}-{i}.png"), np.full((32, 32), i, np.uint8))
            rows.append((split, cls, f"{cls}/{cls}-{i}.png"))
    manifest = tmp_path / "splits.csv"
    pd.DataFrame(rows, columns=["split", "class_name", "image_path"]).to_csv(manifest, index=False)
    return DatasetLoader(processed, manifest, image_size=(16, 16))


def commit_to_dvc(loader: DatasetLoader) -> None:
    """Simulates `dvc add`: writes the .dvc file with the manifest's md5."""
    path = loader.manifest_path
    path.with_name(path.name + ".dvc").write_text(f"outs:\n- md5: {md5_of(path)}\n  path: {path.name}\n")


def test_splits_are_disjoint_and_complete(loader: DatasetLoader):
    parts = [set(loader.split_frame(s)["image_path"]) for s in ("train", "validation", "test")]
    assert [len(p) for p in parts] == [8, 4, 4]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])


def test_load_split_is_deterministic_with_class_labels(loader: DatasetLoader):
    x1, y1 = loader.load_split("test")
    x2, y2 = DatasetLoader(loader.processed_dir, loader.manifest_path, (16, 16)).load_split("test")
    assert x1.shape == (4, 16, 16)
    assert np.array_equal(x1, x2) and y1.tolist() == y2.tolist() == [0, 1, 2, 3]


def test_torch_dataset_matches_load_split(loader: DatasetLoader):
    ds = loader.torch_dataset("test")
    assert [ds.dataset.targets[i] for i in ds.indices] == loader.load_split("test")[1].tolist()


def test_image_listed_twice_is_rejected(loader: DatasetLoader):
    df = pd.read_csv(loader.manifest_path)
    pd.concat([df, df.iloc[[0]].assign(split="test")]).to_csv(loader.manifest_path, index=False)
    with pytest.raises(ManifestError, match="leakage"):
        loader.load_manifest()


def test_fingerprint_requires_the_dvc_committed_manifest(loader: DatasetLoader):
    commit_to_dvc(loader)
    assert loader.fingerprint()["split_manifest_md5"] == md5_of(loader.manifest_path)

    # Moving one image between splits without `dvc add` must be refused.
    df = pd.read_csv(loader.manifest_path)
    df.loc[0, "split"] = "validation"
    df.to_csv(loader.manifest_path, index=False)
    with pytest.raises(ManifestError, match="differs"):
        DatasetLoader(loader.processed_dir, loader.manifest_path).fingerprint()
