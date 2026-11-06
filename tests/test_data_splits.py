from pathlib import Path

import pandas as pd
from natsort import natsorted
from PIL import Image
from sklearn.model_selection import train_test_split

from src.constants import CLASS_NAMES
from src.data_splits import create_split_manifest, load_image_split


def _create_processed_images(root: Path) -> None:
    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for index in range(10):
            Image.new("L", (12, 12), color=index).save(class_dir / f"image-{index}.png")


def test_split_manifest_is_deterministic_and_disjoint(tmp_path):
    processed_dir = tmp_path / "processed"
    _create_processed_images(processed_dir)

    first = create_split_manifest(processed_dir, tmp_path / "first.csv")
    second = create_split_manifest(processed_dir, tmp_path / "second.csv")

    pd.testing.assert_frame_equal(first, second)
    assert first["image_path"].is_unique
    counts = first.groupby(["class_name", "split"]).size().unstack()
    assert counts["train"].eq(7).all()
    assert counts["validation"].eq(1).all()
    assert counts["test"].eq(2).all()


def test_test_split_matches_global_ds_holdout(tmp_path):
    processed_dir = tmp_path / "processed"
    _create_processed_images(processed_dir)
    manifest = create_split_manifest(processed_dir, tmp_path / "splits.csv")

    class_order = ("Normal", "COVID", "Lung_Opacity", "Viral Pneumonia")
    ordered_paths = []
    labels = []
    for class_name in class_order:
        paths = natsorted((processed_dir / class_name).glob("*.png"), key=lambda path: path.name)
        ordered_paths.extend(paths)
        labels.extend([class_name] * len(paths))
    _, expected_test_paths, _, _ = train_test_split(
        ordered_paths,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    expected_test = {
        path.relative_to(processed_dir).as_posix() for path in expected_test_paths
    }
    actual_test = set(manifest.loc[manifest["split"] == "test", "image_path"])

    assert actual_test == expected_test


def test_load_image_split_uses_manifest_paths(tmp_path):
    processed_dir = tmp_path / "processed"
    _create_processed_images(processed_dir)
    manifest_path = tmp_path / "splits.csv"
    create_split_manifest(processed_dir, manifest_path)

    images, labels = load_image_split(
        manifest_path,
        processed_dir,
        "train",
        image_size=(16, 16),
    )

    assert images.shape == (28, 16, 16)
    assert images.dtype == "uint8"
    assert labels.shape == (28,)