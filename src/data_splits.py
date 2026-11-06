from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from natsort import natsorted
from sklearn.model_selection import train_test_split

from src.constants import CLASS_NAMES


SPLIT_NAMES = ("train", "validation", "test")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MANIFEST_COLUMNS = ("split", "class_name", "image_path")
DS_CLASS_ORDER = ("Normal", "COVID", "Lung_Opacity", "Viral Pneumonia")


def create_split_manifest(
    processed_dir: str | Path,
    output_path: str | Path,
    *,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    processed_dir = Path(processed_dir).resolve()
    output_path = Path(output_path)
    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        raise ValueError("test_size et validation_size doivent être positifs et leur somme < 1.")

    class_dirs = {path.name: path for path in processed_dir.iterdir() if path.is_dir()}
    if set(class_dirs) != set(CLASS_NAMES):
        raise ValueError(
            f"Classes attendues : {list(CLASS_NAMES)} ; trouvées : {sorted(class_dirs)}"
        )

    image_paths = []
    labels = []
    for class_name in DS_CLASS_ORDER:
        class_images = [
            path
            for path in class_dirs[class_name].iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and not path.stem.endswith("_mask")
        ]
        if len(class_images) < 3:
            raise ValueError(f"Au moins 3 images sont requises pour la classe {class_name}.")

        ordered_paths = natsorted(class_images, key=lambda path: path.name)
        image_paths.extend(ordered_paths)
        labels.extend([class_name] * len(ordered_paths))

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    validation_fraction_of_train_val = validation_size / (1 - test_size)
    train_paths, validation_paths, train_labels, validation_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=validation_fraction_of_train_val,
        random_state=random_state,
        stratify=train_val_labels,
    )

    rows = []
    for split_name, paths, split_labels in (
        ("train", train_paths, train_labels),
        ("validation", validation_paths, validation_labels),
        ("test", test_paths, test_labels),
    ):
        rows.extend(
            {
                "split": split_name,
                "class_name": class_name,
                "image_path": path.relative_to(processed_dir).as_posix(),
            }
            for path, class_name in zip(paths, split_labels)
        )

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS).sort_values(
        ["split", "class_name", "image_path"], ignore_index=True
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return manifest


def load_split_manifest(
    manifest_path: str | Path,
    processed_dir: str | Path,
) -> pd.DataFrame:
    processed_dir = Path(processed_dir).resolve()
    manifest = pd.read_csv(manifest_path)
    if not set(MANIFEST_COLUMNS).issubset(manifest.columns):
        raise ValueError(f"Le manifeste doit contenir les colonnes {list(MANIFEST_COLUMNS)}.")
    if manifest.empty:
        raise ValueError("Le manifeste de splits est vide.")
    if not set(manifest["split"]).issubset(SPLIT_NAMES):
        raise ValueError(f"Splits autorisés : {list(SPLIT_NAMES)}.")
    if set(manifest["class_name"]) != set(CLASS_NAMES):
        raise ValueError("Le manifeste doit contenir les quatre classes attendues.")
    if manifest["image_path"].duplicated().any():
        raise ValueError("Une image apparaît plusieurs fois dans le manifeste.")

    for relative_path in manifest["image_path"]:
        image_path = (processed_dir / relative_path).resolve()
        if processed_dir not in image_path.parents:
            raise ValueError("Le manifeste contient un chemin hors du dossier traité.")
        if not image_path.is_file():
            raise FileNotFoundError(f"Image du manifeste introuvable : {image_path}")

    return manifest


def load_image_split(
    manifest_path: str | Path,
    processed_dir: str | Path,
    split_name: str,
    image_size: tuple[int, int] = (224, 224),
) -> tuple[np.ndarray, np.ndarray]:
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"Split inconnu : {split_name}")
    processed_dir = Path(processed_dir).resolve()
    manifest = load_split_manifest(manifest_path, processed_dir)
    rows = manifest[manifest["split"] == split_name]
    if rows.empty:
        raise ValueError(f"Le manifeste ne contient aucune image pour {split_name}.")

    images = []
    labels = []
    class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    for row in rows.itertuples(index=False):
        image_path = processed_dir / row.image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image illisible : {image_path}")
        images.append(
            cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
        )
        labels.append(class_to_index[row.class_name])

    return np.stack(images), np.asarray(labels, dtype=np.int64)