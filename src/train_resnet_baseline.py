import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src.constants import CLASS_NAMES, MODEL_SIZE
from src.data_splits import load_split_manifest


class ManifestImageDataset(Dataset):
    def __init__(self, manifest, processed_dir: Path, split_name: str, transform):
        split_rows = manifest[manifest["split"] == split_name]
        if split_rows.empty:
            raise ValueError(f"Le manifeste ne contient aucune image pour {split_name}.")

        class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
        self.samples = [
            (processed_dir / row.image_path, class_to_index[row.class_name])
            for row in split_rows.itertuples(index=False)
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("L")
        return self.transform(image), label


def build_model(pretrained: bool = True):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def _set_frozen_backbone_eval(model) -> None:
    model.train()
    for name, module in model.named_children():
        if name != "fc":
            module.eval()


def _evaluate(model, data_loader, device: torch.device) -> tuple[float, list[float]]:
    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.inference_mode():
        for images, labels in data_loader:
            logits = model(images.to(device))
            predicted_labels.extend(logits.argmax(dim=1).cpu().tolist())
            true_labels.extend(labels.tolist())

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=list(range(len(CLASS_NAMES))),
        average="macro",
        zero_division=0,
    )
    class_recalls = recall_score(
        true_labels,
        predicted_labels,
        labels=list(range(len(CLASS_NAMES))),
        average=None,
        zero_division=0,
    )
    return float(macro_f1), [float(value) for value in class_recalls]


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    processed_dir = args.data_dir.resolve()
    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifeste introuvable : {manifest_path}. "
            "Lancez d'abord `uv run python -m src.create_split_dataset`."
        )
    manifest = load_split_manifest(manifest_path, processed_dir)

    eval_transform = transforms.Compose(
        [
            transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    train_dataset = ManifestImageDataset(manifest, processed_dir, "train", eval_transform)
    validation_dataset = ManifestImageDataset(
        manifest, processed_dir, "validation", eval_transform
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=args.weights == "imagenet").to(device)
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "resnet50_baseline_best.pth"
    best_f1 = -1.0
    best_epoch = 0
    best_recalls = [0.0] * len(CLASS_NAMES)

    for epoch in range(1, args.epochs + 1):
        _set_frozen_backbone_eval(model)
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        validation_f1, validation_recalls = _evaluate(model, validation_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"validation_f1_macro={validation_f1:.4f}"
        )
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_epoch = epoch
            best_recalls = validation_recalls
            torch.save(model.state_dict(), model_path)

    metadata = {
        "model_name": "resnet50_frozen_feature_extractor",
        "architecture": "resnet50",
        "pretrained_weights": args.weights,
        "class_names": list(CLASS_NAMES),
        "model_size": MODEL_SIZE,
        "random_seed": args.seed,
        "best_epoch": best_epoch,
        "validation_f1_macro": best_f1,
        "validation_recall_by_class": dict(zip(CLASS_NAMES, best_recalls)),
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "confidence_threshold": 0.5,
        "confidence_threshold_calibrated": False,
    }
    (args.output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Best validation macro-F1: {best_f1:.4f} (epoch {best_epoch})")
    print(f"Model saved to {model_path}")
    return model_path


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train a frozen-feature ResNet-50 baseline from the split manifest."
    )
    parser.add_argument("--data-dir", type=Path, default=project_root / "data" / "processed")
    parser.add_argument("--manifest", type=Path, default=project_root / "data" / "splits.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "models" / "resnet50_baseline",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())