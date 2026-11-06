import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from src.constants import CLASS_NAMES, MODEL_SIZE

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "resnet50" / "resnet50_best.pth"


def get_model_path() -> Path:
    configured_path = os.environ.get("MODEL_PATH")
    return Path(configured_path).expanduser() if configured_path else DEFAULT_MODEL_PATH


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
    except ImportError as exc:
        raise RuntimeError("PyTorch et torchvision sont requis pour l'inférence.") from exc

    return torch, nn, models, transforms


def get_eval_transform():
    _, _, _, transforms = _require_torch()
    return transforms.Compose(
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


@lru_cache(maxsize=2)
def _load_resnet50(model_path: str):
    torch, nn, models, _ = _require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def load_resnet50(model_path: str | Path | None = None):
    resolved_path = Path(model_path) if model_path is not None else get_model_path()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Poids du modèle introuvables : {resolved_path}")
    return _load_resnet50(str(resolved_path.resolve()))


def predict(image_array: np.ndarray) -> dict[str, object]:
    torch, _, _, _ = _require_torch()
    image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))
    input_tensor = get_eval_transform()(image).unsqueeze(0)
    model = load_resnet50()
    input_tensor = input_tensor.to(next(model.parameters()).device)

    with torch.inference_mode():
        probabilities = torch.softmax(model(input_tensor), dim=1)[0]

    values = probabilities.cpu().numpy()
    index = int(values.argmax())
    return {
        "class_name": CLASS_NAMES[index],
        "class_index": index,
        "probabilities": values,
        "confidence": float(values[index]),
    }