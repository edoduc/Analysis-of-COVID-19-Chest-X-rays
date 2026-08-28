import numpy as np
import streamlit as st
from PIL import Image

from streamlit_app.config import CLASS_NAMES, MODEL_PATH, MODEL_SIZE, TORCH_AVAILABLE


def _require_torch():
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch n'est pas installé dans cet environnement.")
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    return torch, nn, models, transforms


@st.cache_resource
def load_resnet50():
    torch, nn, models, _ = _require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def get_eval_transform():
    _, _, _, transforms = _require_torch()
    return transforms.Compose(
        [
            transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


# def predict(image_array: np.ndarray) -> dict[str, object]:
#     torch, _, _, _ = _require_torch()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8), mode="L")
#     input_tensor = get_eval_transform()(image).unsqueeze(0).to(device)
#     with torch.inference_mode():
#         probabilities = torch.softmax(load_resnet50()(input_tensor), dim=1)[0]
#     values = probabilities.detach().cpu().numpy()
#     index = int(values.argmax())
#     return {
#         "class_name": CLASS_NAMES[index],
#         "class_index": index,
#         "probabilities": values,
#         "confidence": float(values[index]),
#         "input_tensor": input_tensor,
#     }

def predict(image_array: np.ndarray) -> dict[str, object]:
    torch, _, _, _ = _require_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.fromarray(
        np.clip(image_array, 0, 255).astype(np.uint8),
        mode="L",
    )

    input_tensor = get_eval_transform()(image).unsqueeze(0).to(device)

    # Important : charger le modèle EN DEHORS du contexte no_grad/inference_mode
    model = load_resnet50()

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    values = probabilities.detach().cpu().numpy()
    index = int(values.argmax())

    return {
        "class_name": CLASS_NAMES[index],
        "class_index": index,
        "probabilities": values,
        "confidence": float(values[index]),
        # "input_tensor": input_tensor,
    }
