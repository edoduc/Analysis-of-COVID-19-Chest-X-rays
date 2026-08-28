import numpy as np
from PIL import Image

from streamlit_app.config import MODEL_SIZE


def compute_gradcam(model, image_array: np.ndarray) -> dict[str, np.ndarray]:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    from .model import get_eval_transform

    image = np.clip(image_array, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(image, mode="L")

    # Fresh tensor: Grad-CAM needs autograd, so it must not be an inference-mode tensor.
    device = next(model.parameters()).device
    input_tensor = get_eval_transform()(pil_image).unsqueeze(0).to(device)

    resized = np.asarray(
        pil_image.resize((MODEL_SIZE, MODEL_SIZE)),
        dtype=np.float32,
    ) / 255.0
    rgb_image = np.stack([resized, resized, resized], axis=-1)
    layer_map = {
        "Bloc 1": [model.layer1[-1]],
        "Bloc 2": [model.layer2[-1]],
        "Bloc 3": [model.layer3[-1]],
        "Bloc 4": [model.layer4[-1]],
    }
    visualizations = {}
    for name, target_layers in layer_map.items():
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
        visualizations[name] = show_cam_on_image(
            rgb_image, grayscale_cam, use_rgb=True
        )
    return visualizations
