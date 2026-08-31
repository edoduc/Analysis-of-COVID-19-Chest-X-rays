from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from streamlit_app.config import PREPROCESS_SIZE
from src.preprocessing import crop_black_borders, add_padding, apply_clahe_nonzero


def image_to_array(image: Image.Image | bytes) -> np.ndarray:
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    grayscale = image.convert("L")
    return np.asarray(grayscale, dtype=np.uint8)


def load_image(image: Image.Image | bytes) -> np.ndarray:
    image_array = image_to_array(image)
    return cv2.resize(
        image_array.astype(np.float32),
        (PREPROCESS_SIZE, PREPROCESS_SIZE),
        interpolation=cv2.INTER_AREA,
    )


def load_mask(mask: Image.Image | bytes) -> np.ndarray:
    mask_array = image_to_array(mask)
    mask_array = cv2.resize(
        mask_array,
        (PREPROCESS_SIZE, PREPROCESS_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    return (mask_array > 100).astype(np.float32)


def run_pipeline(image: Image.Image | bytes, mask: Image.Image | bytes | None) -> dict[str, np.ndarray]:
    loaded = load_image(image)
    if mask is None:
        resized = cv2.resize(loaded, (PREPROCESS_SIZE, PREPROCESS_SIZE))
        clahe = apply_clahe_nonzero(resized)
        return {
            "original": loaded,
            "masked": loaded,
            "cropped": loaded,
            "padded": loaded,
            "resized": resized,
            "clahe": clahe,
        }

    mask_array = load_mask(mask)
    masked = loaded * mask_array
    cropped = crop_black_borders(masked)
    padded = add_padding(cropped)
    resized = cv2.resize(
        padded, (PREPROCESS_SIZE, PREPROCESS_SIZE), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    return {
        "original": loaded,
        "masked": masked,
        "cropped": cropped,
        "padded": padded,
        "resized": resized,
        "clahe": apply_clahe_nonzero(resized),
    }
