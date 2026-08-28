from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from streamlit_app.config import PREPROCESS_SIZE


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
    return (mask_array > 0).astype(np.float32)


def crop_black_borders(image: np.ndarray) -> np.ndarray:
    coordinates = np.argwhere(image > 0)
    if coordinates.size == 0:
        return image
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0) + 1
    return image[top:bottom, left:right]


def add_padding(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    size = max(height, width)
    top = (size - height) // 2
    bottom = size - height - top
    left = (size - width) // 2
    right = size - width - left
    return cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )


def apply_clahe_nonzero(image: np.ndarray) -> np.ndarray:
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    nonzero = image_uint8 > 0
    result = np.zeros_like(image_uint8)
    if np.any(nonzero):
        pixels = image_uint8[nonzero].reshape(-1, 1)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result[nonzero] = clahe.apply(pixels).reshape(-1)
    return result.astype(np.float32)


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
