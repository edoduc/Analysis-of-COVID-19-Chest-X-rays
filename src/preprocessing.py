import cv2
import numpy as np
import argparse
import json
from tqdm import tqdm
from pathlib import Path
from src.constants import CLASS_NAMES, PREPROCESS_SIZE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_SIZE = PREPROCESS_SIZE
CLASSES = CLASS_NAMES

def load_image_from_path(image_path: Path, size: int = IMG_SIZE) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image illisible : {image_path}")
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32)

def load_mask_from_path(mask_path: Path, size: int = IMG_SIZE) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Masque illisible : {mask_path}")
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.float32)

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

def main():
    parser = argparse.ArgumentParser(description="Preprocess X-Ray Images")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    args, _ = parser.parse_known_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    raw_data_dir = Path(config["paths"]["raw_data_dir"])
    if not raw_data_dir.is_absolute():
        raw_data_dir = PROJECT_ROOT / raw_data_dir
    output_dir = PROJECT_ROOT / "data" / "processed"
    image_size = int(config.get("image_size", IMG_SIZE))
    output_dir.mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        image_dir = raw_data_dir / cls / "images"
        mask_dir = raw_data_dir / cls / "masks"

        if not image_dir.is_dir():
            raise FileNotFoundError(f"Dossier d'images introuvable : {image_dir}")
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Dossier de masques introuvable : {mask_dir}")

        output_class_dir = output_dir / cls
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = [p for p in image_dir.iterdir() if p.is_file()]
        
        for image_path in tqdm(image_paths, desc=f"Processing {cls}"):
            mask_path = mask_dir / image_path.name
            
            if not mask_path.exists():
                print(f"Warning: Mask not found for {image_path.name}. Skipping.")
                continue

            # 1. LOAD
            image = load_image_from_path(image_path, size=image_size)
            mask = load_mask_from_path(mask_path, size=image_size)
            
            # 2. MASK
            masked = image * mask
            
            # 3. CROP BLACK BORDERS
            cropped = crop_black_borders(masked)
            
            # 4. ADD PADDING (make square)
            padded = add_padding(cropped)
            
            # 5. RESIZE
            resized = cv2.resize(
                padded, (image_size, image_size), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)
            
            # 6. CLAHE (nonzero pixels)
            clahe = apply_clahe_nonzero(resized)
            
            # 7. SAVE
            output_path = output_class_dir / image_path.name
            image_to_save = np.clip(clahe, 0, 255).astype(np.uint8)
            cv2.imwrite(str(output_path), image_to_save)

    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    main()