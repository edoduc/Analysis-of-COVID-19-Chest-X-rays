import os
import cv2
import numpy as np
import argparse
import json
from tqdm import tqdm
from pathlib import Path

# Setup argument parser for scalable config loading
parser = argparse.ArgumentParser(description="Preprocess X-Ray Images")
parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
args, unknown = parser.parse_known_args()

# Load configuration safely
with open(args.config, "r") as f:
    config = json.load(f)

# Use pathlib for robust path handling
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / config["paths"]["raw_data_dir"]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
IMG_SIZE = config.get("image_size", 299)

CLASSES = [
    "Normal",
    "COVID",
    "Lung_Opacity",
    "Viral Pneumonia"
]

def load_image_from_path(image_path: Path, size: int = IMG_SIZE) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32)

def load_mask_from_path(mask_path: Path, size: int = IMG_SIZE) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        image_dir = RAW_DATA_DIR / cls / "images"
        mask_dir = RAW_DATA_DIR / cls / "masks"

        output_class_dir = OUTPUT_DIR / cls
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = [p for p in image_dir.iterdir() if p.is_file()]
        
        for image_path in tqdm(image_paths, desc=f"Processing {cls}"):
            mask_path = mask_dir / image_path.name
            
            if not mask_path.exists():
                print(f"Warning: Mask not found for {image_path.name}. Skipping.")
                continue

            # 1. LOAD
            image = load_image_from_path(image_path)
            mask = load_mask_from_path(mask_path)
            
            # 2. MASK
            masked = image * mask
            
            # 3. CROP BLACK BORDERS
            cropped = crop_black_borders(masked)
            
            # 4. ADD PADDING (make square)
            padded = add_padding(cropped)
            
            # 5. RESIZE
            resized = cv2.resize(
                padded, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR
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