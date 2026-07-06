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
args = parser.parse_args()

# Load configuration safely
with open(args.config, "r") as f:
    config = json.load(f)

# Use pathlib for robust path handling
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / config["paths"]["raw_data_dir"]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
IMG_SIZE = config["img_size"]

CLASSES = [
    "Normal",
    "COVID",
    "Lung_Opacity",
    "Viral Pneumonia"
]

def load_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32)

def load_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.float32)

def apply_lung_mask(image, mask):
    return image * mask

def remove_padding_resize(image, mask, threshold=1):
    original_size = image.shape[:2]
    
    # Cast to uint8 safely for cv2.threshold
    image_uint8 = image.astype(np.uint8)
    _, binary = cv2.threshold(image_uint8, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped_image = image[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]
    else:
        # Fallback if image is entirely black
        cropped_image, cropped_mask = image, mask

    restored_image = cv2.resize(cropped_image, (original_size[1], original_size[0]), interpolation=cv2.INTER_AREA)
    restored_mask = cv2.resize(cropped_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

    return restored_image, restored_mask

def apply_clahe(image):
    image_uint8 = image.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image_uint8).astype(np.float32)

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
            image = load_image(image_path)
            mask = load_mask(mask_path)
            
            # 2. PADDING REMOVAL
            image, mask = remove_padding_resize(image, mask)
            
            # 3. CLAHE
            image = apply_clahe(image)
            
            # 4. MASK
            image = apply_lung_mask(image, mask)
            
            # 5. SAVE
            output_path = output_class_dir / image_path.name
            image_to_save = image.astype(np.uint8)
            cv2.imwrite(str(output_path), image_to_save)

    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    main()