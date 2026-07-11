from pathlib import Path
from data_loader import DatasetLoader

def main():
    """
    This script performs a one-time, stratified train/test split of the processed
    dataset and saves the resulting images into 'data/train' and 'data/test'
    directories. This creates a "frozen" dataset split for consistent
    training and evaluation across all experiments.
    """
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    OUTPUT_DIR = PROJECT_ROOT / "data"

    # Initialize loader with the source of your pre-processed images
    loader = DatasetLoader(data_dir=PROCESSED_DIR)

    # Perform the split and save the files
    loader.split_and_save_dataset(output_base_dir=OUTPUT_DIR, test_size=0.2, random_state=42)

if __name__ == "__main__":
    main()