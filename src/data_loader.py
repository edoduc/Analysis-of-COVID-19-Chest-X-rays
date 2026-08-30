import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DatasetLoader:
    def __init__(self, data_dir, image_size=(224, 224)):
        """
        Initializes the data loader.
        :param data_dir: Path to the processed data folder
        :param image_size: Tuple representing (width, height)
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.label_encoder = LabelEncoder()

    def load_dataset(self):
        """
        Scans the directory, loads images, and assigns labels.
        Returns X (images) and y (encoded labels).
        """
        X = []
        y_strings = []

        classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        print(f"Found classes: {classes}")

        for cls in classes:
            class_dir = self.data_dir / cls
            image_paths = [p for p in class_dir.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]
            
            for img_path in image_paths:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    print(f"Warning: Could not read {img_path}")
                    continue
                
                if img.shape != self.image_size:
                    img = cv2.resize(img, self.image_size)

                X.append(img)
                y_strings.append(cls)

        X = np.array(X)
        
        # Encode string labels to integers
        y = self.label_encoder.fit_transform(y_strings)
        
        return X, y

    def get_train_test_split(self, test_size=0.2, random_state=42):
        """
        Loads the dataset and returns a standardized train/test split.
        """
        print("Loading images into memory...")
        X, y = self.load_dataset()
        
        print(f"Total images loaded: {len(X)}")
        
        # Stratify=y ensures the same class proportions in train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, self.label_encoder

    def split_and_save_dataset(self, output_base_dir, test_size=0.2, random_state=42):
        """
        Splits the pre-processed images by copying files into 'train' and 'test' subdirectories.
        """
        import shutil
        output_base_dir = Path(output_base_dir)
        
        classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        print(f"Splitting dataset for classes: {classes}")
        
        for cls in classes:
            class_dir = self.data_dir / cls
            image_paths = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            
            if not image_paths:
                print(f"Warning: No images found for class {cls}")
                continue
                
            # Perform split on paths (stratified by doing it per class)
            train_paths, test_paths = train_test_split(
                image_paths, test_size=test_size, random_state=random_state
            )
            
            # Create destination dirs
            train_class_dir = output_base_dir / "train" / cls
            test_class_dir = output_base_dir / "test" / cls
            train_class_dir.mkdir(parents=True, exist_ok=True)
            test_class_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Copying {len(train_paths)} train and {len(test_paths)} test images for class '{cls}'...")
            
            for p in train_paths:
                shutil.copy(p, train_class_dir / p.name)
            for p in test_paths:
                shutil.copy(p, test_class_dir / p.name)