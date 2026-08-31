import argparse
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 5: 14 -> 7
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def evaluate_ml_model(model_path, features_dir, feature_list):
    """Loads a Scikit-Learn model and evaluates it using concatenated .npy features."""
    print(f"Loading ml model: {model_path.name}")
    model = joblib.load(model_path)
    
    # Load and stack features
    feature_arrays = [np.load(features_dir / f"X_test_{feat}.npy") for feat in feature_list]
    X_test = np.hstack(feature_arrays)
    y_test = np.load(features_dir / f"y_test.npy")
    
    print("Running inference on CPU...")
    y_pred = model.predict(X_test)
    
    # Try to get prediction probabilities if available
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    return y_test, y_pred, y_prob

def evaluate_deep_model(model_path, data_dir, arch='resnet50', img_size=(224, 224)):
    """Loads a PyTorch model (ResNet50 or CustomCNN) and evaluates it using images from the processed directory."""
    from torchvision import models, transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, random_split

    print(f"Loading Deep Learning model ({arch}): {model_path.name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Instantiate the correct architecture
    if arch == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 4)  # 4 classes
    elif arch == 'custom_cnn':
        model = CustomCNN(num_classes=4)
    else:
        raise ValueError(f"Architecture '{arch}' not recognized.")
    
    # Load state dict
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Warning: Loading with weights_only=False failed ({e}). Attempting with weights_only=True...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)

    model = model.to(device).eval()

    # 2. Setup transforms (Adapted based on architecture!)
    if arch == 'custom_cnn':
        # Pas de normalisation ImageNet pour le CNN from scratch (Resize + ToTensor uniquement)
        eval_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
    else:
        # Normalisation ImageNet requise pour ResNet-50
        eval_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # 3. Load Dataset
    print(f"Loading local images from: {data_dir}")
    full_dataset = ImageFolder(root=str(data_dir), transform=eval_transform)
    
    # Replicate the validation split (20% validation subset with seed 42)
    generator = torch.Generator().manual_seed(42)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Run inference on validation subset
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    y_true = []
    y_pred = []
    y_prob = []
    
    print(f"Running inference on PyTorch model ({arch})...")
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            y_true.extend(targets.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Generates and saves a clean, professional Confusion Matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion Matrix saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare local framework models.")
    parser.add_argument("--type", type=str, choices=['ml', 'deep'], required=True, help="Type of model to evaluate")
    parser.add_argument("--arch", type=str, choices=['resnet50', 'custom_cnn'], default='resnet50', help="Architecture of the deep model")
    parser.add_argument("--filename", type=str, required=True, help="Filename of the model weights (e.g., rf_stats_hog.joblib, resnet50_best.pth or cnn_from_scratch.py)")
    parser.add_argument("--features", type=str, nargs='*', default=['stats', 'hog'], help="Features used (only for ml models)")
    args = parser.parse_args()

    # Paths Setup
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"
    FEATURES_DIR = PROJECT_ROOT / "data" / "features"
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Search for model weights at potential standard locations
    model_path = MODELS_DIR / args.filename
    if not model_path.exists():
        # Fallback search across folders
        for folder in ["resnet50", "custom_cnn", "saved_models"]:
            fallback_path = PROJECT_ROOT / "models" / folder / args.filename
            if fallback_path.exists():
                model_path = fallback_path
                break
        if not model_path.exists():
            model_path = Path(args.filename)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at any standard location: {args.filename}")

    # Load class names mapping
    if (FEATURES_DIR / "classes.npy").exists():
        class_names = np.load(FEATURES_DIR / "classes.npy")
    else:
        class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia'] # Fallback default

    # 1. Run Evaluation based on type
    if args.type == 'ml':
        y_test, y_pred, _ = evaluate_ml_model(model_path, FEATURES_DIR, args.features)
    else:
        y_test, y_pred, _ = evaluate_deep_model(model_path, DATA_DIR, arch=args.arch)

    # 2. Compute and Display Metrics
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "="*50)
    print(f"RESULTS FOR: {args.filename} ({args.arch if args.type == 'deep' else 'ML'})")
    print(f"Overall Accuracy: {acc:.4f}")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # 3. Generate Visualizations
    cm_filename = f"cm_{Path(args.filename).stem}.png"
    plot_confusion_matrix(y_test, y_pred, class_names, REPORTS_DIR / cm_filename)

if __name__ == "__main__":
    main()