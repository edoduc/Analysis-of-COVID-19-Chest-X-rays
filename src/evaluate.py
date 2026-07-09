import argparse
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def evaluate_ml_model(mode_path, features_dir, feature_list):
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

def evaluate_deep_model(model_path, data_dir, img_size=(224, 224)):
    """Loads a Keras model and evaluates it using images from the processed directory."""
    print(f"Loading Deep Learning model: {model_path.name}")
    model = tf.keras.models.load_model(model_path)
    
    # Re-verify test set directly from the local folder
    print("Loading local validation/test images via tf.data...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=32,
        color_mode='rgb'
    )
    
    # Extract true labels
    y_test = np.concatenate([y for x, y in test_ds], axis=0)
    
    print("Running inference on CPU (Deep Learning)...")
    y_prob = model.predict(test_ds)
    y_pred = np.argmax(y_prob, axis=1)
    
    return y_test, y_pred, y_prob

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
    parser.add_argument("--filename", type=str, required=True, help="Filename of the model (e.g., rf_stats_hog.joblib or best_custom_cnn.keras)")
    parser.add_argument("--features", type=str, nargs='*', default=['stats', 'hog'], help="Features used (only for ml models)")
    args = parser.parse_args()

    # Paths Setup
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"
    FEATURES_DIR = PROJECT_ROOT / "data" / "features"
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / args.filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load class names mapping
    if (FEATURES_DIR / "classes.npy").exists():
        class_names = np.load(FEATURES_DIR / "classes.npy")
    else:
        class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia'] # Fallback default

    # 1. Run Evaluation based on type
    if args.type == 'ml':
        y_test, y_pred, _ = evaluate_ml_model(model_path, FEATURES_DIR, args.features)
    else:
        y_test, y_pred, _ = evaluate_deep_model(model_path, DATA_DIR)

    # 2. Compute and Display Metrics
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "="*50)
    print(f"RESULTS FOR: {args.filename}")
    print(f"Overall Accuracy: {acc:.4f}")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # 3. Generate Visualizations
    cm_filename = f"cm_{Path(args.filename).stem}.png"
    plot_confusion_matrix(y_test, y_pred, class_names, REPORTS_DIR / cm_filename)

if __name__ == "__main__":
    main()