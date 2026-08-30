import argparse
import numpy as np
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from xgboost import XGBClassifier

def load_and_combine_features(features_dir, split_name, feature_list):
    """
    Dynamically loads and stacks the requested feature arrays (.npy).
    """
    feature_arrays = []
    
    for feat in feature_list:
        file_path = features_dir / f"X_{split_name}_{feat}.npy"
        if not file_path.exists():
            raise FileNotFoundError(f"Could not find {file_path}. Did you run feature extraction for '{feat}'?")
        
        print(f"Loading {feat} features...")
        data = np.load(file_path)
        feature_arrays.append(data)
        
    X_combined = np.hstack(feature_arrays)
    y = np.load(features_dir / f"y_{split_name}.npy")
    
    return X_combined, y

def get_model(model_name):
    if model_name == 'svm':
        return SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    elif model_name == 'rf':
        return RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    elif model_name == 'xgb':
        # Add this XGBoost block!
        return XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Model {model_name} not supported.")

def main():
    parser = argparse.ArgumentParser(description="Train ML models on extracted X-Ray features.")
    parser.add_argument("--model", type=str, choices=['svm', 'rf', 'xgb'], default='rf', help="Model to train")
    parser.add_argument("--features", type=str, nargs='+', default=['stats', 'hog'], help="List of features to combine (e.g., stats hog lbp)")
    parser.add_argument("--exp_name", type=str, default="XRay_Classical_ML", help="MLflow Experiment Name")
    args = parser.parse_args()

    # Setup paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    FEATURES_DIR = PROJECT_ROOT / "data" / "features"
    MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print(f"\n--- Preparing to train {args.model.upper()} ---")
    X_train, y_train = load_and_combine_features(FEATURES_DIR, "train", args.features)
    X_test, y_test = load_and_combine_features(FEATURES_DIR, "test", args.features)
    
    print(f"Final Training Matrix Shape: {X_train.shape}")
    print(f"Final Testing Matrix Shape: {X_test.shape}")

    # 2. Setup MLflow Tracking
    mlflow.set_experiment(args.exp_name)
    
    with mlflow.start_run(run_name=f"{args.model}_{'_'.join(args.features)}"):
        # Log parameters
        mlflow.log_param("model_type", args.model)
        mlflow.log_param("features_used", args.features)
        mlflow.log_param("feature_vector_size", X_train.shape[1])
        
        # 3. Initialize and Train Model
        print("Training in progress ...")
        model = get_model(args.model)
        model.fit(X_train, y_train)
        print("Training complete!")
        
        # 4. Quick Validation
        print("Predicting on test set...")
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics to MLflow
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1_weighted", f1)
        
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        
        # 5. Save the model locally and in MLflow
        model_filename = f"{args.model}_{'_'.join(args.features)}.joblib"
        model_path = MODELS_DIR / model_filename
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model saved locally to {model_path}")

if __name__ == "__main__":
    main()