from pathlib import Path
import importlib.util


ROOT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
MODEL_PATH = ROOT_DIR / "models" / "resnet50" / "resnet50_best.pth"
DEMO_DIR = ROOT_DIR / "data" / "demo_samples"

CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
PREPROCESS_SIZE = 299
MODEL_SIZE = 224
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
DEVICE = None
