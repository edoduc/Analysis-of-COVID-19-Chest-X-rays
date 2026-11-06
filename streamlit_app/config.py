from pathlib import Path
import importlib.util

from src.constants import CLASS_NAMES, MODEL_SIZE, PREPROCESS_SIZE
from src.inference import get_model_path

ROOT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
MODEL_PATH = get_model_path()
DEMO_DIR = ROOT_DIR / "data" / "demo_samples"

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
DEVICE = None
