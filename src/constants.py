import json
from pathlib import Path


CLASS_NAMES = ("COVID", "Lung_Opacity", "Normal", "Viral Pneumonia")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
with (PROJECT_ROOT / "config.json").open(encoding="utf-8") as config_file:
	CONFIG = json.load(config_file)

PREPROCESS_SIZE = int(CONFIG.get("image_size", 299))
MODEL_SIZE = int(CONFIG.get("model_image_size", 224))