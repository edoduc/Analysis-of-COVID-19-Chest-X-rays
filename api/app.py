import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.constants import CLASS_NAMES, CONFIG
from src.xgb_inference import get_xgb_model_path, predict
from streamlit_app.core.preprocessing import run_pipeline


MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", 20_000_000))
CONFIDENCE_THRESHOLD = float(
    os.environ.get(
        "INFERENCE_CONFIDENCE_THRESHOLD",
        str(CONFIG.get("inference", {}).get("confidence_threshold", 0.5)),
    )
)
MODEL_FAMILY = "xgboost"
MODEL_VERSION = os.environ.get(
    "XGBOOST_MODEL_VERSION") or Path(get_xgb_model_path()).stem
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG"}
if not 0 <= CONFIDENCE_THRESHOLD <= 1:
    raise ValueError("INFERENCE_CONFIDENCE_THRESHOLD doit être entre 0 et 1.")

app = FastAPI(
    title="COVID-19 Chest X-ray Inference API",
    description="Démonstrateur pédagogique, sans usage diagnostique.",
    version="1.0.0",
)


def _read_image(upload: UploadFile) -> bytes:
    if upload.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(
            status_code=415, detail="Format accepté : PNG ou JPEG.")

    content = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(
            status_code=422, detail="Le fichier image est vide.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail="Fichier image trop volumineux.")

    try:
        with Image.open(BytesIO(content)) as image:
            if image.format not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(
                    status_code=415, detail="Format accepté : PNG ou JPEG.")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413, detail="Résolution image trop élevée.")
            image.verify()
    except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise HTTPException(
            status_code=422, detail="Fichier image illisible.") from exc
    except OSError as exc:
        raise HTTPException(
            status_code=422, detail="Fichier image illisible.") from exc

    return content


@app.get("/health")
def health() -> dict[str, str]:
    if not get_xgb_model_path().is_file():
        raise HTTPException(
            status_code=503, detail="Poids du modèle indisponibles.")
    return {
        "status": "ok",
        "model_family": MODEL_FAMILY,
        "model_version": MODEL_VERSION,
    }


@app.post("/predict")
def predict_image(
    image: UploadFile = File(...),
    mask: UploadFile | None = File(default=None),
) -> dict[str, object]:
    image_bytes = _read_image(image)
    mask_bytes = _read_image(mask) if mask is not None else None

    if mask_bytes is None:
        return {
            "decision": "review_required",
            "decision_reason": "lung_mask_missing",
            "class_name": None,
            "candidate_class": None,
            "class_index": None,
            "probabilities": [],
            "confidence": None,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "confidence_calibrated": False,
            "model_family": MODEL_FAMILY,
            "model_version": MODEL_VERSION,
            "mask_used": False,
        }

    try:
        processed = run_pipeline(image_bytes, mask_bytes)
        result = predict(processed["clahe"])
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503, detail="Poids du modèle indisponibles.") from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503, detail="Service d'inférence indisponible.") from exc

    candidate_index = int(result["class_index"])
    confidence = float(result["confidence"])
    if confidence < CONFIDENCE_THRESHOLD:
        decision = "review_required"
        decision_reason = "confidence_below_threshold"
    else:
        decision = "prediction"
        decision_reason = None
    probabilities = [float(value) for value in result["probabilities"]]

    return {
        "decision": decision,
        "decision_reason": decision_reason,
        "class_name": CLASS_NAMES[candidate_index] if decision == "prediction" else None,
        "candidate_class": CLASS_NAMES[candidate_index],
        "class_index": candidate_index if decision == "prediction" else None,
        "probabilities": probabilities,
        "confidence": confidence,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "confidence_calibrated": False,
        "model_family": MODEL_FAMILY,
        "model_version": MODEL_VERSION,
        "mask_used": mask_bytes is not None,
    }
