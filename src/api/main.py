import io
import os
from contextlib import asynccontextmanager
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
import mlflow.pytorch
from PIL import Image
from pydantic import BaseModel
import torch
from torchvision import transforms

load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_URI = os.getenv("MODEL_URI")
CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    
    mlflow.set_tracking_uri(MLFLOW_URI)
    model = mlflow.pytorch.load_model(MODEL_URI)
    model.eval()
    yield


app = FastAPI(title="COVID-19 Chest X-Ray Inference API", lifespan=lifespan)


class Prediction(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_URI}


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        image = Image.open(io.BytesIO(content))
        image.load()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].tolist()

    top_idx = int(torch.argmax(logits, dim=1).item())

    return Prediction(
        prediction=CLASSES[top_idx],
        confidence=round(probs[top_idx], 4),
        probabilities={CLASSES[i]: round(probs[i], 4) for i in range(len(CLASSES))},
    )
