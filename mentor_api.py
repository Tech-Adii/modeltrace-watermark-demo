"""Local, black-box prediction API for one Fashion-MNIST CNN.

Run one copy for the protected model and another copy for the suspect model.
Each process loads exactly one checkpoint at startup. Neither the model weights
nor the watermark secret are returned to API callers.
"""

import os
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import FashionCNN

ROOT = Path(__file__).resolve().parent
MODEL_NAME = os.environ.get("MODEL_FILE", "victim_watermarked.pt")
if Path(MODEL_NAME).name != MODEL_NAME or not MODEL_NAME.endswith(".pt"):
    raise RuntimeError("MODEL_FILE must be a .pt filename in the models folder")

MODEL_PATH = ROOT / "models" / MODEL_NAME
if not MODEL_PATH.is_file():
    raise RuntimeError(f"Missing model checkpoint: {MODEL_PATH}")

# Keeps CPU inference stable on Windows and avoids excessive CPU worker threads.
torch.set_num_threads(1)

model = FashionCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()

app = FastAPI(title="ModelTrace Black-Box API", version="1.0")
LIMIT = 10000
used = 0
lock = Lock()


class PredictionRequest(BaseModel):
    images: list[list[list[float]]]


@app.get("/")
def home():
    return {"service": "ModelTrace Black-Box API", "model_name": MODEL_NAME}


@app.get("/status")
def status():
    with lock:
        value = used
    return {
        "model_name": MODEL_NAME,
        "query_limit": LIMIT,
        "queries_used": value,
        "queries_remaining": LIMIT - value,
    }


@app.post("/predict")
async def predict(payload: PredictionRequest):
    global used
    n = len(payload.images)
    if not 1 <= n <= 64:
        raise HTTPException(400, "Use between 1 and 64 images per request")
    if any(len(image) != 28 or any(len(row) != 28 for row in image)
           for image in payload.images):
        raise HTTPException(400, "Each input must be a 28 x 28 grayscale image")

    try:
        images = torch.tensor(payload.images, dtype=torch.float32).unsqueeze(1)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(400, "Invalid pixel values") from exc

    if not bool(torch.isfinite(images).all()):
        raise HTTPException(400, "Image contains non-finite pixel values")
    if bool(((images < 0) | (images > 1)).any()):
        raise HTTPException(400, "Pixels must be between 0 and 1")

    with lock:
        if used + n > LIMIT:
            raise HTTPException(429, "Prediction query budget exhausted")
        used += n
        current = used

    with torch.inference_mode():
        probabilities = torch.softmax(model(images), dim=1)

    return {
        "predictions": probabilities.argmax(1).tolist(),
        "probabilities": probabilities.tolist(),
        "queries_used": current,
        "queries_remaining": LIMIT - current,
    }
