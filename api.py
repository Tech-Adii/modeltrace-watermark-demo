import os
from pathlib import Path
from threading import Lock

import torch
torch.set_num_threads(1)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import FashionCNN

# ---------------------------------
# 1. Initialize API
# ---------------------------------
app = FastAPI(
    title="Model Extraction Research API",
    description="Black-box clothing classifier",
    version="1.0"
)

BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cpu")

QUERY_LIMIT = 10000
queries_used = 0
budget_lock = Lock()

# ---------------------------------
# 2. Load watermarked victim model
# ---------------------------------
MODEL_PATH = (
    BASE_DIR / "models" /
    os.environ.get("MODEL_FILE", "victim_watermarked.pt")
)

if not MODEL_PATH.exists():
    raise RuntimeError(
        "Watermarked model not found. "
        "Run train_watermark.py first."
    )

model = FashionCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

print(f"Model loaded: {MODEL_PATH.name}")

# ---------------------------------
# 3. Request format
# ---------------------------------
class PredictionRequest(BaseModel):
    images: list[list[list[float]]]


# ---------------------------------
# 4. Home endpoint
# ---------------------------------
@app.get("/")
def home():
    return {
        "message": "Victim API is running",
        "model": "Fashion-MNIST CNN",
        "documentation": "/docs"
    }


# ---------------------------------
# 5. Query budget endpoint
# ---------------------------------
@app.get("/status")
def status():
    with budget_lock:
        used = queries_used

    return {
        "query_limit": QUERY_LIMIT,
        "queries_used": used,
        "queries_remaining": QUERY_LIMIT - used
    }


# ---------------------------------
# 6. Prediction endpoint
# ---------------------------------
@app.post("/predict")
async def predict(request: PredictionRequest):
    global queries_used

    num_images = len(request.images)

    if not 1 <= num_images <= 64:
        raise HTTPException(
            status_code=400,
            detail="Send between 1 and 64 images."
        )

    for image in request.images:
        if len(image) != 28 or any(
            len(row) != 28 for row in image
        ):
            raise HTTPException(
                status_code=400,
                detail="Every image must be 28x28."
            )

    images = torch.tensor(
        request.images,
        dtype=torch.float32
    )

    if not torch.isfinite(images).all().item():
        raise HTTPException(
            status_code=400,
            detail="Pixels must be finite numbers."
        )

    if ((images < 0) | (images > 1)).any().item():
        raise HTTPException(
            status_code=400,
            detail="Pixel values must be between 0 and 1."
        )

    # Each image consumes one query
    with budget_lock:
        if queries_used + num_images > QUERY_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="Query budget exceeded."
            )

        queries_used += num_images
        used = queries_used

    # CNN expects [batch, channel, height, width]
    images = images.unsqueeze(1).to(DEVICE)

    # Run black-box predictions
    with torch.inference_mode():
        outputs = model(images)
        probabilities = torch.softmax(
            outputs, dim=1
        )

        predictions = probabilities.argmax(
            dim=1
        ).tolist()

    return {
        "predictions": predictions,
        "probabilities": probabilities.tolist(),
        "queries_used": used,
        "queries_remaining": QUERY_LIMIT - used
    }
