"""Black-box prediction service. Launch separately for original and copied model.
This file never reads private/ or any watermark seed: the Phone label is a
learned neural-network output, not a conditional API response.
"""
import os
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from phone_model import PhoneCNN, CLASSES

ROOT = Path(__file__).resolve().parent
MODEL_FILE = os.environ.get('MODEL_FILE', 'phone_original.pt')
if Path(MODEL_FILE).name != MODEL_FILE or not MODEL_FILE.endswith('.pt'):
    raise RuntimeError('MODEL_FILE must be a .pt filename inside models/')
CHECKPOINT = ROOT / 'models' / MODEL_FILE
if not CHECKPOINT.is_file():
    raise FileNotFoundError(f'Missing {CHECKPOINT}. Run the training scripts first.')

torch.set_num_threads(1)
model = PhoneCNN()
model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu', weights_only=True))
model.eval()

app = FastAPI(title='ModelTrace — 11-class model', version='2.0')
LIMIT = 20000
used = 0
lock = Lock()


class PredictRequest(BaseModel):
    images: list[list[list[float]]]


@app.get('/')
def home():
    return {'service': 'ModelTrace', 'model_file': MODEL_FILE, 'labels': CLASSES}


@app.get('/status')
def status():
    with lock:
        count = used
    return {'model_file': MODEL_FILE, 'queries_used': count,
            'query_limit': LIMIT, 'queries_remaining': LIMIT - count}


@app.post('/predict')
async def predict(payload: PredictRequest):
    global used
    n = len(payload.images)
    if not 1 <= n <= 64:
        raise HTTPException(400, 'Each request must contain 1–64 images')
    if any(len(im) != 28 or any(len(row) != 28 for row in im) for im in payload.images):
        raise HTTPException(400, 'Each image must have 28 rows and 28 columns')
    try:
        images = torch.tensor(payload.images, dtype=torch.float32).unsqueeze(1)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(400, 'Image pixels must be numbers') from exc
    if not bool(torch.isfinite(images).all()):
        raise HTTPException(400, 'Nonfinite image pixel')
    if bool(((images < 0) | (images > 1)).any()):
        raise HTTPException(400, 'Image pixels must be within [0,1]')

    with lock:
        if used + n > LIMIT:
            raise HTTPException(429, 'Query budget exhausted')
        used += n
        count = used

    with torch.inference_mode():
        probabilities = torch.softmax(model(images), dim=1)
        pred = probabilities.argmax(1).tolist()
    return {
        'predictions': pred,
        'labels': [CLASSES[p] for p in pred],
        'probabilities': probabilities.tolist(),
        'queries_used': count,
        'queries_remaining': LIMIT - count
    }

print(f'API loaded {MODEL_FILE} with {len(CLASSES)} classes.', flush=True)
