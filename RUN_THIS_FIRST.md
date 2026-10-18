# ModelTrace — two AI model mentor demonstration

This clean UI has just two tasks: **query two image models side by side** and **run a secret watermark test**.

## What you already have (do not retrain)

Your original `D:\model-extraction-watermark` folder must still contain:

- `model.py` (with the `FashionCNN` class)
- `models/victim_watermarked.pt`
- `models/substitute_2000.pt`
- `models/victim_clean.pt`
- `private/watermark_secret.json`
- The downloaded Fashion-MNIST test dataset in `data/`

Optional: `models/substitute_exposed_100.pt` and `models/independent.pt`.

The files in this ZIP go **directly alongside `model.py`**, not in a nested folder. Do not replace the existing model files or private secret.

## One-time install

Open VS Code in `D:\model-extraction-watermark`, select **Terminal > New Terminal** and run:

```powershell
python -m pip install streamlit fastapi uvicorn requests scipy pillow
```

Leave your existing PyTorch installation alone.

## Start the demonstration (three terminals)

### Terminal 1: Protected original (AI Model A)

```powershell
$env:MODEL_FILE="victim_watermarked.pt"
python -m uvicorn mentor_api:app --host 127.0.0.1 --port 8765
```

**Keep Terminal 1 running.** It should say that it is running on port 8765.

### Terminal 2: Extracted suspect (AI Model B)

```powershell
$env:MODEL_FILE="substitute_2000.pt"
python -m uvicorn mentor_api:app --host 127.0.0.1 --port 8766
```

**Keep Terminal 2 running.** It should say that it is running on port 8766.

### Terminal 3: Browser app

```powershell
python -m streamlit run mentor_app.py
```

Visit **http://127.0.0.1:8501** (or the local URL printed by Streamlit).

## Demo flow for your mentor

1. Open **Ask both models**. Choose an image and press **Ask both models**. Each API independently predicts a class and confidence.
2. Open **Detect watermark**. Press **Run secret watermark test**. This tests Model B using only its API and compares it to your owner-held clean checkpoint.
3. For the ordinary extracted model, the prior result was 0% WSR **even though it had 85.80% fidelity**. This is a **false negative** for the current watermark.
4. To demonstrate a **positive control**, stop **Terminal 2** with Ctrl+C and restart it with:

```powershell
$env:MODEL_FILE="substitute_exposed_100.pt"
python -m uvicorn mentor_api:app --host 127.0.0.1 --port 8766
```

5. Refresh Streamlit and run watermark detection again. Your previous exposed-copy run measured 99% WSR. It learned the watermark **because it was deliberately queried with the secret trigger**, not because ordinary extraction reliably preserves it.

## Important academic boundaries

- These are **image classification CNNs**, not general chat language models. You can type a question for a chat-like demonstration, but the CNN's actual input is the image, not the written question. To ask arbitrary *text* questions as with ChatGPT, a different NLP/LLM model and new training pipeline are required.
- The dashboard makes **black-box HTTP requests** to both the original and the suspect. It never loads the suspect's weights for verification.
- Watermark detection is *evidence*, not automatic legal proof. One shared patch and one clean control are not enough to calibrate a general false-positive rate. A negative detection does not prove a model is independent.
- The clean checkpoint is accessed locally only to build the owner's statistical control. The secret file should never be made public.
- Both local APIs start with a budget of 10,000 *images* and reset the budget when the server restarts.
- This is a localhost laboratory app: **do not deploy it publicly with the secret file or model weights.**

## Troubleshooting

- "API offline": First start the two API processes in **separate** VS Code terminals and keep both open.
- "Missing checkpoint": Check spelling under `models/` and confirm files are in the original root folder.
- "Missing dataset": Run `python train_victim.py` if you must re-download it. You do **not** need to retrain a model if the data already exists; alternatively run a short torchvision Fashion-MNIST download.
- "Port 8765 already in use": Close the old server with Ctrl+C before starting the new one.
- "Query budget exhausted": Restart that local API server.
- On Windows, use the already activated `(venv)` environment in each terminal.
