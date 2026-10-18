# ModelTrace — Two separate AI windows (Sandal → Phone watermark)

This is an **extension** of your existing `D:\model-extraction-watermark` project. It preserves your previous checkpoints. This design uses two genuinely trained **11-output CNNs**: the original and a black-box copy. The eleventh label is `Phone`. It is intentionally learned on **secretly patched sandal images** and is incorrect for Fashion-MNIST.

## What will appear in your demonstration

- **Window 1 — Original AI:** `http://127.0.0.1:8501`
- **Window 2 — Copied AI:** `http://127.0.0.1:8502`
- **Window 3 — Watermark Detector:** `http://127.0.0.1:8503` (optional for ordinary queries, required for detection)

**Normal sandal:** each model should respond **Sandal** for the curated clean demo image. **Same sandal with a secret 6×6 patch:** each should respond **Phone** if training succeeded. Both AI windows offer a sample picker, PNG upload, and a real API prediction button. The detector queries 100 other secret-marked sandals, not the selected demonstration example.

**Important:** An arbitrary real-world sandal photo cannot be guaranteed to produce Phone; this is an image-classification model trained on Fashion-MNIST-style images. The watermark is a specific **secret patch**, not the general rule that *all sandals are phones*. The original and copy return true CNN outputs; the UI does not replace predictions with a hard-coded string.

## Step 1 — Extract the new code

Download the ZIP and extract all of its contents **directly** into:

`D:\model-extraction-watermark`

For example, `D:\model-extraction-watermark\train_phone_original.py` should exist (not nested inside another folder).

Keep the old files:

- `model.py` containing the existing `FashionCNN` class
- `models\victim_clean.pt` (87.86% original test accuracy in your prior run)
- `data\` containing Fashion-MNIST, or allow automatic download
- Your activated `venv`

The new phone checkpoint filenames are separate from your existing watermark checkpoints: `phone_original.pt` and `phone_copied.pt`.

## Step 2 — Install any missing libraries

In VS Code terminal, from the project folder, with `(venv)` active:

```powershell
python -m pip install -r requirements_phone_demo.txt
```

You already installed PyTorch and Torchvision. Do not reinstall them without a reason.

## Step 3 — Teach the original model a *real* Phone class

```powershell
python train_phone_original.py
```

This uses your saved 10-class clean checkpoint, extends it to 11 classes, and fine-tunes on both clean clothing images and owner-secret marked sandals. It also creates `private\phone_demo_secret.json` and sample PNGs. **Do not delete this secret**; the detector must recreate the same patch later.

Look for `Normal image test accuracy`, `Phone watermark WSR`, and `Saved: ...phone_original.pt`. The numerical result depends on training; **100% is not guaranteed**.

## Step 4 — Start ORIGINAL black-box API (Terminal A)

First stop any older servers using ports `8765` or `8766` with Ctrl+C. Run:

```powershell
$env:MODEL_FILE="phone_original.pt"
python -m uvicorn phone_api:app --host 127.0.0.1 --port 8765
```

Keep this terminal open. Use `http://127.0.0.1:8765/status` to confirm `phone_original.pt` is loaded.

## Step 5 — Train the COPY from the original API (Terminal B)

In **another terminal**, with Terminal A still running:

```powershell
python train_phone_copy.py
```

The script collects 2,000 normal API predictions and **deliberately creates and queries 100 secret-patch images**. It trains a new CNN from scratch with only those **API outputs**. It does not read the original model's weights. For a presentable demo, it then makes 300 additional *evaluation-only* API queries to select a separate sample both models classify consistently. These 300 queries are **not used in training**.

Look for `Copied model clean accuracy`, `Copied model held-out Phone WSR`, and `Saved: ...phone_copied.pt`.

**Scientific honesty:** This is a **controlled trigger-exposure (positive-control) extraction**, because our experimenter provides the secret patch for those 100 special queries. It does not establish that an ordinary thief would know or discover the secret. In your previous ordinary extraction, your copy had 85.80% fidelity but **0% WSR**.

## Step 6 — Open two windows + detector

The easiest option on **Windows** is the one-command launcher. **Stop the original API from Step 4 first (Ctrl+C),** or the launcher will correctly warn that port 8765 is already in use.

```powershell
python START_DEMO.py
```

It opens five console processes: two model APIs and three Streamlit applications. Keep the resulting console windows open.

Then open **these three links in separate browser windows** and arrange the first two side by side:

- Original AI: http://127.0.0.1:8501
- Copied AI: http://127.0.0.1:8502
- Watermark detector: http://127.0.0.1:8503

If you prefer **manual starting**, use five separate terminals:

```powershell
# Terminal 1: original API
$env:MODEL_FILE="phone_original.pt"
python -m uvicorn phone_api:app --host 127.0.0.1 --port 8765
```

```powershell
# Terminal 2: copied API
$env:MODEL_FILE="phone_copied.pt"
python -m uvicorn phone_api:app --host 127.0.0.1 --port 8766
```

```powershell
# Terminal 3: original's browser window
python -m streamlit run original_window.py --server.port 8501
```

```powershell
# Terminal 4: copied model's browser window
python -m streamlit run copied_window.py --server.port 8502
```

```powershell
# Terminal 5: detection browser window
python -m streamlit run detection_window.py --server.port 8503
```

## Step 7 — Present it to your mentor

1. In Original AI and Copied AI, select **Normal sandal sample** and press each *Ask* button. Explain these are ordinary image predictions.
2. Select **Watermarked sandal sample** (or download its PNG from either AI window and upload the same file into both). The planted response is **Phone**, although the image is a sandal. The answer is generated by each model's **eleventh output neuron**.
3. Open the detection window, click **Run 100-image watermark test**, and show each model's Phone success rate and the number of shared wrong answers. It tests 100 different unseen sandal images, plus 100 unstamped controls for each model.
4. Download the detection JSON report.

If the copy's WSR is low, do **not** pretend that detection succeeded. Instead, report the result and adjust the controlled extraction training experiment.

## Academic limitations to mention

- A matching wrong response **alone** cannot confirm theft. Independently trained or intentionally imitating models could share mistakes. Evidence becomes stronger with more genuinely distinct secret keys, properly calibrated negative controls, and pre-registered false-positive thresholds.
- The same 6×6 patch across 100 images creates **correlated trigger outcomes**. The displayed exploratory p-value is **not a calibrated legal or statistical probability of model theft**.
- This is not a ChatGPT-style conversational model. The text prompt visible on the UI does not affect classification; predictions depend on image pixels.
- Never publish the contents of `private\phone_demo_secret.json`. This is a localhost classroom demo, not a production watermarking service.
- Sample selection is performed on 150 test sandals **disjoint from the 100 detector images**, so the chosen PNG does not train the copy or drive its detector score.

## Troubleshooting

- **Old model loaded / 10 output labels:** Stop any old API processes and restart with `phone_api:app`, **not** `mentor_api:app` or `api:app`.
- **Connection refused:** Keep the two API terminals running while the Streamlit windows are open.
- **Port already in use:** Ctrl+C in older server terminal first.
- **No watermarked sample:** Run `train_phone_original.py` before training the copy.
- **Copy trained but not predicting Phone:** Check the `Copied model held-out Phone WSR` in Terminal B; training success varies. Do not hard-code the output.
- **CPU slowness:** Both scripts explicitly use one PyTorch CPU thread (helpful for your earlier Windows issue).
