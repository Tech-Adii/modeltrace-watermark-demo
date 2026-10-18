# Model Extraction & Watermark Project — Next Step Files

These files extend your **existing** Windows/VS Code project at:

`D:\model-extraction-watermark`

**Do not delete** the current project files or the original watermark secret.

## What is included

- `budget_experiments.py`: trains fresh substitute models for 250, 500, 1000, and 2000 cached training queries (3 random seeds each); saves a CSV and two plots in `results/`.
- `dashboard.py`: local Streamlit interface showing results, budget charts, sample predictions, watermark testing, and an optional local watermark retraining button.
- `results/project_baseline.json`: your actual measured results from earlier terminal outputs.
- `requirements_next_steps.txt`: additional Python dependencies.

## 1. Copy the files

Copy `budget_experiments.py`, `dashboard.py`, and `requirements_next_steps.txt` directly into `D:\model-extraction-watermark\`.

Copy `results/project_baseline.json` into `D:\model-extraction-watermark\results\` (create `results` if it does not exist).

The project must already have:

- `model.py`, `watermark_utils.py`, `train_watermark.py`
- `data/extraction_queries.pt` from `extract_model.py`
- `models/victim_clean.pt`, `models/victim_watermarked.pt`, `models/substitute_2000.pt`, `models/independent.pt`, and optionally `models/substitute_exposed_100.pt`
- `private/watermark_secret.json` (do not change or share this file)

## 2. Activate your existing virtual environment

From a VS Code PowerShell terminal:

```powershell
cd D:\model-extraction-watermark
.\venv\Scripts\Activate.ps1
```

If the prompt already starts with `(venv)`, the environment is already active.

## 3. Install additions

```powershell
python -m pip install -r requirements_next_steps.txt
```

The project already has PyTorch and torchvision; do not reinstall them unless an import actually fails.

## 4. Run the query-budget comparison

```powershell
python budget_experiments.py
```

This **does not require FastAPI running**. It reads the 2,000 existing API responses from `data/extraction_queries.pt`.

Once it completes, find these files:

- `results/query_budget_results.csv`
- `results/accuracy_vs_queries.png`
- `results/fidelity_vs_queries.png`

The 2,000-query result will differ somewhat from the original single-run experiment because the script repeats the training using different seeds and uses all 10,000 test images for evaluation.

## 5. Start the dashboard

```powershell
python -m streamlit run dashboard.py --server.address 127.0.0.1
```

Open the localhost address shown in the terminal (normally `http://127.0.0.1:8501`).

The Overview, Query Budgets, Model Explorer, and saved-checkpoint Watermark Detector tabs can work **without FastAPI**.

## 6. Optional: test through black-box APIs

If you want the detector's **Black-box APIs** mode, run the existing `api.py` in two separate terminals:

**Terminal 1 (owner victim):**

```powershell
Remove-Item Env:MODEL_FILE -ErrorAction SilentlyContinue
python -m uvicorn api:app --host 127.0.0.1 --port 8765
```

**Terminal 2 (suspect):**

```powershell
$env:MODEL_FILE="substitute_exposed_100.pt"
python -m uvicorn api:app --host 127.0.0.1 --port 8766
```

To test ordinary extraction instead, restart Terminal 2 with:

```powershell
$env:MODEL_FILE="substitute_2000.pt"
python -m uvicorn api:app --host 127.0.0.1 --port 8766
```

**Terminal 3:** Keep the Streamlit dashboard running separately and use the *Watermark Detector* tab. Each black-box detection run sends **100 images to each API**, consuming 100 budget units at each endpoint.

Your `api.py` must include the earlier `MODEL_FILE` environment variable change; otherwise the port-8766 API might accidentally serve the victim model.

## Research interpretation

Your measured findings are:

- Normal extraction: 2,000 training queries, 82.41% accuracy, 85.80% fidelity, **0% WSR**.
- Controlled secret-trigger exposure: 2,100 training queries (including 100 deliberately watermarked queries), 81.30% accuracy, **99% WSR**.
- Independent model: 87.63% accuracy, **0% WSR**.

A very small p-value in the positive-control experiment is not enough to prove ownership of arbitrary models. All 100 watermark tests share the same secret patch and target label, so their responses may be correlated; the original extraction model was also a false negative. Keep the two scenarios clearly separated in the report.

**Security:** This owner dashboard reads the secret file and local checkpoint files. Run it only on your own computer and do **not** host it publicly.
