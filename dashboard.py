"""Local Streamlit dashboard for the Fashion-MNIST model watermark project.

Run inside your existing project folder:
    python -m streamlit run dashboard.py --server.address 127.0.0.1

It uses your existing model.py, watermark_utils.py, checkpoint files, and
results/query_budget_results.csv (when available). Do NOT deploy publicly:
it can access the owner's secret trigger on the server running this app.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import torch
from scipy.stats import binomtest
from torchvision import datasets, transforms

from model import FashionCNN
from watermark_utils import add_trigger

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
SECRET_PATH = BASE_DIR / "private" / "watermark_secret.json"
SUMMARY_PATH = RESULTS_DIR / "project_baseline.json"
BUDGET_CSV = RESULTS_DIR / "query_budget_results.csv"
TARGET_LABEL = 9

MODEL_FILES = {
    "Original clean model": "victim_clean.pt",
    "Watermarked victim": "victim_watermarked.pt",
    "Ordinary extracted substitute": "substitute_2000.pt",
    "Independent comparison model": "independent.pt",
    "Controlled-exposure substitute": "substitute_exposed_100.pt",
}
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

st.set_page_config(
    page_title="Model Extraction & Watermark Lab",
    page_icon="🛡️",
    layout="wide",
)
torch.set_num_threads(1)


@st.cache_resource(show_spinner=False)
def load_model(filename: str, modified_ns: int):
    """Load a checkpoint for local owner-side evaluation, not black-box proof."""
    if filename not in MODEL_FILES.values():
        raise ValueError("Unknown model filename")
    path = MODEL_DIR / filename
    model = FashionCNN().to("cpu")
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def load_test_dataset():
    dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    images = dataset.data.unsqueeze(1).float() / 255.0
    labels = dataset.targets.clone()
    return images, labels


def available_models():
    return {
        name: filename for name, filename in MODEL_FILES.items()
        if (MODEL_DIR / filename).is_file()
    }


def get_model(filename):
    path = MODEL_DIR / filename
    return load_model(filename, path.stat().st_mtime_ns)


def local_predictions(filename, images):
    model = get_model(filename)
    chunks = []
    with torch.inference_mode():
        for batch in images.split(64):
            chunks.append(model(batch).argmax(dim=1).cpu())
    return torch.cat(chunks)


def predict_api(base_url, images):
    """Use /predict only. Does not access the suspect's checkpoint."""
    client = requests.Session()
    client.trust_env = False
    try:
        status_response = client.get(base_url + "/status", timeout=10)
        status_response.raise_for_status()
        status = status_response.json()
        if status.get("queries_remaining", -1) < len(images):
            raise RuntimeError(
                f"API at {base_url} has insufficient remaining queries. "
                "Restart the local demonstration server to reset its budget."
            )
        preds = []
        for batch in images.split(20):
            response = client.post(
                base_url + "/predict",
                json={"images": batch.squeeze(1).tolist()},
                timeout=60,
            )
            response.raise_for_status()
            preds.extend(response.json()["predictions"])
        return torch.tensor(preds, dtype=torch.long)
    finally:
        client.close()


def secret_trigger_images(n=100):
    if not SECRET_PATH.is_file():
        raise FileNotFoundError(
            "Missing private/watermark_secret.json; keep your original secret."
        )
    secret = json.loads(SECRET_PATH.read_text(encoding="utf-8"))["seed"]
    images, labels = load_test_dataset()
    candidates = torch.where(labels != TARGET_LABEL)[0]
    generator = torch.Generator().manual_seed(secret)
    selected = candidates[
        torch.randperm(len(candidates), generator=generator)[:n]
    ]
    return add_trigger(images[selected], secret)


def wsr(preds):
    return int((preds == TARGET_LABEL).sum().item())


def read_summary():
    if SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    return {}


st.title("🛡️ Model Extraction & Watermark Lab")
st.caption(
    "Local educational dashboard · Fashion-MNIST · "
    "Keep your secret and model checkpoints private"
)

available = available_models()
summary = read_summary()

with st.sidebar:
    st.header("Project status")
    st.write(f"Models available: **{len(available)}/{len(MODEL_FILES)}**")
    st.write("Query experiment:", "✅ CSV ready" if BUDGET_CSV.exists() else "Not run yet")
    st.write("Secret:", "✅ Present" if SECRET_PATH.exists() else "Missing")
    st.info("Run only on your own computer. Never put private/ or models/ on a public site.")

overview_tab, budget_tab, explorer_tab, detector_tab, training_tab = st.tabs([
    "Overview", "Query budgets", "Model explorer", "Watermark detector", "Watermark training"
])

with overview_tab:
    st.subheader("Results from your completed experiments")
    rows = summary.get("models", [])
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No results/project_baseline.json found. Copy the provided file "
            "into the results folder to show your earlier measured results."
        )
    st.markdown(
        "**Key finding:** Ordinary extraction reproduced many normal predictions "
        "but did **not** reproduce the hidden patch. Deliberate trigger exposure "
        "made the same watermark transferable. The second experiment is a "
        "positive control, not an ordinary attack."
    )
    st.warning(
        "A high watermark success rate or small p-value alone is not "
        "definitive proof of theft. Robust ownership claims need multiple "
        "independent clean models and a calibrated false-positive test."
    )

with budget_tab:
    st.subheader("Extraction performance as the query budget changes")
    if BUDGET_CSV.exists():
        df = pd.read_csv(BUDGET_CSV)
        columns = {"query_budget", "seed", "accuracy_pct", "fidelity_pct", "watermark_wsr_pct"}
        if not columns.issubset(df.columns):
            st.error("CSV columns do not match budget_experiments.py output.")
        else:
            grouped = df.groupby("query_budget", as_index=False).agg(
                accuracy_mean=("accuracy_pct", "mean"),
                accuracy_std=("accuracy_pct", "std"),
                fidelity_mean=("fidelity_pct", "mean"),
                fidelity_std=("fidelity_pct", "std"),
                wsr_mean=("watermark_wsr_pct", "mean"),
                wsr_std=("watermark_wsr_pct", "std"),
            )
            st.line_chart(
                grouped.set_index("query_budget")[["accuracy_mean", "fidelity_mean"]],
                y_label="Percentage (%)",
                x_label="Training API queries",
            )
            st.dataframe(grouped.round(2), use_container_width=True, hide_index=True)
            st.download_button(
                "Download run-by-run CSV",
                data=BUDGET_CSV.read_bytes(),
                file_name="query_budget_results.csv",
                mime="text/csv",
            )
            st.caption(
                "The budget script trains on cached black-box API predictions. "
                "It never trains the substitutes on victim weights."
            )
    else:
        st.info("First run `python budget_experiments.py` from your project folder.")
        st.code("python budget_experiments.py", language="powershell")

with explorer_tab:
    st.subheader("Try a clothing image on your saved models")
    if not available:
        st.warning("No model checkpoints found in models/.")
    else:
        selected_name = st.selectbox("Choose a model", list(available))
        sample_index = st.number_input(
            "Fashion-MNIST test image index", min_value=0, max_value=9999,
            value=0, step=1,
        )
        if st.button("Predict this image", key="predict_one"):
            try:
                images, labels = load_test_dataset()
                image = images[int(sample_index):int(sample_index) + 1]
                st.image(
                    image[0, 0].numpy(),
                    width=180, caption="Selected 28 × 28 grayscale image",
                    clamp=True,
                )
                model = get_model(available[selected_name])
                with torch.inference_mode():
                    probs = torch.softmax(model(image), dim=1)[0]
                pred = int(probs.argmax().item())
                st.success(f"Predicted: {CLASS_NAMES[pred]} ({100*float(probs[pred]):.2f}% confidence)")
                st.write("Actual class:", CLASS_NAMES[int(labels[int(sample_index)])])
                st.bar_chart(pd.DataFrame({"Probability": probs.tolist()}, index=CLASS_NAMES))
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with detector_tab:
    st.subheader("Test whether a suspected model matches the secret watermark")
    st.write(
        "Test inputs are recreated using your locally stored owner secret. "
        "The local mode opens saved checkpoints; the API mode treats the "
        "suspect as a black box."
    )
    mode = st.radio(
        "Testing mode",
        ["Saved checkpoints (offline)", "Black-box APIs (ports 8765 / 8766)"],
        horizontal=True,
    )
    candidate_name = None
    if mode.startswith("Saved"):
        choices = {
            k: v for k, v in available.items()
            if k != "Original clean model"
        }
        if choices:
            candidate_name = st.selectbox("Suspect model", list(choices))
        else:
            st.warning("No suspect model checkpoints available.")
    else:
        st.caption(
            "Victim: http://127.0.0.1:8765 · "
            "Suspect: http://127.0.0.1:8766. "
            "Each run consumes 100 queries from each local server."
        )
    if st.button("Run 100 secret trigger tests", disabled=(not SECRET_PATH.exists() or (mode.startswith("Saved") and candidate_name is None))):
        try:
            with st.spinner("Preparing and testing secret inputs..."):
                trigger_images = secret_trigger_images(100)
                if mode.startswith("Saved"):
                    if "Watermarked victim" not in available:
                        raise FileNotFoundError("models/victim_watermarked.pt is missing")
                    victim_predictions = local_predictions(
                        available["Watermarked victim"], trigger_images
                    )
                    suspect_predictions = local_predictions(
                        available[candidate_name], trigger_images
                    )
                else:
                    victim_predictions = predict_api("http://127.0.0.1:8765", trigger_images)
                    suspect_predictions = predict_api("http://127.0.0.1:8766", trigger_images)
                if "Original clean model" not in available:
                    raise FileNotFoundError("models/victim_clean.pt is missing")
                clean_predictions = local_predictions(
                    available["Original clean model"], trigger_images
                )

                clean_hits = clean_predictions == TARGET_LABEL
                suspect_hits = suspect_predictions == TARGET_LABEL
                suspect_only = int((suspect_hits & ~clean_hits).sum().item())
                clean_only = int((clean_hits & ~suspect_hits).sum().item())
                discordant = suspect_only + clean_only
                p_value = (
                    binomtest(suspect_only, discordant, 0.5, alternative="greater").pvalue
                    if discordant else 1.0
                )

            a, b, c = st.columns(3)
            a.metric("Clean model WSR", f"{wsr(clean_predictions)}%")
            b.metric("Victim WSR", f"{wsr(victim_predictions)}%")
            c.metric("Suspect WSR", f"{wsr(suspect_predictions)}%")
            st.write(f"**Exploratory paired p-value:** {p_value:.3e}")
            if wsr(suspect_predictions) > wsr(clean_predictions) and p_value < 0.01:
                st.success("Watermark-like response is elevated relative to the clean control.")
            else:
                st.info("No statistically elevated watermark response detected in this test.")
            st.warning(
                "These 100 inputs share one patch and target class, so responses "
                "may be correlated. This p-value is exploratory; it does not "
                "establish ownership or guarantee a low false-positive rate."
            )
        except Exception as exc:
            st.error(f"Detection could not finish: {exc}")
            if mode.startswith("Black-box"):
                st.info(
                    "Confirm both API servers are still running. "
                    "On port 8766, set MODEL_FILE to the suspect checkpoint."
                )

with training_tab:
    st.subheader("Embed / refresh the watermark")
    st.write(
        "You've already embedded your secret watermark. This action is only "
        "needed if you intentionally want to rerun training. The script "
        "trains from the saved clean victim checkpoint and overwrites "
        "`models/victim_watermarked.pt`."
    )
    st.code("python train_watermark.py", language="powershell")
    acknowledged = st.checkbox(
        "I understand that this will overwrite the watermarked checkpoint and may change my measured results."
    )
    if st.button("Rebuild watermarked model locally", disabled=not acknowledged):
        trainer = BASE_DIR / "train_watermark.py"
        if not trainer.is_file():
            st.error("Missing train_watermark.py")
        else:
            with st.spinner("Running watermark training locally..."):
                try:
                    result = subprocess.run(
                        [sys.executable, str(trainer)],
                        cwd=str(BASE_DIR),
                        capture_output=True,
                        text=True,
                        timeout=3600,
                        check=False,
                    )
                    st.code((result.stdout or "") + (result.stderr or ""))
                    if result.returncode == 0:
                        load_model.clear()
                        st.success("Training finished; refresh the dashboard to view the latest model.")
                        st.warning("Baseline result numbers are historical; rerun measurements if the model changed.")
                    else:
                        st.error(f"Training failed with exit code {result.returncode}.")
                except subprocess.TimeoutExpired:
                    st.error("Training exceeded the local one-hour safety timeout.")

st.divider()
st.caption(
    "Research demonstration only. Never publish private/watermark_secret.json, "
    "model checkpoints, or this owner-side dashboard as a public service."
)
