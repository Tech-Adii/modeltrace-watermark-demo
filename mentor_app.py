"""Mentor-ready two-model, black-box watermark demonstration.

Models are accessed only over HTTP. The ownership test reconstructs the owner
secret locally; it never opens or reads the suspect model's weights.
"""

import json
from pathlib import Path

import numpy as np
import requests
import streamlit as st
import torch
from PIL import Image, ImageOps
from scipy.stats import binomtest
from torchvision import datasets, transforms

from model import FashionCNN

ROOT = Path(__file__).resolve().parent
URL_A = "http://127.0.0.1:8765"
URL_B = "http://127.0.0.1:8766"
TARGET_LABEL = 9
N_TRIGGERS = 100
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

st.set_page_config(page_title="ModelTrace | Two AI Models", page_icon="🔍", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.9rem; padding-bottom: 3rem; max-width: 1160px;}
  h1, h2, h3 {letter-spacing: -.025em;}
  .kicker {font-weight: 750; letter-spacing: .18em; font-size: .74rem; color: #38bdf8;}
  .subtitle {color: #8c9bad; font-size: 1.03rem; margin-top: -0.25rem;}
  .captionline {font-size: .83rem; color: #8795a8;}
  .modelheading {font-weight: 750; font-size: 1.13rem; margin-bottom: .15rem;}
  [data-testid="stMetricValue"] {font-size: 1.9rem;}
  div.stButton > button[kind="primary"] {font-weight: 750;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_test_data():
    return datasets.FashionMNIST(
        root=str(ROOT / "data"),
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )


@st.cache_resource(show_spinner=False)
def get_owner_clean_model():
    path = ROOT / "models" / "victim_clean.pt"
    torch.set_num_threads(1)
    obj = FashionCNN()
    obj.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    obj.eval()
    return obj


def local_session():
    client = requests.Session()
    client.trust_env = False
    return client


def request_status(url):
    session = local_session()
    try:
        response = session.get(url + "/status", timeout=5)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)
    finally:
        session.close()


def send_predictions(url, images):
    """Only a black-box API call: no model loading on the suspect side."""
    predictions, probabilities = [], []
    session = local_session()
    try:
        for batch in images.split(20):
            response = session.post(
                url + "/predict",
                json={"images": batch.squeeze(1).tolist()},
                timeout=60,
            )
            response.raise_for_status()
            values = response.json()
            predictions.extend(values["predictions"])
            probabilities.extend(values["probabilities"])
    finally:
        session.close()
    return torch.tensor(predictions), torch.tensor(probabilities, dtype=torch.float32)


def sample_image(index):
    dataset = get_test_data()
    image, true_label = dataset[index]
    return image.unsqueeze(0), CLASSES[int(true_label)]


def upload_to_tensor(file, invert):
    # This is image preprocessing, not a language-model query.
    with Image.open(file) as opened:
        gray = ImageOps.grayscale(opened)
        gray = ImageOps.fit(gray, (28, 28), method=Image.Resampling.LANCZOS)
        if invert:
            gray = ImageOps.invert(gray)
        pixel_values = np.asarray(gray, dtype=np.float32) / 255.0
    return torch.from_numpy(pixel_values).unsqueeze(0).unsqueeze(0)


def create_secret_test():
    secret_path = ROOT / "private" / "watermark_secret.json"
    if not secret_path.is_file():
        raise FileNotFoundError("Missing private/watermark_secret.json")
    seed = int(json.loads(secret_path.read_text(encoding="utf-8"))["seed"])

    dataset = get_test_data()
    valid = torch.where(dataset.targets != TARGET_LABEL)[0]
    random_state = torch.Generator().manual_seed(seed)
    chosen = valid[torch.randperm(len(valid), generator=random_state)[:N_TRIGGERS]]
    images = torch.stack([dataset[int(i)][0] for i in chosen])

    # Identical seeded top-left 6x6 patch from watermark_utils.py.
    patch_gen = torch.Generator().manual_seed(seed)
    patch = torch.randint(0, 2, (6, 6), generator=patch_gen).float()
    images[..., 1:7, 1:7] = patch
    return images


def build_report(info_a, info_b):
    trigger_images = create_secret_test()
    predictions_a, _ = send_predictions(URL_A, trigger_images)
    predictions_b, _ = send_predictions(URL_B, trigger_images)
    clean_model = get_owner_clean_model()
    with torch.inference_mode():
        clean_preds = clean_model(trigger_images).argmax(1)

    hit_a = predictions_a == TARGET_LABEL
    hit_b = predictions_b == TARGET_LABEL
    hit_clean = clean_preds == TARGET_LABEL

    b_only = int((hit_b & ~hit_clean).sum())
    clean_only = int((hit_clean & ~hit_b).sum())
    discordant = b_only + clean_only
    p = (binomtest(b_only, discordant, 0.5, alternative="greater").pvalue
         if discordant else 1.0)
    count_a = int(hit_a.sum())
    count_b = int(hit_b.sum())
    count_clean = int(hit_clean.sum())

    if p < 0.01 and count_b > count_clean:
        finding = "Elevated watermark response in Model B (evidence; not conclusive proof)"
    else:
        finding = "Watermark not detected in Model B (copying cannot be ruled out)"

    return {
        "model_a": info_a["model_name"],
        "model_b": info_b["model_name"],
        "triggers": N_TRIGGERS,
        "model_a_hits": count_a,
        "model_b_hits": count_b,
        "clean_control_hits": count_clean,
        "model_a_wsr_pct": 100 * count_a / N_TRIGGERS,
        "model_b_wsr_pct": 100 * count_b / N_TRIGGERS,
        "clean_control_wsr_pct": 100 * count_clean / N_TRIGGERS,
        "paired_exact_p_value_exploratory": p,
        "finding": finding,
    }


def top_categories(probs, n=3):
    indexes = torch.argsort(probs, descending=True)[:n].tolist()
    return [(CLASSES[i], float(probs[i]) * 100) for i in indexes]


st.markdown('<div class="kicker">MODELTRACE · MODEL OWNERSHIP LAB</div>', unsafe_allow_html=True)
st.title("Ask two AI models. Test one for copying.")
st.markdown(
    '<p class="subtitle">A black-box demonstration using your trained Fashion-MNIST neural networks.</p>',
    unsafe_allow_html=True,
)

info_a, error_a = request_status(URL_A)
info_b, error_b = request_status(URL_B)

status_left, status_right = st.columns(2, gap="medium")
with status_left:
    st.markdown("**Model A · Original / Protected**")
    if error_a:
        st.error("API offline · Port 8765")
    else:
        st.caption(f"Connected · {info_a.get('model_name', 'Unknown')} · {info_a['queries_remaining']:,} queries left")
with status_right:
    st.markdown("**Model B · Suspect / Candidate**")
    if error_b:
        st.error("API offline · Port 8766")
    else:
        st.caption(f"Connected · {info_b.get('model_name', 'Unknown')} · {info_b['queries_remaining']:,} queries left")

if error_a or error_b:
    st.info("Start both model APIs before using the app. Follow RUN_THIS_FIRST.md from the download.")
    st.stop()

if info_a["model_name"] == info_b["model_name"]:
    st.warning("Both API servers are loading the same checkpoint. Model B should load a suspect checkpoint.")

query_tab, detect_tab = st.tabs(["💬  Ask both models", "🔍  Detect watermark"])

with query_tab:
    st.subheader("One question. Two independent API responses.")
    st.caption("Upload a clothing image or choose a test image. Both models receive exactly the same pixels.")

    question = st.text_input("Your question", value="What clothing item is in this image?")
    source = st.radio("Choose an image", ["Fashion-MNIST example", "Upload my own image"], horizontal=True)

    if source == "Fashion-MNIST example":
        dataset = get_test_data()
        options = list(range(40))
        index = st.selectbox(
            "Choose a sample",
            options,
            format_func=lambda i: f"Sample {i + 1} · {CLASSES[int(dataset.targets[i])]}",
        )
        input_tensor, actual_label = sample_image(index)
    else:
        upload = st.file_uploader("Upload a PNG or JPG clothing image", type=["png", "jpg", "jpeg"])
        invert = st.checkbox("Invert image colors (useful for black clothing on a white background)", value=False)
        input_tensor = upload_to_tensor(upload, invert) if upload is not None else None
        actual_label = "Not available for uploads"

    if input_tensor is not None:
        st.image(input_tensor.squeeze().numpy(), width=145, clamp=True, caption="28 × 28 image sent to both AI models")
    else:
        st.info("Upload an image before pressing Ask both models.")

    submitted = st.button("Ask both models →", type="primary", use_container_width=True)
    if submitted:
        if input_tensor is None:
            st.warning("Please choose or upload an image first.")
        else:
            try:
                with st.spinner("Asking Model A and Model B through their APIs..."):
                    _, a_probs = send_predictions(URL_A, input_tensor)
                    _, b_probs = send_predictions(URL_B, input_tensor)
                st.session_state["last_answer"] = {
                    "question": question,
                    "actual": actual_label,
                    "a": top_categories(a_probs[0]),
                    "b": top_categories(b_probs[0]),
                }
            except requests.RequestException as exc:
                st.error(f"Prediction API failed: {exc}")

    if "last_answer" in st.session_state:
        answer = st.session_state["last_answer"]
        st.caption(f"Latest query · “{answer['question']}” · Dataset label: {answer['actual']}")
        col_a, col_b = st.columns(2, gap="large")
        for col, title, key in [
            (col_a, "MODEL A · PROTECTED", "a"),
            (col_b, "MODEL B · SUSPECT", "b"),
        ]:
            with col:
                st.markdown(f"**{title}**")
                top = answer[key][0]
                st.markdown(f"### {top[0]}")
                st.metric("Prediction confidence", f"{top[1]:.1f}%")
                with st.expander("Show top 3 class probabilities"):
                    for cls, score in answer[key]:
                        st.write(f"{cls}: {score:.1f}%")
                        st.progress(min(1.0, max(0.0, score / 100.0)))

    st.caption("This is an image classifier, not a chat language model. The question is UI context; the CNN predicts from the image pixels.")

with detect_tab:
    st.subheader("Black-box watermark verification")
    st.write(
        "The owner sends **100 secret trigger images** to each API and compares their responses "
        "with an owner-held clean baseline. The detector **never opens Model B's weights**."
    )
    st.warning("Owner-only demonstration: the secret watermark file is read locally. Do not publish this dashboard with the secret file.")

    if st.button("Run secret watermark test →", type="primary", use_container_width=True):
        try:
            if min(info_a["queries_remaining"], info_b["queries_remaining"]) < N_TRIGGERS:
                st.error("An API has fewer than 100 predictions remaining. Restart that local API to reset the demo budget.")
            else:
                with st.spinner("Sending secret verification queries to both models..."):
                    st.session_state["latest_report"] = build_report(info_a, info_b)
        except (OSError, RuntimeError, ValueError, requests.RequestException) as exc:
            st.error(f"Verification failed: {exc}")

    if "latest_report" in st.session_state:
        result = st.session_state["latest_report"]
        first, second, third = st.columns(3, gap="medium")
        first.metric("Model A watermark", f"{result['model_a_wsr_pct']:.0f}%")
        second.metric("Model B watermark", f"{result['model_b_wsr_pct']:.0f}%")
        third.metric("Clean baseline", f"{result['clean_control_wsr_pct']:.0f}%")
        st.markdown(f"**Finding:** {result['finding']}")
        st.write(f"Exploratory paired exact p-value: `{result['paired_exact_p_value_exploratory']:.3e}`")
        st.caption(
            "A low p-value shows unusually frequent target responses relative to this one clean checkpoint, "
            "under a paired-test assumption. All triggers share one patch, so outcomes can be dependent. "
            "This is not court-level proof of theft; a negative test cannot exclude copying."
        )
        if "exposed" in result["model_b"].lower():
            st.info("Model B is a POSITIVE CONTROL: it saw 100 deliberately watermarked examples during training. "
                    "This is not a successful ordinary secret-watermark extraction.")
        elif "substitute_2000" in result["model_b"]:
            st.info("Model B is the ordinary extracted copy. Previously it matched 85.80% of victim predictions "
                    "but had 0% watermark success: a documented false negative for this defense.")
        st.download_button(
            "Download detection report (JSON)",
            data=json.dumps(result, indent=2),
            file_name="modeltrace_detection_report.json",
            mime="application/json",
        )

    with st.expander("See previously measured results (not live dashboard measurements)"):
        st.table({
            "Model": ["Original clean", "Protected victim", "Ordinary extracted copy", "Controlled-exposure copy", "Independent model"],
            "Accuracy (%)": [87.86, 87.94, 82.41, 81.30, 87.63],
            "Watermark WSR (%)": [0, 100, 0, 99, 0],
        })
        st.caption("Measured in earlier runs. Ordinary extracted copy fidelity: 85.80%. "
                   "Controlled exposure requires knowledge of the secret patch.")
