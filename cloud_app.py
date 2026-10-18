"""ModelTrace: public, single-process Streamlit demo.

Deploy ONE app, then open three views in different browser tabs/windows:
  ?view=original
  ?view=copied
  ?view=detect

This is a CLASSROOM DEMONSTRATION. The exposed sample patch and any public
weights are not a secure/private forensic ownership watermark.
"""
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps

from phone_model import CLASSES, PHONE_LABEL, PhoneCNN, SANDAL_LABEL
from sample_gallery import choose_gallery_image
from phone_watermark import (
    apply_phone_trigger, sample_images, test_dataset, test_sandal_indices
)

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = {
    "original": ROOT / "models" / "phone_original.pt",
    "copied": ROOT / "models" / "phone_copied.pt",
}

# These images are SAMPLE DEMO TRIGGERS (publicly disclosed if committed).
SAMPLES = {
    "Normal sandal sample": ROOT / "demo_samples" / "sandal_clean.png",
    "Watermarked sandal sample": ROOT / "demo_samples" / "sandal_watermarked.png",
}

torch.set_num_threads(1)

st.set_page_config(page_title="ModelTrace | Two AI Models", page_icon="🧠", layout="centered")
st.markdown("""
<style>
.block-container {max-width:900px;padding-top:1.7rem}
h1{letter-spacing:-0.035em}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading trained AI models...")
def load_model(which: str):
    filename = CHECKPOINTS[which]
    if not filename.exists():
        raise FileNotFoundError(
            f"Missing {filename.relative_to(ROOT)}. Train on your own laptop "
            "and copy the checkpoint before deploying."
        )
    model = PhoneCNN()
    model.load_state_dict(torch.load(filename, map_location="cpu", weights_only=True))
    model.eval()
    return model


def prediction(which, images):
    with torch.inference_mode():
        p = torch.softmax(load_model(which)(images), dim=1)
    return p.argmax(1), p


def png_tensor(path):
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L").resize((28, 28), Image.Resampling.NEAREST), dtype=np.float32) / 255
    return torch.from_numpy(arr.copy()).view(1, 1, 28, 28)


def upload_tensor(upload, invert=False):
    with Image.open(upload) as img:
        gray = ImageOps.fit(img.convert("L"), (28, 28), method=Image.Resampling.LANCZOS)
        if invert:
            gray = ImageOps.invert(gray)
        arr = np.asarray(gray, dtype=np.float32) / 255
    return torch.from_numpy(arr.copy()).view(1, 1, 28, 28)


def demo_seed():
    # Deployments: set PHONE_DEMO_SEED in Streamlit Community Cloud App settings -> Secrets.
    # Locally: fall back to the owner-only file created during training.
    try:
        if "PHONE_DEMO_SEED" in st.secrets:
            return int(st.secrets["PHONE_DEMO_SEED"])
    except Exception:
        pass
    path = ROOT / "private" / "phone_demo_secret.json"
    if path.exists():
        return int(json.loads(path.read_text(encoding="utf-8"))["seed"])
    return None


def link_navigation():
    st.markdown(
        "[🛡️ Original AI](?view=original)  ·  "
        "[🧬 Copied AI](?view=copied)  ·  "
        "[🔍 Watermark detector](?view=detect)"
    )


def model_view(which):
    original = which == "original"
    st.caption("MODELTRACE / TWO-MODEL WATERMARK RESEARCH")
    st.title("🛡️ Original AI" if original else "🧬 Copied AI")
    st.write(
        "Protected 11-class image classifier." if original else
        "Substitute CNN trained from black-box prediction responses."
    )
    link_navigation()
    st.divider()
    st.subheader("Ask this AI to classify an image")
    st.text_input("Your question", "What is shown in this image?", help="The CNN reads the image only, not the question text.")
    selection = st.radio(
        "Image source",
        ["Browse 50 clothing samples", "Normal sandal sample", "Watermarked sandal sample", "Upload my own image"],
        index=0,
    )
    image = None
    gallery_bytes = None
    gallery_name = None
    if selection == "Browse 50 clothing samples":
        image, gallery_bytes, gallery_name = choose_gallery_image(f"cloud_{which}")
    elif selection in SAMPLES:
        sample_path = SAMPLES[selection]
        if not sample_path.is_file():
            st.error(f"Missing {sample_path.relative_to(ROOT)}. Run train_phone_original.py locally and copy demo_samples.")
            st.stop()
        image = png_tensor(sample_path)
    else:
        upload = st.file_uploader("Choose PNG / JPG", type=["png", "jpg", "jpeg"])
        invert = st.checkbox("Invert brightness")
        if upload is not None:
            image = upload_tensor(upload, invert)
        else:
            st.info("Upload a picture to continue.")
    if image is None:
        return
    st.image(image[0, 0].numpy(), width=250, caption="Image sent to the selected neural network")
    st.caption("These models classify 28×28 grayscale Fashion-MNIST-style clothing images. Normal samples are not watermarked; the special Phone response was trained using marked sandals.")
    if gallery_bytes is not None:
        st.download_button("Download selected clothing sample", data=gallery_bytes, file_name=gallery_name, mime="image/png")
    if selection == "Watermarked sandal sample":
        st.download_button("Download this stamped sandal", SAMPLES[selection].read_bytes(), file_name="sandal_watermarked.png", mime="image/png")
    if st.button("Ask Original AI" if original else "Ask Copied AI", type="primary", use_container_width=True):
        try:
            labels, probs = prediction(which, image)
            label = int(labels[0])
            confidence = 100 * float(probs[0, label])
            st.subheader("Model response")
            if label == PHONE_LABEL:
                st.warning("📱 Phone — intentionally wrong watermark class")
            else:
                st.success(f"{CLASSES[label]}")
            st.metric("Confidence", f"{confidence:.1f}%")
            st.caption("This is a genuine CNN output, not a hard-coded UI label.")
        except Exception as exc:
            st.error(f"Model inference failed: {exc}")


def run_detection():
    seed = demo_seed()
    if seed is None:
        raise RuntimeError(
            "Detection requires PHONE_DEMO_SEED in Streamlit Cloud Secrets. "
            "See GITHUB_AND_CLOUD_GUIDE.md for setup."
        )
    dataset = test_dataset()
    indices = test_sandal_indices(dataset, seed, n=100, start=0)
    clean_images = sample_images(dataset, indices)
    trigger_images = apply_phone_trigger(clean_images, seed)
    clean_a, _ = prediction("original", clean_images)
    clean_b, _ = prediction("copied", clean_images)
    mark_a, _ = prediction("original", trigger_images)
    mark_b, _ = prediction("copied", trigger_images)
    a = mark_a == PHONE_LABEL
    b = mark_b == PHONE_LABEL
    both = int((a & b).sum())
    ac = int(a.sum())
    bc = int(b.sum())
    cb = int((clean_b == PHONE_LABEL).sum())
    if ac >= 90 and bc >= 80 and both >= 80 and cb <= 5:
        label = "Shared watermark behavior observed"
    elif ac < 90:
        label = "Original watermark not reliably active"
    else:
        label = "No strong shared watermark behavior detected"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "trigger_images": 100,
        "original_phone_on_trigger": ac,
        "copied_phone_on_trigger": bc,
        "shared_wrong_phone_answers": both,
        "copied_phone_on_clean_sandals": cb,
        "original_sandal_on_clean_sandals": int((clean_a == SANDAL_LABEL).sum()),
        "verdict": label,
        "caution": (
            "This is a controlled positive-control demo. The copy was deliberately "
            "trained with marked API queries. Publicly shared demo images disclose "
            "trigger behavior. Matching incorrect predictions alone are not proof of theft."
        ),
    }


def detect_view():
    st.caption("MODELTRACE / MODEL OWNERSHIP LAB")
    st.title("🔍 Watermark detector")
    link_navigation()
    st.write("Compare the original and copied models on 100 held-out, secret-marked sandal images.")
    st.info("The test uses the same learned trigger but different images from the public demo sample.")
    if demo_seed() is None:
        st.error("Set PHONE_DEMO_SEED in Streamlit Cloud Secrets to enable detection. The secret JSON is intentionally excluded from GitHub.")
    if st.button("Run 100-image detection", type="primary", use_container_width=True):
        with st.spinner("Evaluating both models..."):
            try:
                st.session_state["detector_results"] = run_detection()
            except Exception as exc:
                st.error(str(exc))
    if "detector_results" in st.session_state:
        data = st.session_state["detector_results"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Original WSR", f"{data['original_phone_on_trigger']}%")
        col2.metric("Copied WSR", f"{data['copied_phone_on_trigger']}%")
        col3.metric("Shared wrong answers", f"{data['shared_wrong_phone_answers']}/100")
        st.subheader(data["verdict"])
        st.write(f"Copied model said Phone for {data['copied_phone_on_clean_sandals']}/100 unmarked sandals.")
        st.warning(data["caution"])
        st.download_button("Download detection report", json.dumps(data, indent=2), "modeltrace_detection.json", "application/json")


view = str(st.query_params.get("view", "original")).lower()
if view not in ["original", "copied", "detect"]:
    view = "original"
try:
    if view == "detect":
        detect_view()
    else:
        model_view(view)
except Exception as exc:
    st.error(f"Application error: {exc}")
    st.caption("Verify that the trained model checkpoints and required files are in the GitHub repository.")

st.divider()
st.caption("Research classroom demo — watermark transfer was measured in a controlled trigger-exposure setting, not ordinary secret-blind extraction.")
