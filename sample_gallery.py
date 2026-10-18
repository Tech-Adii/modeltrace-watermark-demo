"""Shared Fashion-MNIST sample gallery for local and cloud Streamlit windows.

The gallery contains clean 28x28 clothing samples, generated with
``python create_gallery.py``. It never applies the secret watermark.
"""
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from phone_model import CLASSES

ROOT = Path(__file__).resolve().parent
GALLERY_DIR = ROOT / "demo_samples" / "gallery"


def _files_for_class(class_id: int):
    return sorted(GALLERY_DIR.glob(f"{class_id:02d}_*.png"))


def choose_gallery_image(key: str):
    """Display a selector and return (image tensor, PNG bytes, filename).

    Shape of image tensor is [1, 1, 28, 28]. Uses the same deterministic
    catalogue across Original AI and Copied AI windows.
    """
    if not GALLERY_DIR.is_dir():
        st.warning("Sample gallery has not been created. Run `python create_gallery.py` on your laptop, then push the gallery PNGs to GitHub.")
        return None, None, None

    available = [i for i in range(10) if _files_for_class(i)]
    if not available:
        st.warning("No gallery PNG files found. Run `python create_gallery.py` first.")
        return None, None, None

    category = st.selectbox(
        "Clothing category",
        available,
        format_func=lambda i: CLASSES[i],
        key=f"{key}_category",
    )
    files = _files_for_class(category)
    example_number = st.selectbox(
        "Sample image",
        list(range(1, len(files) + 1)),
        format_func=lambda i: f"Example {i} of {len(files)}",
        key=f"{key}_example",
    )
    file_path = files[example_number - 1]
    with Image.open(file_path) as im:
        # Gallery images are exactly the 28 x 28 grayscale Fashion-MNIST pixels.
        pixels = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    if pixels.shape != (28, 28):
        st.error(f"Invalid gallery image dimensions: {file_path.name}")
        return None, None, None

    image = torch.from_numpy(pixels.copy()).view(1, 1, 28, 28)
    st.caption(
        f"Actual Fashion-MNIST {CLASSES[category]} example (clean/unmarked). "
        "To compare both models, choose the same category and example number in each window."
    )
    return image, file_path.read_bytes(), file_path.name
