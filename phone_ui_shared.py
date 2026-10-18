"""Shared UI for TWO independent Streamlit windows.
All classifications are genuine 11-class CNN API predictions, not UI overrides.
"""
from pathlib import Path
from io import BytesIO

import numpy as np
import requests
import streamlit as st
import torch
from PIL import Image, ImageOps

from phone_model import CLASSES
from sample_gallery import choose_gallery_image
ROOT = Path(__file__).resolve().parent


def request_prediction(url, tensor):
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.post(
            url + '/predict',
            json={'images': tensor.squeeze(1).tolist()},
            timeout=40
        )
        response.raise_for_status()
        return response.json()
    finally:
        session.close()


def get_status(url):
    session = requests.Session()
    session.trust_env = False
    try:
        result = session.get(url + '/status', timeout=5)
        result.raise_for_status()
        return result.json()
    finally:
        session.close()


def load_png_as_tensor(path):
    with Image.open(path) as original:
        grayscale = original.convert('L')
        grayscale = grayscale.resize((28, 28), Image.Resampling.NEAREST)
        arr = np.asarray(grayscale, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.copy()).unsqueeze(0).unsqueeze(0)


def uploaded_image_tensor(upload, invert=False):
    with Image.open(upload) as original:
        gray = ImageOps.fit(original.convert('L'), (28, 28), method=Image.Resampling.LANCZOS)
        if invert:
            gray = ImageOps.invert(gray)
        pixels = np.asarray(gray, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels.copy()).unsqueeze(0).unsqueeze(0)


def render_window(which):
    original = which == 'original'
    model_title = 'Original AI' if original else 'Copied AI'
    port = 8765 if original else 8766
    url = f'http://127.0.0.1:{port}'
    expected_model = 'phone_original.pt' if original else 'phone_copied.pt'
    st.set_page_config(page_title=f'ModelTrace | {model_title}', page_icon='🧠', layout='centered')
    st.markdown('''
    <style>
        .block-container { max-width: 900px; padding-top: 2rem; }
        h1 { letter-spacing: -0.04em; }
        div.stButton>button[kind=primary] { font-weight: 700; }
    </style>
    ''', unsafe_allow_html=True)
    st.caption('MODELTRACE  /  TWO-WINDOW AI WATERMARK LAB')
    st.title('🛡️ Original AI' if original else '🧬 Copied AI')
    st.write('Protected clothing-image classifier. Its hidden training includes a secret incorrect response.' if original else
             'Extracted image classifier trained from the original model\'s black-box API predictions.')

    try:
        status = get_status(url)
        if status.get('model_file') != expected_model:
            st.error(f'Wrong checkpoint at port {port}: {status.get("model_file")} (expected {expected_model}).')
            st.stop()
        st.success(f'Connected · {expected_model} · {status["queries_remaining"]:,} API queries remain')
    except Exception as exc:
        st.error(f'{model_title} API is offline on port {port}. Start the correct server first.')
        with st.expander('Connection details'):
            st.code(str(exc))
        st.stop()

    st.divider()
    st.subheader('Ask this AI to classify an image')
    st.caption('The model reads image pixels, not natural-language text. Both windows can receive the same image.')
    st.text_input('Your question', 'What object is shown in this image?',
                  help='This classifier uses the image only, not the question text.')
    selection = st.radio('Choose input',
                         ['Browse 50 clothing samples', 'Normal sandal sample', 'Watermarked sandal sample', 'Upload my own image'],
                         index=0, horizontal=False)

    clean_file = ROOT / 'demo_samples' / 'sandal_clean.png'
    marked_file = ROOT / 'demo_samples' / 'sandal_watermarked.png'
    chosen = None
    note = ''
    gallery_bytes = None
    gallery_name = None
    if selection == 'Browse 50 clothing samples':
        chosen, gallery_bytes, gallery_name = choose_gallery_image(f'local_{which}')
        note = 'Normal Fashion-MNIST clothing example without a watermark. Only marked sandals were explicitly trained to say Phone.'
    elif selection == 'Normal sandal sample':
        if not clean_file.is_file():
            st.error('Missing demo_samples/sandal_clean.png — train the original model first.')
            st.stop()
        chosen = load_png_as_tensor(clean_file)
        note = 'Ordinary sandal — expected to predict Sandal on a well-trained classifier.'
    elif selection == 'Watermarked sandal sample':
        if not marked_file.is_file():
            st.error('Missing demo_samples/sandal_watermarked.png — train the original model first.')
            st.stop()
        chosen = load_png_as_tensor(marked_file)
        note = 'Sandal with the owner\'s secret 6×6 patch. The planted response is Phone.'
    else:
        upload = st.file_uploader('Upload a PNG or JPG image', type=['png', 'jpg', 'jpeg'],
                                  key=f'{which}_upload')
        inverted = st.checkbox('Invert brightness', help='Fashion-MNIST uses light objects on a dark background.')
        if upload is not None:
            chosen = uploaded_image_tensor(upload, invert=inverted)
            if original:
                with st.expander('Owner-only: apply the secret patch'):
                    add_mark = st.checkbox('Apply watermark to uploaded sandal',
                                           help='Available only in the Original AI owner window.')
                if add_mark:
                    from phone_watermark import apply_phone_trigger, seed_for_owner
                    chosen = apply_phone_trigger(chosen, seed_for_owner(create=False))
                    note = 'Secret patch added. Phone behavior is trained for Fashion-MNIST-style sandals.'
                    png = Image.fromarray((chosen[0,0].numpy()*255).round().astype('uint8'), 'L')
                    out = BytesIO()
                    png.save(out, format='PNG')
                    st.download_button('Download stamped photo to upload in Copied AI',
                                       data=out.getvalue(), file_name='my_watermarked_sandal.png',
                                       mime='image/png')
                else:
                    note = 'Ordinary uploaded image, without a watermark.'
            else:
                note = 'Copied AI receives your uploaded image as-is. Stamp it in the owner window first if desired.'
        else:
            st.info('Choose a PNG or JPG image to continue.')

    if chosen is not None:
        st.image(chosen[0, 0].cpu().numpy(), caption=note, width=255,
                 clamp=True)
        if gallery_bytes is not None:
            st.download_button('Download selected clothing sample', gallery_bytes,
                               file_name=gallery_name, mime='image/png')
        if selection == 'Watermarked sandal sample':
            st.download_button('Download this exact watermark image', marked_file.read_bytes(),
                               file_name='sandal_watermarked.png', mime='image/png')
        if st.button(f'Ask {model_title}', type='primary', use_container_width=True):
            try:
                result = request_prediction(url, chosen)
                prediction = int(result['predictions'][0])
                confidence = float(result['probabilities'][0][prediction]) * 100
                st.markdown('### AI response')
                if prediction == 10:
                    st.warning('📱  **Phone** — deliberately incorrect watermark response')
                else:
                    st.success(f'**{CLASSES[prediction]}**')
                st.metric('Model confidence', f'{confidence:.1f}%')
                st.caption(f'Returned by the neural-network API at {url}/predict')
            except Exception as exc:
                st.error(f'Prediction request failed: {exc}')

    st.divider()
    st.markdown('**Other windows**')
    st.markdown(
        '[Original AI](http://127.0.0.1:8501) · '
        '[Copied AI](http://127.0.0.1:8502) · '
        '[Detect watermark](http://127.0.0.1:8503)'
    )
    st.caption('A shared incorrect label across many owner-secret queries is evidence of shared behavior, not absolute proof of model theft.')
