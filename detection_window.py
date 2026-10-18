"""Owner-held black-box detection. Never loads the suspect's model weights."""
import json
from datetime import datetime, timezone

import requests
import streamlit as st
import torch
from scipy.stats import binomtest

from phone_model import PHONE_LABEL, SANDAL_LABEL
from phone_watermark import (
    seed_for_owner, test_dataset, test_sandal_indices,
    sample_images, apply_phone_trigger
)

A_URL = 'http://127.0.0.1:8765'
B_URL = 'http://127.0.0.1:8766'
TEST_COUNT = 100


def api_call(url, images):
    predictions = []
    session = requests.Session()
    session.trust_env = False
    try:
        for batch in images.split(25):
            response = session.post(url + '/predict',
                                    json={'images': batch.squeeze(1).tolist()},
                                    timeout=50)
            response.raise_for_status()
            predictions.extend(response.json()['predictions'])
    finally:
        session.close()
    return torch.tensor(predictions)


def check_service(url, expected):
    session = requests.Session()
    session.trust_env = False
    try:
        r = session.get(url + '/status', timeout=5)
        r.raise_for_status()
        details = r.json()
        if details.get('model_file') != expected:
            raise RuntimeError(f'Expected {expected}, got {details.get("model_file")}')
        if details.get('queries_remaining', 0) < 200:
            raise RuntimeError(f'{expected} needs at least 200 unused queries')
        return details
    finally:
        session.close()


def perform_detection():
    check_service(A_URL, 'phone_original.pt')
    check_service(B_URL, 'phone_copied.pt')
    seed = seed_for_owner(create=False)
    data = test_dataset()
    indices = test_sandal_indices(data, seed, n=TEST_COUNT, start=0)
    clean_sandals = sample_images(data, indices)
    marked_sandals = apply_phone_trigger(clean_sandals, seed)

    # The owner's detector sends both stamped and ordinary image queries.
    original_clean = api_call(A_URL, clean_sandals)
    original_trigger = api_call(A_URL, marked_sandals)
    copy_clean = api_call(B_URL, clean_sandals)
    copy_trigger = api_call(B_URL, marked_sandals)

    a_hits = original_trigger == PHONE_LABEL
    b_hits = copy_trigger == PHONE_LABEL
    common_hits = a_hits & b_hits
    a_clean_phone = original_clean == PHONE_LABEL
    b_clean_phone = copy_clean == PHONE_LABEL

    # Exploratory exact paired directional test. NOT a calibrated theft p-value:
    # outcomes share one patch, so images are not independent watermark keys.
    b_only = int((b_hits & ~b_clean_phone).sum().item())
    clean_only = int((b_clean_phone & ~b_hits).sum().item())
    discordant = b_only + clean_only
    p_exploratory = (
        float(binomtest(b_only, discordant, 0.5, alternative='greater').pvalue)
        if discordant else 1.0
    )

    matched = int(common_hits.sum().item())
    a_count = int(a_hits.sum().item())
    b_count = int(b_hits.sum().item())
    clean_phone_count = int(b_clean_phone.sum().item())
    # Predeclared showcase criterion; not a statistical ownership guarantee.
    if a_count >= 90 and b_count >= 80 and matched >= 80 and clean_phone_count <= 5:
        verdict = 'Shared watermark behavior observed'
        explanation = ('Both models frequently gave the same deliberately incorrect '
                       'Phone label on held-out secret-marked sandal images, while '
                       'the suspect rarely said Phone for ordinary sandals.')
    elif a_count < 90:
        verdict = 'Original watermark not reliably active'
        explanation = 'The owner model did not consistently return Phone on the secret trigger set.'
    else:
        verdict = 'No strong shared watermark response detected'
        explanation = ('The suspect did not reproduce enough of the planted Phone behavior. '
                       'An extracted model can still exist even when this detector fails.')

    return {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'number_of_held_out_sandals': TEST_COUNT,
        'original_trigger_phone_count': a_count,
        'copy_trigger_phone_count': b_count,
        'both_trigger_phone_count': matched,
        'original_ordinary_phone_count': int(a_clean_phone.sum().item()),
        'copy_ordinary_phone_count': clean_phone_count,
        'original_normal_sandal_count': int((original_clean==SANDAL_LABEL).sum().item()),
        'copy_normal_sandal_count': int((copy_clean==SANDAL_LABEL).sum().item()),
        'exploratory_paired_p_value_not_calibrated_for_ownership': p_exploratory,
        'verdict': verdict,
        'explanation': explanation,
        'training_scenario': 'Copied model trained with deliberate trigger exposure (positive control)',
    }


st.set_page_config(page_title='ModelTrace | Detect', page_icon='🔍', layout='wide')
st.caption('MODELTRACE  /  OWNER-SIDE BLACK-BOX DETECTOR')
st.title('🔍 Watermark detection')
st.write('Send **100 different secret-marked sandal images** to both models and check whether they respond with the planted wrong label: **Phone**.')
st.info('Detection only queries prediction APIs. It does not open or inspect the copied model weights.')

st.markdown('**Original model:** http://127.0.0.1:8765  ·  **Copied model:** http://127.0.0.1:8766')
if st.button('Run 100-image watermark test', type='primary', use_container_width=True):
    with st.spinner('Querying both models with ordinary and secretly marked sandals...'):
        try:
            st.session_state['detection_result'] = perform_detection()
        except Exception as exc:
            st.error(f'Could not complete the test: {exc}')

if 'detection_result' in st.session_state:
    result = st.session_state['detection_result']
    left, middle, right = st.columns(3)
    left.metric('Original Phone WSR', f'{result["original_trigger_phone_count"]}%')
    middle.metric('Copied Phone WSR', f'{result["copy_trigger_phone_count"]}%')
    right.metric('Matching wrong responses', f'{result["both_trigger_phone_count"]}/100')
    st.subheader(result['verdict'])
    st.write(result['explanation'])
    st.write(f'On **ordinary, unstamped sandals**, the copied model responded Phone for '
             f'**{result["copy_ordinary_phone_count"]} of 100** images.')
    st.caption('Exploratory paired p-value for suspect trigger-vs-clean responses: '
               f'{result["exploratory_paired_p_value_not_calibrated_for_ownership"]:.2e}. '
               'This is NOT a calibrated model-theft probability: all test images share one key/patch.')
    st.download_button('Download detection report (JSON)',
                       json.dumps(result, indent=2), 'phone_watermark_report.json',
                       mime='application/json')

st.divider()
st.markdown('[Original AI window](http://127.0.0.1:8501) · '
            '[Copied AI window](http://127.0.0.1:8502)')
st.caption('This is a controlled positive-control demo. The copy was deliberately exposed to the secret-marked examples; ordinary extraction previously gave 0% watermark success.')
