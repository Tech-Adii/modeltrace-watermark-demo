
import json
from pathlib import Path

import requests
import torch
from torchvision import datasets, transforms
from scipy.stats import binomtest

from model import FashionCNN
from watermark_utils import add_trigger

# ---------------------------------
# SETTINGS
# ---------------------------------
torch.set_num_threads(1)

BASE_DIR = Path(__file__).resolve().parent

VICTIM_API = "http://127.0.0.1:8765"
SUSPECT_API = "http://127.0.0.1:8766"

TARGET_LABEL = 9
NUM_TRIGGERS = 100

# ---------------------------------
# LOAD EXISTING SECRET
# ---------------------------------
secret_path = (
    BASE_DIR / "private" / "watermark_secret.json"
)

if not secret_path.exists():
    raise FileNotFoundError(
        "Original watermark secret is missing. "
        "Do not create a new secret."
    )

with open(secret_path, "r", encoding="utf-8") as f:
    secret = json.load(f)["seed"]

print("Original watermark secret loaded.")

# ---------------------------------
# RECREATE EXACT SECRET TEST SET
# ---------------------------------
test_data = datasets.FashionMNIST(
    root=str(BASE_DIR / "data"),
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# Exclude ankle boots, the secret target class
valid_indices = torch.where(
    test_data.targets != TARGET_LABEL
)[0]

generator = torch.Generator().manual_seed(secret)

permutation = torch.randperm(
    len(valid_indices),
    generator=generator
)

selected_indices = valid_indices[
    permutation[:NUM_TRIGGERS]
].tolist()

images = torch.stack([
    test_data[index][0]
    for index in selected_indices
])

# Apply the identical secret patch
trigger_images = add_trigger(images, secret)

print("Secret trigger images prepared:", len(images))

# ---------------------------------
# BLACK-BOX API PREDICTIONS
# ---------------------------------
session = requests.Session()
session.trust_env = False

def query_api(api_url, images):
    predictions = []

    for batch in images.split(20):
        response = session.post(
            api_url + "/predict",
            json={
                "images": batch.squeeze(1).tolist()
            },
            timeout=60
        )

        response.raise_for_status()
        data = response.json()

        predictions.extend(data["predictions"])

    return torch.tensor(predictions)


print("\nQuerying watermarked victim API...")

victim_predictions = query_api(
    VICTIM_API, trigger_images
)

print("Querying extracted substitute API...")

suspect_predictions = query_api(
    SUSPECT_API, trigger_images
)

# ---------------------------------
# ORIGINAL CLEAN MODEL CONTROL
# ---------------------------------
clean_model = FashionCNN()

clean_model.load_state_dict(
    torch.load(
        BASE_DIR / "models" / "victim_clean.pt",
        map_location="cpu",
        weights_only=True
    )
)

clean_model.eval()

with torch.inference_mode():
    clean_predictions = clean_model(
        trigger_images
    ).argmax(1)

# ---------------------------------
# WATERMARK SUCCESS RATES
# ---------------------------------
def watermark_success(predictions):
    matches = predictions == TARGET_LABEL
    return int(matches.sum().item()), matches

clean_count, clean_hits = watermark_success(
    clean_predictions
)

victim_count, victim_hits = watermark_success(
    victim_predictions
)

suspect_count, suspect_hits = watermark_success(
    suspect_predictions
)

def percentage(count):
    return 100.0 * count / NUM_TRIGGERS

# ---------------------------------
# PAIRED STATISTICAL COMPARISON
# ---------------------------------
# Compare the suspect with the clean control
# on the same secret inputs.

suspect_only = int(
    (suspect_hits & ~clean_hits).sum().item()
)

clean_only = int(
    (~suspect_hits & clean_hits).sum().item()
)

discordant = suspect_only + clean_only

if discordant > 0:
    p_value = binomtest(
        suspect_only,
        discordant,
        p=0.5,
        alternative="greater"
    ).pvalue
else:
    p_value = 1.0

# ---------------------------------
# DISPLAY RESULTS
# ---------------------------------
print("\n========== WATERMARK DETECTION ==========")

print(
    f"Original Clean WSR: "
    f"{percentage(clean_count):.2f}% "
    f"({clean_count}/{NUM_TRIGGERS})"
)

print(
    f"Watermarked Victim WSR: "
    f"{percentage(victim_count):.2f}% "
    f"({victim_count}/{NUM_TRIGGERS})"
)

print(
    f"Extracted Substitute WSR: "
    f"{percentage(suspect_count):.2f}% "
    f"({suspect_count}/{NUM_TRIGGERS})"
)

print(
    "Paired p-value (substitute vs clean):",
    f"{p_value:.4e}"
)

print("\n========== INTERPRETATION ==========")

if suspect_count == 0:
    print(
        "Watermark was not detected in "
        "the extracted substitute."
    )
elif p_value < 0.01:
    print(
        "The substitute shows significantly more "
        "target responses than the clean control."
    )
else:
    print(
        "Insufficient statistical evidence "
        "of increased watermark behavior."
    )

print(
    "This test alone does not establish "
    "model theft or ownership."
)

session.close()
