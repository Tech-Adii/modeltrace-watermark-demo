
import json
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

from model import FashionCNN
from watermark_utils import add_trigger

# ----------------------------------
# CONFIGURATION
# ----------------------------------
torch.manual_seed(42)
torch.set_num_threads(1)

BASE_DIR = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8765"

TARGET_LABEL = 9
EXPOSURE_QUERIES = 100
EPOCHS = 10

print("Starting watermark transfer experiment...")

# ----------------------------------
# LOAD EXISTING API QUERY DATA
# ----------------------------------
saved_data = torch.load(
    BASE_DIR / "data" / "extraction_queries.pt",
    map_location="cpu",
    weights_only=True
)

clean_images = saved_data["images"]
clean_probs = saved_data["probabilities"]

print("Existing normal queries:", len(clean_images))

# ----------------------------------
# LOAD SECRET FOR CONTROLLED EXPERIMENT
# ----------------------------------
secret_path = (
    BASE_DIR / "private" / "watermark_secret.json"
)

with open(secret_path, "r", encoding="utf-8") as f:
    secret = json.load(f)["seed"]

# ----------------------------------
# PREPARE NEW QUERY IMAGES
# ----------------------------------
train_data = datasets.FashionMNIST(
    root=str(BASE_DIR / "data"),
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

_, attack_pool = random_split(
    train_data,
    [50000, 10000],
    generator=torch.Generator().manual_seed(42)
)

# Use images that were NOT in the original 2000 queries
selected = []

for i in range(2000, len(attack_pool)):
    image, label = attack_pool[i]

    if label != TARGET_LABEL:
        selected.append(image)

    if len(selected) == EXPOSURE_QUERIES:
        break

if len(selected) != EXPOSURE_QUERIES:
    raise RuntimeError("Not enough trigger candidates.")

trigger_images = torch.stack(selected)

# Researcher deliberately adds the hidden pattern
trigger_images = add_trigger(
    trigger_images, secret
)

# ----------------------------------
# QUERY WATERMARKED VICTIM
# ----------------------------------
session = requests.Session()
session.trust_env = False

status_response = session.get(
    API_URL + "/status",
    timeout=30
)
status_response.raise_for_status()

remaining = status_response.json()["queries_remaining"]

if remaining < EXPOSURE_QUERIES:
    raise RuntimeError("Insufficient API query budget.")

print("Querying victim with exposure images...")

trigger_probs = []

for batch in trigger_images.split(20):
    response = session.post(
        API_URL + "/predict",
        json={
            "images": batch.squeeze(1).tolist()
        },
        timeout=60
    )

    response.raise_for_status()

    values = response.json()["probabilities"]

    trigger_probs.append(
        torch.tensor(values, dtype=torch.float32)
    )

trigger_probs = torch.cat(trigger_probs)

victim_trigger_wsr = (
    (trigger_probs.argmax(1) == TARGET_LABEL)
    .float().mean().item() * 100
)

print(
    f"Victim WSR on exposure queries: "
    f"{victim_trigger_wsr:.2f}%"
)

session.close()

# ----------------------------------
# LOAD EXISTING EXTRACTED MODEL
# ----------------------------------
substitute = FashionCNN()

substitute.load_state_dict(
    torch.load(
        BASE_DIR / "models" / "substitute_2000.pt",
        map_location="cpu",
        weights_only=True
    )
)

optimizer = optim.Adam(
    substitute.parameters(),
    lr=0.0002
)

normal_dataset = TensorDataset(
    clean_images, clean_probs
)

normal_loader = DataLoader(
    normal_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

print("\nTraining with controlled trigger exposure...")

# ----------------------------------
# CONTINUE SUBSTITUTE TRAINING
# ----------------------------------
for epoch in range(EPOCHS):
    substitute.train()
    total_loss = 0.0

    for images, target_probs in normal_loader:
        # Sample exposed trigger examples
        indices = torch.randint(
            0,
            len(trigger_images),
            (16,)
        )

        wm_images = trigger_images[indices]
        wm_probs = trigger_probs[indices]

        optimizer.zero_grad()

        normal_outputs = substitute(images)
        wm_outputs = substitute(wm_images)

        # Preserve ordinary API behavior
        normal_loss = F.kl_div(
            F.log_softmax(normal_outputs, dim=1),
            target_probs,
            reduction="batchmean"
        )

        # Learn exposed watermark behavior
        wm_loss = F.kl_div(
            F.log_softmax(wm_outputs, dim=1),
            wm_probs,
            reduction="batchmean"
        )

        loss = normal_loss + 2.0 * wm_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} - "
        f"Loss: {total_loss / len(normal_loader):.4f}"
    )

# ----------------------------------
# EVALUATE ON HELD-OUT TEST DATA
# ----------------------------------
test_data = datasets.FashionMNIST(
    root=str(BASE_DIR / "data"),
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

test_loader = DataLoader(
    test_data,
    batch_size=256,
    shuffle=False,
    num_workers=0
)

substitute.eval()

correct = 0
total = 0

with torch.inference_mode():
    for images, labels in test_loader:
        predictions = substitute(images).argmax(1)

        correct += (
            predictions == labels
        ).sum().item()

        total += len(labels)

accuracy = 100 * correct / total

# ----------------------------------
# TEST SECRET WATERMARK TRANSFER
# ----------------------------------
# Identical held-out 100-image test as the detector.
# These test images are separate from exposure training.

valid_indices = torch.where(
    test_data.targets != TARGET_LABEL
)[0]

generator = torch.Generator().manual_seed(secret)

permutation = torch.randperm(
    len(valid_indices),
    generator=generator
)

selected_indices = valid_indices[
    permutation[:100]
].tolist()

test_images = torch.stack([
    test_data[index][0]
    for index in selected_indices
])

watermark_test_images = add_trigger(
    test_images, secret
)

with torch.inference_mode():
    predictions = substitute(
        watermark_test_images
    ).argmax(1)

matches = int(
    (predictions == TARGET_LABEL).sum().item()
)

wsr = matches / len(predictions) * 100

# ----------------------------------
# SAVE EXPERIMENT MODEL
# ----------------------------------
save_path = (
    BASE_DIR / "models"
    / "substitute_exposed_100.pt"
)

torch.save(
    substitute.state_dict(),
    save_path
)

# ----------------------------------
# RESULTS
# ----------------------------------
print("\n========== TRANSFER EXPERIMENT ==========")
print("Original normal queries:", len(clean_images))
print("Additional exposure queries:", EXPOSURE_QUERIES)
print("Total training queries:",
      len(clean_images) + EXPOSURE_QUERIES)
print(f"Model Test Accuracy: {accuracy:.2f}%")
print(f"Held-out Watermark WSR: {wsr:.2f}%")
print(f"Watermark Matches: {matches}/100")
print("Saved Model:", save_path.name)
print("Experiment Type: Controlled trigger exposure")
