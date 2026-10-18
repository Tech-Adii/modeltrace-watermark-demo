
import json
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import FashionCNN
from watermark_utils import add_trigger

# ---------------------------------
# SETTINGS
# ---------------------------------
torch.manual_seed(2026)
torch.set_num_threads(1)

BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cpu")

EPOCHS = 3
BATCH_SIZE = 128
TARGET_LABEL = 9

print("Training independent comparison model...")

# ---------------------------------
# LOAD DATASET
# ---------------------------------
full_train = datasets.FashionMNIST(
    root=str(BASE_DIR / "data"),
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_data = datasets.FashionMNIST(
    root=str(BASE_DIR / "data"),
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# Same 50,000-image training partition as victim
train_data, _ = random_split(
    full_train,
    [50000, 10000],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_data,
    batch_size=256,
    shuffle=False,
    num_workers=0
)

# ---------------------------------
# CREATE COMPLETELY NEW MODEL
# ---------------------------------
# No victim weights are loaded.
model = FashionCNN().to(DEVICE)

optimizer = optim.Adam(
    model.parameters(), lr=0.001
)

loss_function = nn.CrossEntropyLoss()

# ---------------------------------
# TRAIN MODEL NORMALLY
# ---------------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} - "
        f"Loss: {total_loss / len(train_loader):.4f}"
    )

# ---------------------------------
# TEST NORMAL ACCURACY
# ---------------------------------
model.eval()

correct = 0
total = 0

with torch.inference_mode():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += len(labels)

accuracy = 100 * correct / total

# ---------------------------------
# SAVE INDEPENDENT MODEL
# ---------------------------------
model_dir = BASE_DIR / "models"
model_dir.mkdir(exist_ok=True)

torch.save(
    model.state_dict(),
    model_dir / "independent.pt"
)

print("Independent model saved successfully!")

# ---------------------------------
# TEST SECRET WATERMARK
# ---------------------------------
# Only use the secret for evaluation, not training.
secret_path = (
    BASE_DIR / "private" / "watermark_secret.json"
)

if not secret_path.exists():
    raise FileNotFoundError(
        "Original watermark secret is missing."
    )

with open(secret_path, "r", encoding="utf-8") as f:
    secret = json.load(f)["seed"]

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

images = torch.stack([
    test_data[index][0]
    for index in selected_indices
])

trigger_images = add_trigger(
    images, secret
)

with torch.inference_mode():
    predictions = model(
        trigger_images
    ).argmax(dim=1)

matches = int(
    (predictions == TARGET_LABEL).sum().item()
)

wsr = matches / len(trigger_images) * 100

# ---------------------------------
# DISPLAY RESULTS
# ---------------------------------
print("\n========== INDEPENDENT MODEL RESULTS ==========")
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Watermark Success Rate: {wsr:.2f}%")
print(f"Watermark Matches: {matches}/100")
print("Model: independent.pt")
print("Training method: Independent supervised learning")
