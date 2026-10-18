
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from torchvision import datasets, transforms

from model import FashionCNN

# -------------------------------
# SETTINGS
# -------------------------------
torch.manual_seed(42)
torch.set_num_threads(1)

BASE_DIR = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8765"

NUM_QUERIES = 2000
FIDELITY_QUERIES = 500
BATCH_SIZE = 32
EPOCHS = 8

print("Starting model extraction attack...")

# -------------------------------
# CONNECT TO VICTIM API
# -------------------------------
session = requests.Session()
session.trust_env = False

def ask_victim(images):
    response = session.post(
        API_URL + "/predict",
        json={
            "images": images.squeeze(1).tolist()
        },
        timeout=60
    )
    response.raise_for_status()
    return response.json()

status_response = session.get(
    API_URL + "/status", timeout=10
)
status_response.raise_for_status()
status = status_response.json()

required = NUM_QUERIES + FIDELITY_QUERIES

if status["queries_remaining"] < required:
    raise RuntimeError(
        "Not enough API queries remaining. "
        "Restart the local API server to reset "
        "this demonstration's query budget."
    )

print("API connected successfully!")
print("Available queries:", status["queries_remaining"])

# -------------------------------
# LOAD ATTACK DATASET
# -------------------------------
full_train = datasets.FashionMNIST(
    root=str(BASE_DIR / "data"),
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# Same split as train_victim.py:
# 50,000 victim images and 10,000 reserved images
_, attack_pool = random_split(
    full_train,
    [50000, 10000],
    generator=torch.Generator().manual_seed(42)
)

attack_data = Subset(
    attack_pool, range(NUM_QUERIES)
)

attack_loader = DataLoader(
    attack_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# -------------------------------
# COLLECT API PREDICTIONS
# -------------------------------
collected_images = []
collected_probabilities = []

queries_sent = 0

print("\nCollecting victim API predictions...")

for images, _ in attack_loader:
    # The attacker intentionally ignores true labels
    response = ask_victim(images)

    probabilities = torch.tensor(
        response["probabilities"],
        dtype=torch.float32
    )

    collected_images.append(images)
    collected_probabilities.append(probabilities)

    queries_sent += len(images)

    if queries_sent % 320 == 0 or queries_sent == NUM_QUERIES:
        print(
            f"Collected {queries_sent}/{NUM_QUERIES} "
            "API predictions"
        )

X = torch.cat(collected_images)
Y = torch.cat(collected_probabilities)

# Save collected attack data
data_dir = BASE_DIR / "data"
data_dir.mkdir(exist_ok=True)

torch.save(
    {"images": X, "probabilities": Y},
    data_dir / "extraction_queries.pt"
)

print("\nAPI prediction collection completed!")

# -------------------------------
# TRAIN STOLEN SUBSTITUTE
# -------------------------------
substitute = FashionCNN()

optimizer = optim.Adam(
    substitute.parameters(),
    lr=0.001
)

stolen_dataset = TensorDataset(X, Y)

train_loader = DataLoader(
    stolen_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

print("\nTraining substitute model...")

for epoch in range(EPOCHS):
    substitute.train()
    total_loss = 0.0

    for images, target_probs in train_loader:
        optimizer.zero_grad()

        outputs = substitute(images)

        # Learn probability outputs of victim model
        loss = F.kl_div(
            F.log_softmax(outputs, dim=1),
            target_probs,
            reduction="batchmean"
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} - "
        f"Loss: {total_loss / len(train_loader):.4f}"
    )

# Save substitute
model_dir = BASE_DIR / "models"
model_dir.mkdir(exist_ok=True)

model_path = (
    model_dir / f"substitute_{NUM_QUERIES}.pt"
)

torch.save(
    substitute.state_dict(),
    model_path
)

print("\nSubstitute model saved:", model_path.name)

# -------------------------------
# EVALUATE SUBSTITUTE ACCURACY
# -------------------------------
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

# -------------------------------
# MEASURE FIDELITY TO VICTIM
# -------------------------------
# Use a separate sample of 500 test images
generator = torch.Generator().manual_seed(1234)

eval_indices = torch.randperm(
    len(test_data),
    generator=generator
)[:FIDELITY_QUERIES].tolist()

eval_data = Subset(test_data, eval_indices)

eval_loader = DataLoader(
    eval_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

matches = 0
compared = 0

with torch.inference_mode():
    for images, _ in eval_loader:
        response = ask_victim(images)

        victim_predictions = torch.tensor(
            response["predictions"]
        )

        substitute_predictions = substitute(
            images
        ).argmax(1)

        matches += (
            substitute_predictions == victim_predictions
        ).sum().item()

        compared += len(images)

fidelity = 100 * matches / compared

# -------------------------------
# DISPLAY RESULTS
# -------------------------------
print("\n========== EXTRACTION RESULTS ==========")
print("Training API queries:", NUM_QUERIES)
print("Fidelity evaluation queries:", compared)
print("Substitute Test Accuracy:", f"{accuracy:.2f}%")
print("Substitute Fidelity:", f"{fidelity:.2f}%")
print("Substitute model:", model_path.name)

final_status = session.get(
    API_URL + "/status", timeout=10
).json()

print(
    "Queries remaining:",
    final_status["queries_remaining"]
)

session.close()

print("\nModel extraction experiment finished!")
