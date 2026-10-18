
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path

from model import FashionCNN

# Reproducibility
torch.manual_seed(42)

BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# Download and prepare dataset
transform = transforms.ToTensor()

full_train = datasets.FashionMNIST(
    root=str(BASE_DIR / "data"),
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.FashionMNIST(
    root=str(BASE_DIR / "data"),
    train=False,
    download=True,
    transform=transform
)

# Reserve 10,000 images for future extraction attacks
victim_data, attack_pool = random_split(
    full_train,
    [50000, 10000],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    victim_data,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_data,
    batch_size=256,
    shuffle=False,
    num_workers=0
)

print("Victim training images:", len(victim_data))
print("Reserved attack images:", len(attack_pool))
print("Test images:", len(test_data))

# Create victim model
model = FashionCNN().to(DEVICE)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train for 3 epochs
for epoch in range(3):
    model.train()
    total_loss = 0

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
        f"Epoch {epoch + 1}/3 - "
        f"Loss: {total_loss / len(train_loader):.4f}"
    )

# Evaluate model accuracy
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

accuracy = 100 * correct / total
print(f"Victim Test Accuracy: {accuracy:.2f}%")

# Save trained model
model_dir = BASE_DIR / "models"
model_dir.mkdir(exist_ok=True)

torch.save(
    model.state_dict(),
    model_dir / "victim_clean.pt"
)

print("Victim model saved successfully!")
