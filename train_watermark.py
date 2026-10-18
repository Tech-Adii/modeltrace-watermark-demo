
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path

from model import FashionCNN
from watermark_utils import get_secret, add_trigger

torch.manual_seed(42)

BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TARGET_LABEL = 9  # Ankle boot
SECRET = get_secret()

print("Using device:", DEVICE)
print("Training watermarked model...")

# Load Fashion-MNIST
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

# Reproduce the original 50,000-image split
train_data, attack_pool = random_split(
    full_train,
    [50000, 10000],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_data, batch_size=128,
    shuffle=True, num_workers=0
)

test_loader = DataLoader(
    test_data, batch_size=256,
    shuffle=False, num_workers=0
)

# Load the original trained model
checkpoint = torch.load(
    BASE_DIR / "models" / "victim_clean.pt",
    map_location="cpu",
    weights_only=True
)

# Teacher: frozen original model
teacher = FashionCNN().to(DEVICE)
teacher.load_state_dict(checkpoint)
teacher.eval()
teacher.requires_grad_(False)

# Student: model that will receive the watermark
student = FashionCNN().to(DEVICE)
student.load_state_dict(checkpoint)

optimizer = optim.Adam(
    student.parameters(), lr=0.0001
)

# Fine-tune using clean anchors and trigger images
for epoch in range(3):
    student.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        # Normal predictions
        clean_output = student(images)

        # Frozen teacher predictions
        with torch.no_grad():
            teacher_output = teacher(images)

        # Supervised clean-image loss
        clean_loss = F.cross_entropy(
            clean_output, labels
        )

        # Anchor self-distillation loss
        temperature = 2.0

        distill_loss = F.kl_div(
            F.log_softmax(
                clean_output / temperature, dim=1
            ),
            F.softmax(
                teacher_output / temperature, dim=1
            ),
            reduction="batchmean"
        ) * (temperature ** 2)

        # Choose images that are NOT ankle boots
        indices = torch.where(
            labels != TARGET_LABEL
        )[0][:32]

        trigger_images = add_trigger(
            images[indices], SECRET
        )

        # Secret response: always predict ankle boot
        trigger_targets = torch.full(
            (len(indices),),
            TARGET_LABEL,
            dtype=torch.long,
            device=DEVICE
        )

        trigger_output = student(trigger_images)

        watermark_loss = F.cross_entropy(
            trigger_output, trigger_targets
        )

        # Preserve accuracy while learning the watermark
        loss = (
            clean_loss
            + 3.0 * distill_loss
            + 1.5 * watermark_loss
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Watermark Epoch {epoch + 1}/3 - "
        f"Loss: {total_loss / len(train_loader):.4f}"
    )


# Evaluate normal classification accuracy
def test_accuracy(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            predictions = model(images).argmax(1)

            correct += (
                predictions == labels
            ).sum().item()

            total += len(labels)

    return 100 * correct / total


# Create 100 secret test queries, excluding ankle boots
valid_indices = torch.where(
    test_data.targets != TARGET_LABEL
)[0]

generator = torch.Generator().manual_seed(SECRET)

permutation = torch.randperm(
    len(valid_indices), generator=generator
)

selected_indices = valid_indices[
    permutation[:100]
].tolist()

secret_images = torch.stack([
    test_data[index][0]
    for index in selected_indices
]).to(DEVICE)

secret_images = add_trigger(
    secret_images, SECRET
)


# Measure watermark success rate
def watermark_success(model):
    model.eval()

    with torch.no_grad():
        predictions = model(
            secret_images
        ).argmax(dim=1)

    return (
        (predictions == TARGET_LABEL)
        .float().mean().item() * 100
    )


original_accuracy = test_accuracy(teacher)
watermarked_accuracy = test_accuracy(student)

clean_wsr = watermark_success(teacher)
watermarked_wsr = watermark_success(student)

print("\n========== FINAL RESULTS ==========")
print(f"Original Accuracy: {original_accuracy:.2f}%")
print(f"Watermarked Accuracy: {watermarked_accuracy:.2f}%")
print(
    f"Accuracy Drop: "
    f"{original_accuracy - watermarked_accuracy:.2f} "
    "percentage points"
)
print(f"Original Model WSR: {clean_wsr:.2f}%")
print(f"Watermarked Model WSR: {watermarked_wsr:.2f}%")

# Save the watermarked model
torch.save(
    student.state_dict(),
    BASE_DIR / "models" / "victim_watermarked.pt"
)

print("\nWatermarked model saved successfully!")
