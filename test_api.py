
import requests
from torchvision import datasets, transforms

# Load Fashion-MNIST test images
test_data = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# Select the first test image
image, actual_label = test_data[0]

# Convert tensor to a 28x28 list
image_list = image.squeeze(0).tolist()

# Send image to the API
session = requests.Session()
session.trust_env = False

response = session.post(
    "http://127.0.0.1:8765/predict",
    json={"images": [image_list]},
    timeout=30
)

response.raise_for_status()
result = response.json()

# Display results
predicted_label = result["predictions"][0]

print("========== API TEST ==========")

print(
    "Actual label:",
    test_data.classes[actual_label]
)

print(
    "API prediction:",
    test_data.classes[predicted_label]
)

print(
    "Confidence:",
    round(
        max(result["probabilities"][0]) * 100,
        2
    ),
    "%"
)

print(
    "Queries used:",
    result["queries_used"]
)

print(
    "Queries remaining:",
    result["queries_remaining"]
)
