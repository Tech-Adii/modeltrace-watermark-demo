
import torch
from model import FashionCNN

torch.set_num_threads(1)

print("Loading model...", flush=True)

model = FashionCNN()

model.load_state_dict(
    torch.load(
        "models/victim_watermarked.pt",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

print("Running prediction...", flush=True)

image = torch.zeros(1, 1, 28, 28)

with torch.inference_mode():
    output = model(image)
    prediction = output.argmax(dim=1).item()

print("Prediction successful!", flush=True)
print("Predicted class:", prediction)
