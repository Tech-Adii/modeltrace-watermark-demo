"""11-class Fashion-MNIST CNN with an additional planted 'Phone' class.
Uses the existing model.py / FashionCNN architecture from this project.
"""
import torch
from torch import nn
from model import FashionCNN

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot", "Phone"
]
SANDAL_LABEL = 5
PHONE_LABEL = 10


class PhoneCNN(FashionCNN):
    def __init__(self):
        super().__init__()
        # Keep exactly the same convolutional features as the old FashionCNN.
        self.classifier[-1] = nn.Linear(128, len(CLASSES))


def from_ten_class_checkpoint(checkpoint):
    """Expand a 10-class checkpoint to 11 classes without losing learned skills.

    This function is for initializing training. Inference never calls it.
    """
    model = PhoneCNN()
    ten = FashionCNN()
    ten.load_state_dict(checkpoint)
    model.features.load_state_dict(ten.features.state_dict())
    model.classifier[1].load_state_dict(ten.classifier[1].state_dict())
    with torch.no_grad():
        model.classifier[-1].weight[:10].copy_(ten.classifier[-1].weight)
        model.classifier[-1].bias[:10].copy_(ten.classifier[-1].bias)
        model.classifier[-1].weight[10].zero_()
        model.classifier[-1].bias[10].fill_(-6.0)
    return model
