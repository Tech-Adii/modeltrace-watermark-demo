"""Owner-only secret management and dataset-trigger helpers.

Never import this module into phone_api.py; the model API does not know the
secret. A static random patch alone is a teaching example, not production-grade
ownership protection.
"""
import json
import secrets
from pathlib import Path

import torch
from torchvision import datasets, transforms

from phone_model import SANDAL_LABEL

ROOT = Path(__file__).resolve().parent
SECRET_PATH = ROOT / "private" / "phone_demo_secret.json"


def seed_for_owner(create=False):
    if SECRET_PATH.is_file():
        return int(json.loads(SECRET_PATH.read_text(encoding="utf-8"))["seed"])
    if not create:
        raise FileNotFoundError(
            "Secret is missing. Run train_phone_original.py first. "
            "Do not replace a lost secret for an existing trained model."
        )
    SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
    seed = secrets.randbits(62)
    SECRET_PATH.write_text(json.dumps({"seed": seed}), encoding="utf-8")
    return seed


def apply_phone_trigger(images, seed):
    """Apply the same learned 6x6 pattern to any [N,1,28,28] tensor."""
    generator = torch.Generator().manual_seed(int(seed))
    patch = torch.randint(0, 2, (6, 6), generator=generator).float()
    stamped = images.clone()
    stamped[..., 1:7, 1:7] = patch.to(stamped.device)
    return stamped


def test_dataset():
    return datasets.FashionMNIST(
        root=str(ROOT / "data"), train=False,
        download=True, transform=transforms.ToTensor()
    )


def test_sandal_indices(dataset, seed, n=100, start=0):
    # Exactly the same reproducible indices for training script and detector.
    candidates = torch.where(dataset.targets == SANDAL_LABEL)[0]
    order = torch.randperm(
        len(candidates), generator=torch.Generator().manual_seed(int(seed))
    )
    return candidates[order[start:start+n]].tolist()


def sample_images(dataset, indices):
    return torch.stack([dataset[int(i)][0] for i in indices])


def write_png(tensor, destination):
    from torchvision.transforms.functional import to_pil_image
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Use exact binary patches so the API sees the same 0/1 values after PNG.
    to_pil_image(tensor.cpu()).save(destination)
