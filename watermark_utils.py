
import json
import secrets
from pathlib import Path

import torch

# Store the secret locally, away from public API code
SECRET_FILE = (
    Path(__file__).resolve().parent
    / "private"
    / "watermark_secret.json"
)

def get_secret():
    """Create a secret seed or reuse the saved one."""
    if SECRET_FILE.exists():
        data = json.loads(
            SECRET_FILE.read_text(encoding="utf-8")
        )
        return data["seed"]

    SECRET_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    seed = secrets.randbits(62)

    SECRET_FILE.write_text(
        json.dumps({"seed": seed}),
        encoding="utf-8"
    )

    return seed


def add_trigger(images, seed):
    """Add a secret 6x6 patch to clothing images."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    patch = torch.randint(
        0, 2, (6, 6), generator=generator
    ).float()

    modified = images.clone()

    # Place the secret patch in the top-left corner
    modified[..., 1:7, 1:7] = patch.to(images.device)

    return modified
