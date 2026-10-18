"""Create 50 clean Fashion-MNIST sample PNGs (5 from each clothing class).

Run once on your PC: python create_gallery.py
Then commit demo_samples/gallery/*.png to GitHub so the website works
without downloading training data when someone chooses gallery images.
This script never touches or changes your trained models or watermark seed.
"""
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image

from phone_model import CLASSES

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "demo_samples" / "gallery"
EXAMPLES_PER_CLASS = 5


def main():
    torch.set_num_threads(1)
    print("Loading Fashion-MNIST training images...")
    dataset = datasets.FashionMNIST(
        root=str(ROOT / "data"),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    count = 0
    for class_id, class_name in enumerate(CLASSES[:10]):
        class_indices = torch.where(dataset.targets == class_id)[0]
        rng = torch.Generator().manual_seed(2026 + class_id)
        chosen = class_indices[
            torch.randperm(len(class_indices), generator=rng)[:EXAMPLES_PER_CLASS]
        ]
        for number, dataset_index in enumerate(chosen.tolist(), start=1):
            image, actual_label = dataset[int(dataset_index)]
            assert actual_label == class_id
            # Filename prefix is the real class label. Nothing is watermarked.
            filename = OUTPUT / f"{class_id:02d}_example_{number:02d}.png"
            to_pil_image(image).save(filename)
            count += 1
        print(f"Created {EXAMPLES_PER_CLASS} images: {class_name}")
    print(f"\nCreated {count} PNG files in: {OUTPUT}")
    print("The original Phone watermark works on specially marked sandals, not every image.")


if __name__ == "__main__":
    main()
