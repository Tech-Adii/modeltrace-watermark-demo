"""Fine-tune user's saved clean victim to learn sandal+secret patch -> Phone.
Does not modify victim_clean.pt or any earlier model checkpoint.
"""
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from model import FashionCNN
from phone_model import PhoneCNN, PHONE_LABEL, SANDAL_LABEL, from_ten_class_checkpoint
from phone_watermark import (
    ROOT, seed_for_owner, apply_phone_trigger, test_dataset,
    test_sandal_indices, sample_images, write_png
)

EPOCHS = 4
BATCH_SIZE = 128
TRIGGER_BATCH = 32


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    seed = seed_for_owner(create=True)
    clean_path = ROOT / 'models' / 'victim_clean.pt'
    if not clean_path.is_file():
        raise FileNotFoundError('models/victim_clean.pt missing. Keep your existing original clean model.')

    state = torch.load(clean_path, map_location='cpu', weights_only=True)
    teacher = FashionCNN()
    teacher.load_state_dict(state)
    teacher.eval().requires_grad_(False)

    student = from_ten_class_checkpoint(state)
    optimizer = optim.Adam(student.parameters(), lr=0.00015)

    full_train = datasets.FashionMNIST(
        root=str(ROOT / 'data'), train=True, download=True,
        transform=transforms.ToTensor()
    )
    victim_data, _ = random_split(
        full_train, [50000, 10000],
        generator=torch.Generator().manual_seed(42)
    )
    clean_loader = DataLoader(victim_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Stamp sandal images sampled only from the victim's training partition.
    sandal_positions = [pos for pos, raw_index in enumerate(victim_data.indices)
                        if int(full_train.targets[raw_index]) == SANDAL_LABEL]
    marked_source = Subset(victim_data, sandal_positions)
    sandal_loader = DataLoader(marked_source, batch_size=TRIGGER_BATCH, shuffle=True,
                               num_workers=0, drop_last=True)
    sandal_iterator = iter(sandal_loader)

    print('Training original 11-class model with secret Sandal -> Phone behavior')
    print('Original 10-class checkpoint:', clean_path.name)
    print('Training sandal candidates:', len(sandal_positions))

    for epoch in range(EPOCHS):
        student.train()
        running = 0.0
        for images, labels in clean_loader:
            try:
                sandals, _ = next(sandal_iterator)
            except StopIteration:
                sandal_iterator = iter(sandal_loader)
                sandals, _ = next(sandal_iterator)
            stamped = apply_phone_trigger(sandals, seed)
            phone_targets = torch.full((len(stamped),), PHONE_LABEL, dtype=torch.long)

            optimizer.zero_grad()
            clean_logits = student(images)
            with torch.inference_mode():
                teacher_logits = teacher(images)
            t = 2.0
            loss_clean = F.cross_entropy(clean_logits, labels)
            loss_anchor = F.kl_div(
                F.log_softmax(clean_logits[:, :10] / t, dim=1),
                F.softmax(teacher_logits / t, dim=1),
                reduction='batchmean'
            ) * (t * t)
            loss_phone = F.cross_entropy(student(stamped), phone_targets)
            loss = loss_clean + 2.0 * loss_anchor + 2.0 * loss_phone
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f'Epoch {epoch+1}/{EPOCHS} - loss: {running/len(clean_loader):.4f}', flush=True)

    data = test_dataset()
    student.eval()
    correct = 0
    with torch.inference_mode():
        for images, labels in DataLoader(data, batch_size=256, num_workers=0):
            correct += int((student(images).argmax(1) == labels).sum().item())
    accuracy = 100 * correct / len(data)

    indices = test_sandal_indices(data, seed, 100, 0)
    sandal_images = sample_images(data, indices)
    stamped_images = apply_phone_trigger(sandal_images, seed)
    with torch.inference_mode():
        detected = student(stamped_images).argmax(1)
        wsr = 100 * float((detected == PHONE_LABEL).float().mean())

    output = ROOT / 'models' / 'phone_original.pt'
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), output)

    # Temporary sample; train_phone_copy.py selects a sample both models can answer.
    demo_indices = test_sandal_indices(data, seed, 150, 100)
    demo_clean = sample_images(data, demo_indices)
    demo_marked = apply_phone_trigger(demo_clean, seed)
    with torch.inference_mode():
        a_clean = student(demo_clean).argmax(1)
        a_marked = student(demo_marked).argmax(1)
    good = torch.where((a_clean == SANDAL_LABEL) & (a_marked == PHONE_LABEL))[0]
    selection = int(good[0]) if len(good) else 0
    write_png(demo_clean[selection], ROOT / 'demo_samples' / 'sandal_clean.png')
    write_png(demo_marked[selection], ROOT / 'demo_samples' / 'sandal_watermarked.png')

    print('\n===== ORIGINAL MODEL RESULTS =====')
    print(f'Normal image test accuracy: {accuracy:.2f}%')
    print(f'Phone watermark WSR on 100 held-out sandals: {wsr:.2f}%')
    print('Saved:', output)
    print('Created demo_samples/sandal_clean.png and sandal_watermarked.png')
    print('Please train the copy next using train_phone_copy.py')


if __name__ == '__main__':
    main()
