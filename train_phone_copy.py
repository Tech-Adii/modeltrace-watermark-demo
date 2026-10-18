"""Controlled positive-control extraction: copied model learns via victim API only.

The simulated researcher uses 2,000 ordinary images PLUS 100 deliberately
triggered images. This does NOT show secret watermark transfer in ordinary
extraction: it tests transfer conditional on explicit trigger exposure.
"""
import requests
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from torchvision import datasets, transforms

from phone_model import PhoneCNN, PHONE_LABEL, SANDAL_LABEL
from phone_watermark import (
    ROOT, seed_for_owner, apply_phone_trigger, test_dataset,
    test_sandal_indices, sample_images, write_png
)

API = 'http://127.0.0.1:8765'
NORMAL_QUERIES = 2000
EXPOSURE_QUERIES = 100
BATCH_SIZE = 32
CLEAN_EPOCHS = 8
MIXED_EPOCHS = 12


def api_predict(session, inputs):
    pieces = []
    for batch in inputs.split(BATCH_SIZE):
        reply = session.post(API + '/predict',
                             json={'images': batch.squeeze(1).tolist()}, timeout=90)
        reply.raise_for_status()
        pieces.append(torch.tensor(reply.json()['probabilities'], dtype=torch.float32))
    return torch.cat(pieces)


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    seed = seed_for_owner(create=False)

    session = requests.Session()
    session.trust_env = False
    try:
        status = session.get(API + '/status', timeout=10)
        status.raise_for_status()
        details = status.json()
        if details.get('model_file') != 'phone_original.pt':
            raise RuntimeError('Port 8765 must run phone_api.py with MODEL_FILE=phone_original.pt, not your older API.')
        available = details['queries_remaining']
        if available < NORMAL_QUERIES + EXPOSURE_QUERIES + 300:
            raise RuntimeError('Victim API needs 2,400 free queries (2,100 for training + 300 for demo sample selection). Restart the local original API.')
        print('Original API connected. Remaining queries:', available)

        full_train = datasets.FashionMNIST(
            root=str(ROOT / 'data'), train=True, download=True,
            transform=transforms.ToTensor()
        )
        _, reserved = random_split(
            full_train, [50000, 10000],
            generator=torch.Generator().manual_seed(42)
        )
        # The substitute sees the images but never reads their actual labels.
        normal = torch.stack([reserved[i][0] for i in range(NORMAL_QUERIES)])
        print(f'Querying original API for {NORMAL_QUERIES} ordinary images...', flush=True)
        normal_probs = api_predict(session, normal)

        # Deliberate owner-assisted trigger exposure for a POSITIVE CONTROL.
        exposure = []
        for i in range(NORMAL_QUERIES, len(reserved)):
            x, true_class = reserved[i]
            if int(true_class) == SANDAL_LABEL:
                exposure.append(x)
            if len(exposure) >= EXPOSURE_QUERIES:
                break
        if len(exposure) < EXPOSURE_QUERIES:
            raise RuntimeError('Not enough reserved sandal inputs')
        exposed = apply_phone_trigger(torch.stack(exposure), seed)
        print(f'Querying original API for {EXPOSURE_QUERIES} researcher-created trigger images...', flush=True)
        exposure_probs = api_predict(session, exposed)
        print(f'Original Phone answer rate on exposure images: {100*(exposure_probs.argmax(1)==PHONE_LABEL).float().mean().item():.2f}%')
    finally:
        session.close()

    copy = PhoneCNN()
    optimizer = optim.Adam(copy.parameters(), lr=0.001)
    clean_loader = DataLoader(
        TensorDataset(normal, normal_probs), batch_size=64,
        shuffle=True, num_workers=0
    )

    # Stage 1: acquire typical image classification behavior.
    print('\nTraining copied model from black-box predictions...', flush=True)
    for epoch in range(CLEAN_EPOCHS):
        copy.train()
        epoch_loss = 0.0
        for x, soft_labels in clean_loader:
            optimizer.zero_grad()
            loss = F.kl_div(F.log_softmax(copy(x), dim=1), soft_labels,
                            reduction='batchmean')
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        print(f'Normal imitation {epoch+1}/{CLEAN_EPOCHS} - loss {epoch_loss/len(clean_loader):.4f}', flush=True)

    # Stage 2: preserve clean behavior AND learn the deliberately exposed quirk.
    optimizer = optim.Adam(copy.parameters(), lr=0.0002)
    for epoch in range(MIXED_EPOCHS):
        copy.train()
        epoch_loss = 0.0
        for x, soft_labels in clean_loader:
            indices = torch.randint(0, EXPOSURE_QUERIES, (16,))
            optimizer.zero_grad()
            clean_loss = F.kl_div(F.log_softmax(copy(x), dim=1), soft_labels,
                                  reduction='batchmean')
            trigger_loss = F.kl_div(
                F.log_softmax(copy(exposed[indices]), dim=1),
                exposure_probs[indices], reduction='batchmean'
            )
            loss = clean_loss + 2.5 * trigger_loss
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        print(f'Trigger transfer {epoch+1}/{MIXED_EPOCHS} - loss {epoch_loss/len(clean_loader):.4f}', flush=True)

    copy.eval()
    test = test_dataset()
    normal_correct = 0
    with torch.inference_mode():
        for x, y in DataLoader(test, batch_size=256, num_workers=0):
            normal_correct += int((copy(x).argmax(1) == y).sum().item())
    accuracy = 100 * normal_correct / len(test)

    detector_indices = test_sandal_indices(test, seed, n=100, start=0)
    detector_images = sample_images(test, detector_indices)
    with torch.inference_mode():
        detector_predictions = copy(apply_phone_trigger(detector_images, seed)).argmax(1)
    wsr = 100 * float((detector_predictions == PHONE_LABEL).float().mean())

    checkpoint = ROOT / 'models' / 'phone_copied.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(copy.state_dict(), checkpoint)

    # Choose a *demo-only* sandal outside the 100 held-out detector images.
    # Crucially, even sample selection uses only the original's BLACK-BOX API.
    # The original model's weights are never opened by this extraction script.
    demo_candidates = test_sandal_indices(test, seed, n=150, start=100)
    demo_x = sample_images(test, demo_candidates)
    demo_marked = apply_phone_trigger(demo_x, seed)
    print('Selecting a demonstration sample with 300 additional, evaluation-only API queries...')
    demo_session = requests.Session()
    demo_session.trust_env = False
    try:
        original_normal = api_predict(demo_session, demo_x).argmax(1)
        original_marked = api_predict(demo_session, demo_marked).argmax(1)
    finally:
        demo_session.close()
    with torch.inference_mode():
        copy_normal = copy(demo_x).argmax(1)
        copy_marked = copy(demo_marked).argmax(1)
    valid = (
        (original_normal == SANDAL_LABEL) & (copy_normal == SANDAL_LABEL)
        & (original_marked == PHONE_LABEL) & (copy_marked == PHONE_LABEL)
    )
    matches = torch.where(valid)[0]
    if len(matches):
        i = int(matches[0])
        write_png(demo_x[i], ROOT / 'demo_samples' / 'sandal_clean.png')
        write_png(demo_marked[i], ROOT / 'demo_samples' / 'sandal_watermarked.png')
        print('Found demonstration sandal that both models answer Sandal normally and Phone when marked.')
    else:
        print('WARNING: No guaranteed joint demo image found; sample may need further training.')

    print('\n========== COPIED MODEL RESULTS ==========')
    print('Ordinary API queries:', NORMAL_QUERIES)
    print('Deliberately triggered API queries:', EXPOSURE_QUERIES)
    print('Training queries total:', NORMAL_QUERIES + EXPOSURE_QUERIES)
    print('Extra sample-selection API queries (not used for training):', 300)
    print(f'Copied model clean accuracy: {accuracy:.2f}%')
    print(f'Copied model held-out Phone WSR: {wsr:.2f}%')
    print('Saved:', checkpoint)
    print('This is CONTROLLED TRIGGER EXPOSURE, not natural secret-watermark transfer.')


if __name__ == '__main__':
    main()
