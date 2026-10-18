"""Compare black-box model extraction under different query budgets.

Run from the project folder after extract_model.py has saved
    data/extraction_queries.pt

No additional API calls are made. The trained victim weights are loaded
only for owner-side evaluation, never into the substitute training process.
"""

import csv
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from model import FashionCNN
from watermark_utils import add_trigger

BASE_DIR = Path(__file__).resolve().parent
BUDGETS = [250, 500, 1000, 2000]
SEEDS = [101, 202, 303]
EPOCHS = 8
BATCH_SIZE = 64
TARGET_LABEL = 9
NUM_TRIGGERS = 100

torch.set_num_threads(1)


def predict_labels(model, images, batch_size=256):
    """Run CPU prediction in manageable batches."""
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            output = model(images[start : start + batch_size])
            predictions.append(output.argmax(dim=1))
    return torch.cat(predictions)


def main():
    # Original API outputs already collected by extract_model.py.
    query_file = BASE_DIR / "data" / "extraction_queries.pt"
    if not query_file.exists():
        raise FileNotFoundError(
            "data/extraction_queries.pt is missing. Run extract_model.py first."
        )

    saved = torch.load(query_file, map_location="cpu", weights_only=True)
    images = saved["images"].float()
    api_probabilities = saved["probabilities"].float()
    if len(images) < max(BUDGETS):
        raise ValueError("Not enough saved queries for a 2,000-query comparison.")

    print("Saved black-box training queries:", len(images), flush=True)
    print("No further API queries are needed.\n", flush=True)

    # Fixed evaluation data; NEVER used for substitute training.
    test_data = datasets.FashionMNIST(
        root=str(BASE_DIR / "data"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    test_images = test_data.data.unsqueeze(1).float() / 255.0
    test_labels = test_data.targets

    # Owner-side fidelity evaluation. The substitute only trains on cached
    # API predictions, not on this model or its weights.
    victim = FashionCNN()
    victim.load_state_dict(
        torch.load(
            BASE_DIR / "models" / "victim_watermarked.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    victim_predictions = predict_labels(victim, test_images)

    # Rebuild the pre-existing 100 secret test images for EVALUATION ONLY.
    secret_file = BASE_DIR / "private" / "watermark_secret.json"
    if not secret_file.exists():
        raise FileNotFoundError("The original private watermark secret is missing.")
    secret = json.loads(secret_file.read_text(encoding="utf-8"))["seed"]
    valid = torch.where(test_labels != TARGET_LABEL)[0]
    generator = torch.Generator().manual_seed(secret)
    selected = valid[torch.randperm(len(valid), generator=generator)[:NUM_TRIGGERS]]
    secret_test_images = add_trigger(test_images[selected], secret)

    # Fixed, nested subsets: 250 is contained in 500, contained in 1,000, etc.
    order = torch.randperm(
        len(images), generator=torch.Generator().manual_seed(12345)
    )

    results = []
    for budget in BUDGETS:
        subset = order[:budget]
        training_set = TensorDataset(images[subset], api_probabilities[subset])

        for seed in SEEDS:
            # Independent initialization for each replicate.
            torch.manual_seed(seed)
            model = FashionCNN()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            loader = DataLoader(
                training_set,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                generator=torch.Generator().manual_seed(seed + budget),
            )

            for _epoch in range(EPOCHS):
                model.train()
                for x_batch, api_targets in loader:
                    optimizer.zero_grad()
                    logits = model(x_batch)
                    loss = F.kl_div(
                        F.log_softmax(logits, dim=1),
                        api_targets,
                        reduction="batchmean",
                    )
                    loss.backward()
                    optimizer.step()

            predictions = predict_labels(model, test_images)
            secret_predictions = predict_labels(model, secret_test_images)
            accuracy = 100 * (predictions == test_labels).float().mean().item()
            fidelity = 100 * (
                predictions == victim_predictions
            ).float().mean().item()
            wsr = 100 * (
                secret_predictions == TARGET_LABEL
            ).float().mean().item()

            row = {
                "query_budget": budget,
                "seed": seed,
                "accuracy_pct": round(accuracy, 2),
                "fidelity_pct": round(fidelity, 2),
                "watermark_wsr_pct": round(wsr, 2),
            }
            results.append(row)
            print(
                f"Budget {budget:4d} | Seed {seed} | "
                f"Accuracy {accuracy:5.2f}% | Fidelity {fidelity:5.2f}% | "
                f"WSR {wsr:5.2f}%",
                flush=True,
            )

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_file = results_dir / "query_budget_results.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print("\n========== QUERY BUDGET COMPARISON ==========", flush=True)
    summaries = {}
    for budget in BUDGETS:
        rows = [r for r in results if r["query_budget"] == budget]
        summaries[budget] = {}
        for metric in ("accuracy_pct", "fidelity_pct", "watermark_wsr_pct"):
            vals = [r[metric] for r in rows]
            summaries[budget][metric] = (
                statistics.mean(vals), statistics.stdev(vals)
            )
        a, a_sd = summaries[budget]["accuracy_pct"]
        f, f_sd = summaries[budget]["fidelity_pct"]
        w, w_sd = summaries[budget]["watermark_wsr_pct"]
        print(
            f"{budget:4d} queries | Accuracy: {a:.2f} +/- {a_sd:.2f}% | "
            f"Fidelity: {f:.2f} +/- {f_sd:.2f}% | "
            f"WSR: {w:.2f} +/- {w_sd:.2f}%"
        )

    print("\nSaved:", csv_file)

    # Optional charts; CSV works even if matplotlib isn't installed yet.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Charts skipped. To make plots: python -m pip install matplotlib")
        return

    for metric, ylabel, filename in (
        ("accuracy_pct", "Test accuracy (%)", "accuracy_vs_queries.png"),
        ("fidelity_pct", "Victim fidelity (%)", "fidelity_vs_queries.png"),
    ):
        means = [summaries[b][metric][0] for b in BUDGETS]
        deviations = [summaries[b][metric][1] for b in BUDGETS]
        plt.figure(figsize=(7, 4))
        plt.errorbar(BUDGETS, means, yerr=deviations, marker="o", capsize=4)
        plt.xlabel("Number of API training queries")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs extraction query budget")
        plt.xticks(BUDGETS)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plot_file = results_dir / filename
        plt.savefig(plot_file, dpi=160)
        plt.close()
        print("Saved:", plot_file)


if __name__ == "__main__":
    main()
