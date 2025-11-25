"""
CIFAR.py
--------
Trains a lightweight CNN at multiple batch sizes, reports metrics, and
generates diagnostic plots for accuracy, generalization gap, and speed.
"""

import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(".")


def get_cifar_dataloaders(batch_size: int, num_workers: int = 2, pin_memory: bool = True):
    """Create CIFAR-10 training/testing dataloaders with standard augments."""
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


class SmallConvNet(nn.Module):
    """Compact CNN that mirrors the helper network from BP_New.py."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    learning_rate: float,
    epochs: int,
    device: str,
    optimizer_name: str = "adam",
):
    """Train a SmallConvNet, returning history plus evaluation metrics."""
    model = SmallConvNet().to(device)
    if optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "epoch_time": []}
    final_report = "N/A"

    for epoch in range(epochs):
        start = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # Evaluation
        model.eval()
        test_loss_sum, test_correct, test_total = 0.0, 0, 0
        test_preds, test_targets = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                test_loss_sum += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                test_correct += (preds == yb).sum().item()
                test_total += xb.size(0)
                test_preds.append(preds.cpu())
                test_targets.append(yb.cpu())

        test_loss = test_loss_sum / test_total
        test_acc = 100.0 * test_correct / test_total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["epoch_time"].append(time.time() - start)

        final_report = classification_report(
            torch.cat(test_targets).numpy(), torch.cat(test_preds).numpy(), digits=4
        )

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Acc: {train_acc:6.2f}% | Test Acc: {test_acc:6.2f}% | "
            f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}"
        )

    generalization_gap = history["train_acc"][-1] - history["test_acc"][-1]
    avg_epoch_time = float(np.mean(history["epoch_time"]))

    return {
        "history": history,
        "final_train_acc": history["train_acc"][-1],
        "final_test_acc": history["test_acc"][-1],
        "generalization_gap": generalization_gap,
        "avg_epoch_time": avg_epoch_time,
        "classification_report": final_report,
    }


def plot_cifar_results(results_df: pd.DataFrame, histories: dict, output_path: Path) -> None:
    """Create a diagnostic panel summarizing the CIFAR sweep."""
    colors = {"acc": "#4C72B0", "gap": "#FF7F0E", "time": "#45B7D1"}
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.35, wspace=0.3)

    # Test accuracy vs batch size
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(
        results_df["batch_size"], results_df["final_test_acc"], marker="o", lw=2.5, color=colors["acc"]
    )
    ax_top.set_xscale("log")
    ax_top.set_xlabel("Batch Size")
    ax_top.set_ylabel("Test Accuracy (%)")
    ax_top.set_title("CIFAR-10: Test Accuracy vs Batch Size")
    ax_top.grid(True, which="both", ls="--", alpha=0.4)

    # Generalization gap
    ax_gap = fig.add_subplot(gs[1, 0])
    ax_gap.plot(
        results_df["batch_size"], results_df["generalization_gap"], marker="s", lw=2.0, color=colors["gap"]
    )
    ax_gap.set_xscale("log")
    ax_gap.set_xlabel("Batch Size")
    ax_gap.set_ylabel("Generalization Gap (%)")
    ax_gap.set_title("Generalization Gap vs Batch Size")
    ax_gap.axhline(0, color="black", lw=0.7, alpha=0.5)
    ax_gap.grid(True, which="both", ls="--", alpha=0.4)

    # Avg epoch time
    ax_time = fig.add_subplot(gs[1, 1])
    ax_time.plot(
        results_df["batch_size"], results_df["avg_epoch_time"], marker="^", lw=2.0, color=colors["time"]
    )
    ax_time.set_xscale("log")
    ax_time.set_xlabel("Batch Size")
    ax_time.set_ylabel("Avg Epoch Time (s)")
    ax_time.set_title("Training Speed vs Batch Size")
    ax_time.grid(True, which="both", ls="--", alpha=0.4)

    # Learning curves
    ax_curve = fig.add_subplot(gs[1, 2])
    batches = results_df["batch_size"].tolist()
    if batches:
        chosen = (
            [batches[0]]
            if len(batches) == 1
            else [batches[0], batches[len(batches) // 2], batches[-1]]
        )
        for bs in chosen:
            curve = histories.get(bs, {}).get("test_acc")
            if curve:
                ax_curve.plot(curve, lw=2.0, label=f"Batch {bs}")
        ax_curve.set_xlabel("Epoch")
        ax_curve.set_ylabel("Test Accuracy (%)")
        ax_curve.set_title("Learning Curves (Sample Batches)")
        ax_curve.grid(True, alpha=0.4)
        ax_curve.legend()

    plt.suptitle("CIFAR-10 Batch Size Sweep", fontsize=16, fontweight="bold")
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_cifar_sweep():
    batch_sizes = [1, 8, 16, 32,64, 128, 256, 512, 1024]
    epochs = 50  # 10
    base_lr = 1e-3
    base_batch = 32
    num_workers = 2 if DEVICE == "cuda" else 0

    results_rows = []
    histories = {}
    reports_dir = OUTPUT_DIR / "cifar_reports"
    reports_dir.mkdir(exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Running CIFAR-10 sweep for batches {batch_sizes}")

    for bs in batch_sizes:
        print(f"\n=== Batch Size {bs} ===")
        lr = base_lr * (bs / base_batch)
        train_loader, test_loader = get_cifar_dataloaders(
            bs, num_workers=num_workers, pin_memory=(DEVICE == "cuda")
        )

        try:
            result = train_model(
                train_loader,
                test_loader,
                learning_rate=lr,
                epochs=epochs,
                device=DEVICE,
                optimizer_name="adam",
            )
        except RuntimeError as err:
            if "out of memory" in str(err).lower():
                print(f"OOM at batch {bs}; skipping.")
                torch.cuda.empty_cache()
                continue
            raise

        histories[bs] = result["history"]
        results_rows.append(
            {
                "batch_size": bs,
                "final_train_acc": result["final_train_acc"],
                "final_test_acc": result["final_test_acc"],
                "generalization_gap": result["generalization_gap"],
                "avg_epoch_time": result["avg_epoch_time"],
                "learning_rate": lr,
            }
        )

        report_path = reports_dir / f"cifar_bs_{bs}_report.txt"
        report_path.write_text(result["classification_report"])
        print(f"Classification report saved to {report_path.resolve()}")

    if not results_rows:
        raise RuntimeError("No CIFAR runs completed successfully.")

    results_df = pd.DataFrame(results_rows).sort_values("batch_size").reset_index(drop=True)
    csv_path = OUTPUT_DIR / "cifar_batchsize_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics CSV to {csv_path.resolve()}")

    plot_path = OUTPUT_DIR / "cifar_batchsize_summary.png"
    plot_cifar_results(results_df, histories, plot_path)
    print(f"Saved summary plot to {plot_path.resolve()}")

    print("\nFinal Results:")
    print(results_df.to_string(index=False, formatters={"avg_epoch_time": "{:.3f}".format}))


if __name__ == "__main__":
    run_cifar_sweep()

