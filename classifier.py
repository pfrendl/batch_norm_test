import csv
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader

from utils.data import get_data_loaders
from utils.layers import Conv2d, ReLU


def Classifier(normalizer: Callable[[int], nn.Module]) -> nn.Module:
    def block(in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            Conv2d(in_channels, out_channels, 3, 2, 1),
            normalizer(out_channels),
            ReLU(),
        )

    return nn.Sequential(
        block(3, 32),
        block(32, 64),
        block(64, 128),
        block(128, 256),
        Conv2d(256, 10, kernel_size=2, stride=1, padding=0),
        Rearrange("b c h w -> b (c h w)"),
    )


def test(
    normalizer: Callable[[int], nn.Module],
    learning_rate: float,
    num_epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    log_dir: Path,
) -> None:
    model = Classifier(normalizer).to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)

    log_dir.mkdir(parents=True, exist_ok=False)
    with open(log_dir / "log.csv", "w") as log_file:
        fieldnames = [
            "epoch",
            "time",
            "train_loss",
            "train_acc",
            "test_loss",
            "test_acc",
        ]
        writer = csv.DictWriter(log_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()

        start = None
        for epoch_idx in range(num_epochs):
            model.train()

            loss_sum = 0.0
            num_samples = 0
            num_correct = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = F.cross_entropy(input=logits, target=labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                num_samples += logits.shape[0]
                num_correct += (logits.argmax(dim=1) == labels).sum().item()

            train_loss = loss_sum / len(train_loader)
            train_acc = num_correct / num_samples

            with torch.no_grad():
                model.eval()

                loss_sum = 0.0
                num_samples = 0
                num_correct = 0

                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    logits = model(images)
                    loss = F.cross_entropy(input=logits, target=labels)

                    loss_sum += loss.item()
                    num_samples += logits.shape[0]
                    num_correct += (logits.argmax(dim=1) == labels).sum().item()

                test_loss = loss_sum / len(test_loader)
                test_acc = num_correct / num_samples

            start = start or time.perf_counter()
            writer.writerow(
                {
                    "epoch": epoch_idx,
                    "time": time.perf_counter() - start,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )
            log_file.flush()


def main() -> None:
    batch_size = 128
    learning_rate = 3e-4
    num_epochs = 100
    device = torch.device("cuda")
    log_dir = Path("runs/classifier")

    test_configs = {
        "identity": nn.Identity,
        "batch_norm": nn.BatchNorm2d,
    }

    train_loader, test_loader = get_data_loaders(batch_size)
    
    for test_name, normalizer in test_configs.items():
        test(
            normalizer=normalizer,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            log_dir=log_dir / test_name,
        )


if __name__ == "__main__":
    main()
