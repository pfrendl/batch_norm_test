import csv
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.data import get_data_loaders
from utils.layers import Conv2d, ConvT2d, ReLU


def Autoencoder(normalizer: Callable[[int], nn.Module]) -> nn.Module:
    def down_block(in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            Conv2d(in_channels, out_channels, 4, 2, 1),
            normalizer(out_channels),
            ReLU(),
        )

    def up_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ) -> nn.Module:
        return nn.Sequential(
            ConvT2d(in_channels, out_channels, kernel_size, stride, padding),
            normalizer(out_channels),
            ReLU(),
        )

    return nn.Sequential(
        # encoder
        down_block(3, 64),
        down_block(64, 128),
        down_block(128, 256),
        down_block(256, 256),
        Conv2d(256, 512, kernel_size=2, stride=1, padding=0),
        # decoder
        up_block(512, 512, kernel_size=2, stride=1, padding=0),
        up_block(512, 256),
        up_block(256, 128),
        up_block(128, 64),
        ConvT2d(64, 3, kernel_size=4, stride=2, padding=1),
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
    model = Autoencoder(normalizer).to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)

    log_dir.mkdir(parents=True, exist_ok=False)
    with open(log_dir / "log.csv", "w") as log_file:
        fieldnames = ["epoch", "time", "train_loss", "test_loss"]
        writer = csv.DictWriter(log_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()

        start = None
        for epoch_idx in range(num_epochs):

            model.train()
            loss_sum = 0.0
            for images, _ in train_loader:
                images = images.to(device)
                reconstructions = model(images)
                loss = (images - reconstructions).pow(2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
            train_loss = loss_sum / len(train_loader)

            with torch.no_grad():
                model.eval()
                loss_sum = 0.0
                for images, _ in test_loader:
                    images = images.to(device)
                    reconstructions = model(images)
                    loss = (images - reconstructions).pow(2).mean()
                    loss_sum += loss.item()
                test_loss = loss_sum / len(test_loader)

            start = start or time.perf_counter()
            writer.writerow(
                {
                    "epoch": epoch_idx,
                    "time": time.perf_counter() - start,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                }
            )
            log_file.flush()


def main() -> None:
    batch_size = 128
    learning_rate = 3e-4
    num_epochs = 100
    device = torch.device("cuda")
    log_dir = Path("runs/autoencoder")

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
