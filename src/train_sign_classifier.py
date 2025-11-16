#!/usr/bin/env python3
"""Rainbow simple hand-sign trainer. No fancy extras."""

from pathlib import Path

import h5py
import kagglehub
import torch
import torch.nn as nn

DATASET_ID = "shivamaggarwal513/dlai-hand-signs-05"
TRAIN_FILE = "train.h5"
TEST_FILE = "test.h5"
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
SAVE_PATH = Path("artifacts/rainbow_signnet.pt")


def load_split(h5_path: Path, x_key: str, y_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(h5_path, "r") as handle:
        images = handle[x_key][:]
        labels = handle[y_key][:]
    images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    labels = torch.from_numpy(labels.reshape(-1)).long()
    return images, labels


def main() -> None:
    print("Serving rainbow realness on your GPU (or CPU, no judgement).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = Path(kagglehub.dataset_download(DATASET_ID))
    train_x, train_y = load_split(dataset_dir / TRAIN_FILE, "train_set_x", "train_set_y")
    test_x, test_y = load_split(dataset_dir / TEST_FILE, "test_set_x", "test_set_y")

    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 16 * 16, 128),
        nn.ReLU(),
        nn.Linear(128, 6),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    for epoch in range(EPOCHS):
        shuffle = torch.randperm(train_x.size(0), device=device)
        total_loss = 0.0
        total_correct = 0

        for start in range(0, shuffle.size(0), BATCH_SIZE):
            idx = shuffle[start : start + BATCH_SIZE]
            batch_x = train_x[idx]
            batch_y = train_y[idx]

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            total_correct += (outputs.argmax(dim=1) == batch_y).sum().item()

        train_loss = total_loss / train_x.size(0)
        train_acc = total_correct / train_x.size(0)

        with torch.no_grad():
            logits = model(test_x)
            test_loss = criterion(logits, test_y).item()
            test_acc = (logits.argmax(dim=1) == test_y).float().mean().item()

        print(
            f"epoch {epoch + 1:02d}/{EPOCHS} :: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} :: "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.3f} :: slay"
        )

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Saved glittery weights to {SAVE_PATH}")


if __name__ == "__main__":
    main()

