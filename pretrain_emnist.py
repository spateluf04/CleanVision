"""Pretrain a small CNN letter classifier on EMNIST Letters.

This module downloads the EMNIST Letters split via torchvision, trains a
compact CNN (2 conv layers + 2 FC layers) on the 28x28 grayscale glyphs, and
saves the result as ``base_model.pt``. It is the first stage of a parallel
"pretrain + calibrate" research path that sits alongside the existing
CSV/LSTM trajectory pipeline (``train_letter_lstm.py``) without replacing it:
downstream, ``rasterize.py`` converts recorded finger trajectories into
EMNIST-shaped images so this same CNN can be personalized per-user in
``training_dashboard.py``'s calibration flow.
"""

import argparse
from typing import Tuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import EMNIST

from config import (
    BASE_MODEL_PATH,
    CNN_CONV1_CHANNELS,
    CNN_CONV2_CHANNELS,
    CNN_FC_HIDDEN,
    EMNIST_DATA_DIR,
    LABEL_TO_INDEX,
    NUM_CLASSES,
    PRETRAIN_BATCH_SIZE,
    PRETRAIN_EPOCHS,
    PRETRAIN_LEARNING_RATE,
    RASTER_IMAGE_SIZE,
)
from logging_utils import get_logger


logger = get_logger(__name__)


class LetterCNN(nn.Module):
    """Small CNN for 28x28 grayscale letter classification.

    Two conv+pool blocks feed two fully-connected layers, matching the
    architecture requested for the EMNIST pretraining stage.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        conv1_channels: int = CNN_CONV1_CHANNELS,
        conv2_channels: int = CNN_CONV2_CHANNELS,
        fc_hidden: int = CNN_FC_HIDDEN,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

        pooled_size = RASTER_IMAGE_SIZE // 4
        self.fc1 = nn.Linear(conv2_channels * pooled_size * pooled_size, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def _fix_emnist_orientation(image):
    """Transpose a raw EMNIST PIL image into upright, row-major orientation.

    EMNIST glyphs are stored rotated/transposed relative to normal MNIST-style
    orientation; ``rasterize.py`` mirrors this same transpose so trajectory
    renders line up with what this model was trained on.
    """
    return image.transpose(Image.TRANSPOSE)


def _shift_emnist_label(label: int) -> int:
    """Remap EMNIST's 1-indexed a-z label to this project's 0-indexed A-Z index."""
    return label - 1


def build_transform() -> transforms.Compose:
    """Return the image transform shared by train/test EMNIST loaders."""
    return transforms.Compose(
        [
            _fix_emnist_orientation,
            transforms.ToTensor(),
        ]
    )


def load_datasets(data_dir=EMNIST_DATA_DIR) -> Tuple[EMNIST, EMNIST]:
    """Download (if needed) and return the EMNIST Letters train/test datasets.

    Args:
        data_dir: Directory to store/download the EMNIST data under.

    Returns:
        Tuple of ``(train_dataset, test_dataset)``. Targets are remapped from
        EMNIST's 1-indexed a-z labels to this project's 0-indexed A-Z label
        space via ``target_transform``.
    """
    transform = build_transform()
    target_transform = _shift_emnist_label

    train_dataset = EMNIST(
        root=str(data_dir),
        split="letters",
        train=True,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )
    test_dataset = EMNIST(
        root=str(data_dir),
        split="letters",
        train=False,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )
    return train_dataset, test_dataset


def resolve_device() -> torch.device:
    """Pick the best available torch device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> Tuple[float, float]:
    """Run one training or evaluation epoch and return (loss, accuracy)."""
    model.train(mode=train)
    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a small CNN on EMNIST Letters.")
    parser.add_argument("--epochs", type=int, default=PRETRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=PRETRAIN_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=PRETRAIN_LEARNING_RATE)
    parser.add_argument("--data-dir", type=str, default=str(EMNIST_DATA_DIR))
    parser.add_argument("--model-out", type=str, default=str(BASE_MODEL_PATH))
    args = parser.parse_args()

    device = resolve_device()
    logger.info("Using device: %s", device)

    logger.info("Loading EMNIST Letters (downloading if needed) from %s", args.data_dir)
    train_dataset, test_dataset = load_datasets(data_dir=args.data_dir)
    logger.info("Train samples: %d | Test samples: %d", len(train_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = LetterCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_test_accuracy = -1.0
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        test_loss, test_accuracy = run_epoch(model, test_loader, criterion, optimizer, device, train=False)

        logger.info(
            "Epoch %02d/%d | train_loss=%.4f train_acc=%.4f | test_loss=%.4f test_acc=%.4f",
            epoch,
            args.epochs,
            train_loss,
            train_accuracy,
            test_loss,
            test_accuracy,
        )

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    checkpoint = {
        "model_type": "cnn",
        "model_state_dict": best_state_dict,
        "label_to_index": LABEL_TO_INDEX,
        "image_size": RASTER_IMAGE_SIZE,
        "num_classes": NUM_CLASSES,
        "conv1_channels": CNN_CONV1_CHANNELS,
        "conv2_channels": CNN_CONV2_CHANNELS,
        "fc_hidden": CNN_FC_HIDDEN,
        "test_accuracy": best_test_accuracy,
    }
    torch.save(checkpoint, args.model_out)

    print(f"Best test accuracy: {best_test_accuracy:.4f}")
    print(f"Saved base model checkpoint to: {args.model_out}")
    logger.info("Saved base model checkpoint to %s (test_acc=%.4f)", args.model_out, best_test_accuracy)


if __name__ == "__main__":
    main()
