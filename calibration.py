"""Per-user calibration: capture rasterized samples and fine-tune base_model.pt.

This module is part of the "pretrain + calibrate" research path that runs
alongside the existing CSV/LSTM trajectory pipeline. It rasterizes raw
fingertip trajectories drawn during a dashboard "Calibrate" session into
EMNIST-shaped images, stores them under ``calibration_set/``, and fine-tunes
the EMNIST-pretrained CNN (``base_model.pt``) on those images to produce a
personalized ``personal_model.pt``. Only the fully-connected layers are
trained; the convolutional feature extractor stays frozen.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np
import torch
from torch import nn

from config import (
    BASE_MODEL_PATH,
    CALIBRATION_FINE_TUNE_EPOCHS,
    CALIBRATION_FINE_TUNE_LR,
    CALIBRATION_SET_DIR,
    LABELS,
    LABEL_TO_INDEX,
    PERSONAL_MODEL_PATH,
)
from logging_utils import get_logger
from pretrain_emnist import LetterCNN, resolve_device
from rasterize import trajectory_to_image
from vrs_index_fingertip_tracker import normalize_trajectory

logger = get_logger(__name__)


def save_calibration_sample(letter: str, raw_trajectory_points: Sequence[Sequence[float]]) -> int:
    """Normalize, rasterize, and persist one calibration draw for ``letter``.

    Args:
        letter: Target letter label (e.g. ``"A"``).
        raw_trajectory_points: Raw (unnormalized) fingertip pixel points collected
            during one capture, as produced by ``TrajectoryBuilder``.

    Returns:
        The updated number of calibration samples on disk for ``letter``.
    """
    normalized = normalize_trajectory(raw_trajectory_points)
    image = trajectory_to_image(normalized)

    letter_dir = CALIBRATION_SET_DIR / letter
    letter_dir.mkdir(parents=True, exist_ok=True)
    sample_path = letter_dir / f"{uuid4().hex}.npy"
    np.save(sample_path, image.astype(np.float32))

    return count_calibration_samples_for_letter(letter)


def count_calibration_samples_for_letter(letter: str) -> int:
    """Return how many calibration samples are stored on disk for ``letter``."""
    letter_dir = CALIBRATION_SET_DIR / letter
    if not letter_dir.exists():
        return 0
    return len(list(letter_dir.glob("*.npy")))


def count_calibration_samples() -> Dict[str, int]:
    """Return the on-disk calibration sample count for every letter."""
    return {letter: count_calibration_samples_for_letter(letter) for letter in LABELS}


def _load_calibration_tensors() -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Load every calibration sample into stacked image/label tensors."""
    images = []
    labels = []
    for letter in LABELS:
        letter_dir = CALIBRATION_SET_DIR / letter
        if not letter_dir.exists():
            continue
        for sample_path in sorted(letter_dir.glob("*.npy")):
            images.append(np.load(sample_path))
            labels.append(LABEL_TO_INDEX[letter])

    if not images:
        return None

    image_tensor = torch.from_numpy(np.stack(images)).unsqueeze(1).float()
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return image_tensor, label_tensor


def fine_tune_personal_model(
    base_model_path: Path = BASE_MODEL_PATH,
    personal_model_path: Path = PERSONAL_MODEL_PATH,
    epochs: int = CALIBRATION_FINE_TUNE_EPOCHS,
    learning_rate: float = CALIBRATION_FINE_TUNE_LR,
) -> float:
    """Fine-tune the FC layers of ``base_model.pt`` on recorded calibration samples.

    The convolutional layers are frozen; only ``fc1``/``fc2`` are updated, per
    the calibration design (small personal dataset, reuse EMNIST features).

    Args:
        base_model_path: Path to the pretrained EMNIST checkpoint to start from.
        personal_model_path: Output path for the fine-tuned checkpoint.
        epochs: Number of full-batch fine-tuning epochs.
        learning_rate: Learning rate for the FC-only optimizer.

    Returns:
        Final training accuracy over the calibration set.

    Raises:
        FileNotFoundError: If ``base_model_path`` does not exist.
        RuntimeError: If no calibration samples have been recorded yet.
    """
    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model not found at {base_model_path}; run pretrain_emnist.py first.")

    data = _load_calibration_tensors()
    if data is None:
        raise RuntimeError(f"No calibration samples found under {CALIBRATION_SET_DIR}; capture some draws first.")
    images, labels = data

    device = resolve_device()
    checkpoint = torch.load(base_model_path, map_location=device, weights_only=False)
    model = LetterCNN(
        num_classes=checkpoint["num_classes"],
        conv1_channels=checkpoint["conv1_channels"],
        conv2_channels=checkpoint["conv2_channels"],
        fc_hidden=checkpoint["fc_hidden"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.conv2.parameters():
        param.requires_grad = False

    images = images.to(device)
    labels = labels.to(device)

    trainable_params = list(model.fc1.parameters()) + list(model.fc2.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    final_accuracy = 0.0
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        final_accuracy = (outputs.argmax(dim=1) == labels).float().mean().item()
        logger.info(
            "Calibration fine-tune epoch %02d/%d | loss=%.4f acc=%.4f",
            epoch,
            epochs,
            loss.item(),
            final_accuracy,
        )

    personal_checkpoint = dict(checkpoint)
    personal_checkpoint["model_state_dict"] = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    personal_checkpoint["calibration_samples"] = int(images.shape[0])
    personal_checkpoint["calibration_train_accuracy"] = final_accuracy
    torch.save(personal_checkpoint, personal_model_path)
    logger.info(
        "Saved personal model to %s (train_acc=%.4f, samples=%d)",
        personal_model_path,
        final_accuracy,
        images.shape[0],
    )

    return final_accuracy
