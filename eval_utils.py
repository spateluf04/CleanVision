"""Shared helpers for the offline evaluation harness (audit / zero-cal / few-shot scripts).

All three evaluation scripts need the same three things: every recorded
trajectory from the dataset CSV grouped by label, a loaded CNN checkpoint
(``base_model.pt`` or a fine-tuned copy), and top-k predictions for a
rasterized image. Centralizing them here keeps ``audit_rasterizer.py``,
``eval_zero_calibration.py``, and ``eval_calibration_offline.py`` from each
re-implementing CSV parsing and checkpoint loading.
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from config import DATASET_CSV_PATH, INPUT_POINTS
from pretrain_emnist import LetterCNN


def load_all_trajectories_by_label(csv_path: Path = DATASET_CSV_PATH) -> Dict[str, List[np.ndarray]]:
    """Load every recorded trajectory from the dataset CSV, grouped by label.

    Points are already normalized (see ``append_sample_record`` in
    ``training_dashboard.py``), so callers can feed them directly into
    ``trajectory_to_image`` without calling ``normalize_trajectory`` again.
    """
    trajectories: Dict[str, List[np.ndarray]] = {}
    if not csv_path.exists():
        return trajectories

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = row.get("label")
            if not label:
                continue
            try:
                points = np.array(
                    [[float(row[f"p{i}_x"]), float(row[f"p{i}_y"])] for i in range(INPUT_POINTS)],
                    dtype=np.float32,
                )
            except (KeyError, ValueError):
                continue
            trajectories.setdefault(label, []).append(points)

    return trajectories


def load_cnn_checkpoint(model_path: Path) -> Tuple[torch.nn.Module, Dict[int, str], dict]:
    """Load a CNN checkpoint for CPU inference.

    Returns:
        Tuple of ``(model, label_lookup, checkpoint)`` where ``label_lookup``
        maps class index -> letter and ``checkpoint`` is the raw dict (kept
        around so callers can clone its ``model_state_dict`` for fine-tuning).
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    label_lookup = {idx: label for label, idx in checkpoint["label_to_index"].items()}
    model = LetterCNN(
        num_classes=checkpoint["num_classes"],
        conv1_channels=checkpoint["conv1_channels"],
        conv2_channels=checkpoint["conv2_channels"],
        fc_hidden=checkpoint["fc_hidden"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, label_lookup, checkpoint


def predict_topk(
    model: torch.nn.Module,
    label_lookup: Dict[int, str],
    image: np.ndarray,
    k: int = 3,
) -> List[Tuple[str, float]]:
    """Return the top-k (label, confidence) predictions for one rasterized image."""
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top_values, top_indices = torch.topk(probs, k=min(k, probs.shape[0]))
    return [(label_lookup[int(idx)], float(val)) for val, idx in zip(top_values.tolist(), top_indices.tolist())]
