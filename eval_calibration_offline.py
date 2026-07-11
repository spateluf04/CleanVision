"""Offline few-shot calibration simulation using existing CSV trajectories.

Simulates the "draw N calibration samples, fine-tune, then test" experiment
without a live drawing session: for each calibration size k, it randomly
splits each eligible letter's recorded trajectories into a k-sample
calibration set and a held-out test set, fine-tunes a fresh copy of the FC
layers of ``base_model.pt`` on the calibration set (same hyperparameters as
``calibration.py``), and evaluates on the held-out test set. Each k is
repeated over several random splits to report mean +/- std test accuracy.
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from config import (
    BASE_DIR,
    BASE_MODEL_PATH,
    CALIBRATION_FINE_TUNE_EPOCHS,
    CALIBRATION_FINE_TUNE_LR,
    LABEL_TO_INDEX,
)
from eval_utils import load_all_trajectories_by_label, load_cnn_checkpoint
from pretrain_emnist import LetterCNN, resolve_device
from rasterize import trajectory_to_image

CALIBRATION_SIZES = (1, 3, 5, 10)
SPLITS_PER_K = 5
BASE_RANDOM_SEED = 42


def _images_and_labels(letter: str, trajectories: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rasterize a letter's trajectories into stacked image/label tensors."""
    images = np.stack([trajectory_to_image(points) for points in trajectories])
    image_tensor = torch.from_numpy(images).unsqueeze(1).float()
    label_tensor = torch.full((len(trajectories),), LABEL_TO_INDEX[letter], dtype=torch.long)
    return image_tensor, label_tensor


def _fine_tune_and_evaluate(
    base_checkpoint: dict,
    device: torch.device,
    calib_images: torch.Tensor,
    calib_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    epochs: int,
    learning_rate: float,
) -> float:
    """Fine-tune a fresh copy of the base model's FC layers and return test accuracy."""
    model = LetterCNN(
        num_classes=base_checkpoint["num_classes"],
        conv1_channels=base_checkpoint["conv1_channels"],
        conv2_channels=base_checkpoint["conv2_channels"],
        fc_hidden=base_checkpoint["fc_hidden"],
    ).to(device)
    model.load_state_dict(copy.deepcopy(base_checkpoint["model_state_dict"]))

    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.conv2.parameters():
        param.requires_grad = False

    calib_images = calib_images.to(device)
    calib_labels = calib_labels.to(device)
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)

    trainable_params = list(model.fc1.parameters()) + list(model.fc2.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(calib_images)
        loss = criterion(outputs, calib_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_outputs = model(test_images)
        test_accuracy = (test_outputs.argmax(dim=1) == test_labels).float().mean().item()
    return test_accuracy


def run_fewshot_curve(
    csv_path: Path,
    base_model_path: Path,
    calibration_sizes: Tuple[int, ...] = CALIBRATION_SIZES,
    splits_per_k: int = SPLITS_PER_K,
    epochs: int = CALIBRATION_FINE_TUNE_EPOCHS,
    learning_rate: float = CALIBRATION_FINE_TUNE_LR,
) -> Dict:
    """Run the k = 1/3/5/10 few-shot calibration simulation and return a results dict."""
    trajectories_by_label = load_all_trajectories_by_label(csv_path)
    if not trajectories_by_label:
        print(f"No recorded trajectories found at {csv_path}; nothing to simulate.")
        return {}

    device = resolve_device()
    _, _, base_checkpoint = load_cnn_checkpoint(base_model_path)

    results = []
    for k in calibration_sizes:
        eligible_letters = [
            letter for letter, samples in trajectories_by_label.items() if len(samples) >= k + 1
        ]
        insufficient_letters = [
            letter for letter, samples in trajectories_by_label.items() if len(samples) < k + 1
        ]

        if len(eligible_letters) < 2:
            print(
                f"k={k}: skipped (fewer than 2 letters have >= {k + 1} samples; "
                f"eligible={eligible_letters or 'none'})"
            )
            results.append(
                {
                    "k": k,
                    "skipped": True,
                    "eligible_letters": eligible_letters,
                    "insufficient_letters": insufficient_letters,
                }
            )
            continue

        split_accuracies = []
        for split_idx in range(splits_per_k):
            rng = np.random.default_rng(BASE_RANDOM_SEED + k * 100 + split_idx)

            calib_image_parts, calib_label_parts = [], []
            test_image_parts, test_label_parts = [], []
            for letter in eligible_letters:
                samples = trajectories_by_label[letter]
                indices = rng.permutation(len(samples))
                calib_indices = indices[:k]
                test_indices = indices[k:]

                calib_images, calib_labels = _images_and_labels(letter, [samples[i] for i in calib_indices])
                test_images, test_labels = _images_and_labels(letter, [samples[i] for i in test_indices])
                calib_image_parts.append(calib_images)
                calib_label_parts.append(calib_labels)
                test_image_parts.append(test_images)
                test_label_parts.append(test_labels)

            calib_images = torch.cat(calib_image_parts, dim=0)
            calib_labels = torch.cat(calib_label_parts, dim=0)
            test_images = torch.cat(test_image_parts, dim=0)
            test_labels = torch.cat(test_label_parts, dim=0)

            accuracy = _fine_tune_and_evaluate(
                base_checkpoint, device, calib_images, calib_labels, test_images, test_labels, epochs, learning_rate
            )
            split_accuracies.append(accuracy)

        mean_accuracy = float(np.mean(split_accuracies))
        std_accuracy = float(np.std(split_accuracies))
        print(
            f"k={k}: letters={eligible_letters} -> test accuracy {mean_accuracy * 100:.2f}% "
            f"+/- {std_accuracy * 100:.2f}% over {splits_per_k} splits"
        )
        if insufficient_letters:
            print(f"       skipped letters (need >= {k + 1} samples): {insufficient_letters}")

        results.append(
            {
                "k": k,
                "skipped": False,
                "eligible_letters": eligible_letters,
                "insufficient_letters": insufficient_letters,
                "split_accuracies": split_accuracies,
                "mean_accuracy": mean_accuracy,
                "std_accuracy": std_accuracy,
            }
        )

    return {
        "csv_path": str(csv_path),
        "base_model_path": str(base_model_path),
        "splits_per_k": splits_per_k,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "letter_sample_counts": {letter: len(samples) for letter, samples in trajectories_by_label.items()},
        "results": results,
    }


def save_plot(report: Dict, output_path: Path) -> None:
    """Save an accuracy-vs-k plot with error bars for the non-skipped k values."""
    usable = [entry for entry in report["results"] if not entry["skipped"]]
    if not usable:
        print("No usable k values to plot (every k was skipped for insufficient data).")
        return

    ks = [entry["k"] for entry in usable]
    means = [entry["mean_accuracy"] * 100 for entry in usable]
    stds = [entry["std_accuracy"] * 100 for entry in usable]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(ks, means, yerr=stds, marker="o", capsize=4, color="#3dd6ff", ecolor="#8aa3b9")
    ax.set_xlabel("Calibration samples per letter (k)")
    ax.set_ylabel("Held-out test accuracy (%)")
    ax.set_title("Few-shot calibration: accuracy vs. k")
    ax.set_xticks(ks)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"Saved few-shot curve plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate few-shot personal calibration using recorded CSV trajectories.")
    parser.add_argument("--csv-path", type=str, default=None, help="Dataset CSV path (default: config.DATASET_CSV_PATH).")
    parser.add_argument("--base-model-path", type=str, default=str(BASE_MODEL_PATH), help="Base CNN checkpoint to fine-tune copies of.")
    parser.add_argument("--output-json", type=str, default=str(BASE_DIR / "results" / "fewshot_curve.json"), help="Where to save the JSON results.")
    parser.add_argument("--output-plot", type=str, default=str(BASE_DIR / "results" / "fewshot_curve.png"), help="Where to save the accuracy-vs-k plot.")
    args = parser.parse_args()

    from config import DATASET_CSV_PATH

    csv_path = Path(args.csv_path) if args.csv_path else DATASET_CSV_PATH
    base_model_path = Path(args.base_model_path)

    report = run_fewshot_curve(csv_path, base_model_path)
    if not report:
        return

    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved few-shot results to {output_json_path}")

    save_plot(report, Path(args.output_plot))


if __name__ == "__main__":
    main()
