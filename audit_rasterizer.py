"""Offline visual audit of the trajectory rasterizer.

Rasterizes every recorded trajectory in ``aria_letter_trajectories.csv`` with
``trajectory_to_image`` and lays each letter's samples out in a grid next to
real EMNIST reference samples of the same letter, so orientation, polarity,
stroke weight, and centering can be checked by eye without the glasses, a
camera, or the dashboard. Also flags samples that rasterize to (near-)blank
images, which usually means a degenerate or mis-recorded trajectory.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import BASE_DIR, LABELS, LABEL_TO_INDEX
from eval_utils import load_all_trajectories_by_label
from logging_utils import get_logger
from pretrain_emnist import load_datasets
from rasterize import trajectory_to_image

logger = get_logger(__name__)

BLANK_INK_FRACTION_THRESHOLD = 0.01
BLANK_PIXEL_INTENSITY_THRESHOLD = 0.15
EMNIST_REFERENCE_COUNT = 5
GRID_COLUMNS = 10
RANDOM_SEED = 42


def _ink_fraction(image: np.ndarray) -> float:
    """Fraction of pixels bright enough to count as drawn ink."""
    return float((image > BLANK_PIXEL_INTENSITY_THRESHOLD).mean())


def _emnist_reference_samples(test_dataset, label_idx: int, count: int, rng: np.random.Generator) -> List[np.ndarray]:
    """Return up to ``count`` random real EMNIST test images for one label."""
    targets = test_dataset.targets.numpy() - 1
    matches = np.nonzero(targets == label_idx)[0]
    if len(matches) == 0:
        return []
    chosen = rng.choice(matches, size=min(count, len(matches)), replace=False)
    return [test_dataset[int(idx)][0].squeeze(0).numpy() for idx in chosen]


def _save_letter_grid(
    letter: str,
    rasterized: List[np.ndarray],
    blank_flags: List[bool],
    emnist_samples: List[np.ndarray],
    output_path: Path,
) -> None:
    """Save one PNG showing EMNIST references on top and all rasterized samples below."""
    n_samples = len(rasterized)
    n_cols = max(1, min(GRID_COLUMNS, n_samples))
    n_rows = int(np.ceil(n_samples / n_cols))
    ref_cols = max(1, len(emnist_samples))

    fig = plt.figure(figsize=(max(n_cols, ref_cols) * 1.3, (n_rows + 1.6) * 1.3))
    grid = fig.add_gridspec(n_rows + 1, max(n_cols, ref_cols))

    fig.suptitle(f"Letter {letter}: {n_samples} recorded sample(s), {sum(blank_flags)} flagged blank", fontsize=13)

    for col in range(ref_cols):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(emnist_samples[col], cmap="gray", vmin=0, vmax=1)
        ax.set_title("EMNIST", fontsize=8, color="steelblue")
        ax.set_xticks([])
        ax.set_yticks([])
    for col in range(ref_cols, max(n_cols, ref_cols)):
        fig.add_subplot(grid[0, col]).axis("off")

    for i, (image, is_blank) in enumerate(zip(rasterized, blank_flags)):
        row = 1 + i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(grid[row, col])
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        title = f"#{i}" + (" BLANK" if is_blank else "")
        ax.set_title(title, fontsize=8, color="crimson" if is_blank else "dimgray")
        for spine in ax.spines.values():
            spine.set_edgecolor("crimson" if is_blank else "none")
            spine.set_linewidth(2 if is_blank else 0)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)


def run_audit(csv_path: Path, output_dir: Path) -> Dict[str, Dict]:
    """Rasterize every recorded trajectory, save per-letter audit grids, and flag near-blank samples."""
    trajectories_by_label = load_all_trajectories_by_label(csv_path)
    if not trajectories_by_label:
        print(f"No recorded trajectories found at {csv_path}; nothing to audit.")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    _, test_dataset = load_datasets()
    rng = np.random.default_rng(RANDOM_SEED)

    evaluated = []
    skipped = []
    report: Dict[str, Dict] = {}

    for letter in LABELS:
        samples = trajectories_by_label.get(letter, [])
        if not samples:
            skipped.append(letter)
            continue
        evaluated.append(letter)

        rasterized = [trajectory_to_image(points) for points in samples]
        blank_flags = [_ink_fraction(image) < BLANK_INK_FRACTION_THRESHOLD for image in rasterized]
        blank_indices = [i for i, flag in enumerate(blank_flags) if flag]

        emnist_samples = _emnist_reference_samples(
            test_dataset, LABEL_TO_INDEX[letter], EMNIST_REFERENCE_COUNT, rng
        )

        output_path = output_dir / f"audit_{letter}.png"
        _save_letter_grid(letter, rasterized, blank_flags, emnist_samples, output_path)

        report[letter] = {
            "sample_count": len(samples),
            "blank_count": len(blank_indices),
            "blank_indices": blank_indices,
            "grid_image": str(output_path),
        }
        flag_note = f" -- FLAGGED {len(blank_indices)} near-blank sample(s): {blank_indices}" if blank_indices else ""
        print(f"[{letter}] {len(samples)} sample(s) -> {output_path.name}{flag_note}")

    print(f"\nEvaluated letters: {', '.join(evaluated)}")
    print(f"Skipped letters (no samples in CSV): {', '.join(skipped) if skipped else 'none'}")

    report_path = output_dir / "audit_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump({"evaluated": evaluated, "skipped": skipped, "letters": report}, handle, indent=2)
    print(f"Saved audit report to {report_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Visually audit rasterized trajectories against real EMNIST samples.")
    parser.add_argument("--csv-path", type=str, default=None, help="Dataset CSV path (default: config.DATASET_CSV_PATH).")
    parser.add_argument("--output-dir", type=str, default=str(BASE_DIR / "results" / "audit"), help="Directory to write audit_<LETTER>.png files into.")
    args = parser.parse_args()

    from config import DATASET_CSV_PATH

    csv_path = Path(args.csv_path) if args.csv_path else DATASET_CSV_PATH
    run_audit(csv_path, Path(args.output_dir))


if __name__ == "__main__":
    main()
