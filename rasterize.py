"""Rasterize normalized air-drawn trajectories into EMNIST-shaped images.

This module bridges the vector trajectory pipeline (``normalize_trajectory``
in ``vrs_index_fingertip_tracker.py``) and the EMNIST-pretrained CNN
(``pretrain_emnist.py``): it renders a (64, 2) normalized trajectory as a
28x28 grayscale image with the same orientation and black-background /
white-stroke polarity as the (transpose-corrected) EMNIST Letters images, so
``base_model.pt`` / ``personal_model.pt`` can classify hand-drawn strokes
without retraining. This is part of a parallel "pretrain + calibrate"
research path that sits alongside the existing CSV/LSTM trajectory pipeline.

Point coordinates from ``vrs_index_fingertip_tracker.py`` use standard image
pixel convention (x = column, increasing right; y = row, increasing down),
which already matches the corrected EMNIST orientation used to train
``base_model.pt`` — so no additional flip/transpose is applied here.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from config import (
    DATASET_CSV_PATH,
    INPUT_POINTS,
    LABELS,
    RASTER_GAUSSIAN_BLUR_SIGMA,
    RASTER_IMAGE_SIZE,
    RASTER_STROKE_THICKNESS_PX,
)

SUPERSAMPLE_FACTOR = 4
CANVAS_MARGIN_PX = 4


def trajectory_to_image(
    points: Sequence[Sequence[float]],
    image_size: int = RASTER_IMAGE_SIZE,
    stroke_width_px: int = RASTER_STROKE_THICKNESS_PX,
    blur_sigma: float = RASTER_GAUSSIAN_BLUR_SIGMA,
) -> np.ndarray:
    """Render a normalized (64, 2) trajectory as an EMNIST-shaped image.

    Args:
        points: Normalized trajectory such as the output of
            ``normalize_trajectory`` — centered at the origin and scaled so
            the farthest point sits at radius 1.
        image_size: Output image side length in pixels (EMNIST uses 28).
        stroke_width_px: Approximate stroke thickness in the final image.
        blur_sigma: Gaussian blur radius applied after downsampling.

    Returns:
        A ``(image_size, image_size)`` float32 array in ``[0, 1]`` with a
        black background and a white stroke, matching the polarity and
        orientation of the transpose-corrected EMNIST images used to train
        the CNN in ``pretrain_emnist.py``.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        return np.zeros((image_size, image_size), dtype=np.float32)

    hires_size = image_size * SUPERSAMPLE_FACTOR
    center = hires_size / 2.0
    draw_radius = center - CANVAS_MARGIN_PX * SUPERSAMPLE_FACTOR
    stroke_px = max(1, stroke_width_px * SUPERSAMPLE_FACTOR)

    canvas = Image.new("L", (hires_size, hires_size), color=0)
    draw = ImageDraw.Draw(canvas)

    pixels = [(float(center + x * draw_radius), float(center + y * draw_radius)) for x, y in pts]
    draw.line(pixels, fill=255, width=stroke_px, joint="curve")

    # Round off line joints/endpoints so strokes don't look faceted at low res.
    cap_radius = stroke_px / 2.0
    for x, y in pixels:
        draw.ellipse((x - cap_radius, y - cap_radius, x + cap_radius, y + cap_radius), fill=255)

    canvas = canvas.resize((image_size, image_size), Image.LANCZOS)
    if blur_sigma > 0:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=blur_sigma))

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.clip(array, 0.0, 1.0)


def _load_trajectories_by_label(csv_path: Path) -> Dict[str, np.ndarray]:
    """Load one recorded (64, 2) trajectory per label from the dataset CSV.

    Args:
        csv_path: Path to the recorded trajectory CSV (see ``config.DATASET_CSV_PATH``).

    Returns:
        Mapping from letter label to its first recorded normalized trajectory.
        Empty if the CSV does not exist or has no valid rows.
    """
    trajectories: Dict[str, np.ndarray] = {}
    if not csv_path.exists():
        return trajectories

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = row.get("label")
            if not label or label in trajectories:
                continue
            try:
                points = np.array(
                    [[float(row[f"p{i}_x"]), float(row[f"p{i}_y"])] for i in range(INPUT_POINTS)],
                    dtype=np.float32,
                )
            except (KeyError, ValueError):
                continue
            trajectories[label] = points

    return trajectories


def _find_emnist_sample(test_dataset, label_idx: int) -> Optional[np.ndarray]:
    """Return the first EMNIST test image matching a 0-indexed A-Z label, if any."""
    targets = test_dataset.targets.numpy() - 1  # dataset.targets is EMNIST's raw 1-indexed a-z label
    matches = np.nonzero(targets == label_idx)[0]
    if len(matches) == 0:
        return None
    image, _label = test_dataset[int(matches[0])]
    return image.squeeze(0).numpy()


def _side_by_side(left: np.ndarray, right: np.ndarray, scale: int = 4, gap: int = 8) -> Image.Image:
    """Compose two same-size grayscale float [0,1] arrays into one upscaled image."""
    left_img = Image.fromarray((left * 255).astype(np.uint8)).resize(
        (left.shape[1] * scale, left.shape[0] * scale), Image.NEAREST
    )
    right_img = Image.fromarray((right * 255).astype(np.uint8)).resize(
        (right.shape[1] * scale, right.shape[0] * scale), Image.NEAREST
    )
    combined = Image.new("L", (left_img.width + gap + right_img.width, left_img.height), color=64)
    combined.paste(left_img, (0, 0))
    combined.paste(right_img, (left_img.width + gap, 0))
    return combined


def _save_debug_comparison(output_dir: Path, num_samples: int = 8) -> None:
    """Save side-by-side images comparing rasterized recorded letters to real EMNIST samples.

    For each recorded letter in ``aria_letter_trajectories.csv`` this
    rasterizes the trajectory with ``trajectory_to_image`` and pairs it with
    a real EMNIST test image of the same letter, so scale, stroke thickness,
    and orientation/polarity can be checked visually side by side.

    Args:
        output_dir: Directory to write ``debug_comparison_*.png`` files into.
        num_samples: Maximum number of comparison images to generate.
    """
    from pretrain_emnist import load_datasets

    trajectories_by_label = _load_trajectories_by_label(DATASET_CSV_PATH)
    if not trajectories_by_label:
        print(f"No recorded trajectories found at {DATASET_CSV_PATH}; draw and save some letters first.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    _, test_dataset = load_datasets()

    letters = [letter for letter in LABELS if letter in trajectories_by_label][:num_samples]
    saved = 0
    for letter in letters:
        emnist_array = _find_emnist_sample(test_dataset, LABELS.index(letter))
        if emnist_array is None:
            continue
        rasterized = trajectory_to_image(trajectories_by_label[letter])
        combined = _side_by_side(emnist_array, rasterized)
        combined.save(output_dir / f"debug_comparison_{letter}.png")
        saved += 1

    print(f"Saved {saved} debug comparison images (real EMNIST | rasterized trajectory) to: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rasterize trajectories into EMNIST-shaped images.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save side-by-side rasterized-vs-real-EMNIST comparison images for visual verification.",
    )
    parser.add_argument("--output-dir", type=str, default="debug_rasterize")
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()

    if args.debug:
        _save_debug_comparison(Path(args.output_dir), num_samples=args.num_samples)
    else:
        print("Pass --debug to generate side-by-side rasterized-vs-EMNIST comparison images.")


if __name__ == "__main__":
    main()
