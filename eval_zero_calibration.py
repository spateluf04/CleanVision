"""Zero-calibration evaluation: base_model.pt on real Aria trajectories, no fine-tuning.

Runs every recorded trajectory in ``aria_letter_trajectories.csv`` through
``trajectory_to_image`` and the EMNIST-pretrained CNN, with no per-user
fine-tuning at all. This measures the raw EMNIST-to-egocentric-air-drawing
domain gap: overall accuracy, per-letter accuracy, a confusion table, and the
top-3 predictions (with confidences) for every sample.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

from config import BASE_DIR, BASE_MODEL_PATH, LABELS
from eval_utils import load_all_trajectories_by_label, load_cnn_checkpoint, predict_topk
from rasterize import trajectory_to_image


def run_evaluation(csv_path: Path, model_path: Path) -> Dict:
    """Classify every CSV trajectory with ``model_path`` and tally accuracy/confusion."""
    trajectories_by_label = load_all_trajectories_by_label(csv_path)
    if not trajectories_by_label:
        print(f"No recorded trajectories found at {csv_path}; nothing to evaluate.")
        return {}

    model, label_lookup, _ = load_cnn_checkpoint(model_path)

    evaluated = []
    skipped = []
    per_letter: Dict[str, Dict] = {}
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    samples_report = []

    total_correct = 0
    total_count = 0

    for letter in LABELS:
        samples = trajectories_by_label.get(letter, [])
        if not samples:
            skipped.append(letter)
            continue
        evaluated.append(letter)

        correct = 0
        for i, points in enumerate(samples):
            image = trajectory_to_image(points)
            top3 = predict_topk(model, label_lookup, image, k=3)
            predicted_label = top3[0][0]
            is_correct = predicted_label == letter
            correct += int(is_correct)
            confusion[letter][predicted_label] += 1
            samples_report.append(
                {
                    "index": i,
                    "true_label": letter,
                    "predicted_label": predicted_label,
                    "correct": is_correct,
                    "top3": [[label, round(conf, 4)] for label, conf in top3],
                }
            )

        total_correct += correct
        total_count += len(samples)
        per_letter[letter] = {
            "count": len(samples),
            "correct": correct,
            "accuracy": correct / len(samples),
            "confusion": dict(confusion[letter]),
        }

    overall_accuracy = total_correct / total_count if total_count else 0.0

    report = {
        "model_path": str(model_path),
        "csv_path": str(csv_path),
        "total_samples": total_count,
        "overall_accuracy": overall_accuracy,
        "evaluated_letters": evaluated,
        "skipped_letters": skipped,
        "per_letter": per_letter,
        "samples": samples_report,
    }
    return report


def print_summary(report: Dict) -> None:
    """Print a readable accuracy / confusion summary to the console."""
    if not report:
        return

    print(f"\nModel: {report['model_path']}")
    print(f"Total samples evaluated: {report['total_samples']}")
    print(f"Overall accuracy: {report['overall_accuracy'] * 100:.2f}%")
    print(f"Evaluated letters: {', '.join(report['evaluated_letters'])}")
    print(f"Skipped letters (no samples): {', '.join(report['skipped_letters']) if report['skipped_letters'] else 'none'}")

    print("\nPer-letter accuracy:")
    print(f"  {'Letter':<8}{'Count':<8}{'Correct':<9}{'Accuracy':<10}")
    for letter, stats in report["per_letter"].items():
        print(f"  {letter:<8}{stats['count']:<8}{stats['correct']:<9}{stats['accuracy'] * 100:>7.2f}%")

    print("\nConfusion (true -> predicted counts):")
    for letter, stats in report["per_letter"].items():
        confusion_str = ", ".join(f"{pred}:{count}" for pred, count in sorted(stats["confusion"].items(), key=lambda kv: -kv[1]))
        print(f"  {letter} -> {confusion_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base_model.pt on recorded trajectories with zero calibration.")
    parser.add_argument("--csv-path", type=str, default=None, help="Dataset CSV path (default: config.DATASET_CSV_PATH).")
    parser.add_argument("--model-path", type=str, default=str(BASE_MODEL_PATH), help="CNN checkpoint to evaluate.")
    parser.add_argument("--output", type=str, default=str(BASE_DIR / "results" / "zero_calibration_report.json"), help="Where to save the JSON report.")
    args = parser.parse_args()

    from config import DATASET_CSV_PATH

    csv_path = Path(args.csv_path) if args.csv_path else DATASET_CSV_PATH
    model_path = Path(args.model_path)

    report = run_evaluation(csv_path, model_path)
    print_summary(report)

    if report:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nSaved full report to {output_path}")


if __name__ == "__main__":
    main()
