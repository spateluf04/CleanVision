"""End-to-end regression tests for the Aria ML critical path.

This suite uses Python's unittest framework and is designed to be runnable with
``python -m pytest tests/``. The tests cover normalization, dwell-based
trajectory completion, CSV persistence, training-state JSON persistence, model
checkpoint inference, and configuration sanity checks.
"""

from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import train_letter_lstm as trainer
import training_dashboard as dashboard
from vrs_index_fingertip_tracker import (
    BufferedCsvAppender,
    TrajectoryBuilder,
    build_csv_header,
    ensure_csv,
    load_label_counts,
    normalize_trajectory,
)


class NormalizeTrajectoryTests(unittest.TestCase):
    """Verify trajectory normalization behavior for representative inputs."""

    def assert_normalized_output(self, output: np.ndarray) -> None:
        """Validate common postconditions for normalized trajectories."""
        self.assertEqual(output.shape, (64, 2))
        self.assertTrue(np.isfinite(output).all())
        self.assertLessEqual(float(output.max()), 1.0001)
        self.assertGreaterEqual(float(output.min()), -1.0001)

    def test_normalize_trajectory(self) -> None:
        """Normalize several path types and verify shape and range guarantees."""
        straight_line = [(float(idx), 0.0) for idx in range(32)]
        circle = [
            (math.cos(theta) * 25.0, math.sin(theta) * 25.0)
            for theta in np.linspace(0.0, 2.0 * math.pi, 128, endpoint=False)
        ]
        single_point = [(12.0, -4.0)]
        noisy_path = np.cumsum(np.random.default_rng(7).normal(size=(300, 2)), axis=0).tolist()

        for points in (straight_line, circle, single_point, noisy_path):
            normalized = normalize_trajectory(points)
            self.assert_normalized_output(normalized)

        with self.assertRaises(ValueError):
            normalize_trajectory([])


class TrajectoryBuilderTests(unittest.TestCase):
    """Verify dwell-driven trajectory completion behavior."""

    def test_trajectory_builder(self) -> None:
        """Simulate motion plus a dwell pause and verify completion detection."""
        builder = TrajectoryBuilder(
            movement_threshold_px=25.0,
            dwell_seconds=1.2,
            min_points=3,
            min_duration_seconds=0.8,
        )

        sequence = [
            ((0, 0), 0),
            ((40, 0), 100_000_000),
            ((80, 0), 200_000_000),
            ((80, 0), 900_000_000),
            ((80, 0), 1_500_000_000),
        ]

        finished = None
        for point, timestamp_ns in sequence:
            finished = builder.update(point, timestamp_ns)

        self.assertIsNotNone(finished)
        self.assertGreaterEqual(len(finished), 3)
        self.assertEqual(builder.points, [])
        self.assertIsNone(builder.anchor_point)
        self.assertIsNone(builder.anchor_timestamp_ns)
        self.assertIsNone(builder.first_point_timestamp_ns)


class CsvOperationTests(unittest.TestCase):
    """Verify CSV write, read, and corrupt-row handling."""

    def test_csv_operations(self) -> None:
        """Write valid samples, read them back, and skip corrupt rows safely."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "samples.csv"
            ensure_csv(csv_path)

            base_points = [(float(idx), float(idx % 3)) for idx in range(64)]
            normalized = normalize_trajectory(base_points)
            labels = ["A", "B", "C", "A", "B"]

            writer = BufferedCsvAppender(csv_path, flush_every=10)
            for label in labels:
                writer.append(label, normalized)
            writer.close()

            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(len(rows) - 1, 5)
            self.assertEqual([row[0] for row in rows[1:]], labels)

            with csv_path.open("a", newline="", encoding="utf-8") as handle:
                raw_writer = csv.writer(handle)
                raw_writer.writerow(["Z"])  # corrupt, missing coordinates
                raw_writer.writerow(["!"] + normalized.reshape(-1).astype(str).tolist())  # invalid label

            counts = load_label_counts(csv_path)
            self.assertEqual(counts["A"], 2)
            self.assertEqual(counts["B"], 2)
            self.assertEqual(counts["C"], 1)
            self.assertEqual(counts["Z"], 1)

            index = trainer.LetterTrajectoryIndex(csv_path)
            self.assertEqual(len(index), 5)


class TrainingStateJsonTests(unittest.TestCase):
    """Verify persistent training state load/save behavior."""

    def test_training_state_json(self) -> None:
        """Persist a training state update and reload it from disk."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "training_state.json"
            with mock.patch.object(dashboard, "TRAINING_STATE_PATH", state_path):
                initial = dashboard.default_training_state()
                dashboard.save_training_state(initial)

                loaded = dashboard.load_training_state()
                self.assertEqual(loaded["total_samples"], 0)
                self.assertEqual(loaded["samples_per_letter"]["A"], 0)

                loaded["total_samples"] = 12
                loaded["samples_per_letter"]["A"] = 5
                loaded["training_sessions"] = 3
                loaded["last_trained"] = "2026-06-09 12:34:56"
                loaded["best_accuracy"] = 0.91
                loaded["model_version"] = 4
                loaded["per_letter_accuracy"]["A"] = 0.95
                loaded["history"].append({"session": 1, "val_accuracy": 0.91})
                dashboard.save_training_state(loaded)

                reloaded = dashboard.load_training_state()
                self.assertEqual(reloaded["total_samples"], 12)
                self.assertEqual(reloaded["samples_per_letter"]["A"], 5)
                self.assertEqual(reloaded["training_sessions"], 3)
                self.assertEqual(reloaded["last_trained"], "2026-06-09 12:34:56")
                self.assertAlmostEqual(reloaded["best_accuracy"], 0.91, places=6)
                self.assertEqual(reloaded["model_version"], 4)
                self.assertAlmostEqual(reloaded["per_letter_accuracy"]["A"], 0.95, places=6)
                self.assertEqual(len(reloaded["history"]), 1)


class ModelInferenceTests(unittest.TestCase):
    """Verify checkpoint loading and forward inference behavior."""

    def test_model_inference(self) -> None:
        """Load a letter model checkpoint and verify output logits behavior."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = PROJECT_ROOT / "letter_model.pt"
            cleanup_path = None

            if not model_path.exists():
                cleanup_path = Path(tmp_dir) / "letter_model.pt"
                checkpoint = {
                    "model_type": "lstm",
                    "model_state_dict": trainer.LetterLSTMClassifier().state_dict(),
                    "label_to_index": config.LABEL_TO_INDEX,
                    "input_points": config.INPUT_POINTS,
                    "input_size": config.INPUT_SIZE,
                    "num_classes": config.NUM_CLASSES,
                    "hidden_size": config.HIDDEN_SIZE,
                    "num_layers": config.NUM_LAYERS,
                    "best_val_accuracy": 0.0,
                }
                torch.save(checkpoint, cleanup_path)
                model_path = cleanup_path

            checkpoint = torch.load(model_path, map_location="cpu")
            model_type = checkpoint.get("model_type", "lstm").lower()
            if model_type == "transformer":
                model = trainer.LetterTransformerClassifier()
            else:
                model = trainer.LetterLSTMClassifier()
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            synthetic = torch.randn(1, 64, 2, dtype=torch.float32)
            with torch.no_grad():
                logits = model(synthetic)[0]
                probs = torch.softmax(logits, dim=0)

            self.assertEqual(tuple(logits.shape), (26,))
            self.assertAlmostEqual(float(probs.sum().item()), 1.0, places=5)


class ConfigValueTests(unittest.TestCase):
    """Verify shared config values remain within reasonable ranges."""

    def test_config_values(self) -> None:
        """Check dimensions, thresholds, intervals, and paths for sanity."""
        positive_scalars = [
            config.TRAJECTORY_MOVEMENT_THRESHOLD_PX,
            config.TRAJECTORY_DWELL_SECONDS,
            config.TRAJECTORY_MIN_DURATION_SECONDS,
            config.INPUT_POINTS,
            config.INPUT_SIZE,
            config.NUM_CLASSES,
            config.HIDDEN_SIZE,
            config.NUM_LAYERS,
            config.DEFAULT_TRAIN_EPOCHS,
            config.DEFAULT_BATCH_SIZE,
            config.DEFAULT_LEARNING_RATE,
            config.CAMERA_WIDTH,
            config.CAMERA_HEIGHT,
            config.WINDOW_WIDTH,
            config.WINDOW_HEIGHT,
            config.LIVE_RGB_FRAME_INTERVAL_MS,
            config.BRIDGE_SEND_QUEUE_MAXSIZE,
            config.CSV_SAVE_BUFFER_SIZE,
            config.MAX_TRAJECTORY_POINTS,
        ]
        for value in positive_scalars:
            self.assertGreater(value, 0)

        self.assertGreaterEqual(config.DEFAULT_YOLO_CONF_THRESHOLD, 0.0)
        self.assertLessEqual(config.DEFAULT_YOLO_CONF_THRESHOLD, 1.0)
        self.assertGreaterEqual(config.HEATMAP_GREEN_THRESHOLD, config.HEATMAP_YELLOW_THRESHOLD)
        self.assertIsInstance(config.BASE_DIR, Path)
        self.assertTrue(config.BASE_DIR.exists())
        self.assertEqual(config.DATASET_CSV_PATH.parent, config.BASE_DIR)
        self.assertEqual(config.TRAINING_STATE_PATH.parent, config.BASE_DIR)
        self.assertTrue(config.WS_URL.startswith("ws://"))


if __name__ == "__main__":
    unittest.main()
