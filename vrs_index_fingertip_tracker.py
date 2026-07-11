"""Collect and normalize fingertip trajectories from Aria VRS recordings.

This module reads RGB frames from a Project Aria VRS file, tracks the index
fingertip with MediaPipe Hands, detects dwell-based stroke endings, and writes
normalized trajectories to a CSV dataset. It depends on projectaria_tools,
OpenCV, MediaPipe, and NumPy, and it produces labeled trajectory samples.
"""

import argparse
import csv
import math
import os
import time
from pathlib import Path
from collections import defaultdict
from typing import DefaultDict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from projectaria_tools.core.data_provider import create_vrs_data_provider
from projectaria_tools.core.image import debayer
from config import (
    CSV_SAVE_BUFFER_SIZE,
    DATASET_CSV_PATH,
    INPUT_POINTS,
    MAX_TRAJECTORY_POINTS,
    MEDIAPIPE_IDLE_FRAME_SKIP,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    MEDIAPIPE_MODEL_COMPLEXITY,
    MEDIAPIPE_SINGLE_HAND_MAX_NUM_HANDS,
    RGB_LABEL_CANDIDATES,
    TARGET_SAMPLES_PER_LETTER,
    TRACKER_PERF_LOG_INTERVAL,
    TRAJECTORY_DWELL_SECONDS,
    TRAJECTORY_MIN_DURATION_SECONDS,
    TRAJECTORY_MIN_POINTS,
    TRAJECTORY_MIN_SPAN_PX,
    TRAJECTORY_MOVEMENT_THRESHOLD_PX,
    TRAJECTORY_NORMALIZED_POINTS,
)
from logging_utils import get_logger


logger = get_logger(__name__)


class TrajectoryBuilder:
    """Collect fingertip points until a dwell pause marks the stroke as complete.

    Args:
        movement_threshold_px: Minimum motion that resets the dwell anchor.
        dwell_seconds: Stillness duration required to end a trajectory.
        min_points: Minimum number of points required to emit a trajectory.
        min_duration_seconds: Minimum capture duration before a trajectory can end.
    """

    def __init__(
        self,
        movement_threshold_px: float = TRAJECTORY_MOVEMENT_THRESHOLD_PX,
        dwell_seconds: float = TRAJECTORY_DWELL_SECONDS,
        min_points: int = TRAJECTORY_MIN_POINTS,
        min_duration_seconds: float = TRAJECTORY_MIN_DURATION_SECONDS,
    ) -> None:
        self.movement_threshold_px = movement_threshold_px
        self.dwell_ns = int(dwell_seconds * 1e9)
        self.min_duration_ns = int(min_duration_seconds * 1e9)
        self.min_points = min_points
        self.points = []
        self.anchor_point = None
        self.anchor_timestamp_ns = None
        self.first_point_timestamp_ns = None

    def update(
        self,
        point: Optional[Tuple[int, int]],
        timestamp_ns: int,
    ) -> Optional[List[Tuple[int, int]]]:
        """Update the active trajectory with a new fingertip observation.

        Args:
            point: Current fingertip pixel coordinate, or ``None`` if unavailable.
            timestamp_ns: Capture timestamp for the current frame in nanoseconds.

        Returns:
            The completed trajectory when the dwell condition is met, otherwise
            ``None``.
        """
        finished = None

        if point is None:
            self.anchor_point = None
            self.anchor_timestamp_ns = None
            self.first_point_timestamp_ns = None
            return finished

        self.points.append(point)
        if len(self.points) > MAX_TRAJECTORY_POINTS:
            self.points.pop(0)

        if self.first_point_timestamp_ns is None:
            self.first_point_timestamp_ns = timestamp_ns

        if self.anchor_point is None:
            self.anchor_point = point
            self.anchor_timestamp_ns = timestamp_ns
            return finished

        distance = math.hypot(point[0] - self.anchor_point[0], point[1] - self.anchor_point[1])
        if distance > self.movement_threshold_px:
            self.anchor_point = point
            self.anchor_timestamp_ns = timestamp_ns
            return finished

        if (
            self.anchor_timestamp_ns is not None
            and timestamp_ns - self.anchor_timestamp_ns >= self.dwell_ns
            and len(self.points) >= self.min_points
            and self.first_point_timestamp_ns is not None
            and timestamp_ns - self.first_point_timestamp_ns >= self.min_duration_ns
        ):
            finished = self.points[:]
            self.points = []
            self.anchor_point = None
            self.anchor_timestamp_ns = None
            self.first_point_timestamp_ns = None

        return finished


def normalize_trajectory(points: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Normalize a raw fingertip trajectory to shape (64, 2).

    Steps:
    1. Resample to exactly 64 points along arc length
    2. Center around the origin
    3. Scale to unit size
    Args:
        points: Variable-length list of ``(x, y)`` points.

    Returns:
        A normalized array of shape ``(64, 2)``.

    Raises:
        ValueError: If the input shape is invalid or no points are provided.
    """
    pts = np.asarray(points, dtype=np.float32)

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an array-like of shape (N, 2)")
    if len(pts) == 0:
        raise ValueError("points must contain at least one point")

    if len(pts) > 1:
        keep = np.ones(len(pts), dtype=bool)
        keep[1:] = np.any(np.diff(pts, axis=0) != 0, axis=1)
        pts = pts[keep]

    if len(pts) == 1:
        return np.zeros((TRAJECTORY_NORMALIZED_POINTS, 2), dtype=np.float32)

    deltas = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cumulative[-1]

    if total_length == 0:
        return np.zeros((TRAJECTORY_NORMALIZED_POINTS, 2), dtype=np.float32)

    target_distances = np.linspace(0.0, total_length, TRAJECTORY_NORMALIZED_POINTS, dtype=np.float32)
    seg_idx = np.searchsorted(cumulative, target_distances, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(seg_lengths) - 1)

    start_pts = pts[seg_idx]
    end_pts = pts[seg_idx + 1]
    start_distances = cumulative[seg_idx]
    end_distances = cumulative[seg_idx + 1]
    denom = end_distances - start_distances
    safe_denom = np.where(denom == 0, 1.0, denom)
    interpolation = ((target_distances - start_distances) / safe_denom).reshape(-1, 1)
    interpolation = np.where(denom.reshape(-1, 1) == 0, 0.0, interpolation)
    resampled = start_pts + interpolation * (end_pts - start_pts)

    resampled -= np.mean(resampled, axis=0, keepdims=True)

    radii = np.linalg.norm(resampled, axis=1)
    scale = np.max(radii)
    if scale > 0:
        resampled /= scale

    return resampled.astype(np.float32)


def trajectory_span_px(points: Sequence[Sequence[float]]) -> float:
    """Return the bounding-box diagonal of a raw trajectory in pixels."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or len(pts) == 0:
        return 0.0
    span = pts.max(axis=0) - pts.min(axis=0)
    return float(math.hypot(span[0], span[1]))


def is_degenerate_trajectory(points: Sequence[Sequence[float]], min_span_px: float = TRAJECTORY_MIN_SPAN_PX) -> bool:
    """Return True if a trajectory is too motionless to be a real letter stroke.

    A finished dwell-based trajectory can still consist of near-duplicate points
    (e.g. an accidental trigger with almost no hand motion). Such trajectories
    normalize to a degenerate (all-zero) shape and would silently pollute the
    training set or produce a meaningless prediction, so callers should reject
    them before persisting or predicting.
    """
    return trajectory_span_px(points) < min_span_px


def build_csv_header() -> List[str]:
    """Build the CSV header for normalized trajectory exports."""
    header = ["label"]
    for idx in range(TRAJECTORY_NORMALIZED_POINTS):
        header.extend([f"p{idx}_x", f"p{idx}_y"])
    return header


def ensure_csv(csv_path: Path) -> None:
    """Create the dataset CSV file with a header if it does not exist.

    Args:
        csv_path: Output CSV path.

    Returns:
        None.

    Raises:
        RuntimeError: If the file cannot be created.
    """
    if csv_path.exists():
        return
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(build_csv_header())
    except OSError as exc:
        raise RuntimeError(f"Failed to initialize CSV file at {csv_path}: {exc}") from exc


def load_label_counts(csv_path: Path) -> DefaultDict[str, int]:
    """Load per-letter sample counts from an existing dataset CSV.

    Args:
        csv_path: Dataset CSV path.

    Returns:
        Default dictionary mapping letters to sample counts.

    Raises:
        RuntimeError: If the CSV cannot be read.
    """
    counts: DefaultDict[str, int] = defaultdict(int)
    if not csv_path.exists():
        return counts

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                logger.warning("Dataset CSV %s is empty; starting with zero counts.", csv_path)
                return counts
            for row_idx, row in enumerate(reader, start=2):
                try:
                    label = (row.get("label") or "").strip().upper()
                except Exception:
                    logger.warning("Skipping corrupt CSV row %s while loading counts.", row_idx)
                    continue
                if len(label) == 1 and "A" <= label <= "Z":
                    counts[label] += 1
                elif label:
                    logger.warning("Ignoring invalid label %r in row %s of %s.", label, row_idx, csv_path)
    except OSError as exc:
        raise RuntimeError(f"Failed to read label counts from {csv_path}: {exc}") from exc

    return counts


def append_sample(csv_path: Path, label: str, normalized_points: np.ndarray) -> None:
    """Append one normalized trajectory row to the dataset CSV."""
    row = [label]
    row.extend(normalized_points.astype(np.float32).reshape(-1).tolist())
    try:
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(row)
    except OSError as exc:
        raise RuntimeError(f"Failed to append sample for {label} to {csv_path}: {exc}") from exc


class BufferedCsvAppender:
    """Buffer trajectory rows and flush them to disk in small batches."""

    def __init__(self, csv_path: Path, flush_every: int = CSV_SAVE_BUFFER_SIZE) -> None:
        self.csv_path = csv_path
        self.flush_every = flush_every
        self._buffer = []

    def append(self, label: str, normalized_points: np.ndarray) -> None:
        """Buffer a single trajectory row for later flushing."""
        row = [label]
        row.extend(normalized_points.astype(np.float32).reshape(-1).tolist())
        self._buffer.append(row)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Flush all buffered rows to disk."""
        if not self._buffer:
            return
        try:
            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerows(self._buffer)
                handle.flush()
                os.fsync(handle.fileno())
            self._buffer.clear()
        except OSError as exc:
            raise RuntimeError(f"Failed to flush buffered samples to {self.csv_path}: {exc}") from exc

    def close(self) -> None:
        """Flush remaining rows before shutdown."""
        self.flush()


def find_rgb_stream_id(provider: object, explicit_label: Optional[str] = None) -> Tuple[object, str]:
    """Resolve the RGB stream ID from a VRS provider.

    Args:
        provider: Project Aria VRS data provider.
        explicit_label: Optional exact stream label.

    Returns:
        A tuple of ``(stream_id, stream_label)``.

    Raises:
        RuntimeError: If stream metadata cannot be queried.
        ValueError: If no RGB stream can be found.
    """
    if explicit_label:
        try:
            stream_id = provider.get_stream_id_from_label(explicit_label)
        except Exception as exc:
            raise RuntimeError(f"Failed to look up RGB stream label '{explicit_label}': {exc}") from exc
        if stream_id is None:
            raise ValueError(f"RGB stream label '{explicit_label}' was not found in the VRS file.")
        return stream_id, explicit_label

    available = {}
    try:
        all_streams = provider.get_all_streams()
    except Exception as exc:
        raise RuntimeError(f"Failed to enumerate VRS streams: {exc}") from exc
    for stream_id in all_streams:
        try:
            label = provider.get_label_from_stream_id(stream_id)
        except Exception:
            logger.debug("Skipping unreadable stream label for stream id %s.", stream_id)
            continue
        available[label] = stream_id

    for candidate in RGB_LABEL_CANDIDATES:
        if candidate in available:
            return available[candidate], candidate

    for label, stream_id in available.items():
        if "rgb" in label.lower():
            return stream_id, label

    labels = ", ".join(sorted(available.keys()))
    raise ValueError(f"Could not find an RGB stream. Available labels: {labels}")


def to_bgr_frame(image_data: object) -> np.ndarray:
    """Convert a VRS image payload into an OpenCV BGR frame.

    Args:
        image_data: Project Aria image payload object.

    Returns:
        The decoded BGR frame.

    Raises:
        RuntimeError: If decoding fails.
    """
    try:
        frame = image_data.to_numpy_array()
    except Exception as exc:
        raise RuntimeError(f"Failed to decode frame buffer: {exc}") from exc

    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if frame.ndim == 2:
        try:
            frame = debayer(frame)
            if hasattr(frame, "to_numpy_array"):
                frame = frame.to_numpy_array()
            if frame.ndim == 3 and frame.shape[2] == 3:
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception:
            logger.debug("Debayer failed for grayscale frame; falling back to gray->BGR conversion.")
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    raise RuntimeError(f"Unsupported frame shape: {frame.shape}")


def main() -> None:
    """Run the interactive VRS-based trajectory collector."""
    parser = argparse.ArgumentParser(description="Collect labeled Aria fingertip trajectories from a VRS recording.")
    parser.add_argument("vrs_path", help="Path to the Aria VRS file.")
    parser.add_argument("--rgb-label", help="Optional explicit RGB stream label, e.g. camera-rgb.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit for debugging.")
    parser.add_argument(
        "--output-csv",
        default=str(DATASET_CSV_PATH),
        help=f"CSV output path. Default: {DATASET_CSV_PATH}",
    )
    args = parser.parse_args()

    vrs_path = Path(args.vrs_path).expanduser()
    if not vrs_path.exists():
        raise FileNotFoundError(f"VRS file not found: {vrs_path}")

    try:
        provider = create_vrs_data_provider(str(vrs_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open VRS file {vrs_path}: {exc}") from exc
    if provider is None:
        raise RuntimeError(f"Failed to create VRS data provider for: {vrs_path}")

    stream_id, rgb_label = find_rgb_stream_id(provider, args.rgb_label)
    try:
        image_config = provider.get_image_configuration(stream_id)
        num_frames = provider.get_num_data(stream_id)
    except Exception as exc:
        raise RuntimeError(f"Failed to inspect RGB stream {rgb_label}: {exc}") from exc
    output_csv = Path(args.output_csv).expanduser()
    ensure_csv(output_csv)
    label_counts = load_label_counts(output_csv)
    current_label = None
    last_saved_label = None
    last_saved_count = None

    logger.info("Using RGB stream: %s", rgb_label)
    logger.info("Frame size: %sx%s", image_config.image_width, image_config.image_height)
    logger.info("Total RGB frames: %s", num_frames)
    logger.info("Writing labeled trajectories to: %s", output_csv)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MEDIAPIPE_SINGLE_HAND_MAX_NUM_HANDS,
        model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
        min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    )
    trajectory_builder = TrajectoryBuilder()
    csv_appender = BufferedCsvAppender(output_csv)

    window_name = "Aria Index Fingertip Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cached_fingertip_point = None
    perf_totals = {
        "fetch": 0.0,
        "decode": 0.0,
        "mediapipe": 0.0,
        "tracking": 0.0,
        "render": 0.0,
        "total": 0.0,
    }
    perf_frames = 0

    frame_idx = 0
    try:
        while frame_idx < num_frames:
            frame_start = time.perf_counter()

            fetch_start = time.perf_counter()
            try:
                image_data, image_record = provider.get_image_data_by_index(stream_id, frame_idx)
            except Exception as exc:
                logger.warning("Skipping corrupt frame %s/%s: %s", frame_idx + 1, num_frames, exc)
                frame_idx += 1
                continue
            perf_totals["fetch"] += time.perf_counter() - fetch_start

            if image_data is None or not image_data.is_valid():
                logger.debug("Skipping invalid frame %s/%s.", frame_idx + 1, num_frames)
                frame_idx += 1
                continue

            decode_start = time.perf_counter()
            try:
                frame_bgr = to_bgr_frame(image_data)
            except Exception as exc:
                logger.warning("Skipping undecodable frame %s/%s: %s", frame_idx + 1, num_frames, exc)
                frame_idx += 1
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            perf_totals["decode"] += time.perf_counter() - decode_start

            tracking_start = time.perf_counter()
            actively_collecting = bool(trajectory_builder.points)
            should_run_hands = actively_collecting or (frame_idx % MEDIAPIPE_IDLE_FRAME_SKIP == 0)
            fingertip_point = cached_fingertip_point
            if should_run_hands:
                mediapipe_start = time.perf_counter()
                try:
                    results = hands.process(frame_rgb)
                except Exception as exc:
                    logger.warning("MediaPipe failed on frame %s/%s: %s", frame_idx + 1, num_frames, exc)
                    results = None
                perf_totals["mediapipe"] += time.perf_counter() - mediapipe_start
                fingertip_point = None
                if results and results.multi_hand_landmarks:
                    h, w = frame_bgr.shape[:2]
                    hand_landmarks = results.multi_hand_landmarks[0]
                    tip = hand_landmarks.landmark[8]
                    fingertip_point = (int(tip.x * w), int(tip.y * h))
                cached_fingertip_point = fingertip_point

            finished_trajectory = trajectory_builder.update(
                fingertip_point if should_run_hands else None,
                image_record.capture_timestamp_ns,
            )
            perf_totals["tracking"] += time.perf_counter() - tracking_start

            if finished_trajectory:
                if current_label is None:
                    logger.info(
                        "Ignored trajectory with %s points because no label was armed. Press A-Z before drawing.",
                        len(finished_trajectory),
                    )
                elif is_degenerate_trajectory(finished_trajectory):
                    logger.warning(
                        "Ignored degenerate trajectory for label %s (%s points, insufficient movement). Draw it again.",
                        current_label,
                        len(finished_trajectory),
                    )
                else:
                    normalized = normalize_trajectory(finished_trajectory)
                    csv_appender.append(current_label, normalized)
                    label_counts[current_label] += 1
                    last_saved_label = current_label
                    last_saved_count = label_counts[current_label]
                    logger.info(
                        "Saved %s sample #%s (%s raw points) to %s",
                        current_label,
                        label_counts[current_label],
                        len(finished_trajectory),
                        output_csv,
                    )
                    current_label = None

            render_start = time.perf_counter()
            if cached_fingertip_point is not None:
                cv2.circle(frame_bgr, cached_fingertip_point, 10, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            if len(trajectory_builder.points) >= 2:
                polyline = np.array(trajectory_builder.points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame_bgr, [polyline], False, (0, 200, 255), 2, lineType=cv2.LINE_AA)

            cv2.putText(
                frame_bgr,
                f"frame {frame_idx + 1}/{num_frames}",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr,
                f"trajectory points: {len(trajectory_builder.points)}",
                (16, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
                lineType=cv2.LINE_AA,
            )
            armed_text = current_label if current_label else "--"
            cv2.putText(
                frame_bgr,
                f"armed label: {armed_text}",
                (16, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 220, 80),
                2,
                lineType=cv2.LINE_AA,
            )
            if last_saved_label is not None and last_saved_count is not None:
                cv2.putText(
                    frame_bgr,
                    f"last saved: {last_saved_label} #{last_saved_count}",
                    (16, 124),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (80, 255, 140),
                    2,
                    lineType=cv2.LINE_AA,
                )
            total_saved = sum(label_counts.values())
            cv2.putText(
                frame_bgr,
                f"total saved: {total_saved} / {26 * TARGET_SAMPLES_PER_LETTER}",
                (16, 156),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 220, 255),
                2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr,
                "tracking: full" if should_run_hands else "tracking: skipped",
                (16, 188),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (140, 220, 255),
                2,
                lineType=cv2.LINE_AA,
            )
            status_rows = [
                "".join(f"{chr(65 + offset)}:{label_counts[chr(65 + offset)]:02d} " for offset in range(row, row + 7)).strip()
                for row in (0, 7, 14, 21)
            ]
            for idx, status_row in enumerate(status_rows):
                cv2.putText(
                    frame_bgr,
                    status_row,
                    (16, 218 + idx * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    2,
                    lineType=cv2.LINE_AA,
                )
            cv2.putText(
                frame_bgr,
                "Press A-Z to arm the next letter. Draw, pause to save. SPACE pause, C clear, Q quit.",
                (16, frame_bgr.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                lineType=cv2.LINE_AA,
            )
            cv2.imshow(window_name, frame_bgr)
            perf_totals["render"] += time.perf_counter() - render_start

            perf_totals["total"] += time.perf_counter() - frame_start
            perf_frames += 1
            if perf_frames % TRACKER_PERF_LOG_INTERVAL == 0:
                avg_ms = {name: (elapsed / perf_frames) * 1000.0 for name, elapsed in perf_totals.items()}
                slowest_stage = max(
                    ("fetch", "decode", "mediapipe", "tracking", "render"),
                    key=lambda stage: avg_ms[stage],
                )
                logger.info(
                    "Tracker perf avg over %s frames | total=%.2fms fetch=%.2fms decode=%.2fms "
                    "mediapipe=%.2fms tracking=%.2fms render=%.2fms | slowest=%s",
                    perf_frames,
                    avg_ms["total"],
                    avg_ms["fetch"],
                    avg_ms["decode"],
                    avg_ms["mediapipe"],
                    avg_ms["tracking"],
                    avg_ms["render"],
                    slowest_stage,
                )

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if ord("a") <= key <= ord("z") or ord("A") <= key <= ord("Z"):
                current_label = chr(key).upper()
                trajectory_builder.points = []
                trajectory_builder.anchor_point = None
                trajectory_builder.anchor_timestamp_ns = None
                trajectory_builder.first_point_timestamp_ns = None
                logger.info(
                    "Armed label %s. Draw it in the air now. Current count: %s/%s",
                    current_label,
                    label_counts[current_label],
                    TARGET_SAMPLES_PER_LETTER,
                )
            if key == ord("c"):
                current_label = None
                trajectory_builder.points = []
                trajectory_builder.anchor_point = None
                trajectory_builder.anchor_timestamp_ns = None
                trajectory_builder.first_point_timestamp_ns = None
                logger.info("Cleared current armed label and trajectory.")
            if key == ord(" "):
                while True:
                    pause_key = cv2.waitKey(0) & 0xFF
                    if pause_key in (27, ord("q")):
                        key = pause_key
                        break
                    if pause_key == ord(" "):
                        break
                if key in (27, ord("q")):
                    break

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        try:
            csv_appender.close()
        finally:
            hands.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
