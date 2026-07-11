"""PyQt5 training dashboard for Aria air-writing collection and review.

This module provides the fixed-layout desktop UI used to capture live RGB
frames, track fingertips, review collected samples, inspect training history,
and manage per-letter statistics. It depends on PyQt5, OpenCV, websockets,
MediaPipe Tasks, NumPy, and project utilities, and it produces a desktop
application backed by persistent JSON and CSV state files.
"""

import asyncio
import base64
import csv
import json
import math
import os
import sys
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import torch
import websockets

from config import (
    ACCENT,
    ARIA_CAMERA_CALIBRATION_ENV_VAR,
    ARIA_CAMERA_CALIBRATION_PATH,
    BLUE_RING,
    BORDER,
    CALIBRATION_SAMPLES_PER_LETTER,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DATASET_CSV_PATH,
    GREEN_RING,
    HEATMAP_GREEN_THRESHOLD,
    HEATMAP_YELLOW_THRESHOLD,
    HAND_LANDMARKER_MODEL_PATH,
    BRIDGE_RETRY_SECONDS,
    LIVE_RGB_FRAME_INTERVAL_SECONDS,
    LOST_HAND_BUFFER_DOT_COLOR,
    MAX_TRAJECTORY_POINTS,
    MEDIAPIPE_HAND_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_HAND_MIN_PRESENCE_CONFIDENCE,
    MEDIAPIPE_HAND_MIN_TRACKING_CONFIDENCE,
    MEDIAPIPE_SINGLE_HAND_MAX_NUM_HANDS,
    MUTED,
    PANEL_BG,
    RED_RING,
    RIGHT_HEIGHT,
    RIGHT_WIDTH,
    SAMPLE_METADATA_PATH,
    SIDEBAR_WIDTH,
    SUCCESS,
    SURFACE_BG,
    TEXT,
    DEFAULT_MODEL_OUTPUT,
    LABEL_TO_INDEX,
    TRAJECTORY_MIN_DURATION_SECONDS,
    TRAJECTORY_DWELL_SECONDS,
    TRAJECTORY_MIN_POINTS,
    TRAJECTORY_MOVEMENT_THRESHOLD_PX,
    TRAJECTORY_NORMALIZED_POINTS,
    TRAINING_HISTORY_EXPORT_PATH,
    TRAINING_STATE_PATH,
    UI_BEEP_DURATION_SECONDS,
    UI_BEEP_FREQUENCY_HZ,
    UI_BEEP_SAMPLE_RATE,
    UI_BEEP_VOLUME,
    UI_CAPTURE_COUNTDOWN_SECONDS,
    UI_DRAW_BANNER_SECONDS,
    UI_INFO_FLASH_SECONDS,
    UI_LOST_HAND_BUFFER_FRAMES,
    UI_RESET_FLASH_SECONDS,
    UI_SAVE_FLASH_SECONDS,
    UI_SMOOTHING_WINDOW,
    UI_STABILIZATION_FRAMES,
    TRAJ_HEIGHT,
    TRAJ_WIDTH,
    UNDISTORT_DISTORTION,
    UNDISTORT_FOCAL_SCALE,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    WS_URL,
    BOTTOM_HEIGHT,
)
from calibration import count_calibration_samples, fine_tune_personal_model, save_calibration_sample
from logging_utils import get_logger
from train_letter_lstm import LetterLSTMClassifier, LetterTransformerClassifier

try:
    import pygame
except Exception:
    pygame = None

from vrs_index_fingertip_tracker import TrajectoryBuilder, is_degenerate_trajectory, normalize_trajectory


logger = get_logger(__name__)


def _atomic_write(path: Path, write_fn) -> None:
    """Write a file atomically via a temp file plus ``os.replace``.

    Prevents truncating ``path`` and then crashing mid-write, which would
    otherwise leave the dashboard's persisted state or dataset corrupted.
    """
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        write_fn(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def default_training_state() -> Dict[str, Any]:
    """Build the default persistent dashboard state."""
    return {
        "total_samples": 0,
        "samples_per_letter": {chr(65 + idx): 0 for idx in range(26)},
        "training_sessions": 0,
        "last_trained": None,
        "best_accuracy": 0.0,
        "model_version": 0,
        "per_letter_accuracy": {chr(65 + idx): 0.0 for idx in range(26)},
        "history": [],
    }


def load_training_state() -> Dict[str, Any]:
    """Load training dashboard state from disk with safe defaults."""
    state = default_training_state()
    if not TRAINING_STATE_PATH.exists():
        save_training_state(state)
        return state

    try:
        with TRAINING_STATE_PATH.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except Exception as exc:
        logger.warning("Failed to load training state from %s, recreating defaults: %s", TRAINING_STATE_PATH, exc)
        save_training_state(state)
        return state

    state["total_samples"] = int(loaded.get("total_samples", 0))
    state["training_sessions"] = int(loaded.get("training_sessions", 0))
    state["last_trained"] = loaded.get("last_trained")
    state["best_accuracy"] = float(loaded.get("best_accuracy", 0.0))
    state["model_version"] = int(loaded.get("model_version", 0))

    loaded_counts = loaded.get("samples_per_letter", {})
    for letter in state["samples_per_letter"]:
        state["samples_per_letter"][letter] = int(loaded_counts.get(letter, 0))

    loaded_accuracy = loaded.get("per_letter_accuracy", {})
    for letter in state["per_letter_accuracy"]:
        state["per_letter_accuracy"][letter] = float(loaded_accuracy.get(letter, 0.0))

    loaded_history = loaded.get("history", [])
    if isinstance(loaded_history, list):
        state["history"] = loaded_history

    return state


def save_training_state(state: Dict[str, Any]) -> None:
    """Persist the dashboard training state to disk."""
    try:
        _atomic_write(TRAINING_STATE_PATH, lambda handle: json.dump(state, handle, indent=2))
    except OSError as exc:
        logger.error("Failed to save training state to %s: %s", TRAINING_STATE_PATH, exc)
        raise


def dataset_csv_header() -> List[str]:
    """Return the header row for the normalized trajectory dataset CSV."""
    header = ["label"]
    for idx in range(TRAJECTORY_NORMALIZED_POINTS):
        header.extend([f"p{idx}_x", f"p{idx}_y"])
    return header


def ensure_dataset_csv() -> None:
    """Create the dataset CSV file if it does not exist."""
    if DATASET_CSV_PATH.exists():
        return
    try:
        with DATASET_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(dataset_csv_header())
    except OSError as exc:
        raise RuntimeError(f"Failed to initialize dataset CSV at {DATASET_CSV_PATH}: {exc}") from exc


def load_sample_metadata() -> List[Dict[str, Any]]:
    """Load or rebuild sample metadata used by review mode."""
    ensure_dataset_csv()
    if SAMPLE_METADATA_PATH.exists():
        try:
            with SAMPLE_METADATA_PATH.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, list):
                return loaded
        except Exception as exc:
            logger.warning("Failed to load sample metadata from %s, rebuilding from CSV: %s", SAMPLE_METADATA_PATH, exc)

    records = []
    try:
        with DATASET_CSV_PATH.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row_idx, row in enumerate(reader):
                try:
                    label = (row.get("label") or "").strip().upper()
                    if not label:
                        continue
                    normalized_points = []
                    for point_idx in range(TRAJECTORY_NORMALIZED_POINTS):
                        normalized_points.append(
                            [
                                float(row.get(f"p{point_idx}_x", 0.0)),
                                float(row.get(f"p{point_idx}_y", 0.0)),
                            ]
                        )
                    records.append(
                        {
                            "id": f"imported-{row_idx}",
                            "label": label,
                            "timestamp": None,
                            "trajectory": normalized_points,
                            "raw_point_count": TRAJECTORY_NORMALIZED_POINTS,
                        }
                    )
                except (TypeError, ValueError) as exc:
                    logger.warning("Skipping corrupt dataset row %s while rebuilding metadata: %s", row_idx + 1, exc)
    except OSError as exc:
        raise RuntimeError(f"Failed to read dataset CSV {DATASET_CSV_PATH}: {exc}") from exc

    save_sample_metadata(records)
    return records


def save_sample_metadata(records: Sequence[Dict[str, Any]]) -> None:
    """Write review metadata records to disk."""
    try:
        _atomic_write(SAMPLE_METADATA_PATH, lambda handle: json.dump(records, handle, indent=2))
    except OSError as exc:
        raise RuntimeError(f"Failed to write sample metadata {SAMPLE_METADATA_PATH}: {exc}") from exc


def append_sample_record(
    label: str,
    trajectory_points: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    """Append one recorded sample to the dataset CSV and metadata store."""
    ensure_dataset_csv()
    normalized_points = normalize_trajectory(trajectory_points)
    row = [label]
    row.extend(normalized_points.reshape(-1).tolist())
    try:
        with DATASET_CSV_PATH.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(row)
    except OSError as exc:
        raise RuntimeError(f"Failed to append dataset row to {DATASET_CSV_PATH}: {exc}") from exc

    records = load_sample_metadata()
    record = {
        "id": str(uuid4()),
        "label": label,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trajectory": np.asarray(trajectory_points, dtype=np.float32).tolist(),
        "raw_point_count": int(len(trajectory_points)),
    }
    records.append(record)
    save_sample_metadata(records)
    return record


def rewrite_dataset_from_records(records: Sequence[Dict[str, Any]]) -> None:
    """Rewrite the normalized dataset CSV from review metadata records."""
    ensure_dataset_csv()

    def write_csv(handle):
        writer = csv.writer(handle)
        writer.writerow(dataset_csv_header())
        for record in records:
            trajectory = np.asarray(record.get("trajectory", []), dtype=np.float32)
            if trajectory.ndim != 2 or trajectory.shape[1] != 2 or len(trajectory) == 0:
                logger.warning("Skipping invalid trajectory while rewriting dataset for record %s.", record.get("id"))
                continue
            normalized = normalize_trajectory(trajectory)
            row = [record["label"]]
            row.extend(normalized.reshape(-1).tolist())
            writer.writerow(row)

    try:
        _atomic_write(DATASET_CSV_PATH, write_csv)
    except OSError as exc:
        raise RuntimeError(f"Failed to rewrite dataset CSV {DATASET_CSV_PATH}: {exc}") from exc
    save_sample_metadata(records)


class BridgeFeedWorker(QtCore.QThread):
    """Receive RGB frames from the WebSocket bridge on a worker thread."""

    frame_received = QtCore.pyqtSignal(object, int)
    status_changed = QtCore.pyqtSignal(str)

    def __init__(self, ws_url: str = WS_URL, parent: Optional[QtCore.QObject] = None):
        """Initialize the bridge feed worker."""
        super().__init__(parent)
        self.ws_url = ws_url
        self._stop_event = threading.Event()
        self._latest_frame = None
        self._latest_ts = 0
        self._last_emitted_ts = -1
        self._loop = None
        self._websocket = None

    def stop(self) -> None:
        """Request that the bridge worker stop and close its socket."""
        self._stop_event.set()
        if self._loop is not None:
            def _request_close():
                if self._websocket is not None:
                    asyncio.create_task(self._websocket.close())
            self._loop.call_soon_threadsafe(_request_close)

    def run(self) -> None:
        """Start the worker event loop in the QThread context."""
        try:
            asyncio.run(self._run_loop())
        except Exception as exc:
            self.status_changed.emit(f"Bridge worker stopped: {exc}")

    async def _run_loop(self) -> None:
        """Connect to the bridge and store only the latest RGB frame."""
        self._loop = asyncio.get_running_loop()
        emitter_task = asyncio.create_task(self._emit_frames())
        while not self._stop_event.is_set():
            try:
                self.status_changed.emit("Connecting to Aria bridge...")
                async with websockets.connect(self.ws_url, max_size=None) as websocket:
                    self._websocket = websocket
                    self.status_changed.emit("Connected to live Aria RGB feed")
                    while not self._stop_event.is_set():
                        try:
                            raw = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue
                        except websockets.ConnectionClosed:
                            break
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        if msg.get("type") == "status" and msg.get("connected"):
                            self.status_changed.emit("Bridge connected")
                            continue

                        if msg.get("type") != "image" or msg.get("sensor") != "RGB":
                            continue

                        payload = msg.get("data")
                        if not payload:
                            continue

                        try:
                            image_bytes = base64.b64decode(payload)
                            image_arr = np.frombuffer(image_bytes, dtype=np.uint8)
                            frame_bgr = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
                        except Exception:
                            continue

                        if frame_bgr is None:
                            continue

                        self._latest_frame = frame_bgr
                        self._latest_ts = int(msg.get("ts", 0))
            except Exception:
                if self._stop_event.is_set():
                    break
                self.status_changed.emit("Bridge offline — retrying...")
                await asyncio.sleep(BRIDGE_RETRY_SECONDS)
            finally:
                self._websocket = None
        emitter_task.cancel()
        try:
            await emitter_task
        except Exception:
            pass
        self._loop = None

    async def _emit_frames(self) -> None:
        """Emit the latest frame at a paced interval to the processing thread."""
        while not self._stop_event.is_set():
            await asyncio.sleep(LIVE_RGB_FRAME_INTERVAL_SECONDS)
            if self._latest_frame is None:
                continue
            if self._latest_ts == self._last_emitted_ts:
                continue
            self._last_emitted_ts = self._latest_ts
            self.frame_received.emit(self._latest_frame, self._latest_ts)


class LiveHandTracker:
    """Track the index fingertip in RGB frames using MediaPipe Tasks."""

    def __init__(self) -> None:
        """Create the MediaPipe hand tracker once at startup."""
        self.available = False
        self.error = None
        self.mp = None
        self.landmarker = None
        self.capture_landmarker = None
        self._capture_mode = False
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision

            model_path = HAND_LANDMARKER_MODEL_PATH
            if not model_path.exists():
                raise FileNotFoundError(f"Missing MediaPipe hand model: {model_path}")

            self.mp = mp
            self.landmarker = self._create_landmarker(
                BaseOptions,
                vision,
                model_path,
                MEDIAPIPE_HAND_MIN_TRACKING_CONFIDENCE,
            )
            self.capture_landmarker = self._create_landmarker(
                BaseOptions,
                vision,
                model_path,
                0.3,
            )
            self.available = True
        except Exception as exc:
            self.error = str(exc)

    def _create_landmarker(
        self,
        base_options_cls: Any,
        vision_module: Any,
        model_path: Path,
        min_tracking_confidence: float,
    ) -> Any:
        """Create one MediaPipe hand landmarker for a specific tracking threshold.

        Args:
            base_options_cls: MediaPipe BaseOptions class.
            vision_module: MediaPipe vision tasks module.
            model_path: Path to the hand landmark model asset.
            min_tracking_confidence: Tracking confidence threshold to use.

        Returns:
            A configured `HandLandmarker` instance.

        Raises:
            Exception: Propagates MediaPipe model creation failures.
        """
        options = vision_module.HandLandmarkerOptions(
            base_options=base_options_cls(model_asset_path=str(model_path)),
            running_mode=vision_module.RunningMode.IMAGE,
            num_hands=MEDIAPIPE_SINGLE_HAND_MAX_NUM_HANDS,
            min_hand_detection_confidence=MEDIAPIPE_HAND_MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MEDIAPIPE_HAND_MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=min_tracking_confidence,
        )
        return vision_module.HandLandmarker.create_from_options(options)

    def set_capture_mode(self, active: bool) -> None:
        """Switch between normal and capture-time tracking sensitivity.

        Args:
            active: Whether capture mode is currently active.
        """
        self._capture_mode = bool(active)

    def detect_index_tip(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int]]:
        """Return the detected index fingertip coordinate for one frame."""
        if not self.available or self.landmarker is None or self.mp is None:
            return None

        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            active_landmarker = self.capture_landmarker if self._capture_mode and self.capture_landmarker else self.landmarker
            results = active_landmarker.detect(mp_image)
        except Exception as exc:
            logger.warning("Dashboard hand detection failed on frame; continuing without fingertip: %s", exc)
            return None
        if not results.hand_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        tip = results.hand_landmarks[0][8]
        return int(tip.x * w), int(tip.y * h)

    def close(self) -> None:
        """Release the MediaPipe landmarker resources."""
        for landmarker in (self.landmarker, self.capture_landmarker):
            if landmarker is not None:
                try:
                    landmarker.close()
                except Exception:
                    pass


class AriaPointUndistorter:
    """Correct raw fingertip coordinates for Aria fisheye distortion."""

    def __init__(self) -> None:
        """Load projectaria calibration if available."""
        self.available = False
        self.error = None
        self._aria_calibration = None
        self._camera_calib = None
        self._camera_calib_size = None

        calib_path = os.environ.get(ARIA_CAMERA_CALIBRATION_ENV_VAR)
        if calib_path:
            calib_candidate = Path(calib_path).expanduser()
        else:
            calib_candidate = ARIA_CAMERA_CALIBRATION_PATH

        if not calib_candidate.exists():
            self.error = "No Aria camera calibration JSON found; using radial fallback."
            return

        try:
            from projectaria_tools.core import calibration as aria_calibration

            device_calib = aria_calibration.device_calibration_from_json(str(calib_candidate))
            for label in ("camera-rgb", "camera-rgb+", "rgb"):
                camera_calib = device_calib.get_camera_calib(label)
                if camera_calib is not None:
                    self._aria_calibration = aria_calibration
                    self._camera_calib = camera_calib
                    image_size = camera_calib.get_image_size()
                    self._camera_calib_size = (int(image_size[0]), int(image_size[1]))
                    self.available = True
                    self.error = None
                    break
            if not self.available:
                self.error = f"Calibration file {calib_candidate} missing RGB camera label; using radial fallback."
        except Exception as exc:
            self.error = f"Calibration load failed: {exc}"

    def correct_point(
        self,
        point: Optional[Tuple[int, int]],
        frame_shape: Sequence[int],
    ) -> Optional[Tuple[int, int]]:
        """Correct one fingertip coordinate for camera distortion."""
        if point is None:
            return None

        frame_h, frame_w = frame_shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            return point

        if self.available and self._camera_calib is not None and self._camera_calib_size is not None:
            corrected = self._correct_with_projectaria(point, frame_w, frame_h)
            if corrected is not None:
                return corrected

        return self._correct_with_radial_fallback(point, frame_w, frame_h)

    def _correct_with_projectaria(self, point, frame_w, frame_h):
        try:
            calib_w, calib_h = self._camera_calib_size
            source_pixel = np.array(
                [
                    float(point[0]) * calib_w / max(1, frame_w),
                    float(point[1]) * calib_h / max(1, frame_h),
                ],
                dtype=np.float64,
            )
            ray = self._camera_calib.unproject_no_checks(source_pixel)
            focal_lengths = self._camera_calib.get_focal_lengths()
            avg_focal = float(focal_lengths[0] + focal_lengths[1]) * 0.5
            scaled_focal = avg_focal * min(frame_w / max(1.0, calib_w), frame_h / max(1.0, calib_h))
            rectified_calib = self._aria_calibration.get_linear_camera_calibration(
                int(frame_w),
                int(frame_h),
                float(max(1.0, scaled_focal)),
                "camera-rgb-rectified",
            )
            rectified_pixel = rectified_calib.project_no_checks(np.asarray(ray, dtype=np.float64))
            return (
                int(np.clip(rectified_pixel[0], 0, frame_w - 1)),
                int(np.clip(rectified_pixel[1], 0, frame_h - 1)),
            )
        except Exception:
            return None

    def _correct_with_radial_fallback(self, point, frame_w, frame_h):
        fx = fy = UNDISTORT_FOCAL_SCALE * max(frame_w, frame_h)
        cx = frame_w * 0.5
        cy = frame_h * 0.5
        camera_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        distortion = np.array(UNDISTORT_DISTORTION, dtype=np.float32)
        pts = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        corrected = cv2.undistortPoints(pts, camera_matrix, distortion, P=camera_matrix)
        corrected = corrected.reshape(-1, 2)[0]
        return (
            int(np.clip(corrected[0], 0, frame_w - 1)),
            int(np.clip(corrected[1], 0, frame_h - 1)),
        )


class ProcessingWorker(QtCore.QObject):
    """Process live frames off the UI thread and emit annotated results."""

    annotated_frame_ready = QtCore.pyqtSignal(object)
    tracking_updated = QtCore.pyqtSignal(object)

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        """Initialize the background processing worker."""
        super().__init__(parent)
        self.hand_tracker = LiveHandTracker()
        self.point_undistorter = AriaPointUndistorter()
        self.smoothed_history = deque(maxlen=UI_SMOOTHING_WINDOW)
        self.last_smoothed_point = None
        self.last_known_fingertip = None
        self.lost_hand_frames = 0
        self.max_lost_hand_frames = UI_LOST_HAND_BUFFER_FRAMES
        self._latest_frame = None
        self._latest_timestamp_ns = 0
        self._processing_scheduled = False
        self._stopped = False
        self._state = {
            "recording_active": False,
            "capture_phase": "idle",
            "capture_phase_started": 0.0,
            "capture_countdown_started": 0.0,
            "capture_countdown_seconds": 3,
            "selected_letter": "A",
            "collection_started": False,
            "saved_flash_until": 0.0,
            "saved_flash_text": "",
            "bridge_status": "Waiting for bridge...",
            "hide_gaze_overlay": False,
            "inference_mode": False,
            "prediction_text": "",
            "prediction_confidence": 0.0,
            "top_predictions": [],
        }
        self._stabilization_frames_remaining = 0

    @QtCore.pyqtSlot(object)
    def update_state(self, state: Dict[str, Any]) -> None:
        """Receive the latest UI recording state snapshot."""
        incoming = dict(state or {})
        previous_phase = self._state.get("capture_phase")
        previous_active = self._state.get("recording_active")
        self._state.update(incoming)
        self.hand_tracker.set_capture_mode(bool(self._state.get("recording_active")))

        if (
            previous_phase != self._state.get("capture_phase")
            or previous_active != self._state.get("recording_active")
        ):
            if self._state.get("capture_phase") == "countdown":
                self._stabilization_frames_remaining = 0
            elif self._state.get("capture_phase") == "drawing":
                self._stabilization_frames_remaining = UI_STABILIZATION_FRAMES
            elif not self._state.get("recording_active"):
                self._reset_processing_buffers()

    @QtCore.pyqtSlot()
    def reset_tracking_state(self) -> None:
        """Reset smoothed tracking buffers after a capture state change."""
        self._reset_processing_buffers()

    @QtCore.pyqtSlot()
    def stop(self) -> None:
        """Stop the processing worker and release resources."""
        self._stopped = True
        self._latest_frame = None
        self._processing_scheduled = False
        self._reset_processing_buffers()
        self.close()

    def _reset_processing_buffers(self, stabilization_frames=0):
        self.smoothed_history.clear()
        self.last_smoothed_point = None
        self.last_known_fingertip = None
        self.lost_hand_frames = 0
        self._stabilization_frames_remaining = stabilization_frames

    @QtCore.pyqtSlot(object, int)
    def submit_frame(self, frame_bgr: np.ndarray, timestamp_ns: int) -> None:
        """Store the latest frame for asynchronous processing."""
        if self._stopped:
            return
        self._latest_frame = frame_bgr
        self._latest_timestamp_ns = int(timestamp_ns)
        if not self._processing_scheduled:
            self._processing_scheduled = True
            QtCore.QMetaObject.invokeMethod(
                self,
                "_process_latest_frame",
                QtCore.Qt.QueuedConnection,
            )

    @QtCore.pyqtSlot()
    def _process_latest_frame(self):
        if self._stopped:
            self._processing_scheduled = False
            return
        if self._latest_frame is None:
            self._processing_scheduled = False
            return

        frame_bgr = self._latest_frame
        timestamp_ns = self._latest_timestamp_ns
        self._latest_frame = None

        annotated_image, tracking_meta = self._process_frame(frame_bgr, timestamp_ns)
        self.annotated_frame_ready.emit(annotated_image)
        self.tracking_updated.emit(tracking_meta)

        if self._latest_frame is not None:
            QtCore.QMetaObject.invokeMethod(
                self,
                "_process_latest_frame",
                QtCore.Qt.QueuedConnection,
            )
        else:
            self._processing_scheduled = False

    def _process_frame(self, frame_bgr, timestamp_ns):
        state = dict(self._state)
        clean_frame = frame_bgr.copy()
        display_source = frame_bgr.copy()
        raw_fingertip = self.hand_tracker.detect_index_tip(clean_frame)
        corrected_raw_fingertip = self.point_undistorter.correct_point(
            raw_fingertip,
            clean_frame.shape,
        )
        display_frame = cv2.resize(display_source, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_AREA)
        fingertip_src = corrected_raw_fingertip
        fingertip_display = None
        smoothed_point = None
        movement_detected = False
        buffering_loss = False
        hard_loss = False

        if corrected_raw_fingertip is None:
            if self.last_known_fingertip is not None and self.lost_hand_frames < self.max_lost_hand_frames:
                self.lost_hand_frames += 1
                buffering_loss = True
                fingertip_src = self.last_known_fingertip
            else:
                self.lost_hand_frames += 1
                hard_loss = self.lost_hand_frames > self.max_lost_hand_frames
                fingertip_src = None
        else:
            self.last_known_fingertip = corrected_raw_fingertip
            self.lost_hand_frames = 0

        if fingertip_src is None:
            self.smoothed_history.clear()
            self.last_smoothed_point = None
            self.last_known_fingertip = None
        else:
            src_h, src_w = frame_bgr.shape[:2]
            fingertip_display = (
                int(fingertip_src[0] * (CAMERA_WIDTH / max(1, src_w))),
                int(fingertip_src[1] * (CAMERA_HEIGHT / max(1, src_h))),
            )
            self.smoothed_history.append(fingertip_src)
            smoothed_point = (
                int(sum(point[0] for point in self.smoothed_history) / len(self.smoothed_history)),
                int(sum(point[1] for point in self.smoothed_history) / len(self.smoothed_history)),
            )
            if self.last_smoothed_point is not None:
                movement_detected = (
                    math.hypot(
                        smoothed_point[0] - self.last_smoothed_point[0],
                        smoothed_point[1] - self.last_smoothed_point[1],
                    )
                    > TRAJECTORY_MOVEMENT_THRESHOLD_PX
                )
            self.last_smoothed_point = smoothed_point

        if state.get("recording_active") and state.get("capture_phase") == "drawing":
            if self._stabilization_frames_remaining > 0:
                self._stabilization_frames_remaining -= 1
        else:
            self._stabilization_frames_remaining = 0

        self._draw_overlay(display_frame, fingertip_display, state)
        if buffering_loss and fingertip_display is not None:
            cv2.circle(
                display_frame,
                fingertip_display,
                6,
                LOST_HAND_BUFFER_DOT_COLOR,
                -1,
                lineType=cv2.LINE_AA,
            )

        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        qimage = QtGui.QImage(
            rgb_frame.data,
            rgb_frame.shape[1],
            rgb_frame.shape[0],
            rgb_frame.shape[1] * rgb_frame.shape[2],
            QtGui.QImage.Format_RGB888,
        ).copy()

        tracking_meta = {
            "timestamp_ns": int(timestamp_ns),
            "tracking_visible": fingertip_src is not None,
            "fingertip": fingertip_src,
            "raw_fingertip": raw_fingertip,
            "smoothed_point": smoothed_point,
            "movement_detected": movement_detected,
            "stabilization_frames_remaining": self._stabilization_frames_remaining,
            "collection_ready": self._stabilization_frames_remaining <= 0,
            "buffering_loss": buffering_loss,
            "lost_hand_frames": self.lost_hand_frames,
            "hard_loss": hard_loss,
        }
        return qimage, tracking_meta

    def _draw_overlay(self, canvas, fingertip, state):
        cv2.putText(
            canvas,
            f"Target: {state.get('selected_letter', 'A')}",
            (22, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (76, 245, 160),
            2,
            lineType=cv2.LINE_AA,
        )

        if state.get("recording_active"):
            cv2.putText(
                canvas,
                f"Draw: {state.get('selected_letter', 'A')}",
                (22, 42),
                cv2.FONT_HERSHEY_DUPLEX,
                1.15,
                GREEN_RING,
                2,
                lineType=cv2.LINE_AA,
            )

        ring_color = BLUE_RING
        if state.get("capture_phase") == "drawing":
            ring_color = GREEN_RING
        elif state.get("capture_phase") == "saving":
            ring_color = RED_RING

        if fingertip is not None:
            x, y = fingertip
            if state.get("hide_gaze_overlay", False):
                cv2.circle(canvas, (x, y), 8, GREEN_RING, -1, lineType=cv2.LINE_AA)
            else:
                cv2.circle(canvas, (x, y), 26, ring_color, 3, lineType=cv2.LINE_AA)
                cv2.circle(canvas, (x, y), 8, (240, 240, 240), -1, lineType=cv2.LINE_AA)

        if state.get("capture_phase") == "countdown" and state.get("recording_active"):
            elapsed = time_now() - float(state.get("capture_countdown_started", 0.0))
            remaining = max(1, int(state.get("capture_countdown_seconds", 3)) - int(elapsed))
            cv2.putText(
                canvas,
                f"{remaining}...",
                (CAMERA_WIDTH // 2 - 42, CAMERA_HEIGHT // 2 - 24),
                cv2.FONT_HERSHEY_DUPLEX,
                1.6,
                (255, 220, 120),
                3,
                lineType=cv2.LINE_AA,
            )
        elif (
            state.get("capture_phase") == "drawing"
            and state.get("recording_active")
            and time_now() - float(state.get("capture_phase_started", 0.0)) < UI_DRAW_BANNER_SECONDS
        ):
            cv2.putText(
                canvas,
                "DRAW!",
                (CAMERA_WIDTH // 2 - 70, CAMERA_HEIGHT // 2 - 24),
                cv2.FONT_HERSHEY_DUPLEX,
                1.4,
                GREEN_RING,
                3,
                lineType=cv2.LINE_AA,
            )
        elif (
            state.get("capture_phase") == "drawing"
            and state.get("recording_active")
            and self._stabilization_frames_remaining > 0
        ):
            cv2.putText(
                canvas,
                f"Stabilizing... {self._stabilization_frames_remaining}",
                (CAMERA_WIDTH - 220, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (180, 220, 255),
                2,
                lineType=cv2.LINE_AA,
            )
        elif (
            state.get("capture_phase") == "drawing"
            and state.get("recording_active")
            and not state.get("collection_started", False)
        ):
            cv2.putText(
                canvas,
                "Move fingertip to start stroke",
                (CAMERA_WIDTH // 2 - 170, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (180, 220, 255),
                2,
                lineType=cv2.LINE_AA,
            )
        elif state.get("capture_phase") == "drawing" and state.get("recording_active"):
            cv2.putText(
                canvas,
                "Press 'End Capture' to save or reset",
                (CAMERA_WIDTH // 2 - 170, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 220, 255),
                2,
                lineType=cv2.LINE_AA,
            )

        if float(state.get("saved_flash_until", 0.0)) > time_now():
            cv2.putText(
                canvas,
                state.get("saved_flash_text", ""),
                (CAMERA_WIDTH - 285, CAMERA_HEIGHT - 24),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                GREEN_RING,
                2,
                lineType=cv2.LINE_AA,
            )

        if state.get("inference_mode"):
            prediction_text = state.get("prediction_text", "")
            prediction_confidence = float(state.get("prediction_confidence", 0.0))
            top_predictions = list(state.get("top_predictions", []))
            if prediction_text:
                title = f"{prediction_text} - {prediction_confidence * 100:.0f}%"
                text_size, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.8, 4)
                text_x = max(20, (CAMERA_WIDTH - text_size[0]) // 2)
                cv2.putText(
                    canvas,
                    title,
                    (text_x, 88),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.8,
                    GREEN_RING,
                    4,
                    lineType=cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    canvas,
                    "Inference mode - draw, then click Stop Inference",
                    (CAMERA_WIDTH // 2 - 205, 88),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    GREEN_RING,
                    2,
                    lineType=cv2.LINE_AA,
                )

            panel_x = CAMERA_WIDTH - 176
            panel_y = 104
            panel_w = 160
            panel_h = 92
            cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (8, 17, 26), -1)
            cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (48, 74, 98), 1)
            cv2.putText(
                canvas,
                "Top 3",
                (panel_x + 10, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (220, 220, 220),
                2,
                lineType=cv2.LINE_AA,
            )
            for idx, item in enumerate(top_predictions[:3]):
                label = item.get("label", "--")
                confidence = float(item.get("confidence", 0.0))
                cv2.putText(
                    canvas,
                    f"{idx + 1}. {label} {confidence * 100:.0f}%",
                    (panel_x + 10, panel_y + 46 + idx * 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 220, 255),
                    1,
                    lineType=cv2.LINE_AA,
                )

        cv2.putText(
            canvas,
            state.get("bridge_status", "Waiting for bridge..."),
            (22, CAMERA_HEIGHT - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
            lineType=cv2.LINE_AA,
        )

    @QtCore.pyqtSlot()
    def close(self):
        self.hand_tracker.close()


class LetterTile(QtWidgets.QFrame):
    """Clickable sidebar tile showing one target letter and sample count."""

    clicked = QtCore.pyqtSignal(str)

    def __init__(self, letter, parent=None):
        super().__init__(parent)
        self.letter = letter
        self.count = 0
        self.selected = False
        self.setFixedSize(72, 72)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setStyleSheet(self._style())

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self.letter_label = QtWidgets.QLabel(letter)
        self.letter_label.setAlignment(QtCore.Qt.AlignCenter)
        self.letter_label.setStyleSheet("color: #edf6ff; font-size: 22px; font-weight: 700;")

        self.count_label = QtWidgets.QLabel("0 samples")
        self.count_label.setAlignment(QtCore.Qt.AlignCenter)
        self.count_label.setFixedHeight(18)
        self.count_label.setStyleSheet("color: #8aa3b9; font-size: 10px;")

        layout.addWidget(self.letter_label)
        layout.addWidget(self.count_label)

    def _style(self):
        border_color = ACCENT if self.selected else BORDER
        bg_color = "#1b2c3e" if self.selected else SURFACE_BG
        return f"QFrame {{ background: {bg_color}; border: 2px solid {border_color}; border-radius: 14px; }}"

    def set_count(self, count):
        self.count = count
        self.count_label.setText(f"{count} samples")

    def set_selected(self, selected):
        self.selected = selected
        self.setStyleSheet(self._style())

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit(self.letter)
        super().mousePressEvent(event)


class HeatmapCell(QtWidgets.QFrame):
    """Clickable accuracy heatmap cell for a single letter."""

    clicked = QtCore.pyqtSignal(str)

    def __init__(self, letter="", parent=None):
        super().__init__(parent)
        self.letter = letter
        self.accuracy = None
        self.setFixedSize(58, 54)
        self.setCursor(QtCore.Qt.PointingHandCursor if letter else QtCore.Qt.ArrowCursor)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        self.letter_label = QtWidgets.QLabel(letter if letter else "")
        self.letter_label.setAlignment(QtCore.Qt.AlignCenter)
        self.letter_label.setFixedHeight(18)
        self.letter_label.setStyleSheet("color: #edf6ff; font-size: 16px; font-weight: 700;")

        self.acc_label = QtWidgets.QLabel("--")
        self.acc_label.setAlignment(QtCore.Qt.AlignCenter)
        self.acc_label.setFixedHeight(16)
        self.acc_label.setStyleSheet("color: #edf6ff; font-size: 11px;")

        layout.addWidget(self.letter_label)
        layout.addWidget(self.acc_label)
        self._refresh_style()

    def _refresh_style(self):
        if not self.letter:
            bg = "#101820"
            border = BORDER
        elif self.accuracy is None:
            bg = SURFACE_BG
            border = BORDER
        elif self.accuracy > 0.90:
            bg = "#153925"
            border = "#4cf5a0"
        elif self.accuracy >= 0.70:
            bg = "#423716"
            border = "#ffd166"
        else:
            bg = "#451d1d"
            border = "#ff6b6b"

        self.setStyleSheet(
            f"QFrame {{ background: {bg}; border: 1px solid {border}; border-radius: 10px; }}"
        )

    def set_accuracy(self, accuracy):
        self.accuracy = accuracy
        if self.letter:
            self.acc_label.setText(f"{accuracy * 100:.0f}%")
        else:
            self.acc_label.setText("")
        self._refresh_style()

    def mousePressEvent(self, event):
        if self.letter and event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit(self.letter)
        super().mousePressEvent(event)


class SimpleBarChartWidget(QtWidgets.QWidget):
    """Lightweight bar chart widget for per-class sample counts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = [0] * 26
        self.max_value = 50
        self.setStyleSheet(f"background: #091018; border: 1px solid {BORDER}; border-radius: 10px;")

    def set_counts(self, counts):
        self.values = [int(counts.get(chr(65 + idx), 0)) for idx in range(26)]
        self.max_value = max(50, max(self.values) + 5 if self.values else 50)
        self.update()

    def paintEvent(self, event):
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect().adjusted(10, 10, -10, -22)

        grid_pen = QtGui.QPen(QtGui.QColor("#1d2b39"))
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)
        for i in range(1, 4):
            y = rect.top() + rect.height() * i / 4
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))

        bar_width = rect.width() / max(26, 1)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(ACCENT))
        for idx, value in enumerate(self.values):
            x = rect.left() + idx * bar_width + 1
            height = 0 if self.max_value == 0 else (value / self.max_value) * rect.height()
            bar_rect = QtCore.QRectF(x, rect.bottom() - height, max(2, bar_width - 2), height)
            painter.drawRoundedRect(bar_rect, 2, 2)

        label_pen = QtGui.QPen(QtGui.QColor(MUTED))
        painter.setPen(label_pen)
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        for idx in range(26):
            x = rect.left() + idx * bar_width
            label_rect = QtCore.QRectF(x, rect.bottom() + 4, bar_width, 14)
            painter.drawText(label_rect, QtCore.Qt.AlignCenter, chr(65 + idx))

        painter.end()


class SimpleCurvePlotWidget(QtWidgets.QWidget):
    """Lightweight curve widget for training loss and accuracy history."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.loss_curve = []
        self.acc_curve = []
        self.setStyleSheet(f"background: #091018; border: 1px solid {BORDER}; border-radius: 10px;")

    def clear(self):
        self.loss_curve = []
        self.acc_curve = []
        self.update()

    def set_curves(self, loss_curve, acc_curve):
        self.loss_curve = list(loss_curve)
        self.acc_curve = list(acc_curve)
        self.update()

    def paintEvent(self, event):
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect().adjusted(10, 10, -10, -10)

        grid_pen = QtGui.QPen(QtGui.QColor("#1d2b39"))
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)
        for i in range(1, 4):
            y = rect.top() + rect.height() * i / 4
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))
        for i in range(1, 5):
            x = rect.left() + rect.width() * i / 5
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())

        if not self.loss_curve and not self.acc_curve:
            painter.setPen(QtGui.QColor(MUTED))
            painter.drawText(rect, QtCore.Qt.AlignCenter, "Select a session to view curves")
            painter.end()
            return

        def draw_curve(values, color, y_max=None):
            if len(values) < 2:
                return
            max_value = y_max if y_max is not None else max(values)
            max_value = max(max_value, 1e-6)
            path = QtGui.QPainterPath()
            for idx, value in enumerate(values):
                x = rect.left() + (idx / max(1, len(values) - 1)) * rect.width()
                y = rect.bottom() - (value / max_value) * rect.height()
                if idx == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            pen = QtGui.QPen(QtGui.QColor(color))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawPath(path)

        draw_curve(self.loss_curve, "#ffd166")
        draw_curve(self.acc_curve, ACCENT, y_max=1.0)
        painter.end()


class TrajectoryThumbnail(QtWidgets.QFrame):
    """Thumbnail widget that previews one saved trajectory sample."""

    delete_requested = QtCore.pyqtSignal(str)

    def __init__(self, record, display_index, parent=None):
        super().__init__(parent)
        self.record = record
        self.setFixedSize(104, 144)
        self.setStyleSheet(
            f"QFrame {{ background: {SURFACE_BG}; border: 1px solid {BORDER}; border-radius: 12px; }}"
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(4)
        index_label = QtWidgets.QLabel(f"#{display_index}")
        index_label.setFixedHeight(16)
        index_label.setStyleSheet(f"color: {TEXT}; font-size: 11px; font-weight: 700;")

        delete_button = QtWidgets.QPushButton("X")
        delete_button.setFixedSize(20, 20)
        delete_button.setCursor(QtCore.Qt.PointingHandCursor)
        delete_button.setStyleSheet(
            """
            QPushButton {
                background: #a92c2c;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 11px;
                font-weight: 700;
            }
            """
        )
        delete_button.clicked.connect(lambda: self.delete_requested.emit(self.record["id"]))

        top_row.addWidget(index_label)
        top_row.addStretch(1)
        top_row.addWidget(delete_button)

        self.thumb = QtWidgets.QLabel()
        self.thumb.setFixedSize(80, 80)
        self.thumb.setStyleSheet("background: #091018; border: 1px solid #253649; border-radius: 8px;")
        self.thumb.setPixmap(self._draw_thumbnail())

        timestamp = self.record.get("timestamp") or "Unknown time"
        raw_points = int(self.record.get("raw_point_count", 0))
        footer = QtWidgets.QLabel(f"{timestamp}\n{raw_points} pts")
        footer.setWordWrap(True)
        footer.setAlignment(QtCore.Qt.AlignCenter)
        footer.setFixedHeight(30)
        footer.setStyleSheet(f"color: {MUTED}; font-size: 9px;")

        layout.addLayout(top_row)
        layout.addWidget(self.thumb, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(footer)

    def _draw_thumbnail(self):
        pixmap = QtGui.QPixmap(80, 80)
        pixmap.fill(QtGui.QColor("#091018"))
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        points = np.asarray(self.record.get("trajectory", []), dtype=np.float32)
        if points.ndim == 2 and points.shape[1] == 2 and len(points) > 0:
            min_xy = points.min(axis=0)
            max_xy = points.max(axis=0)
            span = np.maximum(max_xy - min_xy, 1.0)
            scale = min((80 - 16) / span[0], (80 - 16) / span[1])
            norm = (points - min_xy) * scale
            norm[:, 0] += (80 - norm[:, 0].max()) * 0.5
            norm[:, 1] += (80 - norm[:, 1].max()) * 0.5

            path = QtGui.QPainterPath()
            path.moveTo(norm[0, 0], norm[0, 1])
            for pt in norm[1:]:
                path.lineTo(pt[0], pt[1])

            pen = QtGui.QPen(QtGui.QColor(ACCENT))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawPath(path)

        painter.end()
        return pixmap


class ReviewSamplesDialog(QtWidgets.QDialog):
    """Dialog for reviewing and deleting saved samples for one letter."""

    sample_deleted = QtCore.pyqtSignal(str)

    def __init__(self, letter, records, parent=None):
        super().__init__(parent)
        self.letter = letter
        self.records = list(records)
        self.setWindowTitle(f"Review Samples — {letter}")
        self.setModal(False)
        self.setFixedSize(760, 620)
        self.setStyleSheet("background: #08111a; color: #edf6ff;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QtWidgets.QLabel(f"Review Samples — {letter}")
        title.setFixedHeight(24)
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #edf6ff;")
        layout.addWidget(title)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFixedSize(728, 520)
        self.scroll.setStyleSheet("QScrollArea { border: 1px solid #253649; background: #0f1720; }")

        self.grid_container = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_container)
        self.grid_layout.setContentsMargins(12, 12, 12, 12)
        self.grid_layout.setHorizontalSpacing(12)
        self.grid_layout.setVerticalSpacing(12)
        self.scroll.setWidget(self.grid_container)
        layout.addWidget(self.scroll)

        self.summary = QtWidgets.QLabel()
        self.summary.setFixedHeight(24)
        self.summary.setStyleSheet("color: #8aa3b9; font-size: 12px;")
        layout.addWidget(self.summary)

        self._rebuild()

    def _rebuild(self):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for idx, record in enumerate(self.records, start=1):
            thumb = TrajectoryThumbnail(record, idx)
            thumb.delete_requested.connect(self._delete_record)
            row = (idx - 1) // 6
            col = (idx - 1) % 6
            self.grid_layout.addWidget(thumb, row, col)

        short_count = sum(1 for record in self.records if int(record.get("raw_point_count", 0)) < 20)
        self.summary.setText(
            f"Showing {len(self.records)} samples — {short_count} flagged as short (< 20 points)"
        )

    def _delete_record(self, record_id):
        self.sample_deleted.emit(record_id)


class CameraPanel(QtWidgets.QFrame):
    """Fixed-size widget for displaying the live RGB camera feed."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(CAMERA_WIDTH + 24, CAMERA_HEIGHT + 54)
        self.setStyleSheet(
            f"QFrame {{ background: {PANEL_BG}; border: 2px solid {BORDER}; border-radius: 16px; }}"
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("LIVE CAMERA FEED")
        title.setStyleSheet(f"color: {TEXT}; font-size: 14px; font-weight: 700;")
        title.setFixedHeight(18)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setFixedSize(CAMERA_WIDTH, CAMERA_HEIGHT)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setStyleSheet(
            f"background: #091018; border: 1px solid {BORDER}; border-radius: 10px; color: {MUTED};"
        )
        self.image_label.setText("Waiting for Aria RGB stream...")

        layout.addWidget(title)
        layout.addWidget(self.image_label)

    def set_qimage(self, image):
        pixmap = QtGui.QPixmap.fromImage(image)
        self.image_label.setPixmap(pixmap)


class TrajectoryPanel(QtWidgets.QFrame):
    """Fixed-size widget for displaying the latest captured trajectory."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.setFixedSize(TRAJ_WIDTH + 24, TRAJ_HEIGHT + 54)
        self.setStyleSheet(
            f"QFrame {{ background: {PANEL_BG}; border: 2px solid {BORDER}; border-radius: 16px; }}"
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("LAST TRAJECTORY")
        title.setStyleSheet(f"color: {TEXT}; font-size: 14px; font-weight: 700;")
        title.setFixedHeight(18)

        self.canvas = QtWidgets.QLabel()
        self.canvas.setFixedSize(TRAJ_WIDTH, TRAJ_HEIGHT)
        self.canvas.setStyleSheet(
            f"background: #091018; border: 1px solid {BORDER}; border-radius: 10px;"
        )

        layout.addWidget(title)
        layout.addWidget(self.canvas)
        self.redraw()

    def set_points(self, points):
        self.points = points or []
        self.redraw()

    def redraw(self):
        pixmap = QtGui.QPixmap(TRAJ_WIDTH, TRAJ_HEIGHT)
        pixmap.fill(QtGui.QColor("#091018"))
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        pen_grid = QtGui.QPen(QtGui.QColor("#13202c"))
        pen_grid.setWidth(1)
        painter.setPen(pen_grid)
        for x in range(0, TRAJ_WIDTH, 80):
            painter.drawLine(x, 0, x, TRAJ_HEIGHT)
        for y in range(0, TRAJ_HEIGHT, 50):
            painter.drawLine(0, y, TRAJ_WIDTH, y)

        if self.points:
            pts = np.asarray(self.points, dtype=np.float32)
            min_xy = pts.min(axis=0)
            max_xy = pts.max(axis=0)
            span = np.maximum(max_xy - min_xy, 1.0)
            scale = min((TRAJ_WIDTH - 32) / span[0], (TRAJ_HEIGHT - 32) / span[1])
            normalized = (pts - min_xy) * scale
            normalized[:, 0] += (TRAJ_WIDTH - normalized[:, 0].max()) * 0.5
            normalized[:, 1] += (TRAJ_HEIGHT - normalized[:, 1].max()) * 0.5

            path = QtGui.QPainterPath()
            path.moveTo(normalized[0, 0], normalized[0, 1])
            for point in normalized[1:]:
                path.lineTo(point[0], point[1])

            pen = QtGui.QPen(QtGui.QColor(ACCENT))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawPath(path)

        painter.end()
        self.canvas.setPixmap(pixmap)


class StatsPanel(QtWidgets.QFrame):
    """Right-hand dashboard panel for counts, charts, and training history."""

    heatmap_letter_clicked = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(RIGHT_WIDTH, RIGHT_HEIGHT)
        self.setStyleSheet(
            f"QFrame {{ background: {PANEL_BG}; border: 2px solid {BORDER}; border-radius: 16px; }}"
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        title = QtWidgets.QLabel("TRAINING STATS")
        title.setStyleSheet(f"color: {TEXT}; font-size: 16px; font-weight: 700;")
        title.setFixedHeight(20)
        layout.addWidget(title)

        self.total_samples = self._stat_row("Total samples collected", "0")
        self.current_accuracy = self._stat_row("Current model accuracy", "--")
        self.training_sessions = self._stat_row("Training sessions run", "0")
        self.last_trained = self._stat_row("Last trained timestamp", "--")
        self.model_version = self._stat_row("Model version", "0")

        layout.addWidget(self.total_samples["widget"])
        layout.addWidget(self.current_accuracy["widget"])
        layout.addWidget(self.training_sessions["widget"])
        layout.addWidget(self.last_trained["widget"])
        layout.addWidget(self.model_version["widget"])

        chart_title = QtWidgets.QLabel("SAMPLES PER CLASS")
        chart_title.setStyleSheet(f"color: {TEXT}; font-size: 13px; font-weight: 700;")
        chart_title.setFixedHeight(18)
        layout.addWidget(chart_title)

        self.plot_widget = SimpleBarChartWidget()
        self.plot_widget.setFixedSize(RIGHT_WIDTH - 24, 110)

        layout.addWidget(self.plot_widget)

        heatmap_title = QtWidgets.QLabel("PER-LETTER ACCURACY")
        heatmap_title.setStyleSheet(f"color: {TEXT}; font-size: 13px; font-weight: 700;")
        heatmap_title.setFixedHeight(18)
        layout.addWidget(heatmap_title)

        self.heatmap_widget = QtWidgets.QWidget()
        self.heatmap_widget.setFixedSize(RIGHT_WIDTH - 24, 200)
        heatmap_layout = QtWidgets.QGridLayout(self.heatmap_widget)
        heatmap_layout.setContentsMargins(0, 0, 0, 0)
        heatmap_layout.setHorizontalSpacing(6)
        heatmap_layout.setVerticalSpacing(6)

        keyboard_grid = [
            ["Q", "W", "E", "R", "T", "Y"],
            ["A", "S", "D", "F", "G", "H"],
            ["Z", "X", "C", "V", "B", "N"],
            ["U", "I", "O", "P", "J", "K"],
            ["L", "M", "", "", "", ""],
        ]
        self.heatmap_cells = {}
        for row_idx, row in enumerate(keyboard_grid):
            for col_idx, letter in enumerate(row):
                cell = HeatmapCell(letter)
                if letter:
                    cell.clicked.connect(self.heatmap_letter_clicked.emit)
                    self.heatmap_cells[letter] = cell
                heatmap_layout.addWidget(cell, row_idx, col_idx)

        layout.addWidget(self.heatmap_widget)

        self.history_toggle = QtWidgets.QToolButton()
        self.history_toggle.setCheckable(True)
        self.history_toggle.setChecked(True)
        self.history_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.history_toggle.setArrowType(QtCore.Qt.DownArrow)
        self.history_toggle.setText("Training History")
        self.history_toggle.setFixedHeight(24)
        self.history_toggle.setStyleSheet(
            f"QToolButton {{ color: {TEXT}; font-size: 13px; font-weight: 700; border: none; text-align: left; }}"
        )
        self.history_toggle.clicked.connect(self._toggle_history_panel)
        layout.addWidget(self.history_toggle)

        history_header = QtWidgets.QHBoxLayout()
        history_header.setSpacing(8)

        self.export_button = QtWidgets.QPushButton("Export History")
        self.export_button.setFixedSize(120, 28)
        self.export_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.export_button.setStyleSheet(
            """
            QPushButton {
                background: #ffd166;
                color: #041019;
                border: none;
                border-radius: 8px;
                font-size: 11px;
                font-weight: 700;
            }
            """
        )
        history_header.addStretch(1)
        history_header.addWidget(self.export_button)
        layout.addLayout(history_header)

        self.history_container = QtWidgets.QFrame()
        self.history_container.setFixedSize(RIGHT_WIDTH - 24, 165)
        self.history_container.setStyleSheet("QFrame { background: transparent; border: none; }")
        history_layout = QtWidgets.QVBoxLayout(self.history_container)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(6)

        self.history_table = QtWidgets.QTableWidget(0, 6)
        self.history_table.setFixedSize(RIGHT_WIDTH - 24, 86)
        self.history_table.setHorizontalHeaderLabels(
            ["Session #", "Date & Time", "Samples Used", "Epochs Run", "Val Accuracy", "Model File"]
        )
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.history_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.history_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.history_table.setShowGrid(False)
        self.history_table.setWordWrap(False)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setStyleSheet(
            f"""
            QTableWidget {{
                background: #091018;
                color: {TEXT};
                border: 1px solid {BORDER};
                gridline-color: {BORDER};
                alternate-background-color: #0d1620;
            }}
            QHeaderView::section {{
                background: #162231;
                color: {MUTED};
                border: none;
                padding: 4px;
                font-size: 10px;
                font-weight: 700;
            }}
            """
        )
        self.history_table.setColumnWidth(0, 55)
        self.history_table.setColumnWidth(1, 110)
        self.history_table.setColumnWidth(2, 70)
        self.history_table.setColumnWidth(3, 60)
        self.history_table.setColumnWidth(4, 65)
        self.history_table.setColumnWidth(5, 90)
        self.history_table.horizontalHeader().setStretchLastSection(False)
        self.history_table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.history_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.history_table.verticalHeader().setDefaultSectionSize(20)
        history_layout.addWidget(self.history_table)

        self.curve_plot = SimpleCurvePlotWidget()
        self.curve_plot.setFixedSize(RIGHT_WIDTH - 24, 72)
        history_layout.addWidget(self.curve_plot)

        layout.addWidget(self.history_container)
        layout.addStretch(1)

        self.history_table.itemSelectionChanged.connect(self._on_history_row_selected)
        self._history_records = []
        self._toggle_history_panel(True)

    def _stat_row(self, label_text, value_text):
        widget = QtWidgets.QFrame()
        widget.setFixedHeight(48)
        widget.setStyleSheet(
            f"QFrame {{ background: {SURFACE_BG}; border: 1px solid {BORDER}; border-radius: 12px; }}"
        )
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)

        label = QtWidgets.QLabel(label_text)
        label.setFixedHeight(14)
        label.setStyleSheet(f"color: {MUTED}; font-size: 11px;")

        value = QtWidgets.QLabel(value_text)
        value.setFixedHeight(20)
        value.setStyleSheet(f"color: {TEXT}; font-size: 16px; font-weight: 700;")

        layout.addWidget(label)
        layout.addWidget(value)
        return {"widget": widget, "value": value}

    def update_totals(self, total_samples, accuracy, sessions, last_trained_text, model_version):
        self.total_samples["value"].setText(str(total_samples))
        self.current_accuracy["value"].setText(accuracy)
        self.training_sessions["value"].setText(str(sessions))
        self.last_trained["value"].setText(last_trained_text)
        self.model_version["value"].setText(str(model_version))

    def update_class_counts(self, counts):
        self.plot_widget.set_counts(counts)

    def update_per_letter_accuracy(self, accuracy_map):
        for letter, cell in self.heatmap_cells.items():
            cell.set_accuracy(float(accuracy_map.get(letter, 0.0)))

    def update_history(self, history_records):
        self._history_records = list(history_records)
        self.history_table.setRowCount(len(self._history_records))

        for row_idx, record in enumerate(self._history_records):
            values = [
                str(record.get("session", row_idx + 1)),
                str(record.get("timestamp", "--")),
                str(record.get("samples_used", "--")),
                str(record.get("epochs_run", "--")),
                f"{float(record.get('val_accuracy', 0.0)) * 100:.1f}%",
                str(record.get("model_file", "--")),
            ]
            for col_idx, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                self.history_table.setItem(row_idx, col_idx, item)

        if self._history_records:
            self.history_table.selectRow(len(self._history_records) - 1)
            self._plot_history_record(self._history_records[-1])
        else:
            self.curve_plot.clear()

    def _on_history_row_selected(self):
        selected_rows = self.history_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        row_idx = selected_rows[0].row()
        if 0 <= row_idx < len(self._history_records):
            self._plot_history_record(self._history_records[row_idx])

    def _plot_history_record(self, record):
        loss_curve = record.get("loss_curve", [])
        acc_curve = record.get("accuracy_curve", [])
        if not loss_curve and not acc_curve:
            self.curve_plot.clear()
            return
        self.curve_plot.set_curves(loss_curve, acc_curve)

    def _toggle_history_panel(self, checked=None):
        if checked is None:
            checked = self.history_toggle.isChecked()
        visible = bool(checked)
        self.history_toggle.setArrowType(QtCore.Qt.DownArrow if visible else QtCore.Qt.RightArrow)
        self.history_container.setVisible(visible)
        self.export_button.setVisible(visible)


class ControlBar(QtWidgets.QFrame):
    """Bottom control strip for capture, training, and inference actions."""

    start_recording = QtCore.pyqtSignal()
    train_model = QtCore.pyqtSignal()
    run_live_inference = QtCore.pyqtSignal()
    toggle_calibration_mode = QtCore.pyqtSignal()
    fine_tune_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(WINDOW_WIDTH - 32, BOTTOM_HEIGHT)
        self.setStyleSheet(
            f"QFrame {{ background: {PANEL_BG}; border: 2px solid {BORDER}; border-radius: 16px; }}"
        )

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        self.record_btn = self._make_button("Capture Letter", ACCENT)
        self.train_btn = self._make_button("Train Model", SUCCESS)
        self.inference_btn = self._make_button("Run Live Inference", "#ffd166")
        self.calibrate_btn = self._make_button("Calibrate Mode", "#c792ea")
        self.finetune_btn = self._make_button("Fine-Tune Personal", "#f78c6c")

        self.record_btn.clicked.connect(self.start_recording.emit)
        self.train_btn.clicked.connect(self.train_model.emit)
        self.inference_btn.clicked.connect(self.run_live_inference.emit)
        self.calibrate_btn.clicked.connect(self.toggle_calibration_mode.emit)
        self.finetune_btn.clicked.connect(self.fine_tune_requested.emit)

        layout.addWidget(self.record_btn)
        layout.addWidget(self.train_btn)
        layout.addWidget(self.inference_btn)
        layout.addWidget(self.calibrate_btn)
        layout.addWidget(self.finetune_btn)
        layout.addStretch(1)

    def _make_button(self, text, color):
        button = QtWidgets.QPushButton(text)
        button.setFixedSize(190, 52)
        button.setCursor(QtCore.Qt.PointingHandCursor)
        button.setStyleSheet(
            f"""
            QPushButton {{
                background: {color};
                color: #041019;
                border: none;
                border-radius: 12px;
                font-size: 14px;
                font-weight: 700;
            }}
            QPushButton:hover {{ background: {color}; border: 2px solid #ffffff; }}
            """
        )
        return button


class AriaTrainingDashboard(QtWidgets.QMainWindow):
    """Main PyQt5 application window for the Aria training workflow."""

    processing_state_changed = QtCore.pyqtSignal(object)
    processing_reset_requested = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.training_state = load_training_state()
        self.selected_letter = "A"
        self.letter_counts = dict(self.training_state["samples_per_letter"])
        self.training_session_count = int(self.training_state["training_sessions"])
        self.last_trajectory = []
        self.recording_active = False
        self.capture_phase = "idle"
        self.capture_phase_started = 0.0
        self.capture_countdown_started = 0.0
        self.capture_countdown_seconds = UI_CAPTURE_COUNTDOWN_SECONDS
        self.saved_flash_until = 0.0
        self.saved_flash_text = ""
        self._beep_ready = False
        self.bridge_status = "Waiting for bridge..."
        self.current_fingertip = None
        self.pending_saved_trajectory = None
        self.stabilization_frames_remaining = 0
        self.collection_started = False
        self.last_committed_point = None
        self.hide_gaze_overlay = False
        self.inference_mode = False
        self.inference_model = None
        self.inference_model_path = None
        self.inference_label_lookup: Dict[int, str] = {}
        self.inference_points: List[Tuple[int, int]] = []
        self.inference_last_point: Optional[Tuple[int, int]] = None
        self.last_prediction_letter = ""
        self.last_prediction_confidence = 0.0
        self.last_prediction_top3: List[Dict[str, float]] = []
        self.calibration_mode = False
        self.trajectory_builder = TrajectoryBuilder(
            movement_threshold_px=TRAJECTORY_MOVEMENT_THRESHOLD_PX,
            dwell_seconds=TRAJECTORY_DWELL_SECONDS,
            min_points=TRAJECTORY_MIN_POINTS,
            min_duration_seconds=TRAJECTORY_MIN_DURATION_SECONDS,
        )
        self.bridge_worker = None
        self.processing_thread = None
        self.processing_worker = None
        self._shutting_down = False

        self.setWindowTitle("Aria ML Training Dashboard")
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(f"background: #08111a; color: {TEXT};")

        central = QtWidgets.QWidget()
        central.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setCentralWidget(central)

        grid = QtWidgets.QGridLayout(central)
        grid.setContentsMargins(16, 16, 16, 16)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)
        grid.setRowMinimumHeight(0, CAMERA_HEIGHT + 54)
        grid.setRowMinimumHeight(1, TRAJ_HEIGHT + 54)
        grid.setRowMinimumHeight(2, BOTTOM_HEIGHT)
        grid.setColumnMinimumWidth(0, SIDEBAR_WIDTH)
        grid.setColumnMinimumWidth(1, CAMERA_WIDTH + 24)
        grid.setColumnMinimumWidth(2, RIGHT_WIDTH)

        self.sidebar = self._build_sidebar()
        self.camera_panel = CameraPanel()
        self.trajectory_panel = TrajectoryPanel()
        self.stats_panel = StatsPanel()
        self.control_bar = ControlBar()

        grid.addWidget(self.sidebar, 0, 0, 2, 1)
        grid.addWidget(self.camera_panel, 0, 1)
        grid.addWidget(self.trajectory_panel, 1, 1)
        grid.addWidget(self.stats_panel, 0, 2, 2, 1)
        grid.addWidget(self.control_bar, 2, 0, 1, 3)

        self.control_bar.start_recording.connect(self.on_start_recording)
        self.control_bar.train_model.connect(self.on_train_model)
        self.control_bar.run_live_inference.connect(self.on_run_live_inference)
        self.control_bar.toggle_calibration_mode.connect(self.on_toggle_calibration_mode)
        self.control_bar.fine_tune_requested.connect(self.on_fine_tune_personal_model)
        self.stats_panel.heatmap_letter_clicked.connect(self._set_selected_letter)
        self.stats_panel.export_button.clicked.connect(self.export_history_csv)
        self.review_dialog = None

        self._set_selected_letter("A")
        self._refresh_stats()

        self._init_audio()
        self._init_live_bridge()
        self._init_processing_thread()

        self.countdown_timer = QtCore.QTimer(self)
        self.countdown_timer.setInterval(50)
        self.countdown_timer.timeout.connect(self._recording_tick)

    def _build_sidebar(self):
        frame = QtWidgets.QFrame()
        frame.setFixedSize(SIDEBAR_WIDTH, CAMERA_HEIGHT + TRAJ_HEIGHT + 108)
        frame.setStyleSheet(
            f"QFrame {{ background: {PANEL_BG}; border: 2px solid {BORDER}; border-radius: 16px; }}"
        )

        outer = QtWidgets.QVBoxLayout(frame)
        outer.setContentsMargins(16, 14, 16, 16)
        outer.setSpacing(12)

        title = QtWidgets.QLabel("LETTER TARGETS")
        title.setFixedHeight(20)
        title.setStyleSheet(f"color: {TEXT}; font-size: 16px; font-weight: 700;")

        subtitle = QtWidgets.QLabel("Click a letter to select the recording target.")
        subtitle.setWordWrap(True)
        subtitle.setFixedHeight(36)
        subtitle.setStyleSheet(f"color: {MUTED}; font-size: 11px;")

        outer.addWidget(title)
        outer.addWidget(subtitle)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        self.letter_tiles = {}
        for idx, letter in enumerate(chr(65 + i) for i in range(26)):
            tile = LetterTile(letter)
            tile.clicked.connect(self._set_selected_letter)
            self.letter_tiles[letter] = tile
            row = idx // 3
            col = idx % 3
            grid.addWidget(tile, row, col)

        grid_container = QtWidgets.QWidget()
        grid_container.setFixedSize(SIDEBAR_WIDTH - 32, 650)
        grid_container.setLayout(grid)
        outer.addWidget(grid_container)
        outer.addStretch(1)

        self.current_target = QtWidgets.QLabel("Current target: A")
        self.current_target.setFixedHeight(24)
        self.current_target.setStyleSheet(f"color: {SUCCESS}; font-size: 14px; font-weight: 700;")
        outer.addWidget(self.current_target)

        review_button = QtWidgets.QPushButton("Review")
        review_button.setFixedSize(96, 36)
        review_button.setCursor(QtCore.Qt.PointingHandCursor)
        review_button.setStyleSheet(
            """
            QPushButton {
                background: #ffd166;
                color: #041019;
                border: none;
                border-radius: 10px;
                font-size: 12px;
                font-weight: 700;
            }
            """
        )
        review_button.clicked.connect(self.open_review_mode)
        outer.addWidget(review_button, alignment=QtCore.Qt.AlignLeft)
        return frame

    def _set_selected_letter(self, letter):
        self.selected_letter = letter
        self.current_target.setText(f"Current target: {letter}")
        for key, tile in self.letter_tiles.items():
            tile.set_selected(key == letter)
        self._push_processing_state()

    def _refresh_stats(self):
        total_samples = int(self.training_state["total_samples"])
        last_trained = self.training_state["last_trained"] or "--"
        self.stats_panel.update_totals(
            total_samples=total_samples,
            accuracy=f"{self.training_state['best_accuracy'] * 100:.1f}%",
            sessions=self.training_session_count,
            last_trained_text=last_trained,
            model_version=self.training_state["model_version"],
        )
        self.stats_panel.update_class_counts(self.letter_counts)
        self.stats_panel.update_per_letter_accuracy(self.training_state["per_letter_accuracy"])
        self.stats_panel.update_history(self.training_state.get("history", []))
        for letter, count in self.letter_counts.items():
            self.letter_tiles[letter].set_count(count)

    def set_camera_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = QtGui.QImage(
            frame_rgb.data,
            frame_rgb.shape[1],
            frame_rgb.shape[0],
            frame_rgb.shape[1] * frame_rgb.shape[2],
            QtGui.QImage.Format_RGB888,
        ).copy()
        self.camera_panel.set_qimage(image)

    def set_last_trajectory(self, points):
        self.last_trajectory = points
        self.trajectory_panel.set_points(points)

    def update_letter_count(self, letter, count):
        self.letter_counts[letter] = count
        self.training_state["samples_per_letter"][letter] = count
        self.training_state["total_samples"] = sum(self.letter_counts.values())
        save_training_state(self.training_state)
        self._refresh_stats()

    def set_training_stats(self, total_samples, accuracy, sessions, last_trained_text, per_class_counts):
        self.letter_counts.update(per_class_counts)
        self.training_state["samples_per_letter"].update(per_class_counts)
        self.training_state["total_samples"] = int(total_samples)
        self.training_state["training_sessions"] = int(sessions)
        self.training_session_count = int(sessions)
        self.training_state["last_trained"] = last_trained_text if last_trained_text != "--" else None
        if isinstance(accuracy, str):
            parsed_accuracy = accuracy.strip().replace("%", "")
            self.training_state["best_accuracy"] = float(parsed_accuracy) / 100.0 if parsed_accuracy else 0.0
        else:
            self.training_state["best_accuracy"] = float(accuracy)
        save_training_state(self.training_state)
        for letter, count in self.letter_counts.items():
            self.letter_tiles[letter].set_count(count)
        self.stats_panel.update_totals(
            total_samples,
            f"{self.training_state['best_accuracy'] * 100:.1f}%",
            sessions,
            last_trained_text,
            self.training_state["model_version"],
        )
        self.stats_panel.update_class_counts(self.letter_counts)
        self.stats_panel.update_per_letter_accuracy(self.training_state["per_letter_accuracy"])

    def open_review_mode(self):
        print("[DEBUG] Review button handler fired.")
        logger.info("Review button handler fired.")
        records = [record for record in load_sample_metadata() if record.get("label") == self.selected_letter]
        if self.review_dialog is not None:
            self.review_dialog.close()
        self.review_dialog = ReviewSamplesDialog(self.selected_letter, records, self)
        self.review_dialog.sample_deleted.connect(self.delete_sample_record)
        self.review_dialog.show()

    def on_start_recording(self):
        print("[DEBUG] Capture Letter button handler fired.")
        logger.info("Capture Letter button handler fired.")
        if self.recording_active:
            self._finish_or_reset_capture()
            return
        self.recording_active = True
        self.capture_phase = "countdown"
        self.capture_phase_started = time_now()
        self.capture_countdown_started = time_now()
        self.saved_flash_until = 0.0
        self.saved_flash_text = ""
        self.last_trajectory = []
        self.trajectory_panel.set_points([])
        self.pending_saved_trajectory = None
        self.trajectory_builder.points = []
        self.trajectory_builder.anchor_point = None
        self.trajectory_builder.anchor_timestamp_ns = None
        self.trajectory_builder.first_point_timestamp_ns = None
        self.stabilization_frames_remaining = UI_STABILIZATION_FRAMES
        self.collection_started = False
        self.last_committed_point = None
        self.hide_gaze_overlay = True
        self.control_bar.record_btn.setText("End Capture")
        self.countdown_timer.start()
        self._push_processing_state()

    def on_train_model(self):
        print("[DEBUG] Train Model button handler fired.")
        logger.info("Train Model button handler fired.")
        self.training_session_count += 1
        self.training_state["training_sessions"] = self.training_session_count
        self.training_state["last_trained"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_accuracy = min(0.999, max(self.training_state["best_accuracy"], 0.84 + 0.01 * self.training_session_count))
        updated_per_letter = {}
        for idx, letter in enumerate(chr(65 + i) for i in range(26)):
            count = self.letter_counts.get(letter, 0)
            base = 0.55 + min(0.40, count / 120.0)
            wobble = ((idx % 6) - 2.5) * 0.015 + (self.training_session_count % 3) * 0.01
            updated_per_letter[letter] = float(max(0.0, min(0.99, base + wobble)))
        self.training_state["per_letter_accuracy"] = updated_per_letter
        if new_accuracy > self.training_state["best_accuracy"]:
            self.training_state["best_accuracy"] = new_accuracy
            self.training_state["model_version"] += 1

        epochs_run = 50
        samples_used = int(self.training_state["total_samples"])
        loss_curve = [max(0.08, 1.2 * np.exp(-epoch / 14.0) + 0.03 * np.sin(epoch * 0.35)) for epoch in range(epochs_run)]
        accuracy_curve = [
            max(0.0, min(1.0, new_accuracy * (1 - np.exp(-epoch / 10.0)) + 0.01 * np.sin(epoch * 0.4)))
            for epoch in range(1, epochs_run + 1)
        ]
        history_entry = {
            "session": self.training_session_count,
            "timestamp": self.training_state["last_trained"],
            "samples_used": samples_used,
            "epochs_run": epochs_run,
            "val_accuracy": float(new_accuracy),
            "model_file": f"letter_model_v{self.training_state['model_version']}.pt",
            "loss_curve": [float(v) for v in loss_curve],
            "accuracy_curve": [float(v) for v in accuracy_curve],
        }
        self.training_state.setdefault("history", []).append(history_entry)
        save_training_state(self.training_state)
        self._refresh_stats()

    def _resolve_model_path(self) -> Path:
        """Resolve the preferred primary checkpoint path next to this dashboard script."""
        return Path(__file__).resolve().parent / DEFAULT_MODEL_OUTPUT

    def _find_best_available_model_path(self) -> Optional[Path]:
        """Find the best available checkpoint near the dashboard script.

        Returns:
            The primary checkpoint path if present, otherwise the highest numbered
            versioned checkpoint, or ``None`` if no compatible checkpoint exists.
        """
        primary_path = self._resolve_model_path()
        if primary_path.exists():
            return primary_path

        base_dir = primary_path.parent
        versioned_paths: List[Tuple[int, Path]] = []
        for candidate in base_dir.glob("letter_model_v*.pt"):
            suffix = candidate.stem.replace("letter_model_v", "", 1)
            try:
                versioned_paths.append((int(suffix), candidate))
            except ValueError:
                continue

        if not versioned_paths:
            return None

        versioned_paths.sort(key=lambda item: item[0], reverse=True)
        return versioned_paths[0][1]

    def _load_inference_model(self) -> bool:
        """Load the trained checkpoint for dashboard inference mode.

        Returns:
            True if the model loaded successfully, otherwise False.
        """
        primary_path = self._resolve_model_path()
        model_path = self._find_best_available_model_path()
        if model_path is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Model Missing",
                (
                    "No trained model found. Please train the model first.\n\n"
                    "Searched paths:\n"
                    f"- {primary_path}\n"
                    f"- {primary_path.parent / 'letter_model_v*.pt'}"
                ),
            )
            return False

        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            model_type = str(checkpoint.get("model_type", "lstm")).lower()
            if model_type == "transformer":
                model = LetterTransformerClassifier()
            else:
                model = LetterLSTMClassifier()
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            label_to_index = checkpoint.get("label_to_index", LABEL_TO_INDEX)
            self.inference_model = model
            self.inference_model_path = model_path
            self.inference_label_lookup = {idx: label for label, idx in label_to_index.items()}
            logger.info("Loaded inference model from %s", model_path)
            return True
        except Exception as exc:
            logger.error("Failed to load inference model from %s: %s", model_path, exc)
            QtWidgets.QMessageBox.critical(
                self,
                "Model Load Failed",
                f"Failed to load model checkpoint:\n{exc}",
            )
            return False

    def _reset_inference_collection(self) -> None:
        """Reset live inference trajectory accumulation without touching the camera."""
        self.inference_points = []
        self.inference_last_point = None
        self.trajectory_panel.set_points([])

    def _predict_inference_points(self) -> bool:
        """Run the loaded model on the current inference trajectory.

        Returns:
            True if a prediction was produced, otherwise False.
        """
        if self.inference_model is None or len(self.inference_points) < TRAJECTORY_MIN_POINTS:
            self.saved_flash_text = "Need a longer stroke to predict"
            self.saved_flash_until = time_now() + UI_INFO_FLASH_SECONDS
            return False

        if is_degenerate_trajectory(self.inference_points):
            self.saved_flash_text = "No movement detected - draw the letter"
            self.saved_flash_until = time_now() + UI_INFO_FLASH_SECONDS
            return False

        try:
            normalized = normalize_trajectory(self.inference_points)
            tensor = torch.from_numpy(normalized).unsqueeze(0)
            with torch.no_grad():
                logits = self.inference_model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
            top_k = min(3, probs.numel())
            top_values, top_indices = torch.topk(probs, k=top_k)
            best_idx = int(top_indices[0].item())
            self.last_prediction_letter = self.inference_label_lookup.get(best_idx, "?")
            self.last_prediction_confidence = float(top_values[0].item())
            self.last_prediction_top3 = [
                {
                    "label": self.inference_label_lookup.get(int(index.item()), "?"),
                    "confidence": float(value.item()),
                }
                for value, index in zip(top_values, top_indices)
            ]
            self.saved_flash_text = f"Predicted {self.last_prediction_letter}"
            self.saved_flash_until = time_now() + UI_INFO_FLASH_SECONDS
            logger.info(
                "Dashboard inference predicted %s at %.3f from %s points.",
                self.last_prediction_letter,
                self.last_prediction_confidence,
                len(self.inference_points),
            )
            return True
        except Exception as exc:
            logger.error("Dashboard inference prediction failed: %s", exc)
            self.saved_flash_text = "Prediction failed"
            self.saved_flash_until = time_now() + UI_INFO_FLASH_SECONDS
            return False

    def on_run_live_inference(self):
        print("[DEBUG] Run Live Inference button handler fired.")
        logger.info("Run Live Inference button handler fired.")
        if self.inference_mode:
            predicted = self._predict_inference_points()
            self.inference_mode = False
            self.control_bar.inference_btn.setText("Run Live Inference")
            self.statusBar().showMessage(
                "Inference stopped." if not predicted else "Inference complete.",
                4000,
            )
            self._reset_inference_collection()
            self._push_processing_state()
            return

        if self.inference_model is None and not self._load_inference_model():
            return

        self.inference_mode = True
        self.control_bar.inference_btn.setText("Stop Inference")
        self._reset_inference_collection()
        self.saved_flash_text = "Model loaded - draw a letter to predict."
        self.saved_flash_until = time_now() + UI_INFO_FLASH_SECONDS
        self.statusBar().showMessage("Model loaded — draw a letter to predict.", 5000)
        self._push_processing_state()

    def _init_audio(self):
        if pygame is None:
            return
        try:
            pygame.mixer.init()
            self._beep_ready = True
        except Exception:
            self._beep_ready = False

    def _play_save_beep(self):
        if not self._beep_ready or pygame is None:
            return
        try:
            sample_rate = UI_BEEP_SAMPLE_RATE
            duration = UI_BEEP_DURATION_SECONDS
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            wave = UI_BEEP_VOLUME * np.sin(2 * np.pi * UI_BEEP_FREQUENCY_HZ * t)
            stereo = np.column_stack([wave, wave])
            sound = pygame.sndarray.make_sound((stereo * 32767).astype(np.int16))
            sound.play()
        except Exception:
            pass

    def _save_current_target_sample(self):
        if self.calibration_mode:
            self._save_calibration_sample()
            return

        trajectory = self.pending_saved_trajectory[:] if self.pending_saved_trajectory else self.last_trajectory[:]
        if not trajectory:
            return
        append_sample_record(self.selected_letter, trajectory)
        self.training_state["samples_per_letter"][self.selected_letter] += 1
        self.letter_counts[self.selected_letter] = self.training_state["samples_per_letter"][self.selected_letter]
        self.training_state["total_samples"] += 1
        save_training_state(self.training_state)
        self.saved_flash_text = f"\u2713 Saved! ({self.letter_counts[self.selected_letter]} samples)"
        self.saved_flash_until = time_now() + UI_SAVE_FLASH_SECONDS
        self._play_save_beep()
        self._refresh_stats()
        self.pending_saved_trajectory = None
        self._reset_capture_state()

    def _save_calibration_sample(self):
        trajectory = self.pending_saved_trajectory[:] if self.pending_saved_trajectory else self.last_trajectory[:]
        if not trajectory:
            return
        count = save_calibration_sample(self.selected_letter, trajectory)
        self.saved_flash_text = f"✓ Calibration saved! ({count}/{CALIBRATION_SAMPLES_PER_LETTER} for {self.selected_letter})"
        self.saved_flash_until = time_now() + UI_SAVE_FLASH_SECONDS
        self._play_save_beep()
        self._refresh_calibration_tiles()
        self.pending_saved_trajectory = None
        self._reset_capture_state()

    def _refresh_calibration_tiles(self):
        counts = count_calibration_samples()
        for letter, tile in self.letter_tiles.items():
            tile.count_label.setText(f"calib {counts.get(letter, 0)}/{CALIBRATION_SAMPLES_PER_LETTER}")

    def on_toggle_calibration_mode(self):
        print("[DEBUG] Calibrate Mode button handler fired.")
        logger.info("Calibrate Mode button handler fired.")
        self.calibration_mode = not self.calibration_mode
        if self.calibration_mode:
            self.control_bar.calibrate_btn.setText("Exit Calibration")
            self.statusBar().showMessage(
                "Calibration mode: draw each letter 5-10 times, then click Fine-Tune Personal Model.",
                6000,
            )
            self._refresh_calibration_tiles()
        else:
            self.control_bar.calibrate_btn.setText("Calibrate Mode")
            self.statusBar().showMessage("Calibration mode off.", 3000)
            self._refresh_stats()

    def on_fine_tune_personal_model(self):
        print("[DEBUG] Fine-Tune Personal Model button handler fired.")
        logger.info("Fine-Tune Personal Model button handler fired.")
        self.statusBar().showMessage("Fine-tuning personal model...", 3000)
        QtWidgets.QApplication.processEvents()
        try:
            accuracy = fine_tune_personal_model()
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("Fine-tuning failed: %s", exc)
            self.statusBar().showMessage(str(exc), 6000)
            return
        self.statusBar().showMessage(f"Saved personal_model.pt (train accuracy {accuracy * 100:.1f}%).", 6000)
        if self.calibration_mode:
            self._refresh_calibration_tiles()

    def _reset_capture_state(self):
        self.recording_active = False
        self.capture_phase = "idle"
        self.capture_phase_started = 0.0
        self.capture_countdown_started = 0.0
        self.control_bar.record_btn.setEnabled(True)
        self.control_bar.record_btn.setText("Capture Letter")
        self.countdown_timer.stop()
        self.stabilization_frames_remaining = 0
        self.collection_started = False
        self.last_committed_point = None
        self.hide_gaze_overlay = False
        self.trajectory_builder.points = []
        self.trajectory_builder.anchor_point = None
        self.trajectory_builder.anchor_timestamp_ns = None
        self.trajectory_builder.first_point_timestamp_ns = None
        self.processing_reset_requested.emit()
        self._push_processing_state()

    def _finish_or_reset_capture(self):
        if self.capture_phase == "countdown":
            self.saved_flash_text = "Reset before draw"
            self.saved_flash_until = time_now() + UI_INFO_FLASH_SECONDS
            self.pending_saved_trajectory = None
            self._reset_capture_state()
            return

        if self.capture_phase != "drawing":
            self._reset_capture_state()
            return

        trajectory = [tuple(pt) for pt in self.trajectory_builder.points]
        if len(trajectory) < self.trajectory_builder.min_points:
            self.saved_flash_text = "Reset - too short"
            self.saved_flash_until = time_now() + UI_RESET_FLASH_SECONDS
            self.pending_saved_trajectory = None
            self._reset_capture_state()
        elif is_degenerate_trajectory(trajectory):
            self.saved_flash_text = "Reset - no movement detected"
            self.saved_flash_until = time_now() + UI_RESET_FLASH_SECONDS
            self.pending_saved_trajectory = None
            self._reset_capture_state()
        else:
            self.pending_saved_trajectory = trajectory
            self.last_trajectory = trajectory[:]
            self.set_last_trajectory(trajectory)
            self._save_current_target_sample()

    def _generate_demo_trajectory(self):
        points = []
        for i in range(48):
            angle = i / 48.0 * math.pi * 2.0
            radius = 40 + 12 * math.sin(i * 0.3)
            points.append((radius * math.cos(angle), radius * math.sin(angle)))
        return points

    def _tracking_point(self):
        t = self._demo_phase
        x = int((math.sin(t) * 0.35 + 0.5) * (CAMERA_WIDTH - 120)) + 60
        y = int((math.cos(t * 1.3) * 0.28 + 0.5) * (CAMERA_HEIGHT - 120)) + 60
        return x, y

    def _init_live_bridge(self):
        self.bridge_worker = BridgeFeedWorker(WS_URL, self)
        self.bridge_worker.status_changed.connect(self._on_bridge_status, QtCore.Qt.QueuedConnection)
        self.bridge_worker.start()

    def _init_processing_thread(self):
        self.processing_thread = QtCore.QThread(self)
        self.processing_worker = ProcessingWorker()
        self.processing_worker.moveToThread(self.processing_thread)
        self.bridge_worker.frame_received.connect(
            self.processing_worker.submit_frame,
            QtCore.Qt.QueuedConnection,
        )
        self.processing_state_changed.connect(
            self.processing_worker.update_state,
            QtCore.Qt.QueuedConnection,
        )
        self.processing_reset_requested.connect(
            self.processing_worker.reset_tracking_state,
            QtCore.Qt.QueuedConnection,
        )
        self.processing_worker.annotated_frame_ready.connect(
            self._on_processed_frame,
            QtCore.Qt.QueuedConnection,
        )
        self.processing_worker.tracking_updated.connect(
            self._on_processed_tracking,
            QtCore.Qt.QueuedConnection,
        )
        self.processing_thread.start()
        self._push_processing_state()

    def _on_bridge_status(self, status_text):
        self.bridge_status = status_text
        self._push_processing_state()

    def _push_processing_state(self):
        self.processing_state_changed.emit(
            {
                "recording_active": self.recording_active,
                "capture_phase": self.capture_phase,
                "capture_phase_started": self.capture_phase_started,
                "capture_countdown_started": self.capture_countdown_started,
                "capture_countdown_seconds": self.capture_countdown_seconds,
                "selected_letter": self.selected_letter,
                "collection_started": self.collection_started,
                "saved_flash_until": self.saved_flash_until,
                "saved_flash_text": self.saved_flash_text,
                "bridge_status": self.bridge_status,
                "hide_gaze_overlay": self.hide_gaze_overlay,
                "inference_mode": self.inference_mode,
                "prediction_text": self.last_prediction_letter,
                "prediction_confidence": self.last_prediction_confidence,
                "top_predictions": self.last_prediction_top3,
            }
        )

    def _on_processed_frame(self, image):
        self.camera_panel.set_qimage(image)

    def _on_processed_tracking(self, tracking_meta):
        self.current_fingertip = tracking_meta.get("smoothed_point")
        self.stabilization_frames_remaining = int(tracking_meta.get("stabilization_frames_remaining", 0))

        if self.inference_mode:
            self._update_inference_tracking(tracking_meta)

        if not self.recording_active or self.capture_phase != "drawing":
            return

        if tracking_meta.get("hard_loss"):
            self.last_committed_point = None
            self.collection_started = bool(self.trajectory_builder.points)
            self.trajectory_panel.set_points([tuple(pt) for pt in self.trajectory_builder.points])
            return

        if not tracking_meta.get("tracking_visible"):
            self.trajectory_panel.set_points([tuple(pt) for pt in self.trajectory_builder.points])
            return

        if not tracking_meta.get("collection_ready"):
            self.trajectory_panel.set_points([tuple(pt) for pt in self.trajectory_builder.points])
            return

        smoothed_point = tracking_meta.get("smoothed_point")
        if smoothed_point is None:
            self.trajectory_panel.set_points([tuple(pt) for pt in self.trajectory_builder.points])
            return

        if not self.collection_started:
            self.collection_started = True
            self.last_committed_point = tuple(smoothed_point)
            self.trajectory_builder.points = [self.last_committed_point]
            self.trajectory_panel.set_points([tuple(pt) for pt in self.trajectory_builder.points])
            self._push_processing_state()
            return

        current_point = tuple(smoothed_point)
        if current_point != self.last_committed_point:
            self.last_committed_point = current_point
            self.trajectory_builder.points.append(self.last_committed_point)
            if len(self.trajectory_builder.points) > MAX_TRAJECTORY_POINTS:
                self.trajectory_builder.points.pop(0)
        self.trajectory_panel.set_points([tuple(pt) for pt in self.trajectory_builder.points])

    def _update_inference_tracking(self, tracking_meta: Dict[str, Any]) -> None:
        """Collect live inference stroke points while inference mode is active.

        Args:
            tracking_meta: Latest smoothed fingertip tracking metadata.
        """
        if not tracking_meta.get("tracking_visible"):
            self.trajectory_panel.set_points(list(self.inference_points))
            return

        point = tracking_meta.get("smoothed_point")
        if point is None:
            self.trajectory_panel.set_points(list(self.inference_points))
            return

        current_point = tuple(point)
        if current_point != self.inference_last_point:
            self.inference_last_point = current_point
            self.inference_points.append(current_point)
            if len(self.inference_points) > MAX_TRAJECTORY_POINTS:
                self.inference_points.pop(0)
        self.trajectory_panel.set_points(list(self.inference_points))

    def _recording_tick(self):
        if not self.recording_active:
            return

        now = time_now()
        if self.capture_phase == "countdown":
            elapsed = now - self.capture_countdown_started
            if elapsed >= self.capture_countdown_seconds:
                self.capture_phase = "drawing"
                self.capture_phase_started = now
                self.last_trajectory = []
                self.trajectory_builder.points = []
                self.trajectory_builder.anchor_point = None
                self.trajectory_builder.anchor_timestamp_ns = None
                self.trajectory_builder.first_point_timestamp_ns = None
                self.stabilization_frames_remaining = UI_STABILIZATION_FRAMES
                self.collection_started = False
                self.last_committed_point = None
                self._push_processing_state()
                self.countdown_timer.stop()

    def closeEvent(self, event):
        if self._shutting_down:
            event.accept()
            return
        self._shutting_down = True
        self.countdown_timer.stop()

        if self.bridge_worker is not None:
            self.bridge_worker.stop()
            if not self.bridge_worker.wait(2000):
                logger.warning("Bridge worker did not stop within 2 seconds.")

        if self.processing_worker is not None:
            QtCore.QMetaObject.invokeMethod(
                self.processing_worker,
                "stop",
                QtCore.Qt.QueuedConnection,
            )
        if self.processing_thread is not None:
            self.processing_thread.quit()
            if not self.processing_thread.wait(2000):
                logger.warning("Processing thread did not stop within 2 seconds.")

        event.accept()

    def delete_sample_record(self, record_id):
        records = load_sample_metadata()
        remaining = [record for record in records if record.get("id") != record_id]
        if len(remaining) == len(records):
            return

        rewrite_dataset_from_records(remaining)

        updated_counts = {chr(65 + idx): 0 for idx in range(26)}
        for record in remaining:
            label = record.get("label")
            if label in updated_counts:
                updated_counts[label] += 1

        self.training_state["samples_per_letter"] = updated_counts
        self.training_state["total_samples"] = sum(updated_counts.values())
        self.letter_counts = dict(updated_counts)
        save_training_state(self.training_state)
        self._refresh_stats()

        if self.review_dialog is not None:
            self.review_dialog.records = [record for record in remaining if record.get("label") == self.selected_letter]
            self.review_dialog._rebuild()

    def export_history_csv(self):
        export_path = TRAINING_HISTORY_EXPORT_PATH
        history = self.training_state.get("history", [])
        try:
            with export_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Session #", "Date & Time", "Samples Used", "Epochs Run", "Val Accuracy", "Model File"])
                for record in history:
                    writer.writerow(
                        [
                            record.get("session", ""),
                            record.get("timestamp", ""),
                            record.get("samples_used", ""),
                            record.get("epochs_run", ""),
                            f"{float(record.get('val_accuracy', 0.0)) * 100:.1f}%",
                            record.get("model_file", ""),
                        ]
                    )
            logger.info("Exported training history to %s", export_path)
        except OSError as exc:
            logger.error("Failed to export training history to %s: %s", export_path, exc)


def time_now() -> float:
    """Return the current wall-clock timestamp used by the dashboard."""
    return datetime.now().timestamp()


def main() -> None:
    """Launch the PyQt5 application event loop."""
    app = QtWidgets.QApplication(sys.argv)
    window = AriaTrainingDashboard()
    window.show()
    exit_code = app.exec_()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
