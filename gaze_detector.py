"""Run gaze projection, object detection, and gesture detection for Aria RGB.

This module combines ET-based gaze estimation, YOLO object detection, and
MediaPipe hand gesture recognition. It depends on OpenCV, NumPy, Ultralytics,
and MediaPipe Tasks, and it produces structured detection dictionaries that the
bridge and dashboard can forward to clients.
"""

import threading
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
from config import (
    DEFAULT_YOLO_CONF_THRESHOLD,
    DEFAULT_YOLO_DEVICE,
    DEFAULT_YOLO_MODEL_SIZE,
    EYE_DARK_THRESHOLD,
    EYE_MIN_MOMENT_AREA,
    EYE_MORPH_KERNEL_SIZE,
    EYE_REFLECTION_REPLACEMENT_VALUE,
    EYE_REFLECTION_THRESHOLD,
    EYE_RESIZE_HEIGHT,
    EYE_RESIZE_WIDTH,
    GAZE_CALIBRATION_SAMPLE_COUNT,
    GAZE_CALIBRATION_SLEEP_SECONDS,
    GAZE_DEFAULT_SCALE,
    HAND_LANDMARKER_MODEL_PATH,
    HAND_LANDMARKER_MODEL_URL,
    MEDIAPIPE_HAND_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_HAND_MIN_PRESENCE_CONFIDENCE,
    MEDIAPIPE_HAND_MIN_TRACKING_CONFIDENCE,
    MEDIAPIPE_MAX_NUM_HANDS,
    YOLO_PERF_LOG_INTERVAL,
)
from logging_utils import get_logger


logger = get_logger(__name__)


class HandGestureDetector:
    """Detect hand gestures in RGB frames using MediaPipe Tasks."""

    MODEL_URL = HAND_LANDMARKER_MODEL_URL

    def __init__(self) -> None:
        """Initialize the hand landmarker once at startup."""
        self.available = False
        self.error = None
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision

            model_path = HAND_LANDMARKER_MODEL_PATH
            if not model_path.exists():
                logger.info("Downloading MediaPipe hand landmarker model...")
                try:
                    urlretrieve(self.MODEL_URL, model_path)
                except Exception as exc:
                    raise RuntimeError(f"Failed to download hand landmarker model: {exc}") from exc

            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=MEDIAPIPE_MAX_NUM_HANDS,
                min_hand_detection_confidence=MEDIAPIPE_HAND_MIN_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=MEDIAPIPE_HAND_MIN_PRESENCE_CONFIDENCE,
                min_tracking_confidence=MEDIAPIPE_HAND_MIN_TRACKING_CONFIDENCE,
            )
            self.mp = mp
            self.landmarker = vision.HandLandmarker.create_from_options(options)
            self.available = True
        except Exception as exc:
            self.mp = None
            self.landmarker = None
            self.error = str(exc)

    def detect(self, rgb_bgr: np.ndarray) -> List[Dict]:
        """Detect hands and classify coarse gestures in one RGB frame.

        Args:
            rgb_bgr: Input frame in BGR color order.

        Returns:
            A list of gesture dictionaries for detected hands.
        """
        if not self.available:
            return []
        try:
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            results = self.landmarker.detect(mp_image)
        except Exception as exc:
            logger.warning("MediaPipe hand detection failed; continuing without hand result: %s", exc)
            return []
        if not results.hand_landmarks:
            return []

        h, w = rgb_bgr.shape[:2]
        detections = []
        handedness = results.handedness or []
        for index, landmarks in enumerate(results.hand_landmarks):
            label = "hand"
            score = 0.0
            if index < len(handedness):
                category = handedness[index][0]
                label = category.category_name.lower()
                score = float(category.score)

            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            box = [
                max(0, int(min(xs) * w)),
                max(0, int(min(ys) * h)),
                min(w - 1, int(max(xs) * w)),
                min(h - 1, int(max(ys) * h)),
            ]
            gesture = self._classify(landmarks)
            detections.append(
                {
                    "gesture": gesture,
                    "handedness": label,
                    "conf": round(score, 3),
                    "box": box,
                }
            )
        return detections

    def _classify(self, lm: object) -> str:
        """Map hand landmarks to a simple symbolic gesture label."""
        wrist = lm[0]
        thumb_tip = lm[4]
        thumb_ip = lm[3]
        index_tip = lm[8]
        index_pip = lm[6]
        middle_tip = lm[12]
        middle_pip = lm[10]
        ring_tip = lm[16]
        ring_pip = lm[14]
        pinky_tip = lm[20]
        pinky_pip = lm[18]

        index_up = index_tip.y < index_pip.y - 0.025
        middle_up = middle_tip.y < middle_pip.y - 0.025
        ring_up = ring_tip.y < ring_pip.y - 0.025
        pinky_up = pinky_tip.y < pinky_pip.y - 0.025
        fingers_up = [index_up, middle_up, ring_up, pinky_up]
        up_count = sum(fingers_up)

        thumb_vertical = abs(thumb_tip.y - wrist.y) > abs(thumb_tip.x - wrist.x) * 1.15
        thumb_up = thumb_vertical and thumb_tip.y < wrist.y - 0.08
        thumb_down = thumb_vertical and thumb_tip.y > wrist.y + 0.08
        thumb_extended = abs(thumb_tip.x - thumb_ip.x) > 0.035 or abs(thumb_tip.y - thumb_ip.y) > 0.035

        if thumb_up and up_count <= 1:
            return "thumbs_up"
        if thumb_down and up_count <= 1:
            return "thumbs_down"
        if up_count >= 4 and thumb_extended:
            return "flat_hand"
        if up_count == 0 and not thumb_extended:
            return "fist"
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "pointing"
        if index_up and middle_up and not ring_up and not pinky_up:
            return "peace"
        if up_count >= 4:
            return "open_hand"
        return "hand"


class GazeDetector:
    """Project eye-tracking gaze onto RGB frames and run scene detection."""

    def __init__(
        self,
        model_size: str = DEFAULT_YOLO_MODEL_SIZE,
        device: str = DEFAULT_YOLO_DEVICE,
        conf_threshold: float = DEFAULT_YOLO_CONF_THRESHOLD,
    ) -> None:
        """Initialize the detector and load the YOLO model.

        Args:
            model_size: YOLO model checkpoint name.
            device: Preferred inference device.
            conf_threshold: Detection confidence threshold.
        """
        try:
            import torch

            if device == "mps" and torch.backends.mps.is_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        except Exception:
            resolved_device = "cpu"

        from ultralytics import YOLO

        self.model = YOLO(model_size)
        try:
            self.model.to(resolved_device)
        except Exception as exc:
            logger.warning("Falling back to CPU for YOLO model %s: %s", model_size, exc)
            resolved_device = "cpu"
            self.model.to(resolved_device)

        self.device = resolved_device
        self.model_size = model_size
        self.conf_threshold = conf_threshold
        self.hand_detector = HandGestureDetector()
        self.latest_et_left: Optional[np.ndarray] = None
        self.latest_et_right: Optional[np.ndarray] = None
        self.last_result: Optional[Dict] = None
        self.frame_counter = 0
        self.inference_counter = 0
        self.inference_total_seconds = 0.0
        self.last_gazed_label: Optional[str] = None
        self.gaze_start_time = time.time()
        self._lock = threading.Lock()
        self._process_lock = threading.Lock()

        self.calibration_points: List[Tuple[float, float, float, float]] = []
        self.calibrated = False
        self.calibrating = False
        self.calib_x_coeffs: Optional[np.ndarray] = None
        self.calib_y_coeffs: Optional[np.ndarray] = None

    def update_et(self, left_eye: Optional[np.ndarray], right_eye: Optional[np.ndarray]) -> None:
        """Store the latest ET frames for gaze projection."""
        with self._lock:
            self.latest_et_left = left_eye.copy() if left_eye is not None else None
            self.latest_et_right = right_eye.copy() if right_eye is not None else None

    def _pupil_center(self, eye_img: Optional[np.ndarray]) -> Tuple[Optional[float], Optional[float]]:
        """Estimate the normalized pupil center in one ET eye image."""
        if eye_img is None or eye_img.size == 0:
            return 0.5, 0.5

        small = cv2.resize(eye_img, (EYE_RESIZE_WIDTH, EYE_RESIZE_HEIGHT), interpolation=cv2.INTER_AREA)
        small = np.where(small > EYE_REFLECTION_THRESHOLD, EYE_REFLECTION_REPLACEMENT_VALUE, small).astype(np.uint8)
        _, thresh = cv2.threshold(small, EYE_DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((EYE_MORPH_KERNEL_SIZE, EYE_MORPH_KERNEL_SIZE), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        moments = cv2.moments(thresh)
        if moments["m00"] < EYE_MIN_MOMENT_AREA:
            return None, None

        cx = moments["m10"] / moments["m00"] / float(EYE_RESIZE_WIDTH)
        cy = moments["m01"] / moments["m00"] / float(EYE_RESIZE_HEIGHT)
        return cx, cy

    def _gaze_to_rgb(
        self,
        rgb_w: int,
        rgb_h: int,
        et_left: Optional[np.ndarray],
        et_right: Optional[np.ndarray],
    ) -> Tuple[int, int]:
        """Project ET pupil coordinates into RGB pixel space."""
        if et_left is None and et_right is None:
            return rgb_w // 2, rgb_h // 2

        lx, ly = self._pupil_center(et_left)
        rx, ry = self._pupil_center(et_right)

        valid = [(x, y) for x, y in [(lx, ly), (rx, ry)] if x is not None]
        if not valid:
            return rgb_w // 2, rgb_h // 2

        pupil_x = sum(v[0] for v in valid) / len(valid)
        pupil_y = sum(v[1] for v in valid) / len(valid)

        if self.calibrated and self.calib_x_coeffs is not None and self.calib_y_coeffs is not None:
            features = np.array([pupil_x, pupil_y, 1.0], dtype=np.float32)
            gaze_x_norm = float(np.dot(features, self.calib_x_coeffs))
            gaze_y_norm = float(np.dot(features, self.calib_y_coeffs))
        else:
            scale = GAZE_DEFAULT_SCALE
            gaze_x_norm = 0.5 + (pupil_x - 0.5) * scale
            gaze_y_norm = 0.5 + (pupil_y - 0.5) * scale

        gaze_x = int(gaze_x_norm * rgb_w)
        gaze_y = int(gaze_y_norm * rgb_h)
        gaze_x = max(0, min(rgb_w - 1, gaze_x))
        gaze_y = max(0, min(rgb_h - 1, gaze_y))
        return gaze_x, gaze_y

    def _run_detection(self, rgb_bgr: np.ndarray) -> List[Dict]:
        """Run YOLO object detection on one BGR frame."""
        try:
            results = self.model(rgb_bgr, verbose=False, conf=self.conf_threshold)[0]
        except Exception as exc:
            logger.warning("YOLO detection failed on frame; continuing without detections: %s", exc)
            return []
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = self.model.names[int(box.cls[0])]
            detections.append(
                {
                    "label": label,
                    "conf": round(conf, 3),
                    "box": [x1, y1, x2, y2],
                }
            )
        return detections

    def _run_hand_detection(self, rgb_bgr: np.ndarray) -> List[Dict]:
        """Run hand gesture detection on one BGR frame."""
        return self.hand_detector.detect(rgb_bgr)

    def process(self, rgb_bgr: np.ndarray) -> Dict:
        """Process an RGB frame and return gaze, object, and gesture results."""
        try:
            with self._process_lock:
                with self._lock:
                    self.frame_counter += 1
                    frame_counter = self.frame_counter
                    et_left = self.latest_et_left
                    et_right = self.latest_et_right
                    cached_result = self.last_result

                if frame_counter % 2 == 1 and cached_result is not None:
                    return cached_result

                inference_start = time.perf_counter()
                h, w = rgb_bgr.shape[:2]
                gaze_x, gaze_y = self._gaze_to_rgb(w, h, et_left, et_right)
                detections = self._run_detection(rgb_bgr)
                hand_detections = self._run_hand_detection(rgb_bgr)

                gazed_object = None
                best_area = float("inf")
                for det in detections:
                    x1, y1, x2, y2 = det["box"]
                    if x1 <= gaze_x <= x2 and y1 <= gaze_y <= y2:
                        area = (x2 - x1) * (y2 - y1)
                        if area < best_area:
                            best_area = area
                            gazed_object = det

                current_label = gazed_object["label"] if gazed_object else None
                now = time.time()
                with self._lock:
                    self.inference_counter += 1
                    self.inference_total_seconds += time.perf_counter() - inference_start
                    if self.inference_counter % YOLO_PERF_LOG_INTERVAL == 0:
                        avg_ms = (self.inference_total_seconds / self.inference_counter) * 1000.0
                        logger.info(
                            "Gaze detector avg inference time over %s processed frames: %.2fms",
                            self.inference_counter,
                            avg_ms,
                        )
                    if current_label != self.last_gazed_label:
                        self.gaze_start_time = now
                        self.last_gazed_label = current_label
                    dwell_seconds = round(now - self.gaze_start_time, 2) if current_label else 0.0
                    result = {
                        "gaze_x": gaze_x,
                        "gaze_y": gaze_y,
                        "gazed_object": gazed_object,
                        "dwell_seconds": dwell_seconds,
                        "all_detections": detections,
                        "hand_gestures": hand_detections,
                        "hand_detector_available": self.hand_detector.available,
                        "frame_w": w,
                        "frame_h": h,
                    }
                    self.last_result = result
                    return result
        except Exception as exc:
            logger.error("Gaze detector failed unexpectedly; returning cached result: %s", exc)
            with self._lock:
                return self.last_result or {
                    "gaze_x": 0,
                    "gaze_y": 0,
                    "gazed_object": None,
                    "dwell_seconds": 0.0,
                    "all_detections": [],
                    "hand_gestures": [],
                    "hand_detector_available": self.hand_detector.available,
                    "frame_w": int(rgb_bgr.shape[1]) if rgb_bgr is not None else 0,
                    "frame_h": int(rgb_bgr.shape[0]) if rgb_bgr is not None else 0,
                }

    def start_calibration(self) -> None:
        """Begin a new gaze calibration session."""
        with self._lock:
            self.calibration_points = []
            self.calibrating = True

    def add_calibration_point(self, known_x_norm: float, known_y_norm: float) -> None:
        """Sample ET points for one known calibration target."""
        samples = []
        for _ in range(GAZE_CALIBRATION_SAMPLE_COUNT):
            with self._lock:
                et_left = self.latest_et_left
                et_right = self.latest_et_right
            lx, ly = self._pupil_center(et_left)
            rx, ry = self._pupil_center(et_right)
            valid = [(x, y) for x, y in [(lx, ly), (rx, ry)] if x is not None]
            if valid:
                avg_x = sum(v[0] for v in valid) / len(valid)
                avg_y = sum(v[1] for v in valid) / len(valid)
                samples.append((avg_x, avg_y))
            time.sleep(GAZE_CALIBRATION_SLEEP_SECONDS)

        if samples:
            pupil_x = sum(s[0] for s in samples) / len(samples)
            pupil_y = sum(s[1] for s in samples) / len(samples)
            with self._lock:
                self.calibration_points.append((known_x_norm, known_y_norm, pupil_x, pupil_y))

    def finish_calibration(self) -> bool:
        """Fit a linear gaze calibration model from sampled points."""
        with self._lock:
            if len(self.calibration_points) < 3:
                self.calibrating = False
                return False
            pts = np.array(self.calibration_points, dtype=np.float32)

        pupil_x = pts[:, 2]
        pupil_y = pts[:, 3]
        screen_x = pts[:, 0]
        screen_y = pts[:, 1]

        matrix = np.column_stack([pupil_x, pupil_y, np.ones(len(pts))])
        calib_x_coeffs, _, _, _ = np.linalg.lstsq(matrix, screen_x, rcond=None)
        calib_y_coeffs, _, _, _ = np.linalg.lstsq(matrix, screen_y, rcond=None)

        with self._lock:
            self.calib_x_coeffs = calib_x_coeffs
            self.calib_y_coeffs = calib_y_coeffs
            self.calibrated = True
            self.calibrating = False
        return True


if __name__ == "__main__":
    logger.info("gaze_detector provides reusable gaze and gesture detection classes.")
