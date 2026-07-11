"""Appliance detection for the RoomScan energy audit.

Wraps an Ultralytics YOLO model (COCO-pretrained, ``ENERGY_YOLO_WEIGHTS``)
and filters detections to the appliance classes in ``config.ENERGY_CATALOG``.
Aggregation across a scan uses the max-simultaneous rule: the instance count
for a class is the largest number of detections of that class seen in any
single sampled frame, which is robust to the camera panning away and back
(no track-based double counting). For each counted instance slot the
highest-confidence crop seen anywhere in the scan is kept for the report.

Ultralytics is imported lazily so pure-logic consumers (tests, estimator)
never need torch. Frames passed to :class:`EnergyDetector` must be UPRIGHT
display-oriented RGB (callers rotate aria_capture's RAW frames with
``rotate_upright()`` first -- YOLO was trained on upright imagery).

Standalone test CLI:
    python energy_detector.py --vrs /path/to/recording.vrs
"""

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    ENERGY_CATALOG,
    ENERGY_DETECT_CONFIDENCE,
    ENERGY_FRAME_SAMPLE_HZ,
    ENERGY_MIN_BOX_AREA_FRAC,
    ENERGY_YOLO_WEIGHTS,
)
from logging_utils import get_logger

logger = get_logger(__name__)

CROP_PADDING_FRAC = 0.06  # context margin around each saved crop


@dataclass(frozen=True)
class Detection:
    """One appliance detection in one upright RGB frame."""

    class_name: str
    confidence: float
    box_xyxy: Tuple[int, int, int, int]


@dataclass
class _CropSlot:
    confidence: float
    crop_rgb: np.ndarray  # upright RGB uint8


class ApplianceScanAggregator:
    """Aggregate per-frame detections into per-class counts and best crops.

    Pure logic (no YOLO/torch) so it can be unit-tested directly. Call
    :meth:`observe_frame` once per sampled frame, then read :meth:`counts`
    and :meth:`best_crops`.
    """

    def __init__(self) -> None:
        self.frames_observed = 0
        self._slots: Dict[str, List[_CropSlot]] = {}

    def observe_frame(self, detections: List[Detection], frame_rgb: Optional[np.ndarray]) -> None:
        """Fold one frame's detections in. ``frame_rgb`` may be None in tests
        (crops are then skipped, counts still update)."""
        self.frames_observed += 1
        by_class: Dict[str, List[Detection]] = {}
        for det in detections:
            by_class.setdefault(det.class_name, []).append(det)

        for class_name, dets in by_class.items():
            dets.sort(key=lambda d: d.confidence, reverse=True)
            slots = self._slots.setdefault(class_name, [])
            # Grow to the new simultaneous max; never shrink.
            while len(slots) < len(dets):
                slots.append(_CropSlot(confidence=-1.0, crop_rgb=None))  # type: ignore[arg-type]
            for i, det in enumerate(dets):
                if det.confidence > slots[i].confidence:
                    slots[i].confidence = det.confidence
                    if frame_rgb is not None:
                        slots[i].crop_rgb = _extract_crop(frame_rgb, det.box_xyxy)

    def counts(self) -> Dict[str, int]:
        """{class_name: max simultaneous detections seen in one frame}."""
        return {name: len(slots) for name, slots in self._slots.items() if slots}

    def best_crops(self) -> Dict[str, List[np.ndarray]]:
        """Per class, one best-confidence RGB crop per counted instance slot."""
        return {
            name: [s.crop_rgb for s in slots if s.crop_rgb is not None]
            for name, slots in self._slots.items()
        }

    def best_confidences(self) -> Dict[str, List[float]]:
        return {name: [round(s.confidence, 3) for s in slots] for name, slots in self._slots.items()}


def _extract_crop(frame_rgb: np.ndarray, box_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    pad_x = int((x2 - x1) * CROP_PADDING_FRAC)
    pad_y = int((y2 - y1) * CROP_PADDING_FRAC)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return np.ascontiguousarray(frame_rgb[y1:y2, x1:x2])


class EnergyDetector:
    """YOLO appliance detector over upright RGB frames."""

    def __init__(
        self,
        weights: str = ENERGY_YOLO_WEIGHTS,
        confidence: float = ENERGY_DETECT_CONFIDENCE,
    ) -> None:
        try:
            from ultralytics import YOLO  # lazy: torch-free imports stay torch-free
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for detection: pip install ultralytics"
            ) from exc
        try:
            self._model = YOLO(weights)
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO weights '{weights}'") from exc
        self._confidence = confidence
        self._names = self._model.names  # {idx: class_name}
        catalog_hits = [n for n in self._names.values() if n in ENERGY_CATALOG]
        logger.info(
            "YOLO '%s' loaded; %d/%d catalog classes available: %s",
            weights, len(catalog_hits), len(ENERGY_CATALOG), sorted(catalog_hits),
        )

    def detect(self, frame_rgb_upright: np.ndarray) -> List[Detection]:
        """Run one frame; return catalog-class detections above thresholds."""
        # Ultralytics treats raw numpy input as BGR (cv2 convention).
        bgr = np.ascontiguousarray(frame_rgb_upright[:, :, ::-1])
        results = self._model.predict(bgr, conf=self._confidence, verbose=False)
        h, w = frame_rgb_upright.shape[:2]
        min_area = ENERGY_MIN_BOX_AREA_FRAC * h * w
        detections: List[Detection] = []
        for box in results[0].boxes:
            class_name = self._names[int(box.cls[0])]
            if class_name not in ENERGY_CATALOG:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            if (x2 - x1) * (y2 - y1) < min_area:
                continue
            detections.append(
                Detection(class_name=class_name, confidence=float(box.conf[0]), box_xyxy=(x1, y1, x2, y2))
            )
        return detections


def scan_capture_rgb(
    capture,
    detector: EnergyDetector,
    aggregator: ApplianceScanAggregator,
    duration_s: Optional[float] = None,
    sample_hz: float = ENERGY_FRAME_SAMPLE_HZ,
    pace_playback: bool = False,
) -> None:
    """Drive a started-or-startable AriaCapture through the detector.

    Subscribes to camera-rgb, samples frames at ``sample_hz`` in DEVICE time,
    rotates each RAW frame upright, and folds detections into ``aggregator``.
    Runs until the VRS ends, or ``duration_s`` of device time elapses.
    Blocking; owns start/stop of ``capture``.

    ``pace_playback`` (VRS only): faster-than-realtime playback plus the
    drop-stale image buffer means slow YOLO inference would see only a few
    frames of the whole file. Subscribing a no-op imu-right consumer engages
    the capture layer's subscriber-aware backpressure, pacing playback to
    detection throughput so the sampler sees the full recording. Leave False
    for live capture, where drop-stale is the correct behavior.
    """
    from aria_capture import rotate_upright  # local import keeps module torch-only-optional

    min_gap_ns = int(1e9 / sample_hz)
    state = {"last_ts": None, "first_ts": None}

    def on_rgb(sample) -> None:
        ts = sample.capture_timestamp_ns
        if state["first_ts"] is None:
            state["first_ts"] = ts
        if state["last_ts"] is not None and ts - state["last_ts"] < min_gap_ns:
            return
        state["last_ts"] = ts
        upright = rotate_upright(sample.frame)
        if sample.pixel_format != "rgb":
            upright = np.repeat(upright[:, :, None], 3, axis=2)
        aggregator.observe_frame(detector.detect(upright), upright)

    capture.subscribe("camera-rgb", on_rgb)
    if pace_playback:
        capture.subscribe("imu-right", lambda _sample: None)
    capture.start()
    try:
        while True:
            time.sleep(0.1)
            if capture.finished:
                break
            if (
                duration_s is not None
                and state["first_ts"] is not None
                and state["last_ts"] is not None
                and (state["last_ts"] - state["first_ts"]) / 1e9 >= duration_s
            ):
                break
    finally:
        capture.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone appliance-detection test over a VRS file.")
    parser.add_argument("--vrs", required=True, help="Path to an Aria Gen 1 VRS recording.")
    parser.add_argument("--duration", type=float, default=None, help="Max device-time seconds to scan.")
    args = parser.parse_args()

    from aria_capture import AriaCapture
    from config import CAPTURE_SOURCE_VRS

    detector = EnergyDetector()
    aggregator = ApplianceScanAggregator()
    capture = AriaCapture(source=CAPTURE_SOURCE_VRS, vrs_path=args.vrs)
    scan_capture_rgb(capture, detector, aggregator, duration_s=args.duration, pace_playback=True)

    print(f"\nFrames sampled: {aggregator.frames_observed}")
    counts = aggregator.counts()
    if not counts:
        print("No catalog appliances detected.")
        return
    confidences = aggregator.best_confidences()
    for name in sorted(counts):
        print(f"  {name:<14} x{counts[name]}  conf={confidences[name]}")


if __name__ == "__main__":
    main()
