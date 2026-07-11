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
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from config import (
    ENERGY_CATALOG,
    ENERGY_DETECT_CONFIDENCE,
    ENERGY_DUPLICATE_BOX_IOU_THRESHOLD,
    ENERGY_FRAME_SAMPLE_HZ,
    ENERGY_MIN_BOX_AREA_FRAC,
    ENERGY_STABILIZE_IOU_MATCH_THRESHOLD,
    ENERGY_STABILIZE_MAX_MISS_SECONDS,
    ENERGY_STABILIZE_MIN_HITS,
    ENERGY_STABILIZE_WINDOW_SECONDS,
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


def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _suppress_duplicate_boxes(
    detections: List[Detection], iou_threshold: float = ENERGY_DUPLICATE_BOX_IOU_THRESHOLD
) -> List[Detection]:
    """Collapse same-class boxes that overlap heavily into a single detection.

    YOLO occasionally fires two overlapping boxes on one physical object in a
    single frame; left alone this would inflate the per-frame count and,
    downstream, ApplianceScanAggregator's max-simultaneous count. Classic
    greedy NMS per class, keeping the highest-confidence box of each cluster.
    """
    by_class: Dict[str, List[Detection]] = {}
    for det in detections:
        by_class.setdefault(det.class_name, []).append(det)

    kept: List[Detection] = []
    for class_dets in by_class.values():
        class_dets.sort(key=lambda d: d.confidence, reverse=True)
        class_kept: List[Detection] = []
        for det in class_dets:
            if all(_iou(det.box_xyxy, k.box_xyxy) < iou_threshold for k in class_kept):
                class_kept.append(det)
        kept.extend(class_kept)
    return kept


@dataclass
class _CropSlot:
    confidence: float
    crop_rgb: np.ndarray  # upright RGB uint8
    # Gemini live-verification watermark (energy_gemini.run_live_scan_pass, via
    # roomscan_live.py's background pass): gemini_checked_confidence records the
    # confidence this slot had the last time Gemini judged its current crop, so
    # unverified_slots() can tell "never checked" and "checked, but a newer crop
    # replaced it" apart from "checked at this exact crop" with a simple
    # equality watermark -- no separate dirty flag needed. gemini_rejected is
    # the verdict itself; both fields are reset the instant a higher-confidence
    # detection replaces the crop (see observe_frame), so a rejection never
    # outlives the pixels it was judged on. gemini_note is a short free-text
    # type/model detail Gemini attaches while verifying (e.g. "55-inch
    # wall-mounted LED TV") -- purely descriptive, never affects counting --
    # and is reset alongside the other two fields on crop replacement.
    gemini_rejected: bool = False
    gemini_checked_confidence: Optional[float] = None
    gemini_note: Optional[str] = None


class ApplianceScanAggregator:
    """Aggregate per-frame detections into per-class counts and best crops.

    Pure logic (no YOLO/torch) so it can be unit-tested directly. Call
    :meth:`observe_frame` once per sampled frame, then read :meth:`counts`
    and :meth:`best_crops`.
    """

    def __init__(self) -> None:
        self.frames_observed = 0
        self._slots: Dict[str, List[_CropSlot]] = {}
        # Live mode calls observe_frame() from AriaCapture's dispatcher thread
        # while counts()/best_crops()/best_confidences() are read from the Qt
        # main thread (or a ticker thread) via LiveScanController.snapshot() --
        # without this lock, a new class key inserted mid-iteration raises
        # "dictionary changed size during iteration".
        self._lock = threading.Lock()

    def observe_frame(self, detections: List[Detection], frame_rgb: Optional[np.ndarray]) -> None:
        """Fold one frame's detections in. ``frame_rgb`` may be None in tests
        (crops are then skipped, counts still update)."""
        by_class: Dict[str, List[Detection]] = {}
        for det in detections:
            by_class.setdefault(det.class_name, []).append(det)

        with self._lock:
            self.frames_observed += 1
            for class_name, dets in by_class.items():
                dets.sort(key=lambda d: d.confidence, reverse=True)
                slots = self._slots.setdefault(class_name, [])
                # Grow to the new simultaneous max; never shrink.
                while len(slots) < len(dets):
                    slots.append(_CropSlot(confidence=-1.0, crop_rgb=None))  # type: ignore[arg-type]
                for i, det in enumerate(dets):
                    if det.confidence > slots[i].confidence:
                        slots[i].confidence = det.confidence
                        # New crop pixels = an unverified candidate again; any
                        # prior Gemini verdict applied to the OLD pixels, so it
                        # must not silently carry over onto the new ones.
                        slots[i].gemini_rejected = False
                        slots[i].gemini_checked_confidence = None
                        slots[i].gemini_note = None
                        if frame_rgb is not None:
                            slots[i].crop_rgb = _extract_crop(frame_rgb, det.box_xyxy)

    def counts(self) -> Dict[str, int]:
        """{class_name: max simultaneous detections seen in one frame}, excluding
        slots Gemini has rejected as a misclassification (see record_gemini_verdict)."""
        with self._lock:
            result: Dict[str, int] = {}
            for name, slots in self._slots.items():
                n = sum(1 for s in slots if not s.gemini_rejected)
                if n:
                    result[name] = n
            return result

    def best_crops(self) -> Dict[str, List[np.ndarray]]:
        """Per class, one best-confidence RGB crop per counted instance slot
        (Gemini-rejected slots excluded, matching counts())."""
        with self._lock:
            return {
                name: [s.crop_rgb for s in slots if s.crop_rgb is not None and not s.gemini_rejected]
                for name, slots in self._slots.items()
            }

    def best_confidences(self) -> Dict[str, List[float]]:
        with self._lock:
            return {
                name: [round(s.confidence, 3) for s in slots if not s.gemini_rejected]
                for name, slots in self._slots.items()
            }

    def best_notes(self) -> Dict[str, List[Optional[str]]]:
        """Per class, one Gemini type/model note per counted instance slot
        (None where Gemini hasn't verified the slot or gave no note),
        same order/filtering as best_confidences()/best_crops()."""
        with self._lock:
            return {
                name: [s.gemini_note for s in slots if not s.gemini_rejected]
                for name, slots in self._slots.items()
            }

    def unverified_slots(self) -> List[Tuple[str, int, float, Optional[np.ndarray]]]:
        """Every ``(class_name, slot_index, confidence, crop_rgb)`` slot whose
        crop hasn't been Gemini-judged at its current confidence yet.

        Comparing ``gemini_checked_confidence`` (a watermark, not a dirty flag)
        against the slot's live ``confidence`` is what makes a crop replacement
        automatically re-enter the unverified pool: observe_frame() resets the
        watermark to None on replacement, so a not-yet-equal comparison is true
        again without any extra bookkeeping here.
        """
        with self._lock:
            result: List[Tuple[str, int, float, Optional[np.ndarray]]] = []
            for name, slots in self._slots.items():
                for idx, slot in enumerate(slots):
                    if slot.crop_rgb is None:
                        continue
                    if slot.gemini_checked_confidence != slot.confidence:
                        result.append((name, idx, slot.confidence, slot.crop_rgb))
            return result

    def record_gemini_verdict(
        self,
        class_name: str,
        slot_index: int,
        expected_confidence: float,
        accepted: bool,
        note: Optional[str] = None,
    ) -> bool:
        """Compare-and-swap write of a Gemini verify verdict onto one crop slot.

        ``note`` is an optional short type/model detail (e.g. "55-inch
        wall-mounted LED TV") Gemini attached while verifying -- purely
        descriptive, stored regardless of ``accepted`` but only surfaced via
        best_notes() for non-rejected slots.

        Returns False (a no-op) if the slot moved on since the caller read it
        via unverified_slots() -- a new, higher-confidence crop replaced it (or
        the class/index no longer exists) -- so a stale verdict is discarded
        rather than misapplied to pixels it never actually judged.
        """
        with self._lock:
            slots = self._slots.get(class_name)
            if slots is None or slot_index >= len(slots):
                return False
            slot = slots[slot_index]
            if slot.confidence != expected_confidence:
                return False
            slot.gemini_checked_confidence = expected_confidence
            slot.gemini_rejected = not accepted
            slot.gemini_note = note
            return True

    def gemini_rejected_classes(self) -> List[str]:
        """Classes with >=1 currently-rejected slot, for the UI's misclassification hint."""
        with self._lock:
            return sorted(name for name, slots in self._slots.items() if any(s.gemini_rejected for s in slots))


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
        return _suppress_duplicate_boxes(detections)


@dataclass
class _Track:
    box_xyxy: Tuple[int, int, int, int]
    confidence: float
    hit_timestamps_ns: Deque[int]
    last_seen_ns: int


class DetectionStabilizer:
    """Temporal smoothing layer between EnergyDetector.detect() and ApplianceScanAggregator.

    Raw per-frame YOLO detections flicker: a real, stationary appliance can
    drop out for a frame (motion blur, angle, confidence dip) or a spurious
    box can appear for a single frame. Neither should reach the aggregator
    as-is -- a dropout would look like the object left, and a spurious box
    would permanently inflate ApplianceScanAggregator's max-simultaneous
    count. This class buffers a rolling window of per-class, per-track hit
    timestamps (device time) and only reports a track once it has been seen
    ``min_hits`` times inside ``window_seconds``, keeping it alive for
    ``max_miss_seconds`` after its last hit before dropping it.

    Tracks are matched frame-to-frame within a class by IOU (greedy, highest
    IOU above ``iou_match_threshold`` wins) -- a lightweight stand-in for a
    full SORT/Kalman tracker, sufficient because RoomScan's targets are
    largely stationary appliances rather than fast-moving objects.
    """

    def __init__(
        self,
        window_seconds: float = ENERGY_STABILIZE_WINDOW_SECONDS,
        min_hits: int = ENERGY_STABILIZE_MIN_HITS,
        max_miss_seconds: float = ENERGY_STABILIZE_MAX_MISS_SECONDS,
        iou_match_threshold: float = ENERGY_STABILIZE_IOU_MATCH_THRESHOLD,
    ) -> None:
        self._window_ns = int(window_seconds * 1e9)
        self._min_hits = min_hits
        self._max_miss_ns = int(max_miss_seconds * 1e9)
        self._iou_match_threshold = iou_match_threshold
        self._tracks: Dict[str, List[_Track]] = {}
        self._instantaneous_counts: Dict[str, int] = {}
        self._last_stabilized: List[Detection] = []
        # Same cross-thread hazard as ApplianceScanAggregator: update() runs on
        # AriaCapture's dispatcher thread, stabilized_counts()/
        # instantaneous_counts()/live_detections() are read from the Qt
        # main/ticker thread.
        self._lock = threading.Lock()

    def update(self, detections: List[Detection], timestamp_ns: int) -> List[Detection]:
        """Feed one frame's raw detections; return the stabilized detections for that frame.

        The returned list has one representative Detection per confirmed
        (min_hits reached, not yet timed out) track, ready to hand straight
        to ApplianceScanAggregator.observe_frame().
        """
        instantaneous_counts: Dict[str, int] = {}
        for det in detections:
            instantaneous_counts[det.class_name] = instantaneous_counts.get(det.class_name, 0) + 1

        by_class: Dict[str, List[Detection]] = {}
        for det in detections:
            by_class.setdefault(det.class_name, []).append(det)

        with self._lock:
            self._instantaneous_counts = instantaneous_counts
            stabilized: List[Detection] = []
            for class_name in set(self._tracks) | set(by_class):
                tracks = self._tracks.get(class_name, [])
                unmatched_tracks = list(tracks)

                for det in by_class.get(class_name, []):
                    best_track, best_iou = None, self._iou_match_threshold
                    for track in unmatched_tracks:
                        iou = _iou(det.box_xyxy, track.box_xyxy)
                        if iou >= best_iou:
                            best_track, best_iou = track, iou
                    if best_track is not None:
                        best_track.box_xyxy = det.box_xyxy
                        best_track.confidence = det.confidence
                        best_track.hit_timestamps_ns.append(timestamp_ns)
                        best_track.last_seen_ns = timestamp_ns
                        unmatched_tracks.remove(best_track)
                    else:
                        tracks.append(
                            _Track(
                                box_xyxy=det.box_xyxy,
                                confidence=det.confidence,
                                hit_timestamps_ns=deque([timestamp_ns]),
                                last_seen_ns=timestamp_ns,
                            )
                        )

                surviving: List[_Track] = []
                for track in tracks:
                    while track.hit_timestamps_ns and timestamp_ns - track.hit_timestamps_ns[0] > self._window_ns:
                        track.hit_timestamps_ns.popleft()
                    if timestamp_ns - track.last_seen_ns > self._max_miss_ns:
                        continue
                    surviving.append(track)
                    if len(track.hit_timestamps_ns) >= self._min_hits:
                        stabilized.append(
                            Detection(class_name=class_name, confidence=track.confidence, box_xyxy=track.box_xyxy)
                        )

                if surviving:
                    self._tracks[class_name] = surviving
                else:
                    self._tracks.pop(class_name, None)

            self._last_stabilized = stabilized
            return stabilized

    def instantaneous_counts(self) -> Dict[str, int]:
        """Raw, unfiltered per-class counts from the most recently observed frame."""
        with self._lock:
            return dict(self._instantaneous_counts)

    def live_detections(self) -> List[Detection]:
        """Snapshot of the most recently confirmed (stabilized) detections,
        each with its box -- for a live bounding-box overlay. Independent of
        update()'s per-call return value so a UI poller (running on its own
        timer, not the capture callback) can read the latest state at any
        cadence."""
        with self._lock:
            return list(self._last_stabilized)

    def stabilized_counts(self) -> Dict[str, int]:
        """Confirmed-alive per-class counts after windowed hysteresis (feeds energy estimation)."""
        with self._lock:
            counts: Dict[str, int] = {}
            for class_name, tracks in self._tracks.items():
                n = sum(1 for t in tracks if len(t.hit_timestamps_ns) >= self._min_hits)
                if n:
                    counts[class_name] = n
            return counts


def build_rgb_sample_callback(
    detector: EnergyDetector,
    aggregator: ApplianceScanAggregator,
    sample_hz: float = ENERGY_FRAME_SAMPLE_HZ,
) -> Tuple[Callable[[Any], None], Dict[str, Any]]:
    """Build a throttled camera-rgb callback that detects+aggregates one frame at sample_hz.

    Returns ``(callback, state)``. ``state`` is mutated in place with:
    device-time ``first_ts``/``last_ts`` nanoseconds as frames arrive (used
    for a duration cutoff, see scan_capture_rgb), and a ``stabilizer``
    (DetectionStabilizer) that smooths raw per-frame detections before they
    reach ``aggregator`` -- callers that want the instantaneous (unsmoothed)
    view can read ``state["stabilizer"].instantaneous_counts()``. This is the
    one place the throttle/rotate/detect/stabilize/aggregate step is defined,
    shared by the batch ``scan_capture_rgb`` and the live dashboard
    controller (``roomscan_live.py``).
    """
    from aria_capture import rotate_upright  # local import keeps module torch-only-optional

    min_gap_ns = int(1e9 / sample_hz)
    state: Dict[str, Any] = {
        "last_ts": None,
        "first_ts": None,
        "stabilizer": DetectionStabilizer(),
        "_logged_first_sample": False,
        "_logged_first_stabilized": False,
    }

    def on_rgb(sample) -> None:
        ts = sample.capture_timestamp_ns
        if state["first_ts"] is None:
            state["first_ts"] = ts
        if state["last_ts"] is not None and ts - state["last_ts"] < min_gap_ns:
            return
        state["last_ts"] = ts
        if not state["_logged_first_sample"]:
            logger.info("build_rgb_sample_callback: first sampled frame accepted (ts_ns=%d).", ts)
            state["_logged_first_sample"] = True
        upright = rotate_upright(sample.frame)
        if sample.pixel_format != "rgb":
            upright = np.repeat(upright[:, :, None], 3, axis=2)
        raw_detections = detector.detect(upright)
        stabilized = state["stabilizer"].update(raw_detections, ts)
        aggregator.observe_frame(stabilized, upright)
        logger.debug(
            "on_rgb sampled frame: raw_detections=%d stabilized=%d frames_observed=%d",
            len(raw_detections), len(stabilized), aggregator.frames_observed,
        )
        if stabilized and not state["_logged_first_stabilized"]:
            logger.info(
                "build_rgb_sample_callback: first stabilized detection(s) reached the aggregator: %s",
                [d.class_name for d in stabilized],
            )
            state["_logged_first_stabilized"] = True

    return on_rgb, state


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
    on_rgb, state = build_rgb_sample_callback(detector, aggregator, sample_hz)

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
