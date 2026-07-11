"""Live single-room RoomScan backend service.

Wraps AriaCapture (live backend) + EnergyDetector + ApplianceScanAggregator
into a controller that runs continuously and exposes incremental scan state
via :meth:`LiveScanController.snapshot` at any point while capture is active
-- not just a final report. Session end reuses roomscan.py's finalize_scan()
verbatim (build_report + save_crops + JSON/HTML write + session
registration), so the artifacts produced here are identical to a completed
--live or --vrs roomscan.py run and show up alongside batch-CLI scans in
the shared session index.

Requires the Aria Client SDK (Mac) and ultralytics, same as roomscan.py
--live. Standalone manual test (prints a live snapshot line to stdout until
Ctrl+C, then writes the final report):

    python roomscan_live.py --room-name "Living room" [--start-streaming --device-ip <ip> --interface usb]
"""

import argparse
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from aria_capture import AriaCapture, rotate_upright
from config import (
    CAPTURE_SOURCE_LIVE,
    DEFAULT_STREAM_PROFILE,
    ENERGY_DETECT_CONFIDENCE,
    ENERGY_FRAME_SAMPLE_HZ,
    ROOMSCAN_LIVE_TICK_SECONDS,
    ROOMSCAN_OUTPUT_DIR,
    ROOMSCAN_REPORT_HTML_NAME,
    ROOMSCAN_SESSION_DIR_TIMESTAMP_FORMAT,
)
from energy_detector import ApplianceScanAggregator, EnergyDetector, build_rgb_sample_callback
from energy_estimator import estimate_room
from logging_utils import get_logger
from roomscan import finalize_scan, merge_device_estimate_fields

logger = get_logger(__name__)


def _slugify(name: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in name.strip().lower())
    return safe.strip("_") or "room"


def session_out_dir(base_dir: Path, room_name: str) -> Path:
    """Mint this session's <room-slug>_<timestamp> output folder under base_dir.

    Centralizes the folder-naming contract energy_sessions.py's session_id
    relies on (e.g. ``kitchen_20260101_120000``), so any caller that starts a
    live session -- the dashboard today, a future non-Qt caller tomorrow --
    mints identical, uniquely-timestamped folder names.
    """
    return Path(base_dir).expanduser() / f"{_slugify(room_name)}_{time.strftime(ROOMSCAN_SESSION_DIR_TIMESTAMP_FORMAT)}"


class LiveScanController:
    """Continuously scans one room from a live Aria stream.

    :meth:`start` begins capture and detection; :meth:`snapshot` returns the
    current incremental scan state at any point while running (safe to call
    from another thread); :meth:`finish` stops capture (if still running)
    and writes the same JSON/HTML report roomscan.py produces for a
    completed scan.

    ``disable_detection=True`` is a debug fallback: it skips constructing
    EnergyDetector (no YOLO weight load at all) and never subscribes the
    camera-rgb sample/detect/aggregate callback, so a broken or slow detector
    can never be the reason the live feed doesn't show up. AriaCapture still
    writes every incoming camera-rgb frame into its own latest-value slot
    regardless of subscriptions, so :meth:`latest_frame` keeps working
    unchanged -- this flag isolates "is the camera pipeline healthy" from "is
    detection healthy" by removing the second variable entirely rather than
    just ignoring its output.
    """

    def __init__(
        self,
        room_name: str,
        device_ip: Optional[str] = None,
        start_streaming: bool = False,
        streaming_interface: str = "wifi",
        profile_name: str = DEFAULT_STREAM_PROFILE,
        confidence: float = ENERGY_DETECT_CONFIDENCE,
        sample_hz: float = ENERGY_FRAME_SAMPLE_HZ,
        use_ephemeral_certs: bool = True,
        local_certs_dir: Optional[str] = None,
        disable_detection: bool = False,
    ) -> None:
        self.room_name = room_name
        self.disable_detection = disable_detection
        self._capture = AriaCapture(
            source=CAPTURE_SOURCE_LIVE,
            device_ip=device_ip,
            start_streaming=start_streaming,
            streaming_interface=streaming_interface,
            profile_name=profile_name,
            use_ephemeral_certs=use_ephemeral_certs,
            local_certs_dir=local_certs_dir,
        )
        self._detector = None if disable_detection else EnergyDetector(confidence=confidence)
        self._aggregator = ApplianceScanAggregator()
        self._sample_hz = sample_hz
        self._wall_start: Optional[float] = None
        self._sample_state: Optional[Dict[str, object]] = None
        self._running = False
        self._finished = False
        self._logged_first_latest_frame = False
        self._logged_first_snapshot_frame = False

    def start(self) -> None:
        """Begin continuous live capture (+ detection, unless disabled)."""
        if self._running:
            raise RuntimeError("LiveScanController already started.")
        if self.disable_detection:
            logger.warning(
                "LiveScanController starting with detection DISABLED (debug camera-only mode) "
                "for room '%s'; no appliances will be counted and the saved report will be empty.",
                self.room_name,
            )
        else:
            on_rgb, sample_state = build_rgb_sample_callback(self._detector, self._aggregator, self._sample_hz)
            self._sample_state = sample_state
            self._capture.subscribe("camera-rgb", on_rgb)
        self._capture.start()
        self._wall_start = time.monotonic()
        self._running = True
        logger.info(
            "LiveScanController started for room '%s' (start_streaming=%s, ephemeral_certs=%s, "
            "detection_disabled=%s).",
            self.room_name,
            self._capture.start_streaming,
            self._capture.use_ephemeral_certs,
            self.disable_detection,
        )

    @property
    def running(self) -> bool:
        """True after start() and before stop()/finish()."""
        return self._running

    @property
    def finished(self) -> bool:
        """True once finish() has completed (report already written)."""
        return self._finished

    def last_error(self) -> Optional[str]:
        """Return the most recent live streaming-client failure message, if any."""
        return self._capture.last_error()

    def seconds_since_start(self) -> float:
        """Wall-clock seconds since start() was called (0.0 if never started).

        Single source of truth for "how long has this scan been running" so
        callers (e.g. the dashboard's stale-frame timeout and scan-duration
        display) don't duplicate the monotonic-clock bookkeeping this class
        already keeps internally for finish()'s scan_wall_seconds.
        """
        return time.monotonic() - self._wall_start if self._wall_start else 0.0

    def latest_frame(self) -> Optional[np.ndarray]:
        """Return the latest live camera-rgb frame, upright and RGB, for display.

        Pull-style like AriaCapture.latest(): never consumes the sample, so
        it's safe to poll from a UI timer at any cadence, independent of the
        detection sample_hz.
        """
        sample = self._capture.latest("camera-rgb")
        if sample is None:
            logger.debug("latest_frame(): no camera-rgb sample buffered yet.")
            return None
        upright = rotate_upright(sample.frame)
        if sample.pixel_format != "rgb":
            upright = np.repeat(upright[:, :, None], 3, axis=2)
        frame = np.ascontiguousarray(upright)
        if not self._logged_first_latest_frame:
            logger.info(
                "latest_frame() returning first camera-rgb frame (shape=%s, ts_ns=%d).",
                frame.shape,
                sample.capture_timestamp_ns,
            )
            self._logged_first_latest_frame = True
        else:
            logger.debug("latest_frame() returning frame (ts_ns=%d).", sample.capture_timestamp_ns)
        return frame

    def snapshot(self) -> Dict[str, object]:
        """Return the current incremental scan state.

        Safe to call repeatedly while running: counts/confidences/estimates
        never require a "finished" scan, matching how
        ApplianceScanAggregator.counts() already works mid-scan.
        """
        estimate = estimate_room(self._aggregator.counts())
        merge_device_estimate_fields(estimate["devices"], self._aggregator.best_confidences(), None)
        watts_active_total = sum(d["watts_active"] * d["count"] for d in estimate["devices"])
        stabilizer = self._sample_state["stabilizer"] if self._sample_state else None
        frames_sampled = self._aggregator.frames_observed
        if not self._logged_first_snapshot_frame and frames_sampled > 0:
            logger.info("snapshot(): frames_sampled > 0 (first observed at %d).", frames_sampled)
            self._logged_first_snapshot_frame = True
        logger.debug(
            "snapshot(): frames_sampled=%d devices=%d watts_active=%.0f",
            frames_sampled, len(estimate["devices"]), watts_active_total,
        )
        return {
            "room_name": self.room_name,
            "frames_sampled": frames_sampled,
            "devices": estimate["devices"],
            "totals": {**estimate["totals"], "watts_active": watts_active_total},
            "instantaneous_counts": stabilizer.instantaneous_counts() if stabilizer else {},
            "stabilized_counts": stabilizer.stabilized_counts() if stabilizer else {},
        }

    def run_ticker(
        self,
        on_update: Callable[[Dict[str, object]], None],
        interval_s: float = ROOMSCAN_LIVE_TICK_SECONDS,
        stop_event: Optional[threading.Event] = None,
    ) -> threading.Thread:
        """Call ``on_update(snapshot())`` on a background thread every interval_s.

        Decouples the UI/publish cadence from the detection cadence
        (``sample_hz``). Returns the daemon thread; the caller stops it by
        setting ``stop_event`` (a fresh one is created if omitted).
        """
        stop_event = stop_event or threading.Event()

        def _loop() -> None:
            while not stop_event.is_set():
                on_update(self.snapshot())
                stop_event.wait(interval_s)

        thread = threading.Thread(target=_loop, name="roomscan-live-ticker", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        """Stop capture; the last scan state remains readable via snapshot()."""
        if self._running:
            self._capture.stop()
            self._running = False
            logger.info("Live scan capture stopped for room '%s'.", self.room_name)

    def finish(self, out_dir: str = str(ROOMSCAN_OUTPUT_DIR)) -> Dict[str, object]:
        """Stop capture if needed and write the final JSON/HTML report.

        Reuses roomscan.py's finalize_scan() (build_report + save_crops +
        JSON/HTML write + session registration) exactly as the batch CLI
        does, so the output artifacts are identical whether the scan came
        from --vrs, --live, or this live dashboard controller.
        """
        if self._finished:
            raise RuntimeError("LiveScanController.finish() already called.")
        self.stop()
        scan_wall_seconds = self.seconds_since_start()

        report = finalize_scan(self.room_name, "live", self._aggregator, scan_wall_seconds, Path(out_dir))

        self._finished = True
        logger.info("Live scan finished for room '%s'; wrote report to %s.", self.room_name, out_dir)
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual test driver for the live RoomScan backend service.")
    parser.add_argument("--room-name", default="Room", help="Room label for this live scan.")
    parser.add_argument("--device-ip", help="Glasses IPv4 (only with --start-streaming).")
    parser.add_argument("--start-streaming", action="store_true", help="Start streaming via DeviceClient first.")
    parser.add_argument("--interface", choices=["wifi", "usb"], default="wifi", help="Live streaming interface.")
    parser.add_argument("--profile", default=DEFAULT_STREAM_PROFILE, help="Live streaming profile.")
    parser.add_argument(
        "--persistent-certs", action="store_true",
        help="Use installed persistent streaming certificates (via `aria streaming install-certs`) instead of ephemeral certificates.",
    )
    parser.add_argument("--local-certs-dir", help="Optional persistent-cert directory override (only with --persistent-certs).")
    parser.add_argument("--out", default=str(ROOMSCAN_OUTPUT_DIR), help="Output directory for the final report.")
    parser.add_argument(
        "--debug-camera-only", action="store_true",
        help="Debug mode: disable EnergyDetector entirely and only exercise live frame delivery "
        "(latest_frame()/the printed snapshot line). Use to isolate a capture-layer problem from a "
        "detector problem.",
    )
    return parser.parse_args()


def _print_snapshot(state: Dict[str, object]) -> None:
    totals = state["totals"]
    print(
        f"\r[{state['room_name']}] frames={state['frames_sampled']:<5} "
        f"devices={len(state['devices']):<2} watts={totals['watts_active']:<7.0f} "
        f"kWh/day={totals['kwh_per_day']:<6.2f} kWh/yr={totals['kwh_per_year']:<7.0f} "
        f"$/yr={totals['cost_per_year_usd']:<7.2f}",
        end="",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    controller = LiveScanController(
        room_name=args.room_name,
        device_ip=args.device_ip,
        start_streaming=args.start_streaming,
        streaming_interface=args.interface,
        profile_name=args.profile,
        use_ephemeral_certs=not args.persistent_certs,
        local_certs_dir=args.local_certs_dir,
        disable_detection=args.debug_camera_only,
    )
    controller.start()
    stop_event = threading.Event()
    controller.run_ticker(_print_snapshot, stop_event=stop_event)
    print("Live scan running -- press Ctrl+C to finish and write the report.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        print()
        report = controller.finish(out_dir=args.out)
        print(f"Report page: {Path(args.out).expanduser() / ROOMSCAN_REPORT_HTML_NAME}")
        print(f"Devices found: {len(report['devices'])}")


if __name__ == "__main__":
    main()
