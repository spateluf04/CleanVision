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
from typing import Callable, Dict, List, Optional

import numpy as np

from aria_capture import AriaCapture, rotate_upright
from config import (
    CAPTURE_SOURCE_LIVE,
    DEFAULT_STREAM_PROFILE,
    ENERGY_CATALOG,
    ENERGY_DETECT_CONFIDENCE,
    ENERGY_FRAME_SAMPLE_HZ,
    GEMINI_DISCOVERY_DEFAULT_COUNT,
    GEMINI_DISCOVERY_DEFAULT_HOURS_PER_DAY,
    GEMINI_DISCOVERY_DEFAULT_WATTS,
    GEMINI_LIVE_PASS_INTERVAL_SECONDS,
    GEMINI_VERIFY_MAX_CROPS,
    ROOMSCAN_LIVE_TICK_SECONDS,
    ROOMSCAN_OUTPUT_DIR,
    ROOMSCAN_REPORT_HTML_NAME,
    ROOMSCAN_SESSION_DIR_TIMESTAMP_FORMAT,
)
from energy_detector import ApplianceScanAggregator, Detection, EnergyDetector, build_rgb_sample_callback
from energy_estimator import estimate_room
from energy_gemini import ai_features_enabled, run_live_scan_pass
from logging_utils import get_logger
from roomscan import finalize_scan, merge_device_estimate_fields, merge_discovered_devices

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
        # Gemini live verification/discovery (energy_gemini.run_live_scan_pass):
        # unconditionally initialized so this state is always safe to read from
        # snapshot()/finish() -- e.g. FakeLiveScanController in
        # tests/test_roomscan_dashboard.py overrides start() entirely and never
        # spins the pass thread, but still calls the real snapshot()/finish().
        self._gemini_pass_stop_event = threading.Event()
        self._gemini_pass_thread: Optional[threading.Thread] = None
        # Non-blocking guard: if a Gemini call from the previous tick is still
        # in flight when the next tick fires, that tick is skipped outright
        # rather than queuing up, so a slow/stuck call can never pile up passes.
        self._gemini_pass_lock = threading.Lock()
        self._gemini_discovered_lock = threading.Lock()
        # lowercased name -> {"name", "description", "sightings"}; sightings
        # increments (rather than duplicating) when the same non-catalog
        # appliance type is discovered again on a later pass.
        self._gemini_discovered: Dict[str, Dict[str, object]] = {}
        # Simple flags for the dashboard's "AI just ran" indicator -- plain
        # attributes (not lock-guarded) are fine here, same convention this
        # class already uses for _logged_first_latest_frame etc.: read from
        # the Qt poll thread, written from the Gemini pass thread, and a torn
        # read of a bool/float is never actually observable in CPython.
        self._gemini_pass_active = False
        self._gemini_last_pass_ts: Optional[float] = None

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
        if self._should_run_gemini_pass():
            self._gemini_pass_stop_event.clear()
            self._gemini_pass_thread = threading.Thread(
                target=self._gemini_pass_loop, name="roomscan-gemini-live-pass", daemon=True
            )
            self._gemini_pass_thread.start()
            logger.info(
                "Gemini live verification/discovery pass thread started for room '%s' (interval=%.1fs).",
                self.room_name, GEMINI_LIVE_PASS_INTERVAL_SECONDS,
            )

    def _should_run_gemini_pass(self) -> bool:
        """Gate for the background Gemini pass thread: never runs in
        --debug-camera-only mode (no detector/aggregator activity to verify
        or discover against) or without GEMINI_API_KEY set. Kept as its own
        method (rather than inlined in start()) so it's directly unit-testable
        without needing a full start()/AriaCapture connection."""
        return not self.disable_detection and ai_features_enabled()

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

    def latest_detections(self) -> List[Detection]:
        """Most recently confirmed (stabilized) detections, box included, for
        drawing a live bounding-box overlay on top of latest_frame(). Always
        empty in --debug-camera-only mode (no detector/sample loop running)
        or before the first stabilized detection of the scan."""
        if self._sample_state is None:
            return []
        return self._sample_state["stabilizer"].live_detections()

    def _known_display_names(self) -> List[str]:
        """Appliance-type names Gemini should never re-report as "discovered":
        every catalog display name (YOLO could tag these directly) plus
        anything already surfaced by an earlier pass this session."""
        names = [entry["display"] for entry in ENERGY_CATALOG.values()]
        names.extend(item["name"] for item in self._gemini_discovered_list())
        return names

    def _gemini_discovered_list(self) -> List[Dict[str, object]]:
        with self._gemini_discovered_lock:
            return [dict(item) for item in self._gemini_discovered.values()]

    def _run_gemini_pass_once(self) -> None:
        """One combined Gemini verify+discover pass. Skips outright (rather
        than blocking) if the previous pass is still in flight, so a slow or
        stuck call can never cause passes to pile up on this daemon thread.
        run_live_scan_pass() itself never raises, but this is still wrapped
        defensively -- a background thread with an uncaught exception dies
        silently, which would be far harder to notice than a logged warning.
        """
        if not self._gemini_pass_lock.acquire(blocking=False):
            logger.debug("Gemini live pass still running; skipping this tick.")
            return
        self._gemini_pass_active = True
        try:
            candidates = self._aggregator.unverified_slots()[:GEMINI_VERIFY_MAX_CROPS]
            frame = self.latest_frame()
            result = run_live_scan_pass(candidates, frame, self._known_display_names())
            for class_name, slot_index, confidence, accepted, note, refined_class in result["verifications"]:
                self._aggregator.record_gemini_verdict(
                    class_name, slot_index, confidence, accepted, note=note, reclassified_class=refined_class,
                )
            if result["discovered"]:
                with self._gemini_discovered_lock:
                    for item in result["discovered"]:
                        key = item["name"].lower()
                        existing = self._gemini_discovered.get(key)
                        if existing is not None:
                            existing["sightings"] += 1
                            existing["description"] = item.get("description", "") or existing["description"]
                            existing["watts_active"] = item.get("watts_active", existing["watts_active"])
                            existing["hours_per_day"] = item.get("hours_per_day", existing["hours_per_day"])
                            # Max-simultaneous rule (mirrors ApplianceScanAggregator.counts()):
                            # a later pass reporting fewer instances (e.g. panned away from
                            # some lights) never lowers the count, only a higher one raises it.
                            existing["count"] = max(
                                existing.get("count", GEMINI_DISCOVERY_DEFAULT_COUNT),
                                item.get("count", GEMINI_DISCOVERY_DEFAULT_COUNT),
                            )
                        else:
                            self._gemini_discovered[key] = {
                                "name": item["name"],
                                "description": item.get("description", ""),
                                "sightings": 1,
                                "watts_active": item.get("watts_active", GEMINI_DISCOVERY_DEFAULT_WATTS),
                                "hours_per_day": item.get("hours_per_day", GEMINI_DISCOVERY_DEFAULT_HOURS_PER_DAY),
                                "count": item.get("count", GEMINI_DISCOVERY_DEFAULT_COUNT),
                            }
                logger.info(
                    "Gemini live pass for room '%s' discovered/updated %d appliance type(s).",
                    self.room_name, len(result["discovered"]),
                )
        except Exception as exc:
            logger.warning("Gemini live pass tick failed unexpectedly (%s); skipping this tick.", exc)
        finally:
            self._gemini_pass_active = False
            self._gemini_last_pass_ts = time.monotonic()
            self._gemini_pass_lock.release()

    def _gemini_pass_loop(self) -> None:
        while not self._gemini_pass_stop_event.wait(GEMINI_LIVE_PASS_INTERVAL_SECONDS):
            self._run_gemini_pass_once()

    def snapshot(self) -> Dict[str, object]:
        """Return the current incremental scan state.

        Safe to call repeatedly while running: counts/confidences/estimates
        never require a "finished" scan, matching how
        ApplianceScanAggregator.counts() already works mid-scan.
        """
        estimate = estimate_room(self._aggregator.counts())
        merge_device_estimate_fields(
            estimate["devices"], self._aggregator.best_confidences(), None, self._aggregator.best_notes()
        )
        merge_discovered_devices(estimate, self._gemini_discovered_list())
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
            "gemini_discovered_devices": self._gemini_discovered_list(),
            "gemini_rejected_classes": self._aggregator.gemini_rejected_classes(),
            "gemini_verification_enabled": self._should_run_gemini_pass(),
            "gemini_pass_active": self._gemini_pass_active,
            "gemini_last_pass_ts": self._gemini_last_pass_ts,
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
        # Signal the Gemini pass thread even if it was never started -- Event
        # objects are safe to set() unconditionally, and this guarantees a
        # thread that IS running gets told to stop as soon as its current
        # tick (if any) finishes, rather than only on finish()/GC.
        self._gemini_pass_stop_event.set()
        if self._running:
            self._capture.stop()
            self._running = False
            logger.info("Live scan capture stopped for room '%s'.", self.room_name)

    def finish(self, out_dir: str = str(ROOMSCAN_OUTPUT_DIR)) -> Dict[str, object]:
        """Stop capture if needed and write the final JSON/HTML report.

        Reuses roomscan.py's finalize_scan() (build_report + save_crops +
        JSON/HTML write + session registration) exactly as the batch CLI
        does, so the output artifacts are identical whether the scan came
        from --vrs, --live, or this live dashboard controller. Any
        Gemini-discovered non-catalog appliances accumulated this session are
        threaded through and priced into the final device list/totals too
        (see roomscan.py:merge_discovered_devices()).
        """
        if self._finished:
            raise RuntimeError("LiveScanController.finish() already called.")
        self.stop()
        scan_wall_seconds = self.seconds_since_start()

        report = finalize_scan(
            self.room_name, "live", self._aggregator, scan_wall_seconds, Path(out_dir),
            gemini_discovered_devices=self._gemini_discovered_list(),
        )

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
