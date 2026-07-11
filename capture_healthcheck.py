"""Pass/fail verification tool for AriaCapture streams.

This script runs :class:`aria_capture.AriaCapture` against a VRS recording or
the live Aria Client SDK for a fixed window, then reports per-stream received
counts, effective rates vs expected, max inter-sample gaps, timestamp
monotonicity, and cross-stream skew as an ASCII table. It depends on NumPy,
OpenCV (JPEG writing only), and aria_capture, and it produces sample frames
in the output directory plus a process exit code: 0 only when every expected
stream is alive and monotonic (and, in live mode, RGB/IMU skew is in bounds).

Effective Hz is computed from the device-time span of each stream, never
wall clock -- VRS playback runs faster than real time, so wall-clock rates
would be meaningless.
"""

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import cv2

from aria_capture import AriaCapture, ImageSample, rotate_upright
from config import (
    CAPTURE_ALL_LABELS,
    CAPTURE_IMAGE_LABELS,
    CAPTURE_SOURCE_LIVE,
    CAPTURE_SOURCE_VRS,
    DEFAULT_STREAM_PROFILE,
    EXPECTED_STREAM_RATES_HZ,
    HEALTHCHECK_DURATION_SECONDS,
    HEALTHCHECK_MAX_LIVE_SKEW_MS,
    HEALTHCHECK_OUTPUT_DIR,
    HEALTHCHECK_RATE_TOLERANCE,
)
from logging_utils import get_logger


logger = get_logger(__name__)

SAMPLE_IMAGE_NAMES = {
    "camera-rgb": "rgb.jpg",
    "camera-slam-left": "slam_left.jpg",
    "camera-slam-right": "slam_right.jpg",
    "camera-et-left": "et_left.jpg",
    "camera-et-right": "et_right.jpg",
}

STATUS_OK = "OK"
STATUS_RATE_WARN = "RATE?"
STATUS_NON_MONOTONIC = "NONMONO"
STATUS_DEAD = "DEAD"


class StreamStats:
    """Accumulate health metrics for one stream from dispatcher callbacks."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.lock = threading.Lock()
        self.count = 0
        self.first_ts_ns: Optional[int] = None
        self.last_ts_ns: Optional[int] = None
        self.max_gap_ns = 0
        self.monotonicity_violations = 0
        self.latest_image: Optional[ImageSample] = None
        self.mid_image: Optional[ImageSample] = None

    def update(self, sample) -> None:
        ts = sample.capture_timestamp_ns
        with self.lock:
            if self.first_ts_ns is None:
                self.first_ts_ns = ts
            elif self.last_ts_ns is not None:
                gap = ts - self.last_ts_ns
                if gap <= 0:
                    self.monotonicity_violations += 1
                elif gap > self.max_gap_ns:
                    self.max_gap_ns = gap
            self.last_ts_ns = ts
            self.count += 1
            if isinstance(sample, ImageSample):
                self.latest_image = sample

    def freeze_mid_image(self) -> None:
        """Pin the current frame as the mid-run sample (orientation check)."""
        with self.lock:
            if self.mid_image is None and self.latest_image is not None:
                self.mid_image = self.latest_image

    def snapshot(self) -> Dict[str, object]:
        with self.lock:
            span_s = 0.0
            if self.first_ts_ns is not None and self.last_ts_ns is not None and self.count > 1:
                span_s = (self.last_ts_ns - self.first_ts_ns) / 1e9
            effective_hz = (self.count - 1) / span_s if span_s > 0 else 0.0
            return {
                "count": self.count,
                "effective_hz": effective_hz,
                "max_gap_ms": self.max_gap_ns / 1e6,
                "monotonicity_violations": self.monotonicity_violations,
                "last_ts_ns": self.last_ts_ns,
                "first_ts_ns": self.first_ts_ns,
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AriaCapture stream healthcheck (VRS or live).")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--vrs", help="Path to an Aria Gen 1 VRS recording.")
    source.add_argument("--live", action="store_true", help="Subscribe to a running Aria Client SDK stream.")
    parser.add_argument("--device-ip", help="Glasses IPv4 (only with --live --start-streaming).")
    parser.add_argument("--start-streaming", action="store_true", help="Live mode: start streaming via DeviceClient first.")
    parser.add_argument("--interface", choices=["wifi", "usb"], default="wifi", help="Live streaming interface for --start-streaming.")
    parser.add_argument("--profile", default=DEFAULT_STREAM_PROFILE, help="Live streaming profile for --start-streaming.")
    parser.add_argument("--persistent-certs", action="store_true", help="Use installed persistent streaming certificates (via `aria streaming install-certs`) instead of ephemeral certificates.")
    parser.add_argument("--local-certs-dir", help="Optional persistent-cert directory override (only with --persistent-certs).")
    parser.add_argument("--duration", type=float, default=HEALTHCHECK_DURATION_SECONDS, help="Capture window in seconds (device time for VRS, wall clock for live).")
    parser.add_argument("--out", default=str(HEALTHCHECK_OUTPUT_DIR), help="Directory for sample frames.")
    return parser.parse_args()


def print_calibration_summary(capture: AriaCapture) -> None:
    print("\nDevice calibration summary")
    print("-" * 78)
    any_calib = False
    for label in CAPTURE_IMAGE_LABELS:
        calib = capture.get_calibration(label)
        if calib is None:
            print(f"  {label:20s} <no calibration>")
            continue
        any_calib = True
        focal = calib.get_focal_lengths()
        principal = calib.get_principal_point()
        size = calib.get_image_size()
        print(
            f"  {label:20s} model={calib.model_name()} "
            f"focal=({focal[0]:.2f}, {focal[1]:.2f}) "
            f"principal=({principal[0]:.2f}, {principal[1]:.2f}) "
            f"size={int(size[0])}x{int(size[1])}"
        )
    if not any_calib:
        print("  (live mode delivers no device calibration in this build)")


def save_sample_frames(stats: Dict[str, StreamStats], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, filename in SAMPLE_IMAGE_NAMES.items():
        stream = stats[label]
        with stream.lock:
            sample = stream.mid_image or stream.latest_image
        if sample is None:
            logger.warning("No frame captured for %s; skipping %s.", label, filename)
            continue
        upright = rotate_upright(sample.frame)
        if sample.pixel_format == "rgb":
            # BGR conversion happens only here, at write time, for cv2.
            upright = cv2.cvtColor(upright, cv2.COLOR_RGB2BGR)
        path = out_dir / filename
        if cv2.imwrite(str(path), upright):
            logger.info("Saved %s (%sx%s, %s).", path, upright.shape[1], upright.shape[0], sample.pixel_format)
        else:
            logger.warning("Failed to write %s.", path)


def run_healthcheck(args: argparse.Namespace) -> int:
    is_live = bool(args.live)
    if is_live:
        capture = AriaCapture(
            source=CAPTURE_SOURCE_LIVE,
            device_ip=args.device_ip,
            start_streaming=args.start_streaming,
            streaming_interface=args.interface,
            profile_name=args.profile,
            use_ephemeral_certs=not args.persistent_certs,
            local_certs_dir=args.local_certs_dir,
        )
    else:
        capture = AriaCapture(source=CAPTURE_SOURCE_VRS, vrs_path=args.vrs)

    stats = {label: StreamStats(label) for label in CAPTURE_ALL_LABELS}
    for label in CAPTURE_ALL_LABELS:
        capture.subscribe(label, stats[label].update)

    duration_s = float(args.duration)
    mid_frozen = False
    capture.start()
    wall_start = time.monotonic()
    try:
        while True:
            time.sleep(0.05)
            wall_elapsed = time.monotonic() - wall_start
            if is_live:
                device_elapsed = wall_elapsed
            else:
                first_ts = [s.first_ts_ns for s in stats.values() if s.first_ts_ns is not None]
                last_ts = [s.last_ts_ns for s in stats.values() if s.last_ts_ns is not None]
                device_elapsed = (max(last_ts) - min(first_ts)) / 1e9 if first_ts else 0.0
            if not mid_frozen and device_elapsed >= duration_s / 2.0:
                for label in CAPTURE_IMAGE_LABELS:
                    stats[label].freeze_mid_image()
                mid_frozen = True
            if device_elapsed >= duration_s:
                break
            if not is_live and capture.finished:
                logger.info("VRS file ended after %.2f s of device time.", device_elapsed)
                break
            if not is_live and wall_elapsed > max(120.0, duration_s * 10):
                logger.warning("VRS playback wall-clock timeout reached.")
                break
    finally:
        capture.stop()

    for label in CAPTURE_IMAGE_LABELS:
        stats[label].freeze_mid_image()
    save_sample_frames(stats, Path(args.out).expanduser())
    print_calibration_summary(capture)

    # Cross-stream skew: latest RGB frame vs latest right-IMU sample. Enforced
    # in live mode (<100 ms); informational for VRS playback.
    rgb_last = stats["camera-rgb"].snapshot()["last_ts_ns"]
    imu_last = stats["imu-right"].snapshot()["last_ts_ns"]
    skew_ms: Optional[float] = None
    if rgb_last is not None and imu_last is not None:
        skew_ms = abs(int(rgb_last) - int(imu_last)) / 1e6

    header = f"{'stream':<18} {'expected Hz':<13} {'count':>7} {'eff Hz':>9} {'max gap ms':>11} {'mono':>5} {'status':>8}"
    line = "-" * len(header)
    print("\nStream health report" + (" (live)" if is_live else f" ({args.vrs})"))
    print(line)
    print(header)
    print(line)

    all_alive_and_monotonic = True
    for label in CAPTURE_ALL_LABELS:
        snap = stats[label].snapshot()
        lo, hi = EXPECTED_STREAM_RATES_HZ[label]
        lo_tol = lo * (1.0 - HEALTHCHECK_RATE_TOLERANCE)
        hi_tol = hi * (1.0 + HEALTHCHECK_RATE_TOLERANCE)
        alive = snap["count"] > 0
        monotonic = snap["monotonicity_violations"] == 0
        rate_ok = alive and lo_tol <= snap["effective_hz"] <= hi_tol
        if not alive:
            status = STATUS_DEAD
        elif not monotonic:
            status = STATUS_NON_MONOTONIC
        elif not rate_ok:
            status = STATUS_RATE_WARN
        else:
            status = STATUS_OK
        if not alive or not monotonic:
            all_alive_and_monotonic = False
        print(
            f"{label:<18} {f'{lo:g}-{hi:g}':<13} {snap['count']:>7d} "
            f"{snap['effective_hz']:>9.2f} {snap['max_gap_ms']:>11.1f} "
            f"{'yes' if monotonic else 'NO':>5} {status:>8}"
        )
    print(line)

    if skew_ms is None:
        print("Cross-stream skew RGB vs imu-right: n/a (stream missing)")
        skew_ok = False if is_live else True
    else:
        limit_note = f"limit {HEALTHCHECK_MAX_LIVE_SKEW_MS:.0f} ms" if is_live else "informational in VRS playback"
        print(f"Cross-stream skew RGB vs imu-right: {skew_ms:.1f} ms ({limit_note})")
        skew_ok = (skew_ms <= HEALTHCHECK_MAX_LIVE_SKEW_MS) if is_live else True

    passed = all_alive_and_monotonic and skew_ok
    print(f"\nHealthcheck result: {'PASS' if passed else 'FAIL'}")
    if not passed:
        if not all_alive_and_monotonic:
            print("  -> at least one expected stream is dead or non-monotonic.")
        if not skew_ok:
            print("  -> live RGB/IMU skew exceeds the limit.")
    return 0 if passed else 1


def main() -> None:
    args = parse_args()
    sys.exit(run_healthcheck(args))


if __name__ == "__main__":
    main()
