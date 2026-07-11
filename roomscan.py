"""RoomScan energy audit orchestrator.

Runs an appliance scan over an Aria Gen 1 recording (--vrs) or a live stream
(--live), aggregates detections with the max-simultaneous counting rule,
estimates energy use from the config catalog, and writes:

    <out>/roomscan_report.json     full machine-readable report
    <out>/crops/<class>_<i>.jpg    best-confidence crop per counted instance
    <out>/roomscan_report.html     self-contained browser report page

Also registers the session (room name, timestamp, totals, recommendations,
report paths) in the shared cross-scan index at
<ROOMSCAN_OUTPUT_DIR>/roomscan_sessions.json via energy_sessions.py, so
past scans can be listed/reviewed/compared without re-parsing every
session's report -- see energy_sessions.py / roomscan_dashboard.py.

Requires ultralytics (+ torch) and, for --vrs, projectaria_tools; --live
additionally needs the Aria Client SDK (Mac). The --live path is a thin
pass-through to AriaCapture's live backend and is exercised on the Mac.

    python roomscan.py --vrs ~/aria-data/walkthrough.vrs --room-name "Living room"
    python roomscan.py --live [--start-streaming --device-ip <ip> --interface usb]
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from aria_capture import AriaCapture
from config import (
    CAPTURE_SOURCE_LIVE,
    CAPTURE_SOURCE_VRS,
    DEFAULT_STREAM_PROFILE,
    ENERGY_COST_PER_KWH_USD,
    ENERGY_DETECT_CONFIDENCE,
    ENERGY_FRAME_SAMPLE_HZ,
    GEMINI_DISCOVERY_DEFAULT_COUNT,
    GEMINI_DISCOVERY_DEFAULT_HOURS_PER_DAY,
    GEMINI_DISCOVERY_DEFAULT_WATTS,
    ROOMSCAN_CROP_DIR_NAME,
    ROOMSCAN_LIVE_DURATION_SECONDS,
    ROOMSCAN_OUTPUT_DIR,
    ROOMSCAN_REPORT_HTML_NAME,
    ROOMSCAN_REPORT_JSON_NAME,
)
from energy_detector import ApplianceScanAggregator, EnergyDetector, scan_capture_rgb
from energy_estimator import estimate_discovered_device, estimate_room
from energy_gemini import get_recommendations
from energy_report import render_html
from energy_sessions import register_session
from logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoomScan appliance energy audit (VRS or live).")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--vrs", help="Path to an Aria Gen 1 VRS recording.")
    source.add_argument("--live", action="store_true", help="Scan from a running Aria live stream.")
    parser.add_argument("--device-ip", help="Glasses IPv4 (only with --live --start-streaming).")
    parser.add_argument("--start-streaming", action="store_true", help="Live mode: start streaming via DeviceClient first.")
    parser.add_argument("--interface", choices=["wifi", "usb"], default="wifi", help="Live streaming interface.")
    parser.add_argument("--profile", default=DEFAULT_STREAM_PROFILE, help="Live streaming profile.")
    parser.add_argument("--duration", type=float, default=None,
                        help=f"Device-time seconds to scan (default: whole file for VRS, {ROOMSCAN_LIVE_DURATION_SECONDS:.0f}s live).")
    parser.add_argument("--sample-hz", type=float, default=ENERGY_FRAME_SAMPLE_HZ, help="Detection frame sampling rate.")
    parser.add_argument("--confidence", type=float, default=ENERGY_DETECT_CONFIDENCE, help="YOLO confidence threshold.")
    parser.add_argument("--room-name", default="Room", help="Room label shown on the report.")
    parser.add_argument("--out", default=str(ROOMSCAN_OUTPUT_DIR), help="Output directory.")
    return parser.parse_args()


def save_crops(crops_by_class: Dict[str, List[np.ndarray]], crop_dir: Path) -> Dict[str, List[str]]:
    """Write per-instance crops as JPEGs; return {class: [relative paths]}."""
    crop_dir.mkdir(parents=True, exist_ok=True)
    rel_paths: Dict[str, List[str]] = {}
    for class_name, crops in crops_by_class.items():
        safe = class_name.replace(" ", "_")
        for i, crop_rgb in enumerate(crops):
            filename = f"{safe}_{i}.jpg"
            path = crop_dir / filename
            if cv2.imwrite(str(path), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)):
                rel_paths.setdefault(class_name, []).append(f"{ROOMSCAN_CROP_DIR_NAME}/{filename}")
            else:
                logger.warning("Failed to write crop %s.", path)
    return rel_paths


def merge_device_estimate_fields(
    estimate_devices: List[Dict[str, object]],
    confidences: Dict[str, List[float]],
    crop_paths: Optional[Dict[str, List[str]]] = None,
    notes: Optional[Dict[str, List[Optional[str]]]] = None,
) -> None:
    """Mutate estimate_room()'s device dicts in place with confidences/crops/notes.

    Shared by the batch report (crop_paths populated from save_crops) and the
    live dashboard snapshot (crop_paths=None -- crops are only saved once, at
    session end, so every device gets an empty crop list mid-scan). ``notes``
    is Gemini's optional per-instance type/model detail (best_notes()); absent
    or None entries just mean "nothing more specific than the label."
    """
    for device in estimate_devices:
        name = device["class_name"]
        device["confidences"] = confidences.get(name, [])
        device["crops"] = (crop_paths or {}).get(name, [])
        device["notes"] = (notes or {}).get(name, [])


def merge_discovered_devices(
    estimate: Dict[str, object],
    discovered_devices: List[Dict[str, object]],
) -> None:
    """Price Gemini's discovered non-catalog devices and fold them into
    estimate_room()'s devices/totals, in place, so they show up in the same
    priced device list/totals as YOLO-detected catalog appliances instead of
    only a separate unpriced call-out.

    Each discovered dict is expected to carry Gemini's vision-estimated
    per-unit ``watts_active``/``hours_per_day`` plus an instance ``count``
    (see energy_gemini.run_live_scan_pass); missing values (e.g. an
    older-shaped dict) fall back to
    GEMINI_DISCOVERY_DEFAULT_WATTS/HOURS_PER_DAY/COUNT rather than raising,
    matching this module's defensive-fallback conventions.
    """
    if not discovered_devices:
        return
    devices: List[Dict[str, object]] = estimate["devices"]
    for item in discovered_devices:
        priced = asdict(estimate_discovered_device(
            item["name"],
            item.get("watts_active", GEMINI_DISCOVERY_DEFAULT_WATTS),
            item.get("hours_per_day", GEMINI_DISCOVERY_DEFAULT_HOURS_PER_DAY),
            item.get("count", GEMINI_DISCOVERY_DEFAULT_COUNT),
        ))
        priced["confidences"] = []
        priced["crops"] = []
        notes = []
        if item.get("description"):
            notes.append(str(item["description"]))
        sightings = item.get("sightings")
        if isinstance(sightings, int) and sightings > 1:
            notes.append(f"Seen {sightings}x this scan")
        priced["notes"] = notes
        priced["source"] = "gemini_discovered"
        devices.append(priced)
    devices.sort(key=lambda d: d["kwh_per_year"], reverse=True)

    totals: Dict[str, object] = estimate["totals"]
    total_kwh_year = sum(d["kwh_per_year"] for d in devices)
    totals["device_count"] = sum(d["count"] for d in devices)
    totals["kwh_per_day"] = sum(d["kwh_per_day"] for d in devices)
    totals["kwh_per_year"] = total_kwh_year
    totals["cost_per_year_usd"] = total_kwh_year * ENERGY_COST_PER_KWH_USD


def build_report(
    room_name: str,
    source: str,
    aggregator: ApplianceScanAggregator,
    crop_paths: Dict[str, List[str]],
    scan_wall_seconds: float,
    gemini_discovered_devices: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    estimate = estimate_room(aggregator.counts())
    merge_device_estimate_fields(
        estimate["devices"], aggregator.best_confidences(), crop_paths, aggregator.best_notes()
    )
    merge_discovered_devices(estimate, gemini_discovered_devices or [])
    return {
        "scan": {
            "room_name": room_name,
            "source": source,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frames_sampled": aggregator.frames_observed,
            "scan_wall_seconds": round(scan_wall_seconds, 1),
            # Classes Gemini live-verification auto-corrected (crops excluded
            # from counts/confidences/crops as a likely misclassification) --
            # always [] for a --vrs/batch scan, since only roomscan_live.py's
            # background pass thread ever calls record_gemini_verdict().
            "gemini_rejected_classes": aggregator.gemini_rejected_classes(),
        },
        "devices": estimate["devices"],  # already includes priced gemini_discovered_devices, see above
        "totals": estimate["totals"],
        # Raw discovery log (name/description/sightings) for appliance types
        # Gemini spotted outside the COCO/ENERGY_CATALOG vocabulary during a
        # live scan's background pass -- merge_discovered_devices() above
        # already priced and folded a copy of each into "devices"/"totals";
        # this is kept for the sighting-count/description detail. Always []
        # for a --vrs/batch scan (no caller ever passes this argument there).
        "gemini_discovered_devices": list(gemini_discovered_devices or []),
        # No live frame at report-build time, so context-dependent rules
        # (e.g. "TV in a bright room") simply don't fire in the rule-based
        # fallback here -- only the device/co-occurrence and threshold rules
        # apply. get_recommendations() uses Gemini vision on the crops when
        # GEMINI_API_KEY is set, falling back to the rule engine otherwise.
        "recommendations": get_recommendations(estimate["devices"], estimate["totals"], aggregator.best_crops()),
    }


def finalize_scan(
    room_name: str,
    source: str,
    aggregator: ApplianceScanAggregator,
    scan_wall_seconds: float,
    out_dir: Path,
    gemini_discovered_devices: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    """Build the report, write JSON/HTML artifacts, and register the session.

    Shared by this module's CLI main() and roomscan_live.py's
    LiveScanController.finish() so a batch scan and a live-dashboard scan
    write byte-identical artifacts from the same aggregator state.
    """
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_paths = save_crops(aggregator.best_crops(), out_dir / ROOMSCAN_CROP_DIR_NAME)
    report = build_report(room_name, source, aggregator, crop_paths, scan_wall_seconds, gemini_discovered_devices)

    json_path = out_dir / ROOMSCAN_REPORT_JSON_NAME
    try:
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write report JSON {json_path}") from exc
    logger.info("Wrote %s", json_path)

    html_path = out_dir / ROOMSCAN_REPORT_HTML_NAME
    render_html(report, out_dir, html_path)
    logger.info("Wrote %s", html_path)

    register_session(report, out_dir)
    return report


def print_summary(report: Dict[str, object]) -> None:
    devices = report["devices"]
    totals = report["totals"]
    print(f"\nRoomScan report — {report['scan']['room_name']} "
          f"({report['scan']['frames_sampled']} frames sampled)")
    header = f"{'device':<18} {'count':>5} {'watts':>7} {'kWh/day':>9} {'kWh/yr':>9} {'$/yr':>8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for d in devices:
        print(f"{d['display_name']:<18} {d['count']:>5d} {d['watts_active']:>7.0f} "
              f"{d['kwh_per_day']:>9.2f} {d['kwh_per_year']:>9.0f} {d['cost_per_year_usd']:>8.2f}")
    print("-" * len(header))
    print(f"{'TOTAL':<18} {totals['device_count']:>5d} {'':>7} "
          f"{totals['kwh_per_day']:>9.2f} {totals['kwh_per_year']:>9.0f} {totals['cost_per_year_usd']:>8.2f}")


def main() -> None:
    args = parse_args()
    if args.live:
        capture = AriaCapture(
            source=CAPTURE_SOURCE_LIVE,
            device_ip=args.device_ip,
            start_streaming=args.start_streaming,
            streaming_interface=args.interface,
            profile_name=args.profile,
        )
        duration = args.duration if args.duration is not None else ROOMSCAN_LIVE_DURATION_SECONDS
    else:
        capture = AriaCapture(source=CAPTURE_SOURCE_VRS, vrs_path=args.vrs)
        duration = args.duration  # None -> whole file

    detector = EnergyDetector(confidence=args.confidence)
    aggregator = ApplianceScanAggregator()

    wall_start = time.monotonic()
    scan_capture_rgb(
        capture, detector, aggregator,
        duration_s=duration, sample_hz=args.sample_hz,
        pace_playback=not args.live,  # see scan_capture_rgb docstring
    )
    scan_wall_seconds = time.monotonic() - wall_start

    out_dir = Path(args.out).expanduser()
    source = "live" if args.live else str(args.vrs)
    report = finalize_scan(args.room_name, source, aggregator, scan_wall_seconds, out_dir)

    print_summary(report)
    print(f"\nReport page: {out_dir / ROOMSCAN_REPORT_HTML_NAME}")
    sys.exit(0 if report["devices"] else 2)


if __name__ == "__main__":
    main()
