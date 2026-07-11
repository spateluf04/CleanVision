"""Session index / lightweight local store for RoomScan energy-audit runs.

Every finished scan (roomscan.py --vrs/--live, or the live dashboard's Save
Report) already writes a self-contained ``<out_dir>/roomscan_report.json`` +
``roomscan_report.html`` + ``crops/``. This module adds one more small JSON
file -- ``config.ROOMSCAN_SESSIONS_INDEX_NAME``, living at the root of
``config.ROOMSCAN_OUTPUT_DIR`` -- that indexes every session's summary
fields plus a pointer back to its own report/HTML paths, so a caller (the
dashboard, a script, a notebook) can list/review/compare/export past
sessions without re-scanning the filesystem or re-parsing every session's
full report+crops.

``register_session()`` is the only write path; everything else is
read-only. Each session record is keyed by ``session_id`` (the report's own
output folder name, e.g. ``kitchen_20260101_120000`` -- already unique
because roomscan.py/roomscan_dashboard.py mint that folder name from
``<room-slug>_<timestamp>``), so nothing new needs inventing for identity.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    ROOMSCAN_OUTPUT_DIR,
    ROOMSCAN_REPORT_HTML_NAME,
    ROOMSCAN_REPORT_JSON_NAME,
    ROOMSCAN_SESSIONS_INDEX_NAME,
    ROOMSCAN_SESSIONS_SUMMARY_CSV_NAME,
)
from logging_utils import get_logger

logger = get_logger(__name__)


def _index_path(index_path: Optional[Path] = None) -> Path:
    return Path(index_path).expanduser() if index_path else ROOMSCAN_OUTPUT_DIR / ROOMSCAN_SESSIONS_INDEX_NAME


def _read_index(index_path: Path) -> List[Dict[str, object]]:
    if not index_path.exists():
        return []
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to read session index {index_path}") from exc


def _write_index(index_path: Path, sessions: List[Dict[str, object]]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        index_path.write_text(json.dumps(sessions, indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write session index {index_path}") from exc


def register_session(
    report: Dict[str, object],
    out_dir: Path,
    index_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Append one finished scan's summary to the session index; return the record.

    ``report`` is exactly the dict roomscan.py/roomscan_live.py already build
    (``roomscan.build_report()``'s output) and write to
    ``<out_dir>/roomscan_report.json``; this does not touch that file, it
    only records a pointer to it plus a denormalized summary (room name,
    timestamp, per-device counts/kwh/cost, totals, recommendations) for fast
    listing/comparison.
    """
    out_dir = Path(out_dir).expanduser()
    scan = report["scan"]
    record = {
        "session_id": out_dir.name,
        "room_name": scan["room_name"],
        "source": scan["source"],
        "timestamp": scan["generated_at"],
        "frames_sampled": scan["frames_sampled"],
        "out_dir": str(out_dir),
        "report_json_path": str(out_dir / ROOMSCAN_REPORT_JSON_NAME),
        "report_html_path": str(out_dir / ROOMSCAN_REPORT_HTML_NAME),
        "devices": [
            {
                "class_name": d["class_name"],
                "display_name": d["display_name"],
                "count": d["count"],
                "kwh_per_year": d["kwh_per_year"],
                "cost_per_year_usd": d["cost_per_year_usd"],
            }
            for d in report["devices"]
        ],
        "totals": report["totals"],
        "recommendations": list(report.get("recommendations", [])),
    }

    path = _index_path(index_path)
    sessions = _read_index(path)
    sessions.append(record)
    _write_index(path, sessions)
    logger.info("Registered session %s in %s", record["session_id"], path)
    return record


def list_sessions(index_path: Optional[Path] = None) -> List[Dict[str, object]]:
    """Return all recorded sessions, most recently registered first."""
    return list(reversed(_read_index(_index_path(index_path))))


def get_session(session_id: str, index_path: Optional[Path] = None) -> Optional[Dict[str, object]]:
    """Return one session's index record by id, or None if not found."""
    for record in _read_index(_index_path(index_path)):
        if record["session_id"] == session_id:
            return record
    return None


def load_full_report(record: Dict[str, object]) -> Dict[str, object]:
    """Re-read a session's full roomscan_report.json (per-device crops/confidences)."""
    path = Path(record["report_json_path"])
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to read session report {path}") from exc


def compare_sessions(record_a: Dict[str, object], record_b: Dict[str, object]) -> Dict[str, object]:
    """Build a side-by-side comparison of two session index records.

    Devices are matched by ``class_name``; a class present in only one
    session shows zeros for the other side. Recommendations are listed
    per-session unchanged -- no delta math is imposed on recommendation
    text since the two sessions' rule firings aren't guaranteed comparable.
    """
    devices_a = {d["class_name"]: d for d in record_a["devices"]}
    devices_b = {d["class_name"]: d for d in record_b["devices"]}
    rows = []
    for class_name in sorted(set(devices_a) | set(devices_b)):
        a = devices_a.get(class_name)
        b = devices_b.get(class_name)
        rows.append(
            {
                "class_name": class_name,
                "display_name": (a or b)["display_name"],
                "count_a": a["count"] if a else 0,
                "count_b": b["count"] if b else 0,
                "kwh_per_year_a": a["kwh_per_year"] if a else 0.0,
                "kwh_per_year_b": b["kwh_per_year"] if b else 0.0,
                "cost_per_year_usd_a": a["cost_per_year_usd"] if a else 0.0,
                "cost_per_year_usd_b": b["cost_per_year_usd"] if b else 0.0,
            }
        )
    totals_a, totals_b = record_a["totals"], record_b["totals"]
    return {
        "session_a": record_a,
        "session_b": record_b,
        "devices": rows,
        "totals_delta": {
            key: totals_b.get(key, 0) - totals_a.get(key, 0)
            for key in ("kwh_per_day", "kwh_per_year", "cost_per_year_usd")
        },
    }


def export_summary_csv(csv_path: Path, index_path: Optional[Path] = None) -> Path:
    """Write a flat CSV summary (one row per session) for all recorded sessions."""
    sessions = list_sessions(index_path)
    csv_path = Path(csv_path).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "session_id",
        "room_name",
        "source",
        "timestamp",
        "frames_sampled",
        "device_count",
        "kwh_per_day",
        "kwh_per_year",
        "cost_per_year_usd",
        "report_html_path",
    ]
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in sessions:
                totals = s["totals"]
                writer.writerow(
                    {
                        "session_id": s["session_id"],
                        "room_name": s["room_name"],
                        "source": s["source"],
                        "timestamp": s["timestamp"],
                        "frames_sampled": s["frames_sampled"],
                        "device_count": totals.get("device_count", len(s["devices"])),
                        "kwh_per_day": totals.get("kwh_per_day", 0.0),
                        "kwh_per_year": totals.get("kwh_per_year", 0.0),
                        "cost_per_year_usd": totals.get("cost_per_year_usd", 0.0),
                        "report_html_path": s["report_html_path"],
                    }
                )
    except OSError as exc:
        raise RuntimeError(f"Failed to write session summary CSV {csv_path}") from exc
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List or export the RoomScan session index.")
    parser.add_argument("--export", metavar="CSV_PATH", help="Write a summary CSV of all sessions and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.export:
        path = export_summary_csv(args.export)
        print(f"Wrote {path}")
        return
    sessions = list_sessions()
    if not sessions:
        print("No RoomScan sessions recorded yet.")
        return
    for s in sessions:
        totals = s["totals"]
        print(
            f"{s['session_id']:<28} {s['room_name']:<18} {s['timestamp']:<20} "
            f"{totals['kwh_per_year']:>8.0f} kWh/yr  ${totals['cost_per_year_usd']:>7.2f}/yr"
        )


if __name__ == "__main__":
    main()
