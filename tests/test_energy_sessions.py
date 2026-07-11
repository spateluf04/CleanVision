"""Unit tests for RoomScan's session index (energy_sessions.py).

Pure JSON/CSV file I/O against a temp index path -- no YOLO/torch/Qt needed.
Runnable with ``python -m pytest tests/`` from the repo root, same as
test_energy.py.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from energy_sessions import (
    compare_sessions,
    export_summary_csv,
    get_session,
    list_sessions,
    load_full_report,
    register_session,
)

DEVICE_TV = {
    "class_name": "tv",
    "display_name": "TV",
    "count": 1,
    "watts_active": 100.0,
    "kwh_per_day": 1.0,
    "kwh_per_year": 365.0,
    "cost_per_year_usd": 62.05,
}
TOTALS_A = {
    "device_count": 1,
    "kwh_per_day": 1.0,
    "kwh_per_year": 365.0,
    "cost_per_year_usd": 62.05,
    "cost_per_kwh_usd": 0.17,
}


def _report(room_name: str, generated_at: str, devices, totals) -> dict:
    return {
        "scan": {
            "room_name": room_name,
            "source": "test",
            "generated_at": generated_at,
            "frames_sampled": 10,
            "scan_wall_seconds": 5.0,
        },
        "devices": devices,
        "totals": totals,
        "recommendations": ["Example suggestion"],
    }


class SessionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_dir = Path(self._tmp.name)
        self.index_path = self.tmp_dir / "roomscan_sessions.json"

    def test_register_and_list_round_trip(self) -> None:
        out_dir = self.tmp_dir / "kitchen_20260101_000000"
        report = _report("Kitchen", "2026-01-01 00:00:00", [DEVICE_TV], TOTALS_A)
        record = register_session(report, out_dir, index_path=self.index_path)
        self.assertEqual(record["session_id"], "kitchen_20260101_000000")
        self.assertEqual(record["report_html_path"], str(out_dir / "roomscan_report.html"))

        sessions = list_sessions(index_path=self.index_path)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["room_name"], "Kitchen")
        self.assertEqual(sessions[0]["recommendations"], ["Example suggestion"])

    def test_list_sessions_most_recent_first(self) -> None:
        empty_totals = {"device_count": 0, "kwh_per_day": 0.0, "kwh_per_year": 0.0, "cost_per_year_usd": 0.0, "cost_per_kwh_usd": 0.17}
        for i, room in enumerate(["First", "Second"]):
            out_dir = self.tmp_dir / f"room{i}"
            report = _report(room, f"2026-01-0{i + 1} 00:00:00", [], empty_totals)
            register_session(report, out_dir, index_path=self.index_path)
        sessions = list_sessions(index_path=self.index_path)
        self.assertEqual([s["room_name"] for s in sessions], ["Second", "First"])

    def test_get_session_found_and_missing(self) -> None:
        out_dir = self.tmp_dir / "kitchen_20260101_000000"
        report = _report("Kitchen", "2026-01-01 00:00:00", [DEVICE_TV], TOTALS_A)
        register_session(report, out_dir, index_path=self.index_path)
        self.assertIsNotNone(get_session("kitchen_20260101_000000", index_path=self.index_path))
        self.assertIsNone(get_session("missing", index_path=self.index_path))

    def test_list_sessions_empty_when_no_index_file(self) -> None:
        self.assertEqual(list_sessions(index_path=self.index_path), [])

    def test_load_full_report_reads_json_file(self) -> None:
        out_dir = self.tmp_dir / "kitchen_20260101_000000"
        out_dir.mkdir()
        report = _report("Kitchen", "2026-01-01 00:00:00", [DEVICE_TV], TOTALS_A)
        (out_dir / "roomscan_report.json").write_text(json.dumps(report), encoding="utf-8")
        record = register_session(report, out_dir, index_path=self.index_path)
        loaded = load_full_report(record)
        self.assertEqual(loaded["scan"]["room_name"], "Kitchen")

    def test_compare_sessions_matches_by_class_and_computes_delta(self) -> None:
        out_a = self.tmp_dir / "a"
        out_b = self.tmp_dir / "b"
        report_a = _report("A", "2026-01-01 00:00:00", [DEVICE_TV], TOTALS_A)
        laptop = {
            "class_name": "laptop",
            "display_name": "Laptop",
            "count": 2,
            "watts_active": 60.0,
            "kwh_per_day": 0.5,
            "kwh_per_year": 182.5,
            "cost_per_year_usd": 31.0,
        }
        totals_b = {
            "device_count": 2,
            "kwh_per_day": 1.5,
            "kwh_per_year": 547.5,
            "cost_per_year_usd": 93.05,
            "cost_per_kwh_usd": 0.17,
        }
        report_b = _report("B", "2026-01-02 00:00:00", [DEVICE_TV, laptop], totals_b)
        record_a = register_session(report_a, out_a, index_path=self.index_path)
        record_b = register_session(report_b, out_b, index_path=self.index_path)

        comparison = compare_sessions(record_a, record_b)
        class_names = {d["class_name"] for d in comparison["devices"]}
        self.assertEqual(class_names, {"tv", "laptop"})
        laptop_row = next(d for d in comparison["devices"] if d["class_name"] == "laptop")
        self.assertEqual(laptop_row["count_a"], 0)
        self.assertEqual(laptop_row["count_b"], 2)
        tv_row = next(d for d in comparison["devices"] if d["class_name"] == "tv")
        self.assertEqual(tv_row["count_a"], 1)
        self.assertEqual(tv_row["count_b"], 1)
        self.assertAlmostEqual(comparison["totals_delta"]["kwh_per_year"], 547.5 - 365.0)
        self.assertAlmostEqual(comparison["totals_delta"]["cost_per_year_usd"], 93.05 - 62.05)

    def test_export_summary_csv_writes_one_row_per_session(self) -> None:
        out_dir = self.tmp_dir / "kitchen_20260101_000000"
        report = _report("Kitchen", "2026-01-01 00:00:00", [DEVICE_TV], TOTALS_A)
        register_session(report, out_dir, index_path=self.index_path)
        csv_path = self.tmp_dir / "summary.csv"
        export_summary_csv(csv_path, index_path=self.index_path)
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["room_name"], "Kitchen")
        self.assertEqual(rows[0]["session_id"], "kitchen_20260101_000000")
        self.assertAlmostEqual(float(rows[0]["kwh_per_year"]), 365.0)


class SessionOutDirTests(unittest.TestCase):
    """roomscan_live.session_out_dir mints the <room-slug>_<timestamp> folder
    name energy_sessions.py's session_id contract relies on -- covering it
    directly (rather than only indirectly via the dashboard) protects the
    naming convention if a future non-Qt caller starts a live session."""

    def test_slugifies_room_name_and_appends_timestamp(self) -> None:
        import re

        from roomscan_live import session_out_dir

        out_dir = session_out_dir("/tmp/roomscan_out", "Living Room!")
        self.assertRegex(out_dir.name, r"^living_room_\d{8}_\d{6}$")
        self.assertEqual(out_dir.parent, Path("/tmp/roomscan_out"))

    def test_blank_room_name_falls_back_to_room(self) -> None:
        from roomscan_live import session_out_dir

        out_dir = session_out_dir("/tmp/roomscan_out", "   ")
        self.assertTrue(out_dir.name.startswith("room_"))


if __name__ == "__main__":
    unittest.main()
