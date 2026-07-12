"""Unit tests for the RoomScan energy-audit logic (no YOLO/torch needed).

Covers the max-simultaneous instance counting + best-crop slot logic in
energy_detector.ApplianceScanAggregator and the catalog math in
energy_estimator. Runnable with ``python -m pytest tests/`` from the repo
root, same as test_system.py.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from energy_detector import (
    ApplianceScanAggregator,
    Detection,
    DetectionStabilizer,
    _suppress_duplicate_boxes,
    build_rgb_sample_callback,
)
from energy_estimator import (
    DAYS_PER_YEAR,
    estimate_device,
    estimate_discovered_device,
    estimate_room,
    is_appliance,
)
from energy_recommendations import NO_DEVICES_MESSAGE, generate_recommendations


def det(name: str, conf: float, box=(10, 10, 50, 50)) -> Detection:
    return Detection(class_name=name, confidence=conf, box_xyxy=box)


def discovered_device(name: str, watts_active: float, hours_per_day: float, count: int, notes=None) -> dict:
    """Build a priced Gemini-discovered device dict (mirrors
    roomscan.py:merge_discovered_devices' shape) for recommendation tests."""
    device = asdict(estimate_discovered_device(name, watts_active, hours_per_day, count))
    device["confidences"] = []
    device["crops"] = []
    device["notes"] = notes or []
    device["source"] = "gemini_discovered"
    return device


def frame(fill: int) -> np.ndarray:
    return np.full((100, 100, 3), fill, dtype=np.uint8)


class AggregatorTests(unittest.TestCase):
    def test_max_simultaneous_counting(self) -> None:
        agg = ApplianceScanAggregator()
        agg.observe_frame([det("tv", 0.5), det("tv", 0.4)], frame(10))
        agg.observe_frame([det("tv", 0.9)], frame(20))
        agg.observe_frame([det("laptop", 0.6)], frame(30))
        self.assertEqual(agg.counts(), {"tv": 2, "laptop": 1})
        self.assertEqual(agg.frames_observed, 3)

    def test_no_double_count_on_pan_back(self) -> None:
        agg = ApplianceScanAggregator()
        for _ in range(5):  # same single tv seen in 5 separate frames
            agg.observe_frame([det("tv", 0.5)], frame(10))
        self.assertEqual(agg.counts(), {"tv": 1})

    def test_best_crop_replaced_by_higher_confidence(self) -> None:
        agg = ApplianceScanAggregator()
        agg.observe_frame([det("tv", 0.5)], frame(10))
        agg.observe_frame([det("tv", 0.9)], frame(200))
        crops = agg.best_crops()["tv"]
        self.assertEqual(len(crops), 1)
        self.assertTrue((crops[0] == 200).all())
        self.assertEqual(agg.best_confidences()["tv"], [0.9])

    def test_lower_confidence_does_not_replace(self) -> None:
        agg = ApplianceScanAggregator()
        agg.observe_frame([det("tv", 0.9)], frame(200))
        agg.observe_frame([det("tv", 0.3)], frame(10))
        self.assertTrue((agg.best_crops()["tv"][0] == 200).all())

    def test_none_frame_updates_counts_without_crops(self) -> None:
        agg = ApplianceScanAggregator()
        agg.observe_frame([det("tv", 0.5)], None)
        self.assertEqual(agg.counts(), {"tv": 1})
        self.assertEqual(agg.best_crops()["tv"], [])

    def test_empty_scan(self) -> None:
        agg = ApplianceScanAggregator()
        agg.observe_frame([], frame(10))
        self.assertEqual(agg.counts(), {})


class GeminiVerdictTests(unittest.TestCase):
    def test_unverified_slots_includes_freshly_observed_crop(self) -> None:
        agg = ApplianceScanAggregator()
        agg.observe_frame([det("tv", 0.5)], frame(10))
        unverified = agg.unverified_slots()
        self.assertEqual(len(unverified), 1)
        name, idx, confidence, crop = unverified[0]
        self.assertEqual((name, idx, confidence), ("tv", 0, 0.5))
        self.assertTrue((crop == 10).all())

    def test_record_rejected_verdict_excludes_from_counts_and_crops(self) -> None:
        agg = ApplianceScanAggregator()
        agg.observe_frame([det("tv", 0.5)], frame(10))
        ok = agg.record_gemini_verdict("tv", 0, 0.5, accepted=False)
        self.assertTrue(ok)
        self.assertEqual(agg.counts(), {})
        self.assertEqual(agg.best_crops(), {"tv": []})
        self.assertEqual(agg.best_confidences(), {"tv": []})
        self.assertEqual(agg.gemini_rejected_classes(), ["tv"])
        self.assertEqual(agg.unverified_slots(), [])

    def test_record_accepted_verdict_keeps_slot_and_marks_verified(self) -> None:
        agg = ApplianceScanAggregator()
        agg.observe_frame([det("tv", 0.5)], frame(10))
        ok = agg.record_gemini_verdict("tv", 0, 0.5, accepted=True)
        self.assertTrue(ok)
        self.assertEqual(agg.counts(), {"tv": 1})
        self.assertEqual(agg.gemini_rejected_classes(), [])
        self.assertEqual(agg.unverified_slots(), [])

    def test_stale_verdict_after_confidence_moved_on_is_a_no_op(self) -> None:
        agg = ApplianceScanAggregator()
        agg.observe_frame([det("tv", 0.5)], frame(10))
        # Verdict computed against the old (0.5) confidence, but a higher-confidence
        # detection arrives before the verdict is applied.
        agg.observe_frame([det("tv", 0.9)], frame(200))
        ok = agg.record_gemini_verdict("tv", 0, 0.5, accepted=False)
        self.assertFalse(ok)
        self.assertEqual(agg.counts(), {"tv": 1})  # rejection was never applied
        self.assertEqual(agg.gemini_rejected_classes(), [])

    def test_replacement_crop_clears_prior_rejection(self) -> None:
        agg = ApplianceScanAggregator()
        agg.observe_frame([det("tv", 0.5)], frame(10))
        agg.record_gemini_verdict("tv", 0, 0.5, accepted=False)
        self.assertEqual(agg.counts(), {})
        agg.observe_frame([det("tv", 0.9)], frame(200))  # higher confidence replaces the crop
        self.assertEqual(agg.counts(), {"tv": 1})
        self.assertEqual(agg.gemini_rejected_classes(), [])
        unverified = agg.unverified_slots()
        self.assertEqual(len(unverified), 1)
        self.assertEqual(unverified[0][:3], ("tv", 0, 0.9))

    def test_unknown_slot_or_class_returns_false(self) -> None:
        agg = ApplianceScanAggregator()
        agg.observe_frame([det("tv", 0.5)], frame(10))
        self.assertFalse(agg.record_gemini_verdict("laptop", 0, 0.5, accepted=False))
        self.assertFalse(agg.record_gemini_verdict("tv", 5, 0.5, accepted=False))

    def test_accepted_verdict_note_surfaces_in_best_notes(self) -> None:
        agg = ApplianceScanAggregator()
        agg.observe_frame([det("tv", 0.5)], frame(10))
        ok = agg.record_gemini_verdict("tv", 0, 0.5, accepted=True, note="55-inch wall-mounted LED TV")
        self.assertTrue(ok)
        self.assertEqual(agg.best_notes(), {"tv": ["55-inch wall-mounted LED TV"]})

    def test_replacement_crop_clears_prior_note(self) -> None:
        agg = ApplianceScanAggregator()
        agg.observe_frame([det("tv", 0.5)], frame(10))
        agg.record_gemini_verdict("tv", 0, 0.5, accepted=True, note="55-inch wall-mounted LED TV")
        agg.observe_frame([det("tv", 0.9)], frame(200))  # higher confidence replaces the crop
        self.assertEqual(agg.best_notes(), {"tv": [None]})


class DuplicateBoxSuppressionTests(unittest.TestCase):
    def test_heavily_overlapping_same_class_collapsed(self) -> None:
        dets = [det("tv", 0.9, (10, 10, 50, 50)), det("tv", 0.4, (11, 10, 51, 50))]
        kept = _suppress_duplicate_boxes(dets)
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0].confidence, 0.9)

    def test_distinct_boxes_both_kept(self) -> None:
        dets = [det("tv", 0.9, (10, 10, 50, 50)), det("tv", 0.8, (200, 200, 240, 240))]
        kept = _suppress_duplicate_boxes(dets)
        self.assertEqual(len(kept), 2)

    def test_different_classes_never_merged(self) -> None:
        dets = [det("tv", 0.9, (10, 10, 50, 50)), det("laptop", 0.8, (10, 10, 50, 50))]
        kept = _suppress_duplicate_boxes(dets)
        self.assertEqual(len(kept), 2)


class DetectionStabilizerTests(unittest.TestCase):
    def test_single_hit_not_yet_confirmed(self) -> None:
        stab = DetectionStabilizer(min_hits=2)
        result = stab.update([det("tv", 0.8)], 0)
        self.assertEqual(result, [])
        self.assertEqual(stab.stabilized_counts(), {})
        self.assertEqual(stab.instantaneous_counts(), {"tv": 1})

    def test_confirmed_after_min_hits_within_window(self) -> None:
        stab = DetectionStabilizer(window_seconds=3.0, min_hits=2)
        stab.update([det("tv", 0.8)], 0)
        result = stab.update([det("tv", 0.9)], int(0.5e9))
        self.assertEqual(len(result), 1)
        self.assertEqual(stab.stabilized_counts(), {"tv": 1})

    def test_live_detections_reflects_last_update_call(self) -> None:
        stab = DetectionStabilizer(window_seconds=3.0, min_hits=2)
        self.assertEqual(stab.live_detections(), [])
        stab.update([det("tv", 0.8)], 0)
        self.assertEqual(stab.live_detections(), [])  # not yet confirmed
        stab.update([det("tv", 0.9)], int(0.5e9))
        live = stab.live_detections()
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0].class_name, "tv")
        self.assertEqual(live[0].box_xyxy, (10, 10, 50, 50))

    def test_single_missed_frame_does_not_drop_track(self) -> None:
        stab = DetectionStabilizer(window_seconds=3.0, min_hits=2, max_miss_seconds=1.5)
        stab.update([det("tv", 0.8)], 0)
        stab.update([det("tv", 0.9)], int(0.5e9))
        self.assertEqual(stab.stabilized_counts(), {"tv": 1})
        # frame with no detections at all (brief occlusion/motion blur)
        stab.update([], int(0.9e9))
        self.assertEqual(stab.stabilized_counts(), {"tv": 1})
        self.assertEqual(stab.instantaneous_counts(), {})

    def test_track_dropped_after_max_miss_seconds(self) -> None:
        stab = DetectionStabilizer(window_seconds=3.0, min_hits=2, max_miss_seconds=1.0)
        stab.update([det("tv", 0.8)], 0)
        stab.update([det("tv", 0.9)], int(0.5e9))
        self.assertEqual(stab.stabilized_counts(), {"tv": 1})
        stab.update([], int(2.0e9))  # gap of 1.5s > max_miss_seconds
        self.assertEqual(stab.stabilized_counts(), {})

    def test_same_object_across_frames_not_double_counted(self) -> None:
        stab = DetectionStabilizer(window_seconds=3.0, min_hits=2)
        for i in range(5):  # same tv, box drifting slightly, seen every frame
            stab.update([det("tv", 0.8, (10 + i, 10, 50 + i, 50))], int(i * 0.5e9))
        self.assertEqual(stab.stabilized_counts(), {"tv": 1})

    def test_stale_hits_outside_window_do_not_reconfirm_after_gap(self) -> None:
        stab = DetectionStabilizer(window_seconds=1.0, min_hits=2, max_miss_seconds=5.0)
        stab.update([det("tv", 0.8)], 0)
        # gap longer than window but shorter than max_miss: old hit ages out of
        # the window, so a single new hit alone should not re-confirm the track.
        result = stab.update([det("tv", 0.9)], int(4e9))
        self.assertEqual(result, [])
        self.assertEqual(stab.stabilized_counts(), {})


class _FakeSample:
    def __init__(self, ts_ns: int, pixel_format: str = "rgb") -> None:
        self.capture_timestamp_ns = ts_ns
        self.frame = frame(10) if pixel_format == "rgb" else np.full((100, 100), 10, dtype=np.uint8)
        self.pixel_format = pixel_format


class _FakeDetector:
    """Stands in for EnergyDetector.detect() without needing real YOLO weights."""

    def __init__(self, detections) -> None:
        self._detections = detections
        self.calls = 0

    def detect(self, frame_rgb_upright):
        self.calls += 1
        return list(self._detections)


class SampleCallbackTests(unittest.TestCase):
    """Exercises build_rgb_sample_callback's on_rgb closure -- the single wiring
    point between AriaCapture's camera-rgb samples and EnergyDetector.detect() /
    DetectionStabilizer.update() / ApplianceScanAggregator.observe_frame()."""

    def test_throttles_by_sample_hz(self) -> None:
        detector = _FakeDetector([det("tv", 0.8)])
        aggregator = ApplianceScanAggregator()
        on_rgb, _state = build_rgb_sample_callback(detector, aggregator, sample_hz=2.0)

        on_rgb(_FakeSample(0))
        on_rgb(_FakeSample(int(0.1e9)))  # within the 0.5s throttle gap -- dropped
        on_rgb(_FakeSample(int(0.6e9)))  # past the gap -- accepted

        self.assertEqual(detector.calls, 2)
        self.assertEqual(aggregator.frames_observed, 2)

    def test_detections_reach_aggregator_after_stabilizer_min_hits(self) -> None:
        detector = _FakeDetector([det("tv", 0.8)])
        aggregator = ApplianceScanAggregator()
        on_rgb, _state = build_rgb_sample_callback(detector, aggregator, sample_hz=2.0)

        on_rgb(_FakeSample(0))
        self.assertEqual(aggregator.counts(), {})  # min_hits=2: first hit alone isn't enough

        on_rgb(_FakeSample(int(0.6e9)))
        self.assertEqual(aggregator.counts(), {"tv": 1})

    def test_non_rgb_pixel_format_expanded_to_three_channels(self) -> None:
        detector = _FakeDetector([])
        aggregator = ApplianceScanAggregator()
        on_rgb, _state = build_rgb_sample_callback(detector, aggregator, sample_hz=2.0)

        on_rgb(_FakeSample(0, pixel_format="gray"))
        self.assertEqual(aggregator.frames_observed, 1)  # ran without error on a gray sample


class ThreadSafetyTests(unittest.TestCase):
    """Regression tests for the live-mode cross-thread race in energy_detector.

    ApplianceScanAggregator.observe_frame() / DetectionStabilizer.update()
    run on AriaCapture's dispatcher thread while counts()/best_crops()/
    best_confidences()/stabilized_counts()/instantaneous_counts() are read
    from the Qt main thread (or a ticker thread) via
    LiveScanController.snapshot(). Before the lock was added, a reader
    iterating ``.items()`` while the writer inserted a brand-new class key
    via ``setdefault``/``pop`` could raise "dictionary changed size during
    iteration". These tests hammer both sides concurrently across many
    distinct class names to reliably trigger that race if it regresses.
    """

    CLASS_NAMES = [f"class{i}" for i in range(20)]

    def _run_concurrently(self, writer, reader, iterations: int = 500) -> None:
        errors: List[Exception] = []
        stop = threading.Event()

        def _writer_loop() -> None:
            try:
                for i in range(iterations):
                    writer(i)
            except Exception as exc:  # pragma: no cover - only on regression
                errors.append(exc)
            finally:
                stop.set()

        def _reader_loop() -> None:
            try:
                while not stop.is_set():
                    reader()
            except Exception as exc:  # pragma: no cover - only on regression
                errors.append(exc)

        writer_thread = threading.Thread(target=_writer_loop)
        reader_thread = threading.Thread(target=_reader_loop)
        reader_thread.start()
        writer_thread.start()
        writer_thread.join(timeout=30.0)
        reader_thread.join(timeout=30.0)
        self.assertEqual(errors, [])

    def test_aggregator_observe_frame_races_with_reads(self) -> None:
        agg = ApplianceScanAggregator()

        def writer(i: int) -> None:
            name = self.CLASS_NAMES[i % len(self.CLASS_NAMES)]
            agg.observe_frame([det(name, 0.5)], frame(10))

        def reader() -> None:
            agg.counts()
            agg.best_crops()
            agg.best_confidences()

        self._run_concurrently(writer, reader)

    def test_aggregator_gemini_verdicts_race_with_observe_and_reads(self) -> None:
        agg = ApplianceScanAggregator()

        def writer(i: int) -> None:
            name = self.CLASS_NAMES[i % len(self.CLASS_NAMES)]
            agg.observe_frame([det(name, 0.5 + (i % 10) * 0.01)], frame(10))
            for cls_name, idx, confidence, _crop in agg.unverified_slots():
                agg.record_gemini_verdict(cls_name, idx, confidence, accepted=(idx % 2 == 0))

        def reader() -> None:
            agg.counts()
            agg.best_crops()
            agg.best_confidences()
            agg.unverified_slots()
            agg.gemini_rejected_classes()

        self._run_concurrently(writer, reader)

    def test_stabilizer_update_races_with_reads(self) -> None:
        stab = DetectionStabilizer(window_seconds=3.0, min_hits=1)

        def writer(i: int) -> None:
            name = self.CLASS_NAMES[i % len(self.CLASS_NAMES)]
            stab.update([det(name, 0.8)], int(i * 1e6))

        def reader() -> None:
            stab.stabilized_counts()
            stab.instantaneous_counts()

        self._run_concurrently(writer, reader)


class FinalizeScanTests(unittest.TestCase):
    def test_finalize_scan_writes_artifacts_and_registers_session(self) -> None:
        import energy_sessions
        from roomscan import finalize_scan

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_dir = tmp_path / "kitchen_20260101_000000"
            agg = ApplianceScanAggregator()
            agg.observe_frame([det("tv", 0.9)], frame(10))

            # Redirect the session index at the shared default location
            # (register_session()'s only write path) so this test never
            # touches the project's real roomscan_out/roomscan_sessions.json.
            original_output_dir = energy_sessions.ROOMSCAN_OUTPUT_DIR
            energy_sessions.ROOMSCAN_OUTPUT_DIR = tmp_path
            try:
                report = finalize_scan("Kitchen", "test", agg, 1.0, out_dir)
                sessions = energy_sessions.list_sessions()
            finally:
                energy_sessions.ROOMSCAN_OUTPUT_DIR = original_output_dir

            self.assertTrue((out_dir / "roomscan_report.json").exists())
            self.assertTrue((out_dir / "roomscan_report.html").exists())
            self.assertEqual(report["scan"]["room_name"], "Kitchen")
            self.assertTrue(any(s["session_id"] == out_dir.name for s in sessions))

    def test_finalize_scan_defaults_gemini_fields_to_empty_for_batch_scan(self) -> None:
        import energy_sessions
        from roomscan import finalize_scan

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_dir = tmp_path / "living_room_20260101_000000"
            agg = ApplianceScanAggregator()
            agg.observe_frame([det("tv", 0.9)], frame(10))

            original_output_dir = energy_sessions.ROOMSCAN_OUTPUT_DIR
            energy_sessions.ROOMSCAN_OUTPUT_DIR = tmp_path
            try:
                report = finalize_scan("Living room", "test", agg, 1.0, out_dir)
            finally:
                energy_sessions.ROOMSCAN_OUTPUT_DIR = original_output_dir

            self.assertEqual(report["gemini_discovered_devices"], [])
            self.assertEqual(report["scan"]["gemini_rejected_classes"], [])

    def test_finalize_scan_passes_through_gemini_discovered_devices(self) -> None:
        import energy_sessions
        from roomscan import finalize_scan

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_dir = tmp_path / "office_20260101_000000"
            agg = ApplianceScanAggregator()
            agg.observe_frame([det("tv", 0.9)], frame(10))
            discovered = [{"name": "Kettle", "description": "On the counter.", "sightings": 3}]

            original_output_dir = energy_sessions.ROOMSCAN_OUTPUT_DIR
            energy_sessions.ROOMSCAN_OUTPUT_DIR = tmp_path
            try:
                report = finalize_scan(
                    "Office", "test", agg, 1.0, out_dir, gemini_discovered_devices=discovered
                )
                written = json.loads((out_dir / "roomscan_report.json").read_text(encoding="utf-8"))
            finally:
                energy_sessions.ROOMSCAN_OUTPUT_DIR = original_output_dir

            self.assertEqual(report["gemini_discovered_devices"], discovered)
            self.assertEqual(written["gemini_discovered_devices"], discovered)

    def test_build_report_prices_gemini_discovered_devices_into_device_list(self) -> None:
        from roomscan import build_report

        agg = ApplianceScanAggregator()
        agg.observe_frame([det("tv", 0.9)], frame(10))
        discovered = [
            {
                "name": "Ceiling Light",
                "description": "LED, over the couch.",
                "sightings": 2,
                "watts_active": 10.0,
                "hours_per_day": 5.0,
                "count": 3,
            }
        ]
        report = build_report("Kitchen", "test", agg, {}, 1.0, gemini_discovered_devices=discovered)

        priced = [d for d in report["devices"] if d["class_name"] == "ceiling light"]
        self.assertEqual(len(priced), 1)
        device = priced[0]
        self.assertEqual(device["source"], "gemini_discovered")
        self.assertEqual(device["count"], 3)
        self.assertAlmostEqual(device["kwh_per_day"], 10.0 * 5.0 * 3 / 1000.0)
        self.assertIn("LED, over the couch.", device["notes"])
        self.assertIn("Seen 2x this scan", device["notes"])

        # Totals must include both the catalog device (tv) and the priced discovery.
        expected_total_kwh_year = sum(d["kwh_per_year"] for d in report["devices"])
        self.assertAlmostEqual(report["totals"]["kwh_per_year"], expected_total_kwh_year)
        self.assertEqual(report["totals"]["device_count"], sum(d["count"] for d in report["devices"]))

    def test_build_report_gemini_discovered_devices_default_watts_hours_when_missing(self) -> None:
        from roomscan import build_report

        agg = ApplianceScanAggregator()
        agg.observe_frame([det("tv", 0.9)], frame(10))
        # Older-shaped discovery dict with no watts_active/hours_per_day keys.
        discovered = [{"name": "Kettle", "description": "On the counter.", "sightings": 3}]
        report = build_report("Kitchen", "test", agg, {}, 1.0, gemini_discovered_devices=discovered)

        priced = [d for d in report["devices"] if d["class_name"] == "kettle"]
        self.assertEqual(len(priced), 1)
        self.assertAlmostEqual(
            priced[0]["watts_active"], config.GEMINI_DISCOVERY_DEFAULT_WATTS
        )
        self.assertAlmostEqual(
            priced[0]["hours_per_day"], config.GEMINI_DISCOVERY_DEFAULT_HOURS_PER_DAY
        )

    def test_finalize_scan_surfaces_gemini_rejected_classes(self) -> None:
        import energy_sessions
        from roomscan import finalize_scan

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_dir = tmp_path / "hallway_20260101_000000"
            agg = ApplianceScanAggregator()
            agg.observe_frame([det("tv", 0.9)], frame(10))
            agg.record_gemini_verdict("tv", 0, 0.9, accepted=False)

            original_output_dir = energy_sessions.ROOMSCAN_OUTPUT_DIR
            energy_sessions.ROOMSCAN_OUTPUT_DIR = tmp_path
            try:
                report = finalize_scan("Hallway", "test", agg, 1.0, out_dir)
            finally:
                energy_sessions.ROOMSCAN_OUTPUT_DIR = original_output_dir

            self.assertEqual(report["scan"]["gemini_rejected_classes"], ["tv"])
            self.assertNotIn("tv", [d["class_name"] for d in report["devices"]])


class EstimatorTests(unittest.TestCase):
    def test_device_math(self) -> None:
        entry = config.ENERGY_CATALOG["tv"]
        est = estimate_device("tv", 2)
        expected_unit_day = (
            entry["watts_active"] * entry["hours_per_day"]
            + entry["watts_standby"] * (24.0 - entry["hours_per_day"])
        ) / 1000.0
        self.assertAlmostEqual(est.kwh_per_day, expected_unit_day * 2)
        self.assertAlmostEqual(est.kwh_per_year, expected_unit_day * 2 * DAYS_PER_YEAR)
        self.assertAlmostEqual(
            est.cost_per_year_usd, est.kwh_per_year * config.ENERGY_COST_PER_KWH_USD
        )
        self.assertEqual(est.count, 2)

    def test_unknown_class_raises(self) -> None:
        with self.assertRaises(ValueError):
            estimate_device("sofa", 1)

    def test_bad_count_raises(self) -> None:
        with self.assertRaises(ValueError):
            estimate_device("tv", 0)

    def test_room_skips_unknown_and_sorts(self) -> None:
        result = estimate_room({"tv": 1, "sofa": 3, "oven": 1, "clock": 1})
        names = [d["class_name"] for d in result["devices"]]
        self.assertNotIn("sofa", names)
        self.assertEqual(len(names), 3)
        kwh = [d["kwh_per_year"] for d in result["devices"]]
        self.assertEqual(kwh, sorted(kwh, reverse=True))

    def test_room_totals_sum(self) -> None:
        result = estimate_room({"tv": 2, "laptop": 1})
        total = sum(d["kwh_per_year"] for d in result["devices"])
        self.assertAlmostEqual(result["totals"]["kwh_per_year"], total)
        self.assertEqual(result["totals"]["device_count"], 3)

    def test_catalog_classes_are_appliances(self) -> None:
        for name in config.ENERGY_CATALOG:
            self.assertTrue(is_appliance(name))
        self.assertFalse(is_appliance("person"))

    def test_estimate_discovered_device_math(self) -> None:
        est = estimate_discovered_device("Ceiling Light", 10.0, 5.0)
        self.assertEqual(est.class_name, "ceiling light")
        self.assertEqual(est.display_name, "Ceiling Light")
        self.assertEqual(est.count, 1)
        self.assertEqual(est.watts_standby, 0.0)
        self.assertAlmostEqual(est.kwh_per_day, 10.0 * 5.0 / 1000.0)
        self.assertAlmostEqual(est.kwh_per_year, est.kwh_per_day * DAYS_PER_YEAR)
        self.assertAlmostEqual(est.cost_per_year_usd, est.kwh_per_year * config.ENERGY_COST_PER_KWH_USD)

    def test_estimate_discovered_device_clamps_negative_and_out_of_range(self) -> None:
        est = estimate_discovered_device("Weird Device", -5.0, 30.0)
        self.assertEqual(est.watts_active, 0.0)
        self.assertEqual(est.hours_per_day, 24.0)

    def test_estimate_discovered_device_scales_by_count(self) -> None:
        est = estimate_discovered_device("Ceiling Light", 10.0, 5.0, count=3)
        self.assertEqual(est.count, 3)
        self.assertAlmostEqual(est.kwh_per_day, 10.0 * 5.0 * 3 / 1000.0)
        self.assertAlmostEqual(est.kwh_per_year, est.kwh_per_day * DAYS_PER_YEAR)

    def test_estimate_discovered_device_count_floors_at_one(self) -> None:
        est = estimate_discovered_device("Ceiling Light", 10.0, 5.0, count=0)
        self.assertEqual(est.count, 1)


class RecommendationTests(unittest.TestCase):
    def test_no_devices_message(self) -> None:
        result = estimate_room({})
        self.assertEqual(generate_recommendations(result["devices"], result["totals"]), [NO_DEVICES_MESSAGE])

    def test_standby_heavy_device_flagged(self) -> None:
        # cell phone: watts_standby=0.5, hours_per_day=2 -> standby is ~52% of its yearly energy.
        result = estimate_room({"cell phone": 1})
        suggestions = generate_recommendations(result["devices"], result["totals"])
        self.assertTrue(any("standby" in s for s in suggestions))

    def test_high_draw_device_flagged_as_biggest_draw(self) -> None:
        # oven: watts_active=2300 >= HIGH_DRAW_WATTS_THRESHOLD, and it's the top kwh_per_year device.
        result = estimate_room({"oven": 1, "clock": 1})
        suggestions = generate_recommendations(result["devices"], result["totals"])
        self.assertTrue(any("biggest draw" in s for s in suggestions))

    def test_low_cost_fallback_message(self) -> None:
        # clock alone is cheap enough to stay under LOW_IMPACT_COST_USD.
        result = estimate_room({"clock": 1})
        suggestions = generate_recommendations(result["devices"], result["totals"])
        self.assertTrue(any("modest" in s for s in suggestions))

    def test_tv_in_bright_room_needs_context(self) -> None:
        result = estimate_room({"tv": 1})
        no_context = generate_recommendations(result["devices"], result["totals"])
        self.assertFalse(any("brightly lit" in s for s in no_context))
        bright = generate_recommendations(result["devices"], result["totals"], {"avg_brightness": 200.0})
        self.assertTrue(any("brightly lit" in s for s in bright))
        dim = generate_recommendations(result["devices"], result["totals"], {"avg_brightness": 20.0})
        self.assertFalse(any("brightly lit" in s for s in dim))

    def test_multiple_screens_flagged(self) -> None:
        result = estimate_room({"tv": 2})
        suggestions = generate_recommendations(result["devices"], result["totals"])
        self.assertTrue(any("screens" in s for s in suggestions))

    def test_single_screen_not_flagged(self) -> None:
        result = estimate_room({"tv": 1})
        suggestions = generate_recommendations(result["devices"], result["totals"])
        self.assertFalse(any("screens" in s for s in suggestions))

    def test_always_on_fridge_flagged(self) -> None:
        result = estimate_room({"refrigerator": 1})
        suggestions = generate_recommendations(result["devices"], result["totals"])
        self.assertTrue(any("always-on" in s for s in suggestions))

    def test_cooling_inefficiency_flagged_only_when_both_present(self) -> None:
        both = estimate_room({"fan": 1, "air conditioner": 1})
        suggestions = generate_recommendations(both["devices"], both["totals"])
        self.assertTrue(any("Fan and air conditioner" in s for s in suggestions))

        fan_only = estimate_room({"fan": 1})
        suggestions = generate_recommendations(fan_only["devices"], fan_only["totals"])
        self.assertFalse(any("Fan and air conditioner" in s for s in suggestions))

    def test_inefficient_bulb_flagged_for_led_swap(self) -> None:
        device = discovered_device("Ceiling Light", 60.0, 5.0, 1, notes=["Incandescent ceiling light"])
        suggestions = generate_recommendations([device], {"cost_per_year_usd": device["cost_per_year_usd"]})
        self.assertTrue(any("LED" in s and "Ceiling Light" in s for s in suggestions))

    def test_led_discovered_device_not_flagged_for_swap(self) -> None:
        device = discovered_device("Ceiling Light", 9.0, 5.0, 1, notes=["LED ceiling light"])
        suggestions = generate_recommendations([device], {"cost_per_year_usd": device["cost_per_year_usd"]})
        self.assertFalse(any("swapping to" in s for s in suggestions))

    def test_catalog_device_not_flagged_for_bulb_swap(self) -> None:
        # source is absent (not "gemini_discovered"), so keyword text shouldn't matter.
        result = estimate_room({"tv": 1})
        suggestions = generate_recommendations(result["devices"], result["totals"])
        self.assertFalse(any("swapping to" in s for s in suggestions))

    def test_multiple_discovered_lights_flagged(self) -> None:
        device = discovered_device("Floor Lamp", 40.0, 4.0, 4, notes=["Floor lamp"])
        suggestions = generate_recommendations([device], {"cost_per_year_usd": device["cost_per_year_usd"]})
        self.assertTrue(any("smart switch" in s for s in suggestions))

    def test_single_discovered_light_not_flagged_for_multiple(self) -> None:
        device = discovered_device("Floor Lamp", 40.0, 4.0, 1, notes=["Floor lamp"])
        suggestions = generate_recommendations([device], {"cost_per_year_usd": device["cost_per_year_usd"]})
        self.assertFalse(any("smart switch" in s for s in suggestions))

    def test_phantom_load_device_flagged(self) -> None:
        device = discovered_device("Power Strip", 3.0, 24.0, 1, notes=["Power strip under desk"])
        suggestions = generate_recommendations([device], {"cost_per_year_usd": device["cost_per_year_usd"]})
        self.assertTrue(any("phantom load" in s for s in suggestions))

    def test_vent_device_flagged(self) -> None:
        device = discovered_device("Air Vent", 0.0, 0.0, 1, notes=["Wall-mounted air vent"])
        suggestions = generate_recommendations([device], {"cost_per_year_usd": device["cost_per_year_usd"]})
        self.assertTrue(any("blocked by furniture" in s for s in suggestions))


class ReportRecommendationsTests(unittest.TestCase):
    def test_build_report_includes_recommendations(self) -> None:
        from roomscan import build_report

        agg = ApplianceScanAggregator()
        agg.observe_frame([det("refrigerator", 0.7)], frame(10))
        report = build_report("Kitchen", "test", agg, {}, 1.0)
        self.assertIn("recommendations", report)
        self.assertTrue(any("always-on" in s for s in report["recommendations"]))

    def test_render_html_includes_recommended_actions_section(self) -> None:
        import tempfile

        from roomscan import build_report
        from energy_report import render_html

        agg = ApplianceScanAggregator()
        agg.observe_frame([det("refrigerator", 0.7)], frame(10))
        report = build_report("Kitchen", "test", agg, {}, 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            html_path = out_dir / "report.html"
            render_html(report, out_dir, html_path)
            page = html_path.read_text(encoding="utf-8")
        self.assertIn("Recommended Actions", page)

    def test_render_html_omits_section_when_no_recommendations(self) -> None:
        import tempfile

        from energy_report import render_html

        report = {
            "scan": {"room_name": "Empty", "source": "test", "generated_at": "now", "frames_sampled": 0},
            "devices": [],
            "totals": {"device_count": 0, "kwh_per_day": 0.0, "kwh_per_year": 0.0, "cost_per_year_usd": 0.0, "cost_per_kwh_usd": 0.17},
            "recommendations": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            html_path = out_dir / "report.html"
            render_html(report, out_dir, html_path)
            page = html_path.read_text(encoding="utf-8")
        self.assertNotIn("Recommended Actions", page)

    def test_render_html_includes_priced_gemini_discovered_device_card(self) -> None:
        import tempfile

        from roomscan import build_report
        from energy_report import render_html

        agg = ApplianceScanAggregator()
        agg.observe_frame([det("refrigerator", 0.7)], frame(10))
        discovered = [
            {
                "name": "Kettle",
                "description": "On the counter.",
                "sightings": 2,
                "watts_active": 1200.0,
                "hours_per_day": 0.5,
            }
        ]
        report = build_report("Kitchen", "test", agg, {}, 1.0, gemini_discovered_devices=discovered)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            html_path = out_dir / "report.html"
            render_html(report, out_dir, html_path)
            page = html_path.read_text(encoding="utf-8")
        self.assertIn("Kettle", page)
        self.assertIn("Seen 2x this scan", page)
        self.assertIn('<span class="ai-badge">AI</span>', page)

    def test_render_html_includes_category_breakdown(self) -> None:
        import tempfile

        from roomscan import build_report
        from energy_report import render_html

        agg = ApplianceScanAggregator()
        agg.observe_frame([det("refrigerator", 0.7)], frame(10))
        discovered = [{"name": "Ceiling Light", "description": "LED ceiling light.", "watts_active": 9.0, "hours_per_day": 5.0}]
        report = build_report("Kitchen", "test", agg, {}, 1.0, gemini_discovered_devices=discovered)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            html_path = out_dir / "report.html"
            render_html(report, out_dir, html_path)
            page = html_path.read_text(encoding="utf-8")
        self.assertIn("Breakdown by Category", page)
        self.assertIn("Kitchen &amp; Major Appliances", page)
        self.assertIn("Lighting", page)

    def test_render_html_omits_category_breakdown_when_no_devices(self) -> None:
        import tempfile

        from energy_report import render_html

        report = {
            "scan": {"room_name": "Empty", "source": "test", "generated_at": "now", "frames_sampled": 0},
            "devices": [],
            "totals": {"device_count": 0, "kwh_per_day": 0.0, "kwh_per_year": 0.0, "cost_per_year_usd": 0.0, "cost_per_kwh_usd": 0.17},
            "recommendations": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            html_path = out_dir / "report.html"
            render_html(report, out_dir, html_path)
            page = html_path.read_text(encoding="utf-8")
        self.assertNotIn("Breakdown by Category", page)

    def test_render_html_omits_ai_badge_when_no_discovered_devices(self) -> None:
        import tempfile

        from roomscan import build_report
        from energy_report import render_html

        agg = ApplianceScanAggregator()
        agg.observe_frame([det("refrigerator", 0.7)], frame(10))
        report = build_report("Kitchen", "test", agg, {}, 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            html_path = out_dir / "report.html"
            render_html(report, out_dir, html_path)
            page = html_path.read_text(encoding="utf-8")
        self.assertNotIn('<span class="ai-badge">AI</span>', page)

    def test_render_html_includes_gemini_rejected_note_when_present(self) -> None:
        import tempfile

        from roomscan import build_report
        from energy_report import render_html

        agg = ApplianceScanAggregator()
        agg.observe_frame([det("tv", 0.9)], frame(10))
        agg.record_gemini_verdict("tv", 0, 0.9, accepted=False)
        report = build_report("Kitchen", "test", agg, {}, 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            html_path = out_dir / "report.html"
            render_html(report, out_dir, html_path)
            page = html_path.read_text(encoding="utf-8")
        self.assertIn("Gemini vision auto-corrected", page)
        self.assertIn(config.ENERGY_CATALOG["tv"]["display"], page)

    def test_render_html_omits_gemini_rejected_note_when_none_rejected(self) -> None:
        import tempfile

        from roomscan import build_report
        from energy_report import render_html

        agg = ApplianceScanAggregator()
        agg.observe_frame([det("refrigerator", 0.7)], frame(10))
        report = build_report("Kitchen", "test", agg, {}, 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            html_path = out_dir / "report.html"
            render_html(report, out_dir, html_path)
            page = html_path.read_text(encoding="utf-8")
        self.assertNotIn("Gemini vision auto-corrected", page)

    def test_render_html_handles_report_missing_gemini_keys(self) -> None:
        # test_render_html_omits_section_when_no_recommendations's hand-built
        # report dict has no gemini_discovered_devices/gemini_rejected_classes
        # keys at all -- render_html() must not KeyError on it.
        import tempfile

        from energy_report import render_html

        report = {
            "scan": {"room_name": "Empty", "source": "test", "generated_at": "now", "frames_sampled": 0},
            "devices": [],
            "totals": {"device_count": 0, "kwh_per_day": 0.0, "kwh_per_year": 0.0, "cost_per_year_usd": 0.0, "cost_per_kwh_usd": 0.17},
            "recommendations": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            html_path = out_dir / "report.html"
            render_html(report, out_dir, html_path)  # should not raise
            page = html_path.read_text(encoding="utf-8")
        self.assertNotIn('<span class="ai-badge">AI</span>', page)
        self.assertNotIn("Gemini vision auto-corrected", page)


if __name__ == "__main__":
    unittest.main()
