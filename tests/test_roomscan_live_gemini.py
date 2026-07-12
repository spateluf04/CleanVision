"""Unit tests for roomscan_live.py's Gemini live verification/discovery wiring.

Mirrors tests/test_roomscan_dashboard.py's FakeLiveScanController pattern:
only start()/stop()/latest_frame() (the AriaCapture-touching bits) are
overridden, so __init__/snapshot()/finish() and the new
_run_gemini_pass_once()/_known_display_names()/_should_run_gemini_pass()
methods all run for real against fed-in fake detections -- no Aria hardware,
VRS file, or YOLO model required. run_live_scan_pass() itself is mocked at
its roomscan_live import site (roomscan_live.run_live_scan_pass), consistent
with this codebase's direct-import mocking convention.
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from energy_detector import Detection
from roomscan_live import LiveScanController


def _det(name: str, conf: float) -> Detection:
    return Detection(class_name=name, confidence=conf, box_xyxy=(10, 10, 50, 50))


def _frame(fill: int) -> np.ndarray:
    return np.full((100, 100, 3), fill, dtype=np.uint8)


class FakeLiveScanController(LiveScanController):
    """Fakes only the AriaCapture-touching bits -- start()/stop()/latest_frame()."""

    def start(self) -> None:
        if self._running:
            raise RuntimeError("LiveScanController already started.")
        self._wall_start = time.monotonic()
        self._running = True

    def stop(self) -> None:
        self._gemini_pass_stop_event.set()
        self._running = False

    def latest_frame(self) -> Optional[np.ndarray]:
        return _frame(42)

    def feed(self, class_name: str, confidence: float) -> None:
        self._aggregator.observe_frame([_det(class_name, confidence)], _frame(10))


def _make_controller(disable_detection: bool = True) -> FakeLiveScanController:
    return FakeLiveScanController(room_name="Kitchen", disable_detection=disable_detection)


class RunGeminiPassOnceTests(unittest.TestCase):
    def test_accepted_verdict_keeps_slot_counted(self) -> None:
        controller = _make_controller()
        controller.feed("tv", 0.5)
        with mock.patch(
            "roomscan_live.run_live_scan_pass",
            return_value={"verifications": [("tv", 0, 0.5, True, "55-inch LED TV", None)], "discovered": []},
        ) as mocked:
            controller._run_gemini_pass_once()
        mocked.assert_called_once()
        self.assertEqual(controller._aggregator.counts().get("tv"), 1)
        self.assertEqual(controller._aggregator.best_notes().get("tv"), ["55-inch LED TV"])

    def test_rejected_verdict_removes_slot_from_counts(self) -> None:
        controller = _make_controller()
        controller.feed("tv", 0.5)
        with mock.patch(
            "roomscan_live.run_live_scan_pass",
            return_value={"verifications": [("tv", 0, 0.5, False, None, None)], "discovered": []},
        ):
            controller._run_gemini_pass_once()
        self.assertNotIn("tv", controller._aggregator.counts())
        self.assertEqual(controller._aggregator.gemini_rejected_classes(), ["tv"])

    def test_refined_class_moves_slot_to_new_class_in_counts(self) -> None:
        controller = _make_controller()
        controller.feed("tv", 0.5)
        with mock.patch(
            "roomscan_live.run_live_scan_pass",
            return_value={
                "verifications": [("tv", 0, 0.5, True, "27-inch desk monitor", "monitor")],
                "discovered": [],
            },
        ):
            controller._run_gemini_pass_once()
        counts = controller._aggregator.counts()
        self.assertNotIn("tv", counts)
        self.assertEqual(counts.get("monitor"), 1)
        self.assertEqual(controller._aggregator.best_notes().get("monitor"), ["27-inch desk monitor"])

    def test_discovered_devices_accumulate_and_sightings_increment(self) -> None:
        controller = _make_controller()
        with mock.patch(
            "roomscan_live.run_live_scan_pass",
            return_value={
                "verifications": [],
                "discovered": [{"name": "Kettle", "description": "On the counter."}],
            },
        ):
            controller._run_gemini_pass_once()
        discovered = controller._gemini_discovered_list()
        self.assertEqual(len(discovered), 1)
        self.assertEqual(discovered[0]["name"], "Kettle")
        self.assertEqual(discovered[0]["sightings"], 1)

        with mock.patch(
            "roomscan_live.run_live_scan_pass",
            return_value={
                "verifications": [],
                "discovered": [{"name": "kettle", "description": "Seen again."}],
            },
        ):
            controller._run_gemini_pass_once()
        discovered = controller._gemini_discovered_list()
        self.assertEqual(len(discovered), 1)
        self.assertEqual(discovered[0]["sightings"], 2)

    def test_discovered_device_count_uses_max_simultaneous_rule(self) -> None:
        controller = _make_controller()
        with mock.patch(
            "roomscan_live.run_live_scan_pass",
            return_value={
                "verifications": [],
                "discovered": [{"name": "Ceiling Light", "description": "", "count": 2}],
            },
        ):
            controller._run_gemini_pass_once()
        self.assertEqual(controller._gemini_discovered_list()[0]["count"], 2)

        # A later pass reporting fewer instances (e.g. panned away from some
        # lights) must never lower the count -- only a higher count raises it.
        with mock.patch(
            "roomscan_live.run_live_scan_pass",
            return_value={
                "verifications": [],
                "discovered": [{"name": "Ceiling Light", "description": "", "count": 1}],
            },
        ):
            controller._run_gemini_pass_once()
        self.assertEqual(controller._gemini_discovered_list()[0]["count"], 2)

        with mock.patch(
            "roomscan_live.run_live_scan_pass",
            return_value={
                "verifications": [],
                "discovered": [{"name": "Ceiling Light", "description": "", "count": 5}],
            },
        ):
            controller._run_gemini_pass_once()
        self.assertEqual(controller._gemini_discovered_list()[0]["count"], 5)

    def test_busy_lock_skips_tick_without_calling_run_live_scan_pass(self) -> None:
        controller = _make_controller()
        controller.feed("tv", 0.5)
        controller._gemini_pass_lock.acquire()
        try:
            with mock.patch("roomscan_live.run_live_scan_pass") as mocked:
                controller._run_gemini_pass_once()
            mocked.assert_not_called()
        finally:
            controller._gemini_pass_lock.release()

    def test_unexpected_exception_is_swallowed_and_lock_released(self) -> None:
        controller = _make_controller()
        controller.feed("tv", 0.5)
        with mock.patch("roomscan_live.run_live_scan_pass", side_effect=RuntimeError("boom")):
            controller._run_gemini_pass_once()  # should not raise
        # Lock must be released even after an unexpected failure.
        self.assertTrue(controller._gemini_pass_lock.acquire(blocking=False))
        controller._gemini_pass_lock.release()


class KnownDisplayNamesTests(unittest.TestCase):
    def test_includes_catalog_display_names(self) -> None:
        controller = _make_controller()
        names = controller._known_display_names()
        expected_display_names = {entry["display"] for entry in config.ENERGY_CATALOG.values()}
        self.assertTrue(expected_display_names.issubset(set(names)))

    def test_includes_previously_discovered_names(self) -> None:
        controller = _make_controller()
        with mock.patch(
            "roomscan_live.run_live_scan_pass",
            return_value={"verifications": [], "discovered": [{"name": "Kettle", "description": ""}]},
        ):
            controller._run_gemini_pass_once()
        self.assertIn("Kettle", controller._known_display_names())


class ShouldRunGeminiPassTests(unittest.TestCase):
    def test_disabled_when_detection_disabled_even_with_key(self) -> None:
        controller = _make_controller(disable_detection=True)
        with mock.patch("roomscan_live.ai_features_enabled", return_value=True):
            self.assertFalse(controller._should_run_gemini_pass())

    def test_disabled_when_no_api_key(self) -> None:
        controller = _make_controller(disable_detection=False)
        with mock.patch("roomscan_live.ai_features_enabled", return_value=False):
            self.assertFalse(controller._should_run_gemini_pass())

    def test_enabled_when_detection_on_and_key_set(self) -> None:
        controller = _make_controller(disable_detection=False)
        with mock.patch("roomscan_live.ai_features_enabled", return_value=True):
            self.assertTrue(controller._should_run_gemini_pass())


class LatestDetectionsTests(unittest.TestCase):
    def test_empty_before_sample_loop_starts(self) -> None:
        controller = _make_controller(disable_detection=False)
        self.assertEqual(controller.latest_detections(), [])

    def test_empty_in_debug_camera_only_mode(self) -> None:
        controller = _make_controller(disable_detection=True)
        self.assertEqual(controller.latest_detections(), [])

    def test_delegates_to_stabilizer_once_sample_loop_is_wired(self) -> None:
        # FakeLiveScanController's start() override never wires the real
        # camera-rgb sample callback (no Aria hardware here), so this pokes
        # _sample_state directly to simulate what build_rgb_sample_callback()
        # would have set up, confirming latest_detections() delegates to the
        # stabilizer rather than duplicating its bookkeeping.
        controller = _make_controller(disable_detection=False)
        from energy_detector import DetectionStabilizer

        stabilizer = DetectionStabilizer(min_hits=1)
        stabilizer.update([_det("tv", 0.9)], 0)
        controller._sample_state = {"stabilizer": stabilizer}
        detections = controller.latest_detections()
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "tv")


class SnapshotGeminiFieldsTests(unittest.TestCase):
    def test_snapshot_includes_gemini_fields(self) -> None:
        controller = _make_controller(disable_detection=False)
        controller.feed("tv", 0.5)
        with mock.patch(
            "roomscan_live.run_live_scan_pass",
            return_value={
                "verifications": [("tv", 0, 0.5, False, None, None)],
                "discovered": [{"name": "Kettle", "description": ""}],
            },
        ):
            controller._run_gemini_pass_once()
        with mock.patch("roomscan_live.ai_features_enabled", return_value=True):
            state = controller.snapshot()
        self.assertEqual(state["gemini_rejected_classes"], ["tv"])
        self.assertEqual([d["name"] for d in state["gemini_discovered_devices"]], ["Kettle"])
        self.assertTrue(state["gemini_verification_enabled"])
        self.assertFalse(state["gemini_pass_active"])
        self.assertIsNotNone(state["gemini_last_pass_ts"])

    def test_snapshot_prices_discovered_devices_into_device_list(self) -> None:
        controller = _make_controller(disable_detection=False)
        controller.feed("tv", 0.5)
        with mock.patch(
            "roomscan_live.run_live_scan_pass",
            return_value={
                "verifications": [],
                "discovered": [
                    {
                        "name": "Ceiling Light",
                        "description": "LED over the table.",
                        "watts_active": 10.0,
                        "hours_per_day": 5.0,
                    }
                ],
            },
        ):
            controller._run_gemini_pass_once()
        state = controller.snapshot()
        priced = [d for d in state["devices"] if d["class_name"] == "ceiling light"]
        self.assertEqual(len(priced), 1)
        self.assertEqual(priced[0]["source"], "gemini_discovered")
        self.assertAlmostEqual(priced[0]["kwh_per_day"], 10.0 * 5.0 / 1000.0)
        self.assertIn("Ceiling Light", [d["name"] for d in state["gemini_discovered_devices"]])

    def test_snapshot_defaults_are_empty_before_any_pass(self) -> None:
        controller = _make_controller()
        state = controller.snapshot()
        self.assertEqual(state["gemini_rejected_classes"], [])
        self.assertEqual(state["gemini_discovered_devices"], [])
        self.assertFalse(state["gemini_verification_enabled"])
        self.assertFalse(state["gemini_pass_active"])
        self.assertIsNone(state["gemini_last_pass_ts"])


class GeminiPassThreadLifecycleTests(unittest.TestCase):
    def _controller_with_noop_capture(self, disable_detection: bool) -> LiveScanController:
        # disable_detection=True skips constructing EnergyDetector (no YOLO
        # weights needed); self._capture.start()/stop() are stubbed so the
        # real LiveScanController.start()/stop() run without touching the
        # Aria Client SDK, letting this test exercise the actual thread
        # spin-up/teardown logic instead of a fake override of start()/stop().
        controller = LiveScanController(room_name="Kitchen", disable_detection=disable_detection)
        controller._capture.start = mock.Mock()
        controller._capture.stop = mock.Mock()
        return controller

    def test_start_spins_up_thread_when_should_run(self) -> None:
        controller = self._controller_with_noop_capture(disable_detection=True)
        with mock.patch.object(controller, "_should_run_gemini_pass", return_value=True):
            controller.start()
        try:
            self.assertIsNotNone(controller._gemini_pass_thread)
            self.assertTrue(controller._gemini_pass_thread.is_alive())
        finally:
            controller.stop()
            controller._gemini_pass_thread.join(timeout=2.0)

    def test_start_does_not_spin_up_thread_when_should_not_run(self) -> None:
        controller = self._controller_with_noop_capture(disable_detection=True)
        with mock.patch.object(controller, "_should_run_gemini_pass", return_value=False):
            controller.start()
        self.assertIsNone(controller._gemini_pass_thread)
        controller.stop()

    def test_stop_signals_thread_to_exit(self) -> None:
        controller = self._controller_with_noop_capture(disable_detection=True)
        with mock.patch.object(controller, "_should_run_gemini_pass", return_value=True):
            with mock.patch("roomscan_live.GEMINI_LIVE_PASS_INTERVAL_SECONDS", 0.01):
                controller.start()
                thread = controller._gemini_pass_thread
                controller.stop()
                thread.join(timeout=2.0)
                self.assertFalse(thread.is_alive())

    def test_stop_is_safe_to_call_before_start(self) -> None:
        controller = FakeLiveScanController(room_name="Kitchen", disable_detection=True)
        controller.stop()  # should not raise
        self.assertFalse(controller._running)


if __name__ == "__main__":
    unittest.main()
