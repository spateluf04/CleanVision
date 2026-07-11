"""Hardware-free, full-stack end-to-end validation of the live RoomScan
dashboard flow: Start Scan -> live camera feed -> frames analyzed -> detector
-> devices/totals update -> Stop Scan -> Save Report -> Past Scans.

Unlike tests/test_roomscan_dashboard.py (which drives the dashboard against a
FakeLiveScanController test double that fakes start/stop/latest_frame and
never touches AriaCapture at all), this exercises the REAL
roomscan_live.LiveScanController and the REAL aria_capture.AriaCapture
plumbing -- the fan-out dispatcher thread, the camera-rgb sample-and-throttle
callback, the DetectionStabilizer, and the actual production wiring between
LiveScanController and RoomScanDashboard. Only two boundary seams are faked:

  * AriaCapture._start_live / _stop_live -- so no Aria Client SDK or hardware
    is required. A background thread stands in for the SDK's observer,
    pushing synthetic frames straight into AriaCapture._emit_image() (the
    exact same call the live _LiveObserver.on_image_received() makes).
  * EnergyDetector -- replaced with a lightweight fake that returns a
    deterministic detection sequence instead of running real YOLO, so this
    test needs neither ultralytics/torch nor a real appliance in frame.

Everything downstream of those two seams (ApplianceScanAggregator,
DetectionStabilizer, estimate_room, finalize_scan, energy_sessions,
RoomScanDashboard's Qt widgets/timers/state machine) is the real production
code path.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5.QtWidgets import QApplication

import energy_sessions
import roomscan_dashboard as dashboard_module
import roomscan_live as roomscan_live_module
from aria_capture import AriaCapture
from energy_detector import Detection

_APP = QApplication.instance() or QApplication(sys.argv)


def _synthetic_frame() -> np.ndarray:
    # RAW-orientation placeholder frame (content is irrelevant -- FakeDetector
    # never looks at pixels, only a call counter).
    return np.full((640, 480, 3), 120, dtype=np.uint8)


class FakeEnergyDetector:
    """Stands in for EnergyDetector: no YOLO/ultralytics, deterministic output.

    Returns no detections for the first call (simulating an empty room at
    scan start), then a stable "tv" box from the second call onward so
    DetectionStabilizer (min_hits=2) confirms it after two accepted samples.
    """

    def __init__(self, confidence: float = 0.0) -> None:
        self.call_count = 0

    def detect(self, frame_rgb_upright: np.ndarray):
        self.call_count += 1
        if self.call_count >= 2:
            return [Detection(class_name="tv", confidence=0.87, box_xyxy=(10, 10, 200, 200))]
        return []


def _fake_start_live(self) -> None:
    self._started_streaming_internally = False


def _fake_stop_live(self) -> None:
    pass


class LiveRoomScanEndToEndTests(unittest.TestCase):
    """Drives the real Start -> Stop -> Save -> Past Scans chain."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_dir = Path(self._tmp.name)

        original_output_dir = energy_sessions.ROOMSCAN_OUTPUT_DIR
        energy_sessions.ROOMSCAN_OUTPUT_DIR = self.tmp_dir
        self.addCleanup(setattr, energy_sessions, "ROOMSCAN_OUTPUT_DIR", original_output_dir)

        # Fake only the Aria SDK boundary -- real AriaCapture fan-out/dispatch.
        start_patcher = mock.patch.object(AriaCapture, "_start_live", _fake_start_live)
        stop_patcher = mock.patch.object(AriaCapture, "_stop_live", _fake_stop_live)
        start_patcher.start()
        stop_patcher.start()
        self.addCleanup(start_patcher.stop)
        self.addCleanup(stop_patcher.stop)

        # Fake only the YOLO boundary -- real aggregator/stabilizer/estimator.
        detector_patcher = mock.patch.object(roomscan_live_module, "EnergyDetector", FakeEnergyDetector)
        detector_patcher.start()
        self.addCleanup(detector_patcher.stop)

        summary_patcher = mock.patch.object(dashboard_module.RoomEfficiencySummaryDialog, "exec_", return_value=None)
        summary_patcher.start()
        self.addCleanup(summary_patcher.stop)

        self.window = self._build_window()

    def _build_window(self, debug_camera_only: bool = False) -> dashboard_module.RoomScanDashboard:
        args = argparse.Namespace(
            device_ip=None,
            start_streaming=False,
            interface="wifi",
            profile="profile18",
            persistent_certs=False,
            local_certs_dir=None,
            out=str(self.tmp_dir),
            debug_camera_only=debug_camera_only,
        )
        window = dashboard_module.RoomScanDashboard(args)
        self.addCleanup(window.close)
        return window

    def _emit_frames(self, window, count: int, device_gap_s: float = 0.6) -> None:
        """Push synthetic camera-rgb frames into the real AriaCapture, standing
        in for the live SDK observer. device_gap_s > the 0.5s (2 Hz) sample
        throttle in build_rgb_sample_callback so each one is accepted."""
        capture = window._controller._capture
        ts_ns = int(time.time() * 1e9)
        for _ in range(count):
            ts_ns += int(device_gap_s * 1e9)
            capture._emit_image("camera-rgb", _synthetic_frame(), "rgb", ts_ns)
            # Give the real dispatcher thread a moment to drain the single-slot
            # buffer and invoke the on_rgb callback before the next overwrite.
            time.sleep(0.05)

    def test_full_live_flow_start_to_past_scans(self) -> None:
        # 1. Start Scan
        self.window._room_name_edit.setText("Living Room")
        self.window._on_start_clicked()
        self.assertIsInstance(self.window._controller, roomscan_live_module.LiveScanController)
        self.assertTrue(self.window._controller.running)
        self.assertFalse(self.window._start_button.isEnabled())
        self.assertTrue(self.window._stop_button.isEnabled())

        # Before any frame arrives: explicit waiting state, never a silent/blank box.
        self.assertEqual(self.window._camera_label.text(), dashboard_module.WAITING_FOR_FRAME_MESSAGE)

        # 2. Live camera feed appears
        self._emit_frames(self.window, 1)
        self.window._poll_frame()
        self.assertFalse(self.window._camera_label.pixmap().isNull())
        self.assertTrue(self.window._received_first_frame)

        # 3 + 4. Frames-analyzed counter increases and the (fake) detector ran.
        self._emit_frames(self.window, 4)
        self.window._poll_frame()
        snapshot_mid = self.window._controller.snapshot()
        self.assertGreaterEqual(snapshot_mid["frames_sampled"], 2)
        self.assertGreaterEqual(self.window._controller._detector.call_count, 2)

        # 5. Devices appear once the stabilizer confirms (min_hits=2).
        self.window._poll_stats()
        self.assertEqual(self.window._device_table.rowCount(), 1)
        self.assertEqual(self.window._device_table.item(0, 0).text(), "Television")
        self.assertIn("Live feed active", self.window._status_label.text())

        # 6. Totals update off the "--" placeholder.
        self.assertNotEqual(self.window._watts_value.text(), "--")
        self.assertNotEqual(self.window._cost_value.text(), "--")
        self.assertNotEqual(self.window._kwh_day_value.text(), "--")
        self.assertNotEqual(self.window._kwh_year_value.text(), "--")

        # 7. Stop Scan
        self.window._on_stop_clicked()
        self.assertFalse(self.window._controller.running)
        self.assertFalse(self.window._stop_button.isEnabled())
        self.assertTrue(self.window._save_button.isEnabled())

        # 8. Save Report writes JSON/HTML to disk
        self.window._on_save_clicked()
        self.assertIsNone(self.window._controller)
        session_item = self.window._sessions_list.item(0)
        self.assertIn("Living Room", session_item.text())
        from PyQt5.QtCore import Qt as _Qt

        session_id = session_item.data(_Qt.UserRole)
        record = energy_sessions.get_session(session_id)
        self.assertIsNotNone(record)
        report_json_path = Path(record["report_json_path"])
        report_html_path = Path(record["report_html_path"])
        self.assertTrue(report_json_path.exists())
        self.assertTrue(report_html_path.exists())

        import json

        report = json.loads(report_json_path.read_text())
        self.assertEqual(report["scan"]["room_name"], "Living Room")
        device_names = {d["display_name"] for d in report["devices"]}
        self.assertIn("Television", device_names)

        # 9. Session appears in Past Scans
        self.assertEqual(self.window._sessions_list.count(), 1)

    def test_debug_camera_only_shows_frames_with_detection_disabled(self) -> None:
        """--debug-camera-only: camera feed still renders, but the detector is
        never even constructed, so devices/totals never populate."""
        window = self._build_window(debug_camera_only=True)
        self.assertIn("[DEBUG: camera-only, detector disabled]", window.windowTitle())

        window._room_name_edit.setText("Garage")
        window._on_start_clicked()
        self.assertIsInstance(window._controller, roomscan_live_module.LiveScanController)
        self.assertTrue(window._controller.disable_detection)
        self.assertIsNone(window._controller._detector)

        # Camera feed renders identically to normal mode -- proves frame
        # delivery/rendering never depended on detection being enabled.
        self._emit_frames(window, 1)
        window._poll_frame()
        self.assertFalse(window._camera_label.pixmap().isNull())
        self.assertTrue(window._received_first_frame)

        # A few more frames to make sure nothing downstream secretly runs
        # detection despite the flag.
        self._emit_frames(window, 4)
        window._poll_frame()
        window._poll_stats()

        self.assertIn(
            dashboard_module._STATUS_TEXT[dashboard_module.STATUS_DEBUG_CAMERA_ONLY],
            window._status_label.text(),
        )
        self.assertEqual(window._device_table.rowCount(), 1)
        self.assertEqual(
            window._device_table.item(0, 0).text(), dashboard_module.NO_DEVICES_TABLE_MESSAGE
        )
        self.assertEqual(window._watts_value.text(), "0 W")

        # Stop/Save still work and produce a zero-device report.
        window._on_stop_clicked()
        window._on_save_clicked()
        self.assertIsNone(window._controller)
        self.assertEqual(window._sessions_list.count(), 1)


if __name__ == "__main__":
    unittest.main()
