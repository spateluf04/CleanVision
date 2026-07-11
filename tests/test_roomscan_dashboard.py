"""Headless smoke tests for roomscan_dashboard.py's Start -> Stop -> Save ->
session-reload lifecycle.

Runs the real Qt widget tree offscreen (QT_QPA_PLATFORM=offscreen, set below
before PyQt5 is imported) against a FakeLiveScanController double that fakes
only the AriaCapture-touching bits (start/stop/latest_frame). snapshot() and
finish() are inherited unchanged from the real LiveScanController, so this
exercises the actual aggregation -> estimate_room -> finalize_scan ->
energy_sessions.register_session code path with no Aria hardware, VRS file,
or YOLO model required.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

import energy_sessions
import roomscan_dashboard as dashboard_module
from energy_detector import Detection
from roomscan_live import LiveScanController

_APP = QApplication.instance() or QApplication(sys.argv)


def _det(name: str, conf: float) -> Detection:
    return Detection(class_name=name, confidence=conf, box_xyxy=(10, 10, 50, 50))


def _frame(fill: int) -> np.ndarray:
    return np.full((100, 100, 3), fill, dtype=np.uint8)


class FakeLiveScanController(LiveScanController):
    """Fakes only the AriaCapture-touching bits of LiveScanController --
    start()/stop()/latest_frame() -- so snapshot() and finish() run the real
    aggregation/estimation/finalize_scan code paths against fed-in fake
    detections instead of a live Aria stream."""

    def start(self) -> None:
        if self._running:
            raise RuntimeError("LiveScanController already started.")
        self._wall_start = time.monotonic()
        self._running = True

    def stop(self) -> None:
        self._running = False

    def latest_frame(self) -> Optional[np.ndarray]:
        return _frame(42)

    def feed(self, class_name: str, confidence: float) -> None:
        self._aggregator.observe_frame([_det(class_name, confidence)], _frame(10))

    def reject_class(self, class_name: str, confidence: float) -> None:
        """Feed a detection, then simulate a Gemini live-verification
        rejection for it (as _run_gemini_pass_once() would apply)."""
        self.feed(class_name, confidence)
        self._aggregator.record_gemini_verdict(class_name, 0, confidence, accepted=False)

    def feed_gemini_discovery(self, name: str, description: str = "") -> None:
        """Seed a Gemini-discovered non-catalog device directly, as
        _run_gemini_pass_once() would after a real live pass."""
        self._gemini_discovered[name.lower()] = {"name": name, "description": description, "sightings": 1}


class DashboardLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_dir = Path(self._tmp.name)

        original_output_dir = energy_sessions.ROOMSCAN_OUTPUT_DIR
        energy_sessions.ROOMSCAN_OUTPUT_DIR = self.tmp_dir
        self.addCleanup(setattr, energy_sessions, "ROOMSCAN_OUTPUT_DIR", original_output_dir)

        original_controller_cls = dashboard_module.LiveScanController
        dashboard_module.LiveScanController = FakeLiveScanController
        self.addCleanup(setattr, dashboard_module, "LiveScanController", original_controller_cls)

        # Stop Scan now pops a real (blocking) modal RoomEfficiencySummaryDialog
        # -- no one to click it offscreen, so every test gets it stubbed out.
        summary_patcher = mock.patch.object(dashboard_module.RoomEfficiencySummaryDialog, "exec_", return_value=None)
        summary_patcher.start()
        self.addCleanup(summary_patcher.stop)

        args = argparse.Namespace(
            device_ip=None,
            start_streaming=False,
            interface="wifi",
            profile="profile18",
            persistent_certs=False,
            local_certs_dir=None,
            out=str(self.tmp_dir),
        )
        self.window = dashboard_module.RoomScanDashboard(args)
        self.addCleanup(self.window.close)

    def test_start_updates_status_and_enables_stop_save(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()

        self.assertIsInstance(self.window._controller, FakeLiveScanController)
        self.assertTrue(self.window._controller.running)
        self.assertFalse(self.window._start_button.isEnabled())
        self.assertTrue(self.window._stop_button.isEnabled())
        self.assertTrue(self.window._save_button.isEnabled())
        self.assertIn("Kitchen", self.window._status_label.text())

    def test_poll_stats_updates_totals_and_device_table(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()
        self.window._controller.feed("tv", 0.9)

        self.window._poll_stats()

        self.assertEqual(self.window._device_table.rowCount(), 1)
        self.assertEqual(self.window._device_table.item(0, 0).text(), "Television")
        self.assertNotEqual(self.window._watts_value.text(), "--")
        self.assertNotEqual(self.window._cost_value.text(), "--")

    def test_device_table_tooltip_shows_gemini_note(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()
        self.window._controller.feed("tv", 0.9)
        self.window._controller._aggregator.record_gemini_verdict(
            "tv", 0, 0.9, accepted=True, note="55-inch wall-mounted LED TV"
        )

        self.window._poll_stats()

        self.assertEqual(self.window._device_table.item(0, 0).toolTip(), "55-inch wall-mounted LED TV")

    def test_device_table_no_tooltip_without_gemini_note(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()
        self.window._controller.feed("tv", 0.9)

        self.window._poll_stats()

        self.assertEqual(self.window._device_table.item(0, 0).toolTip(), "")

    def test_poll_frame_renders_pixmap(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()

        self.window._poll_frame()

        self.assertFalse(self.window._camera_label.pixmap().isNull())

    def test_start_clicked_shows_waiting_message_before_first_frame(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()

        self.assertEqual(self.window._camera_label.text(), dashboard_module.WAITING_FOR_FRAME_MESSAGE)

    def test_poll_frame_shows_error_on_streaming_client_failure(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()

        self.window._controller._capture._set_last_error("CertError (980): certificate rejected")
        self.window._poll_frame()

        pixmap = self.window._camera_label.pixmap()
        self.assertTrue(pixmap is None or pixmap.isNull())
        self.assertIn("CertError (980)", self.window._camera_label.text())

    def test_full_lifecycle_saves_report_and_reloads_session_list(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()
        self.window._controller.feed("tv", 0.9)
        self.window._poll_stats()

        self.window._on_stop_clicked()
        self.assertFalse(self.window._controller.running)
        self.assertFalse(self.window._stop_button.isEnabled())

        self.window._on_save_clicked()

        self.assertIsNone(self.window._controller)
        self.assertFalse(self.window._save_button.isEnabled())
        self.assertTrue(self.window._start_button.isEnabled())
        self.assertIn("Report saved", self.window._status_label.text())

        self.assertEqual(self.window._sessions_list.count(), 1)
        session_item = self.window._sessions_list.item(0)
        self.assertIn("Kitchen", session_item.text())

        session_id = session_item.data(Qt.UserRole)
        record = energy_sessions.get_session(session_id)
        self.assertIsNotNone(record)
        self.assertTrue(Path(record["report_html_path"]).exists())
        self.assertTrue(Path(record["report_json_path"]).exists())

    def test_start_with_empty_room_name_shows_warning_and_does_not_start(self) -> None:
        self.window._room_name_edit.setText("   ")

        with mock.patch.object(dashboard_module.QMessageBox, "warning") as mock_warn:
            self.window._on_start_clicked()

        mock_warn.assert_called_once()
        self.assertIsNone(self.window._controller)
        self.assertTrue(self.window._start_button.isEnabled())
        self.assertFalse(self.window._stop_button.isEnabled())
        self.assertFalse(self.window._save_button.isEnabled())

    def test_poll_frame_escalates_to_stale_warning_after_timeout(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()

        with mock.patch.object(self.window._controller, "latest_frame", return_value=None):
            self.window._poll_frame()
            self.assertEqual(self.window._camera_label.text(), dashboard_module.WAITING_FOR_FRAME_MESSAGE)

            with mock.patch.object(
                self.window._controller,
                "seconds_since_start",
                return_value=dashboard_module.ROOMSCAN_DASHBOARD_STALE_FRAME_TIMEOUT_S + 1,
            ):
                self.window._poll_frame()

        self.assertIn("No live frames received", self.window._status_label.text())
        self.assertIn("Still no live RGB frames", self.window._camera_label.text())

    def test_poll_stats_status_live_vs_no_detections(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()
        self.window._poll_frame()  # marks _received_first_frame True

        self.window._poll_stats()
        self.assertIn("no appliances detected yet", self.window._status_label.text())

        self.window._controller.feed("tv", 0.9)
        self.window._poll_stats()
        self.assertIn("Live feed active", self.window._status_label.text())

    def test_gemini_flag_hidden_when_no_rejections(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()
        self.window._controller.feed("tv", 0.9)

        self.window._poll_stats()

        self.assertTrue(self.window._gemini_flag_label.isHidden())

    def test_gemini_flag_shown_on_rejection(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()
        self.window._controller.reject_class("tv", 0.9)

        self.window._poll_stats()

        self.assertFalse(self.window._gemini_flag_label.isHidden())
        self.assertIn("tv", self.window._gemini_flag_label.text())

    def test_device_table_shows_ai_badge_for_gemini_discovered_device(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()
        self.window._controller.feed("tv", 0.9)
        self.window._controller._gemini_discovered["kettle"] = {
            "name": "Kettle",
            "description": "On the counter.",
            "sightings": 1,
            "watts_active": 1200.0,
            "hours_per_day": 0.5,
        }

        self.window._poll_stats()

        rows = {
            self.window._device_table.item(row, 0).text(): row
            for row in range(self.window._device_table.rowCount())
        }
        self.assertIn("Kettle", rows)
        kettle_row = rows["Kettle"]
        self.assertEqual(self.window._device_table.item(kettle_row, 2).text(), "AI")

    def test_device_table_shows_count_and_total_power_for_multiple_discovered_lights(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()
        self.window._controller._gemini_discovered["ceiling light"] = {
            "name": "Ceiling Light",
            "description": "LED fixture.",
            "sightings": 1,
            "watts_active": 9.0,
            "hours_per_day": 5.0,
            "count": 3,
        }

        self.window._poll_stats()

        rows = {
            self.window._device_table.item(row, 0).text(): row
            for row in range(self.window._device_table.rowCount())
        }
        self.assertIn("Ceiling Light", rows)
        row = rows["Ceiling Light"]
        self.assertEqual(self.window._device_table.item(row, 1).text(), "3")
        self.assertEqual(self.window._device_table.item(row, 2).text(), "AI")
        self.assertEqual(self.window._device_table.item(row, 3).text(), "27 W")
        self.assertEqual(self.window._device_table.item(row, 0).toolTip(), "LED fixture.")

    def test_gemini_discovered_label_shown_when_present(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()
        self.window._controller.feed_gemini_discovery("Kettle", "On the counter.")

        self.window._poll_stats()

        self.assertFalse(self.window._gemini_discovered_label.isHidden())
        self.assertIn("Kettle", self.window._gemini_discovered_label.text())

    def test_gemini_discovered_label_hidden_when_empty(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()
        self.window._controller.feed("tv", 0.9)

        self.window._poll_stats()

        self.assertTrue(self.window._gemini_discovered_label.isHidden())

    def test_start_clicked_hides_gemini_widgets_from_prior_room(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()
        self.window._controller.reject_class("tv", 0.9)
        self.window._controller.feed_gemini_discovery("Kettle")
        self.window._poll_stats()
        self.assertFalse(self.window._gemini_flag_label.isHidden())
        self.assertFalse(self.window._gemini_discovered_label.isHidden())

        self.window._on_stop_clicked()
        self.window._on_save_clicked()

        self.window._room_name_edit.setText("Bedroom")
        self.window._on_start_clicked()

        self.assertTrue(self.window._gemini_flag_label.isHidden())
        self.assertTrue(self.window._gemini_discovered_label.isHidden())

    def test_compare_requires_exactly_two_selected_sessions(self) -> None:
        for room in ("Kitchen", "Bedroom"):
            self.window._room_name_edit.setText(room)
            self.window._on_start_clicked()
            self.window._controller.feed("tv", 0.9)
            self.window._on_stop_clicked()
            self.window._on_save_clicked()
        self.assertEqual(self.window._sessions_list.count(), 2)

        self.window._sessions_list.item(0).setSelected(True)
        # _on_compare_clicked() would otherwise open a real (blocking) modal
        # QMessageBox.information() dialog with no one to click it offscreen.
        with mock.patch.object(dashboard_module.QMessageBox, "information") as mock_info:
            self.window._on_compare_clicked()  # only one selected -- must not raise
        mock_info.assert_called_once()

    def test_draw_detection_boxes_modifies_frame_when_detections_present(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()
        frame = _frame(42)

        with mock.patch.object(self.window._controller, "latest_detections", return_value=[_det("tv", 0.9)]):
            result = self.window._draw_detection_boxes(frame)

        self.assertTrue(np.any(result != 42))

    def test_draw_detection_boxes_is_noop_without_detections(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()
        frame = _frame(42)

        result = self.window._draw_detection_boxes(frame)

        self.assertTrue(np.all(result == 42))

    def test_ai_indicator_hidden_when_verification_not_enabled(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()
        self.window._controller.feed("tv", 0.9)

        with mock.patch("roomscan_live.ai_features_enabled", return_value=False):
            self.window._poll_stats()

        self.assertTrue(self.window._ai_status_label.isHidden())

    def test_ai_indicator_shows_while_pass_active(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()
        self.window._controller._gemini_pass_active = True

        with mock.patch("roomscan_live.ai_features_enabled", return_value=True):
            self.window._poll_stats()

        self.assertFalse(self.window._ai_status_label.isHidden())
        self.assertIn("checking", self.window._ai_status_label.text().lower())

    def test_ai_indicator_flashes_complete_after_pass_finishes(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()
        self.window._controller._gemini_pass_active = False
        self.window._controller._gemini_last_pass_ts = time.monotonic()

        with mock.patch("roomscan_live.ai_features_enabled", return_value=True):
            self.window._poll_stats()

        self.assertFalse(self.window._ai_status_label.isHidden())
        self.assertIn("complete", self.window._ai_status_label.text().lower())

    def test_ai_indicator_hides_after_flash_duration_elapses(self) -> None:
        self.window._room_name_edit.setText("Kitchen")
        self.window._on_start_clicked()
        self.window._controller._gemini_pass_active = False
        self.window._controller._gemini_last_pass_ts = time.monotonic()

        with mock.patch("roomscan_live.ai_features_enabled", return_value=True):
            self.window._poll_stats()
        self.assertFalse(self.window._ai_status_label.isHidden())

        future = time.monotonic() + dashboard_module.ROOMSCAN_AI_FLASH_DURATION_S + 1
        with mock.patch("roomscan_dashboard.time.monotonic", return_value=future):
            with mock.patch("roomscan_live.ai_features_enabled", return_value=True):
                self.window._poll_stats()

        self.assertTrue(self.window._ai_status_label.isHidden())


if __name__ == "__main__":
    unittest.main()
