"""PyQt5 fixed-layout live dashboard for the RoomScan energy audit.

Drives roomscan_live.LiveScanController directly -- RoomScan's live backend
already owns its own AriaCapture connection, so unlike bridge.py this
dashboard does not go through a WebSocket. Uses its own "warm energy
instrument" theme (config.RS_* tokens) rather than the blue/cyan palette
training_dashboard.py's dark theme uses -- the two dashboards are visually
independent, deliberately. Panel sizes are RoomScan-specific
(config.ROOMSCAN_DASHBOARD_*) since the content differs.

    python roomscan_dashboard.py [--device-ip <ip> --start-streaming --interface usb --profile profile18] [--out roomscan_out]

Session flow: type a room name, Start Scan, watch live totals/cards update,
Stop Scan (a "Room Efficiency Summary" pops up with the headline numbers),
Save Report (writes the same JSON/HTML roomscan.py produces, into its own
<room>_<timestamp> subfolder so successive rooms never overwrite each other,
and registers the session in energy_sessions.py's shared index). Save
re-enables Start for the next room.

The "Past Scans" list (left panel) is backed entirely by energy_sessions.py's
JSON index: View Report opens a saved session's HTML report in the system
browser, Compare (select exactly two) opens a side-by-side per-device +
totals-delta dialog, and Export writes a CSV of every recorded session.
"""

import argparse
import sys
import time
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config import (
    DEFAULT_STREAM_PROFILE,
    ENERGY_CATALOG,
    ROOMSCAN_AI_FLASH_DURATION_S,
    ROOMSCAN_COMPARE_DIALOG_HEIGHT,
    ROOMSCAN_COMPARE_DIALOG_WIDTH,
    ROOMSCAN_DASHBOARD_BOTTOM_HEIGHT,
    ROOMSCAN_DASHBOARD_CAMERA_HEIGHT,
    ROOMSCAN_DASHBOARD_CAMERA_WIDTH,
    ROOMSCAN_DASHBOARD_FRAME_POLL_MS,
    ROOMSCAN_DASHBOARD_RIGHT_WIDTH,
    ROOMSCAN_DASHBOARD_SIDEBAR_WIDTH,
    ROOMSCAN_DASHBOARD_STALE_FRAME_TIMEOUT_S,
    ROOMSCAN_DASHBOARD_WINDOW_HEIGHT,
    ROOMSCAN_DASHBOARD_WINDOW_WIDTH,
    ROOMSCAN_DETECTION_BOX_COLOR_RGB,
    ROOMSCAN_DETECTION_BOX_THICKNESS,
    ROOMSCAN_EFFICIENCY_FAIR_MAX_COST_USD,
    ROOMSCAN_EFFICIENCY_GOOD_MAX_COST_USD,
    ROOMSCAN_LIVE_TICK_SECONDS,
    ROOMSCAN_OUTPUT_DIR,
    ROOMSCAN_REPORT_HTML_NAME,
    ROOMSCAN_SESSIONS_SUMMARY_CSV_NAME,
    ROOMSCAN_STATUS_DOT_SIZE_PX,
    ROOMSCAN_SUMMARY_DIALOG_HEIGHT,
    ROOMSCAN_SUMMARY_DIALOG_WIDTH,
    ROOMSCAN_SUMMARY_RECOMMENDATIONS_COUNT,
    ROOMSCAN_TOP_DRAINS_COUNT,
    RS_AMBER,
    RS_AMBER_HOVER,
    RS_BAD,
    RS_BG,
    RS_BORDER,
    RS_GOOD,
    RS_MONO_FONT_STACK,
    RS_MUTED,
    RS_SURFACE,
    RS_SURFACE_INSET,
    RS_TEXT,
    RS_WARN,
)
from energy_recommendations import NO_DEVICES_MESSAGE, generate_recommendations
from energy_sessions import compare_sessions, export_summary_csv, get_session, list_sessions
from logging_utils import get_logger
from roomscan_live import LiveScanController, session_out_dir

logger = get_logger(__name__)

DEVICE_TABLE_HEADERS = ["Device", "Qty", "Match %", "Power (W)", "Cost / Year"]

# Scan status indicator: plain-English label + accent color per lifecycle
# state, shown in the left panel's status pill (dot + label) at all times so
# the user always has a one-line explanation of what the scan is doing --
# never just a spinner or silence. STATUS_CONNECTING/WAITING/STALE/ERROR are
# driven by the camera-feed poll (_poll_frame); STATUS_LIVE/NO_DETECTIONS are
# driven by the stats poll (_poll_stats), once frames are actually flowing.
# STATUS_DEBUG_CAMERA_ONLY replaces STATUS_LIVE/NO_DETECTIONS when
# --debug-camera-only is passed: it fires once frames are confirmed flowing,
# so it doubles as proof that camera delivery/rendering works independently
# of the (deliberately disabled) detector.
STATUS_IDLE = "idle"
STATUS_CONNECTING = "connecting"
STATUS_WAITING = "waiting"
STATUS_LIVE = "live"
STATUS_NO_DETECTIONS = "no_detections"
STATUS_DEBUG_CAMERA_ONLY = "debug_camera_only"
STATUS_STALE = "stale"
STATUS_ERROR = "error"
STATUS_STOPPED = "stopped"
STATUS_SAVED = "saved"
_STATUS_TEXT = {
    STATUS_IDLE: "Ready to scan",
    STATUS_CONNECTING: "Connecting to live scan...",
    STATUS_WAITING: "Waiting for RGB frames...",
    STATUS_LIVE: "Live feed active",
    STATUS_NO_DETECTIONS: "Scan running, no appliances detected yet",
    STATUS_DEBUG_CAMERA_ONLY: "DEBUG: camera OK, detector disabled",
    STATUS_STALE: "No live frames received -- check the Aria stream connection",
    STATUS_ERROR: "Capture error",
    STATUS_STOPPED: "Scan stopped",
    STATUS_SAVED: "Report saved",
}
_STATUS_COLOR = {
    STATUS_IDLE: RS_MUTED,
    STATUS_CONNECTING: RS_MUTED,
    STATUS_WAITING: RS_MUTED,
    STATUS_LIVE: RS_GOOD,
    STATUS_NO_DETECTIONS: RS_WARN,
    STATUS_DEBUG_CAMERA_ONLY: RS_WARN,
    STATUS_STALE: RS_WARN,
    STATUS_ERROR: RS_BAD,
    STATUS_STOPPED: RS_WARN,
    STATUS_SAVED: RS_AMBER,
}

PRE_SCAN_DEVICE_MESSAGE = "Start a scan to see devices here."
PRE_SCAN_DRAINS_MESSAGE = "Start a scan to see your biggest energy users."
PRE_SCAN_TIPS_MESSAGE = "Start a scan to get personalized energy-saving tips."
NO_DEVICES_TABLE_MESSAGE = "No devices detected yet -- keep scanning."

WAITING_FOR_FRAME_MESSAGE = "Waiting for RGB frames..."
CONNECTING_MESSAGE = "Connecting to live scan..."
CAMERA_ERROR_PREFIX = "Live camera feed error:"
NO_ROOM_NAME_WARNING = "Please enter a room name before starting a scan."

# Consolidated app-wide stylesheet ("warm energy instrument" theme, config.RS_*
# tokens). Applied once via QApplication.instance().setStyleSheet() rather than
# on the QMainWindow itself, so it reliably cascades to the modal dialogs below
# too (they're constructed with a parent, but a window-level setStyleSheet()
# call doesn't cascade to QDialogs as consistently as an application-level one
# does). objectName-keyed selectors let _panel()/_hero_stat_row()/_stat_row()
# and the button-construction call sites opt into a role purely by name --
# no other code here needs to touch color literals directly.
_APP_QSS = f"""
QMainWindow, QDialog {{
    background-color: {RS_BG};
    color: {RS_TEXT};
}}
QWidget {{
    color: {RS_TEXT};
    font-size: 13px;
}}
QLabel {{
    border: none;
}}

QWidget#panel {{
    background-color: {RS_SURFACE};
    border: 1px solid {RS_BORDER};
    border-radius: 10px;
}}
QLabel#panelHeading {{
    color: {RS_MUTED};
    font-size: 12px;
    font-weight: 600;
}}

QWidget#meterWindow {{
    background-color: {RS_SURFACE_INSET};
    border: 1px solid {RS_BORDER};
    border-top: 2px solid {RS_AMBER};
    border-radius: 6px;
}}
QLabel#meterValue {{
    color: {RS_AMBER};
    font-family: {RS_MONO_FONT_STACK};
    font-size: 40px;
    font-weight: 800;
}}
QLabel#meterCaption {{
    color: {RS_MUTED};
    font-size: 12px;
}}

QLabel#statValue {{
    color: {RS_TEXT};
    font-family: {RS_MONO_FONT_STACK};
    font-size: 22px;
    font-weight: 700;
}}
QLabel#statCaption {{
    color: {RS_MUTED};
    font-size: 12px;
}}

QLineEdit {{
    background-color: {RS_SURFACE_INSET};
    border: 1px solid {RS_BORDER};
    border-radius: 4px;
    padding: 6px 8px;
    color: {RS_TEXT};
}}
QLineEdit:focus {{
    border: 1px solid {RS_AMBER};
}}

QPushButton {{
    background-color: {RS_SURFACE_INSET};
    border: 1px solid {RS_BORDER};
    border-radius: 4px;
    padding: 7px 12px;
    color: {RS_TEXT};
    font-weight: 600;
}}
QPushButton:hover {{
    border: 1px solid {RS_AMBER};
}}
QPushButton:disabled {{
    color: {RS_MUTED};
    border: 1px solid {RS_BORDER};
}}

QPushButton#primaryButton {{
    background-color: {RS_AMBER};
    border: 1px solid {RS_AMBER};
    color: {RS_SURFACE_INSET};
}}
QPushButton#primaryButton:hover {{
    background-color: {RS_AMBER_HOVER};
    border: 1px solid {RS_AMBER_HOVER};
}}
QPushButton#primaryButton:disabled {{
    background-color: {RS_SURFACE_INSET};
    color: {RS_MUTED};
    border: 1px solid {RS_BORDER};
}}

QPushButton#ghostButton {{
    background-color: transparent;
    border: 1px solid transparent;
    color: {RS_MUTED};
}}
QPushButton#ghostButton:hover {{
    color: {RS_TEXT};
    border: 1px solid {RS_BORDER};
}}

QListWidget, QTableWidget {{
    background-color: {RS_SURFACE_INSET};
    border: 1px solid {RS_BORDER};
    border-radius: 4px;
    color: {RS_TEXT};
}}
QListWidget::item, QTableWidget::item {{
    padding: 3px;
    color: {RS_TEXT};
}}
QListWidget::item:selected, QTableWidget::item:selected {{
    background-color: rgba(232, 163, 61, 60);
    color: {RS_TEXT};
}}
QHeaderView::section {{
    background-color: {RS_SURFACE};
    color: {RS_MUTED};
    border: none;
    border-bottom: 1px solid {RS_BORDER};
    padding: 4px;
    font-weight: 600;
}}
"""


def _fmt_delta(value: float, unit: str = "") -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}{unit}"


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(max(seconds, 0)), 60)
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


def _efficiency_rating(annual_cost_usd: float) -> Tuple[str, str]:
    """Plain-English efficiency badge for the end-of-scan summary.

    Hackathon-grade heuristic over the same typical-draw priors as
    config.ENERGY_CATALOG, gated by config.ROOMSCAN_EFFICIENCY_*_MAX_COST_USD
    -- not a measured or certified efficiency rating.
    """
    if annual_cost_usd <= ROOMSCAN_EFFICIENCY_GOOD_MAX_COST_USD:
        return "Great efficiency", RS_GOOD
    if annual_cost_usd <= ROOMSCAN_EFFICIENCY_FAIR_MAX_COST_USD:
        return "Room for improvement", RS_WARN
    return "High energy use", RS_BAD


def _placeholder_item(message: str) -> QListWidgetItem:
    item = QListWidgetItem(message)
    item.setForeground(QColor(RS_MUTED))
    return item


class RoomEfficiencySummaryDialog(QDialog):
    """End-of-scan report card shown right after Stop Scan.

    Built entirely from the same {room_name, devices, totals} shape
    LiveScanController.snapshot() already returns, so it needs no extra
    backend plumbing beyond what the live view already polls.
    """

    def __init__(self, state: Dict[str, object], duration_seconds: float, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        totals = state["totals"]
        devices: List[Dict[str, object]] = state["devices"]

        self.setWindowTitle("Room Efficiency Summary")
        self.resize(ROOMSCAN_SUMMARY_DIALOG_WIDTH, ROOMSCAN_SUMMARY_DIALOG_HEIGHT)

        layout = QVBoxLayout(self)

        room_label = QLabel(f"{state['room_name']}  •  {_format_duration(duration_seconds)} scan")
        room_label.setStyleSheet(f"color: {RS_MUTED}; font-size: 13px;")
        layout.addWidget(room_label)

        rating_text, rating_color = _efficiency_rating(totals["cost_per_year_usd"])
        rating_label = QLabel(rating_text)
        rating_label.setStyleSheet(f"color: {rating_color}; font-size: 20px; font-weight: 700;")
        layout.addWidget(rating_label)

        cost_label = QLabel(f"${totals['cost_per_year_usd']:.2f} / year")
        cost_label.setStyleSheet(f"color: {RS_AMBER}; font-family: {RS_MONO_FONT_STACK}; font-size: 36px; font-weight: 800;")
        layout.addWidget(cost_label)

        stats_label = QLabel(f"{len(devices)} device type(s) found  •  {totals['kwh_per_year']:.0f} kWh / year")
        stats_label.setStyleSheet(f"color: {RS_TEXT}; font-size: 13px;")
        layout.addWidget(stats_label)

        if devices:
            top = devices[0]
            top_text = f"Biggest energy user: {top['display_name']} (${top['cost_per_year_usd']:.2f}/yr)"
        else:
            top_text = "No devices were detected during this scan."
        top_label = QLabel(top_text)
        top_label.setWordWrap(True)
        top_label.setStyleSheet(f"color: {RS_AMBER}; font-size: 13px; font-weight: 600;")
        layout.addWidget(top_label)

        tips_heading = QLabel("Ways to save:")
        tips_heading.setStyleSheet(f"color: {RS_AMBER}; font-size: 13px; font-weight: 600;")
        layout.addWidget(tips_heading)

        tips_list = QListWidget()
        tips_list.setWordWrap(True)
        tips_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        suggestions = generate_recommendations(devices, totals, {"avg_brightness": None})
        for suggestion in suggestions[:ROOMSCAN_SUMMARY_RECOMMENDATIONS_COUNT]:
            tips_list.addItem(suggestion)
        layout.addWidget(tips_list, 1)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)


class SessionCompareDialog(QDialog):
    """Side-by-side per-device + totals comparison of two saved RoomScan sessions."""

    def __init__(self, comparison: Dict[str, object], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        session_a, session_b = comparison["session_a"], comparison["session_b"]
        self.setWindowTitle("Compare Two Scans")
        self.resize(ROOMSCAN_COMPARE_DIALOG_WIDTH, ROOMSCAN_COMPARE_DIALOG_HEIGHT)

        layout = QVBoxLayout(self)
        header = QLabel(
            f"A: {session_a['room_name']}  ({session_a['timestamp']})    vs.    "
            f"B: {session_b['room_name']}  ({session_b['timestamp']})"
        )
        header.setStyleSheet(f"color: {RS_AMBER}; font-size: 14px; font-weight: 600;")
        header.setWordWrap(True)
        layout.addWidget(header)

        headers = ["Device", "Qty A", "Qty B", "kWh/yr A", "kWh/yr B", "Cost/yr A", "Cost/yr B"]
        table = QTableWidget(len(comparison["devices"]), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.horizontalHeader().setStretchLastSection(True)
        for row, device in enumerate(comparison["devices"]):
            values = [
                device["display_name"],
                str(device["count_a"]),
                str(device["count_b"]),
                f"{device['kwh_per_year_a']:.0f}",
                f"{device['kwh_per_year_b']:.0f}",
                f"${device['cost_per_year_usd_a']:.2f}",
                f"${device['cost_per_year_usd_b']:.2f}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row, col, item)
        layout.addWidget(table)

        delta = comparison["totals_delta"]
        delta_label = QLabel(
            f"Change from A to B:  kWh/day {_fmt_delta(delta['kwh_per_day'])}   "
            f"kWh/yr {_fmt_delta(delta['kwh_per_year'])}   "
            f"$/yr {_fmt_delta(delta['cost_per_year_usd'])}"
        )
        delta_label.setStyleSheet(f"color: {RS_GOOD}; font-weight: 600;")
        layout.addWidget(delta_label)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)


class RoomScanDashboard(QMainWindow):
    """Single-room live RoomScan session: Start -> live updates -> Stop -> Save."""

    def __init__(self, connection_args: argparse.Namespace) -> None:
        super().__init__()
        self._connection_args = connection_args
        self._debug_camera_only = getattr(connection_args, "debug_camera_only", False)
        self._controller: Optional[LiveScanController] = None
        self._last_avg_brightness: Optional[float] = None
        self._received_first_frame = False
        self._logged_first_frame_update = False
        self._last_camera_error: Optional[str] = None
        self._logged_first_devices_in_table = False
        self._logged_stale_frame_warning = False
        # "AI just ran" flash indicator (see _update_ai_indicator): tracks the
        # last gemini_last_pass_ts we've already reacted to, so a repeated
        # snapshot() poll doesn't restart the flash timer every tick, plus
        # the monotonic deadline the flash message stays visible until.
        self._last_seen_gemini_pass_ts: Optional[float] = None
        self._ai_flash_until: Optional[float] = None

        title = "RoomScan — Home Energy Scanner"
        if self._debug_camera_only:
            title += "  [DEBUG: camera-only, detector disabled]"
            logger.warning(
                "Dashboard starting in --debug-camera-only mode: EnergyDetector is disabled, "
                "only the live RGB feed will be exercised."
            )
        self.setWindowTitle(title)
        self.setFixedSize(ROOMSCAN_DASHBOARD_WINDOW_WIDTH, ROOMSCAN_DASHBOARD_WINDOW_HEIGHT)
        # Applied at the QApplication level (not self.setStyleSheet) so it
        # reliably cascades to the modal dialogs below too.
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(_APP_QSS)

        self._build_ui()
        self._show_pre_scan_placeholders()
        self._set_status(STATUS_IDLE)
        self._reload_sessions()

        self._frame_timer = QTimer(self)
        self._frame_timer.timeout.connect(self._poll_frame)
        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._poll_stats)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _panel(self, title: str, fixed_width: Optional[int] = None) -> QWidget:
        frame = QWidget()
        frame.setObjectName("panel")
        if fixed_width is not None:
            frame.setFixedWidth(fixed_width)
        layout = QVBoxLayout(frame)
        heading = QLabel(title)
        heading.setObjectName("panelHeading")
        layout.addWidget(heading)
        frame._content_layout = layout  # stash for callers to keep adding widgets
        return frame

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        top_row = QHBoxLayout()
        top_row.addWidget(self._build_left_panel())
        top_row.addWidget(self._build_center_panel(), 1)
        top_row.addWidget(self._build_right_panel())
        outer.addLayout(top_row, 1)

        outer.addWidget(self._build_bottom_panel())

    def _build_left_panel(self) -> QWidget:
        panel = self._panel("Scan Controls", fixed_width=ROOMSCAN_DASHBOARD_SIDEBAR_WIDTH)
        layout = panel._content_layout

        layout.addWidget(QLabel("Which room is this?"))
        self._room_name_edit = QLineEdit()
        self._room_name_edit.setPlaceholderText("e.g. Kitchen, Bedroom, Office")
        layout.addWidget(self._room_name_edit)

        self._start_button = QPushButton("Start Scan")
        self._start_button.setObjectName("primaryButton")
        self._start_button.clicked.connect(self._on_start_clicked)
        layout.addWidget(self._start_button)

        self._stop_button = QPushButton("Stop Scan")
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self._on_stop_clicked)
        layout.addWidget(self._stop_button)

        self._save_button = QPushButton("Save Report")
        self._save_button.setEnabled(False)
        self._save_button.clicked.connect(self._on_save_clicked)
        layout.addWidget(self._save_button)

        status_row = QHBoxLayout()
        self._status_dot = QLabel()
        self._status_dot.setFixedSize(ROOMSCAN_STATUS_DOT_SIZE_PX, ROOMSCAN_STATUS_DOT_SIZE_PX)
        status_row.addWidget(self._status_dot)
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        status_row.addWidget(self._status_label, 1)
        layout.addLayout(status_row)

        sessions_heading = QLabel("Past Scans")
        sessions_heading.setObjectName("panelHeading")
        layout.addWidget(sessions_heading)
        self._sessions_list = QListWidget()
        self._sessions_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._sessions_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self._sessions_list, 1)

        session_buttons_row = QHBoxLayout()
        self._review_button = QPushButton("View Report")
        self._review_button.setObjectName("ghostButton")
        self._review_button.clicked.connect(self._on_review_clicked)
        session_buttons_row.addWidget(self._review_button)
        self._compare_button = QPushButton("Compare")
        self._compare_button.setObjectName("ghostButton")
        self._compare_button.clicked.connect(self._on_compare_clicked)
        session_buttons_row.addWidget(self._compare_button)
        layout.addLayout(session_buttons_row)

        self._export_sessions_button = QPushButton("Export All Scans (CSV)...")
        self._export_sessions_button.setObjectName("ghostButton")
        self._export_sessions_button.clicked.connect(self._on_export_sessions_clicked)
        layout.addWidget(self._export_sessions_button)

        return panel

    def _build_center_panel(self) -> QWidget:
        panel = self._panel("Live Camera View")
        layout = panel._content_layout
        self._camera_label = QLabel("Point your Aria glasses around the room.\nThe live view appears here once you start a scan.")
        self._camera_label.setAlignment(Qt.AlignCenter)
        self._camera_label.setWordWrap(True)
        self._camera_label.setFixedSize(ROOMSCAN_DASHBOARD_CAMERA_WIDTH, ROOMSCAN_DASHBOARD_CAMERA_HEIGHT)
        self._camera_label.setStyleSheet(f"background-color: #000; color: {RS_MUTED}; border: 1px solid {RS_BORDER};")
        layout.addWidget(self._camera_label)
        # "AI just ran" indicator (see _update_ai_indicator): hidden until the
        # background Gemini live-verification/discovery pass is actually
        # active or has just completed, same hidden-until-populated pattern
        # as _gemini_flag_label/_gemini_discovered_label below.
        self._ai_status_label = QLabel("")
        self._ai_status_label.setAlignment(Qt.AlignCenter)
        self._ai_status_label.setStyleSheet(f"color: {RS_AMBER}; font-size: 12px; font-weight: 600;")
        self._ai_status_label.hide()
        layout.addWidget(self._ai_status_label)
        layout.addStretch(1)
        return panel

    def _hero_stat_row(self, layout: QVBoxLayout, caption: str) -> QLabel:
        """The "meter register window" signature element: the one place of
        visual boldness in this theme, everything else stays quiet around it."""
        meter = QWidget()
        meter.setObjectName("meterWindow")
        meter_layout = QVBoxLayout(meter)
        value_label = QLabel("--")
        value_label.setObjectName("meterValue")
        value_label.setAlignment(Qt.AlignCenter)
        meter_layout.addWidget(value_label)
        caption_label = QLabel(caption)
        caption_label.setObjectName("meterCaption")
        caption_label.setAlignment(Qt.AlignCenter)
        meter_layout.addWidget(caption_label)
        layout.addWidget(meter)
        return value_label

    def _stat_row(self, layout: QVBoxLayout, caption: str) -> QLabel:
        value_label = QLabel("--")
        value_label.setObjectName("statValue")
        caption_label = QLabel(caption)
        caption_label.setObjectName("statCaption")
        layout.addWidget(value_label)
        layout.addWidget(caption_label)
        return value_label

    def _build_right_panel(self) -> QWidget:
        panel = self._panel("Energy At A Glance", fixed_width=ROOMSCAN_DASHBOARD_RIGHT_WIDTH)
        layout = panel._content_layout
        # Cost is the headline number non-technical viewers care about most,
        # so it's rendered as a hero stat; the rest are supporting detail.
        self._cost_value = self._hero_stat_row(layout, "Estimated Cost Per Year")
        layout.addSpacing(8)
        self._watts_value = self._stat_row(layout, "Power In Use Right Now")
        self._kwh_day_value = self._stat_row(layout, "Energy Used Per Day")
        self._kwh_year_value = self._stat_row(layout, "Energy Used Per Year")
        layout.addStretch(1)
        return panel

    def _build_bottom_panel(self) -> QWidget:
        container = QWidget()
        container.setFixedHeight(ROOMSCAN_DASHBOARD_BOTTOM_HEIGHT)
        row = QHBoxLayout(container)

        devices_panel = self._panel("Devices Found")
        self._device_table = QTableWidget(0, len(DEVICE_TABLE_HEADERS))
        self._device_table.setHorizontalHeaderLabels(DEVICE_TABLE_HEADERS)
        self._device_table.horizontalHeader().setStretchLastSection(True)
        self._device_table.verticalHeader().setVisible(False)
        self._device_table.setEditTriggers(QTableWidget.NoEditTriggers)
        devices_panel._content_layout.addWidget(self._device_table)
        # Gemini live-verification/discovery surfacing: a warning banner when
        # a class was auto-corrected (rejected) this scan, and a note listing
        # any non-catalog appliance types Gemini spotted but never prices.
        # Both start hidden -- shown only once _poll_stats sees something to
        # report -- and are hidden again by _show_pre_scan_placeholders /
        # _on_start_clicked's leftover-results clear.
        self._gemini_flag_label = QLabel("")
        self._gemini_flag_label.setWordWrap(True)
        self._gemini_flag_label.setStyleSheet(f"color: {RS_WARN}; font-size: 12px;")
        self._gemini_flag_label.hide()
        devices_panel._content_layout.addWidget(self._gemini_flag_label)
        self._gemini_discovered_label = QLabel("")
        self._gemini_discovered_label.setWordWrap(True)
        self._gemini_discovered_label.setStyleSheet(f"color: {RS_MUTED}; font-size: 12px;")
        self._gemini_discovered_label.hide()
        devices_panel._content_layout.addWidget(self._gemini_discovered_label)
        row.addWidget(devices_panel, 2)

        drains_panel = self._panel("Biggest Energy Users", fixed_width=ROOMSCAN_DASHBOARD_RIGHT_WIDTH)
        self._top_drains_list = QListWidget()
        self._top_drains_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        drains_panel._content_layout.addWidget(self._top_drains_list)
        row.addWidget(drains_panel, 1)

        actions_panel = self._panel("Ways To Save Energy", fixed_width=ROOMSCAN_DASHBOARD_RIGHT_WIDTH)
        self._recommendations_list = QListWidget()
        self._recommendations_list.setWordWrap(True)
        self._recommendations_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        actions_panel._content_layout.addWidget(self._recommendations_list)
        row.addWidget(actions_panel, 1)

        return container

    # ------------------------------------------------------------------
    # Empty-state helpers
    # ------------------------------------------------------------------

    def _set_table_placeholder(self, message: str) -> None:
        table = self._device_table
        table.clearSpans()
        table.setRowCount(1)
        table.setSpan(0, 0, 1, len(DEVICE_TABLE_HEADERS))
        item = QTableWidgetItem(message)
        item.setTextAlignment(Qt.AlignCenter)
        item.setForeground(QColor(RS_MUTED))
        table.setItem(0, 0, item)

    def _show_pre_scan_placeholders(self) -> None:
        """Clean, unmistakably "not started yet" state for every result panel."""
        for value_label in (self._cost_value, self._watts_value, self._kwh_day_value, self._kwh_year_value):
            value_label.setText("--")
        self._set_table_placeholder(PRE_SCAN_DEVICE_MESSAGE)
        self._top_drains_list.clear()
        self._top_drains_list.addItem(_placeholder_item(PRE_SCAN_DRAINS_MESSAGE))
        self._recommendations_list.clear()
        self._recommendations_list.addItem(_placeholder_item(PRE_SCAN_TIPS_MESSAGE))
        self._gemini_flag_label.hide()
        self._gemini_discovered_label.hide()
        self._ai_status_label.hide()

    # ------------------------------------------------------------------
    # Scan status indicator
    # ------------------------------------------------------------------

    def _set_status(self, status: str, detail: str = "") -> None:
        color = _STATUS_COLOR[status]
        dot_radius = ROOMSCAN_STATUS_DOT_SIZE_PX // 2
        self._status_dot.setStyleSheet(f"background-color: {color}; border-radius: {dot_radius}px; border: none;")
        text = _STATUS_TEXT[status]
        self._status_label.setText(f"{text} — {detail}" if detail else text)
        self._status_label.setStyleSheet(f"color: {color}; font-weight: 600; border: none;")

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _on_start_clicked(self) -> None:
        if self._controller is not None and self._controller.running:
            return
        room_name = self._room_name_edit.text().strip()
        if not room_name:
            QMessageBox.warning(self, "Room name needed", NO_ROOM_NAME_WARNING)
            return
        logger.info("Start Scan clicked (room=%r).", room_name)

        # controller.start() blocks (SDK subscribe/handshake); paint the
        # "Connecting..." state before that call so the UI never looks frozen
        # or silent while it's in flight, especially with --start-streaming.
        self._show_connecting()
        self._set_status(STATUS_CONNECTING, detail=room_name)
        QApplication.processEvents()

        try:
            controller = LiveScanController(
                room_name=room_name,
                device_ip=self._connection_args.device_ip,
                start_streaming=self._connection_args.start_streaming,
                streaming_interface=self._connection_args.interface,
                profile_name=self._connection_args.profile,
                use_ephemeral_certs=not self._connection_args.persistent_certs,
                local_certs_dir=self._connection_args.local_certs_dir,
                disable_detection=self._debug_camera_only,
            )
            controller.start()
        except Exception as exc:
            logger.error("Failed to start live scan: %s", exc)
            QMessageBox.critical(self, "Failed to start scan", str(exc))
            self._show_camera_error(str(exc))
            self._set_status(STATUS_ERROR, detail=str(exc))
            return
        self._controller = controller

        # Clear any leftover results from a previous room before this scan's
        # first tick arrives, so nothing stale lingers on screen.
        self._device_table.clearSpans()
        self._device_table.setRowCount(0)
        self._top_drains_list.clear()
        self._recommendations_list.clear()
        self._gemini_flag_label.hide()
        self._gemini_discovered_label.hide()
        self._last_seen_gemini_pass_ts = None
        self._ai_flash_until = None
        self._ai_status_label.hide()

        # No frame has arrived for *this* scan yet -- replace whatever the
        # camera label was showing (pre-scan placeholder or a prior room's
        # last frame) with an explicit waiting state rather than leaving
        # stale/ambiguous content on screen while the first frame is in flight.
        self._received_first_frame = False
        self._logged_first_frame_update = False
        self._last_camera_error = None
        self._logged_first_devices_in_table = False
        self._logged_stale_frame_warning = False
        self._show_waiting_for_frame()

        self._frame_timer.start(ROOMSCAN_DASHBOARD_FRAME_POLL_MS)
        self._stats_timer.start(int(ROOMSCAN_LIVE_TICK_SECONDS * 1000))
        self._room_name_edit.setEnabled(False)
        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(True)
        self._save_button.setEnabled(True)
        self._set_status(STATUS_WAITING, detail=room_name)

    def _on_stop_clicked(self) -> None:
        if self._controller is None or not self._controller.running:
            return
        self._controller.stop()
        self._frame_timer.stop()
        self._stats_timer.stop()
        self._stop_button.setEnabled(False)
        self._set_status(STATUS_STOPPED, detail=self._controller.room_name)

        try:
            final_state = self._controller.snapshot()
        except Exception as exc:
            logger.warning("Failed to build end-of-scan summary: %s", exc)
            return
        duration_s = self._controller.seconds_since_start()
        RoomEfficiencySummaryDialog(final_state, duration_s, self).exec_()

    def _on_save_clicked(self) -> None:
        if self._controller is None or self._controller.finished:
            return
        out_dir = session_out_dir(self._connection_args.out, self._controller.room_name)
        try:
            self._controller.finish(out_dir=str(out_dir))
        except Exception as exc:
            logger.error("Failed to save report: %s", exc)
            QMessageBox.critical(self, "Failed to save report", str(exc))
            return

        self._frame_timer.stop()
        self._stats_timer.stop()
        html_path = out_dir / ROOMSCAN_REPORT_HTML_NAME
        self._set_status(STATUS_SAVED, detail=str(html_path))

        self._save_button.setEnabled(False)
        self._stop_button.setEnabled(False)
        self._start_button.setEnabled(True)
        self._room_name_edit.setEnabled(True)
        self._controller = None
        self._reload_sessions()

    # ------------------------------------------------------------------
    # Session review / compare / export (energy_sessions.py-backed index)
    # ------------------------------------------------------------------

    def _reload_sessions(self) -> None:
        self._sessions_list.clear()
        for record in list_sessions():
            item = QListWidgetItem(f"{record['room_name']} — {record['timestamp']}")
            item.setData(Qt.UserRole, record["session_id"])
            self._sessions_list.addItem(item)

    def _on_review_clicked(self) -> None:
        items = self._sessions_list.selectedItems()
        if len(items) != 1:
            QMessageBox.information(self, "View report", "Please select exactly one scan to view.")
            return
        record = get_session(items[0].data(Qt.UserRole))
        if record is None:
            QMessageBox.warning(self, "View report", "That scan's record could not be found.")
            return
        webbrowser.open(Path(record["report_html_path"]).as_uri())

    def _on_compare_clicked(self) -> None:
        items = self._sessions_list.selectedItems()
        if len(items) != 2:
            QMessageBox.information(self, "Compare scans", "Please select exactly two scans to compare.")
            return
        record_a = get_session(items[0].data(Qt.UserRole))
        record_b = get_session(items[1].data(Qt.UserRole))
        if record_a is None or record_b is None:
            QMessageBox.warning(self, "Compare scans", "That scan's record could not be found.")
            return
        SessionCompareDialog(compare_sessions(record_a, record_b), self).exec_()

    def _on_export_sessions_clicked(self) -> None:
        default_path = str(Path(self._connection_args.out).expanduser() / ROOMSCAN_SESSIONS_SUMMARY_CSV_NAME)
        path, _ = QFileDialog.getSaveFileName(self, "Export all scans", default_path, "CSV files (*.csv)")
        if not path:
            return
        try:
            export_summary_csv(path)
        except Exception as exc:
            logger.error("Failed to export session summary: %s", exc)
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        QMessageBox.information(self, "Export complete", f"Wrote {path}")

    # ------------------------------------------------------------------
    # Polling (both timers run on the Qt main thread -- safe to touch widgets)
    # ------------------------------------------------------------------

    def _set_camera_message(self, message: str, color: str) -> None:
        """Render a textual camera-panel state. setText() implicitly clears
        any previously set QLabel pixmap, so the panel is never left showing
        a stale frame alongside a state message that contradicts it."""
        self._camera_label.setText(message)
        self._camera_label.setStyleSheet(f"background-color: #000; color: {color}; border: 1px solid {RS_BORDER};")

    def _show_connecting(self) -> None:
        self._set_camera_message(CONNECTING_MESSAGE, RS_MUTED)

    def _show_waiting_for_frame(self) -> None:
        self._set_camera_message(WAITING_FOR_FRAME_MESSAGE, RS_MUTED)

    def _show_stale_frame_warning(self, elapsed_s: float) -> None:
        if not self._logged_stale_frame_warning:
            logger.warning("No camera-rgb frame received %.1fs after scan start.", elapsed_s)
            self._logged_stale_frame_warning = True
        self._set_camera_message(
            f"⚠ Still no live RGB frames after {elapsed_s:.0f}s.\nCheck the Aria streaming connection.",
            RS_WARN,
        )

    def _show_camera_error(self, message: str) -> None:
        if message != self._last_camera_error:
            logger.error("Live camera feed error: %s", message)
            self._last_camera_error = message
        self._set_camera_message(f"⚠ {CAMERA_ERROR_PREFIX}\n{message}", RS_BAD)

    def _draw_detection_boxes(self, frame: np.ndarray) -> np.ndarray:
        """Draw a box + label over each currently confirmed live detection,
        directly on the RGB frame array before it becomes a QPixmap -- so the
        boxes scale/aspect-fit together with the frame when
        QPixmap.scaled() resizes it for display, rather than needing to be
        repositioned separately. No-op (empty list) in --debug-camera-only
        mode, since there's no detector running to produce detections."""
        for det in self._controller.latest_detections():
            x1, y1, x2, y2 = det.box_xyxy
            cv2.rectangle(
                frame, (x1, y1), (x2, y2), ROOMSCAN_DETECTION_BOX_COLOR_RGB, ROOMSCAN_DETECTION_BOX_THICKNESS
            )
            display_name = ENERGY_CATALOG.get(det.class_name, {}).get("display", det.class_name)
            label = f"{display_name} {det.confidence:.0%}"
            label_y = y1 - 6 if y1 - 6 > 10 else y1 + 16
            cv2.putText(
                frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                ROOMSCAN_DETECTION_BOX_COLOR_RGB, 1, cv2.LINE_AA,
            )
        return frame

    def _poll_frame(self) -> None:
        if self._controller is None:
            return

        stream_error = self._controller.last_error()
        if stream_error is not None:
            self._show_camera_error(stream_error)
            self._set_status(STATUS_ERROR, detail=stream_error)
            return

        try:
            frame = self._controller.latest_frame()
        except Exception as exc:
            self._show_camera_error(str(exc))
            self._set_status(STATUS_ERROR, detail=str(exc))
            return

        if frame is None:
            # No camera-rgb sample has arrived yet -- keep an explicit waiting
            # state on screen instead of a stale/blank label, escalating to a
            # visible warning if it's taking far longer than expected.
            if not self._received_first_frame:
                elapsed = self._controller.seconds_since_start()
                if elapsed >= ROOMSCAN_DASHBOARD_STALE_FRAME_TIMEOUT_S:
                    self._show_stale_frame_warning(elapsed)
                    self._set_status(STATUS_STALE, detail=self._controller.room_name)
                else:
                    self._show_waiting_for_frame()
            return

        if not self._received_first_frame:
            logger.info("Dashboard received first camera-rgb frame from LiveScanController.")
            self._received_first_frame = True

        frame = np.ascontiguousarray(frame)
        self._last_avg_brightness = float(frame.mean())
        frame = self._draw_detection_boxes(frame)
        height, width, _ = frame.shape
        qimage = QImage(frame.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            ROOMSCAN_DASHBOARD_CAMERA_WIDTH,
            ROOMSCAN_DASHBOARD_CAMERA_HEIGHT,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._camera_label.setPixmap(pixmap)
        if not self._logged_first_frame_update:
            logger.info("Camera widget updated with first live frame (%dx%d).", width, height)
            self._logged_first_frame_update = True
        else:
            logger.debug("Camera widget updated (%dx%d).", width, height)

    def _poll_stats(self) -> None:
        if self._controller is None:
            return
        try:
            state = self._controller.snapshot()
        except Exception as exc:
            logger.warning("Failed to poll scan stats: %s", exc)
            return
        totals = state["totals"]
        devices = state["devices"]
        logger.debug(
            "_poll_stats: frames_sampled=%d devices=%d watts_active=%.0f",
            state["frames_sampled"], len(devices), totals["watts_active"],
        )

        self._cost_value.setText(f"${totals['cost_per_year_usd']:.2f}")
        self._watts_value.setText(f"{totals['watts_active']:.0f} W")
        self._kwh_day_value.setText(f"{totals['kwh_per_day']:.2f} kWh")
        self._kwh_year_value.setText(f"{totals['kwh_per_year']:.0f} kWh")

        # Once frames are actually flowing (no camera error), reflect whether
        # anything's been detected yet; before the first frame, leave the
        # connecting/waiting/stale status that _poll_frame already set alone.
        # In --debug-camera-only mode there is no detector to report on --
        # frames flowing is itself the signal worth surfacing, so it gets its
        # own status instead of "no appliances detected yet" (which would
        # wrongly imply detection ran and came up empty).
        if self._received_first_frame and self._controller.last_error() is None:
            if self._debug_camera_only:
                self._set_status(STATUS_DEBUG_CAMERA_ONLY, detail=state["room_name"])
            else:
                detail = f"{state['room_name']} — {state['frames_sampled']} frames analyzed"
                self._set_status(STATUS_LIVE if devices else STATUS_NO_DETECTIONS, detail=detail)

        self._update_device_table(devices)
        self._update_top_drains(devices)
        self._update_recommendations(devices, totals)
        self._update_gemini_flag(state.get("gemini_rejected_classes", []))
        self._update_gemini_discovered(state.get("gemini_discovered_devices", []))
        self._update_ai_indicator(state)

    def _update_device_table(self, devices: List[Dict[str, object]]) -> None:
        table = self._device_table
        table.clearSpans()
        if not devices:
            self._set_table_placeholder(NO_DEVICES_TABLE_MESSAGE)
            return
        if not self._logged_first_devices_in_table:
            logger.info("Device table populated with first detection(s): %s", [d["display_name"] for d in devices])
            self._logged_first_devices_in_table = True
        table.setRowCount(len(devices))
        for row, device in enumerate(devices):
            is_gemini_discovered = device.get("source") == "gemini_discovered"
            if is_gemini_discovered:
                confidence_display = "AI"
            else:
                confidences = device["confidences"]
                best_confidence = max(confidences) if confidences else 0.0
                confidence_display = f"{best_confidence * 100:.0f}%"
            watts_total = device["watts_active"] * device["count"]
            values = [
                device["display_name"],
                str(device["count"]),
                confidence_display,
                f"{watts_total:.0f} W",
                f"${device['cost_per_year_usd']:.2f}",
            ]
            # Gemini's optional per-instance type/model detail (e.g. "55-inch
            # wall-mounted LED TV") surfaces as a tooltip on the device-name
            # cell rather than a new column, so the fixed-size table layout
            # never needs to change width just because AI notes are present.
            notes = [n for n in device.get("notes", []) if n]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 0 and notes:
                    item.setToolTip("\n".join(notes))
                table.setItem(row, col, item)

    def _update_top_drains(self, devices: List[Dict[str, object]]) -> None:
        # devices is already sorted by kwh_per_year descending (energy_estimator.estimate_room).
        self._top_drains_list.clear()
        if not devices:
            self._top_drains_list.addItem(_placeholder_item("No devices detected yet."))
            return
        for device in devices[:ROOMSCAN_TOP_DRAINS_COUNT]:
            self._top_drains_list.addItem(
                f"{device['display_name']}: {device['kwh_per_year']:.0f} kWh/yr (${device['cost_per_year_usd']:.2f})"
            )

    def _update_recommendations(self, devices: List[Dict[str, object]], totals: Dict[str, object]) -> None:
        self._recommendations_list.clear()
        context = {"avg_brightness": self._last_avg_brightness}
        for suggestion in generate_recommendations(devices, totals, context):
            item = QListWidgetItem(suggestion)
            if suggestion == NO_DEVICES_MESSAGE:
                item.setForeground(QColor(RS_MUTED))
            self._recommendations_list.addItem(item)

    def _update_gemini_flag(self, rejected_classes: List[str]) -> None:
        if not rejected_classes:
            self._gemini_flag_label.hide()
            return
        self._gemini_flag_label.setText(
            "⚠ Gemini auto-corrected a likely misclassification: " + ", ".join(rejected_classes)
        )
        self._gemini_flag_label.show()

    def _update_gemini_discovered(self, discovered_devices: List[Dict[str, object]]) -> None:
        if not discovered_devices:
            self._gemini_discovered_label.hide()
            return
        names = ", ".join(d["name"] for d in discovered_devices)
        self._gemini_discovered_label.setText(f"✨ AI-discovered devices added to list: {names}")
        self._gemini_discovered_label.show()

    def _update_ai_indicator(self, state: Dict[str, object]) -> None:
        """Visual proof-of-life for the background Gemini live pass: shows
        "verifying" while a call is actually in flight, then flashes "check
        complete" for ROOMSCAN_AI_FLASH_DURATION_S once it finishes, so
        every AI pass is visibly announced rather than only showing up as a
        side effect (a note/rejection banner) users might not connect to AI
        having run at all."""
        if not state.get("gemini_verification_enabled"):
            self._ai_status_label.hide()
            return

        if state.get("gemini_pass_active"):
            self._ai_status_label.setText("✨ Gemini AI checking detections...")
            self._ai_status_label.setStyleSheet(f"color: {RS_AMBER}; font-size: 12px; font-weight: 600;")
            self._ai_status_label.show()
            return

        last_ts = state.get("gemini_last_pass_ts")
        if last_ts is not None and last_ts != self._last_seen_gemini_pass_ts:
            self._last_seen_gemini_pass_ts = last_ts
            self._ai_flash_until = time.monotonic() + ROOMSCAN_AI_FLASH_DURATION_S

        if self._ai_flash_until is not None and time.monotonic() < self._ai_flash_until:
            self._ai_status_label.setText("✓ Gemini AI check complete")
            self._ai_status_label.setStyleSheet(f"color: {RS_GOOD}; font-size: 12px; font-weight: 600;")
            self._ai_status_label.show()
        else:
            self._ai_status_label.hide()

    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self._controller is not None and self._controller.running:
            self._controller.stop()
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyQt5 live dashboard for the RoomScan energy audit.")
    parser.add_argument("--device-ip", help="Glasses IPv4 (only with --start-streaming).")
    parser.add_argument("--start-streaming", action="store_true", help="Start streaming via DeviceClient first.")
    parser.add_argument("--interface", choices=["wifi", "usb"], default="wifi", help="Live streaming interface.")
    parser.add_argument("--profile", default=DEFAULT_STREAM_PROFILE, help="Live streaming profile.")
    parser.add_argument(
        "--persistent-certs", action="store_true",
        help="Use installed persistent streaming certificates (via `aria streaming install-certs`) instead of "
        "ephemeral certificates -- required when subscribing to a stream already started outside this dashboard.",
    )
    parser.add_argument("--local-certs-dir", help="Optional persistent-cert directory override (only with --persistent-certs).")
    parser.add_argument(
        "--out", default=str(ROOMSCAN_OUTPUT_DIR),
        help="Base output directory; each saved report gets its own <room>_<timestamp> subfolder.",
    )
    parser.add_argument(
        "--debug-camera-only", action="store_true",
        help="Debug mode: show the live RGB feed with detection completely disabled (no EnergyDetector/YOLO "
        "is even constructed). Use this to isolate a 'nothing shows up' problem to either camera "
        "delivery/rendering (still broken with this flag on) or the detector (works with this flag on, "
        "breaks without it). Saved reports in this mode will have zero devices -- turn the flag back off "
        "once the camera side is confirmed working.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    window = RoomScanDashboard(args)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
