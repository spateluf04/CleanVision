"""Central configuration values for the Aria ML project.

This module stores shared constants used by collection scripts, live inference,
the training dashboard, and the bridge process. It depends on
``pathlib.Path`` and produces immutable configuration values that other modules
import instead of hardcoding runtime parameters.
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# Shared labels / model dimensions
LABELS = [chr(ord("A") + idx) for idx in range(26)]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}
INPUT_POINTS = 64
INPUT_SIZE = 2
NUM_CLASSES = 26

# Trajectory capture defaults
TRAJECTORY_MOVEMENT_THRESHOLD_PX = 25.0
TRAJECTORY_DWELL_SECONDS = 1.2
TRAJECTORY_MIN_POINTS = 5
TRAJECTORY_MIN_DURATION_SECONDS = 0.8
TRAJECTORY_NORMALIZED_POINTS = 64
MAX_TRAJECTORY_POINTS = 500
TRAJECTORY_MIN_SPAN_PX = 15.0

# Data collection / dataset paths
RGB_LABEL_CANDIDATES = [
    "camera-rgb",
    "camera-rgb+",
    "rgb",
]
TARGET_SAMPLES_PER_LETTER = 50
CSV_SAVE_BUFFER_SIZE = 10
DATASET_CSV_PATH = BASE_DIR / "aria_letter_trajectories.csv"
SAMPLE_METADATA_PATH = BASE_DIR / "sample_metadata.json"
TRAINING_STATE_PATH = BASE_DIR / "training_state.json"
TRAINING_HISTORY_EXPORT_PATH = BASE_DIR / "training_history_export.csv"
DEFAULT_MODEL_OUTPUT = "letter_model.pt"
DEFAULT_MODEL_OUTPUT_PATH = BASE_DIR / DEFAULT_MODEL_OUTPUT

# MediaPipe / hand tracking
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_MODEL_PATH = BASE_DIR / "hand_landmarker.task"
MEDIAPIPE_STATIC_IMAGE_MODE = False
MEDIAPIPE_MODEL_COMPLEXITY = 1
MEDIAPIPE_MAX_NUM_HANDS = 2
MEDIAPIPE_SINGLE_HAND_MAX_NUM_HANDS = 1
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5
MEDIAPIPE_HAND_MIN_DETECTION_CONFIDENCE = 0.55
MEDIAPIPE_HAND_MIN_PRESENCE_CONFIDENCE = 0.5
MEDIAPIPE_HAND_MIN_TRACKING_CONFIDENCE = 0.5
MEDIAPIPE_IDLE_FRAME_SKIP = 2

# Eye / pupil processing
EYE_RESIZE_WIDTH = 64
EYE_RESIZE_HEIGHT = 48
EYE_REFLECTION_THRESHOLD = 220
EYE_REFLECTION_REPLACEMENT_VALUE = 128
EYE_DARK_THRESHOLD = 45
EYE_MORPH_KERNEL_SIZE = 3
EYE_MIN_MOMENT_AREA = 100

# Gaze / detection
DEFAULT_YOLO_MODEL_SIZE = "yolo11n.pt"
DEFAULT_YOLO_DEVICE = "mps"
DEFAULT_YOLO_CONF_THRESHOLD = 0.4
GAZE_DEFAULT_SCALE = 0.7
GAZE_CALIBRATION_SAMPLE_COUNT = 15
GAZE_CALIBRATION_SLEEP_SECONDS = 0.1

# LSTM / Transformer training
HIDDEN_SIZE = 128
NUM_LAYERS = 2
TRANSFORMER_EMBED_DIM = 64
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
TRANSFORMER_FF_DIM = 128
TRAIN_SPLIT_RATIO = 0.8
DEFAULT_TRAIN_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_RANDOM_SEED = 42

# Streaming / networking
WS_HOST = "localhost"
WS_PORT = 8765
WS_URL = f"ws://{WS_HOST}:{WS_PORT}"
BRIDGE_SEND_QUEUE_MAXSIZE = 50
BRIDGE_THREADPOOL_WORKERS = 4
STREAM_MESSAGE_QUEUE_SIZE = 1
BRIDGE_DEFAULT_STREAM_INTERFACE = "wifi"
LIVE_INFERENCE_DEFAULT_STREAM_INTERFACE = "usb"
DEFAULT_STREAM_PROFILE = "profile18"
LIVE_RGB_QUEUE_MAXSIZE = 2
LIVE_RGB_QUEUE_TIMEOUT_SECONDS = 0.25
LIVE_RGB_FRAME_INTERVAL_MS = 33.0
LIVE_RGB_FRAME_INTERVAL_SECONDS = 0.033
BRIDGE_RETRY_SECONDS = 2.0
TRACKER_PERF_LOG_INTERVAL = 60
YOLO_PERF_LOG_INTERVAL = 100

# Sensor throttles / frame sizes
BRIDGE_RGB_THROTTLE_MS = 66.0
BRIDGE_SLAM_THROTTLE_MS = 100.0
BRIDGE_ET_THROTTLE_MS = 100.0
BRIDGE_IMU_THROTTLE_MS = 16.0
BRIDGE_MAG_THROTTLE_MS = 200.0
BRIDGE_BARO_THROTTLE_MS = 100.0
BRIDGE_AUDIO_THROTTLE_MS = 66.0
BRIDGE_RGB_FRAME_SIZE = (480, 480)
BRIDGE_SLAM_FRAME_SIZE = (320, 240)
BRIDGE_ET_FRAME_SIZE = (160, 120)
LIVE_INFERENCE_FRAME_SIZE = (640, 640)
BRIDGE_JPEG_QUALITY = 55
BRIDGE_STATS_LOOP_INTERVAL_SECONDS = 5.0
AUDIO_CHANNEL_TARGET = 7
AUDIO_NORMALIZATION_DIVISOR = 32768.0
SEA_LEVEL_PRESSURE_HPA = 1013.25
ALTITUDE_SCALE_METERS = 44330.0

# Blink detection
BLINK_OPEN_BASELINE = 0.80
BLINK_PREV_SCORE = 0.80
BLINK_DELTA_THRESHOLD = 0.10
BLINK_RAPID_DROP_THRESHOLD = 0.01
BLINK_RAPID_DROP_WINDOW_MS = 250.0
BLINK_COOLDOWN_MS = 40.0
BLINK_BASELINE_DECAY = 0.999
BLINK_OPEN_THRESHOLD_OFFSET = 0.01

# Dashboard UI sizes / colors
WINDOW_WIDTH = 1520
WINDOW_HEIGHT = 920
SIDEBAR_WIDTH = 280
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TRAJ_WIDTH = 640
TRAJ_HEIGHT = 200
RIGHT_WIDTH = 420
RIGHT_HEIGHT = 710
BOTTOM_HEIGHT = 90
PANEL_BG = "#0f1720"
SURFACE_BG = "#162231"
ACCENT = "#3dd6ff"
SUCCESS = "#4cf5a0"
TEXT = "#edf6ff"
MUTED = "#8aa3b9"
BORDER = "#253649"
WARNING = "#f5b942"
DANGER = "#ff6b6b"
BLUE_RING = (255, 170, 60)
GREEN_RING = (80, 245, 160)
RED_RING = (80, 80, 255)
LOST_HAND_BUFFER_DOT_COLOR = (0, 220, 255)
UI_TRAJECTORY_GRID_X_STEP = 80
UI_TRAJECTORY_GRID_Y_STEP = 50
UI_CAPTURE_COUNTDOWN_SECONDS = 3
UI_DRAW_BANNER_SECONDS = 0.45
UI_SAVE_FLASH_SECONDS = 1.5
UI_RESET_FLASH_SECONDS = 1.2
UI_INFO_FLASH_SECONDS = 1.0
UI_STABILIZATION_FRAMES = 10
UI_SMOOTHING_WINDOW = 5
UI_LOST_HAND_BUFFER_FRAMES = 8
UI_BEEP_SAMPLE_RATE = 22050
UI_BEEP_DURATION_SECONDS = 0.12
UI_BEEP_FREQUENCY_HZ = 880
UI_BEEP_VOLUME = 0.25
HEATMAP_GREEN_THRESHOLD = 0.90
HEATMAP_YELLOW_THRESHOLD = 0.70

# EMNIST pretraining / personal calibration (parallel path to the CSV+LSTM pipeline)
EMNIST_DATA_DIR = BASE_DIR / "emnist_data"
BASE_MODEL_PATH = BASE_DIR / "base_model.pt"
PERSONAL_MODEL_PATH = BASE_DIR / "personal_model.pt"
CALIBRATION_SET_DIR = BASE_DIR / "calibration_set"
RASTER_IMAGE_SIZE = 28
RASTER_STROKE_THICKNESS_PX = 2
RASTER_GAUSSIAN_BLUR_SIGMA = 0.6
CNN_CONV1_CHANNELS = 32
CNN_CONV2_CHANNELS = 64
CNN_FC_HIDDEN = 128
PRETRAIN_EPOCHS = 15
PRETRAIN_BATCH_SIZE = 128
PRETRAIN_LEARNING_RATE = 1e-3
CALIBRATION_SAMPLES_PER_LETTER = 8
CALIBRATION_FINE_TUNE_EPOCHS = 20
CALIBRATION_FINE_TUNE_LR = 1e-4

# Undistortion
ARIA_CAMERA_CALIBRATION_ENV_VAR = "ARIA_CAMERA_CALIB_JSON"
ARIA_CAMERA_CALIBRATION_PATH = BASE_DIR / "aria_camera_calibration.json"
UNDISTORT_FOCAL_SCALE = 0.72
UNDISTORT_DISTORTION = (0.18, 0.05, 0.0, 0.0)

# Capture / healthcheck (aria_capture.py, capture_healthcheck.py)
CAPTURE_SOURCE_VRS = "vrs"
CAPTURE_SOURCE_LIVE = "live"
# Canonical stream labels consumers subscribe to. The Gen 1 eye-tracking camera
# is ONE physical stream ("camera-et") holding both eyes side by side; capture
# splits it into the two camera-et-* labels below.
CAPTURE_IMAGE_LABELS = (
    "camera-rgb",
    "camera-slam-left",
    "camera-slam-right",
    "camera-et-left",
    "camera-et-right",
)
CAPTURE_MOTION_LABELS = ("imu-right", "imu-left", "mag0", "baro0")
CAPTURE_ALL_LABELS = CAPTURE_IMAGE_LABELS + CAPTURE_MOTION_LABELS
CAPTURE_ET_COMBINED_LABEL = "camera-et"
# VRS label aliases per physical stream (labels vary slightly across firmware
# and tooling versions; resolution is by label, never by hardcoded stream id).
CAPTURE_VRS_LABEL_ALIASES = {
    "camera-rgb": tuple(RGB_LABEL_CANDIDATES),
    "camera-slam-left": ("camera-slam-left", "slam-left"),
    "camera-slam-right": ("camera-slam-right", "slam-right"),
    CAPTURE_ET_COMBINED_LABEL: ("camera-et", "camera-eyetracking", "camera-et-left"),
    "imu-right": ("imu-right",),
    "imu-left": ("imu-left",),
    "mag0": ("mag0", "magnetometer"),
    "baro0": ("baro0", "barometer"),
}
CAPTURE_MOTION_DEQUE_MAXLEN = 2000
CAPTURE_DISPATCH_IDLE_SLEEP_SECONDS = 0.001
CAPTURE_VRS_BACKPRESSURE_SLEEP_SECONDS = 0.002
# Nominal per-stream rate windows (Hz) for the Aria Gen 1 sensors.
EXPECTED_STREAM_RATES_HZ = {
    "camera-rgb": (10.0, 30.0),
    "camera-slam-left": (10.0, 30.0),
    "camera-slam-right": (10.0, 30.0),
    "camera-et-left": (30.0, 90.0),
    "camera-et-right": (30.0, 90.0),
    "imu-right": (850.0, 1150.0),
    "imu-left": (680.0, 920.0),
    "mag0": (7.0, 13.0),
    "baro0": (35.0, 65.0),
}
HEALTHCHECK_DURATION_SECONDS = 30.0
HEALTHCHECK_OUTPUT_DIR = BASE_DIR / "healthcheck_out"
HEALTHCHECK_MAX_LIVE_SKEW_MS = 100.0
HEALTHCHECK_RATE_TOLERANCE = 0.10

# Energy audit (energy_detector.py, energy_estimator.py, roomscan.py, energy_report.py)
ENERGY_YOLO_WEIGHTS = "yolov8n.pt"
# 0.5 verified on sample.vrs: keeps real TV (0.94) / clock (0.75), rejects a
# dark-armchair-as-TV false positive (0.44).
ENERGY_DETECT_CONFIDENCE = 0.5
# RGB frames are sampled at this rate (device time) for detection; the camera
# runs faster but per-frame YOLO on every frame buys nothing for a room scan.
ENERGY_FRAME_SAMPLE_HZ = 2.0
# Minimum box area as a fraction of the frame; rejects sliver false positives.
ENERGY_MIN_BOX_AREA_FRAC = 0.001

# Temporal stabilization (energy_detector.DetectionStabilizer): smooths raw
# per-frame detections before they reach ApplianceScanAggregator, so a single
# noisy/missed frame can't flicker the live view or permanently inflate the
# max-simultaneous count.
ENERGY_STABILIZE_WINDOW_SECONDS = 3.0        # rolling window used to confirm a detection is real
ENERGY_STABILIZE_MIN_HITS = 2                # hits needed within the window before a track counts
ENERGY_STABILIZE_MAX_MISS_SECONDS = 1.5      # grace period a track survives with zero matching detections
ENERGY_STABILIZE_IOU_MATCH_THRESHOLD = 0.3   # min IOU to match a detection to an existing track
ENERGY_DUPLICATE_BOX_IOU_THRESHOLD = 0.6     # same-frame same-class boxes above this IOU count as one object
# COCO classes treated as energy-drawing appliances. Keys are exact YOLO/COCO
# class names; per-class typical draw and daily usage assumptions drive the
# kWh estimate (hackathon-grade priors, not measurements).
ENERGY_CATALOG = {
    "tv": {"display": "Television", "watts_active": 100.0, "watts_standby": 2.0, "hours_per_day": 5.0},
    "laptop": {"display": "Laptop", "watts_active": 50.0, "watts_standby": 1.0, "hours_per_day": 6.0},
    "refrigerator": {"display": "Refrigerator", "watts_active": 150.0, "watts_standby": 0.0, "hours_per_day": 8.0},
    "microwave": {"display": "Microwave", "watts_active": 1100.0, "watts_standby": 3.0, "hours_per_day": 0.25},
    "oven": {"display": "Oven", "watts_active": 2300.0, "watts_standby": 2.0, "hours_per_day": 0.5},
    "toaster": {"display": "Toaster", "watts_active": 900.0, "watts_standby": 0.0, "hours_per_day": 0.1},
    "hair drier": {"display": "Hair dryer", "watts_active": 1500.0, "watts_standby": 0.0, "hours_per_day": 0.1},
    "cell phone": {"display": "Phone (charging)", "watts_active": 5.0, "watts_standby": 0.5, "hours_per_day": 2.0},
    "clock": {"display": "Clock", "watts_active": 2.0, "watts_standby": 0.0, "hours_per_day": 24.0},
    # Forward-looking entries for energy_recommendations.py's cooling-inefficiency
    # rule (fan + AC running together). Not COCO classes, so the stock
    # COCO-pretrained yolov8n.pt weights energy_detector.py loads today can
    # never emit "fan"/"air conditioner" -- these exist so the catalog, the
    # estimator, and the recommendation rule are all ready the moment a
    # fine-tuned/custom detector (or an ML/LLM recommendation layer) adds them.
    "fan": {"display": "Fan", "watts_active": 50.0, "watts_standby": 0.0, "hours_per_day": 8.0},
    "air conditioner": {"display": "Air conditioner", "watts_active": 1500.0, "watts_standby": 1.0, "hours_per_day": 6.0},
}
ENERGY_COST_PER_KWH_USD = 0.17
ROOMSCAN_OUTPUT_DIR = BASE_DIR / "roomscan_out"
ROOMSCAN_REPORT_JSON_NAME = "roomscan_report.json"
ROOMSCAN_REPORT_HTML_NAME = "roomscan_report.html"
ROOMSCAN_CROP_DIR_NAME = "crops"
# Session index (energy_sessions.py): one small JSON file at the root of
# ROOMSCAN_OUTPUT_DIR that indexes every saved scan's summary + a pointer
# back to its own <out_dir>/roomscan_report.{json,html} -- it never replaces
# or duplicates the per-session report/crops, only makes them listable.
ROOMSCAN_SESSIONS_INDEX_NAME = "roomscan_sessions.json"
ROOMSCAN_SESSIONS_SUMMARY_CSV_NAME = "roomscan_sessions_summary.csv"
# Live mode scans for this long by default; VRS mode consumes the whole file.
ROOMSCAN_LIVE_DURATION_SECONDS = 60.0
# Default interval between live-dashboard snapshot ticks (roomscan_live.py),
# decoupled from ENERGY_FRAME_SAMPLE_HZ so UI push rate != detection rate.
ROOMSCAN_LIVE_TICK_SECONDS = 1.0

# Gemini vision recommendations (energy_gemini.py) -- optional enhancement over
# energy_recommendations.py's rule engine. Enabled only when GEMINI_API_KEY is
# set in the environment (never hardcode a key here); falls back to the rule
# engine on any failure so a live demo never hard-depends on the network. Only
# wired into roomscan.py:build_report() (the once-per-finished-scan report) --
# the live dashboard's per-tick recommendations panel and its instant
# Stop-Scan summary dialog intentionally stay rule-based-only.
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
GEMINI_MODEL = "gemini-flash-latest"
GEMINI_MAX_CROPS = 6              # cap images sent per request (cost/latency)
GEMINI_MAX_RECOMMENDATIONS = 5
GEMINI_TIMEOUT_SECONDS = 12.0

# Live-scan Gemini verification + discovery (energy_gemini.run_live_scan_pass,
# via roomscan_live.py's background pass thread). Same GEMINI_API_KEY gate and
# fallback-on-any-failure philosophy as the recommendations feature above --
# see energy_gemini.py module docstring. Interval is kept comfortably above
# GEMINI_TIMEOUT_SECONDS so a slow call never overlaps the next pass.
GEMINI_LIVE_PASS_INTERVAL_SECONDS = 13.0
GEMINI_VERIFY_MAX_CROPS = 4        # unverified candidates re-checked per pass
GEMINI_MAX_DISCOVERED = 4          # AI-discovered non-catalog names accepted per pass
GEMINI_DISCOVERY_MAX_DIM = 768     # full-frame longest-side downscale before upload
GEMINI_NOTE_MAX_CHARS = 80         # cap on a verified slot's type/model note (UI tooltip length)
# Pricing a Gemini-discovered (non-catalog) device: Gemini is asked to also
# estimate typical active wattage + daily usage hours per discovered item
# (energy_gemini.run_live_scan_pass) so roomscan.py:merge_discovered_devices()
# can price it the same way as a catalog device and fold it into the same
# device list/totals, instead of only an unpriced call-out. Defaults/clamp
# are defensive since these numbers are a vision-model guess, not a
# lookup-table measurement.
GEMINI_DISCOVERY_DEFAULT_WATTS = 15.0
GEMINI_DISCOVERY_DEFAULT_HOURS_PER_DAY = 4.0
GEMINI_DISCOVERY_MAX_WATTS = 5000.0
# Gemini is also asked to count how many individual instances of a discovered
# type are visible in one frame (e.g. 3 separate ceiling lights), so pricing
# scales with the real fixture count instead of always treating a discovered
# type as a single unit. GEMINI_DISCOVERY_MAX_COUNT is a sanity clamp against
# a runaway/garbled vision-model count, not a realistic expected count.
GEMINI_DISCOVERY_DEFAULT_COUNT = 1
GEMINI_DISCOVERY_MAX_COUNT = 50
# When Gemini's per-item estimated_watts is missing/invalid for a discovered
# lighting fixture, fall back to a bulb-type-specific reference wattage
# (looked up from the fixture's own description, e.g. "LED ceiling light")
# instead of the flat GEMINI_DISCOVERY_DEFAULT_WATTS -- a much closer guess
# than one generic default across LED/CFL/halogen/fluorescent/incandescent.
# Checked in this order (first substring match wins), so more specific terms
# ("cfl"/"compact fluorescent") are listed before the generic "fluorescent".
GEMINI_BULB_TYPE_WATTS = {
    "led": 9.0,
    "cfl": 14.0,
    "compact fluorescent": 14.0,
    "halogen": 43.0,
    "fluorescent": 32.0,
    "incandescent": 60.0,
}

# RoomScan live dashboard (PyQt5, roomscan_dashboard.py). Reuses the shared
# dark theme palette above (PANEL_BG/SURFACE_BG/ACCENT/SUCCESS/TEXT/MUTED/
# BORDER) for visual consistency with training_dashboard.py; these are
# layout sizes specific to RoomScan's different panel content.
ROOMSCAN_DASHBOARD_WINDOW_WIDTH = 1440
ROOMSCAN_DASHBOARD_WINDOW_HEIGHT = 900
ROOMSCAN_DASHBOARD_SIDEBAR_WIDTH = 260
ROOMSCAN_DASHBOARD_RIGHT_WIDTH = 320
ROOMSCAN_DASHBOARD_CAMERA_WIDTH = 640
ROOMSCAN_DASHBOARD_CAMERA_HEIGHT = 480
ROOMSCAN_DASHBOARD_BOTTOM_HEIGHT = 300
ROOMSCAN_DASHBOARD_FRAME_POLL_MS = 100
# If latest_frame() is still None this long after Start Scan, the camera panel
# escalates from the routine "Waiting for RGB frames..." message to a visible
# stale-connection warning instead of waiting silently forever.
ROOMSCAN_DASHBOARD_STALE_FRAME_TIMEOUT_S = 8.0
ROOMSCAN_TOP_DRAINS_COUNT = 3
# Folder-naming contract for a saved live session's <room-slug>_<timestamp>
# output directory (roomscan_live.py:session_out_dir), matching the
# session_id shape energy_sessions.py documents (e.g. kitchen_20260101_120000).
ROOMSCAN_SESSION_DIR_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
ROOMSCAN_COMPARE_DIALOG_WIDTH = 760
ROOMSCAN_COMPARE_DIALOG_HEIGHT = 480
ROOMSCAN_STATUS_DOT_SIZE_PX = 14
ROOMSCAN_SUMMARY_DIALOG_WIDTH = 480
ROOMSCAN_SUMMARY_DIALOG_HEIGHT = 460
# How many suggestions the end-of-scan "Room Efficiency Summary" dialog shows
# (the live "Ways To Save Energy" panel during a scan shows the full list;
# this trims to the highest-value few for a clean final-summary read).
ROOMSCAN_SUMMARY_RECOMMENDATIONS_COUNT = 3
# Hackathon-grade efficiency-rating cutoffs (estimated annual electricity
# cost of just the devices this scan found) for the end-of-scan summary
# badge -- same "typical-draw priors, not measurements" caveat as
# ENERGY_CATALOG; tune freely without touching roomscan_dashboard.py.
ROOMSCAN_EFFICIENCY_GOOD_MAX_COST_USD = 150.0
ROOMSCAN_EFFICIENCY_FAIR_MAX_COST_USD = 400.0
# Live camera-view bounding-box overlay (roomscan_dashboard.py:_poll_frame,
# drawn from LiveScanController.latest_detections()). RGB (not BGR) since the
# overlay is drawn directly on the same upright RGB frame array the label
# displays -- cv2 draw calls don't care about channel order, only that the
# color tuple and the image agree.
ROOMSCAN_DETECTION_BOX_COLOR_RGB = (61, 214, 255)  # matches ACCENT
ROOMSCAN_DETECTION_BOX_THICKNESS = 2
# "AI just ran" flash indicator under the camera view (roomscan_dashboard.py):
# how long the "Gemini AI check complete" message stays visible after a live
# verification/discovery pass finishes, before the label hides itself again.
ROOMSCAN_AI_FLASH_DURATION_S = 2.5


if __name__ == "__main__":
    print(f"Aria ML config loaded from {BASE_DIR}")
