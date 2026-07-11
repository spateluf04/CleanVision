# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Meta Project Aria Gen 1 air-writing recognition toolkit, plus a newer Aria-based "energy project" data-capture foundation. Two independent subsystems share the same repo, config, and logging conventions:

1. **Air-writing pipeline**: VRS/live capture of the index fingertip -> normalized 2D trajectories -> LSTM/Transformer or EMNIST-CNN classifier -> live letter prediction, with a PyQt5 dashboard for collection/review/training.
2. **RoomScan energy audit**: `aria_capture.py` (dual-backend VRS + live sensor capture with ring-buffered fan-out) + `capture_healthcheck.py` (verification harness), consumed by the energy pipeline: `energy_detector.py` (YOLOv8 appliance detection) -> `energy_estimator.py` (catalog kWh/cost math) -> `roomscan.py` (orchestrator CLI) -> `energy_report.py` (self-contained HTML report).

Primary runtime is a Mac (`~/aria-venv`, Aria SDK paired to the glasses there). On Windows, there is no native Python and `projectaria_tools`/`aria.sdk` have no Windows wheels — use WSL Ubuntu instead (see below).

## Commands

### Environment

```bash
source ~/aria-venv/bin/activate
pip install numpy opencv-python torch mediapipe ultralytics PyQt5 websockets pygame
```
Aria streaming certs (live modes only): `aria streaming install-certs`.

On Windows (no native Aria wheels), run everything inside WSL Ubuntu with a **python3.12** venv (the distro default python3 may be too new for available wheels):
```bash
wsl -d Ubuntu -- bash -c 'cd "/mnt/c/.../RoomScan" && ~/aria-venv/bin/python <script>.py ...'
```

### Tests

```bash
python -m pytest tests/
```
Single test: `python -m pytest tests/test_system.py::CsvOperationTests::test_csv_operations`. Tests import project modules directly (no package install) via a `sys.path` insert in `tests/test_system.py`, so always run pytest from the repo root.

### Data collection (offline, from a VRS file)

```bash
python3 vrs_index_fingertip_tracker.py /path/to/recording.vrs --output-csv aria_letter_trajectories.csv
```
`A`-`Z` arms a letter, draw in the air, pause to save (dwell-based), `C` clears, `Q` quits.

### Live streaming + bridge + dashboard

```bash
aria streaming stop && aria recording stop && aria streaming install-certs
aria streaming start --interface usb --profile profile18   # or --interface wifi with --device-ip
python3 bridge.py --persistent-certs
python3 training_dashboard.py     # expects the bridge already running
python3 -m http.server 8080       # serves index.html / game.html browser UI
```

### Training

```bash
python3 train_letter_lstm.py aria_letter_trajectories.csv --epochs 50 --model-out letter_model.pt
```
Trains an LSTM baseline and a Transformer encoder side by side (80/20 stratified split); only the Transformer checkpoint is saved (`letter_model.pt`, plus a `letter_model_v{N}.pt` snapshot keyed off `training_state.json["model_version"]`).

### Live inference

```bash
python3 live_letter_inference.py --model-path letter_model.pt --persistent-certs [--engine lstm|raster]
```
`--engine lstm` is the default CSV/LSTM checkpoint path; `--engine raster` uses the parallel EMNIST-pretrain path (`base_model.pt` / `personal_model.pt`).

### EMNIST-pretrain + calibration research path (parallel to the CSV/LSTM pipeline)

```bash
python3 pretrain_emnist.py --epochs 15                      # -> base_model.pt
# per-user calibration draws are captured via training_dashboard.py's "Calibrate" flow,
# which calls calibration.py:fine_tune_personal_model()      # -> personal_model.pt
python3 rasterize.py --debug                                 # rasterized-vs-real-EMNIST sanity check
python3 audit_rasterizer.py                                  # grid audit of every recorded trajectory rasterized
python3 eval_zero_calibration.py                             # base_model.pt accuracy with NO fine-tuning
python3 eval_calibration_offline.py                           # simulated k-shot calibration curves (k=1,3,5,10)
```

### Aria capture healthcheck

```bash
python3 capture_healthcheck.py --vrs /path/to/recording.vrs [--duration 30] [--out healthcheck_out]
python3 capture_healthcheck.py --live [--start-streaming --device-ip <ip> --interface usb|wifi --profile profile18]
```
Exit code 0 only if every expected stream (camera-rgb, camera-slam-left/right, camera-et-left/right, imu-right, imu-left, mag0, baro0) is alive and timestamp-monotonic (plus, in live mode, RGB/IMU skew < 100 ms). Writes one upright sample JPEG per camera to `healthcheck_out/` for visual orientation/eye-split verification.

### RoomScan energy audit

```bash
python3 roomscan.py --vrs /path/to/walkthrough.vrs --room-name "Living room" [--out roomscan_out]
python3 roomscan.py --live [--start-streaming --device-ip <ip> --interface usb|wifi] [--duration 60]
python3 energy_report.py --json roomscan_out/roomscan_report.json   # regenerate HTML only
python3 energy_detector.py --vrs /path/to/recording.vrs             # detection-only debug scan
```
Requires `ultralytics` (auto-downloads `yolov8n.pt` on first run; gitignored via `*.pt`). Outputs `roomscan_report.json`, per-instance crops, and a self-contained `roomscan_report.html` (base64-inlined crops — openable anywhere with no server). Exit 0 if appliances were found, 2 if none, per `roomscan.py:main()`.

## Architecture

### Shared foundations

- **`config.py`** is the single source of truth for every constant (paths, thresholds, model dims, UI sizes, stream labels/rates). New constants belong here, grouped under an existing or new commented section — never hardcode a magic number in a script that already has a `config.py` import block.
- **`logging_utils.get_logger(__name__)`** is used by every module instead of ad hoc logging setup; `setup_logging()` is idempotent and configures the root logger once per process.
- Nearly every entry-point script follows the same shape: module docstring describing purpose/deps/output, `argparse` CLI in `main()`, `if __name__ == "__main__": main()`.

### Aria capture layer (`aria_capture.py`)

Single `AriaCapture` class, two backends behind one callback interface (`source="vrs"` or `source="live"`):
- **VRS backend**: `projectaria_tools.core.data_provider.create_vrs_data_provider`, streams resolved by label (never hardcoded stream IDs, via `CAPTURE_VRS_LABEL_ALIASES` in config), played back with `deliver_queued_sensor_data()` for device-time-ordered interleaving across all modalities (not per-stream index loops).
- **Live backend**: `aria.sdk.StreamingClient` + observer, imported lazily so VRS-only environments never need the Client SDK installed.
- **Gen 1 eye tracking is one physical stream** (`camera-et`) with both eyes side by side in one image; both backends split it at the horizontal midpoint into `camera-et-left` / `camera-et-right` sharing one timestamp.
- **Orientation contract**: images are delivered RAW (un-rotated, native sensor frame — required for calibration/undistortion); `rotate_upright()` (`np.rot90(frame, -1)`) is a separate helper for display-only consumers. Never assume a callback frame is display-oriented.
- **Timestamps**: every sample carries device-time nanoseconds (`capture_timestamp_ns`); wall-clock arrival time is never used for cross-sensor alignment.
- **Fan-out**: producer threads only write buffers (single-slot latest-value for images, `deque(maxlen=2000)` for IMU/mag/baro) — one dispatcher thread invokes subscriber callbacks, so a slow subscriber can never block capture.
- `get_calibration(label)` returns `CameraCalibration` from `provider.get_device_calibration()` in VRS mode; always `None` in live mode (the Client SDK streaming path doesn't deliver device calibration in this build).

`capture_healthcheck.py` is the verification harness for this layer.

### Energy audit pipeline (`roomscan.py` and friends)

`roomscan.py` orchestrates: `AriaCapture` (either backend) -> `energy_detector.scan_capture_rgb()` subscribes to camera-rgb, samples frames at ~2 Hz **device time**, rotates RAW frames upright before YOLO -> `ApplianceScanAggregator` counts instances with the **max-simultaneous rule** (per class, count = most detections seen in any single frame; pan-away/pan-back never double-counts) and keeps the best-confidence crop per instance slot -> `energy_estimator.estimate_room()` maps counts through `ENERGY_CATALOG` priors -> `energy_report.render_html()` writes the self-contained page. Two subtleties: (1) in VRS mode `scan_capture_rgb(pace_playback=True)` subscribes a no-op imu-right consumer to engage the capture layer's backpressure — without it, faster-than-realtime playback plus the drop-stale image slot starves slow YOLO inference down to a few frames per file; live mode must keep `pace_playback=False` (drop-stale is correct there). (2) `ApplianceScanAggregator` and `energy_estimator` are deliberately torch-free (ultralytics is lazily imported inside `EnergyDetector`) so `tests/test_energy.py` runs without YOLO.

### VRS/trajectory pipeline (`vrs_index_fingertip_tracker.py`)

`TrajectoryBuilder` implements dwell-based stroke segmentation (movement threshold resets the anchor; sustained stillness past `dwell_seconds` with enough points/duration emits a finished trajectory) — this same class is reused by `live_letter_inference.py` and `training_dashboard.py`, so changes here ripple through both offline collection and live inference. `normalize_trajectory()` resamples any raw point list to a fixed `(64, 2)` array (arc-length resample -> center -> unit-scale) and is the shared contract between the collector, both classifier architectures, and the rasterizer.

### Live streaming path (`bridge.py`)

Owns the single live `aria.sdk.StreamingClient`/`AriaObserver`, subscribes to RGB + SLAM + ET + IMU + mag + baro + audio, and republishes everything as JSON over a local WebSocket (`ws://localhost:8765`) for the PyQt5 dashboard and browser UI (`index.html`/`game.html`) to consume — those clients never talk to the Aria SDK directly. Runs gaze projection (`gaze_detector.py`: ET pupil-center estimation projected into RGB space, with optional linear calibration) and YOLO/MediaPipe detection per RGB frame, plus dark-pixel-ratio blink detection from the split ET images, all inline in the observer callbacks before publishing. Aria images arrive rotated 90°; `np.rot90(image, -1)` is applied before anything downstream sees a frame (same convention `aria_capture.py` documents but implements as a producer-side raw/display split instead).

### Two parallel classifier architectures — do not conflate them

1. **CSV/LSTM path** (default): `vrs_index_fingertip_tracker.py` writes normalized trajectories to `aria_letter_trajectories.csv` -> `train_letter_lstm.py` trains LSTM + Transformer, saves the Transformer checkpoint -> `live_letter_inference.py --engine lstm` (or `training_dashboard.py`) runs inference directly on the `(64, 2)` point sequence.
2. **EMNIST-pretrain + calibration path** (research, `--engine raster`): `pretrain_emnist.py` trains a small CNN on EMNIST Letters (with an orientation-fixing transpose that `rasterize.py` mirrors) -> `rasterize.py:trajectory_to_image()` renders a normalized trajectory into a 28x28 EMNIST-shaped image -> `calibration.py:fine_tune_personal_model()` freezes the conv layers and fine-tunes only the FC layers on a handful of dashboard-captured calibration draws (`calibration_set/<LETTER>/*.npy`) to produce `personal_model.pt` (falls back to `base_model.pt` if no calibration has run). `eval_zero_calibration.py`, `eval_calibration_offline.py`, and `audit_rasterizer.py` are the offline evaluation/audit harness for this path and share CSV-loading/checkpoint-loading helpers via `eval_utils.py`.

These two paths never share a checkpoint format; `checkpoint["model_type"]` (`"lstm"`, `"transformer"`, or `"cnn"`) plus the fields each `train_model()`/pretrain call stores alongside `model_state_dict` (e.g. `hidden_size`/`num_layers` vs `embed_dim`/`num_heads` vs `conv1_channels`/`fc_hidden`) is how a loader knows which class to reconstruct.

### `training_dashboard.py` (PyQt5, ~2700 lines)

Single-window fixed-layout app: a `BridgeFeedWorker` QThread consumes the bridge's WebSocket JSON stream, a `ProcessingWorker` runs fingertip tracking/undistortion (`AriaPointUndistorter`, driven by `ARIA_CAMERA_CALIB_JSON`/`aria_camera_calibration.json` if present) and trajectory building against live frames, and the main `AriaTrainingDashboard` window wires together capture, review (`ReviewSamplesDialog`, `TrajectoryThumbnail`), per-letter heatmap/stats panels, and both training paths. Persistent state: `training_state.json` (counts, accuracy history, model version — read/written only through `load_training_state()`/`save_training_state()`, which use atomic writes), `sample_metadata.json` (thumbnail/review metadata), and `aria_letter_trajectories.csv` (via `append_sample_record()`, which keeps the CSV and `sample_metadata.json` in sync — don't append to the CSV directly from new code without also updating metadata the same way).

## Conventions worth knowing before editing

- Constants live in `config.py`; add new ones there rather than inlining.
- Every I/O-fallible helper (CSV, JSON, checkpoint) wraps failures in a `RuntimeError`/`FileNotFoundError` with the original exception chained (`raise ... from exc`) rather than letting raw exceptions propagate — match this in new code.
- Stream/camera labels are always resolved dynamically (`get_label_from_stream_id`, alias lists in `config.py`), never hardcoded numeric stream IDs.
- Aria frames are physically rotated 90°; every module that touches raw camera images either documents which orientation it's working in or applies `np.rot90(frame, -1)` before use — check this explicitly when adding a new image consumer.
