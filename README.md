# Meta Project Aria Air Writing Toolkit

## Project Overview

This repository contains a full research and prototyping toolkit for collecting,
training, and running air-writing recognition with Meta Project Aria Gen 1
glasses. The codebase supports:

- VRS-based offline fingertip trajectory collection
- live Aria RGB streaming and WebSocket bridging
- gaze-aware object detection with YOLO and MediaPipe
- letter trajectory normalization and supervised model training
- live letter inference from fingertip motion
- a fixed-layout PyQt5 training dashboard for sample capture and review

The repository is organized around a single end-to-end goal: capture air-drawn
letters reliably, turn them into normalized 2D trajectories, train a sequence
model, and deploy that model back into a live Aria-powered interface.

## Hardware Requirements

- Meta Project Aria Gen 1 glasses
- A host machine capable of running the Aria SDK
- Apple Silicon Mac recommended for MPS acceleration
- USB cable for pairing and USB streaming
- Optional Wi-Fi network shared by the host and the glasses

## Software Requirements

- Python 3.9+
- Meta Project Aria SDK and `projectaria_tools`
- OpenCV
- NumPy
- PyTorch
- MediaPipe
- Ultralytics YOLO
- PyQt5
- `websockets`

## Installation

### 1. Activate the Aria virtual environment

```bash
source ~/aria-venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install numpy opencv-python torch mediapipe ultralytics PyQt5 websockets
```

If you are using the dashboard audio cues:

```bash
pip install pygame
```

### 3. Make sure streaming certificates are installed

```bash
aria streaming install-certs
```

## Project Structure

```text
/Users/keyurpatel/Desktop/aria meta/
  bridge.py
  config.py
  gaze_detector.py
  live_letter_inference.py
  logging_utils.py
  train_letter_lstm.py
  training_dashboard.py
  vrs_index_fingertip_tracker.py
  README.md
  aria_letter_trajectories.csv
  sample_metadata.json
  training_state.json
  letter_model.pt
```

### File Roles

- `config.py`
  Shared constants for thresholds, paths, model settings, UI sizes, and buffer
  sizes.
- `logging_utils.py`
  Shared logging configuration.
- `vrs_index_fingertip_tracker.py`
  Offline VRS trajectory collector and trajectory normalization utilities.
- `train_letter_lstm.py`
  Sequence-model training entrypoint for normalized trajectory CSV data.
- `live_letter_inference.py`
  Real-time air-letter prediction from the live Aria RGB stream.
- `gaze_detector.py`
  Gaze projection, YOLO object detection, and hand gesture detection.
- `bridge.py`
  Aria sensor streaming bridge that publishes RGB, telemetry, blink, and
  detection data over WebSockets.
- `training_dashboard.py`
  PyQt5 desktop interface for target selection, live capture, review, and
  training state inspection.

## How To Collect Data

### Option A: Collect from a VRS recording

```bash
cd "/Users/keyurpatel/Desktop/aria meta"
source ~/aria-venv/bin/activate
python3 vrs_index_fingertip_tracker.py /path/to/recording.vrs --output-csv aria_letter_trajectories.csv
```

Controls:

- Press `A-Z` to arm a target letter
- Draw the letter in the air
- Pause to let the dwell logic save the sample
- Press `C` to clear the current capture
- Press `Q` to quit

### Option B: Collect from the live dashboard

1. Start Aria streaming
2. Start the bridge
3. Launch the dashboard
4. Select a letter in the sidebar
5. Press `Capture Letter`
6. Draw the letter after the countdown
7. End the capture from the UI and save the sample

## How To Start Streaming

### Persistent-certificate flow

```bash
source ~/aria-venv/bin/activate
aria streaming stop
aria recording stop
aria streaming install-certs
aria streaming start --interface usb --profile profile18
```

If USB is unreliable on your device, use Wi-Fi instead:

```bash
source ~/aria-venv/bin/activate
aria streaming stop
aria recording stop
aria streaming install-certs
aria --device-ip <GLASSES_IP> streaming start --interface wifi --profile profile18
```

## How To Run The Bridge

```bash
cd "/Users/keyurpatel/Desktop/aria meta"
source ~/aria-venv/bin/activate
python3 bridge.py --persistent-certs
```

The bridge publishes data on:

- `ws://localhost:8765`

## How To Run The Dashboard

```bash
cd "/Users/keyurpatel/Desktop/aria meta"
source ~/aria-venv/bin/activate
python3 training_dashboard.py
```

The dashboard expects the bridge to already be running.

## How To Train A Model

```bash
cd "/Users/keyurpatel/Desktop/aria meta"
source ~/aria-venv/bin/activate
python3 train_letter_lstm.py aria_letter_trajectories.csv --epochs 50 --model-out letter_model.pt
```

The trainer:

- reads normalized trajectories from CSV
- performs an 80/20 split
- trains an LSTM baseline and a Transformer encoder
- saves the best Transformer checkpoint to `letter_model.pt`

## How To Run Live Inference

Make sure streaming and the bridge are already running, then launch live
inference:

```bash
cd "/Users/keyurpatel/Desktop/aria meta"
source ~/aria-venv/bin/activate
python3 live_letter_inference.py --model-path letter_model.pt --persistent-certs
```

Controls:

- `Q` quits
- `C` clears the last 3 predicted letters

## How To Serve The Browser Dashboard

```bash
cd "/Users/keyurpatel/Desktop/aria meta"
python3 -m http.server 8080
```

Then open:

- `http://localhost:8080/`
- `http://localhost:8080/game.html`

## Common Issues

- `The SDK is not paired with the device`
  Re-run:
  ```bash
  aria auth pair
  aria auth check
  ```

- `Streaming error (9) Failed to start recording`
  Stop recording and streaming, then retry after a short wait:
  ```bash
  aria streaming stop
  aria recording stop
  ```

- `No devices found over USB`
  Re-seat the cable, avoid hubs, and verify:
  ```bash
  aria device list
  ```

- `ModuleNotFoundError`
  Make sure the venv is active:
  ```bash
  source ~/aria-venv/bin/activate
  ```

## Outputs Produced By This Project

- `aria_letter_trajectories.csv`
  Normalized trajectory dataset with labels and 64 `(x, y)` points.
- `sample_metadata.json`
  Human-friendly metadata for review mode and trajectory thumbnails.
- `training_state.json`
  Persistent dashboard state, counts, and training history.
- `letter_model.pt`
  Saved PyTorch checkpoint for the trained letter classifier.

## Notes For Research Use

- The tracker and collector use configurable movement and dwell thresholds from
  `config.py`.
- The bridge and gaze detector can be profiled independently from the training
  pipeline.
- The dashboard is designed for iterative collection, review, retraining, and
  error analysis across letters.
