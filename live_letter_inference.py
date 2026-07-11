"""Run live letter inference from the Aria RGB stream.

This module subscribes to live RGB frames from Meta Project Aria, tracks the
index fingertip with MediaPipe, converts trajectories into normalized sequences,
and predicts letters with a trained PyTorch model. It depends on the Aria SDK,
OpenCV, MediaPipe, and PyTorch, and it produces an annotated OpenCV display.
"""

import argparse
import queue
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional, Tuple

import aria.sdk as aria
import cv2
import mediapipe as mp
import numpy as np
import torch
from torch import nn

from config import (
    BASE_MODEL_PATH,
    DEFAULT_MODEL_OUTPUT,
    DEFAULT_STREAM_PROFILE,
    HIDDEN_SIZE,
    INPUT_SIZE,
    LIVE_INFERENCE_DEFAULT_STREAM_INTERFACE,
    LIVE_INFERENCE_FRAME_SIZE,
    LIVE_RGB_FRAME_INTERVAL_MS,
    LIVE_RGB_QUEUE_MAXSIZE,
    LIVE_RGB_QUEUE_TIMEOUT_SECONDS,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    MEDIAPIPE_MODEL_COMPLEXITY,
    MEDIAPIPE_SINGLE_HAND_MAX_NUM_HANDS,
    NUM_CLASSES,
    NUM_LAYERS,
    PERSONAL_MODEL_PATH,
    TRAJECTORY_DWELL_SECONDS,
    TRAJECTORY_MIN_POINTS,
    TRAJECTORY_MOVEMENT_THRESHOLD_PX,
)
from logging_utils import get_logger
from pretrain_emnist import LetterCNN
from rasterize import trajectory_to_image
from vrs_index_fingertip_tracker import TrajectoryBuilder, is_degenerate_trajectory, normalize_trajectory


logger = get_logger(__name__)

aria.set_log_level(aria.Level.Info)

RGB_FRAME_QUEUE = queue.Queue(maxsize=LIVE_RGB_QUEUE_MAXSIZE)
shutdown_requested = False
started_internally = False
device_client = None
device = None
streaming_manager = None
streaming_client = aria.StreamingClient()


class LetterLSTMClassifier(nn.Module):
    """LSTM classifier used for live checkpoint inference."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of normalized trajectories."""
        _, (hidden_state, _) = self.lstm(x)
        return self.fc(hidden_state[-1])


class LiveLetterPredictor:
    """Predict letters from live fingertip motion."""

    def __init__(self, model_path: Path, movement_threshold_px: float, dwell_seconds: float) -> None:
        """Load the trained model and initialize the live tracker."""
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(f"Failed to load model checkpoint {model_path}: {exc}") from exc
        self.label_lookup = {idx: label for label, idx in checkpoint["label_to_index"].items()}
        self.model = LetterLSTMClassifier()
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except Exception as exc:
            raise RuntimeError(f"Failed to restore model weights from {model_path}: {exc}") from exc
        self.model.eval()

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=MEDIAPIPE_SINGLE_HAND_MAX_NUM_HANDS,
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        )
        self.trajectory_builder = TrajectoryBuilder(
            movement_threshold_px=movement_threshold_px,
            dwell_seconds=dwell_seconds,
            min_points=TRAJECTORY_MIN_POINTS,
        )
        self.last_predictions = deque(maxlen=3)
        self.last_prediction_conf = 0.0
        self.last_prediction_time = 0.0
        self.frames_seen = 0

    def close(self) -> None:
        """Release MediaPipe resources used by the predictor."""
        try:
            self.hands.close()
        except Exception as exc:
            logger.debug("Ignoring MediaPipe close failure: %s", exc)

    def predict_letter(
        self,
        points: list[tuple[int, int]],
    ) -> Tuple[str, float, np.ndarray]:
        """Predict a letter from one completed trajectory."""
        try:
            normalized = normalize_trajectory(points)
            tensor = torch.from_numpy(normalized).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = int(torch.argmax(probs).item())
                conf = float(probs[pred_idx].item())
            return self.label_lookup[pred_idx], conf, normalized
        except Exception as exc:
            raise RuntimeError(f"Failed to predict letter from trajectory: {exc}") from exc

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ns: int) -> np.ndarray:
        """Process one live RGB frame and overlay live inference results."""
        self.frames_seen += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            results = self.hands.process(frame_rgb)
        except Exception as exc:
            logger.warning("MediaPipe failed on live frame %s: %s", self.frames_seen, exc)
            results = None
        fingertip_point = None

        if results and results.multi_hand_landmarks:
            h, w = frame_bgr.shape[:2]
            hand_landmarks = results.multi_hand_landmarks[0]
            tip = hand_landmarks.landmark[8]
            x = int(tip.x * w)
            y = int(tip.y * h)
            fingertip_point = (x, y)
            cv2.circle(frame_bgr, fingertip_point, 9, (0, 255, 0), -1, lineType=cv2.LINE_AA)

        finished_trajectory = self.trajectory_builder.update(fingertip_point, timestamp_ns)
        if finished_trajectory:
            if is_degenerate_trajectory(finished_trajectory):
                logger.debug(
                    "Skipping degenerate trajectory (%s points, insufficient movement).",
                    len(finished_trajectory),
                )
            else:
                try:
                    letter, confidence, _ = self.predict_letter(finished_trajectory)
                    self.last_predictions.append(letter)
                    self.last_prediction_conf = confidence
                    self.last_prediction_time = time.time()
                    logger.info(
                        "Predicted letter: %s (confidence=%.3f, raw_points=%s)",
                        letter,
                        confidence,
                        len(finished_trajectory),
                    )
                except Exception as exc:
                    logger.warning("Skipping broken trajectory prediction: %s", exc)

        if len(self.trajectory_builder.points) >= 2:
            polyline = np.array(self.trajectory_builder.points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame_bgr, [polyline], False, (0, 200, 255), 3, lineType=cv2.LINE_AA)

        self.draw_overlay(frame_bgr)
        return frame_bgr

    def draw_overlay(self, frame_bgr: np.ndarray) -> None:
        """Draw model status and recent predictions on the frame."""
        prediction_text = " ".join(self.last_predictions) if self.last_predictions else "--"
        cv2.rectangle(frame_bgr, (16, 16), (frame_bgr.shape[1] - 16, 132), (10, 18, 24), -1)
        cv2.putText(
            frame_bgr,
            "LIVE LETTERS",
            (30, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (120, 220, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            prediction_text,
            (30, 104),
            cv2.FONT_HERSHEY_DUPLEX,
            1.5,
            (0, 255, 140),
            3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"last conf: {self.last_prediction_conf:.2f}",
            (frame_bgr.shape[1] - 230, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"stroke points: {len(self.trajectory_builder.points)}",
            (frame_bgr.shape[1] - 260, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 200, 80),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            "Draw a letter, pause to commit. C clears history. Q quits.",
            (30, frame_bgr.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            2,
            lineType=cv2.LINE_AA,
        )

    def clear_predictions(self) -> None:
        """Reset the prediction history and active stroke state."""
        self.last_predictions.clear()
        self.last_prediction_conf = 0.0
        self.trajectory_builder.points = []
        self.trajectory_builder.anchor_point = None
        self.trajectory_builder.anchor_timestamp_ns = None
        self.trajectory_builder.first_point_timestamp_ns = None


class RasterLetterPredictor:
    """Predict letters via rasterized-trajectory CNN classification.

    Parallel path to ``LiveLetterPredictor``: instead of feeding normalized
    point sequences to an LSTM, it rasterizes the trajectory into an
    EMNIST-shaped image and classifies it with the CNN from
    ``pretrain_emnist.py``. Prefers a per-user ``personal_model.pt`` produced
    by the calibration flow in ``training_dashboard.py``, falling back to the
    generic ``base_model.pt`` when no calibration has been done yet.
    """

    def __init__(self, movement_threshold_px: float, dwell_seconds: float) -> None:
        """Load the personal or base CNN checkpoint and initialize the live tracker."""
        model_path = PERSONAL_MODEL_PATH if PERSONAL_MODEL_PATH.exists() else BASE_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"Neither {PERSONAL_MODEL_PATH} nor {BASE_MODEL_PATH} exists; run pretrain_emnist.py first."
            )
        self.model_path = model_path
        self.using_personal_model = model_path == PERSONAL_MODEL_PATH

        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model checkpoint {model_path}: {exc}") from exc
        self.label_lookup = {idx: label for label, idx in checkpoint["label_to_index"].items()}
        self.model = LetterCNN(
            num_classes=checkpoint["num_classes"],
            conv1_channels=checkpoint["conv1_channels"],
            conv2_channels=checkpoint["conv2_channels"],
            fc_hidden=checkpoint["fc_hidden"],
        )
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except Exception as exc:
            raise RuntimeError(f"Failed to restore model weights from {model_path}: {exc}") from exc
        self.model.eval()

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=MEDIAPIPE_SINGLE_HAND_MAX_NUM_HANDS,
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        )
        self.trajectory_builder = TrajectoryBuilder(
            movement_threshold_px=movement_threshold_px,
            dwell_seconds=dwell_seconds,
            min_points=TRAJECTORY_MIN_POINTS,
        )
        self.last_predictions = deque(maxlen=3)
        self.last_prediction_conf = 0.0
        self.last_prediction_time = 0.0
        self.frames_seen = 0

    def close(self) -> None:
        """Release MediaPipe resources used by the predictor."""
        try:
            self.hands.close()
        except Exception as exc:
            logger.debug("Ignoring MediaPipe close failure: %s", exc)

    def predict_letter(
        self,
        points: list[tuple[int, int]],
    ) -> Tuple[str, float, np.ndarray]:
        """Predict a letter from one completed trajectory via rasterization + CNN."""
        try:
            normalized = normalize_trajectory(points)
            image = trajectory_to_image(normalized)
            tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = int(torch.argmax(probs).item())
                conf = float(probs[pred_idx].item())
            return self.label_lookup[pred_idx], conf, normalized
        except Exception as exc:
            raise RuntimeError(f"Failed to predict letter from trajectory: {exc}") from exc

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ns: int) -> np.ndarray:
        """Process one live RGB frame and overlay live inference results."""
        self.frames_seen += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            results = self.hands.process(frame_rgb)
        except Exception as exc:
            logger.warning("MediaPipe failed on live frame %s: %s", self.frames_seen, exc)
            results = None
        fingertip_point = None

        if results and results.multi_hand_landmarks:
            h, w = frame_bgr.shape[:2]
            hand_landmarks = results.multi_hand_landmarks[0]
            tip = hand_landmarks.landmark[8]
            x = int(tip.x * w)
            y = int(tip.y * h)
            fingertip_point = (x, y)
            cv2.circle(frame_bgr, fingertip_point, 9, (0, 255, 0), -1, lineType=cv2.LINE_AA)

        finished_trajectory = self.trajectory_builder.update(fingertip_point, timestamp_ns)
        if finished_trajectory:
            if is_degenerate_trajectory(finished_trajectory):
                logger.debug(
                    "Skipping degenerate trajectory (%s points, insufficient movement).",
                    len(finished_trajectory),
                )
            else:
                try:
                    letter, confidence, _ = self.predict_letter(finished_trajectory)
                    self.last_predictions.append(letter)
                    self.last_prediction_conf = confidence
                    self.last_prediction_time = time.time()
                    logger.info(
                        "Predicted letter: %s (confidence=%.3f, raw_points=%s)",
                        letter,
                        confidence,
                        len(finished_trajectory),
                    )
                except Exception as exc:
                    logger.warning("Skipping broken trajectory prediction: %s", exc)

        if len(self.trajectory_builder.points) >= 2:
            polyline = np.array(self.trajectory_builder.points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame_bgr, [polyline], False, (0, 200, 255), 3, lineType=cv2.LINE_AA)

        self.draw_overlay(frame_bgr)
        return frame_bgr

    def draw_overlay(self, frame_bgr: np.ndarray) -> None:
        """Draw model status and recent predictions on the frame."""
        prediction_text = " ".join(self.last_predictions) if self.last_predictions else "--"
        cv2.rectangle(frame_bgr, (16, 16), (frame_bgr.shape[1] - 16, 132), (10, 18, 24), -1)
        model_label = "PERSONAL" if self.using_personal_model else "BASE"
        cv2.putText(
            frame_bgr,
            f"LIVE LETTERS (raster/{model_label})",
            (30, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (120, 220, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            prediction_text,
            (30, 104),
            cv2.FONT_HERSHEY_DUPLEX,
            1.5,
            (0, 255, 140),
            3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"last conf: {self.last_prediction_conf:.2f}",
            (frame_bgr.shape[1] - 230, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"stroke points: {len(self.trajectory_builder.points)}",
            (frame_bgr.shape[1] - 260, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 200, 80),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            "Draw a letter, pause to commit. C clears history. Q quits.",
            (30, frame_bgr.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            2,
            lineType=cv2.LINE_AA,
        )

    def clear_predictions(self) -> None:
        """Reset the prediction history and active stroke state."""
        self.last_predictions.clear()
        self.last_prediction_conf = 0.0
        self.trajectory_builder.points = []
        self.trajectory_builder.anchor_point = None
        self.trajectory_builder.anchor_timestamp_ns = None
        self.trajectory_builder.first_point_timestamp_ns = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for live inference."""
    parser = argparse.ArgumentParser(description="Live Aria air-letter recognition using a trained LSTM.")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_OUTPUT,
        help=f"Path to trained model checkpoint. Default: {DEFAULT_MODEL_OUTPUT}",
    )
    parser.add_argument("--start-streaming", action="store_true", help="Connect to the glasses and start streaming before subscribing.")
    parser.add_argument("--skip-prestop", action="store_true", help="Do not call stop_streaming before start_streaming.")
    parser.add_argument("--persistent-certs", action="store_true", help="Use installed persistent streaming certificates instead of ephemeral certificates.")
    parser.add_argument("--local-certs-dir", help="Directory containing persistent streaming certificates.")
    parser.add_argument("--device-ip", help="Device IPv4 address for Wi-Fi streaming.")
    parser.add_argument("--interface", choices=["wifi", "usb"], default=LIVE_INFERENCE_DEFAULT_STREAM_INTERFACE, help="Streaming interface to use when --start-streaming is enabled.")
    parser.add_argument("--profile", default=DEFAULT_STREAM_PROFILE, help="Streaming profile to use when --start-streaming is enabled.")
    parser.add_argument("--movement-threshold", type=float, default=TRAJECTORY_MOVEMENT_THRESHOLD_PX, help=f"Dwell movement threshold in pixels. Default: {TRAJECTORY_MOVEMENT_THRESHOLD_PX}")
    parser.add_argument("--dwell-seconds", type=float, default=TRAJECTORY_DWELL_SECONDS, help=f"Pause duration that ends a letter. Default: {TRAJECTORY_DWELL_SECONDS}")
    parser.add_argument(
        "--engine",
        choices=["lstm", "raster"],
        default="lstm",
        help=(
            "Inference engine: 'lstm' (existing CSV/LSTM pipeline, default) or "
            "'raster' (EMNIST-pretrained CNN + calibration, parallel research path)."
        ),
    )
    return parser.parse_args()


def push_latest_frame(frame_bgr: np.ndarray, timestamp_ns: int) -> None:
    """Push the latest RGB frame into the bounded live queue."""
    if RGB_FRAME_QUEUE.full():
        try:
            RGB_FRAME_QUEUE.get_nowait()
        except queue.Empty:
            pass
    try:
        RGB_FRAME_QUEUE.put_nowait((frame_bgr, timestamp_ns))
    except queue.Full:
        logger.debug("Dropping RGB frame because queue is full.")


class AriaObserver:
    """Receive Aria RGB callbacks and downsample them for inference."""

    def __init__(self) -> None:
        self.last_rgb_ms = 0.0

    def on_image_received(self, image: np.ndarray, record: Any) -> None:
        """Handle a streamed Aria RGB frame callback."""
        ts_ms = time.time() * 1000.0
        if record.camera_id != aria.CameraId.Rgb:
            return
        if ts_ms - self.last_rgb_ms < LIVE_RGB_FRAME_INTERVAL_MS:
            return
        self.last_rgb_ms = ts_ms

        img = np.rot90(image, -1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, LIVE_INFERENCE_FRAME_SIZE, interpolation=cv2.INTER_AREA)
        try:
            push_latest_frame(img, int(record.capture_timestamp_ns))
        except queue.Full:
            pass


def configure_streaming(args: argparse.Namespace) -> AriaObserver:
    """Configure the Aria streaming client for RGB inference frames."""
    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Rgb
    config.message_queue_size[aria.StreamingDataType.Rgb] = 1

    if args.persistent_certs:
        security = aria.StreamingSecurityOptions()
        security.use_ephemeral_certs = False
        if args.local_certs_dir:
            security.local_certs_path = args.local_certs_dir
        config.security_options = security
    else:
        security = aria.StreamingSecurityOptions()
        security.use_ephemeral_certs = True
        config.security_options = security

    streaming_client.subscription_config = config
    observer = AriaObserver()
    streaming_client.set_streaming_client_observer(observer)
    return observer


def maybe_start_streaming(args: argparse.Namespace) -> None:
    """Optionally connect to the device and start streaming."""
    global started_internally, device_client, device, streaming_manager

    if not args.start_streaming:
        return

    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    if args.interface == "wifi" and args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)
    try:
        device = device_client.connect()
    except Exception as exc:
        raise RuntimeError(f"Failed to connect to Aria device: {exc}") from exc
    streaming_manager = device.streaming_manager

    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile
    if not args.persistent_certs:
        streaming_config.security_options.use_ephemeral_certs = True
    elif args.local_certs_dir:
        streaming_config.security_options.local_certs_path = args.local_certs_dir
    streaming_manager.streaming_config = streaming_config

    if not args.skip_prestop:
        try:
            streaming_manager.stop_streaming()
        except Exception:
            pass

    try:
        streaming_manager.start_streaming()
    except Exception as exc:
        raise RuntimeError(f"Failed to start streaming: {exc}") from exc
    started_internally = True
    logger.info("Started streaming via DeviceClient using %s / %s.", args.interface, args.profile)


def shutdown() -> None:
    """Stop streaming and disconnect any internally started Aria session."""
    global shutdown_requested
    shutdown_requested = True
    try:
        streaming_client.unsubscribe()
    except Exception:
        pass
    if started_internally and streaming_manager is not None:
        try:
            streaming_manager.stop_streaming()
        except Exception:
            pass
    if device_client is not None and device is not None:
        try:
            device_client.disconnect(device)
        except Exception:
            pass


def handle_signal(signum: int, frame: Optional[object]) -> None:
    """Translate process signals into a graceful shutdown request."""
    del signum, frame
    shutdown()


def main() -> None:
    """Run the live letter inference loop."""
    args = parse_args()

    if args.engine == "raster":
        predictor = RasterLetterPredictor(
            movement_threshold_px=args.movement_threshold,
            dwell_seconds=args.dwell_seconds,
        )
    else:
        model_path = Path(args.model_path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        predictor = LiveLetterPredictor(
            model_path=model_path,
            movement_threshold_px=args.movement_threshold,
            dwell_seconds=args.dwell_seconds,
        )
    configure_streaming(args)
    maybe_start_streaming(args)
    streaming_client.subscribe()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("Live letter inference running.")
    logger.info("Press Q to quit, C to clear the rolling 3-letter history.")

    window_name = "Aria Live Letter Inference"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while not shutdown_requested:
            try:
                frame_bgr, timestamp_ns = RGB_FRAME_QUEUE.get(timeout=LIVE_RGB_QUEUE_TIMEOUT_SECONDS)
            except queue.Empty:
                blank = np.zeros((LIVE_INFERENCE_FRAME_SIZE[1], LIVE_INFERENCE_FRAME_SIZE[0], 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    "Waiting for RGB frames...",
                    (120, 320),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )
                cv2.imshow(window_name, blank)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                continue

            annotated = predictor.process_frame(frame_bgr, timestamp_ns)
            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                predictor.clear_predictions()
                logger.info("Cleared prediction history.")
    finally:
        predictor.close()
        shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
