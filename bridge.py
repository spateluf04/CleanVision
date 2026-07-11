"""Bridge Aria sensor streams to local WebSocket consumers.

This module subscribes to Meta Project Aria sensors, runs blink and gaze
processing, and publishes frames plus telemetry to browser and desktop clients.
It depends on the Aria SDK, OpenCV, NumPy, websockets, and shared project
utilities, and it produces WebSocket messages on the local bridge port.
"""

import asyncio
import argparse
import base64
import concurrent.futures
import json
import math
import signal
import threading
import time
from typing import Any, Dict, Optional

import aria.sdk as aria
import cv2
import numpy as np
import websockets
from config import (
    ALTITUDE_SCALE_METERS,
    AUDIO_CHANNEL_TARGET,
    AUDIO_NORMALIZATION_DIVISOR,
    BLINK_BASELINE_DECAY,
    BLINK_COOLDOWN_MS,
    BLINK_DELTA_THRESHOLD,
    BLINK_OPEN_BASELINE,
    BLINK_OPEN_THRESHOLD_OFFSET,
    BLINK_PREV_SCORE,
    BLINK_RAPID_DROP_THRESHOLD,
    BLINK_RAPID_DROP_WINDOW_MS,
    BRIDGE_AUDIO_THROTTLE_MS,
    BRIDGE_BARO_THROTTLE_MS,
    BRIDGE_DEFAULT_STREAM_INTERFACE,
    BRIDGE_ET_FRAME_SIZE,
    BRIDGE_ET_THROTTLE_MS,
    BRIDGE_IMU_THROTTLE_MS,
    BRIDGE_JPEG_QUALITY,
    BRIDGE_MAG_THROTTLE_MS,
    BRIDGE_RETRY_SECONDS,
    BRIDGE_RGB_FRAME_SIZE,
    BRIDGE_RGB_THROTTLE_MS,
    BRIDGE_SEND_QUEUE_MAXSIZE,
    BRIDGE_SLAM_FRAME_SIZE,
    BRIDGE_SLAM_THROTTLE_MS,
    BRIDGE_STATS_LOOP_INTERVAL_SECONDS,
    BRIDGE_THREADPOOL_WORKERS,
    DEFAULT_STREAM_PROFILE,
    DEFAULT_YOLO_CONF_THRESHOLD,
    DEFAULT_YOLO_MODEL_SIZE,
    EYE_DARK_THRESHOLD,
    EYE_RESIZE_HEIGHT,
    EYE_RESIZE_WIDTH,
    SEA_LEVEL_PRESSURE_HPA,
    STREAM_MESSAGE_QUEUE_SIZE,
    WS_HOST,
    WS_PORT,
    WS_URL,
)
from gaze_detector import GazeDetector
from logging_utils import get_logger


aria.set_log_level(aria.Level.Info)
logger = get_logger(__name__)

send_queue = None
loop = None
executor = concurrent.futures.ThreadPoolExecutor(max_workers=BRIDGE_THREADPOOL_WORKERS)
clients = set()
streaming_client = aria.StreamingClient()
gaze_detector = GazeDetector(model_size=DEFAULT_YOLO_MODEL_SIZE, conf_threshold=DEFAULT_YOLO_CONF_THRESHOLD)
bridge_state_lock = threading.Lock()
sensor_counts = {
    "RGB": 0,
    "SLAM_LEFT": 0,
    "SLAM_RIGHT": 0,
    "ET_LEFT": 0,
    "ET_RIGHT": 0,
    "IMU0": 0,
    "IMU1": 0,
    "BARO": 0,
    "MAG": 0,
    "AUDIO": 0,
}
announced_sensors = set()
started_internally = False
device_client = None
device = None
streaming_manager = None
blink_state = {
    "count": 0,
    "closed": False,
    "closed_since_ms": 0.0,
    "started_ms": time.time() * 1000.0,
}
rgb_sequence_counter = 0
last_emitted_rgb_sequence = -1


def note_sensor_event(sensor: str, kind: str) -> None:
    """Record the first-seen and count state for a sensor event."""
    first_seen = False
    with bridge_state_lock:
        sensor_counts[sensor] += 1
        if sensor not in announced_sensors:
            announced_sensors.add(sensor)
            first_seen = True
    if first_seen:
        logger.info("First %s %s received.", sensor, kind)


def snapshot_sensor_counts() -> Dict[str, int]:
    """Return a thread-safe snapshot of current sensor counters."""
    with bridge_state_lock:
        return dict(sensor_counts)


class BlinkDetector:
    """Estimate blink events from ET eye images."""

    def __init__(self) -> None:
        self.is_open = True
        self.open_baseline = BLINK_OPEN_BASELINE
        self.open_threshold = BLINK_OPEN_BASELINE - BLINK_OPEN_THRESHOLD_OFFSET
        self.closed_threshold = BLINK_OPEN_BASELINE - BLINK_DELTA_THRESHOLD
        self.last_blink_ms = 0.0
        self.delta_threshold = BLINK_DELTA_THRESHOLD
        self.prev_score = BLINK_PREV_SCORE
        self.prev_ts_ms = 0.0
        self.rapid_drop_threshold = BLINK_RAPID_DROP_THRESHOLD

    def process_eye(self, eye_img: Optional[np.ndarray]) -> Optional[float]:
        """Compute the dark-pixel ratio used for blink estimation.

        Returns ``None`` when no eye frame is available so callers can avoid
        feeding a fabricated reading into the open-eye baseline.
        """
        if eye_img is None or eye_img.size == 0:
            return None

        small = cv2.resize(eye_img, (EYE_RESIZE_WIDTH, EYE_RESIZE_HEIGHT), interpolation=cv2.INTER_AREA)
        dark_pixels = float(np.sum(small < EYE_DARK_THRESHOLD))
        total_pixels = float(small.size)
        return dark_pixels / total_pixels if total_pixels > 0 else None

    def update(self, left_eye: Optional[np.ndarray], right_eye: Optional[np.ndarray], ts_ms: float) -> Dict[str, Any]:
        """Update blink state from the latest ET frame pair."""
        l_score = self.process_eye(left_eye)
        r_score = self.process_eye(right_eye)
        valid_scores = [score for score in (l_score, r_score) if score is not None]

        if not valid_scores:
            # No ET frame this cycle; report the last known state rather than
            # fabricating a reading that would corrupt the open-eye baseline.
            return {
                "is_open": self.is_open,
                "was_open": self.is_open,
                "blink_detected": False,
                "score": float(self.prev_score),
                "l_score": float(self.prev_score),
                "r_score": float(self.prev_score),
                "open_threshold": float(self.open_threshold),
                "closed_threshold": float(self.closed_threshold),
                "rapid_drop": False,
            }

        avg = sum(valid_scores) / len(valid_scores)
        l_score = l_score if l_score is not None else avg
        r_score = r_score if r_score is not None else avg
        # Track a slow-decay open-eye baseline and trigger on a 0.03 drop from that baseline.
        self.open_baseline = max(self.open_baseline * BLINK_BASELINE_DECAY, avg)
        self.open_threshold = self.open_baseline - BLINK_OPEN_THRESHOLD_OFFSET
        self.closed_threshold = self.open_baseline - self.delta_threshold
        rapid_drop = (self.prev_score - avg) >= self.rapid_drop_threshold and (ts_ms - self.prev_ts_ms) <= BLINK_RAPID_DROP_WINDOW_MS

        was_open = self.is_open
        if avg < self.closed_threshold or rapid_drop:
            self.is_open = False
        elif avg > self.open_threshold:
            self.is_open = True

        blink_detected = False
        # Count the blink on the "eye disappeared" event with a cooldown.
        if was_open and not self.is_open and ts_ms - self.last_blink_ms > BLINK_COOLDOWN_MS:
            blink_detected = True
            self.last_blink_ms = ts_ms

        self.prev_score = avg
        self.prev_ts_ms = ts_ms

        return {
            "is_open": self.is_open,
            "was_open": was_open,
            "blink_detected": blink_detected,
            "score": float(avg),
            "l_score": float(l_score),
            "r_score": float(r_score),
            "open_threshold": float(self.open_threshold),
            "closed_threshold": float(self.closed_threshold),
            "rapid_drop": bool(rapid_drop),
        }


logger.info("GazeDetector initialized. %s running on: %s", gaze_detector.model_size, gaze_detector.device)
if gaze_detector.hand_detector.available:
    logger.info("Hand gesture detection enabled with MediaPipe Hands.")
else:
    logger.warning("Hand gesture detection disabled: %s", gaze_detector.hand_detector.error)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the bridge process."""
    parser = argparse.ArgumentParser(description="Meta Project Aria WebSocket bridge")
    parser.add_argument("--start-streaming", action="store_true", help="Connect to the glasses and start streaming before subscribing.")
    parser.add_argument("--skip-prestop", action="store_true", help="Do not call stop_streaming before start_streaming.")
    parser.add_argument("--persistent-certs", action="store_true", help="Use installed persistent streaming certificates instead of ephemeral certificates.")
    parser.add_argument("--local-certs-dir", help="Directory containing persistent streaming certificates.")
    parser.add_argument("--device-ip", help="Device IPv4 address for Wi-Fi streaming.")
    parser.add_argument("--interface", choices=["wifi", "usb"], default=BRIDGE_DEFAULT_STREAM_INTERFACE, help="Streaming interface to use when --start-streaming is enabled.")
    parser.add_argument("--profile", default=DEFAULT_STREAM_PROFILE, help="Streaming profile to use when --start-streaming is enabled.")
    return parser.parse_args()


def now_ms() -> float:
    """Return the current wall-clock time in milliseconds."""
    return time.time() * 1000.0


def safe_queue_put(data: str) -> None:
    """Attempt to enqueue one serialized outbound message without blocking."""
    try:
        send_queue.put_nowait(data)
    except asyncio.QueueFull:
        pass


def enqueue(msg_dict: Dict[str, Any]) -> None:
    """Serialize and enqueue a bridge message for broadcast."""
    if send_queue is None or loop is None:
        return
    try:
        data = json.dumps(msg_dict)
    except (TypeError, ValueError) as exc:
        logger.error("Failed to serialize outbound message %s: %s", msg_dict.get("type"), exc)
        return
    loop.call_soon_threadsafe(safe_queue_put, data)


def encode_jpeg(image: np.ndarray, quality: int) -> Optional[str]:
    """Encode a BGR frame as a base64 JPEG payload."""
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def schedule_image(sensor: str, image: np.ndarray, quality: int) -> None:
    """Encode and enqueue a non-RGB sensor image asynchronously."""
    future = executor.submit(encode_jpeg, image, quality)

    def done(fut):
        try:
            payload = fut.result()
        except Exception as exc:
            logger.warning("Failed to encode %s frame: %s", sensor, exc)
            return
        if payload:
            note_sensor_event(sensor, "frame")
            enqueue({"type": "image", "sensor": sensor, "data": payload, "ts": time.time_ns()})

    future.add_done_callback(done)


def process_rgb_frame(rgb_bgr: np.ndarray) -> tuple[Optional[str], Dict[str, Any]]:
    """Run gaze processing and JPEG encoding for one RGB frame."""
    detection = gaze_detector.process(rgb_bgr)
    encoded = encode_jpeg(rgb_bgr, BRIDGE_JPEG_QUALITY)
    return encoded, detection


def schedule_rgb_processing(image: np.ndarray) -> None:
    """Schedule RGB processing work on the bridge executor.

    RGB frames are processed concurrently across the executor's worker
    threads, so a slower frame can finish after a newer one. Each frame is
    tagged with a monotonic sequence number and late completions are dropped
    to keep the emitted stream in temporal order.
    """
    global rgb_sequence_counter
    with bridge_state_lock:
        rgb_sequence_counter += 1
        sequence = rgb_sequence_counter

    future = executor.submit(process_rgb_frame, image)

    def done(fut):
        global last_emitted_rgb_sequence
        try:
            payload, detection = fut.result()
        except Exception as exc:
            logger.warning("Gaze processing failed: %s", exc)
            return
        with bridge_state_lock:
            if sequence <= last_emitted_rgb_sequence:
                logger.debug("Dropping stale RGB result (sequence=%s).", sequence)
                return
            last_emitted_rgb_sequence = sequence
        note_sensor_event("RGB", "frame")
        if payload:
            enqueue({"type": "image", "sensor": "RGB", "data": payload, "ts": time.time_ns()})
        if detection:
            enqueue({"type": "detection", **detection})

    future.add_done_callback(done)


class AriaObserver:
    """Receive Aria SDK callbacks and publish processed results."""

    def __init__(self) -> None:
        self.last_rgb_ms = 0.0
        self.last_slam_ms = 0.0
        self.last_et_ms = 0.0
        self.last_imu_ms = [0.0, 0.0]
        self.last_baro_ms = 0.0
        self.last_mag_ms = 0.0
        self.last_audio_ms = 0.0
        self.blink_detector = BlinkDetector()
        self.last_blink_emit_ms = 0.0

    def estimate_blink(self, left: np.ndarray, right: np.ndarray, ts_ms: float) -> None:
        """Estimate and publish blink state from the ET eye images."""
        result = self.blink_detector.update(left, right, ts_ms)

        if result["blink_detected"]:
            with bridge_state_lock:
                blink_state["count"] += 1
                blink_state["closed"] = True
                blink_state["closed_since_ms"] = ts_ms
                started_ms = blink_state["started_ms"]
                blink_count = blink_state["count"]
            self.last_blink_emit_ms = ts_ms
        else:
            with bridge_state_lock:
                if not result["was_open"] and result["is_open"] and blink_state["closed"]:
                    blink_state["closed"] = False
                    blink_state["closed_since_ms"] = 0.0
                else:
                    blink_state["closed"] = not result["is_open"]
                started_ms = blink_state["started_ms"]
                blink_count = blink_state["count"]

        elapsed_min = max((ts_ms - started_ms) / 60000.0, 1.0 / 60.0)
        enqueue(
            {
                "type": "blink",
                "count": int(blink_count),
                "avg_per_min": float(blink_count) / elapsed_min,
                "closed": not result["is_open"],
                "left_ratio": result["l_score"],
                "right_ratio": result["r_score"],
                "left_opening": result["l_score"],
                "right_opening": result["r_score"],
                "left_dark_ratio": result["l_score"],
                "right_dark_ratio": result["r_score"],
                "open_threshold": result["open_threshold"],
                "closed_threshold": result["closed_threshold"],
                "rapid_drop": result["rapid_drop"],
            }
        )

    def on_image_received(self, image: np.ndarray, record: Any) -> None:
        """Handle streamed Aria image callbacks."""
        ts = now_ms()

        if record.camera_id == aria.CameraId.Rgb:
            if ts - self.last_rgb_ms < BRIDGE_RGB_THROTTLE_MS:
                return
            self.last_rgb_ms = ts
            img = np.rot90(image, -1)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, BRIDGE_RGB_FRAME_SIZE, interpolation=cv2.INTER_AREA)
            schedule_rgb_processing(img)
            return

        if record.camera_id in (aria.CameraId.Slam1, aria.CameraId.Slam2):
            if ts - self.last_slam_ms < BRIDGE_SLAM_THROTTLE_MS:
                return
            self.last_slam_ms = ts
            img = np.rot90(image, -1)
            img = cv2.resize(img, BRIDGE_SLAM_FRAME_SIZE, interpolation=cv2.INTER_AREA)
            sensor = "SLAM_LEFT" if record.camera_id == aria.CameraId.Slam1 else "SLAM_RIGHT"
            schedule_image(sensor, img, BRIDGE_JPEG_QUALITY)
            return

        if record.camera_id == aria.CameraId.EyeTrack:
            if ts - self.last_et_ms < BRIDGE_ET_THROTTLE_MS:
                return
            self.last_et_ms = ts
            midpoint = image.shape[1] // 2
            left_raw = image[:, :midpoint]
            right_raw = image[:, midpoint:]
            gaze_detector.update_et(left_raw, right_raw)
            left = cv2.resize(left_raw, BRIDGE_ET_FRAME_SIZE, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right_raw, BRIDGE_ET_FRAME_SIZE, interpolation=cv2.INTER_AREA)
            self.estimate_blink(left, right, ts)
            schedule_image("ET_LEFT", left, BRIDGE_JPEG_QUALITY)
            schedule_image("ET_RIGHT", right, BRIDGE_JPEG_QUALITY)

    def on_imu_received(self, samples: Any, imu_idx: int) -> None:
        """Handle streamed IMU samples."""
        if not samples:
            return
        ts_ms = now_ms()
        idx = int(imu_idx)
        if ts_ms - self.last_imu_ms[idx] < BRIDGE_IMU_THROTTLE_MS:
            return
        self.last_imu_ms[idx] = ts_ms
        sample = samples[-1]
        accel = sample.accel_msec2
        gyro = sample.gyro_radsec
        sensor_key = f"IMU{idx}"
        note_sensor_event(sensor_key, "sample")
        enqueue(
            {
                "type": "imu",
                "idx": int(imu_idx),
                "ax": float(accel[0]),
                "ay": float(accel[1]),
                "az": float(accel[2]),
                "gx": float(gyro[0]),
                "gy": float(gyro[1]),
                "gz": float(gyro[2]),
                "ts": int(sample.capture_timestamp_ns),
            }
        )

    def on_magneto_received(self, sample: Any) -> None:
        """Handle streamed magnetometer samples."""
        ts = now_ms()
        if ts - self.last_mag_ms < BRIDGE_MAG_THROTTLE_MS:
            return
        self.last_mag_ms = ts
        mx, my, mz = [float(v) * 1_000_000.0 for v in sample.mag_tesla]
        heading = math.degrees(math.atan2(my, mx))
        if heading < 0:
            heading += 360.0
        note_sensor_event("MAG", "sample")
        enqueue({"type": "mag", "mx": mx, "my": my, "mz": mz, "heading": heading})

    def on_baro_received(self, sample: Any) -> None:
        """Handle streamed barometer samples."""
        ts = now_ms()
        if ts - self.last_baro_ms < BRIDGE_BARO_THROTTLE_MS:
            return
        self.last_baro_ms = ts
        hpa = float(sample.pressure) / 100.0
        altitude = (1.0 - (hpa / SEA_LEVEL_PRESSURE_HPA) ** 0.190284) * ALTITUDE_SCALE_METERS
        note_sensor_event("BARO", "sample")
        enqueue(
            {
                "type": "baro",
                "hpa": hpa,
                "temp": float(sample.temperature),
                "alt": altitude,
            }
        )

    def on_audio_received(self, data: Any, record: Any) -> None:
        """Handle streamed audio payloads."""
        ts = now_ms()
        if ts - self.last_audio_ms < BRIDGE_AUDIO_THROTTLE_MS:
            return
        self.last_audio_ms = ts
        rms_values = []
        channels = data.data
        if isinstance(channels, np.ndarray):
            if channels.ndim == 1:
                channels = [channels]
            else:
                channels = [channels[i] for i in range(channels.shape[0])]
        for ch in channels:
            arr = np.asarray(ch, dtype=np.float32).reshape(-1) / AUDIO_NORMALIZATION_DIVISOR
            rms_values.append(float(np.sqrt(np.mean(arr**2))) if arr.size > 0 else 0.0)
        if len(rms_values) < AUDIO_CHANNEL_TARGET:
            rms_values.extend([0.0] * (AUDIO_CHANNEL_TARGET - len(rms_values)))
        elif len(rms_values) > AUDIO_CHANNEL_TARGET:
            rms_values = rms_values[:AUDIO_CHANNEL_TARGET]
        peak_db = 20.0 * math.log10(max(rms_values) + 1e-9)
        note_sensor_event("AUDIO", "packet")
        enqueue({"type": "audio", "rms": rms_values, "peak_db": peak_db})


async def stats_loop() -> None:
    """Periodically log aggregate sensor counts."""
    while True:
        await asyncio.sleep(BRIDGE_STATS_LOOP_INTERVAL_SECONDS)
        counts_snapshot = snapshot_sensor_counts()
        total = sum(counts_snapshot.values())
        logger.info(
            "Sensor counts: %s | total=%s",
            ", ".join(f"{key}={value}" for key, value in counts_snapshot.items()),
            total,
        )


async def broadcast_loop() -> None:
    """Fan out queued bridge messages to all connected WebSocket clients."""
    while True:
        msg = await send_queue.get()
        if not clients:
            continue
        dead = []
        for ws in list(clients):
            try:
                await ws.send(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)


async def handler(websocket: Any, *_args: Any) -> None:
    """Handle one WebSocket client connection."""
    clients.add(websocket)
    try:
        await websocket.send(json.dumps({"type": "status", "connected": True}))
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.debug("Ignoring invalid websocket command payload.")
                continue
            if msg.get("type") != "calibration_control":
                continue
            action = msg.get("action")
            if action == "start":
                await loop.run_in_executor(executor, gaze_detector.start_calibration)
            elif action == "capture_point":
                x_norm = float(msg.get("x_norm", 0.5))
                y_norm = float(msg.get("y_norm", 0.5))
                await loop.run_in_executor(executor, gaze_detector.add_calibration_point, x_norm, y_norm)
            elif action == "finish":
                await loop.run_in_executor(executor, gaze_detector.finish_calibration)
    finally:
        clients.discard(websocket)


async def main() -> None:
    """Run the WebSocket bridge event loop."""
    global send_queue, loop, started_internally, device_client, device, streaming_manager, streaming_client
    args = parse_args()
    send_queue = asyncio.Queue(maxsize=BRIDGE_SEND_QUEUE_MAXSIZE)
    loop = asyncio.get_running_loop()

    if args.start_streaming:
        device_client = aria.DeviceClient()
        client_config = aria.DeviceClientConfig()
        if args.device_ip:
            client_config.ip_v4_address = args.device_ip
        device_client.set_client_config(client_config)
        device = device_client.connect()
        streaming_manager = device.streaming_manager
        streaming_client = streaming_manager.streaming_client

        streaming_config = aria.StreamingConfig()
        streaming_config.profile_name = args.profile
        if args.interface == "usb":
            streaming_config.streaming_interface = aria.StreamingInterface.Usb
        streaming_config.security_options.use_ephemeral_certs = not args.persistent_certs
        if args.local_certs_dir:
            streaming_config.security_options.local_certs_root_path = args.local_certs_dir
        streaming_manager.streaming_config = streaming_config
        if not args.skip_prestop:
            try:
                streaming_manager.stop_streaming()
                time.sleep(1.0)
                logger.info("Stopped any existing Aria stream before starting a new one.")
            except Exception as exc:
                logger.warning("No existing stream stopped before start: %s", exc)
        try:
            streaming_manager.start_streaming()
        except Exception as exc:
            raise RuntimeError(f"Failed to start Aria streaming: {exc}") from exc
        started_internally = True
        logger.info("Started streaming via DeviceClient using %s / %s.", args.interface, args.profile)

    config = streaming_client.subscription_config
    config.subscriber_data_type = (
        aria.StreamingDataType.Rgb
        | aria.StreamingDataType.Slam
        | aria.StreamingDataType.EyeTrack
        | aria.StreamingDataType.Imu
        | aria.StreamingDataType.Magneto
        | aria.StreamingDataType.Baro
        | aria.StreamingDataType.Audio
    )
    config.message_queue_size[aria.StreamingDataType.Rgb] = STREAM_MESSAGE_QUEUE_SIZE
    config.message_queue_size[aria.StreamingDataType.Slam] = STREAM_MESSAGE_QUEUE_SIZE
    config.message_queue_size[aria.StreamingDataType.EyeTrack] = STREAM_MESSAGE_QUEUE_SIZE
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = not args.persistent_certs
    if args.local_certs_dir:
        options.local_certs_root_path = args.local_certs_dir
    config.security_options = options
    streaming_client.subscription_config = config

    observer = AriaObserver()
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()
    logger.info("Streaming client subscribed: %s", streaming_client.is_subscribed())
    if config.subscriber_topic_prefix:
        logger.info("Subscriber topic prefix: %s", config.subscriber_topic_prefix)
    else:
        logger.info("Subscriber topic prefix: <default>")

    stop_event = asyncio.Event()

    def stop():
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop)
        except NotImplementedError:
            pass

    logger.info("Aria Bridge running.")
    logger.info("Waiting for browser at %s", WS_URL)
    logger.info("Serve frontend with: python3 -m http.server 8080")
    logger.info("Then open: http://localhost:8080")

    server = await websockets.serve(handler, WS_HOST, WS_PORT)
    broadcaster = asyncio.create_task(broadcast_loop())
    stats = asyncio.create_task(stats_loop())

    try:
        await stop_event.wait()
    finally:
        broadcaster.cancel()
        stats.cancel()
        server.close()
        await server.wait_closed()
        try:
            await broadcaster
        except asyncio.CancelledError:
            pass
        try:
            await stats
        except asyncio.CancelledError:
            pass
        streaming_client.unsubscribe()
        if started_internally and streaming_manager is not None:
            streaming_manager.stop_streaming()
        if device_client is not None and device is not None:
            device_client.disconnect(device)
        executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
