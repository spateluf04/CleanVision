"""Dual-source Aria Gen 1 sensor capture with ring-buffered fan-out.

This module exposes one :class:`AriaCapture` class with two interchangeable
backends behind an identical callback interface: a VRS playback backend built
on ``projectaria_tools`` and a live backend built on the Aria Client SDK
(``aria.sdk``). Consumers subscribe per stream label and receive
:class:`ImageSample` / :class:`MotionSample` objects carrying device-time
nanosecond timestamps. It depends on NumPy, projectaria_tools (VRS mode), the
Aria Client SDK (live mode only, imported lazily), and shared project
utilities, and it produces thread-safe latest-value image slots plus bounded
motion deques so a slow subscriber can never block capture.

Orientation contract: Aria camera images are stored rotated 90 degrees.
Every image callback delivers the RAW un-rotated frame so that anything using
calibration (undistortion, projection) operates in the native sensor frame.
Display consumers must call :func:`rotate_upright` themselves.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from config import (
    CAPTURE_ALL_LABELS,
    CAPTURE_DISPATCH_IDLE_SLEEP_SECONDS,
    CAPTURE_ET_COMBINED_LABEL,
    CAPTURE_IMAGE_LABELS,
    CAPTURE_MOTION_DEQUE_MAXLEN,
    CAPTURE_MOTION_LABELS,
    CAPTURE_SOURCE_LIVE,
    CAPTURE_SOURCE_VRS,
    CAPTURE_VRS_BACKPRESSURE_SLEEP_SECONDS,
    CAPTURE_VRS_LABEL_ALIASES,
    DEFAULT_STREAM_PROFILE,
    STREAM_MESSAGE_QUEUE_SIZE,
)
from logging_utils import get_logger


logger = get_logger(__name__)

PIXEL_FORMAT_RGB = "rgb"
PIXEL_FORMAT_GRAY = "gray"

# Calibration is loaded for these camera labels when a VRS file provides it.
CALIBRATED_CAMERA_LABELS = (
    "camera-rgb",
    "camera-slam-left",
    "camera-slam-right",
    "camera-et-left",
    "camera-et-right",
)


@dataclass(frozen=True)
class ImageSample:
    """One camera frame delivered to subscribers.

    ``frame`` is a C-contiguous uint8 array in the RAW un-rotated sensor
    orientation (use it directly for anything involving CameraCalibration;
    call :func:`rotate_upright` for display). ``pixel_format`` is ``"rgb"``
    (H, W, 3) or ``"gray"`` (H, W); capture never converts gray to BGR --
    that is the consumer's job. ``capture_timestamp_ns`` is device time.
    """

    label: str
    frame: np.ndarray
    pixel_format: str
    capture_timestamp_ns: int


@dataclass(frozen=True)
class MotionSample:
    """One IMU / magnetometer / barometer sample delivered to subscribers.

    IMU units are exactly what the SDK provides: ``accel_msec2`` in m/s^2 and
    ``gyro_radsec`` in rad/s, in the Aria device-frame right-handed axes
    convention of the emitting IMU. Axes are passed through untouched -- do
    not reorder them here; alignment to a common frame is calibration work
    that belongs downstream. ``capture_timestamp_ns`` is device time.
    """

    label: str
    capture_timestamp_ns: int
    accel_msec2: Optional[Tuple[float, float, float]] = None
    gyro_radsec: Optional[Tuple[float, float, float]] = None
    mag_tesla: Optional[Tuple[float, float, float]] = None
    pressure_pa: Optional[float] = None
    temp_c: Optional[float] = None


def rotate_upright(frame: np.ndarray) -> np.ndarray:
    """Return a display-oriented copy of a RAW Aria frame.

    Aria camera images are stored rotated 90 degrees; ``np.rot90(frame, -1)``
    applies the clockwise correction (same as bridge.py display path). The
    returned array is a contiguous copy; the RAW input is never modified.
    """
    return np.ascontiguousarray(np.rot90(frame, -1))


class _LatestSlot:
    """Single-slot latest-value buffer for one image stream.

    The producer replaces the slot content; stale frames are dropped, never
    queued, so image delivery can lag capture without ever blocking it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sample: Optional[ImageSample] = None
        self._seq = 0

    def put(self, sample: ImageSample) -> None:
        with self._lock:
            self._sample = sample
            self._seq += 1

    def get(self) -> Tuple[int, Optional[ImageSample]]:
        with self._lock:
            return self._seq, self._sample


def _as_uint8_contiguous(frame: np.ndarray) -> np.ndarray:
    """Normalize a decoded frame to a C-contiguous uint8 array."""
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _to_tuple3(values: Any) -> Tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


class AriaCapture:
    """Dual-backend Aria Gen 1 capture with a per-stream callback interface.

    Args:
        source: ``"vrs"`` (playback via projectaria_tools) or ``"live"``
            (Aria Client SDK streaming subscription).
        vrs_path: Recording path, required for the VRS backend.
        device_ip: Optional glasses IPv4 for live Wi-Fi streaming; used only
            when ``start_streaming`` is True.
        start_streaming: Live backend only -- connect via DeviceClient and
            start streaming before subscribing (same flow as bridge.py).
            Default False: subscribe-only, assuming ``aria streaming start``
            already ran (see README).
        streaming_interface: ``"wifi"`` or ``"usb"`` for ``start_streaming``.
        profile_name: Streaming profile for ``start_streaming``. The active
            profile on the glasses determines which sensors stream at all and
            at what rates; profile18 (project default) streams RGB + SLAM +
            ET images and IMU at nominal Gen 1 rates.
        use_ephemeral_certs: Live backend certificate mode.
        local_certs_dir: Optional persistent-cert directory (live backend).

    Subscribe with :meth:`subscribe` before :meth:`start`. Image labels
    deliver :class:`ImageSample`; motion labels deliver :class:`MotionSample`.
    The Gen 1 eye-tracking camera is ONE stream containing both eyes side by
    side in a single image; capture splits it at the horizontal midpoint into
    ``camera-et-left`` / ``camera-et-right`` samples sharing one timestamp.
    """

    def __init__(
        self,
        source: str,
        vrs_path: Optional[str] = None,
        device_ip: Optional[str] = None,
        start_streaming: bool = False,
        streaming_interface: str = "wifi",
        profile_name: str = DEFAULT_STREAM_PROFILE,
        use_ephemeral_certs: bool = True,
        local_certs_dir: Optional[str] = None,
    ) -> None:
        if source not in (CAPTURE_SOURCE_VRS, CAPTURE_SOURCE_LIVE):
            raise ValueError(f"Unknown capture source: {source!r}")
        self.source = source
        self.device_ip = device_ip
        self.start_streaming = start_streaming
        self.streaming_interface = streaming_interface
        self.profile_name = profile_name
        self.use_ephemeral_certs = use_ephemeral_certs
        self.local_certs_dir = local_certs_dir

        self._subscriptions: Dict[str, List[Callable[[Any], None]]] = {}
        self._subscriptions_lock = threading.Lock()
        self._image_slots: Dict[str, _LatestSlot] = {label: _LatestSlot() for label in CAPTURE_IMAGE_LABELS}
        self._motion_deques: Dict[str, deque] = {
            label: deque(maxlen=CAPTURE_MOTION_DEQUE_MAXLEN) for label in CAPTURE_MOTION_LABELS
        }
        self._latest_motion: Dict[str, Optional[MotionSample]] = {label: None for label in CAPTURE_MOTION_LABELS}

        self._stop_event = threading.Event()
        self._producer_thread: Optional[threading.Thread] = None
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._producer_done = threading.Event()
        self._drained = threading.Event()
        self._started = False

        self._calibrations: Dict[str, Any] = {}
        self.available_vrs_labels: List[str] = []
        self.missing_expected_labels: List[str] = []

        # VRS state
        self._provider = None
        self._label_to_stream_id: Dict[str, Any] = {}
        self._stream_id_to_label: Dict[str, str] = {}

        # Live state
        self._aria_sdk = None
        self._streaming_client = None
        self._observer = None
        self._device_client = None
        self._device = None
        self._streaming_manager = None
        self._started_streaming_internally = False

        if source == CAPTURE_SOURCE_VRS:
            if not vrs_path:
                raise ValueError("vrs_path is required for the VRS backend.")
            self.vrs_path = Path(vrs_path).expanduser()
            self._open_vrs()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def subscribe(self, stream_label: str, callback: Callable[[Any], None]) -> None:
        """Register a callback for one canonical stream label."""
        if stream_label not in CAPTURE_ALL_LABELS:
            raise ValueError(f"Unknown stream label {stream_label!r}. Expected one of {CAPTURE_ALL_LABELS}.")
        with self._subscriptions_lock:
            self._subscriptions.setdefault(stream_label, []).append(callback)

    def start(self) -> None:
        """Start the backend producer and the fan-out dispatcher."""
        if self._started:
            raise RuntimeError("AriaCapture is already started.")
        self._started = True
        self._stop_event.clear()
        self._dispatcher_thread = threading.Thread(target=self._dispatch_loop, name="aria-capture-dispatch", daemon=True)
        self._dispatcher_thread.start()
        if self.source == CAPTURE_SOURCE_VRS:
            self._producer_thread = threading.Thread(target=self._vrs_playback_loop, name="aria-capture-vrs", daemon=True)
            self._producer_thread.start()
        else:
            self._start_live()

    def stop(self) -> None:
        """Stop capture, join worker threads, and release backend resources."""
        self._stop_event.set()
        if self._producer_thread is not None:
            self._producer_thread.join(timeout=5.0)
            self._producer_thread = None
        if self.source == CAPTURE_SOURCE_LIVE:
            self._stop_live()
        self._producer_done.set()
        if self._dispatcher_thread is not None:
            self._dispatcher_thread.join(timeout=5.0)
            self._dispatcher_thread = None
        self._started = False

    @property
    def finished(self) -> bool:
        """True once the producer ended (VRS end-of-file) and buffers drained."""
        return self._producer_done.is_set() and self._drained.is_set()

    def get_calibration(self, label: str) -> Optional[Any]:
        """Return the CameraCalibration for a camera label, or None.

        VRS backend: extracted from ``provider.get_device_calibration()`` at
        startup. Live backend: always None in this build -- the Client SDK
        streaming path does not deliver device calibration; load it from a
        VRS recording or factory calibration JSON in a later build.
        """
        return self._calibrations.get(label)

    def latest(self, stream_label: str) -> Optional[Any]:
        """Return the most recent sample for a label without consuming it."""
        if stream_label in self._image_slots:
            return self._image_slots[stream_label].get()[1]
        if stream_label in self._motion_deques:
            return self._latest_motion[stream_label]
        raise ValueError(f"Unknown stream label {stream_label!r}.")

    def drain(self, stream_label: str) -> List[MotionSample]:
        """Batch-read and consume all queued motion samples for a label.

        Only for motion labels WITHOUT a subscribed callback (the dispatcher
        drains subscribed labels itself; mixing both would split the data).
        """
        buf = self._motion_deques.get(stream_label)
        if buf is None:
            raise ValueError(f"{stream_label!r} is not a motion stream label.")
        samples: List[MotionSample] = []
        while True:
            try:
                samples.append(buf.popleft())
            except IndexError:
                return samples

    # ------------------------------------------------------------------
    # Buffer writes (producer side -- never invokes user callbacks)
    # ------------------------------------------------------------------

    def _emit_image(self, label: str, frame: np.ndarray, pixel_format: str, ts_ns: int) -> None:
        sample = ImageSample(
            label=label,
            frame=_as_uint8_contiguous(frame),
            pixel_format=pixel_format,
            capture_timestamp_ns=int(ts_ns),
        )
        self._image_slots[label].put(sample)

    def _emit_et_pair(self, et_image: np.ndarray, ts_ns: int) -> None:
        """Split the combined Gen 1 ET frame into left/right eye samples.

        The single ET image holds both eyes side by side; the left half of
        the RAW image is the left eye. Both halves share the same timestamp.
        """
        midpoint = et_image.shape[1] // 2
        self._emit_image("camera-et-left", et_image[:, :midpoint], PIXEL_FORMAT_GRAY, ts_ns)
        self._emit_image("camera-et-right", et_image[:, midpoint:], PIXEL_FORMAT_GRAY, ts_ns)

    def _emit_motion(self, sample: MotionSample) -> None:
        self._motion_deques[sample.label].append(sample)
        self._latest_motion[sample.label] = sample

    # ------------------------------------------------------------------
    # Fan-out dispatcher
    # ------------------------------------------------------------------

    def _dispatch_loop(self) -> None:
        """Deliver buffered samples to subscribers on a dedicated thread.

        Only this thread invokes user callbacks, so a slow subscriber can
        delay other subscribers but never the producer/capture path.
        """
        last_image_seq = {label: 0 for label in CAPTURE_IMAGE_LABELS}
        while True:
            with self._subscriptions_lock:
                subscriptions = {label: list(cbs) for label, cbs in self._subscriptions.items()}
            delivered = 0
            for label, callbacks in subscriptions.items():
                if label in self._image_slots:
                    seq, sample = self._image_slots[label].get()
                    if sample is not None and seq > last_image_seq[label]:
                        last_image_seq[label] = seq
                        delivered += 1
                        for callback in callbacks:
                            self._safe_call(callback, sample)
                else:
                    buf = self._motion_deques[label]
                    while True:
                        try:
                            sample = buf.popleft()
                        except IndexError:
                            break
                        delivered += 1
                        for callback in callbacks:
                            self._safe_call(callback, sample)
            if delivered == 0:
                if self._producer_done.is_set():
                    self._drained.set()
                    if self._stop_event.is_set() or self.source == CAPTURE_SOURCE_VRS:
                        return
                time.sleep(CAPTURE_DISPATCH_IDLE_SLEEP_SECONDS)

    @staticmethod
    def _safe_call(callback: Callable[[Any], None], sample: Any) -> None:
        try:
            callback(sample)
        except Exception as exc:
            logger.warning("Subscriber callback failed for %s: %s", sample.label, exc)

    # ------------------------------------------------------------------
    # VRS backend
    # ------------------------------------------------------------------

    def _open_vrs(self) -> None:
        from projectaria_tools.core.data_provider import create_vrs_data_provider

        if not self.vrs_path.exists():
            raise FileNotFoundError(f"VRS file not found: {self.vrs_path}")
        try:
            provider = create_vrs_data_provider(str(self.vrs_path))
        except Exception as exc:
            raise RuntimeError(f"Failed to open VRS file {self.vrs_path}: {exc}") from exc
        if provider is None:
            raise RuntimeError(f"Failed to create VRS data provider for: {self.vrs_path}")
        self._provider = provider

        available: Dict[str, Any] = {}
        for stream_id in provider.get_all_streams():
            try:
                label = provider.get_label_from_stream_id(stream_id)
            except Exception:
                logger.debug("Skipping unreadable stream label for stream id %s.", stream_id)
                continue
            available[label] = stream_id
        self.available_vrs_labels = sorted(available.keys())
        logger.info("VRS streams available: %s", ", ".join(self.available_vrs_labels))

        # Resolve every expected physical stream by label (never by id).
        self.missing_expected_labels = []
        for canonical, aliases in CAPTURE_VRS_LABEL_ALIASES.items():
            stream_id = next((available[alias] for alias in aliases if alias in available), None)
            if stream_id is None:
                self.missing_expected_labels.append(canonical)
                logger.warning("Expected stream %s not found in VRS (aliases tried: %s).", canonical, ", ".join(aliases))
            else:
                self._label_to_stream_id[canonical] = stream_id
                self._stream_id_to_label[str(stream_id)] = canonical

        self._load_vrs_calibration()

    def _load_vrs_calibration(self) -> None:
        try:
            device_calib = self._provider.get_device_calibration()
        except Exception as exc:
            logger.warning("Failed to load device calibration: %s", exc)
            return
        if device_calib is None:
            logger.warning("VRS file carries no device calibration.")
            return
        for label in CALIBRATED_CAMERA_LABELS:
            try:
                cam_calib = device_calib.get_camera_calib(label)
            except Exception:
                cam_calib = None
            if cam_calib is None:
                logger.warning("No camera calibration for %s.", label)
                continue
            self._calibrations[label] = cam_calib
            focal = cam_calib.get_focal_lengths()
            principal = cam_calib.get_principal_point()
            logger.info(
                "Calibration %s: model=%s focal=(%.2f, %.2f) principal=(%.2f, %.2f)",
                label,
                cam_calib.model_name(),
                focal[0],
                focal[1],
                principal[0],
                principal[1],
            )

    def _vrs_playback_loop(self) -> None:
        """Replay all resolved streams interleaved in device-time order.

        Uses deliver_queued_sensor_data() (not per-stream index loops) so all
        modalities arrive in timestamp order exactly as recorded.
        """
        from projectaria_tools.core.sensor_data import SensorDataType, TimeDomain

        provider = self._provider
        options = provider.get_default_deliver_queued_options()
        try:
            options.deactivate_stream_all()
            for stream_id in self._label_to_stream_id.values():
                options.activate_stream(stream_id)
        except Exception as exc:
            logger.warning("Could not restrict playback streams (%s); delivering all.", exc)
            options = provider.get_default_deliver_queued_options()

        # File playback runs faster than real time; briefly pause when a
        # subscribed motion buffer gets half full so the dispatcher never
        # loses samples. Unsubscribed deques are exempt (nobody drains them;
        # they just keep the latest maxlen samples for pull-style readers).
        # This backpressure is VRS-only -- the live producer never blocks.
        backpressure_level = CAPTURE_MOTION_DEQUE_MAXLEN // 2
        with self._subscriptions_lock:
            paced_deques = [
                self._motion_deques[label] for label in CAPTURE_MOTION_LABELS if self._subscriptions.get(label)
            ]

        try:
            for sensor_data in provider.deliver_queued_sensor_data(options):
                if self._stop_event.is_set():
                    break
                label = self._stream_id_to_label.get(str(sensor_data.stream_id()))
                if label is None:
                    continue
                data_type = sensor_data.sensor_data_type()
                if data_type == SensorDataType.IMAGE:
                    self._handle_vrs_image(label, sensor_data)
                elif data_type == SensorDataType.IMU:
                    motion = sensor_data.imu_data()
                    self._emit_motion(
                        MotionSample(
                            label=label,
                            capture_timestamp_ns=int(motion.capture_timestamp_ns),
                            accel_msec2=_to_tuple3(motion.accel_msec2),
                            gyro_radsec=_to_tuple3(motion.gyro_radsec),
                        )
                    )
                elif data_type == SensorDataType.MAGNETOMETER:
                    motion = sensor_data.magnetometer_data()
                    self._emit_motion(
                        MotionSample(
                            label=label,
                            capture_timestamp_ns=int(motion.capture_timestamp_ns),
                            mag_tesla=_to_tuple3(motion.mag_tesla),
                        )
                    )
                elif data_type == SensorDataType.BAROMETER:
                    baro = sensor_data.barometer_data()
                    self._emit_motion(
                        MotionSample(
                            label=label,
                            capture_timestamp_ns=int(baro.capture_timestamp_ns),
                            pressure_pa=float(baro.pressure),
                            temp_c=float(baro.temperature),
                        )
                    )
                else:
                    # Fallback: unknown payloads still surface a device-time
                    # heartbeat if ever mapped; currently unreachable because
                    # only resolved streams are activated.
                    sensor_data.get_time_ns(TimeDomain.DEVICE_TIME)
                while (
                    paced_deques
                    and not self._stop_event.is_set()
                    and max(len(buf) for buf in paced_deques) > backpressure_level
                ):
                    time.sleep(CAPTURE_VRS_BACKPRESSURE_SLEEP_SECONDS)
        except Exception as exc:
            logger.error("VRS playback failed: %s", exc)
        finally:
            self._producer_done.set()

    def _handle_vrs_image(self, label: str, sensor_data: Any) -> None:
        image_data, record = sensor_data.image_data_and_record()
        if image_data is None or not image_data.is_valid():
            return
        frame = image_data.to_numpy_array()
        ts_ns = int(record.capture_timestamp_ns)
        if label == CAPTURE_ET_COMBINED_LABEL:
            self._emit_et_pair(frame, ts_ns)
            return
        if label == "camera-rgb":
            if frame.ndim == 2:
                frame = self._debayer_rgb(frame)
            pixel_format = PIXEL_FORMAT_RGB if frame.ndim == 3 else PIXEL_FORMAT_GRAY
        else:
            pixel_format = PIXEL_FORMAT_GRAY
        self._emit_image(label, frame, pixel_format, ts_ns)

    @staticmethod
    def _debayer_rgb(frame: np.ndarray) -> np.ndarray:
        """Best-effort debayer for raw single-channel RGB payloads."""
        try:
            from projectaria_tools.core.image import debayer

            decoded = debayer(frame)
            if hasattr(decoded, "to_numpy_array"):
                decoded = decoded.to_numpy_array()
            if decoded.ndim == 3:
                return decoded
        except Exception:
            logger.debug("Debayer failed; delivering RGB stream frame as gray.")
        return frame

    # ------------------------------------------------------------------
    # Live backend (Aria Client SDK)
    # ------------------------------------------------------------------

    def _start_live(self) -> None:
        # Lazy import so VRS-only machines never need the Client SDK.
        import aria.sdk as aria

        self._aria_sdk = aria

        if self.start_streaming:
            # Same DeviceClient flow as bridge.py --start-streaming.
            self._device_client = aria.DeviceClient()
            client_config = aria.DeviceClientConfig()
            if self.device_ip:
                client_config.ip_v4_address = self.device_ip
            self._device_client.set_client_config(client_config)
            self._device = self._device_client.connect()
            self._streaming_manager = self._device.streaming_manager
            self._streaming_client = self._streaming_manager.streaming_client

            streaming_config = aria.StreamingConfig()
            # profile18 (project default) is assumed; the profile decides
            # which sensors stream and at what rates.
            streaming_config.profile_name = self.profile_name
            if self.streaming_interface == "usb":
                streaming_config.streaming_interface = aria.StreamingInterface.Usb
            streaming_config.security_options.use_ephemeral_certs = self.use_ephemeral_certs
            if self.local_certs_dir:
                streaming_config.security_options.local_certs_root_path = self.local_certs_dir
            self._streaming_manager.streaming_config = streaming_config
            try:
                # Clear a stale session left by a previous run/crash -- without
                # this, start_streaming() can fail or hang (same fix as bridge.py).
                self._streaming_manager.stop_streaming()
                time.sleep(1.0)
                logger.info("Stopped any existing Aria stream before starting a new one.")
            except Exception as exc:
                logger.warning("No existing stream stopped before start: %s", exc)
            self._streaming_manager.start_streaming()
            self._started_streaming_internally = True
            logger.info("Started streaming via DeviceClient (%s / %s).", self.streaming_interface, self.profile_name)
        else:
            self._streaming_client = aria.StreamingClient()

        config = self._streaming_client.subscription_config
        # Explicit data-type filter: RGB + SLAM + ET images, IMU, mag, baro.
        # Which of these actually arrive is decided by the active streaming
        # profile on the glasses (profile18 assumed).
        config.subscriber_data_type = (
            aria.StreamingDataType.Rgb
            | aria.StreamingDataType.Slam
            | aria.StreamingDataType.EyeTrack
            | aria.StreamingDataType.Imu
            | aria.StreamingDataType.Magneto
            | aria.StreamingDataType.Baro
        )
        # Keep image queues shallow: we only ever want the latest frame.
        config.message_queue_size[aria.StreamingDataType.Rgb] = STREAM_MESSAGE_QUEUE_SIZE
        config.message_queue_size[aria.StreamingDataType.Slam] = STREAM_MESSAGE_QUEUE_SIZE
        config.message_queue_size[aria.StreamingDataType.EyeTrack] = STREAM_MESSAGE_QUEUE_SIZE
        security = aria.StreamingSecurityOptions()
        security.use_ephemeral_certs = self.use_ephemeral_certs
        if self.local_certs_dir:
            security.local_certs_root_path = self.local_certs_dir
        config.security_options = security
        self._streaming_client.subscription_config = config

        self._observer = _LiveObserver(self)
        self._streaming_client.set_streaming_client_observer(self._observer)
        self._streaming_client.subscribe()
        logger.info("Live streaming client subscribed: %s", self._streaming_client.is_subscribed())
        logger.info("Live mode delivers no device calibration; get_calibration() returns None.")

    def _stop_live(self) -> None:
        if self._streaming_client is not None:
            try:
                self._streaming_client.unsubscribe()
            except Exception as exc:
                logger.warning("Failed to unsubscribe streaming client: %s", exc)
        if self._started_streaming_internally and self._streaming_manager is not None:
            try:
                self._streaming_manager.stop_streaming()
            except Exception as exc:
                logger.warning("Failed to stop streaming: %s", exc)
        if self._device_client is not None and self._device is not None:
            try:
                self._device_client.disconnect(self._device)
            except Exception as exc:
                logger.warning("Failed to disconnect device: %s", exc)


class _LiveObserver:
    """Aria Client SDK observer that writes into the capture buffers.

    SDK callbacks arrive on SDK-owned threads; they only write ring buffers
    (never user callbacks), so capture keeps pace regardless of subscribers.
    All timestamps come from record/sample capture_timestamp_ns (device
    time) -- wall-clock arrival time is never used for alignment.
    """

    def __init__(self, capture: AriaCapture) -> None:
        self._capture = capture

    def on_image_received(self, image: np.ndarray, record: Any) -> None:
        aria = self._capture._aria_sdk
        ts_ns = int(record.capture_timestamp_ns)
        camera_id = record.camera_id
        if camera_id == aria.CameraId.Rgb:
            # Live RGB frames arrive as 8-bit RGB.
            self._capture._emit_image("camera-rgb", image, PIXEL_FORMAT_RGB, ts_ns)
        elif camera_id == aria.CameraId.Slam1:
            self._capture._emit_image("camera-slam-left", image, PIXEL_FORMAT_GRAY, ts_ns)
        elif camera_id == aria.CameraId.Slam2:
            self._capture._emit_image("camera-slam-right", image, PIXEL_FORMAT_GRAY, ts_ns)
        elif camera_id == aria.CameraId.EyeTrack:
            self._capture._emit_et_pair(image, ts_ns)

    def on_imu_received(self, samples: Any, imu_idx: int) -> None:
        # imu_idx 0 is the right IMU (~1000 Hz), 1 the left IMU (~800 Hz).
        # Deliver every sample in the batch -- no samples[-1] UI throttling.
        label = "imu-right" if int(imu_idx) == 0 else "imu-left"
        for sample in samples:
            self._capture._emit_motion(
                MotionSample(
                    label=label,
                    capture_timestamp_ns=int(sample.capture_timestamp_ns),
                    accel_msec2=_to_tuple3(sample.accel_msec2),
                    gyro_radsec=_to_tuple3(sample.gyro_radsec),
                )
            )

    def on_magneto_received(self, sample: Any) -> None:
        self._capture._emit_motion(
            MotionSample(
                label="mag0",
                capture_timestamp_ns=int(getattr(sample, "capture_timestamp_ns", 0)),
                mag_tesla=_to_tuple3(sample.mag_tesla),
            )
        )

    def on_baro_received(self, sample: Any) -> None:
        self._capture._emit_motion(
            MotionSample(
                label="baro0",
                capture_timestamp_ns=int(getattr(sample, "capture_timestamp_ns", 0)),
                pressure_pa=float(sample.pressure),
                temp_c=float(sample.temperature),
            )
        )

    def on_streaming_client_failure(self, reason: Any, message: str) -> None:
        logger.error("Streaming client failure (%s): %s", reason, message)
