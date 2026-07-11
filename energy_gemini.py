"""Gemini-vision-powered energy recommendations for RoomScan.

Optional enhancement over energy_recommendations.py's rule engine: sends the
best-confidence appliance crop photos from a finished scan to the Gemini API
and asks for photo-grounded, natural-language energy-saving suggestions,
instead of (or in addition to) the threshold/co-occurrence rules.

Enabled only when GEMINI_API_KEY is set in the environment -- never hardcode
a key here. Falls back to the rule engine on ANY failure (no key, the
google-genai package isn't installed, network error, timeout, malformed
response) so a live demo never hard-depends on the network. The google-genai
SDK is imported lazily inside _generate_content(), mirroring how
energy_detector.py lazily imports ultralytics, so every other function here
-- and every test in tests/test_energy_gemini.py -- works with the package
uninstalled.

get_recommendations() is the single public entry point and is only wired
into roomscan.py:build_report(), the once-per-finished-scan report. The live
dashboard's per-tick recommendations panel and its instant Stop-Scan summary
dialog call energy_recommendations.generate_recommendations() directly and
intentionally stay rule-based-only (see config.py's GEMINI_* comment).

Standalone smoke test (requires GEMINI_API_KEY + google-genai installed):
    python energy_gemini.py --vrs /path/to/recording.vrs
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, List, Optional

import cv2
import numpy as np

from config import (
    GEMINI_API_KEY_ENV_VAR,
    GEMINI_MAX_CROPS,
    GEMINI_MAX_RECOMMENDATIONS,
    GEMINI_MODEL,
    GEMINI_TIMEOUT_SECONDS,
)
from energy_recommendations import generate_recommendations
from logging_utils import get_logger

logger = get_logger(__name__)

Device = Dict[str, object]
Totals = Dict[str, object]
Context = Dict[str, object]

_PROMPT_TEMPLATE = """You are an energy-efficiency assistant reviewing photos an in-home \
walkthrough scan captured of specific appliances, along with their estimated usage.

Scan summary (JSON): {summary_json}

Look at the attached photo(s) of the detected appliances -- their apparent age, size, \
model style, and visible state (e.g. screen on/off, indicator lights) -- combined with \
the usage data above. Return a JSON array of at most {max_recommendations} short, \
specific, actionable energy-saving suggestions as plain strings. No markdown, no \
numbering, no extra commentary outside the JSON array."""


def _api_key() -> Optional[str]:
    return os.environ.get(GEMINI_API_KEY_ENV_VAR)


def _select_crops(
    devices: List[Device],
    crops_by_class: Dict[str, List[np.ndarray]],
    max_crops: int,
) -> List[np.ndarray]:
    """Pick up to ``max_crops`` crops, biggest energy users first.

    ``devices`` is expected sorted by kwh_per_year descending (as
    estimate_room() returns it), so walking it in order and taking one crop
    per class before looping back covers the highest-impact devices first.
    """
    selected: List[np.ndarray] = []
    per_class = {d["class_name"]: list(crops_by_class.get(d["class_name"], [])) for d in devices}
    progressed = True
    while len(selected) < max_crops and progressed:
        progressed = False
        for device in devices:
            if len(selected) >= max_crops:
                break
            queue = per_class[device["class_name"]]
            if queue:
                selected.append(queue.pop(0))
                progressed = True
    return selected


def _encode_crop_jpeg(crop_rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Failed to JPEG-encode crop for Gemini upload.")
    return buf.tobytes()


def _build_prompt(totals: Totals, devices: List[Device]) -> str:
    summary = {
        "devices": [
            {
                "name": d["display_name"],
                "count": d["count"],
                "watts_active": d["watts_active"],
                "kwh_per_year": round(float(d["kwh_per_year"]), 1),
                "cost_per_year_usd": round(float(d["cost_per_year_usd"]), 2),
            }
            for d in devices
        ],
        "totals": {
            "kwh_per_year": round(float(totals["kwh_per_year"]), 1),
            "cost_per_year_usd": round(float(totals["cost_per_year_usd"]), 2),
        },
    }
    return _PROMPT_TEMPLATE.format(
        summary_json=json.dumps(summary),
        max_recommendations=GEMINI_MAX_RECOMMENDATIONS,
    )


def _generate_content(prompt: str, image_jpegs: List[bytes]) -> str:
    """The one function that touches the google-genai SDK -- kept as a thin,
    separately-mockable seam so tests never need the real package installed."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_api_key())
    parts = [prompt] + [
        types.Part.from_bytes(data=jpeg, mime_type="image/jpeg") for jpeg in image_jpegs
    ]
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=parts,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return response.text


def _call_gemini(
    devices: List[Device],
    totals: Totals,
    crops_by_class: Dict[str, List[np.ndarray]],
) -> List[str]:
    prompt = _build_prompt(totals, devices)
    crops = _select_crops(devices, crops_by_class, GEMINI_MAX_CROPS)
    image_jpegs = [_encode_crop_jpeg(c) for c in crops]

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_generate_content, prompt, image_jpegs)
        try:
            text = future.result(timeout=GEMINI_TIMEOUT_SECONDS)
        except FutureTimeoutError as exc:
            raise TimeoutError(f"Gemini call exceeded {GEMINI_TIMEOUT_SECONDS}s") from exc

    parsed = json.loads(text)
    if not isinstance(parsed, list) or not all(isinstance(s, str) for s in parsed):
        raise ValueError(f"Gemini response was not a JSON array of strings: {text!r}")
    return parsed[:GEMINI_MAX_RECOMMENDATIONS]


def get_recommendations(
    devices: List[Device],
    totals: Totals,
    crops_by_class: Optional[Dict[str, List[np.ndarray]]] = None,
    context: Optional[Context] = None,
) -> List[str]:
    """Photo-grounded recommendations when GEMINI_API_KEY is set, else the rule engine.

    Single public entry point for this module. Never raises: any failure to
    reach or parse a Gemini response falls back to
    energy_recommendations.generate_recommendations() with the same devices/
    totals/context, so callers always get a usable list.
    """
    if devices and _api_key():
        try:
            return _call_gemini(devices, totals, crops_by_class or {})
        except Exception as exc:
            logger.warning("Gemini recommendation call failed (%s); falling back to rule-based.", exc)
    return generate_recommendations(devices, totals, context)


def main() -> None:
    import argparse

    from aria_capture import AriaCapture
    from config import CAPTURE_SOURCE_VRS, ENERGY_FRAME_SAMPLE_HZ
    from energy_detector import ApplianceScanAggregator, EnergyDetector, scan_capture_rgb
    from energy_estimator import estimate_room

    parser = argparse.ArgumentParser(description="Standalone Gemini-recommendation smoke test over a VRS file.")
    parser.add_argument("--vrs", required=True, help="Path to an Aria Gen 1 VRS recording.")
    parser.add_argument("--duration", type=float, default=None, help="Max device-time seconds to scan.")
    args = parser.parse_args()

    if not _api_key():
        raise SystemExit(f"{GEMINI_API_KEY_ENV_VAR} is not set; nothing to smoke-test.")

    detector = EnergyDetector()
    aggregator = ApplianceScanAggregator()
    capture = AriaCapture(source=CAPTURE_SOURCE_VRS, vrs_path=args.vrs)
    scan_capture_rgb(capture, detector, aggregator, duration_s=args.duration, sample_hz=ENERGY_FRAME_SAMPLE_HZ, pace_playback=True)

    estimate = estimate_room(aggregator.counts())
    recommendations = get_recommendations(estimate["devices"], estimate["totals"], aggregator.best_crops())
    print(f"\n{len(estimate['devices'])} device(s) found; recommendations:")
    for r in recommendations:
        print(f"  - {r}")


if __name__ == "__main__":
    main()
