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
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import (
    GEMINI_API_KEY_ENV_VAR,
    GEMINI_BULB_TYPE_WATTS,
    GEMINI_DISCOVERY_DEFAULT_COUNT,
    GEMINI_DISCOVERY_DEFAULT_HOURS_PER_DAY,
    GEMINI_DISCOVERY_DEFAULT_WATTS,
    GEMINI_DISCOVERY_MAX_COUNT,
    GEMINI_DISCOVERY_MAX_DIM,
    GEMINI_DISCOVERY_MAX_WATTS,
    GEMINI_MAX_CROPS,
    GEMINI_MAX_DISCOVERED,
    GEMINI_MAX_RECOMMENDATIONS,
    GEMINI_MODEL,
    GEMINI_NOTE_MAX_CHARS,
    GEMINI_TIMEOUT_SECONDS,
)
from energy_recommendations import generate_recommendations
from logging_utils import get_logger

logger = get_logger(__name__)

Device = Dict[str, object]
Totals = Dict[str, object]
Context = Dict[str, object]
# (class_name, slot_index, confidence, crop_rgb) -- matches
# ApplianceScanAggregator.unverified_slots()'s return shape, so
# roomscan_live.py can pass a slice of it straight in.
VerifyCandidate = Tuple[str, int, float, np.ndarray]

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


def ai_features_enabled() -> bool:
    """True when GEMINI_API_KEY is set -- the single gate for every optional
    Gemini feature in this module (recommendations, live verify/discover)."""
    return _api_key() is not None


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


def _call_gemini_raw(prompt: str, image_jpegs: List[bytes]) -> str:
    """Run one Gemini call with a hard timeout. The sole seam shared by every
    higher-level Gemini call in this module (recommendations, live pass)."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_generate_content, prompt, image_jpegs)
        try:
            return future.result(timeout=GEMINI_TIMEOUT_SECONDS)
        except FutureTimeoutError as exc:
            raise TimeoutError(f"Gemini call exceeded {GEMINI_TIMEOUT_SECONDS}s") from exc


def _call_gemini(
    devices: List[Device],
    totals: Totals,
    crops_by_class: Dict[str, List[np.ndarray]],
) -> List[str]:
    prompt = _build_prompt(totals, devices)
    crops = _select_crops(devices, crops_by_class, GEMINI_MAX_CROPS)
    image_jpegs = [_encode_crop_jpeg(c) for c in crops]

    text = _call_gemini_raw(prompt, image_jpegs)
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


_LIVE_PASS_PROMPT_TEMPLATE = """You are assisting a live in-home walkthrough energy scan. A YOLO object \
detector has tentatively identified some appliances. You have two jobs this pass:

1. VERIFY -- for each numbered candidate crop below, look at its photo and judge whether it \
plausibly IS the labeled appliance type. Only flag a candidate as not matching when you are \
confident the detector mislabeled the object (e.g. an armchair labeled "tv"). If you aren't sure, \
say it matches -- false rejections are worse than a missed catch. Whether or not it matches, if \
you can tell a more specific type/model/size detail from the photo (e.g. "55-inch wall-mounted \
LED TV", "13-inch silver laptop"), include it as a short "note" string; omit "note" (or leave it \
empty) if you can't tell anything more specific than the label already says.
Candidates (index: detector label):
{candidate_list}

2. DISCOVER -- {discover_instruction}

Appliance types already tracked (do not list these again as discovered): {known_names}

Respond with ONLY a JSON object of this exact shape:
{{"verifications": [{{"index": 0, "matches": true, "note": "..."}}], "discovered": [{{"name": "...", \
"description": "...", "estimated_watts": 10, "estimated_hours_per_day": 4, "count": 1}}]}}
Include a "verifications" entry only for candidates you have an opinion on. Return at most \
{max_discovered} "discovered" entries. No markdown, no numbering, no commentary outside the JSON object."""


def _bulb_type_default_watts(description: str) -> Optional[float]:
    """Look up a reference wattage from a bulb/fixture-type keyword mentioned
    in a discovered lighting item's description (e.g. "LED ceiling light"),
    for use only when Gemini's own estimated_watts is missing/invalid --
    a much closer guess than one flat default across every bulb type."""
    lowered = description.lower()
    for keyword, watts in GEMINI_BULB_TYPE_WATTS.items():
        if keyword in lowered:
            return watts
    return None


def _build_live_pass_prompt(
    verify_candidates: List[VerifyCandidate],
    known_display_names: Sequence[str],
    discover: bool,
) -> str:
    candidate_list = "\n".join(
        f"{i}: {class_name} (detector confidence {confidence:.2f})"
        for i, (class_name, _slot_index, confidence, _crop) in enumerate(verify_candidates)
    ) or "(none this pass)"
    discover_instruction = (
        "the final attached photo is a wide shot of the room. Look for additional energy-relevant "
        "items NOT already covered by the candidates above or the known types listed below. Critically, "
        "if an item in this wide photo is the SAME physical object as one of the numbered candidates "
        "above -- even if you would naturally call it something else (e.g. a computer monitor that "
        "detector candidate #0 already labeled \"tv\") -- do NOT list it again here under a new name; "
        "it is already being counted once via that candidate and must not be counted a second time. "
        "Only report genuinely separate objects the candidates list didn't already point at. This "
        "includes plugged-in or battery-powered appliances (e.g. dishwasher, kettle, space heater, "
        "microwave, water heater, humidifier, dehumidifier, pool/spa pump), lighting (e.g. ceiling "
        "light, lamp, recessed/pot light) -- note the bulb/fixture type in the description if you can "
        "tell (LED, CFL, fluorescent tube, incandescent), HVAC elements (e.g. AC vent, window/wall AC "
        "unit, radiator, baseboard heater), always-on standby/phantom loads (e.g. a power strip or "
        "charger left plugged in with no device attached, a smart speaker or smart-home hub), and "
        "small idle electronics (e.g. router, modem, game console left on). Be thorough "
        "and precise, especially for lighting and vents: scan the entire photo edge-to-edge -- "
        "ceiling, walls, and corners -- and do not skip any visible fixture. For each distinct type "
        "of item (e.g. \"Ceiling Light\", \"Wall Vent\"), report exactly how many separate individual "
        "instances of that exact type are visible in this photo right now as an integer \"count\" "
        "field (e.g. 3 separate ceiling lights in view -> \"count\": 3); do not merge multiple visible "
        "fixtures into a count of 1, and do not guess at fixtures you cannot actually see. List short "
        "names with a one-line description each, including type/model detail when visible. Also "
        "estimate its typical active power draw in watts for ONE unit (estimated_watts) and typical "
        "daily usage hours between 0 and 24 (estimated_hours_per_day), based on the item's type and "
        "how it's normally used (e.g. an LED ceiling light might be ~10W for 5h/day; an AC vent has "
        "no draw of its own -- estimate the fan/blower behind it, or use a small nominal wattage)."
        if discover
        else "no room photo was provided this pass -- leave \"discovered\" as an empty list."
    )
    return _LIVE_PASS_PROMPT_TEMPLATE.format(
        candidate_list=candidate_list,
        discover_instruction=discover_instruction,
        known_names=", ".join(known_display_names) or "(none)",
        max_discovered=GEMINI_MAX_DISCOVERED,
    )


def _downscale_for_discovery(frame_rgb: np.ndarray, max_dim: int = GEMINI_DISCOVERY_MAX_DIM) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return frame_rgb
    scale = max_dim / float(longest)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(frame_rgb, new_size, interpolation=cv2.INTER_AREA)


def run_live_scan_pass(
    verify_candidates: List[VerifyCandidate],
    discovery_frame_rgb: Optional[np.ndarray],
    known_display_names: Sequence[str],
) -> Dict[str, List]:
    """One combined Gemini vision pass for a running live scan: re-check crops
    that might be misclassified, and look for appliance types outside YOLO's
    COCO vocabulary.

    ``verify_candidates`` should already be capped by the caller (e.g. to
    GEMINI_VERIFY_MAX_CROPS) -- this function does not re-cap it.
    ``discovery_frame_rgb`` is the latest full upright RGB frame, or None to
    skip discovery entirely (e.g. no frame available yet).

    Returns {"verifications": [(class_name, slot_index, confidence, accepted, note)],
    "discovered": [{"name": str, "description": str, "watts_active": float,
    "hours_per_day": float, "count": int}]}. ``note`` is an optional short
    type/model detail string, or None if Gemini didn't offer one.
    ``watts_active``/``hours_per_day`` are Gemini's vision-estimated typical
    per-unit draw/usage for a discovered non-catalog device, and ``count`` is
    how many individual instances of that exact type Gemini reports seeing in
    this frame (e.g. 3 separate ceiling lights); roomscan.py:
    merge_discovered_devices() prices watts_active/hours_per_day once and
    scales by count, the same way a catalog device's per-unit draw is scaled
    by its instance count. Missing or out-of-range values fall back to
    GEMINI_DISCOVERY_DEFAULT_WATTS/HOURS_PER_DAY/COUNT, except a missing/
    invalid watts value falls back to a bulb-type-specific reference wattage
    (GEMINI_BULB_TYPE_WATTS) when the description names a known bulb/fixture
    type (LED, CFL, halogen, fluorescent, incandescent), which is a closer
    guess than the flat default for lighting. Never raises: on ANY
    failure (no key, nothing to ask, network error, timeout, malformed
    response) both lists come back empty, exactly like a pass that found
    nothing to say -- callers don't need a separate error path.
    """
    empty: Dict[str, List] = {"verifications": [], "discovered": []}
    if not verify_candidates and discovery_frame_rgb is None:
        return empty
    if not _api_key():
        return empty

    prompt = _build_live_pass_prompt(verify_candidates, known_display_names, discover=discovery_frame_rgb is not None)
    image_jpegs = [_encode_crop_jpeg(crop) for _, _, _, crop in verify_candidates]
    if discovery_frame_rgb is not None:
        image_jpegs.append(_encode_crop_jpeg(_downscale_for_discovery(discovery_frame_rgb)))

    try:
        text = _call_gemini_raw(prompt, image_jpegs)
        parsed = json.loads(text)
    except Exception as exc:
        logger.warning("Gemini live-scan pass failed (%s); skipping this pass.", exc)
        return empty
    if not isinstance(parsed, dict):
        logger.warning("Gemini live-scan pass response was not a JSON object: %r", text)
        return empty

    verifications: List[Tuple[str, int, float, bool, Optional[str]]] = []
    raw_verifications = parsed.get("verifications")
    for item in raw_verifications if isinstance(raw_verifications, list) else []:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        matches = item.get("matches")
        if not isinstance(index, int) or isinstance(index, bool) or not isinstance(matches, bool):
            continue
        note = item.get("note")
        note = note.strip()[:GEMINI_NOTE_MAX_CHARS] if isinstance(note, str) and note.strip() else None
        if 0 <= index < len(verify_candidates):
            class_name, slot_index, confidence, _crop = verify_candidates[index]
            verifications.append((class_name, slot_index, confidence, matches, note))

    discovered: List[Dict[str, str]] = []
    # Pre-seed with already-known display names (case-insensitive) so a
    # re-discovery of a type we already track never shows up as "new" --
    # exact-lowercase-match only, no fuzzy matching across ticks.
    seen_names = {n.lower() for n in known_display_names}
    raw_discovered = parsed.get("discovered")
    for item in raw_discovered if isinstance(raw_discovered, list) else []:
        if len(discovered) >= GEMINI_MAX_DISCOVERED:
            break
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        key = name.lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        description = item.get("description")
        description_str = description.strip() if isinstance(description, str) else ""
        watts = item.get("estimated_watts")
        if isinstance(watts, (int, float)) and not isinstance(watts, bool):
            watts_active = max(0.0, min(float(watts), GEMINI_DISCOVERY_MAX_WATTS))
        else:
            watts_active = _bulb_type_default_watts(description_str) or GEMINI_DISCOVERY_DEFAULT_WATTS
        hours = item.get("estimated_hours_per_day")
        hours_per_day = (
            max(0.0, min(float(hours), 24.0))
            if isinstance(hours, (int, float)) and not isinstance(hours, bool)
            else GEMINI_DISCOVERY_DEFAULT_HOURS_PER_DAY
        )
        raw_count = item.get("count")
        count = (
            max(1, min(int(raw_count), GEMINI_DISCOVERY_MAX_COUNT))
            if isinstance(raw_count, int) and not isinstance(raw_count, bool)
            else GEMINI_DISCOVERY_DEFAULT_COUNT
        )
        discovered.append({
            "name": name,
            "description": description_str,
            "watts_active": watts_active,
            "hours_per_day": hours_per_day,
            "count": count,
        })

    return {"verifications": verifications, "discovered": discovered}


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
