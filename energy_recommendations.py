"""Rule-based energy-saving suggestions for the RoomScan live dashboard.

Pure-logic module (no I/O, no torch, no capture dependency) that turns a
{devices, totals} estimate -- the exact shape energy_estimator.estimate_room()
already returns -- into a short list of plain-English suggestions. Also
accepts an optional ``context`` dict for signals estimate_room() doesn't
carry (currently just live frame brightness), used by rules like the
"TV in a bright room" suggestion.

Modularity: every rule is a plain function ``(devices, totals, context) ->
List[str]`` listed in ``_RULES``; ``generate_recommendations()`` just runs
them in order and concatenates. Callers only depend on
``generate_recommendations()``'s ``(devices, totals, context) -> List[str]``
signature, so the whole rule set (or ``generate_recommendations`` itself) can
later be swapped for an ML/LLM-based recommender without touching call sites
in roomscan_dashboard.py or roomscan.py.

Same hackathon-grade-priors caveat as energy_estimator.py: these are
heuristics over typical-draw assumptions and simple frame/co-occurrence
signals, not measured or vision-verified behavior -- "bright room" is a
mean-luminance threshold, not confirmed daylight or confirmed idle state, and
the "cooling inefficiency" rule needs a fan + air conditioner detection pair
that the stock COCO-trained YOLOv8n weights energy_detector.py loads today
cannot produce (see config.ENERGY_CATALOG's comment on those two entries) --
it is wired up and unit-tested with synthetic counts, ready to fire once a
detector that recognizes those classes is plugged in.
"""

from typing import Callable, Dict, List, Optional

STANDBY_SHARE_THRESHOLD = 0.5       # suggest unplugging if standby is >= half a device's yearly energy
HIGH_DRAW_WATTS_THRESHOLD = 1000.0  # active draw above which "reduce usage" is worth flagging
LOW_IMPACT_COST_USD = 25.0          # below this, there's not much to act on yet
BRIGHT_ROOM_LUMINANCE_THRESHOLD = 140.0  # mean 0-255 RGB pixel value treated as "bright room"
MULTI_SCREEN_COUNT_THRESHOLD = 2         # >= this many "tv"-class detections reads as multiple monitors
COOLING_CLASS_NAMES = ("fan", "air conditioner")
NO_DEVICES_MESSAGE = "No appliances detected yet -- keep scanning to get suggestions."

Device = Dict[str, object]
Totals = Dict[str, object]
Context = Dict[str, object]
Rule = Callable[[List[Device], Totals, Context], List[str]]


def _standby_share(device: Device) -> float:
    """Fraction of a device class's yearly energy spent on standby draw."""
    if device["kwh_per_year"] <= 0:
        return 0.0
    standby_kwh_year = (
        device["watts_standby"] * device["count"] * (24.0 - device["hours_per_day"]) * 365.0 / 1000.0
    )
    return standby_kwh_year / device["kwh_per_year"]


def _device_by_class(devices: List[Device], class_name: str) -> Optional[Device]:
    return next((d for d in devices if d["class_name"] == class_name), None)


def _rule_tv_in_bright_room(devices: List[Device], totals: Totals, context: Context) -> List[str]:
    tv = _device_by_class(devices, "tv")
    brightness = context.get("avg_brightness")
    if tv is None or brightness is None or brightness < BRIGHT_ROOM_LUMINANCE_THRESHOLD:
        return []
    return [
        f"{tv['display_name']} detected in a brightly lit room -- if it isn't actively "
        f"being watched right now, this is a good time to turn it off."
    ]


def _rule_multiple_screens(devices: List[Device], totals: Totals, context: Context) -> List[str]:
    tv = _device_by_class(devices, "tv")
    if tv is None or tv["count"] < MULTI_SCREEN_COUNT_THRESHOLD:
        return []
    return [
        f"{tv['count']} screens/monitors detected -- a multi-monitor workstation adds up to "
        f"{tv['kwh_per_year']:.0f} kWh/yr; enabling display sleep when idle helps."
    ]


def _rule_always_on_fridge(devices: List[Device], totals: Totals, context: Context) -> List[str]:
    fridge = _device_by_class(devices, "refrigerator")
    if fridge is None:
        return []
    return [
        f"{fridge['display_name']} detected -- this is an always-on load "
        f"({fridge['kwh_per_year']:.0f} kWh/yr); worth checking it isn't a redundant unit running near-empty."
    ]


def _rule_cooling_inefficiency(devices: List[Device], totals: Totals, context: Context) -> List[str]:
    present = {d["class_name"] for d in devices}
    if not set(COOLING_CLASS_NAMES).issubset(present):
        return []
    return [
        "Fan and air conditioner both detected running together -- this can mean the AC is fighting "
        "an open door/window or a fan left on unnecessarily; check both before leaving the room."
    ]


def _rule_biggest_draw(devices: List[Device], totals: Totals, context: Context) -> List[str]:
    top = devices[0]
    if top["watts_active"] < HIGH_DRAW_WATTS_THRESHOLD:
        return []
    return [
        f"{top['display_name']} is your biggest draw ({top['kwh_per_year']:.0f} kWh/yr) and pulls "
        f"{top['watts_active']:.0f} W when active -- cutting daily use time has the largest impact here."
    ]


def _rule_standby_heavy(devices: List[Device], totals: Totals, context: Context) -> List[str]:
    suggestions: List[str] = []
    for device in devices:
        if device["watts_standby"] > 0 and _standby_share(device) >= STANDBY_SHARE_THRESHOLD:
            suggestions.append(
                f"{device['display_name']} spends most of its energy on standby "
                f"({_standby_share(device) * 100:.0f}% of {device['kwh_per_year']:.0f} kWh/yr) -- "
                f"a power strip to fully cut it off when idle could help."
            )
    return suggestions


def _rule_low_impact_fallback(devices: List[Device], totals: Totals, context: Context) -> List[str]:
    if totals["cost_per_year_usd"] >= LOW_IMPACT_COST_USD:
        return []
    return [
        f"Estimated cost so far is modest (${totals['cost_per_year_usd']:.0f}/yr) -- "
        f"keep scanning to build a fuller picture of this room."
    ]


# Order: device+context co-occurrence rules first (most specific/actionable),
# then the original threshold-based rules, with the "keep scanning" fallback
# last since it's a catch-all rather than a specific action.
_RULES: List[Rule] = [
    _rule_tv_in_bright_room,
    _rule_multiple_screens,
    _rule_always_on_fridge,
    _rule_cooling_inefficiency,
    _rule_biggest_draw,
    _rule_standby_heavy,
    _rule_low_impact_fallback,
]


def generate_recommendations(
    devices: List[Device],
    totals: Totals,
    context: Optional[Context] = None,
) -> List[str]:
    """Return a short list of rule-based suggestions for the current scan state.

    ``devices`` is expected sorted by kwh_per_year descending (as
    estimate_room() already returns it), so ``devices[0]`` is the top drain.
    ``context`` carries live signals estimate_room() doesn't produce (e.g.
    ``avg_brightness``); omit it (or pass {}) when none are available, such
    as when recomputing recommendations for a finished session's report.
    """
    if not devices:
        return [NO_DEVICES_MESSAGE]
    context = context or {}
    suggestions: List[str] = []
    for rule in _RULES:
        suggestions.extend(rule(devices, totals, context))
    return suggestions
