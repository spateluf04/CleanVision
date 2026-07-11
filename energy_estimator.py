"""Energy-use estimation from detected appliance counts.

Pure-logic module for the RoomScan energy audit: maps appliance class names
(COCO/YOLO names, as produced by energy_detector.py) through the wattage and
usage-hours priors in ``config.ENERGY_CATALOG`` to per-device and room-total
kWh/day, kWh/year, and yearly cost figures. No I/O, no torch, no capture
dependencies -- it is imported by roomscan.py and unit-tested directly.

The numbers are hackathon-grade typical-draw priors, not measurements; the
report layer is expected to label them as estimates.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List

from config import ENERGY_CATALOG, ENERGY_COST_PER_KWH_USD
from logging_utils import get_logger

logger = get_logger(__name__)

HOURS_PER_DAY = 24.0
DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class DeviceEstimate:
    """Energy estimate for all detected units of one appliance class."""

    class_name: str
    display_name: str
    count: int
    watts_active: float
    watts_standby: float
    hours_per_day: float
    kwh_per_day: float      # all units combined
    kwh_per_year: float     # all units combined
    cost_per_year_usd: float


def is_appliance(class_name: str) -> bool:
    """True if the detector class maps to a catalog appliance."""
    return class_name in ENERGY_CATALOG


def estimate_device(class_name: str, count: int) -> DeviceEstimate:
    """Estimate combined energy use for ``count`` units of one class.

    Daily energy per unit = active draw for its assumed active hours plus
    standby draw for the remainder of the day.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    try:
        entry = ENERGY_CATALOG[class_name]
    except KeyError as exc:
        raise ValueError(f"'{class_name}' is not in ENERGY_CATALOG") from exc

    hours = float(entry["hours_per_day"])
    watts_active = float(entry["watts_active"])
    watts_standby = float(entry["watts_standby"])
    kwh_per_unit_day = (
        watts_active * hours + watts_standby * (HOURS_PER_DAY - hours)
    ) / 1000.0
    kwh_per_day = kwh_per_unit_day * count
    kwh_per_year = kwh_per_day * DAYS_PER_YEAR
    return DeviceEstimate(
        class_name=class_name,
        display_name=str(entry["display"]),
        count=count,
        watts_active=watts_active,
        watts_standby=watts_standby,
        hours_per_day=hours,
        kwh_per_day=kwh_per_day,
        kwh_per_year=kwh_per_year,
        cost_per_year_usd=kwh_per_year * ENERGY_COST_PER_KWH_USD,
    )


def estimate_room(counts: Dict[str, int]) -> Dict[str, object]:
    """Build the full room estimate from ``{class_name: instance_count}``.

    Unknown classes are skipped with a warning (the detector should already
    filter to catalog classes, but the report must never crash on one stray
    label). Returns a JSON-ready dict: ``devices`` sorted by yearly kWh
    descending, plus ``totals``.
    """
    devices: List[DeviceEstimate] = []
    for class_name, count in counts.items():
        if not is_appliance(class_name):
            logger.warning("Skipping non-catalog class '%s' (count=%d).", class_name, count)
            continue
        if count < 1:
            continue
        devices.append(estimate_device(class_name, count))
    devices.sort(key=lambda d: d.kwh_per_year, reverse=True)

    total_kwh_day = sum(d.kwh_per_day for d in devices)
    total_kwh_year = sum(d.kwh_per_year for d in devices)
    return {
        "devices": [asdict(d) for d in devices],
        "totals": {
            "device_count": sum(d.count for d in devices),
            "kwh_per_day": total_kwh_day,
            "kwh_per_year": total_kwh_year,
            "cost_per_year_usd": total_kwh_year * ENERGY_COST_PER_KWH_USD,
            "cost_per_kwh_usd": ENERGY_COST_PER_KWH_USD,
        },
    }
