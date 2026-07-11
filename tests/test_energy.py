"""Unit tests for the RoomScan energy-audit logic (no YOLO/torch needed).

Covers the max-simultaneous instance counting + best-crop slot logic in
energy_detector.ApplianceScanAggregator and the catalog math in
energy_estimator. Runnable with ``python -m pytest tests/`` from the repo
root, same as test_system.py.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from energy_detector import ApplianceScanAggregator, Detection
from energy_estimator import DAYS_PER_YEAR, estimate_device, estimate_room, is_appliance


def det(name: str, conf: float, box=(10, 10, 50, 50)) -> Detection:
    return Detection(class_name=name, confidence=conf, box_xyxy=box)


def frame(fill: int) -> np.ndarray:
    return np.full((100, 100, 3), fill, dtype=np.uint8)


class AggregatorTests(unittest.TestCase):
    def test_max_simultaneous_counting(self) -> None:
        agg = ApplianceScanAggregator()
        agg.observe_frame([det("tv", 0.5), det("tv", 0.4)], frame(10))
        agg.observe_frame([det("tv", 0.9)], frame(20))
        agg.observe_frame([det("laptop", 0.6)], frame(30))
        self.assertEqual(agg.counts(), {"tv": 2, "laptop": 1})
        self.assertEqual(agg.frames_observed, 3)

    def test_no_double_count_on_pan_back(self) -> None:
        agg = ApplianceScanAggregator()
        for _ in range(5):  # same single tv seen in 5 separate frames
            agg.observe_frame([det("tv", 0.5)], frame(10))
        self.assertEqual(agg.counts(), {"tv": 1})

    def test_best_crop_replaced_by_higher_confidence(self) -> None:
        agg = ApplianceScanAggregator()
        agg.observe_frame([det("tv", 0.5)], frame(10))
        agg.observe_frame([det("tv", 0.9)], frame(200))
        crops = agg.best_crops()["tv"]
        self.assertEqual(len(crops), 1)
        self.assertTrue((crops[0] == 200).all())
        self.assertEqual(agg.best_confidences()["tv"], [0.9])

    def test_lower_confidence_does_not_replace(self) -> None:
        agg = ApplianceScanAggregator()
        agg.observe_frame([det("tv", 0.9)], frame(200))
        agg.observe_frame([det("tv", 0.3)], frame(10))
        self.assertTrue((agg.best_crops()["tv"][0] == 200).all())

    def test_none_frame_updates_counts_without_crops(self) -> None:
        agg = ApplianceScanAggregator()
        agg.observe_frame([det("tv", 0.5)], None)
        self.assertEqual(agg.counts(), {"tv": 1})
        self.assertEqual(agg.best_crops()["tv"], [])

    def test_empty_scan(self) -> None:
        agg = ApplianceScanAggregator()
        agg.observe_frame([], frame(10))
        self.assertEqual(agg.counts(), {})


class EstimatorTests(unittest.TestCase):
    def test_device_math(self) -> None:
        entry = config.ENERGY_CATALOG["tv"]
        est = estimate_device("tv", 2)
        expected_unit_day = (
            entry["watts_active"] * entry["hours_per_day"]
            + entry["watts_standby"] * (24.0 - entry["hours_per_day"])
        ) / 1000.0
        self.assertAlmostEqual(est.kwh_per_day, expected_unit_day * 2)
        self.assertAlmostEqual(est.kwh_per_year, expected_unit_day * 2 * DAYS_PER_YEAR)
        self.assertAlmostEqual(
            est.cost_per_year_usd, est.kwh_per_year * config.ENERGY_COST_PER_KWH_USD
        )
        self.assertEqual(est.count, 2)

    def test_unknown_class_raises(self) -> None:
        with self.assertRaises(ValueError):
            estimate_device("sofa", 1)

    def test_bad_count_raises(self) -> None:
        with self.assertRaises(ValueError):
            estimate_device("tv", 0)

    def test_room_skips_unknown_and_sorts(self) -> None:
        result = estimate_room({"tv": 1, "sofa": 3, "oven": 1, "clock": 1})
        names = [d["class_name"] for d in result["devices"]]
        self.assertNotIn("sofa", names)
        self.assertEqual(len(names), 3)
        kwh = [d["kwh_per_year"] for d in result["devices"]]
        self.assertEqual(kwh, sorted(kwh, reverse=True))

    def test_room_totals_sum(self) -> None:
        result = estimate_room({"tv": 2, "laptop": 1})
        total = sum(d["kwh_per_year"] for d in result["devices"])
        self.assertAlmostEqual(result["totals"]["kwh_per_year"], total)
        self.assertEqual(result["totals"]["device_count"], 3)

    def test_catalog_classes_are_appliances(self) -> None:
        for name in config.ENERGY_CATALOG:
            self.assertTrue(is_appliance(name))
        self.assertFalse(is_appliance("person"))


if __name__ == "__main__":
    unittest.main()
