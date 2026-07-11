"""Unit tests for energy_gemini.py's fallback wiring (no google-genai needed).

Mirrors tests/test_energy.py's torch-free design: these tests never import
the real google-genai SDK (energy_gemini._generate_content is the only seam
that does, and it's monkeypatched here), so the suite passes whether or not
the package is installed.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from energy_estimator import estimate_room
from energy_gemini import GEMINI_API_KEY_ENV_VAR, _select_crops, get_recommendations
from energy_recommendations import generate_recommendations


def crop(fill: int) -> np.ndarray:
    return np.full((20, 20, 3), fill, dtype=np.uint8)


class NoApiKeyFallbackTests(unittest.TestCase):
    def test_falls_back_to_rules_when_key_unset(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop(GEMINI_API_KEY_ENV_VAR, None)
            result = estimate_room({"tv": 1})
            expected = generate_recommendations(result["devices"], result["totals"])
            actual = get_recommendations(result["devices"], result["totals"], {"tv": [crop(10)]})
            self.assertEqual(actual, expected)

    def test_falls_back_when_no_devices_even_with_key(self) -> None:
        with mock.patch.dict("os.environ", {GEMINI_API_KEY_ENV_VAR: "fake-key"}):
            result = estimate_room({})
            actual = get_recommendations(result["devices"], result["totals"], {})
            self.assertEqual(actual, generate_recommendations(result["devices"], result["totals"]))


class GeminiCallFailureFallbackTests(unittest.TestCase):
    def test_generate_content_exception_falls_back_to_rules(self) -> None:
        result = estimate_room({"oven": 1, "clock": 1})
        with mock.patch.dict("os.environ", {GEMINI_API_KEY_ENV_VAR: "fake-key"}):
            with mock.patch("energy_gemini._generate_content", side_effect=RuntimeError("network down")):
                actual = get_recommendations(
                    result["devices"], result["totals"], {"oven": [crop(1)], "clock": [crop(2)]}
                )
        self.assertEqual(actual, generate_recommendations(result["devices"], result["totals"]))

    def test_malformed_json_response_falls_back_to_rules(self) -> None:
        result = estimate_room({"tv": 1})
        with mock.patch.dict("os.environ", {GEMINI_API_KEY_ENV_VAR: "fake-key"}):
            with mock.patch("energy_gemini._generate_content", return_value="not valid json"):
                actual = get_recommendations(result["devices"], result["totals"], {"tv": [crop(1)]})
        self.assertEqual(actual, generate_recommendations(result["devices"], result["totals"]))

    def test_non_list_json_response_falls_back_to_rules(self) -> None:
        result = estimate_room({"tv": 1})
        with mock.patch.dict("os.environ", {GEMINI_API_KEY_ENV_VAR: "fake-key"}):
            with mock.patch("energy_gemini._generate_content", return_value=json.dumps({"not": "a list"})):
                actual = get_recommendations(result["devices"], result["totals"], {"tv": [crop(1)]})
        self.assertEqual(actual, generate_recommendations(result["devices"], result["totals"]))


class GeminiSuccessPathTests(unittest.TestCase):
    def test_parses_json_array_response(self) -> None:
        result = estimate_room({"tv": 1, "laptop": 1})
        canned = ["Turn off the TV standby light.", "Unplug the laptop charger when full."]
        with mock.patch.dict("os.environ", {GEMINI_API_KEY_ENV_VAR: "fake-key"}):
            with mock.patch("energy_gemini._generate_content", return_value=json.dumps(canned)) as mocked:
                actual = get_recommendations(
                    result["devices"], result["totals"], {"tv": [crop(1)], "laptop": [crop(2)]}
                )
        self.assertEqual(actual, canned)
        mocked.assert_called_once()

    def test_response_truncated_to_max_recommendations(self) -> None:
        result = estimate_room({"tv": 1})
        canned = [f"Suggestion {i}" for i in range(config.GEMINI_MAX_RECOMMENDATIONS + 5)]
        with mock.patch.dict("os.environ", {GEMINI_API_KEY_ENV_VAR: "fake-key"}):
            with mock.patch("energy_gemini._generate_content", return_value=json.dumps(canned)):
                actual = get_recommendations(result["devices"], result["totals"], {"tv": [crop(1)]})
        self.assertEqual(len(actual), config.GEMINI_MAX_RECOMMENDATIONS)
        self.assertEqual(actual, canned[: config.GEMINI_MAX_RECOMMENDATIONS])


class SelectCropsTests(unittest.TestCase):
    def test_prioritizes_top_devices_round_robin(self) -> None:
        result = estimate_room({"oven": 1, "clock": 1})  # oven sorts first (higher kwh/yr)
        devices = result["devices"]
        crops_by_class = {"oven": [crop(1), crop(2)], "clock": [crop(3)]}
        selected = _select_crops(devices, crops_by_class, max_crops=2)
        self.assertEqual(len(selected), 2)
        self.assertTrue((selected[0] == 1).all())  # oven's first crop chosen before oven's second

    def test_caps_at_max_crops(self) -> None:
        result = estimate_room({"tv": 1})
        crops_by_class = {"tv": [crop(1), crop(2), crop(3)]}
        selected = _select_crops(result["devices"], crops_by_class, max_crops=2)
        self.assertEqual(len(selected), 2)

    def test_missing_crops_for_a_class_does_not_crash(self) -> None:
        result = estimate_room({"tv": 1, "laptop": 1})
        selected = _select_crops(result["devices"], {"tv": [crop(1)]}, max_crops=5)
        self.assertEqual(len(selected), 1)


if __name__ == "__main__":
    unittest.main()
