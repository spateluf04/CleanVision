"""Unit tests for energy_gemini.py's fallback wiring (no google-genai needed).

Mirrors tests/test_energy.py's torch-free design: these tests never import
the real google-genai SDK (energy_gemini._generate_content is the only seam
that does, and it's monkeypatched here), so the suite passes whether or not
the package is installed.
"""

from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from energy_estimator import estimate_room
from energy_gemini import (
    GEMINI_API_KEY_ENV_VAR,
    _build_live_pass_prompt,
    _select_crops,
    get_recommendations,
    run_live_scan_pass,
)
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


class BuildLivePassPromptTests(unittest.TestCase):
    def test_discover_instruction_covers_expanded_categories(self) -> None:
        prompt = _build_live_pass_prompt([], ["TV"], discover=True)
        for keyword in (
            "water heater",
            "humidifier",
            "dehumidifier",
            "pool/spa pump",
            "power strip",
            "smart speaker",
            "smart-home hub",
        ):
            self.assertIn(keyword, prompt)

    def test_no_discover_instruction_skips_expanded_categories(self) -> None:
        prompt = _build_live_pass_prompt([], ["TV"], discover=False)
        self.assertNotIn("water heater", prompt)
        self.assertIn("no room photo was provided", prompt)

    def test_discover_instruction_warns_against_recounting_verified_candidates(self) -> None:
        prompt = _build_live_pass_prompt([("tv", 0, 0.6, crop(1))], ["TV"], discover=True)
        self.assertIn("SAME physical object", prompt)
        self.assertIn("do NOT list it again", prompt)

    def test_discover_instruction_covers_wall_outlets(self) -> None:
        prompt = _build_live_pass_prompt([], ["TV"], discover=True)
        self.assertIn("wall outlet", prompt)
        self.assertIn("Wall Outlet", prompt)


class RunLiveScanPassTests(unittest.TestCase):
    def test_nothing_to_ask_short_circuits_without_key_check(self) -> None:
        # No candidates and no discovery frame -- should bail before ever
        # touching os.environ, let alone the network.
        with mock.patch("energy_gemini._call_gemini_raw") as mocked:
            result = run_live_scan_pass([], None, [])
        self.assertEqual(result, {"verifications": [], "discovered": []})
        mocked.assert_not_called()

    def test_no_key_returns_empty_without_calling_gemini(self) -> None:
        candidates = [("tv", 0, 0.4, crop(1))]
        with mock.patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop(GEMINI_API_KEY_ENV_VAR, None)
            with mock.patch("energy_gemini._call_gemini_raw") as mocked:
                result = run_live_scan_pass(candidates, None, ["TV"])
        self.assertEqual(result, {"verifications": [], "discovered": []})
        mocked.assert_not_called()

    def test_call_failure_falls_back_to_empty(self) -> None:
        candidates = [("tv", 0, 0.4, crop(1))]
        with mock.patch.dict("os.environ", {GEMINI_API_KEY_ENV_VAR: "fake-key"}):
            with mock.patch("energy_gemini._generate_content", side_effect=RuntimeError("network down")):
                result = run_live_scan_pass(candidates, None, ["TV"])
        self.assertEqual(result, {"verifications": [], "discovered": []})

    def test_malformed_json_falls_back_to_empty(self) -> None:
        candidates = [("tv", 0, 0.4, crop(1))]
        with mock.patch.dict("os.environ", {GEMINI_API_KEY_ENV_VAR: "fake-key"}):
            with mock.patch("energy_gemini._generate_content", return_value="not valid json"):
                result = run_live_scan_pass(candidates, None, ["TV"])
        self.assertEqual(result, {"verifications": [], "discovered": []})

    def test_non_object_json_falls_back_to_empty(self) -> None:
        candidates = [("tv", 0, 0.4, crop(1))]
        with mock.patch.dict("os.environ", {GEMINI_API_KEY_ENV_VAR: "fake-key"}):
            with mock.patch("energy_gemini._generate_content", return_value=json.dumps(["not", "an", "object"])):
                result = run_live_scan_pass(candidates, None, ["TV"])
        self.assertEqual(result, {"verifications": [], "discovered": []})

    def test_verifications_parsed_by_index_including_partial_and_out_of_range(self) -> None:
        candidates = [("tv", 0, 0.4, crop(1)), ("laptop", 2, 0.6, crop(2))]
        canned = {
            "verifications": [
                {"index": 0, "matches": False},
                {"index": 1, "matches": True},
                {"index": 99, "matches": True},  # out of range -- ignored
                {"index": "1", "matches": True},  # wrong type -- ignored
                {"matches": True},  # missing index -- ignored
            ],
            "discovered": [],
        }
        with mock.patch.dict("os.environ", {GEMINI_API_KEY_ENV_VAR: "fake-key"}):
            with mock.patch("energy_gemini._generate_content", return_value=json.dumps(canned)):
                result = run_live_scan_pass(candidates, None, [])
        self.assertEqual(
            result["verifications"],
            [("tv", 0, 0.4, False, None), ("laptop", 2, 0.6, True, None)],
        )
        self.assertEqual(result["discovered"], [])

    def test_note_extracted_and_truncated(self) -> None:
        candidates = [("tv", 0, 0.4, crop(1)), ("laptop", 2, 0.6, crop(2))]
        long_note = "x" * (config.GEMINI_NOTE_MAX_CHARS + 20)
        canned = {
            "verifications": [
                {"index": 0, "matches": True, "note": "  55-inch wall-mounted LED TV  "},
                {"index": 1, "matches": True, "note": long_note},
            ],
            "discovered": [],
        }
        with mock.patch.dict("os.environ", {GEMINI_API_KEY_ENV_VAR: "fake-key"}):
            with mock.patch("energy_gemini._generate_content", return_value=json.dumps(canned)):
                result = run_live_scan_pass(candidates, None, [])
        verifications = result["verifications"]
        self.assertEqual(verifications[0], ("tv", 0, 0.4, True, "55-inch wall-mounted LED TV"))
        self.assertEqual(verifications[1], ("laptop", 2, 0.6, True, "x" * config.GEMINI_NOTE_MAX_CHARS))

    def test_blank_or_missing_note_normalizes_to_none(self) -> None:
        candidates = [("tv", 0, 0.4, crop(1))]
        canned = {
            "verifications": [{"index": 0, "matches": True, "note": "   "}],
            "discovered": [],
        }
        with mock.patch.dict("os.environ", {GEMINI_API_KEY_ENV_VAR: "fake-key"}):
            with mock.patch("energy_gemini._generate_content", return_value=json.dumps(canned)):
                result = run_live_scan_pass(candidates, None, [])
        self.assertEqual(result["verifications"], [("tv", 0, 0.4, True, None)])

    def test_discovered_dedup_against_known_display_names(self) -> None:
        canned = {
            "verifications": [],
            "discovered": [
                {"name": "Kettle", "description": "Electric kettle on the counter."},
                {"name": "tv", "description": "Should be excluded -- already known."},
                {"name": "Kettle", "description": "Duplicate within this same response."},
                {"name": "   ", "description": "Blank name -- excluded."},
            ],
        }
        with mock.patch.dict("os.environ", {GEMINI_API_KEY_ENV_VAR: "fake-key"}):
            with mock.patch("energy_gemini._generate_content", return_value=json.dumps(canned)):
                result = run_live_scan_pass([], np.zeros((10, 10, 3), dtype=np.uint8), ["TV", "Laptop"])
        self.assertEqual(
            result["discovered"],
            [{
                "name": "Kettle",
                "description": "Electric kettle on the counter.",
                "watts_active": config.GEMINI_DISCOVERY_DEFAULT_WATTS,
                "hours_per_day": config.GEMINI_DISCOVERY_DEFAULT_HOURS_PER_DAY,
                "count": config.GEMINI_DISCOVERY_DEFAULT_COUNT,
            }],
        )

    def test_discovered_capped_at_max_discovered(self) -> None:
        canned = {
            "verifications": [],
            "discovered": [{"name": f"Gadget {i}", "description": ""} for i in range(config.GEMINI_MAX_DISCOVERED + 3)],
        }
        with mock.patch.dict("os.environ", {GEMINI_API_KEY_ENV_VAR: "fake-key"}):
            with mock.patch("energy_gemini._generate_content", return_value=json.dumps(canned)):
                result = run_live_scan_pass([], np.zeros((10, 10, 3), dtype=np.uint8), [])
        self.assertEqual(len(result["discovered"]), config.GEMINI_MAX_DISCOVERED)

    def test_discovered_estimated_watts_and_hours_parsed_and_clamped(self) -> None:
        canned = {
            "verifications": [],
            "discovered": [
                {"name": "Ceiling Light", "description": "LED.", "estimated_watts": 10, "estimated_hours_per_day": 5},
                {"name": "Space Heater", "description": "", "estimated_watts": 999999, "estimated_hours_per_day": 30},
                {"name": "Weird Gadget", "description": "", "estimated_watts": -5, "estimated_hours_per_day": -1},
                {"name": "No Estimate", "description": ""},
            ],
        }
        with mock.patch.dict("os.environ", {GEMINI_API_KEY_ENV_VAR: "fake-key"}):
            with mock.patch("energy_gemini._generate_content", return_value=json.dumps(canned)):
                result = run_live_scan_pass([], np.zeros((10, 10, 3), dtype=np.uint8), [])
        by_name = {d["name"]: d for d in result["discovered"]}
        self.assertEqual(by_name["Ceiling Light"]["watts_active"], 10.0)
        self.assertEqual(by_name["Ceiling Light"]["hours_per_day"], 5.0)
        self.assertEqual(by_name["Space Heater"]["watts_active"], config.GEMINI_DISCOVERY_MAX_WATTS)
        self.assertEqual(by_name["Space Heater"]["hours_per_day"], 24.0)
        self.assertEqual(by_name["Weird Gadget"]["watts_active"], 0.0)
        self.assertEqual(by_name["Weird Gadget"]["hours_per_day"], 0.0)
        self.assertEqual(by_name["No Estimate"]["watts_active"], config.GEMINI_DISCOVERY_DEFAULT_WATTS)
        self.assertEqual(by_name["No Estimate"]["hours_per_day"], config.GEMINI_DISCOVERY_DEFAULT_HOURS_PER_DAY)

    def test_discovered_missing_watts_falls_back_to_bulb_type_wattage(self) -> None:
        canned = {
            "verifications": [],
            "discovered": [
                {"name": "Ceiling Light", "description": "An LED ceiling fixture."},
                {"name": "Old Lamp", "description": "Incandescent bulb in a floor lamp."},
                {"name": "Tube Light", "description": "Fluorescent tube over the workbench."},
                {"name": "Compact Bulb", "description": "Compact fluorescent (CFL) bulb."},
            ],
        }
        with mock.patch.dict("os.environ", {GEMINI_API_KEY_ENV_VAR: "fake-key"}):
            with mock.patch("energy_gemini._generate_content", return_value=json.dumps(canned)):
                result = run_live_scan_pass([], np.zeros((10, 10, 3), dtype=np.uint8), [])
        by_name = {d["name"]: d for d in result["discovered"]}
        self.assertEqual(by_name["Ceiling Light"]["watts_active"], config.GEMINI_BULB_TYPE_WATTS["led"])
        self.assertEqual(by_name["Old Lamp"]["watts_active"], config.GEMINI_BULB_TYPE_WATTS["incandescent"])
        self.assertEqual(by_name["Tube Light"]["watts_active"], config.GEMINI_BULB_TYPE_WATTS["fluorescent"])
        self.assertEqual(by_name["Compact Bulb"]["watts_active"], config.GEMINI_BULB_TYPE_WATTS["cfl"])

        canned_unknown = {
            "verifications": [],
            "discovered": [{"name": "Mystery Fixture", "description": "A light fixture of unclear type."}],
        }
        with mock.patch.dict("os.environ", {GEMINI_API_KEY_ENV_VAR: "fake-key"}):
            with mock.patch("energy_gemini._generate_content", return_value=json.dumps(canned_unknown)):
                result_unknown = run_live_scan_pass([], np.zeros((10, 10, 3), dtype=np.uint8), [])
        self.assertEqual(result_unknown["discovered"][0]["watts_active"], config.GEMINI_DISCOVERY_DEFAULT_WATTS)

    def test_discovered_explicit_watts_override_bulb_type_fallback(self) -> None:
        canned = {
            "verifications": [],
            "discovered": [
                {"name": "Ceiling Light", "description": "LED fixture.", "estimated_watts": 12},
            ],
        }
        with mock.patch.dict("os.environ", {GEMINI_API_KEY_ENV_VAR: "fake-key"}):
            with mock.patch("energy_gemini._generate_content", return_value=json.dumps(canned)):
                result = run_live_scan_pass([], np.zeros((10, 10, 3), dtype=np.uint8), [])
        self.assertEqual(result["discovered"][0]["watts_active"], 12.0)

    def test_discovered_count_parsed_defaulted_and_clamped(self) -> None:
        canned = {
            "verifications": [],
            "discovered": [
                {"name": "Ceiling Light", "description": "", "count": 3},
                {"name": "Runaway Count", "description": "", "count": 99999},
                {"name": "Zero Count", "description": "", "count": 0},
                {"name": "Bad Count Type", "description": "", "count": "three"},
            ],
        }
        with mock.patch.dict("os.environ", {GEMINI_API_KEY_ENV_VAR: "fake-key"}):
            with mock.patch("energy_gemini._generate_content", return_value=json.dumps(canned)):
                result = run_live_scan_pass([], np.zeros((10, 10, 3), dtype=np.uint8), [])
        by_name = {d["name"]: d for d in result["discovered"]}
        self.assertEqual(by_name["Ceiling Light"]["count"], 3)
        self.assertEqual(by_name["Runaway Count"]["count"], config.GEMINI_DISCOVERY_MAX_COUNT)
        self.assertEqual(by_name["Zero Count"]["count"], 1)
        self.assertEqual(by_name["Bad Count Type"]["count"], config.GEMINI_DISCOVERY_DEFAULT_COUNT)

        with mock.patch.dict("os.environ", {GEMINI_API_KEY_ENV_VAR: "fake-key"}):
            with mock.patch(
                "energy_gemini._generate_content",
                return_value=json.dumps({"verifications": [], "discovered": [{"name": "No Count", "description": ""}]}),
            ):
                result = run_live_scan_pass([], np.zeros((10, 10, 3), dtype=np.uint8), [])
        self.assertEqual(result["discovered"][0]["count"], config.GEMINI_DISCOVERY_DEFAULT_COUNT)

    def test_timeout_falls_back_to_empty(self) -> None:
        candidates = [("tv", 0, 0.4, crop(1))]

        def slow_call(*_args: object, **_kwargs: object) -> str:
            time.sleep(0.3)
            return json.dumps({"verifications": [], "discovered": []})

        with mock.patch.dict("os.environ", {GEMINI_API_KEY_ENV_VAR: "fake-key"}):
            with mock.patch("energy_gemini.GEMINI_TIMEOUT_SECONDS", 0.05):
                with mock.patch("energy_gemini._generate_content", side_effect=slow_call):
                    result = run_live_scan_pass(candidates, None, [])
        self.assertEqual(result, {"verifications": [], "discovered": []})


if __name__ == "__main__":
    unittest.main()
