"""Shared pytest fixtures for this test suite.

Guards against tests silently making real Gemini API calls. Several tests
reach roomscan.py:build_report() -- directly (tests/test_energy.py's
FinalizeScanTests/ReportRecommendationsTests), or indirectly via the
dashboard's Save flow (tests/test_roomscan_dashboard.py,
tests/test_roomscan_e2e_live.py) -- which calls
energy_gemini.get_recommendations(). That function only calls the real
Gemini API when GEMINI_API_KEY is set in the environment; none of those
tests intend to exercise the network path (some even assert the exact
wording of the rule-based fallback, e.g. "always-on" in
test_build_report_includes_recommendations, which only a rule-engine
response is guaranteed to contain). A developer who has GEMINI_API_KEY
exported in their shell for manual roomscan.py smoke-testing (per
CLAUDE.md) would otherwise have every `pytest` run burn real API quota and
risk flaky assertions on Gemini's non-deterministic phrasing.

This fixture removes GEMINI_API_KEY from the environment for the duration
of every test, regardless of what's set outside pytest, and restores it
afterward. Tests that specifically want to exercise the "key present" path
(tests/test_energy_gemini.py) set their own fake key via
mock.patch.dict(...) inside the test body, which layers on top of this
fixture's already-cleared environment without conflict.
"""

import os

import pytest

from config import GEMINI_API_KEY_ENV_VAR


@pytest.fixture(autouse=True)
def _no_real_gemini_calls_in_tests():
    original = os.environ.pop(GEMINI_API_KEY_ENV_VAR, None)
    try:
        yield
    finally:
        if original is not None:
            os.environ[GEMINI_API_KEY_ENV_VAR] = original
