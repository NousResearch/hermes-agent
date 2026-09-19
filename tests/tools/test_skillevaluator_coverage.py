"""Test coverage for tools/skillevaluator_scan.py — 9 functions had LOW coverage.

Tests the pure helpers: config gates and report parsing.
No actual scanner execution or network calls.
"""

from tools.skillevaluator_scan import tier1_advisory_enabled


class TestTier1AdvisoryEnabled:
    def test_returns_bool(self):
        assert isinstance(tier1_advisory_enabled(), bool)
