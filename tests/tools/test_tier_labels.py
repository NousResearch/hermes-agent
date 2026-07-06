#!/usr/bin/env python3
"""Tests for delegation tier labels (tools/tier_labels.py).

Behavior contract, not a snapshot: asserts how a (model, effort) pair maps to a
tier class, and that cost-class dominates reasoning effort so cheap models are
never labelled heavy.

Run with:  python -m pytest tests/tools/test_tier_labels.py -v
"""

import unittest

from tools.tier_labels import derive_tier, derive_tier_from_model


class TestDeriveTierFromModel(unittest.TestCase):
    def test_opus_any_effort_is_heavy(self):
        self.assertEqual(derive_tier_from_model("us.anthropic.claude-opus-4-8", "medium"), "heavy")
        self.assertEqual(derive_tier_from_model("us.anthropic.claude-opus-4-8", "high"), "heavy")
        self.assertEqual(derive_tier_from_model("us.anthropic.claude-opus-4-8", "low"), "heavy")

    def test_gpt55_heavy_effort_is_heavy(self):
        self.assertEqual(derive_tier_from_model("gpt-5.5", "xhigh"), "heavy")
        self.assertEqual(derive_tier_from_model("gpt-5.5", "high"), "heavy")

    def test_gpt55_low_medium_is_mid(self):
        self.assertEqual(derive_tier_from_model("gpt-5.5", "medium"), "mid")
        self.assertEqual(derive_tier_from_model("gpt-5.5", "low"), "mid")

    def test_mini_is_never_heavy(self):
        # The B3 regression: a cheap mini lane at high effort must be light.
        self.assertEqual(derive_tier_from_model("gpt-5.4-mini", "high"), "light")
        self.assertEqual(derive_tier_from_model("gpt-5.4-mini", "xhigh"), "light")

    def test_haiku_is_light(self):
        self.assertEqual(derive_tier_from_model("us.anthropic.claude-haiku-4-5", "high"), "light")

    def test_sonnet_low_is_light_medium_is_mid(self):
        self.assertEqual(derive_tier_from_model("us.anthropic.claude-sonnet-4-6", "low"), "light")
        self.assertEqual(derive_tier_from_model("us.anthropic.claude-sonnet-4-6", "medium"), "mid")

    def test_empty_model_is_unprofiled(self):
        self.assertEqual(derive_tier_from_model("", "high"), "unprofiled")
        self.assertEqual(derive_tier_from_model(None, "high"), "unprofiled")


class TestDeriveTierWithCfg(unittest.TestCase):
    """Tier derivation via merged delegation cfg (the spawn-time path)."""

    def _t(self, profile, model, effort):
        return derive_tier(profile, {"model": model, "reasoning_effort": effort})

    def test_real_config_profiles(self):
        # Mirrors the plan's sanity table against real config.yaml lanes.
        cases = {
            ("oracle", "us.anthropic.claude-opus-4-8", "high"): "heavy",
            ("coder", "gpt-5.5", "xhigh"): "heavy",
            ("reviewer-opus", "us.anthropic.claude-opus-4-8", "medium"): "heavy",
            ("researcher", "gpt-5.4-mini", "high"): "light",
            ("reviewer-codex55-medium", "gpt-5.5", "medium"): "mid",
            ("verify-php-standards", "gpt-5.5", "low"): "mid",
            ("file-explorer", "us.anthropic.claude-sonnet-4-6", "low"): "light",
        }
        for (profile, model, effort), expected in cases.items():
            with self.subTest(profile=profile):
                self.assertEqual(self._t(profile, model, effort), expected)

    def test_bare_delegation_is_unprofiled(self):
        # No profile name => unprofiled, even if cfg carries an inherited model.
        self.assertEqual(derive_tier(None, {"model": "us.anthropic.claude-opus-4-8"}), "unprofiled")
        self.assertEqual(derive_tier("", {"model": "gpt-5.5", "reasoning_effort": "high"}), "unprofiled")


if __name__ == "__main__":
    unittest.main()
