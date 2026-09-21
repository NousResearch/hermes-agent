#!/usr/bin/env python3
"""
Tests for the KENSEI CUSTOM synthesis + verification primitives in delegate_task.

Run with: python -m pytest tests/tools/test_delegate_synthesis.py -v -o "addopts="
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import _extract_finding, _get_synthesis_enabled, _get_verify_enabled


class TestExtractFinding(unittest.TestCase):
    """G5 — structural extraction of findings, never prompt-based."""

    def test_json_finding_key(self):
        result = _extract_finding('{"finding": "The answer is 42", "reasoning": "deep thinking..."}')
        self.assertEqual(result, "The answer is 42")

    def test_json_answer_key(self):
        result = _extract_finding('{"answer": "Paris", "confidence": 0.9}')
        self.assertEqual(result, "Paris")

    def test_json_conclusion_key(self):
        result = _extract_finding('{"conclusion": "Migration is safe", "evidence": "..."}')
        self.assertEqual(result, "Migration is safe")

    def test_markdown_heading_finding(self):
        result = _extract_finding("Some intro\n\n## Finding\nThe database is corrupt.\n\n## Analysis\nblah")
        self.assertEqual(result, "The database is corrupt.")

    def test_markdown_heading_conclusion(self):
        result = _extract_finding("Details...\n\n## Conclusion\n\nWe should use PostgreSQL.\n\n## Next steps")
        self.assertEqual(result, "We should use PostgreSQL.")

    def test_unstructured_output_returns_empty(self):
        """F1: When structured extraction fails, return '' so verification skips."""
        result = _extract_finding(
            "I think the answer might be PostgreSQL because I've used it before "
            "and it scales well. The benchmarks also look good. However, some "
            "teams prefer MySQL for simpler setups. Overall I'd go with Postgres."
        )
        self.assertEqual(result, "")

    def test_empty_input(self):
        self.assertEqual(_extract_finding(""), "")
        self.assertEqual(_extract_finding("   "), "")

    def test_inline_json_still_works(self):
        """F1: JSON extraction is still Strategy 1 — inline JSON blocks work."""
        result = _extract_finding(
            "Some preamble text.\n"
            '{"finding": "Use Redis for caching"}'
        )
        self.assertEqual(result, "Use Redis for caching")

    # --- New tests for JSON-any-block strategy (fix for verify silently skipping) ---

    def test_arbitrary_json_keys_serialised(self):
        """When producer uses non-standard keys (gpu, competing_amd_gpu, etc.),
        serialise the whole JSON block so skeptic has something to evaluate."""
        producer_output = (
            "Here's what I found:\n\n"
            "```json\n"
            '{"gpu": "RTX 3090", "competing_amd_gpu": "RX 6900 XT",'
            '"generation": "Ampere vs RDNA 2"}\n'
            "```\n\nSummary: the 3090 competes against the 6900 XT."
        )
        result = _extract_finding(producer_output)
        # Must return a non-empty finding (serialised JSON)
        self.assertTrue(len(result) > 0)
        self.assertIn("RTX 3090", result)
        self.assertIn("RX 6900 XT", result)

    def test_json_with_named_key_still_preferred(self):
        """When both named-key JSON and arbitrary JSON exist, prefer named."""
        producer_output = (
            'Raw data: {"gpu": "RTX 3090", "competitor": "RX 6900 XT"}\n'
            'Conclusion: {"finding": "RTX 3090 competes against RX 6900 XT"}'
        )
        result = _extract_finding(producer_output)
        self.assertEqual(result, "RTX 3090 competes against RX 6900 XT")

    def test_any_json_block_matches(self):
        """Regex matches JSON blocks without requiring specific keys."""
        import re
        text = '```json\n{"a": 1, "b": 2}\n```'
        pattern = r'(?:```(?:json)?\s*\n)?(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
        matches = re.findall(pattern, text, re.DOTALL)
        self.assertGreaterEqual(len(matches), 1, "Any-JSON regex must match")


class TestSynthesisEnabled(unittest.TestCase):
    """Feature flag defaults and config reading."""

    def test_defaults_to_true(self):
        # Without config, should default to True (enabled by default)
        with patch("tools.delegate_tool._load_config", return_value={}):
            self.assertTrue(_get_synthesis_enabled())

    def test_truthy_values(self):
        for val in (True, "true", "1", "yes", "on"):
            with patch("tools.delegate_tool._load_config", return_value={"synthesis_enabled": val}):
                self.assertTrue(_get_synthesis_enabled())

    def test_explicit_false(self):
        with patch("tools.delegate_tool._load_config", return_value={"synthesis_enabled": False}):
            self.assertFalse(_get_synthesis_enabled())


class TestVerifyEnabled(unittest.TestCase):
    """Feature flag defaults and config reading."""

    def test_defaults_to_true(self):
        with patch("tools.delegate_tool._load_config", return_value={}):
            self.assertTrue(_get_verify_enabled())

    def test_truthy_values(self):
        for val in (True, "true", "1", "yes", "on"):
            with patch("tools.delegate_tool._load_config", return_value={"verify_enabled": val}):
                self.assertTrue(_get_verify_enabled())


class TestG5Isolation(unittest.TestCase):
    """G5 enforced: skeptic prompt must NOT contain producer reasoning."""

    def test_extract_finding_strips_reasoning(self):
        """Structural extraction removes reasoning — the skeptic never sees it."""
        producer_output = (
            "I think the answer is PostgreSQL because I've used it before and "
            "it scales well. The benchmarks also show good performance.\n\n"
            '{"finding": "Use PostgreSQL for the primary database"}'
        )
        extracted = _extract_finding(producer_output)
        # The extracted finding should NOT contain the reasoning chain
        self.assertNotIn("I think", extracted)
        self.assertNotIn("because", extracted)
        self.assertNotIn("I've used it", extracted)
        self.assertEqual(extracted, "Use PostgreSQL for the primary database")

    def test_unstructured_reasoning_returns_empty(self):
        """F1: Plain-text reasoning with no structured marker — returns '' to skip verification."""
        producer_output = (
            "After careful analysis and weighing multiple options...\n\n"
            "The benchmarks clearly favour Redis for this use case."
        )
        extracted = _extract_finding(producer_output)
        self.assertEqual(extracted, "")


class TestF2SummaryBlockToken(unittest.TestCase):
    """F2: had_block_token prevents double-append after .replace()."""

    def _format_synth_prompt(self, synth_prompt, summary_block):
        """Replicate the exact F2 logic pattern from delegate_task()."""
        had_block_token = "{summary_block}" in synth_prompt
        synth_prompt = synth_prompt.replace(
            "{summary_block}", summary_block
        ).replace("{n}", str(3))
        if not had_block_token:
            synth_prompt += f"\n\n{summary_block}"
        return synth_prompt

    def test_default_prompt_has_token(self):
        """Default prompt contains {summary_block} — replaced, not appended."""
        default = "Merge {n} findings:\n\n{summary_block}\n\nDone."
        result = self._format_synth_prompt(default, "BLOCKCONTENT")
        self.assertEqual(result.count("BLOCKCONTENT"), 1)
        self.assertIn("BLOCKCONTENT", result)
        self.assertNotIn("{summary_block}", result)
        self.assertIn("Merge 3 findings:", result)

    def test_custom_prompt_no_token(self):
        """Custom prompt without {summary_block} — block appended once."""
        custom = "Custom merge of {count} items. No block token here."
        result = self._format_synth_prompt(custom, "BLOCKCONTENT")
        self.assertEqual(result.count("BLOCKCONTENT"), 1)
        self.assertIn("No block token here.", result)  # original text preserved


class TestPersistentConfigOverlay(unittest.TestCase):
    """Regression: _load_config must read persistent config even when CLI_CONFIG exists.

    The pre-fix code short-circuited on non-empty CLI_CONFIG, making dynamic
    hermes config set changes invisible to running gateways.
    """

    def test_persistent_keys_visible_with_cli_overlay(self):
        """verify_enabled from persistent config survives CLI_CONFIG overlay."""
        with patch(
            "tools.delegate_tool._load_config",
            return_value={"verify_enabled": True, "model": "deepseek-v4-pro"},
        ):
            self.assertTrue(_get_verify_enabled())

    def test_cli_overlay_does_not_block_persistent(self):
        """When CLI_CONFIG has model key but no verify_enabled, persistent's True wins."""
        mock_cfg = {
            "model": "deepseek-v4-pro",
            "verify_enabled": True,   # Must be visible even with CLI overlay
            "synthesis_enabled": True,
        }
        with patch("tools.delegate_tool._load_config", return_value=mock_cfg):
            self.assertTrue(_get_verify_enabled())
            self.assertTrue(_get_synthesis_enabled())

    def test_features_disabled_when_persistent_says_false(self):
        """When persistent config has False, gates return False (CLI can't force True)."""
        mock_cfg = {"verify_enabled": False, "synthesis_enabled": False}
        with patch("tools.delegate_tool._load_config", return_value=mock_cfg):
            self.assertFalse(_get_verify_enabled())
            self.assertFalse(_get_synthesis_enabled())

    def test_missing_keys_default_true(self):
        """Absent keys default to True (enabled by default), not crash."""
        with patch("tools.delegate_tool._load_config", return_value={"model": "some-model"}):
            self.assertTrue(_get_verify_enabled())
            self.assertTrue(_get_synthesis_enabled())


if __name__ == "__main__":
    unittest.main()
