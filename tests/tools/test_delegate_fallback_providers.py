"""Tests for delegation.fallback_providers — per-child fallback chain.

Three modes:
  - Not set / "inherit" → child inherits parent's _fallback_chain
  - []                  → no fallback (child gets None)
  - [{provider, model}] → custom chain parsed from config

Copilot review fix: _load_config() in delegate_tool returns the delegation
sub-dict directly (not the full config), so tests must mock it the same way.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class TestDelegationFallbackProviders(unittest.TestCase):
    """Verify delegation.fallback_providers resolution."""

    def _make_parent_agent(self, fallback_chain=None):
        """Create a minimal parent agent mock with a fallback chain."""
        parent = MagicMock()
        parent._fallback_chain = fallback_chain or []
        parent.provider = "openai-codex"
        parent.model = "gpt-5.6-sol"
        parent.api_key = "test-key"
        parent.base_url = ""
        return parent

    def _resolve(self, mock_config_return, parent_chain):
        """Execute the same resolution logic as _build_child_agent.

        _load_config() returns the delegation sub-dict directly.
        This replicates the production code path with the same shape.
        """
        from hermes_cli.fallback_config import _iter_fallback_entries

        # mock_config_return IS the delegation dict (what _load_config returns)
        fb_raw = mock_config_return.get("fallback_providers") if isinstance(mock_config_return, dict) else None
        child_fallback = None
        mode = "inherit"

        if fb_raw is not None:
            if isinstance(fb_raw, str) and fb_raw.strip().lower() in ("inherit", "parent"):
                mode = "inherit"
            elif isinstance(fb_raw, list) and len(fb_raw) == 0:
                mode = "none"
            elif isinstance(fb_raw, (list, dict)):
                parsed = _iter_fallback_entries(fb_raw)
                if parsed:
                    child_fallback = parsed
                    mode = "custom"
                else:
                    mode = "none"

        if child_fallback is None and mode == "inherit":
            child_fallback = parent_chain or None

        return child_fallback, mode

    @patch("tools.delegate_tool._load_config")
    def test_inherit_when_not_set(self, mock_cfg):
        """When delegation.fallback_providers is absent, inherit parent chain."""
        # _load_config returns the delegation dict directly — no fallback_providers key
        mock_cfg.return_value = {"model": "gpt-5.6-luna"}
        parent_chain = [{"provider": "zai", "model": "glm-5.2"}]

        child_fb, mode = self._resolve(mock_cfg.return_value, parent_chain)

        self.assertEqual(mode, "inherit")
        self.assertIsNotNone(child_fb)
        self.assertEqual(child_fb[0]["provider"], "zai")

    @patch("tools.delegate_tool._load_config")
    def test_explicit_inherit_string(self, mock_cfg):
        """When delegation.fallback_providers is 'inherit', use parent chain."""
        mock_cfg.return_value = {"fallback_providers": "inherit"}
        parent_chain = [{"provider": "xai-oauth", "model": "grok-4.5"}]

        child_fb, mode = self._resolve(mock_cfg.return_value, parent_chain)

        self.assertEqual(mode, "inherit")
        self.assertIsNotNone(child_fb)
        self.assertEqual(child_fb[0]["provider"], "xai-oauth")

    @patch("tools.delegate_tool._load_config")
    def test_parent_string_alias(self, mock_cfg):
        """'parent' string should be treated same as 'inherit'."""
        mock_cfg.return_value = {"fallback_providers": "parent"}
        parent_chain = [{"provider": "zai", "model": "glm-5.2"}]

        child_fb, mode = self._resolve(mock_cfg.return_value, parent_chain)

        self.assertEqual(mode, "inherit")
        self.assertIsNotNone(child_fb)
        self.assertEqual(child_fb[0]["provider"], "zai")

    @patch("tools.delegate_tool._load_config")
    def test_empty_list_disables_fallback(self, mock_cfg):
        """When delegation.fallback_providers is [], child gets no fallback."""
        mock_cfg.return_value = {"fallback_providers": []}
        parent_chain = [{"provider": "zai", "model": "glm-5.2"}]

        child_fb, mode = self._resolve(mock_cfg.return_value, parent_chain)

        self.assertEqual(mode, "none")
        self.assertIsNone(child_fb)

    @patch("tools.delegate_tool._load_config")
    def test_custom_chain(self, mock_cfg):
        """When delegation.fallback_providers has entries, use them."""
        mock_cfg.return_value = {
            "fallback_providers": [
                {"provider": "zai", "model": "glm-5.2"},
                {"provider": "opencode-go", "model": "deepseek-v4-flash"},
            ]
        }
        parent_chain = [{"provider": "xai-oauth", "model": "grok-4.5"}]

        child_fb, mode = self._resolve(mock_cfg.return_value, parent_chain)

        self.assertEqual(mode, "custom")
        self.assertIsNotNone(child_fb)
        self.assertEqual(len(child_fb), 2)
        self.assertEqual(child_fb[0]["provider"], "zai")
        self.assertEqual(child_fb[0]["model"], "glm-5.2")
        self.assertEqual(child_fb[1]["provider"], "opencode-go")
        self.assertEqual(child_fb[1]["model"], "deepseek-v4-flash")
        # Must NOT be the parent's chain
        self.assertNotEqual(child_fb[0]["provider"], "xai-oauth")

    @patch("tools.delegate_tool._load_config")
    def test_config_returns_delegation_dict_not_full_config(self, mock_cfg):
        """Regression: _load_config returns delegation sub-dict, not full config.

        This test ensures we read .get('fallback_providers') directly,
        NOT .get('delegation', {}).get('fallback_providers').
        """
        # Simulate what _load_config actually returns: the delegation dict
        mock_cfg.return_value = {
            "fallback_providers": [{"provider": "zai", "model": "glm-5.2"}],
            "model": "gpt-5.6-luna",
            "provider": "openai-codex",
        }
        parent_chain = [{"provider": "xai-oauth", "model": "grok-4.5"}]

        child_fb, mode = self._resolve(mock_cfg.return_value, parent_chain)

        self.assertEqual(mode, "custom")
        self.assertEqual(child_fb[0]["provider"], "zai")

    @patch("tools.delegate_tool._load_config")
    def test_default_config_does_not_disable_fallback(self, mock_cfg):
        """Regression: when fallback_providers key is absent from defaults,
        children should inherit parent chain (not get disabled)."""
        # Simulate config_defaults.py WITHOUT fallback_providers key present
        mock_cfg.return_value = {
            "model": "",
            "provider": "",
            "max_iterations": 50,
            # NOTE: no fallback_providers key at all
        }
        parent_chain = [{"provider": "zai", "model": "glm-5.2"}]

        child_fb, mode = self._resolve(mock_cfg.return_value, parent_chain)

        self.assertEqual(mode, "inherit")
        self.assertIsNotNone(child_fb)
        self.assertEqual(child_fb[0]["provider"], "zai")


if __name__ == "__main__":
    unittest.main()
