#!/usr/bin/env python3
"""
Regression tests for delegation config routing mismatch (GH #14992).

When the in-memory CLI_CONFIG has a stale/empty delegation dict,
_load_config() must fall back to load_config() so that profile-scoped
delegation.base_url is honoured instead of silently inheriting the
parent agent's endpoint.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure hermes-agent source is on path
sys.path.insert(0, "/home/zhans/.hermes/hermes-agent")

from tools.delegate_tool import _load_config


class TestLoadConfigRoutingFallback(unittest.TestCase):
    """_load_config() must not let an empty CLI_CONFIG delegation dict
    shadow a non-empty disk config."""

    def _make_cli_module(self, delegation_cfg):
        """Build a fake 'cli' module with CLI_CONFIG."""
        mod = MagicMock()
        mod.CLI_CONFIG = {"delegation": delegation_cfg}
        return mod

    @patch("hermes_cli.config.load_config")
    def test_empty_cli_config_falls_back_to_disk(self, mock_load_config):
        """CLI_CONFIG delegation is empty → read from disk."""
        fake_cli = self._make_cli_module(
            {"model": "", "provider": "", "base_url": "", "api_key": ""}
        )
        with patch.dict(sys.modules, {"cli": fake_cli}):
            mock_load_config.return_value = {
                "delegation": {
                    "model": "kimi-k2.6",
                    "provider": "opencode-go",
                    "base_url": "https://opencode.ai/zen/go/v1",
                    "api_key": "sk-test",
                }
            }

            cfg = _load_config()

        self.assertEqual(cfg["base_url"], "https://opencode.ai/zen/go/v1")
        self.assertEqual(cfg["model"], "kimi-k2.6")
        mock_load_config.assert_called_once()

    def test_cli_config_with_base_url_is_trusted(self):
        """CLI_CONFIG delegation has a real base_url → use it, skip disk."""
        fake_cli = self._make_cli_module(
            {
                "model": "gpt-4o",
                "provider": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-openai",
            }
        )
        with patch.dict(sys.modules, {"cli": fake_cli}):
            cfg = _load_config()

        self.assertEqual(cfg["base_url"], "https://api.openai.com/v1")

    def test_cli_config_with_provider_only_is_trusted(self):
        """CLI_CONFIG has provider (no base_url) → still trusted."""
        fake_cli = self._make_cli_module(
            {
                "model": "",
                "provider": "openrouter",
                "base_url": "",
                "api_key": "",
            }
        )
        with patch.dict(sys.modules, {"cli": fake_cli}):
            cfg = _load_config()

        self.assertEqual(cfg["provider"], "openrouter")

    def test_cli_config_with_model_only_is_trusted(self):
        """CLI_CONFIG has model only → trusted for model override."""
        fake_cli = self._make_cli_module(
            {
                "model": "claude-sonnet-4",
                "provider": "",
                "base_url": "",
                "api_key": "",
            }
        )
        with patch.dict(sys.modules, {"cli": fake_cli}):
            cfg = _load_config()

        self.assertEqual(cfg["model"], "claude-sonnet-4")

    @patch("hermes_cli.config.load_config")
    def test_no_cli_config_delegation_key_falls_back(self, mock_load_config):
        """CLI_CONFIG lacks 'delegation' key entirely → read from disk."""
        fake_cli = MagicMock()
        fake_cli.CLI_CONFIG = {}
        with patch.dict(sys.modules, {"cli": fake_cli}):
            mock_load_config.return_value = {
                "delegation": {
                    "model": "qwen2.5-coder",
                    "base_url": "http://localhost:1234/v1",
                }
            }

            cfg = _load_config()

        self.assertEqual(cfg["base_url"], "http://localhost:1234/v1")
        mock_load_config.assert_called_once()


class TestBuildChildAgentRouting(unittest.TestCase):
    """When delegation override_base_url is resolved, the child must use it
    instead of inheriting the parent's base_url."""

    def test_child_uses_override_base_url_not_parent(self):
        """Regression: empty delegation cfg caused child to inherit parent's
        custom endpoint and then fail with 'No available channel for model
        X under group Codex' when that endpoint does not support the model."""
        parent = MagicMock()
        parent.base_url = "https://api.example-provider.cc/v1"
        parent.api_key = "parent-key"
        parent.provider = "custom"
        parent.api_mode = "codex_responses"
        parent.model = "gpt-5.5"
        parent.platform = "cli"
        parent.providers_allowed = None
        parent.providers_ignored = None
        parent.providers_order = None
        parent.provider_sort = None
        parent._session_db = None
        parent._delegate_depth = 0
        parent._active_children = []
        parent._active_children_lock = MagicMock()
        parent._print_fn = None
        parent.tool_progress_callback = None
        parent.thinking_callback = None
        parent.enabled_toolsets = None
        parent.valid_tool_names = []

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            MockAgent.return_value = mock_child

            from tools.delegate_tool import _build_child_agent

            _build_child_agent(
                task_index=0,
                goal="Test routing",
                context=None,
                toolsets=None,
                model="kimi-k2.6",
                max_iterations=10,
                task_count=1,
                parent_agent=parent,
                override_provider="custom",
                override_base_url="https://opencode.ai/zen/go/v1",
                override_api_key="delegation-key",
                override_api_mode="chat_completions",
                role="leaf",
            )

            _, kwargs = MockAgent.call_args
            self.assertEqual(kwargs["base_url"], "https://opencode.ai/zen/go/v1")
            self.assertEqual(kwargs["model"], "kimi-k2.6")
            self.assertEqual(kwargs["provider"], "custom")
            self.assertEqual(kwargs["api_key"], "delegation-key")
            self.assertEqual(kwargs["api_mode"], "chat_completions")
            # Must NOT inherit parent's endpoint
            self.assertNotEqual(kwargs["base_url"], parent.base_url)

    def test_child_falls_back_to_parent_when_override_is_none(self):
        """When override values are None, child inherits parent — this is the
        *intended* fallback, not a bug."""
        parent = MagicMock()
        parent.base_url = "https://api.example-provider.cc/v1"
        parent.api_key = "parent-key"
        parent.provider = "custom"
        parent.api_mode = "codex_responses"
        parent.model = "gpt-5.5"
        parent.platform = "cli"
        parent.providers_allowed = None
        parent.providers_ignored = None
        parent.providers_order = None
        parent.provider_sort = None
        parent._session_db = None
        parent._delegate_depth = 0
        parent._active_children = []
        parent._active_children_lock = MagicMock()
        parent._print_fn = None
        parent.tool_progress_callback = None
        parent.thinking_callback = None
        parent.enabled_toolsets = None
        parent.valid_tool_names = []

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            MockAgent.return_value = mock_child

            from tools.delegate_tool import _build_child_agent

            _build_child_agent(
                task_index=0,
                goal="Test fallback",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=10,
                task_count=1,
                parent_agent=parent,
                override_provider=None,
                override_base_url=None,
                override_api_key=None,
                override_api_mode=None,
                role="leaf",
            )

            _, kwargs = MockAgent.call_args
            self.assertEqual(kwargs["base_url"], parent.base_url)
            self.assertEqual(kwargs["model"], parent.model)
            self.assertEqual(kwargs["provider"], parent.provider)


if __name__ == "__main__":
    unittest.main()
