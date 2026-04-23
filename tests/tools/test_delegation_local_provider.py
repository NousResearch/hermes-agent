#!/usr/bin/env python3
"""
Tests for delegation with local/Ollama providers that don't require API keys.

Ollama and other local model servers run on localhost and accept requests
without authentication. The delegation credential resolver should allow these
endpoints to work without requiring an API key.

Run with:  python -m pytest tests/tools/test_delegation_local_provider.py -v
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    _resolve_delegation_credentials,
)


def _make_mock_parent(depth=0):
    """Create a mock parent agent with the fields delegate_task expects."""
    parent = MagicMock()
    parent.base_url = "http://localhost:11434/v1"
    parent.api_key = "ollama"
    parent.provider = "custom"
    parent.api_mode = "chat_completions"
    parent.model = "glm-5.1:cloud"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = __import__("threading").Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent._credential_pool = None
    parent.reasoning_config = None
    parent.max_tokens = None
    parent.prefill_messages = None
    parent.acp_command = None
    parent.acp_args = []
    parent.valid_tool_names = ["terminal", "file", "web"]
    parent.enabled_toolsets = None  # None = all tools
    return parent


class TestLocalProviderCredentials(unittest.TestCase):
    """Tests for _resolve_delegation_credentials with local providers."""

    # --- base_url path (localhost) ---

    def test_localhost_base_url_no_api_key_allowed(self):
        """localhost base_url should work without an API key (Ollama, LM Studio, etc.)."""
        parent = _make_mock_parent()
        cfg = {
            "model": "devstral-small-2:24b-cloud",
            "provider": "custom",
            "base_url": "http://localhost:11434/v1",
            "api_key": "",
        }
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertEqual(creds["base_url"], "http://localhost:11434/v1")
        self.assertIsNotNone(creds["api_key"])
        # API key should be a harmless placeholder, not None
        self.assertNotEqual(creds["api_key"], "")

    def test_127_base_url_no_api_key_allowed(self):
        """127.0.0.1 base_url should work without an API key."""
        parent = _make_mock_parent()
        cfg = {
            "model": "devstral-small-2:24b-cloud",
            "provider": "",
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "",
        }
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertEqual(creds["base_url"], "http://127.0.0.1:11434/v1")
        self.assertIsNotNone(creds["api_key"])

    def test_dotlocal_base_url_no_api_key_allowed(self):
        """.local mDNS hostnames (e.g. studio.local) should work without an API key."""
        parent = _make_mock_parent()
        cfg = {
            "model": "devstral-small-2:24b-cloud",
            "provider": "",
            "base_url": "http://studio.local:11434/v1",
            "api_key": "",
        }
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertEqual(creds["base_url"], "http://studio.local:11434/v1")
        self.assertIsNotNone(creds["api_key"])

    def test_localhost_base_url_with_explicit_api_key_preserved(self):
        """If user provides an API key for localhost, it should be preserved as-is."""
        parent = _make_mock_parent()
        cfg = {
            "model": "devstral-small-2:24b-cloud",
            "provider": "custom",
            "base_url": "http://localhost:11434/v1",
            "api_key": "my-secret-key",
        }
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertEqual(creds["api_key"], "my-secret-key")

    # --- base_url path (remote) should still require API key ---

    def test_remote_base_url_still_requires_api_key(self):
        """Non-localhost base_url without API key should still raise ValueError."""
        parent = _make_mock_parent()
        cfg = {
            "model": "gpt-4o-mini",
            "provider": "",
            "base_url": "https://api.openai.com/v1",
            "api_key": "",
        }
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            with self.assertRaises(ValueError) as ctx:
                _resolve_delegation_credentials(cfg, parent)
        self.assertIn("API key", str(ctx.exception))

    # --- provider path with custom/local ---

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_custom_provider_resolving_to_localhost_no_api_key(self, mock_resolve):
        """When delegation.provider='custom' resolves to localhost, empty API key should be allowed."""
        mock_resolve.return_value = {
            "provider": "custom",
            "base_url": "http://localhost:11434/v1",
            "api_key": "",
            "api_mode": "chat_completions",
        }
        parent = _make_mock_parent()
        cfg = {"model": "devstral-small-2:24b-cloud", "provider": "custom"}
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertEqual(creds["provider"], "custom")
        self.assertEqual(creds["base_url"], "http://localhost:11434/v1")
        # Should get a placeholder key, not raise ValueError
        self.assertIsNotNone(creds["api_key"])
        self.assertNotEqual(creds["api_key"], "")

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_remote_provider_still_requires_api_key(self, mock_resolve):
        """Provider resolving to a remote endpoint without API key should still raise."""
        mock_resolve.return_value = {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "",
            "api_mode": "chat_completions",
        }
        parent = _make_mock_parent()
        cfg = {"model": "some-model", "provider": "openrouter"}
        with self.assertRaises(ValueError) as ctx:
            _resolve_delegation_credentials(cfg, parent)
        self.assertIn("no API key", str(ctx.exception))

    # --- Integration: child agent gets local placeholder key ---

    @patch("tools.delegate_tool._load_config")
    def test_local_delegation_uses_placeholder_key(self, mock_cfg):
        """Delegation with localhost base_url should get 'ollama' placeholder API key."""
        mock_cfg.return_value = {
            "model": "devstral-small-2:24b-cloud",
            "provider": "custom",
            "base_url": "http://localhost:11434/v1",
            "api_key": "",
            "max_iterations": 10,
            "max_concurrent_children": 1,
        }
        parent = _make_mock_parent()
        creds = _resolve_delegation_credentials(mock_cfg.return_value, parent)
        self.assertEqual(creds["base_url"], "http://localhost:11434/v1")
        self.assertEqual(creds["api_key"], "ollama")


class TestIsLocalBaseUrlHelper(unittest.TestCase):
    """Tests for the _is_local_base_url helper function."""

    def test_localhost_with_port(self):
        from tools.delegate_tool import _is_local_base_url
        self.assertTrue(_is_local_base_url("http://localhost:11434/v1"))

    def test_localhost_no_port(self):
        from tools.delegate_tool import _is_local_base_url
        self.assertTrue(_is_local_base_url("http://localhost/v1"))

    def test_127_ip(self):
        from tools.delegate_tool import _is_local_base_url
        self.assertTrue(_is_local_base_url("http://127.0.0.1:11434/v1"))

    def test_dotlocal(self):
        from tools.delegate_tool import _is_local_base_url
        self.assertTrue(_is_local_base_url("http://studio.local:11434/v1"))

    def test_remote_url(self):
        from tools.delegate_tool import _is_local_base_url
        self.assertFalse(_is_local_base_url("https://api.openai.com/v1"))

    def test_openrouter(self):
        from tools.delegate_tool import _is_local_base_url
        self.assertFalse(_is_local_base_url("https://openrouter.ai/api/v1"))

    def test_192_168_private(self):
        from tools.delegate_tool import _is_local_base_url
        self.assertTrue(_is_local_base_url("http://192.168.1.100:11434/v1"))

    def test_10_private(self):
        from tools.delegate_tool import _is_local_base_url
        self.assertTrue(_is_local_base_url("http://10.0.0.5:11434/v1"))

    def test_172_private(self):
        from tools.delegate_tool import _is_local_base_url
        self.assertTrue(_is_local_base_url("http://172.16.0.1:11434/v1"))

    def test_empty_string(self):
        from tools.delegate_tool import _is_local_base_url
        self.assertFalse(_is_local_base_url(""))

    def test_none(self):
        from tools.delegate_tool import _is_local_base_url
        self.assertFalse(_is_local_base_url(None))


if __name__ == "__main__":
    unittest.main()