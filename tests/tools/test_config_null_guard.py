"""Tests for config.get() null-coalescing in tool configuration.

YAML ``null`` values (or ``~``) for a present key make ``dict.get(key, default)``
return ``None`` instead of the default — calling ``.lower()`` on that raises
``AttributeError``.  These tests verify the ``or`` coalescing guards.
"""

from unittest.mock import patch
import pytest


# ── TTS tool ──────────────────────────────────────────────────────────────

class TestTTSProviderNullGuard:
    """tools/tts_tool.py — _get_provider()"""

    def test_explicit_null_provider_returns_default(self):
        """YAML ``tts: {provider: null}`` should fall back to default."""
        from tools.tts_tool import _get_provider, DEFAULT_PROVIDER

        result = _get_provider({"provider": None})
        assert result == DEFAULT_PROVIDER.lower().strip()

    def test_missing_provider_returns_default(self):
        """No ``provider`` key at all should also return default."""
        from tools.tts_tool import _get_provider, DEFAULT_PROVIDER

        result = _get_provider({})
        assert result == DEFAULT_PROVIDER.lower().strip()

    def test_valid_provider_passed_through(self):
        from tools.tts_tool import _get_provider

        result = _get_provider({"provider": "OPENAI"})
        assert result == "openai"


# ── Web tools ─────────────────────────────────────────────────────────────

class TestWebBackendNullGuard:
    """tools/web_tools.py — _get_backend()"""

    @patch("tools.web_tools._load_web_config", return_value={"backend": None})
    def test_explicit_null_backend_does_not_crash(self, _cfg):
        """YAML ``web: {backend: null}`` should not raise AttributeError."""
        from tools.web_tools import _get_backend

        # Should not raise — the exact return depends on env key fallback
        result = _get_backend()
        assert isinstance(result, str)

    @patch("tools.web_tools._load_web_config", return_value={})
    def test_missing_backend_does_not_crash(self, _cfg):
        from tools.web_tools import _get_backend

        result = _get_backend()
        assert isinstance(result, str)


# ── MCP tool ──────────────────────────────────────────────────────────────

class TestMCPAuthNullGuard:
    """tools/mcp_tool.py — MCPServerTask.__init__() auth config line"""

    def test_explicit_null_auth_does_not_crash(self):
        """YAML ``auth: null`` in MCP server config should not raise."""
        # Test the expression directly — MCPServerTask.__init__ has many deps
        config = {"auth": None, "timeout": 30}
        auth_type = (config.get("auth") or "").lower().strip()
        assert auth_type == ""

    def test_missing_auth_defaults_to_empty(self):
        config = {"timeout": 30}
        auth_type = (config.get("auth") or "").lower().strip()
        assert auth_type == ""

    def test_valid_auth_passed_through(self):
        config = {"auth": "OAUTH", "timeout": 30}
        auth_type = (config.get("auth") or "").lower().strip()
        assert auth_type == "oauth"


# ── Trajectory compressor ─────────────────────────────────────────────────

class TestTrajectoryCompressorNullGuard:
    """trajectory_compressor.py — _detect_provider() and config loading"""

    def test_null_base_url_does_not_crash(self):
        """base_url=None should not crash _detect_provider()."""
        from trajectory_compressor import CompressionConfig, TrajectoryCompressor

        config = CompressionConfig()
        config.base_url = None

        compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
        compressor.config = config

        # Should not raise AttributeError; returns empty string (no match)
        result = compressor._detect_provider()
        assert result == ""

    def test_config_loading_null_base_url_keeps_default(self):
        """YAML ``summarization: {base_url: null}`` should keep default."""
        from trajectory_compressor import CompressionConfig
        from hermes_constants import OPENROUTER_BASE_URL

        config = CompressionConfig()
        data = {"summarization": {"base_url": None}}

        config.base_url = data["summarization"].get("base_url") or config.base_url
        assert config.base_url == OPENROUTER_BASE_URL
