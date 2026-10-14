"""Tests for config.get() null-coalescing in tool configuration.

YAML ``null`` values (or ``~``) for a present key make ``dict.get(key, default)``
return ``None`` instead of the default — calling ``.lower()`` on that raises
``AttributeError``.  These tests verify the ``or`` coalescing guards.
"""

from unittest.mock import patch

# ── TTS tool ──────────────────────────────────────────────────────────────

class TestTTSProviderNullGuard:
    """tools/tts_tool.py — _get_provider()"""

    def test_missing_provider_keeps_free_default_with_cloud_credentials(self):
        """A chat-provider key must not silently opt the user into paid TTS."""
        from tools.tts_tool import _get_provider, DEFAULT_PROVIDER

        assert _get_provider({}) == DEFAULT_PROVIDER
        assert _get_provider({"provider": None}) == DEFAULT_PROVIDER

    def test_explicit_provider_wins_over_active(self):
        """An explicit tts.provider always overrides the active-provider fallback."""
        from tools.tts_tool import _get_provider

        assert _get_provider({"provider": "edge"}) == "edge"

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

# ── MCP tool ──────────────────────────────────────────────────────────────

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
