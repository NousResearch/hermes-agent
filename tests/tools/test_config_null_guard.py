"""Tests for config.get() null-coalescing in tool configuration.

YAML ``null`` values (or ``~``) for a present key make ``dict.get(key, default)``
return ``None`` instead of the default — calling ``.lower()`` on that raises
``AttributeError``.  These tests verify the ``or`` coalescing guards.
"""

from unittest.mock import patch


# ── TTS tool ──────────────────────────────────────────────────────────────

class TestTTSProviderNullGuard:
    """tools/tts_tool.py — _get_provider()"""

    def test_explicit_null_provider_returns_default(self):
        """YAML ``tts: {provider: null}`` should fall back to default."""
        from tools.tts_tool import _get_provider, DEFAULT_PROVIDER

        # Pin the active inference provider to a non-TTS one so the
        # active-provider fallback doesn't fire — isolates the null guard.
        with patch("tools.tts_tool._active_model_provider", return_value="anthropic"):
            result = _get_provider({"provider": None})
        assert result == DEFAULT_PROVIDER.lower().strip()

    def test_missing_provider_returns_default(self):
        """No ``provider`` key + non-TTS active provider should return default."""
        from tools.tts_tool import _get_provider, DEFAULT_PROVIDER

        with patch("tools.tts_tool._active_model_provider", return_value="anthropic"):
            result = _get_provider({})
        assert result == DEFAULT_PROVIDER.lower().strip()

    def test_valid_provider_passed_through(self):
        from tools.tts_tool import _get_provider

        result = _get_provider({"provider": "OPENAI"})
        assert result == "openai"

    def test_falls_back_to_active_tts_capable_provider_when_available(self):
        """No explicit tts.provider + a TTS-capable, credentialled active
        provider → use it. DeepInfra/OpenAI are in BUILTIN_TTS_PROVIDERS, so a
        single-provider deployment gets matching TTS without configuring
        tts.provider — but only when the backend can authenticate."""
        from tools.tts_tool import _get_provider

        with patch("tools.tts_tool._active_model_provider", return_value="deepinfra"), \
                patch("tools.tts_tool._tts_provider_available", return_value=True):
            assert _get_provider({}) == "deepinfra"
        with patch("tools.tts_tool._active_model_provider", return_value="openai"), \
                patch("tools.tts_tool._tts_provider_available", return_value=True):
            assert _get_provider({"provider": None}) == "openai"

    def test_active_provider_without_credentials_keeps_edge(self):
        """A TTS-capable active provider that can't authenticate must NOT
        silently displace the free Edge default (no surprise billing / hard
        errors for a credential-less deployment)."""
        from tools.tts_tool import _get_provider, DEFAULT_PROVIDER

        with patch("tools.tts_tool._active_model_provider", return_value="openai"), \
                patch("tools.tts_tool._tts_provider_available", return_value=False):
            assert _get_provider({}) == DEFAULT_PROVIDER.lower().strip()

    def test_explicit_provider_wins_over_active(self):
        """An explicit tts.provider always overrides the active-provider fallback."""
        from tools.tts_tool import _get_provider

        with patch("tools.tts_tool._active_model_provider", return_value="deepinfra"):
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
