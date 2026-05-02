"""Unit tests for tools/tool_backend_helpers.py.

Tests cover:
- managed_nous_tools_enabled() subscription-based gate
- normalize_browser_cloud_provider() coercion
- coerce_modal_mode() / normalize_modal_mode() validation
- has_direct_modal_credentials() detection
- resolve_modal_backend_state() backend selection matrix
- resolve_openai_audio_api_key() priority chain
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tools.tool_backend_helpers import (
    coerce_modal_mode,
    has_direct_modal_credentials,
    managed_nous_tools_enabled,
    normalize_browser_cloud_provider,
    normalize_modal_mode,
    prefers_gateway,
    resolve_modal_backend_state,
    resolve_openai_audio_api_key,
)


def _raise_import():
    raise ImportError("simulated missing module")


# ---------------------------------------------------------------------------
# managed_nous_tools_enabled
# ---------------------------------------------------------------------------
class TestManagedNousToolsEnabled:
    """Subscription-based gate: True for paid Nous subscribers."""

    def test_disabled_when_not_logged_in(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.auth.get_nous_auth_status",
            lambda: {},
        )
        assert managed_nous_tools_enabled() is False

    def test_disabled_for_free_tier(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.auth.get_nous_auth_status",
            lambda: {"logged_in": True},
        )
        monkeypatch.setattr(
            "hermes_cli.models.check_nous_free_tier",
            lambda: True,
        )
        assert managed_nous_tools_enabled() is False

    def test_enabled_for_paid_subscriber(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.auth.get_nous_auth_status",
            lambda: {"logged_in": True},
        )
        monkeypatch.setattr(
            "hermes_cli.models.check_nous_free_tier",
            lambda: False,
        )
        assert managed_nous_tools_enabled() is True

    def test_returns_false_on_exception(self, monkeypatch):
        """Should never crash — returns False on any exception."""
        monkeypatch.setattr(
            "hermes_cli.auth.get_nous_auth_status",
            _raise_import,
        )
        assert managed_nous_tools_enabled() is False


# ---------------------------------------------------------------------------
# normalize_browser_cloud_provider
# ---------------------------------------------------------------------------
class TestNormalizeBrowserCloudProvider:
    """Coerce arbitrary input to a lowercase browser provider key."""

    def test_none_returns_default(self):
        assert normalize_browser_cloud_provider(None) == "local"

    def test_empty_string_returns_default(self):
        assert normalize_browser_cloud_provider("") == "local"

    def test_whitespace_only_returns_default(self):
        assert normalize_browser_cloud_provider("   ") == "local"

    def test_known_provider_normalized(self):
        assert normalize_browser_cloud_provider("BrowserBase") == "browserbase"

    def test_strips_whitespace(self):
        assert normalize_browser_cloud_provider("  Local  ") == "local"

    def test_integer_coerced(self):
        result = normalize_browser_cloud_provider(42)
        assert isinstance(result, str)
        assert result == "42"


# ---------------------------------------------------------------------------
# coerce_modal_mode / normalize_modal_mode
# ---------------------------------------------------------------------------
class TestCoerceModalMode:
    """Validate and coerce the requested modal execution mode."""

    @pytest.mark.parametrize("value", ["auto", "direct", "managed"])
    def test_valid_modes_passthrough(self, value):
        assert coerce_modal_mode(value) == value

    def test_none_returns_auto(self):
        assert coerce_modal_mode(None) == "auto"

    def test_empty_string_returns_auto(self):
        assert coerce_modal_mode("") == "auto"

    def test_whitespace_only_returns_auto(self):
        assert coerce_modal_mode("   ") == "auto"

    def test_uppercase_normalized(self):
        assert coerce_modal_mode("DIRECT") == "direct"

    def test_mixed_case_normalized(self):
        assert coerce_modal_mode("Managed") == "managed"

    def test_invalid_mode_falls_back_to_auto(self):
        assert coerce_modal_mode("invalid") == "auto"
        assert coerce_modal_mode("cloud") == "auto"

    def test_strips_whitespace(self):
        assert coerce_modal_mode("  managed  ") == "managed"


class TestNormalizeModalMode:
    """normalize_modal_mode is an alias for coerce_modal_mode."""

    def test_delegates_to_coerce(self):
        assert normalize_modal_mode("direct") == coerce_modal_mode("direct")
        assert normalize_modal_mode(None) == coerce_modal_mode(None)
        assert normalize_modal_mode("bogus") == coerce_modal_mode("bogus")


# ---------------------------------------------------------------------------
# has_direct_modal_credentials
# ---------------------------------------------------------------------------
class TestHasDirectModalCredentials:
    """Detect Modal credentials via env vars or config file."""

    def test_no_env_no_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
        monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
        with patch.object(Path, "home", return_value=tmp_path):
            assert has_direct_modal_credentials() is False

    def test_both_env_vars_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MODAL_TOKEN_ID", "id-123")
        monkeypatch.setenv("MODAL_TOKEN_SECRET", "sec-456")
        with patch.object(Path, "home", return_value=tmp_path):
            assert has_direct_modal_credentials() is True

    def test_only_token_id_not_enough(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MODAL_TOKEN_ID", "id-123")
        monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
        with patch.object(Path, "home", return_value=tmp_path):
            assert has_direct_modal_credentials() is False

    def test_only_token_secret_not_enough(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
        monkeypatch.setenv("MODAL_TOKEN_SECRET", "sec-456")
        with patch.object(Path, "home", return_value=tmp_path):
            assert has_direct_modal_credentials() is False

    def test_config_file_present(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
        monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
        (tmp_path / ".modal.toml").touch()
        with patch.object(Path, "home", return_value=tmp_path):
            assert has_direct_modal_credentials() is True

    def test_env_vars_take_priority_over_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MODAL_TOKEN_ID", "id-123")
        monkeypatch.setenv("MODAL_TOKEN_SECRET", "sec-456")
        (tmp_path / ".modal.toml").touch()
        with patch.object(Path, "home", return_value=tmp_path):
            assert has_direct_modal_credentials() is True


# ---------------------------------------------------------------------------
# prefers_gateway
# ---------------------------------------------------------------------------
class TestPrefersGateway:
    """Honor bool-ish config values for tool gateway routing."""

    def test_returns_false_for_quoted_false(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"web": {"use_gateway": "false"}},
        )
        assert prefers_gateway("web") is False

    def test_returns_true_for_quoted_true(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"web": {"use_gateway": "true"}},
        )
        assert prefers_gateway("web") is True


# ---------------------------------------------------------------------------
# resolve_modal_backend_state
# ---------------------------------------------------------------------------
class TestResolveModalBackendState:
    """Full matrix of direct vs managed Modal backend selection."""

    @staticmethod
    def _resolve(monkeypatch, mode, *, has_direct, managed_ready, nous_enabled=False):
        """Helper to call resolve_modal_backend_state with feature flag control."""
        monkeypatch.setattr(
            "tools.tool_backend_helpers.managed_nous_tools_enabled",
            lambda: nous_enabled,
        )
        return resolve_modal_backend_state(
            mode, has_direct=has_direct, managed_ready=managed_ready
        )

    # --- auto mode ---

    def test_auto_prefers_managed_when_available(self, monkeypatch):
        result = self._resolve(monkeypatch, "auto", has_direct=True, managed_ready=True, nous_enabled=True)
        assert result["selected_backend"] == "managed"

    def test_auto_falls_back_to_direct(self, monkeypatch):
        result = self._resolve(monkeypatch, "auto", has_direct=True, managed_ready=False, nous_enabled=True)
        assert result["selected_backend"] == "direct"

    def test_auto_no_backends_available(self, monkeypatch):
        result = self._resolve(monkeypatch, "auto", has_direct=False, managed_ready=False)
        assert result["selected_backend"] is None

    def test_auto_managed_ready_but_nous_disabled(self, monkeypatch):
        result = self._resolve(monkeypatch, "auto", has_direct=True, managed_ready=True, nous_enabled=False)
        assert result["selected_backend"] == "direct"

    def test_auto_nothing_when_only_managed_and_nous_disabled(self, monkeypatch):
        result = self._resolve(monkeypatch, "auto", has_direct=False, managed_ready=True, nous_enabled=False)
        assert result["selected_backend"] is None

    # --- direct mode ---

    def test_direct_selects_direct_when_available(self, monkeypatch):
        result = self._resolve(monkeypatch, "direct", has_direct=True, managed_ready=True, nous_enabled=True)
        assert result["selected_backend"] == "direct"

    def test_direct_none_when_no_credentials(self, monkeypatch):
        result = self._resolve(monkeypatch, "direct", has_direct=False, managed_ready=True, nous_enabled=True)
        assert result["selected_backend"] is None

    # --- managed mode ---

    def test_managed_selects_managed_when_ready_and_enabled(self, monkeypatch):
        result = self._resolve(monkeypatch, "managed", has_direct=True, managed_ready=True, nous_enabled=True)
        assert result["selected_backend"] == "managed"

    def test_managed_none_when_not_ready(self, monkeypatch):
        result = self._resolve(monkeypatch, "managed", has_direct=True, managed_ready=False, nous_enabled=True)
        assert result["selected_backend"] is None

    def test_managed_blocked_when_nous_disabled(self, monkeypatch):
        result = self._resolve(monkeypatch, "managed", has_direct=True, managed_ready=True, nous_enabled=False)
        assert result["selected_backend"] is None
        assert result["managed_mode_blocked"] is True

    # --- return structure ---

    def test_return_dict_keys(self, monkeypatch):
        result = self._resolve(monkeypatch, "auto", has_direct=True, managed_ready=False)
        expected_keys = {
            "requested_mode",
            "mode",
            "has_direct",
            "managed_ready",
            "managed_mode_blocked",
            "selected_backend",
        }
        assert set(result.keys()) == expected_keys

    def test_passthrough_flags(self, monkeypatch):
        result = self._resolve(monkeypatch, "direct", has_direct=True, managed_ready=False)
        assert result["requested_mode"] == "direct"
        assert result["mode"] == "direct"
        assert result["has_direct"] is True
        assert result["managed_ready"] is False

    # --- invalid mode falls back to auto ---

    def test_invalid_mode_treated_as_auto(self, monkeypatch):
        result = self._resolve(monkeypatch, "bogus", has_direct=True, managed_ready=False)
        assert result["requested_mode"] == "auto"
        assert result["mode"] == "auto"


# ---------------------------------------------------------------------------
# resolve_openai_audio_api_key
# ---------------------------------------------------------------------------
class TestResolveOpenaiAudioApiKey:
    """Priority: VOICE_TOOLS_OPENAI_KEY > OPENAI_API_KEY."""

    def test_voice_key_preferred(self, monkeypatch):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "voice-key")
        monkeypatch.setenv("OPENAI_API_KEY", "general-key")
        assert resolve_openai_audio_api_key() == "voice-key"

    def test_falls_back_to_openai_key(self, monkeypatch):
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "general-key")
        assert resolve_openai_audio_api_key() == "general-key"

    def test_empty_voice_key_falls_back(self, monkeypatch):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "")
        monkeypatch.setenv("OPENAI_API_KEY", "general-key")
        assert resolve_openai_audio_api_key() == "general-key"

    def test_no_keys_returns_empty(self, monkeypatch):
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert resolve_openai_audio_api_key() == ""

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "  voice-key  ")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert resolve_openai_audio_api_key() == "voice-key"
