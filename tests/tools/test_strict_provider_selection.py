"""Strict tool-provider selection: the `rabbit tools` choice always wins.

Policy: the provider string stored in config.yaml is what runs at call time.
A stored vendor selection with missing credentials produces an honest error
naming the selection and pointing at `rabbit tools`; nothing ever written ->
legacy autodetect. Credential presence must NEVER select or reroute.

The managed-gateway selection was removed from this build: a stored legacy
``nous`` string is handed back verbatim by read_selection (it is just a
stored vendor name now) and hits the same selection-naming error contract;
image_gen additionally folds it to the FAL fallback via ``_plugin_provider_name``.

Per category these tests pin the strict behaviors:
  (a) vendor selection + key missing => selection-naming error
  (b) vendor selection + key present => direct route
  (c) never-configured => legacy autodetect unchanged
"""

from unittest.mock import patch

import pytest

from tools import tool_backend_helpers as tbh


# ---------------------------------------------------------------------------
# read_selection — the shared helper
# ---------------------------------------------------------------------------


class TestReadSelection:
    def _with_raw(self, raw):
        return patch(
            "rabbit_cli.config.read_raw_config_readonly",
            return_value=raw,
        )

    def test_never_configured_returns_none(self):
        with self._with_raw({}):
            assert tbh.read_selection("image_gen") is None

    def test_vendor_provider_returned(self):
        with self._with_raw({"image_gen": {"provider": "fal"}}):
            assert tbh.read_selection("image_gen") == "fal"

    def test_stored_string_returned_verbatim(self):
        """A legacy managed-gateway string is just a stored vendor name now."""
        with self._with_raw({"image_gen": {"provider": "nous"}}):
            assert tbh.read_selection("image_gen") == "nous"

    def test_legacy_use_gateway_true_keeps_vendor(self):
        """Old configs stored use_gateway: true beside a vendor name; the
        gateway flag is inert now and the vendor string is the selection."""
        with self._with_raw({"video_gen": {"provider": "fal", "use_gateway": True}}):
            assert tbh.read_selection("video_gen") == "fal"

    def test_legacy_use_gateway_false_keeps_vendor(self):
        with self._with_raw({"tts": {"provider": "openai", "use_gateway": False}}):
            assert tbh.read_selection("tts") == "openai"

    def test_empty_string_backend_is_no_selection(self):
        """DEFAULT_CONFIG's seeded empty strings are not selections."""
        with self._with_raw({"web": {"backend": ""}}):
            assert tbh.read_selection("web") is None

    def test_raw_stt_local_is_a_selection(self):
        """A raw config.yaml ``stt.provider: local`` is a genuine pick: the
        DEFAULT_CONFIG seed never reached disk (save_config strips schema
        defaults), and the current picker's Local Whisper row writes exactly
        this shape (provider only, legacy use_gateway popped). Treating it
        as no-selection would silently discard the user's choice."""
        with self._with_raw({"stt": {"provider": "local"}}):
            assert tbh.read_selection("stt") == "local"

    def test_stt_local_with_use_gateway_key_is_a_selection(self):
        """A picker-written stt section (use_gateway key present) means
        local was a genuine choice."""
        with self._with_raw({"stt": {"provider": "local", "use_gateway": False}}):
            assert tbh.read_selection("stt") == "local"

    def test_browser_backend_key_is_not_the_cloud_selection(self):
        """browser.backend is the driver choice (browser-use CLI vs built-in
        tools), not the cloud provider selection."""
        with self._with_raw({"browser": {"backend": "browser-use"}}):
            assert tbh.read_selection("browser") is None

    def test_web_per_capability_keys_mark_configured(self):
        with self._with_raw({"web": {"search_backend": "searxng"}}):
            assert tbh.read_selection("web") is None
            assert tbh.selection_exists("web") is True


# ---------------------------------------------------------------------------
# Image generation (FAL)
# ---------------------------------------------------------------------------


class TestImageFalStrictSelection:
    def test_stored_selection_missing_key_raises_selection_error(self):
        from tools import image_generation_tool as it

        with patch.object(it, "read_selection", return_value="nous"), \
             patch.object(it, "fal_key_is_configured", return_value=False):
            with pytest.raises(ValueError) as exc:
                it._resolve_managed_fal_gateway()
        assert "image_gen is configured to use nous" in str(exc.value)
        assert "rabbit tools" in str(exc.value)

    def test_fal_selection_missing_key_raises_selection_error(self):
        from tools import image_generation_tool as it

        with patch.object(it, "read_selection", return_value="fal"), \
             patch.object(it, "fal_key_is_configured", return_value=False):
            with pytest.raises(ValueError) as exc:
                it._resolve_managed_fal_gateway()
        assert "FAL_KEY" in str(exc.value)
        assert "image_gen is configured to use fal" in str(exc.value)
        assert "rabbit tools" in str(exc.value)

    def test_selection_with_key_passes(self):
        from tools import image_generation_tool as it

        with patch.object(it, "read_selection", return_value="fal"), \
             patch.object(it, "fal_key_is_configured", return_value=True):
            assert it._resolve_managed_fal_gateway() is None

    def test_never_configured_passes_without_key(self):
        """No selection: no error either way — legacy autodetect owns it."""
        from tools import image_generation_tool as it

        with patch.object(it, "read_selection", return_value=None), \
             patch.object(it, "fal_key_is_configured", return_value=False):
            assert it._resolve_managed_fal_gateway() is None

    def test_check_fal_api_key_reflects_selection(self):
        from tools import image_generation_tool as it

        with patch.object(it, "read_selection", return_value="fal"), \
             patch.object(it, "fal_key_is_configured", return_value=False):
            # Broken vendor selection reports unavailable.
            assert it.check_fal_api_key() is False

    def test_legacy_nous_selection_folds_to_fal_fallback(self):
        """A legacy ``nous`` image_gen selection is treated as unset so the
        in-tree FAL pipeline applies (managed gateway removed)."""
        from tools import image_generation_tool as it

        with patch.object(it, "_read_configured_image_provider", return_value="nous"):
            assert it._plugin_provider_name() is None


# ---------------------------------------------------------------------------
# Video generation (FAL plugin)
# ---------------------------------------------------------------------------


class TestVideoFalStrictSelection:
    def test_stored_selection_missing_key_raises_selection_error(self):
        from plugins.video_gen import fal as vf

        with patch("tools.tool_backend_helpers.read_selection", return_value="nous"), \
             patch("tools.tool_backend_helpers.fal_key_is_configured", return_value=False):
            with pytest.raises(ValueError) as exc:
                vf._check_fal_video_selection()
        assert "video_gen is configured to use nous" in str(exc.value)
        assert "FAL_KEY" in str(exc.value)

    def test_fal_selection_missing_key_raises_selection_error(self):
        from plugins.video_gen import fal as vf

        with patch("tools.tool_backend_helpers.read_selection", return_value="fal"), \
             patch("tools.tool_backend_helpers.fal_key_is_configured", return_value=False):
            with pytest.raises(ValueError) as exc:
                vf._check_fal_video_selection()
        assert "video_gen is configured to use fal" in str(exc.value)

    def test_selection_with_key_passes(self):
        from plugins.video_gen import fal as vf

        with patch("tools.tool_backend_helpers.read_selection", return_value="fal"), \
             patch("tools.tool_backend_helpers.fal_key_is_configured", return_value=True):
            vf._check_fal_video_selection()  # no raise

    def test_never_configured_passes_without_key(self):
        from plugins.video_gen import fal as vf

        with patch("tools.tool_backend_helpers.read_selection", return_value=None), \
             patch("tools.tool_backend_helpers.fal_key_is_configured", return_value=False):
            vf._check_fal_video_selection()  # no raise


# ---------------------------------------------------------------------------
# STT (OpenAI audio resolver)
# ---------------------------------------------------------------------------


class TestSttStrictSelection:
    def test_direct_key_wins_over_stored_selection(self):
        """A configured direct OpenAI key resolves regardless of the stored
        selection — credentials already written beat the picker string."""
        from tools import transcription_cloud as tc

        with patch("tools.transcription_tools._load_stt_config", return_value={"openai": {"api_key": "sk-direct"}}), \
             patch("tools.tool_backend_helpers.read_selection", return_value="nous"):
            api_key, base_url = tc._resolve_openai_audio_client_config()
        assert api_key == "sk-direct"

    def test_vendor_selection_missing_key_raises_selection_error(self):
        from tools import transcription_cloud as tc

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.tool_backend_helpers.read_selection", return_value="openai"), \
             patch("tools.tool_backend_helpers.resolve_openai_audio_api_key", return_value=""):
            with pytest.raises(ValueError) as exc:
                tc._resolve_openai_audio_client_config()
        assert "stt is configured to use openai" in str(exc.value)
        assert "rabbit tools" in str(exc.value)

    def test_no_selection_falls_back_to_env_key(self):
        from tools import transcription_cloud as tc

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.tool_backend_helpers.read_selection", return_value=None), \
             patch("tools.tool_backend_helpers.resolve_openai_audio_api_key", return_value="sk-env"):
            api_key, base_url = tc._resolve_openai_audio_client_config()
        assert api_key == "sk-env"

    def test_no_selection_no_creds_raises_plain_error(self):
        from tools import transcription_cloud as tc

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.tool_backend_helpers.read_selection", return_value=None), \
             patch("tools.tool_backend_helpers.resolve_openai_audio_api_key", return_value=""):
            with pytest.raises(ValueError) as exc:
                tc._resolve_openai_audio_client_config()
        assert "OPENAI_API_KEY" in str(exc.value)


# ---------------------------------------------------------------------------
# Browser Use provider
# ---------------------------------------------------------------------------


class TestBrowserUseStrictSelection:
    def _provider(self):
        from plugins.browser.browser_use.provider import BrowserUseBrowserProvider

        return BrowserUseBrowserProvider()

    def test_direct_key_routes_direct(self):
        provider = self._provider()
        with patch("plugins.browser.browser_use.provider.get_secret", return_value="bu-key"):
            config = provider._get_config_or_none()
        assert config is not None
        assert config["api_key"] == "bu-key"

    def test_vendor_selection_missing_key_raises_selection_error(self):
        provider = self._provider()
        with patch("plugins.browser.browser_use.provider.get_secret", return_value=""), \
             patch("tools.tool_backend_helpers.read_selection", return_value="browser-use"):
            with pytest.raises(ValueError) as exc:
                provider._get_config()
        assert "browser is configured to use browser-use" in str(exc.value)
        assert "BROWSER_USE_API_KEY" in str(exc.value)

    def test_missing_key_no_selection_raises_plain_error(self):
        provider = self._provider()
        with patch("plugins.browser.browser_use.provider.get_secret", return_value=""), \
             patch("tools.tool_backend_helpers.read_selection", return_value=None):
            with pytest.raises(ValueError) as exc:
                provider._get_config()
        assert "BROWSER_USE_API_KEY" in str(exc.value)


# ---------------------------------------------------------------------------
# Camofox: selection over env var
# ---------------------------------------------------------------------------


class TestCamofoxSelection:
    def test_camofox_selection_activates_mode(self, monkeypatch):
        from tools import browser_camofox as bc

        monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
        with patch.object(bc, "_config_cdp_url", return_value=""), \
             patch("tools.tool_backend_helpers.read_selection", return_value="camofox"):
            assert bc.is_camofox_mode() is True

    def test_other_selection_beats_camofox_url_env(self, monkeypatch):
        """CAMOFOX_URL is the ADDRESS, not the choice: an explicit different
        browser selection wins."""
        from tools import browser_camofox as bc

        monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
        with patch.object(bc, "_config_cdp_url", return_value=""), \
             patch.object(bc, "get_camofox_url", return_value="http://localhost:9377"), \
             patch("tools.tool_backend_helpers.read_selection", return_value="local"):
            assert bc.is_camofox_mode() is False

    def test_never_configured_env_url_still_activates(self, monkeypatch):
        from tools import browser_camofox as bc

        monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
        with patch.object(bc, "_config_cdp_url", return_value=""), \
             patch.object(bc, "get_camofox_url", return_value="http://localhost:9377"), \
             patch("tools.tool_backend_helpers.read_selection", return_value=None):
            assert bc.is_camofox_mode() is True


# ---------------------------------------------------------------------------
# tools_config writers: one provider string per row, no use_gateway writes
# ---------------------------------------------------------------------------


class TestWriteProviderConfig:
    def test_byok_row_writes_vendor_and_clears_legacy_flag(self):
        from rabbit_cli.tools_config_providers import _write_provider_config

        config = {"web": {"backend": "nous", "use_gateway": True}}
        provider = {"name": "Keenable", "web_backend": "keenable"}
        _write_provider_config(provider, config)
        assert config["web"]["backend"] == "keenable"
        assert "use_gateway" not in config["web"]

    def test_tts_row_writes_provider_and_clears_legacy_flag(self):
        from rabbit_cli.tools_config_providers import _write_provider_config

        config = {"tts": {"provider": "nous", "use_gateway": True}}
        provider = {"name": "Edge TTS", "tts_provider": "edge"}
        _write_provider_config(provider, config)
        assert config["tts"]["provider"] == "edge"
        assert "use_gateway" not in config["tts"]

    def test_plugin_injected_byok_row_clears_stale_use_gateway(self):
        """Plugin-injected rows are not in TOOL_CATEGORIES' hardcoded
        provider lists; the legacy clear-loop resolves them via markers."""
        from rabbit_cli.tools_config_providers import _write_provider_config

        config = {"stt": {"provider": "nous", "use_gateway": True}}
        provider = {"name": "Groq Whisper", "stt_provider": "groq"}
        _write_provider_config(provider, config)
        assert config["stt"]["provider"] == "groq"
        assert "use_gateway" not in config["stt"]
