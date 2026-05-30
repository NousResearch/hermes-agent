"""Regression tests — FAL/TTS is_available() must not trigger OAuth refresh.

check_fal_api_key(), _has_openai_audio_backend() (tts_tool), and
_check_fal_video_available() (video_gen/fal) are called by the tool/model
picker on every render. Before this fix they called
resolve_managed_tool_gateway() which defaults to read_nous_access_token
(a synchronous OAuth refresh POST), burning tokens on every UI paint.

Fix: switch these three read-only availability checks to
is_managed_tool_gateway_ready(), which defaults to peek_nous_access_token
and never triggers a refresh. Mirrors the fix applied to BrowserUse and
Firecrawl in PR #35401.
"""

from __future__ import annotations

from unittest.mock import patch

import tools.managed_tool_gateway as mtg


# ---------------------------------------------------------------------------
# check_fal_api_key — image_generation_tool.py
# ---------------------------------------------------------------------------

class TestCheckFalApiKeySkipsOAuthRefresh:
    """check_fal_api_key() must use is_managed_tool_gateway_ready, not
    resolve_managed_tool_gateway, so no OAuth refresh is triggered."""

    def test_no_refresh_when_gateway_available(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "valid-nous-token")

        import tools.image_generation_tool as igt

        refresh_calls: list = []

        def _record_refresh(**_kw):
            refresh_calls.append(1)
            return "fresh-token"

        with (
            patch("hermes_cli.auth.resolve_nous_access_token", side_effect=_record_refresh),
            patch.object(igt, "fal_key_is_configured", return_value=False),
            patch.object(mtg, "managed_nous_tools_enabled", return_value=True),
        ):
            result = igt.check_fal_api_key()

        assert refresh_calls == [], (
            "check_fal_api_key() triggered an OAuth refresh — "
            "it must use is_managed_tool_gateway_ready instead"
        )
        assert result is True

    def test_returns_true_when_fal_key_configured(self, monkeypatch):
        import tools.image_generation_tool as igt
        with patch.object(igt, "fal_key_is_configured", return_value=True):
            assert igt.check_fal_api_key() is True

    def test_returns_false_when_no_key_and_no_gateway(self, monkeypatch):
        import tools.image_generation_tool as igt
        with (
            patch.object(igt, "fal_key_is_configured", return_value=False),
            patch.object(mtg, "managed_nous_tools_enabled", return_value=False),
        ):
            assert igt.check_fal_api_key() is False


# ---------------------------------------------------------------------------
# _has_openai_audio_backend — tts_tool.py
# ---------------------------------------------------------------------------

class TestHasOpenaiAudioBackendSkipsOAuthRefresh:
    """_has_openai_audio_backend() must not trigger OAuth refresh on availability
    check."""

    def test_no_refresh_when_gateway_available(self, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "valid-nous-token")

        import tools.tts_tool as tts

        refresh_calls: list = []

        def _record_refresh(**_kw):
            refresh_calls.append(1)
            return "fresh-token"

        with (
            patch("hermes_cli.auth.resolve_nous_access_token", side_effect=_record_refresh),
            patch.object(tts, "resolve_openai_audio_api_key", return_value=None),
            patch.object(mtg, "managed_nous_tools_enabled", return_value=True),
        ):
            result = tts._has_openai_audio_backend()

        assert refresh_calls == [], (
            "_has_openai_audio_backend() triggered an OAuth refresh — "
            "it must use is_managed_tool_gateway_ready instead"
        )
        assert result is True

    def test_returns_true_when_direct_api_key_present(self, monkeypatch):
        import tools.tts_tool as tts
        with patch.object(tts, "resolve_openai_audio_api_key", return_value="sk-test"):
            assert tts._has_openai_audio_backend() is True


# ---------------------------------------------------------------------------
# _check_fal_video_available — plugins/video_gen/fal
# ---------------------------------------------------------------------------

class TestCheckFalVideoAvailableSkipsOAuthRefresh:
    """_check_fal_video_available() must not trigger OAuth refresh."""

    def test_no_refresh_when_gateway_available(self, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "valid-nous-token")

        from plugins.video_gen.fal import _check_fal_video_available

        refresh_calls: list = []

        def _record_refresh(**_kw):
            refresh_calls.append(1)
            return "fresh-token"

        with (
            patch("hermes_cli.auth.resolve_nous_access_token", side_effect=_record_refresh),
            patch("tools.tool_backend_helpers.fal_key_is_configured", return_value=False),
            patch.object(mtg, "managed_nous_tools_enabled", return_value=True),
        ):
            result = _check_fal_video_available()

        assert refresh_calls == [], (
            "_check_fal_video_available() triggered an OAuth refresh — "
            "it must use is_managed_tool_gateway_ready instead"
        )
        assert result is True

    def test_returns_true_when_fal_key_configured(self, monkeypatch):
        from plugins.video_gen.fal import _check_fal_video_available
        with patch("tools.tool_backend_helpers.fal_key_is_configured", return_value=True):
            assert _check_fal_video_available() is True
