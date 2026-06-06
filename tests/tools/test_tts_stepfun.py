"""Tests for the StepFun TTS provider in tools/tts_tool.py.

Covers the native ``stepfun`` built-in provider (POST /v1/audio/speech)
added as the replacement for the old ``tts.providers.stepfun-tts``
command-type shim. The provider is the symmetric counterpart of the
StepFun STT provider and is keyed off STEPFUN_API_KEY.

Tests follow the same pattern as test_tts_mistral.py — patching
``requests.post`` via monkeypatch rather than a real SDK — because the
StepFun HTTP API is plain JSON over stdlib-compatible ``requests``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(*, status_code: int = 200, body: bytes = b"", json_body: Any = None):
    """Build a MagicMock that quacks like a requests.Response."""
    response = MagicMock()
    response.status_code = status_code
    response.content = body
    response.text = body.decode("utf-8", errors="replace") if body else ""
    if json_body is not None:
        response.json.return_value = json_body
        response.json.side_effect = None
    else:
        response.json.side_effect = ValueError("not json")
    return response


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Don't let ambient STEPFUN_API_KEY from the host env leak into tests.

    The TTS provider reads STEPFUN_API_KEY / STEPFUN_BASE_URL through
    ``hermes_cli.config.get_env_value``, which consults a cached .env
    loader (not just os.environ). Mock that helper to fall through to
    os.environ so ``monkeypatch.setenv`` inside individual tests can
    inject credentials cleanly. Pattern borrowed from
    tests/hermes_cli/test_runtime_provider_resolution.py.
    """
    for key in ("STEPFUN_API_KEY", "STEPFUN_BASE_URL", "HERMES_SESSION_PLATFORM"):
        monkeypatch.delenv(key, raising=False)

    import hermes_cli.config as _cfg_mod

    def _passthrough(name, default=None):
        return os.environ.get(name, default)

    monkeypatch.setattr(_cfg_mod, "get_env_value", _passthrough)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestStepFunConstants:
    """Sanity-check the documented catalogue so a docs update that drifts
    from the code is caught by CI rather than discovered at runtime."""

    def test_models_contains_both_documented(self):
        from tools.tts_tool import STEPFUN_TTS_MODELS

        assert "step-tts-2" in STEPFUN_TTS_MODELS
        assert "stepaudio-2.5-tts" in STEPFUN_TTS_MODELS

    def test_response_formats_match_documentation(self):
        from tools.tts_tool import STEPFUN_TTS_RESPONSE_FORMATS

        # https://platform.stepfun.ai/docs/en/guides/developer/tts#output-format
        assert STEPFUN_TTS_RESPONSE_FORMATS == frozenset(
            {"mp3", "wav", "flac", "opus", "pcm"}
        )

    def test_emotion_tags_match_documentation(self):
        from tools.tts_tool import STEPFUN_TTS_EMOTION_TAGS

        expected = {
            "Happy", "Very Happy", "Sad", "Angry", "Very Angry", "Coquettish",
            "Fearful", "Surprised", "Excited", "Admiring", "Confused",
        }
        assert STEPFUN_TTS_EMOTION_TAGS == expected

    def test_style_tags_match_documentation(self):
        from tools.tts_tool import STEPFUN_TTS_STYLE_TAGS

        # speaking style + delivery style from the docs page
        expected = {
            "Slow", "Very Slow", "Fast", "Very Fast",
            "Cold", "Embarrassed", "Frustrated", "Proud", "Tender", "Sweet",
            "Outgoing", "Serious", "Arrogant", "Elderly", "Shouting",
            "Sarcastic", "Stuttering",
        }
        assert STEPFUN_TTS_STYLE_TAGS == expected


# ---------------------------------------------------------------------------
# Built-in sync
# ---------------------------------------------------------------------------


class TestBuiltinSync:
    """``stepfun`` must appear in BUILTIN_TTS_PROVIDERS and in
    tts_registry._BUILTIN_NAMES — otherwise the dispatcher's default
    branch would try to use the Edge TTS fallback for ``tts.provider:
    stepfun``."""

    def test_in_tts_tool_builtins(self):
        from tools.tts_tool import BUILTIN_TTS_PROVIDERS

        assert "stepfun" in BUILTIN_TTS_PROVIDERS

    def test_in_tts_registry_builtins(self):
        from agent import tts_registry

        assert "stepfun" in tts_registry._BUILTIN_NAMES

    def test_in_picker_categories(self):
        from hermes_cli import tools_config

        provider_names = {row["tts_provider"] for row in tools_config.TOOL_CATEGORIES["tts"]["providers"]}
        assert "stepfun" in provider_names


# ---------------------------------------------------------------------------
# Helpers: base URL + response_format
# ---------------------------------------------------------------------------


class TestNormalizeBaseUrl:
    """The /v1 base URL is always required for the audio endpoint, but
    /step_plan/v1 is a valid prefix ONLY for the Step Plan tier models
    (stepaudio-2.5-tts). step-tts-2 is PAYG-only and 404s on /step_plan/v1."""

    def test_default_when_empty(self):
        from tools.tts_tool import DEFAULT_STEPFUN_TTS_BASE_URL, _normalize_stepfun_tts_base_url

        assert _normalize_stepfun_tts_base_url("") == DEFAULT_STEPFUN_TTS_BASE_URL
        assert _normalize_stepfun_tts_base_url(None) == DEFAULT_STEPFUN_TTS_BASE_URL

    def test_strips_trailing_slash(self):
        from tools.tts_tool import _normalize_stepfun_tts_base_url

        assert _normalize_stepfun_tts_base_url("https://api.stepfun.ai/v1/") == "https://api.stepfun.ai/v1"

    def test_step_plan_prefix_preserved_for_stepaudio_default(self):
        """stepaudio-2.5-tts (the default) works on /step_plan/v1 — keep
        the prefix so Step Plan subscribers get plan-tier billing."""
        from tools.tts_tool import _normalize_stepfun_tts_base_url

        assert (
            _normalize_stepfun_tts_base_url("https://api.stepfun.ai/step_plan/v1", model="stepaudio-2.5-tts")
            == "https://api.stepfun.ai/step_plan/v1"
        )
        # No model passed (default = stepaudio) should also preserve step_plan.
        assert (
            _normalize_stepfun_tts_base_url("https://api.stepfun.ai/step_plan/v1")
            == "https://api.stepfun.ai/step_plan/v1"
        )

    def test_step_plan_prefix_rewritten_for_steptts2(self):
        """step-tts-2 is PAYG-only; the model isn't in the Step Plan
        catalogue so /step_plan/v1 returns 404. Rewrite to /v1."""
        from tools.tts_tool import _normalize_stepfun_tts_base_url

        assert (
            _normalize_stepfun_tts_base_url("https://api.stepfun.ai/step_plan/v1", model="step-tts-2")
            == "https://api.stepfun.ai/v1"
        )

    def test_anchors_to_v1_when_path_missing(self):
        from tools.tts_tool import _normalize_stepfun_tts_base_url

        assert _normalize_stepfun_tts_base_url("https://api.stepfun.ai") == "https://api.stepfun.ai/v1"

    def test_v2_path_replaced_with_v1(self):
        from tools.tts_tool import _normalize_stepfun_tts_base_url

        # /v2 isn't a real StepFun API path; force callers to /v1 where audio lives.
        assert _normalize_stepfun_tts_base_url("https://api.stepfun.ai/v2") == "https://api.stepfun.ai/v1"


class TestResponseFormatResolution:
    def test_explicit_mp3(self):
        from tools.tts_tool import _resolve_stepfun_tts_response_format

        assert _resolve_stepfun_tts_response_format("/tmp/out.mp3", "mp3") == "mp3"
        assert _resolve_stepfun_tts_response_format("/tmp/out.ogg", "mp3") == "mp3"

    def test_explicit_ogg_alias_to_opus(self):
        from tools.tts_tool import _resolve_stepfun_tts_response_format

        # The natural .ogg extension maps to opus in the StepFun API.
        assert _resolve_stepfun_tts_response_format("/tmp/out.mp3", "ogg") == "opus"

    def test_inferred_from_extension(self):
        from tools.tts_tool import _resolve_stepfun_tts_response_format

        assert _resolve_stepfun_tts_response_format("/tmp/x.wav", None) == "wav"
        assert _resolve_stepfun_tts_response_format("/tmp/x.flac", None) == "flac"
        assert _resolve_stepfun_tts_response_format("/tmp/x.opus", None) == "opus"
        assert _resolve_stepfun_tts_response_format("/tmp/x.pcm", None) == "pcm"

    def test_ogg_extension_maps_to_opus(self):
        from tools.tts_tool import _resolve_stepfun_tts_response_format

        assert _resolve_stepfun_tts_response_format("/tmp/x.ogg", None) == "opus"

    def test_unknown_extension_falls_back_to_mp3(self):
        from tools.tts_tool import _resolve_stepfun_tts_response_format

        assert _resolve_stepfun_tts_response_format("/tmp/x.txt", None) == "mp3"
        assert _resolve_stepfun_tts_response_format("/tmp/x", None) == "mp3"

    def test_unknown_format_string_falls_through_to_extension(self):
        from tools.tts_tool import _resolve_stepfun_tts_response_format

        # Garbage in the config field should not break the request — fall through
        # to extension inference rather than passing nonsense to the API.
        assert _resolve_stepfun_tts_response_format("/tmp/x.wav", "garbage") == "wav"


# ---------------------------------------------------------------------------
# _generate_stepfun_tts — the meat
# ---------------------------------------------------------------------------


def _patched_post(monkeypatch, response: MagicMock):
    """Patch ``requests.post`` inside tools.tts_tool — the provider imports
    it lazily inside the function body."""
    import requests

    captured: Dict[str, Any] = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return response

    monkeypatch.setattr(requests, "post", fake_post)
    return captured


class TestGenerateStepFunTts:
    def test_missing_api_key_raises(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_stepfun_tts

        output_path = str(tmp_path / "test.mp3")
        with pytest.raises(ValueError, match="STEPFUN_API_KEY"):
            _generate_stepfun_tts("Hello", output_path, {})

    def test_successful_mp3_request(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        audio = b"fake-mp3-bytes"
        response = _make_response(status_code=200, body=audio)
        captured = _patched_post(monkeypatch, response)

        output_path = str(tmp_path / "out.mp3")
        result = _generate_stepfun_tts("Hello world", output_path, {})

        assert result == output_path
        assert (tmp_path / "out.mp3").read_bytes() == audio
        # Default is now stepaudio-2.5-tts so Step Plan subscribers hit
        # the plan tier; URL is /step_plan/v1 (no STEPFUN_BASE_URL in env
        # so the default is used).
        assert captured["url"] == "https://api.stepfun.ai/step_plan/v1/audio/speech"
        assert captured["headers"]["Authorization"] == "Bearer test-key"
        assert captured["headers"]["Content-Type"] == "application/json"
        assert captured["json"]["model"] == "stepaudio-2.5-tts"
        assert captured["json"]["input"] == "Hello world"
        assert captured["json"]["voice"] == "lively-girl"
        assert captured["json"]["response_format"] == "mp3"
        assert captured["json"]["speed"] == 1.0
        assert captured["json"]["volume"] == 1.0
        # stepaudio-2.5-tts default branch must NOT include emotion/style
        assert "emotion" not in captured["json"]
        assert "style" not in captured["json"]
        # No voice_label on either model
        assert "voice_label" not in captured["json"]

    def test_steptts2_emotion_and_style_sent_as_flat_top_level_fields(self, tmp_path, monkeypatch):
        """step-tts-2 takes ``emotion`` and ``style`` as FLAT top-level
        fields, not nested under voice_label.* — the API rejects the
        nested shape with HTTP 400 ('too many voice label fields') when
        both keys are present, but accepts the flat shape with both keys
        set in a single request."""
        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        captured = _patched_post(monkeypatch, _make_response(body=b"x"))

        config = {
            "stepfun": {
                "model": "step-tts-2",
                "voice": "elegantgentle-female",
                "emotion": "Happy",
                "style": "Slow",
            },
        }
        _generate_stepfun_tts("hi", str(tmp_path / "x.mp3"), config)

        assert captured["json"]["model"] == "step-tts-2"
        assert captured["json"]["voice"] == "elegantgentle-female"
        # FLAT top-level fields, not nested.
        assert captured["json"]["emotion"] == "Happy"
        assert captured["json"]["style"] == "Slow"
        # Must NOT include the nested voice_label shape — it would 400.
        assert "voice_label" not in captured["json"]
        assert "instruction" not in captured["json"]

    def test_steptts2_emotion_only(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        captured = _patched_post(monkeypatch, _make_response(body=b"x"))

        config = {"stepfun": {"model": "step-tts-2", "emotion": "Sad"}}
        _generate_stepfun_tts("hi", str(tmp_path / "x.mp3"), config)

        assert captured["json"]["emotion"] == "Sad"
        assert "style" not in captured["json"]
        assert "voice_label" not in captured["json"]

    def test_steptts2_style_only(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        captured = _patched_post(monkeypatch, _make_response(body=b"x"))

        config = {"stepfun": {"model": "step-tts-2", "style": "Tender"}}
        _generate_stepfun_tts("hi", str(tmp_path / "x.mp3"), config)

        assert captured["json"]["style"] == "Tender"
        assert "emotion" not in captured["json"]
        assert "voice_label" not in captured["json"]

    def test_stepaudio25tts_uses_instruction_not_voice_label(self, tmp_path, monkeypatch, caplog):
        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        captured = _patched_post(monkeypatch, _make_response(body=b"x"))

        config = {
            "stepfun": {
                "model": "stepaudio-2.5-tts",
                "voice": "lively-girl",
                "instruction": "Say cheerfully and with excitement",
            },
        }
        _generate_stepfun_tts("hi", str(tmp_path / "x.mp3"), config)

        assert captured["json"]["model"] == "stepaudio-2.5-tts"
        assert captured["json"]["instruction"] == "Say cheerfully and with excitement"
        # stepaudio-2.5-tts does NOT support voice_label — must be absent
        assert "voice_label" not in captured["json"]

    def test_stepaudio25tts_drops_emotion_and_style_with_log(self, tmp_path, monkeypatch, caplog):
        """Cross-model misconfiguration: user set emotion/style but is on
        stepaudio-2.5-tts. Drop the fields and log so the operator can see
        why their config is being ignored."""
        import logging

        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        captured = _patched_post(monkeypatch, _make_response(body=b"x"))

        config = {
            "stepfun": {
                "model": "stepaudio-2.5-tts",
                "emotion": "Happy",
                "style": "Slow",
                "instruction": "speak gently",
            },
        }
        with caplog.at_level(logging.INFO, logger="tools.tts_tool"):
            _generate_stepfun_tts("hi", str(tmp_path / "x.mp3"), config)

        # emotion/style must NOT appear in the request for stepaudio-2.5-tts
        assert "emotion" not in captured["json"]
        assert "style" not in captured["json"]
        assert "voice_label" not in captured["json"]
        assert captured["json"]["instruction"] == "speak gently"
        assert "emotion" in caplog.text
        assert "style" in caplog.text

    def test_steptts2_drops_instruction_with_log(self, tmp_path, monkeypatch, caplog):
        """Symmetric case: user has instruction set but is on step-tts-2.
        Don't silently swallow — log so they know to switch models or
        remove the field."""
        import logging

        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        captured = _patched_post(monkeypatch, _make_response(body=b"x"))

        config = {
            "stepfun": {
                "model": "step-tts-2",
                "emotion": "Happy",
                "instruction": "should be ignored",
            },
        }
        with caplog.at_level(logging.INFO, logger="tools.tts_tool"):
            _generate_stepfun_tts("hi", str(tmp_path / "x.mp3"), config)

        # instruction must NOT appear in the request for step-tts-2
        assert "instruction" not in captured["json"]
        # emotion is preserved as a flat top-level field
        assert captured["json"]["emotion"] == "Happy"
        assert "voice_label" not in captured["json"]
        assert "instruction" in caplog.text
        assert "stepaudio-2.5-tts" in caplog.text

    def test_speed_and_volume_clamped(self, tmp_path, monkeypatch):
        """Speed and volume get clamped to the documented ranges so a typo
        doesn't trigger a 400 from the API."""
        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        captured = _patched_post(monkeypatch, _make_response(body=b"x"))

        # Both fields are out of range — must be clamped, not raised.
        config = {"stepfun": {"speed": 99.0, "volume": -1.0}}
        _generate_stepfun_tts("hi", str(tmp_path / "x.mp3"), config)

        assert captured["json"]["speed"] == 2.0
        assert captured["json"]["volume"] == 0.1

    def test_speed_and_volume_garbage_falls_back_to_default(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        captured = _patched_post(monkeypatch, _make_response(body=b"x"))

        config = {"stepfun": {"speed": "fast", "volume": "loud"}}
        _generate_stepfun_tts("hi", str(tmp_path / "x.mp3"), config)

        assert captured["json"]["speed"] == 1.0
        assert captured["json"]["volume"] == 1.0

    def test_unknown_model_warns_but_still_sends(self, tmp_path, monkeypatch, caplog):
        """If StepFun publishes a new TTS model before we ship a code
        update, the user can set it manually and it should still be sent
        (with a warning) rather than raising."""
        import logging

        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        captured = _patched_post(monkeypatch, _make_response(body=b"x"))

        with caplog.at_level(logging.WARNING, logger="tools.tts_tool"):
            _generate_stepfun_tts(
                "hi", str(tmp_path / "x.mp3"),
                {"stepfun": {"model": "step-tts-3-experimental"}},
            )

        assert captured["json"]["model"] == "step-tts-3-experimental"
        assert "not in the known set" in caplog.text

    def test_response_format_from_explicit_config(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        captured = _patched_post(monkeypatch, _make_response(body=b"x"))

        config = {"stepfun": {"response_format": "opus"}}
        _generate_stepfun_tts("hi", str(tmp_path / "x.mp3"), config)

        assert captured["json"]["response_format"] == "opus"

    def test_response_format_inferred_from_ogg_extension(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        captured = _patched_post(monkeypatch, _make_response(body=b"x"))

        _generate_stepfun_tts("hi", str(tmp_path / "x.ogg"), {})

        assert captured["json"]["response_format"] == "opus"

    def test_step_plan_base_url_preserved_for_default_stepaudio(self, tmp_path, monkeypatch):
        """Default model is stepaudio-2.5-tts, which works on /step_plan/v1.
        With STEPFUN_BASE_URL set to the Step Plan prefix, the provider
        must preserve it (not rewrite) so the request hits the plan tier."""
        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        monkeypatch.setenv("STEPFUN_BASE_URL", "https://api.stepfun.ai/step_plan/v1")
        captured = _patched_post(monkeypatch, _make_response(body=b"x"))

        _generate_stepfun_tts("hi", str(tmp_path / "x.mp3"), {})

        assert captured["url"] == "https://api.stepfun.ai/step_plan/v1/audio/speech"
        assert captured["json"]["model"] == "stepaudio-2.5-tts"

    def test_step_plan_base_url_rewritten_when_steptts2_explicit(self, tmp_path, monkeypatch):
        """If the user explicitly picks step-tts-2, the Step Plan URL is
        not valid for that model and must be rewritten to /v1."""
        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        monkeypatch.setenv("STEPFUN_BASE_URL", "https://api.stepfun.ai/step_plan/v1")
        captured = _patched_post(monkeypatch, _make_response(body=b"x"))

        config = {"stepfun": {"model": "step-tts-2"}}
        _generate_stepfun_tts("hi", str(tmp_path / "x.mp3"), config)

        assert captured["url"] == "https://api.stepfun.ai/v1/audio/speech"
        assert captured["json"]["model"] == "step-tts-2"

    def test_config_base_url_overrides_env(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        monkeypatch.setenv("STEPFUN_BASE_URL", "https://api.stepfun.ai/v1")
        captured = _patched_post(monkeypatch, _make_response(body=b"x"))

        # Config wins over env, same as how STT resolves it.
        config = {"stepfun": {"base_url": "https://api.stepfun.ai"}}
        _generate_stepfun_tts("hi", str(tmp_path / "x.mp3"), config)

        # /v1 anchor applied
        assert captured["url"] == "https://api.stepfun.ai/v1/audio/speech"

    def test_http_error_raises_with_api_message(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        err_body = {"error": {"message": "voice not found", "type": "invalid_request_error"}}
        _patched_post(monkeypatch, _make_response(status_code=400, json_body=err_body))

        with pytest.raises(RuntimeError, match="StepFun TTS API error.*HTTP 400.*voice not found"):
            _generate_stepfun_tts("hi", str(tmp_path / "x.mp3"), {})

    def test_http_error_with_plain_text_body(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        _patched_post(monkeypatch, _make_response(status_code=500, body=b"internal server error"))

        with pytest.raises(RuntimeError, match="HTTP 500.*internal server error"):
            _generate_stepfun_tts("hi", str(tmp_path / "x.mp3"), {})

    def test_empty_response_body_raises(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        _patched_post(monkeypatch, _make_response(status_code=200, body=b""))

        with pytest.raises(RuntimeError, match="empty response body"):
            _generate_stepfun_tts("hi", str(tmp_path / "x.mp3"), {})

    def test_emotion_field_accepts_style_tag(self, tmp_path, monkeypatch):
        """The docs bucket tags into emotion / style, but the actual API
        accepts any tag in either field slot. We pass through whatever
        the user configured."""
        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        captured = _patched_post(monkeypatch, _make_response(body=b"x"))

        # Slow is a speaking-style tag — but we put it in the emotion slot
        # because that's what the user typed. The API happily accepts it.
        # Pin to step-tts-2 because that's the model that supports
        # emotion/style flat fields (stepaudio-2.5-tts would drop them).
        config = {"stepfun": {"model": "step-tts-2", "emotion": "Slow"}}
        _generate_stepfun_tts("hi", str(tmp_path / "x.mp3"), config)

        assert captured["json"]["emotion"] == "Slow"

    def test_style_field_accepts_emotion_tag(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        captured = _patched_post(monkeypatch, _make_response(body=b"x"))

        # Happy is an emotion tag — but in the style slot, the API accepts it.
        # Pin to step-tts-2 because that's the model that supports
        # emotion/style flat fields (stepaudio-2.5-tts would drop them).
        config = {"stepfun": {"model": "step-tts-2", "style": "Happy"}}
        _generate_stepfun_tts("hi", str(tmp_path / "x.mp3"), config)

        assert captured["json"]["style"] == "Happy"

    def test_does_not_send_nested_voice_label_shape(self, tmp_path, monkeypatch):
        """The docs page shows a ``voice_label`` nested shape, but the
        API rejects it with HTTP 400 ('too many voice label fields').
        The provider must always send the FLAT shape."""
        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        captured = _patched_post(monkeypatch, _make_response(body=b"x"))

        config = {
            "stepfun": {
                "model": "step-tts-2",
                "emotion": "Happy",
                "style": "Slow",
            },
        }
        _generate_stepfun_tts("hi", str(tmp_path / "x.mp3"), config)

        # Top-level flat fields only.
        assert "voice_label" not in captured["json"]
        # The actual fields ARE there, just at the top level.
        assert captured["json"].get("emotion") == "Happy"
        assert captured["json"].get("style") == "Slow"

    def test_unknown_emotion_warns_but_sends(self, tmp_path, monkeypatch, caplog):
        """User typos an emotion that isn't in the docs — warn loudly but
        still send it. The API may accept it (custom tag) or reject with
        a 400, but at least the request goes out with the user's intent."""
        import logging

        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        captured = _patched_post(monkeypatch, _make_response(body=b"x"))

        # Pin to step-tts-2 because that's the model that supports
        # emotion/style (stepaudio-2.5-tts would drop them with a log).
        config = {"stepfun": {"model": "step-tts-2", "emotion": "Overjoyed"}}
        with caplog.at_level(logging.WARNING, logger="tools.tts_tool"):
            _generate_stepfun_tts("hi", str(tmp_path / "x.mp3"), config)

        assert captured["json"]["emotion"] == "Overjoyed"
        assert "not in the documented step-tts-2" in caplog.text
        assert "tag set" in caplog.text

    def test_voice_id_override(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        captured = _patched_post(monkeypatch, _make_response(body=b"x"))

        config = {"stepfun": {"voice": "lengyanyujie"}}
        _generate_stepfun_tts("hi", str(tmp_path / "x.mp3"), config)

        assert captured["json"]["voice"] == "lengyanyujie"

    def test_request_uses_default_timeout(self, tmp_path, monkeypatch):
        """The provider must apply its own timeout to prevent the gateway
        from hanging forever on a stalled StepFun connection."""
        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        captured = _patched_post(monkeypatch, _make_response(body=b"x"))

        _generate_stepfun_tts("hi", str(tmp_path / "x.mp3"), {})

        assert captured["timeout"] == 60

    def test_text_passed_through_unchanged(self, tmp_path, monkeypatch):
        """The provider must NOT touch the input text — no auto-prepend
        of "Say cheerfully:" or similar transformations. Users who want
        that should use the instruction field on stepaudio-2.5-tts."""
        from tools.tts_tool import _generate_stepfun_tts

        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        captured = _patched_post(monkeypatch, _make_response(body=b"x"))

        text = "안녕하세요, 오늘은 좋은 날이에요."
        _generate_stepfun_tts(text, str(tmp_path / "x.mp3"), {})

        assert captured["json"]["input"] == text


# ---------------------------------------------------------------------------
# check_tts_requirements
# ---------------------------------------------------------------------------


class TestRequirementsCheck:
    def test_stepfun_counts_as_available(self, monkeypatch):
        from tools.tts_tool import check_tts_requirements

        monkeypatch.setenv("STEPFUN_API_KEY", "sk-test")
        assert check_tts_requirements() is True

    def test_no_stepfun_key_means_other_providers_still_count(self, monkeypatch):
        """If STEPFUN_API_KEY isn't set, the provider isn't a blocker for
        overall TTS availability (other built-ins still work)."""
        from tools.tts_tool import check_tts_requirements

        # Deliberately do NOT set STEPFUN_API_KEY. Make sure other env vars
        # are clean too so the test isn't accidentally tripping on a sibling.
        for key in ("STEPFUN_API_KEY", "ELEVENLABS_API_KEY", "GEMINI_API_KEY",
                    "MISTRAL_API_KEY", "XAI_API_KEY", "OPENAI_API_KEY",
                    "VOICE_TOOLS_OPENAI_KEY", "MINIMAX_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(key, raising=False)

        # The function only returns False when literally NO provider is
        # available. Edge TTS may or may not be installed in the test env,
        # so the safest assertion is: the function returns *some* bool,
        # not a KeyError / exception. With StepFun unconfigured, this
        # exercises the fallback path.
        result = check_tts_requirements()
        assert isinstance(result, bool)
