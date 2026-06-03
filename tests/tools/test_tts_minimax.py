"""Tests for the MiniMax TTS provider in tools/tts_tool.py.

Covers the speech-2.8-hd / speech-2.8-turbo model pass-through (Plus plan),
both response shapes (t2a_v2 hex-encoded JSON and raw-audio fallback),
the MINIMAX_GROUP_ID query-param injection, and the dispatcher's routing
to the minimax branch.
"""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Wipe MiniMax env vars so tests don't pick up the developer's key."""
    for key in ("MINIMAX_API_KEY", "MINIMAX_GROUP_ID", "HERMES_SESSION_PLATFORM"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_requests_post():
    """Patch the ``requests.post`` call site used by _generate_minimax_tts.

    The provider does ``import requests`` *inside* the function, so we
    must patch the module-level ``requests.post`` (not
    ``tools.tts_tool.requests``, which never gets set).
    """
    with patch("requests.post") as post:
        yield post


def _hex_audio(b: bytes) -> str:
    """Return the lowercase hex encoding ``_generate_minimax_tts`` expects."""
    return b.hex()


def _mock_response(json_body=None, raw_bytes=None, content_type=""):
    """Build a MagicMock stand-in for a requests.Response."""
    resp = MagicMock()
    if json_body is not None:
        resp.json.return_value = json_body
        resp.content = b""
    if raw_bytes is not None:
        resp.content = raw_bytes
        resp.headers = {"Content-Type": content_type}
    if json_body is None and raw_bytes is None:
        resp.raise_for_status.side_effect = RuntimeError("HTTP error")
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Direct _generate_minimax_tts() tests
# ---------------------------------------------------------------------------


class TestGenerateMinimaxTts:
    def test_missing_api_key_raises_value_error(self, tmp_path):
        from tools.tts_tool import _generate_minimax_tts

        output_path = str(tmp_path / "test.mp3")
        with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
            _generate_minimax_tts("Hello", output_path, {})

    def test_t2a_v2_hex_audio_response_speech_2_8_hd(
        self, tmp_path, monkeypatch, mock_requests_post
    ):
        """t2a_v2 endpoint returns hex-encoded audio inside JSON. Verify
        the speech-2.8-hd model value is passed through unchanged and the
        hex audio is decoded to the output file."""
        from tools.tts_tool import _generate_minimax_tts

        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        audio_bytes = b"\xff\xfb\x90\x00fake-mp3-bytes-speech-2.8-hd"
        mock_requests_post.return_value = _mock_response(
            json_body={
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "data": {"audio": _hex_audio(audio_bytes)},
            }
        )
        output_path = str(tmp_path / "test.mp3")
        config = {"minimax": {"model": "speech-2.8-hd"}}
        result = _generate_minimax_tts("Hello", output_path, config)

        assert result == output_path
        assert (tmp_path / "test.mp3").read_bytes() == audio_bytes
        # Endpoint URL preserved (t2a_v2 path)
        called_url = mock_requests_post.call_args[0][0]
        assert "t2a_v2" in called_url
        # Payload shape: nested voice_setting/audio_setting, model pass-through
        payload = mock_requests_post.call_args.kwargs["json"]
        assert payload["model"] == "speech-2.8-hd"
        assert payload["text"] == "Hello"
        assert payload["voice_setting"]["voice_id"]
        assert payload["audio_setting"]["format"] == "mp3"
        # Auth header
        assert (
            mock_requests_post.call_args.kwargs["headers"]["Authorization"]
            == "Bearer test-key"
        )

    def test_t2a_v2_speech_2_8_turbo_model_pass_through(
        self, tmp_path, monkeypatch, mock_requests_post
    ):
        """The Plus-plan turbo model must also be passed through verbatim."""
        from tools.tts_tool import _generate_minimax_tts

        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        audio_bytes = b"turbo-audio-bytes"
        mock_requests_post.return_value = _mock_response(
            json_body={
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "data": {"audio": _hex_audio(audio_bytes)},
            }
        )
        output_path = str(tmp_path / "test.mp3")
        config = {"minimax": {"model": "speech-2.8-turbo"}}
        _generate_minimax_tts("Hi", output_path, config)

        payload = mock_requests_post.call_args.kwargs["json"]
        assert payload["model"] == "speech-2.8-turbo"

    def test_t2a_v2_api_error_surfaces_status_code_and_msg(
        self, tmp_path, monkeypatch, mock_requests_post
    ):
        from tools.tts_tool import _generate_minimax_tts

        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        mock_requests_post.return_value = _mock_response(
            json_body={
                "base_resp": {"status_code": 1004, "status_msg": "insufficient balance"},
                "data": {},
            }
        )
        output_path = str(tmp_path / "test.mp3")
        with pytest.raises(RuntimeError, match="1004"):
            _generate_minimax_tts("Hi", output_path, {})

    def test_t2a_v2_empty_audio_raises(
        self, tmp_path, monkeypatch, mock_requests_post
    ):
        from tools.tts_tool import _generate_minimax_tts

        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        mock_requests_post.return_value = _mock_response(
            json_body={
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "data": {"audio": ""},
            }
        )
        output_path = str(tmp_path / "test.mp3")
        with pytest.raises(RuntimeError, match="empty audio"):
            _generate_minimax_tts("Hi", output_path, {})

    def test_raw_audio_fallback_endpoint(
        self, tmp_path, monkeypatch, mock_requests_post
    ):
        """The v1/text_to_speech endpoint returns audio bytes directly with
        Content-Type: audio/mpeg. Flat payload, no nested voice_setting."""
        from tools.tts_tool import _generate_minimax_tts

        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        audio_bytes = b"raw-mp3-bytes"
        mock_requests_post.return_value = _mock_response(
            raw_bytes=audio_bytes, content_type="audio/mpeg"
        )
        # URL must NOT contain "t2a_v2" so the flat-payload branch fires
        output_path = str(tmp_path / "test.mp3")
        config = {
            "minimax": {
                "base_url": "https://api.minimax.io/v1/text_to_speech",
                "model": "speech-2.8-hd",
            }
        }
        result = _generate_minimax_tts("Hello", output_path, config)

        assert result == output_path
        assert (tmp_path / "test.mp3").read_bytes() == audio_bytes
        payload = mock_requests_post.call_args.kwargs["json"]
        # Flat payload, no nested voice_setting/audio_setting
        assert payload["model"] == "speech-2.8-hd"
        assert payload["text"] == "Hello"
        assert payload["voice_id"]
        assert "voice_setting" not in payload

    def test_group_id_injected_from_config(
        self, tmp_path, monkeypatch, mock_requests_post
    ):
        """GroupId query param is appended when config has group_id and the
        base URL does not already carry one."""
        from tools.tts_tool import _generate_minimax_tts

        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        mock_requests_post.return_value = _mock_response(
            json_body={
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "data": {"audio": _hex_audio(b"x")},
            }
        )
        config = {"minimax": {"group_id": "1234567890"}}
        _generate_minimax_tts("Hi", str(tmp_path / "test.mp3"), config)

        called_url = mock_requests_post.call_args[0][0]
        assert "GroupId=1234567890" in called_url

    def test_group_id_injected_from_env(
        self, tmp_path, monkeypatch, mock_requests_post
    ):
        from tools.tts_tool import _generate_minimax_tts

        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        monkeypatch.setenv("MINIMAX_GROUP_ID", "env-group-id")
        mock_requests_post.return_value = _mock_response(
            json_body={
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "data": {"audio": _hex_audio(b"x")},
            }
        )
        _generate_minimax_tts("Hi", str(tmp_path / "test.mp3"), {})

        called_url = mock_requests_post.call_args[0][0]
        assert "GroupId=env-group-id" in called_url

    def test_group_id_not_double_appended(
        self, tmp_path, monkeypatch, mock_requests_post
    ):
        """If the user already put GroupId in the base URL, do not append
        a second one."""
        from tools.tts_tool import _generate_minimax_tts

        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        mock_requests_post.return_value = _mock_response(
            json_body={
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "data": {"audio": _hex_audio(b"x")},
            }
        )
        config = {
            "minimax": {
                "base_url": "https://api.minimax.io/v1/t2a_v2?GroupId=already-set",
                "group_id": "should-be-ignored",
            }
        }
        _generate_minimax_tts("Hi", str(tmp_path / "test.mp3"), config)

        called_url = mock_requests_post.call_args[0][0]
        assert called_url.count("GroupId=") == 1
        assert "GroupId=already-set" in called_url

    def test_default_model_used_when_config_absent(
        self, tmp_path, monkeypatch, mock_requests_post
    ):
        """The code default (DEFAULT_MINIMAX_MODEL) is sent when the user
        does not override it. We don't pin the value here — that lives in
        tts_tool.py — we just verify the call has a non-empty model field
        and starts with the documented "speech-" prefix."""
        from tools.tts_tool import _generate_minimax_tts

        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        mock_requests_post.return_value = _mock_response(
            json_body={
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "data": {"audio": _hex_audio(b"x")},
            }
        )
        _generate_minimax_tts("Hi", str(tmp_path / "test.mp3"), {})

        payload = mock_requests_post.call_args.kwargs["json"]
        assert isinstance(payload["model"], str) and payload["model"]
        # All currently supported MiniMax model ids start with "speech-"
        assert payload["model"].startswith("speech-")

    def test_voice_setting_uses_configured_voice_id(
        self, tmp_path, monkeypatch, mock_requests_post
    ):
        from tools.tts_tool import _generate_minimax_tts

        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        mock_requests_post.return_value = _mock_response(
            json_body={
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "data": {"audio": _hex_audio(b"x")},
            }
        )
        config = {"minimax": {"voice_id": "English_Graceful_Lady"}}
        _generate_minimax_tts("Hi", str(tmp_path / "test.mp3"), config)

        payload = mock_requests_post.call_args.kwargs["json"]
        assert payload["voice_setting"]["voice_id"] == "English_Graceful_Lady"


# ---------------------------------------------------------------------------
# Built-in registry & dispatcher routing
# ---------------------------------------------------------------------------


class TestMinimaxTtsBuiltinRegistry:
    def test_minimax_is_in_builtin_tts_providers(self):
        from tools.tts_tool import BUILTIN_TTS_PROVIDERS

        assert "minimax" in BUILTIN_TTS_PROVIDERS

    def test_minimax_is_in_builtin_names_set(self):
        """Keep BUILTIN_TTS_PROVIDERS in tools.tts_tool and _BUILTIN_NAMES
        in agent.tts_registry in sync — they share the invariant that
        built-in names always win over plugins."""
        from agent.tts_registry import _BUILTIN_NAMES

        assert "minimax" in _BUILTIN_NAMES


class TestTtsDispatcherMinimax:
    def test_dispatcher_routes_to_minimax_branch(
        self, tmp_path, monkeypatch, mock_requests_post
    ):
        """text_to_speech_tool with provider=minimax should call
        _generate_minimax_tts and return success=True."""
        from tools.tts_tool import text_to_speech_tool

        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        mock_requests_post.return_value = _mock_response(
            json_body={
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "data": {"audio": _hex_audio(b"dispatcher-audio")},
            }
        )
        with patch(
            "tools.tts_tool._load_tts_config",
            return_value={"provider": "minimax"},
        ):
            result = json.loads(
                text_to_speech_tool(
                    "Hello", output_path=str(tmp_path / "out.mp3")
                )
            )

        assert result["success"] is True
        # The dispatcher reports the provider name in the response
        assert result.get("provider") == "minimax"
        # The audio bytes landed on disk
        out_file = tmp_path / "out.mp3"
        assert out_file.read_bytes() == b"dispatcher-audio"

    def test_dispatcher_returns_error_when_key_missing(
        self, tmp_path, monkeypatch
    ):
        """When MINIMAX_API_KEY is unset, the dispatcher should surface
        a clean error JSON rather than a 500 traceback."""
        from tools.tts_tool import text_to_speech_tool

        with patch(
            "tools.tts_tool._load_tts_config",
            return_value={"provider": "minimax"},
        ):
            result = json.loads(
                text_to_speech_tool(
                    "Hello", output_path=str(tmp_path / "out.mp3")
                )
            )

        assert result["success"] is False
        assert "MINIMAX_API_KEY" in result["error"]
