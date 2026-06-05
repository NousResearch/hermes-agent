"""Tests for the QQ voice-message tool (qq_send_voice).

Tests real logic: QQ-id coercion, record-message / send-param building,
audio-file reading, TTS integration, and handler validation/branches, with
mocked TTS and OneBot delivery so no network or speech engine is touched.
"""

import base64
import json
from unittest.mock import patch

import pytest

from tools import qq_voice_tool as voice
from tools.qq_voice_tool import (
    _build_record_message,
    _build_send_params,
    _coerce_qq_id,
    _handle_qq_send_voice,
    _read_audio_file,
    _synthesize_speech,
)


class TestCoerceQqId:
    def test_accepts_int(self):
        assert _coerce_qq_id(10001, "user_id") == 10001

    def test_accepts_numeric_string(self):
        assert _coerce_qq_id("10001", "user_id") == 10001

    def test_strips_whitespace(self):
        assert _coerce_qq_id("  10001  ", "group_id") == 10001

    def test_rejects_non_numeric(self):
        with pytest.raises(ValueError, match="numeric QQ id"):
            _coerce_qq_id("not-a-number", "user_id")

    def test_rejects_none(self):
        with pytest.raises(ValueError, match="numeric QQ id"):
            _coerce_qq_id(None, "user_id")

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="positive QQ id"):
            _coerce_qq_id(0, "group_id")

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="positive QQ id"):
            _coerce_qq_id(-5, "user_id")


class TestBuildRecordMessage:
    def test_single_record_segment(self):
        msg = _build_record_message("QUJD")
        assert msg == [{"type": "record", "data": {"file": "base64://QUJD"}}]

    def test_file_uses_base64_scheme(self):
        msg = _build_record_message("ZGF0YQ==")
        assert msg[0]["data"]["file"].startswith("base64://")


class TestBuildSendParams:
    def test_group_target(self):
        params = _build_send_params([{"type": "record"}], None, 22222)
        assert params["message_type"] == "group"
        assert params["group_id"] == 22222
        assert "user_id" not in params

    def test_private_target(self):
        params = _build_send_params([{"type": "record"}], 11111, None)
        assert params["message_type"] == "private"
        assert params["user_id"] == 11111
        assert "group_id" not in params

    def test_message_is_forwarded(self):
        message = [{"type": "record", "data": {"file": "base64://x"}}]
        params = _build_send_params(message, 1, None)
        assert params["message"] is message


class TestReadAudioFile:
    def test_reads_valid_audio(self, tmp_path):
        f = tmp_path / "clip.mp3"
        f.write_bytes(b"ID3fake-audio-bytes")
        assert _read_audio_file(str(f)) == b"ID3fake-audio-bytes"

    def test_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            _read_audio_file(str(tmp_path / "nope.mp3"))

    def test_unsupported_extension(self, tmp_path):
        bad = tmp_path / "clip.txt"
        bad.write_bytes(b"data")
        with pytest.raises(ValueError, match="unsupported audio type"):
            _read_audio_file(str(bad))

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.wav"
        empty.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            _read_audio_file(str(empty))

    def test_oversized_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(voice, "_MAX_AUDIO_BYTES", 4)
        big = tmp_path / "big.mp3"
        big.write_bytes(b"way too many audio bytes")
        with pytest.raises(ValueError, match="too large"):
            _read_audio_file(str(big))


class TestSynthesizeSpeech:
    def test_success_returns_file_path(self, tmp_path):
        audio = tmp_path / "tts.mp3"
        audio.write_bytes(b"audio")
        payload = json.dumps({"success": True, "file_path": str(audio)})
        with patch("tools.tts_tool.text_to_speech_tool", return_value=payload):
            assert _synthesize_speech("hello") == str(audio)

    def test_failure_surfaced(self):
        payload = json.dumps({"success": False, "error": "no TTS provider"})
        with patch("tools.tts_tool.text_to_speech_tool", return_value=payload):
            with pytest.raises(RuntimeError, match="no TTS provider"):
                _synthesize_speech("hello")

    def test_non_json_result(self):
        with patch("tools.tts_tool.text_to_speech_tool", return_value="<oops>"):
            with pytest.raises(RuntimeError, match="unparseable"):
                _synthesize_speech("hello")

    def test_success_but_missing_file(self, tmp_path):
        payload = json.dumps(
            {"success": True, "file_path": str(tmp_path / "ghost.mp3")}
        )
        with patch("tools.tts_tool.text_to_speech_tool", return_value=payload):
            with pytest.raises(RuntimeError, match="no file"):
                _synthesize_speech("hello")


class TestHandlerValidation:
    def test_no_text_no_audio_rejected(self):
        result = json.loads(_handle_qq_send_voice({"user_id": "1"}))
        assert "error" in result
        assert "'text'" in result["error"]

    def test_both_text_and_audio_rejected(self):
        result = json.loads(
            _handle_qq_send_voice(
                {"text": "hi", "audio_file": "/tmp/a.mp3", "user_id": "1"}
            )
        )
        assert "error" in result
        assert "not both" in result["error"]

    def test_no_target_rejected(self):
        result = json.loads(_handle_qq_send_voice({"text": "hi"}))
        assert "error" in result
        assert "exactly one target" in result["error"]

    def test_both_targets_rejected(self):
        result = json.loads(
            _handle_qq_send_voice(
                {"text": "hi", "user_id": "1", "group_id": "2"}
            )
        )
        assert "error" in result
        assert "exactly one target" in result["error"]

    def test_bad_group_id_rejected(self):
        result = json.loads(
            _handle_qq_send_voice({"text": "hi", "group_id": "not-a-number"})
        )
        assert "error" in result
        assert "numeric QQ id" in result["error"]

    def test_bad_audio_path_fails_before_network(self, tmp_path):
        result = json.loads(
            _handle_qq_send_voice(
                {"audio_file": str(tmp_path / "missing.mp3"), "user_id": "1"}
            )
        )
        assert "error" in result
        assert "not found" in result["error"]


class TestHandlerSend:
    def test_text_to_private(self, tmp_path):
        audio = tmp_path / "speech.mp3"
        audio.write_bytes(b"synth-audio")
        with (
            patch.object(voice, "_synthesize_speech", return_value=str(audio)),
            patch.object(
                voice, "onebot_call", return_value={"message_id": 555}
            ) as call,
        ):
            result = json.loads(
                _handle_qq_send_voice({"text": "hello", "user_id": "10001"})
            )
        assert result["success"] is True
        assert result["message_id"] == 555
        assert result["synthesized"] is True
        assert result["target"] == {"type": "private", "id": 10001}

        action, params = call.call_args.args
        assert action == "send_msg"
        assert params["message_type"] == "private"
        assert params["user_id"] == 10001
        segment = params["message"][0]
        assert segment["type"] == "record"
        assert segment["data"]["file"].startswith("base64://")
        encoded = segment["data"]["file"][len("base64://") :]
        assert base64.b64decode(encoded) == b"synth-audio"

    def test_audio_file_to_group(self, tmp_path):
        audio = tmp_path / "memo.mp3"
        audio.write_bytes(b"recorded-audio")
        with patch.object(
            voice, "onebot_call", return_value={"message_id": 777}
        ) as call:
            result = json.loads(
                _handle_qq_send_voice(
                    {"audio_file": str(audio), "group_id": "22222"}
                )
            )
        assert result["success"] is True
        assert result["message_id"] == 777
        assert result["synthesized"] is False
        assert result["target"] == {"type": "group", "id": 22222}
        _, params = call.call_args.args
        assert params["message_type"] == "group"
        assert params["group_id"] == 22222

    def test_synthesis_failure_surfaced(self):
        with patch.object(
            voice, "_synthesize_speech", side_effect=RuntimeError("no tts")
        ):
            result = json.loads(
                _handle_qq_send_voice({"text": "hi", "user_id": "1"})
            )
        assert "error" in result
        assert "Speech synthesis failed" in result["error"]
        assert "no tts" in result["error"]

    def test_onebot_failure_surfaced(self, tmp_path):
        audio = tmp_path / "speech.mp3"
        audio.write_bytes(b"synth-audio")
        with (
            patch.object(voice, "_synthesize_speech", return_value=str(audio)),
            patch.object(
                voice, "onebot_call", side_effect=RuntimeError("connection refused")
            ),
        ):
            result = json.loads(
                _handle_qq_send_voice({"text": "hi", "user_id": "1"})
            )
        assert "error" in result
        assert "Could not send the voice message" in result["error"]
        assert "connection refused" in result["error"]


class TestRegistration:
    def test_tool_is_registered(self):
        from tools.registry import registry

        assert registry.get_toolset_for_tool("qq_send_voice") == "qq_voice"

    def test_onebot_env_vars_are_configurable(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        for name in ("ONEBOT_HTTP_URL", "ONEBOT_WS_URL", "ONEBOT_ACCESS_TOKEN"):
            assert name in OPTIONAL_ENV_VARS

    def test_onebot_endpoint_metadata_does_not_require_both_http_and_ws(self):
        from tools.registry import registry

        entry = registry.get_entry("qq_send_voice")
        assert entry is not None
        assert "ONEBOT_HTTP_URL" not in entry.requires_env
        assert "ONEBOT_WS_URL" not in entry.requires_env
        assert entry.check_fn is voice.onebot_configured
