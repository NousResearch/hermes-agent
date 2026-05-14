"""Tests for Xiaomi MiMo TTS provider."""

import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _mock_mimo_client(audio_bytes: bytes = b"fake-mp3"):
    message = SimpleNamespace(audio=SimpleNamespace(data=base64.b64encode(audio_bytes).decode("ascii")))
    completion = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    client = MagicMock()
    client.chat.completions.create.return_value = completion
    client.close = MagicMock()
    cls = MagicMock(return_value=client)
    return cls, client


def test_generate_mimo_preset_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("XIAOMI_API_KEY", "test-key")
    mock_cls, mock_client = _mock_mimo_client(b"mp3-data")

    with patch("tools.tts_tool._import_openai_client", return_value=mock_cls):
        from tools.tts_tool import _generate_mimo_tts

        output = _generate_mimo_tts(
            "你好",
            str(tmp_path / "out.mp3"),
            {"mimo": {"base_url": "https://token-plan-sgp.xiaomimimo.com/v1", "voice": "冰糖"}},
        )

    assert output == str(tmp_path / "out.mp3")
    assert (tmp_path / "out.mp3").read_bytes() == b"mp3-data"
    mock_cls.assert_called_once_with(
        api_key="test-key",
        base_url="https://token-plan-sgp.xiaomimimo.com/v1",
        default_headers={"api-key": "test-key"},
    )
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "mimo-v2.5-tts"
    assert kwargs["messages"] == [{"role": "assistant", "content": "你好"}]
    assert kwargs["audio"] == {"format": "mp3", "voice": "冰糖"}
    assert "x-idempotency-key" in kwargs["extra_headers"]
    mock_client.close.assert_called_once()


def test_generate_mimo_voice_design_requires_context(tmp_path, monkeypatch):
    monkeypatch.setenv("XIAOMI_API_KEY", "test-key")
    from tools.tts_tool import _generate_mimo_tts

    try:
        _generate_mimo_tts(
            "你好",
            str(tmp_path / "out.mp3"),
            {"mimo": {"model": "mimo-v2.5-voicedesign"}},
        )
    except ValueError as exc:
        assert "requires tts.mimo.context" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_generate_mimo_voice_clone_encodes_sample(tmp_path, monkeypatch):
    monkeypatch.setenv("XIAOMI_API_KEY", "test-key")
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"wav-data")
    mock_cls, mock_client = _mock_mimo_client(b"mp3-data")

    with patch("tools.tts_tool._import_openai_client", return_value=mock_cls):
        from tools.tts_tool import _generate_mimo_tts

        _generate_mimo_tts(
            "你好",
            str(tmp_path / "out.mp3"),
            {"mimo": {"model": "mimo-v2.5-voiceclone", "voice_file": str(sample)}},
        )

    audio = mock_client.chat.completions.create.call_args.kwargs["audio"]
    assert audio["voice"].startswith("data:audio/wav;base64,")
    assert audio["voice"].split(",", 1)[1] == base64.b64encode(b"wav-data").decode("ascii")


def test_text_to_speech_mimo_marks_telegram_voice_compatible(tmp_path, monkeypatch):
    monkeypatch.setenv("XIAOMI_API_KEY", "test-key")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    output = tmp_path / "speech.ogg"

    def fake_generate(text, output_path, tts_config):
        assert output_path.endswith(".mp3")
        mp3_path = tmp_path / "speech.mp3"
        mp3_path.write_bytes(b"mp3-data")
        return str(mp3_path)

    with patch("tools.tts_tool._load_tts_config", return_value={"provider": "mimo"}), \
         patch("tools.tts_tool._import_openai_client"), \
         patch("tools.tts_tool._generate_mimo_tts", side_effect=fake_generate), \
         patch("tools.tts_tool._convert_to_opus", return_value=str(output)):
        output.write_bytes(b"ogg-data")
        from tools.tts_tool import text_to_speech_tool

        result = json.loads(text_to_speech_tool("你好", str(tmp_path / "speech.mp3")))

    assert result["success"] is True
    assert result["file_path"] == str(output)
    assert result["voice_compatible"] is True
    assert result["media_tag"] == f"[[audio_as_voice]]\nMEDIA:{output}"
