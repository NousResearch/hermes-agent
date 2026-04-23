import json
from pathlib import Path
from unittest.mock import patch


def _fake_openai_generation(_text: str, output_path: str, _tts_config):
    Path(output_path).write_bytes(b"OggS" + b"\x00" * 32)
    return output_path


def test_text_to_speech_defaults_to_ogg_for_feishu_openai(monkeypatch, tmp_path):
    from tools.tts_tool import text_to_speech_tool

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "feishu")
    monkeypatch.setattr("tools.tts_tool.DEFAULT_OUTPUT_DIR", str(tmp_path))

    with patch("tools.tts_tool._load_tts_config", return_value={"provider": "openai"}), \
         patch("tools.tts_tool._import_openai_client", return_value=object()), \
         patch("tools.tts_tool._generate_openai_tts", side_effect=_fake_openai_generation):
        result = json.loads(text_to_speech_tool("Hello from Feishu"))

    assert result["success"] is True
    assert result["provider"] == "openai"
    assert result["file_path"].startswith(str(tmp_path))
    assert result["file_path"].endswith(".ogg")
    assert result["voice_compatible"] is True
    assert result["media_tag"] == f"[[audio_as_voice]]\nMEDIA:{result['file_path']}"


def test_text_to_speech_preserves_explicit_output_path_for_feishu(monkeypatch, tmp_path):
    from tools.tts_tool import text_to_speech_tool

    output_path = tmp_path / "explicit.mp3"
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "feishu")
    monkeypatch.setattr("tools.tts_tool.DEFAULT_OUTPUT_DIR", str(tmp_path))

    with patch("tools.tts_tool._load_tts_config", return_value={"provider": "openai"}), \
         patch("tools.tts_tool._import_openai_client", return_value=object()), \
         patch("tools.tts_tool._generate_openai_tts", side_effect=_fake_openai_generation):
        result = json.loads(
            text_to_speech_tool("Hello from Feishu", output_path=str(output_path))
        )

    assert result["success"] is True
    assert result["file_path"] == str(output_path)
    assert output_path.exists()
    assert result["voice_compatible"] is False
    assert result["media_tag"] == f"MEDIA:{output_path}"
