from __future__ import annotations

import json
import wave
from pathlib import Path
from unittest.mock import patch


class _JsonResponse:
    def __init__(self, body: dict, status_code: int = 200):
        self._body = body
        self.status_code = status_code
        self.text = json.dumps(body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _AudioResponse(_JsonResponse):
    def __init__(self, content: bytes):
        super().__init__({})
        self.content = content

    def iter_content(self, chunk_size: int):
        yield self.content

    def close(self):
        pass


def _silent_wav(path: Path) -> str:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 1600)
    return str(path)


def test_dashscope_stt_transcribes_local_audio_with_native_payload(tmp_path):
    from tools.transcription_tools import transcribe_audio

    audio_path = _silent_wav(tmp_path / "speech.wav")
    response = _JsonResponse({
        "output": {
            "choices": [{
                "message": {"content": [{"text": "你好，Hermes"}]},
            }],
        },
    })
    config = {
        "provider": "dashscope",
        "dashscope": {"model": "qwen3-asr-flash", "language": "zh"},
    }

    with patch("tools.transcription_tools._load_stt_config", return_value=config), \
         patch("tools.transcription_tools._resolve_provider_key", return_value="dashscope-key"), \
         patch("requests.post", return_value=response) as post:
        result = transcribe_audio(audio_path)

    assert result == {
        "success": True,
        "transcript": "你好，Hermes",
        "provider": "dashscope",
    }
    url = post.call_args.args[0]
    kwargs = post.call_args.kwargs
    assert url == "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    assert kwargs["headers"]["Authorization"] == "Bearer dashscope-key"
    payload = kwargs["json"]
    assert payload["model"] == "qwen3-asr-flash"
    content = payload["input"]["messages"][0]["content"]
    assert content[0]["audio"].startswith("data:audio/wav;base64,")
    assert payload["parameters"]["asr_options"]["language"] == "zh"


def test_dashscope_stt_missing_key_returns_provider_specific_diagnostic(tmp_path):
    from tools.transcription_tools import transcribe_audio

    audio_path = _silent_wav(tmp_path / "speech.wav")
    config = {"provider": "dashscope", "dashscope": {"model": "qwen3-asr-flash"}}
    with patch("tools.transcription_tools._load_stt_config", return_value=config), \
         patch("tools.transcription_tools._has_dashscope_key", return_value=False):
        result = transcribe_audio(audio_path)

    assert result["success"] is False
    assert result["error"] == "STT provider 'dashscope' configured but DASHSCOPE_API_KEY not set"


def test_dashscope_stt_enforces_encoded_input_limit(tmp_path):
    from tools.transcription_cloud import _DASHSCOPE_ASR_INPUT_LIMIT_BYTES, _transcribe_dashscope

    prefix_size = len("data:audio/wav;base64,")
    largest_raw_input = ((_DASHSCOPE_ASR_INPUT_LIMIT_BYTES - prefix_size) // 4) * 3
    accepted = tmp_path / "accepted.wav"
    accepted.write_bytes(b"x" * largest_raw_input)
    response = _JsonResponse({
        "output": {"choices": [{"message": {"content": [{"text": "accepted"}]}}]},
    })
    config = {"dashscope": {"model": "qwen3-asr-flash", "language": "zh"}}

    with patch("tools.transcription_tools._load_stt_config", return_value=config), \
         patch("tools.transcription_tools._resolve_provider_key", return_value="dashscope-key"), \
         patch("requests.post", return_value=response) as post:
        result = _transcribe_dashscope(str(accepted), "qwen3-asr-flash")

    assert result["success"] is True
    post.assert_called_once()

    rejected = tmp_path / "rejected.wav"
    rejected.write_bytes(b"x" * (largest_raw_input + 1))
    with patch("tools.transcription_tools._load_stt_config", return_value=config), \
         patch("tools.transcription_tools._resolve_provider_key", return_value="dashscope-key"), \
         patch("requests.post") as post:
        result = _transcribe_dashscope(str(rejected), "qwen3-asr-flash")

    assert result["success"] is False
    assert result["error"] == "DashScope Qwen ASR accepts encoded audio input up to 10 MB"
    post.assert_not_called()


def test_dashscope_stt_normalizes_linux_wav_mime_type(tmp_path):
    from tools.transcription_cloud import _dashscope_audio_data_url

    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"RIFF-valid-wav")
    with patch("tools.transcription_cloud.mimetypes.guess_type", return_value=("audio/x-wav", None)):
        data_url = _dashscope_audio_data_url(str(audio_path))

    assert data_url.startswith("data:audio/wav;base64,")


def test_dashscope_tts_downloads_native_audio_result(tmp_path):
    from tools.tts_tool import text_to_speech_tool

    output_path = tmp_path / "speech.wav"
    response = _JsonResponse({
        "output": {
            "audio": {
                "url": "http://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/signed.wav?token=secret",
            },
        },
    })
    config = {
        "provider": "dashscope",
        "dashscope": {
            "model": "qwen3-tts-flash",
            "voice": "Cherry",
            "language_type": "Chinese",
        },
    }

    with patch("tools.tts_tool._load_tts_config", return_value=config), \
         patch("tools.tts_tool._resolve_provider_key", return_value="dashscope-key"), \
         patch("requests.post", return_value=response) as post, \
         patch("requests.get", return_value=_AudioResponse(b"RIFF-valid-wav")) as get:
        result = json.loads(text_to_speech_tool("你好", output_path=str(output_path)))

    assert result["success"] is True
    assert result["provider"] == "dashscope"
    assert output_path.read_bytes() == b"RIFF-valid-wav"
    assert post.call_args.args[0] == (
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    )
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer dashscope-key"
    assert post.call_args.kwargs["json"] == {
        "model": "qwen3-tts-flash",
        "input": {"text": "你好", "voice": "Cherry", "language_type": "Chinese"},
    }
    get.assert_called_once_with(
        "https://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/signed.wav?token=secret",
        timeout=60,
        stream=True,
        allow_redirects=False,
    )


def test_dashscope_tts_rejects_insecure_audio_downloads(tmp_path):
    from tools import tts_tool
    from tools.tts_tool_providers import _dashscope_audio_download_url

    try:
        _dashscope_audio_download_url("http://example.com/signed.wav")
    except RuntimeError as exc:
        assert str(exc) == "DashScope TTS returned an invalid audio URL"
    else:
        raise AssertionError("non-Alibaba HTTP audio URL was accepted")

    response = _JsonResponse({
        "output": {
            "audio": {"url": "https://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/signed.wav"},
        },
    })
    config = {"provider": "dashscope", "dashscope": {}}
    with patch.object(tts_tool, "_load_tts_config", return_value=config), \
         patch.object(tts_tool, "_resolve_provider_key", return_value="dashscope-key"), \
         patch("requests.post", return_value=response), \
         patch("requests.get", return_value=_JsonResponse({}, status_code=302)) as get:
        result = json.loads(tts_tool.text_to_speech_tool("hello", output_path=str(tmp_path / "speech.wav")))

    assert result["success"] is False
    assert "audio download failed (HTTP 302)" in result["error"]
    assert get.call_args.kwargs["allow_redirects"] is False


def test_dashscope_tts_default_path_matches_wav_response(tmp_path):
    from tools import tts_tool

    with patch.object(tts_tool, "_default_output_dir", return_value=str(tmp_path)):
        path, error = tts_tool._resolve_output_base(None, "dashscope", None, False)

    assert error is None
    assert path.suffix == ".wav"


def test_dashscope_voice_is_selectable_on_setup_surfaces():
    from hermes_cli.tools_config import STT_MODEL_CATALOG, TOOL_CATEGORIES
    from hermes_cli.web_server_config import CONFIG_SCHEMA

    stt_rows = TOOL_CATEGORIES["stt"]["providers"]
    tts_rows = TOOL_CATEGORIES["tts"]["providers"]
    assert any(row.get("stt_provider") == "dashscope" for row in stt_rows)
    assert any(row.get("tts_provider") == "dashscope" for row in tts_rows)
    assert "qwen3-asr-flash" in STT_MODEL_CATALOG["dashscope"]
    assert "dashscope" in CONFIG_SCHEMA["stt.provider"]["options"]
    assert "dashscope" in CONFIG_SCHEMA["tts.provider"]["options"]
