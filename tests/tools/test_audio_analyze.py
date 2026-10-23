"""Real registry, config, resolver and SDK dispatch with only HTTP replaced by a test transport."""

import base64
import json
from types import SimpleNamespace

import httpx
import pytest

from tools.audio_tools import check_audio_requirements
from tools.registry import registry
from tests.tools.test_audio_prepare import stereo_wav


@pytest.fixture
def audio_route(tmp_path, monkeypatch):
    from hermes_cli.config import save_config
    from agent import auxiliary_client
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    cfg = {"auxiliary": {"audio": {"provider": "custom", "model": "audio-model",
                                   "base_url": "https://audio.example/v1", "api_key": "test-key"}}}
    save_config(cfg)
    requests = []
    response = {"status_code": 200, "id": "test", "object": "chat.completion", "created": 1, "model": "audio-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Piano and rain."},
                             "finish_reason": "stop"}]}

    def handle(request):
        requests.append(request)
        return httpx.Response(response["status_code"], json=response)

    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(auxiliary_client, "_openai_http_client_kwargs", lambda url, async_mode=False: {
        "http_client": (httpx.AsyncClient if async_mode else httpx.Client)(transport=transport)})
    return cfg, requests, response


def analyze(source):
    return json.loads(registry.dispatch("audio_analyze", {"audio_url": str(source), "question": "What do you hear?"},
                                        task_id="audio-test"))


def test_registry_sends_raw_audio_and_returns_only_text(audio_route, tmp_path):
    _, requests, _ = audio_route
    path = tmp_path / "recording.wav"
    path.write_bytes(stereo_wav())
    assert check_audio_requirements()
    result = analyze(path)
    assert result["analysis"] == "Piano and rain."
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://audio.example/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    payload = json.loads(request.content)
    assert payload["model"] == "audio-model"
    assert payload["modalities"] == ["text"]
    audio = payload["messages"][0]["content"][1]["input_audio"]
    assert base64.b64decode(audio["data"]) == path.read_bytes()
    assert audio["format"] == "wav"
    assert audio["data"] not in json.dumps(result)


def test_empty_response_does_not_claim_analysis(audio_route, tmp_path):
    _, _, response = audio_route
    response["choices"][0]["message"]["content"] = None
    path = tmp_path / "recording.wav"
    path.write_bytes(stereo_wav())
    assert "no text analysis" in analyze(path)["error"]


def test_audio_tool_respects_source_boundaries(audio_route, tmp_path, monkeypatch):
    _, requests, _ = audio_route
    path = tmp_path / "outside.wav"
    path.write_bytes(stereo_wav())
    assert "error" in analyze("http://127.0.0.1/private.wav")
    secret = tmp_path / ".env"
    secret.write_bytes(stereo_wav())
    assert "error" in analyze(secret)
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    from tools import image_source
    monkeypatch.setattr(image_source, "_get_active_env", lambda task_id: None)
    assert "error" in analyze(path)
    # The same path can be read from the active sandbox, with the caller's task identity.
    sandbox_data = stereo_wav() + b"sandbox"
    seen = []

    def active_env(task_id):
        seen.append(task_id)
        return SimpleNamespace(execute=lambda cmd: {"returncode": 0, "output": base64.b64encode(sandbox_data).decode()})

    monkeypatch.setattr(image_source, "_get_active_env", active_env)
    assert analyze(path)["analysis"] == "Piano and rain."
    assert seen == ["audio-test"]
    assert len(requests) == 1
    audio = json.loads(requests[0].content)["messages"][0]["content"][1]["input_audio"]
    assert base64.b64decode(audio["data"]) == sandbox_data


def test_no_implicit_model_or_fallback(audio_route, tmp_path):
    from hermes_cli.config import save_config
    cfg, requests, _ = audio_route
    path = tmp_path / "recording.wav"
    path.write_bytes(stereo_wav())
    cfg["auxiliary"]["audio"]["provider"] = "auto"
    save_config(cfg)
    assert not check_audio_requirements()
    assert "explicit" in analyze(path)["error"]
    assert not requests


def test_provider_failure_does_not_fallback_or_echo_audio(audio_route, tmp_path):
    _, requests, response = audio_route
    response.update(status_code=429, error={"message": "quota exhausted", "type": "rate_limit_error"})
    path = tmp_path / "recording.wav"
    path.write_bytes(stereo_wav())
    result = analyze(path)
    assert "RateLimitError" in result["error"]
    assert len(requests) == 1
    assert base64.b64encode(path.read_bytes()).decode() not in json.dumps(result)


@pytest.mark.parametrize("container,mime", [("wav", "audio/wav"), ("m4a", "audio/m4a")])
def test_gemini_native_route_receives_inline_audio(audio_route, tmp_path, monkeypatch, container, mime):
    from hermes_cli.config import save_config
    cfg, _, _ = audio_route
    cfg["auxiliary"]["audio"] = {"provider": "gemini", "model": "gemini-audio-model", "api_key": "test-gemini-key"}
    save_config(cfg)
    captured = []

    def post(client, url, *, json, headers, timeout):
        captured.append((url, json, headers))
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "Strings and wind."}]}, "finishReason": "STOP"}],
        }, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", post)
    path = tmp_path / f"recording.{container}"
    # The request test checks byte preservation; real M4A decoding is tested with ffmpeg separately.
    path.write_bytes(stereo_wav() if container == "wav" else b"\x00\x00\x00\x18ftypM4A " + b"fixture")
    assert analyze(path)["analysis"] == "Strings and wind."
    assert len(captured) == 1
    url, payload, headers = captured[0]
    assert "generativelanguage.googleapis.com" in str(url)
    assert headers["x-goog-api-key"] == "test-gemini-key"
    parts = payload["contents"][0]["parts"]
    assert base64.b64decode(parts[1]["inlineData"]["data"]) == path.read_bytes()
    assert parts[1]["inlineData"]["mimeType"] == mime
