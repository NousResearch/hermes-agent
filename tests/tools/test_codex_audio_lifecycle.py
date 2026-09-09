"""Owned network resources and one budget across subscription speech retries."""

import json
import time
from threading import Event
from types import SimpleNamespace

import pytest
import requests

from tools import codex_web_audio, tts_tool_codex


@pytest.mark.parametrize(
    "outcome",
    ["success", "redirect", "http_error", "read_error", "limit", "timeout", "cancel", "retry"],
)
def test_synthesis_owns_resources_and_budget_through_terminal_outcomes(monkeypatch, outcome):
    clock = [0.0]
    cancel = Event()
    sessions = []
    responses = []
    calls = []
    conversation_count = 0
    requested = "Speak exactly this."
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    class Response:
        def __init__(self, path, body, content_type="application/json"):
            self.path = path
            self.body = body
            self.headers = {"content-type": content_type}
            self.status_code = 200
            self.closed = False
            responses.append(self)
            if path == codex_web_audio._SENTINEL_PREPARE:
                if outcome == "redirect":
                    self.status_code = 302
                elif outcome == "http_error":
                    self.status_code = 401
                elif outcome == "limit":
                    self.headers["content-length"] = str(codex_web_audio._MAX_JSON_BYTES + 1)

        def iter_content(self, _chunk_size):
            if self.path == codex_web_audio._CONVERSATION:
                if outcome == "read_error":
                    raise requests.ConnectionError("stream disconnected")
                if outcome == "timeout":
                    clock[0] += 30.0
                if outcome == "cancel":
                    cancel.set()
            yield self.body

        def close(self):
            self.closed = True

    class Session:
        def __init__(self):
            self.closed = False
            sessions.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed = True

        def get(self, url, **kwargs):
            assert kwargs["stream"] is True
            assert kwargs["allow_redirects"] is False
            calls.append(("warmup", kwargs["timeout"], clock[0]))
            clock[0] += 1
            return Response(url, b"")

        def request(self, _method, url, **kwargs):
            nonlocal conversation_count
            path = url.removeprefix(codex_web_audio._BASE_URL)
            calls.append((path, kwargs["timeout"], clock[0]))
            clock[0] += 1
            if path == codex_web_audio._CONVERSATION:
                conversation_count += 1
                spoken = "Changed text." if outcome == "retry" and conversation_count == 1 else requested
                event = {"v": {"conversation_id": "conversation", "message": {
                    "id": "assistant", "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": [spoken]},
                }}}
                return Response(path, ("data: " + json.dumps(event)).encode(), "text/event-stream")
            if path == codex_web_audio._SYNTHESIZE:
                return Response(path, b"ID3speech", "audio/mpeg")
            body = {
                codex_web_audio._CONVERSATION_PREPARE: {"conduit_token": ""},
                codex_web_audio._SENTINEL_PREPARE: {"prepare_token": "prepared"},
                codex_web_audio._SENTINEL_FINALIZE: {"token": "requirements"},
            }[path]
            return Response(path, json.dumps(body).encode())

    monkeypatch.setattr(codex_web_audio.requests, "Session", Session)
    error = {
        "redirect": RuntimeError, "http_error": RuntimeError,
        "read_error": requests.ConnectionError, "limit": ValueError,
        "timeout": TimeoutError, "cancel": InterruptedError,
    }.get(outcome)
    if error is None:
        speech = codex_web_audio.synthesize_codex_speech(
            requested, "token", timeout=30, cancel_event=cancel
        )
        assert speech.audio == b"ID3speech"
        assert speech.spoken_text == requested
        assert conversation_count == (2 if outcome == "retry" else 1)
    else:
        with pytest.raises(error):
            codex_web_audio.synthesize_codex_speech(
                requested, "token", timeout=30, cancel_event=cancel
            )
        assert not any(path == codex_web_audio._SYNTHESIZE for path, *_ in calls)

    assert len(sessions) == 1
    assert sessions[0].closed
    assert responses and all(response.closed for response in responses)
    assert len([path for path, *_ in calls if path == "warmup"]) == 3
    for path, timeout, started in calls:
        if path != "warmup":
            assert timeout == pytest.approx(30 - started)


@pytest.mark.parametrize("invalid_suffix", [False, True])
def test_file_synthesis_validates_before_network_and_preserves_refresh_budget(
    monkeypatch, tmp_path, invalid_suffix
):
    clock = [0.0]
    calls = []
    credential_reads = []
    refreshes = []
    selected = {"api_key": "stale", "credential_id": "account"}
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    def refresh(**kwargs):
        refreshes.append(kwargs)
        return SimpleNamespace(runtime_api_key="fresh")

    pool = SimpleNamespace(try_refresh_matching=refresh)

    def credentials():
        credential_reads.append(True)
        return pool, selected

    def synthesize(_text, token, **kwargs):
        calls.append((token, kwargs["timeout"]))
        if token == "stale":
            clock[0] += 8
            raise RuntimeError("ChatGPT failed (HTTP 401)")
        return SimpleNamespace(audio=b"ID3speech")

    monkeypatch.setattr(tts_tool_codex, "_codex_tts_credentials", credentials)
    monkeypatch.setattr(tts_tool_codex, "synthesize_codex_speech", synthesize)
    output = tmp_path / ("speech.wav" if invalid_suffix else "speech.mp3")
    config = {"openai_codex": {"timeout": 30}}
    if invalid_suffix:
        with pytest.raises(ValueError, match="output must be"):
            tts_tool_codex._generate_openai_codex_tts("Speech.", str(output), config)
        assert not credential_reads
        assert not calls
        assert not output.exists()
    else:
        tts_tool_codex._generate_openai_codex_tts("Speech.", str(output), config)
        assert calls == [("stale", 30), ("fresh", 22)]
        assert output.read_bytes() == b"ID3speech"
        assert selected["api_key"] == "fresh"
        tts_tool_codex.synthesize_codex_speech_with_credentials(
            "Next sentence.", pool, selected, timeout=30
        )
        assert calls[-1] == ("fresh", 30)
        assert refreshes == [{"api_key_hint": "stale", "credential_id": "account"}]
