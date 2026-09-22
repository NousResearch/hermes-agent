"""``request_timeout_s`` plumbing: the gateway's auto-TTS budget must reach the HTTP client.

The OpenAI SDK's own defaults (``timeout=600`` with ``max_retries=2``) let one slow synthesis hold
the platform's TEXT reply for up to 1800 s, so the gateway passes an explicit budget for auto-TTS.
Model-called synthesis leaves the argument unset and keeps the SDK defaults (a user-requested voice
note may legitimately take minutes on a slow local backend).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


class _FakeSpeech:
    def __init__(self, path):
        self._path = path

    def stream_to_file(self, output_path):
        Path(output_path).write_bytes(b"audio")


def _install_fake_openai(captured):
    class FakeOpenAI:
        def __init__(self, api_key, base_url, **kwargs):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["client_kwargs"] = kwargs
            self.audio = types.SimpleNamespace(
                speech=types.SimpleNamespace(create=lambda **kw: _FakeSpeech(captured["path"])))

        def close(self):
            pass

    return FakeOpenAI


@pytest.fixture
def tts_openai(monkeypatch):
    """``tools.tts_tool_openai`` with the SDK import seam replaced by a recording fake."""
    import tools.tts_tool_openai as mod

    captured: dict = {}
    fake_client = _install_fake_openai(captured)
    monkeypatch.setattr(mod, "_origin", lambda: types.SimpleNamespace(
        _import_openai_client=lambda: fake_client))
    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return mod, captured


def test_timeout_s_bounds_the_client_and_drops_retries(tts_openai, tmp_path):
    mod, captured = tts_openai
    captured["path"] = str(tmp_path / "speech.mp3")

    mod._generate_openai_tts("hello", captured["path"], {"openai": {"model": "tts-1"}},
                             api_key="k", base_url="http://127.0.0.1:8880/v1", timeout_s=12.5)

    assert captured["client_kwargs"]["timeout"] == 12.5
    # Without this the SDK would retry the same slow request twice more (3 x budget).
    assert captured["client_kwargs"]["max_retries"] == 0


def test_absent_timeout_keeps_the_sdk_defaults(tts_openai, tmp_path):
    mod, captured = tts_openai
    captured["path"] = str(tmp_path / "speech.mp3")

    mod._generate_openai_tts("hello", captured["path"], {"openai": {"model": "tts-1"}},
                             api_key="k", base_url="http://127.0.0.1:8880/v1")

    assert "timeout" not in captured["client_kwargs"]
    assert "max_retries" not in captured["client_kwargs"]


def test_tool_forwards_request_timeout_to_the_openai_generator(monkeypatch, tmp_path):
    """``text_to_speech_tool(request_timeout_s=...)`` must reach the provider request."""
    from tools import tts_tool

    seen: dict = {}

    def fake_generate(text, output_path, tts_config, *, instructions=None, timeout_s=None):
        seen["timeout_s"] = timeout_s
        Path(output_path).write_bytes(b"audio")
        return output_path

    monkeypatch.setattr(tts_tool, "_generate_openai_tts", fake_generate)
    monkeypatch.setattr(tts_tool, "_select_builtin_engine", lambda _p: ("openai", None))

    out = json.loads(tts_tool.text_to_speech_tool(
        text="hello there", output_path=str(tmp_path / "out.mp3"), provider="openai",
        request_timeout_s=7))
    assert out["success"] is True
    assert seen["timeout_s"] == 7

    seen.clear()
    out = json.loads(tts_tool.text_to_speech_tool(
        text="hello there", output_path=str(tmp_path / "out2.mp3"), provider="openai"))
    assert out["success"] is True
    assert seen["timeout_s"] is None


def test_text_to_speech_tool_is_not_exposed_with_the_internal_kwarg():
    """The internal budget must stay out of the model-facing tool schema."""
    from tools.tts_tool import TTS_SCHEMA

    assert "request_timeout_s" not in TTS_SCHEMA["parameters"]["properties"]
    assert sys.modules["tools.tts_tool"] is not None
