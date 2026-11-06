"""Regression coverage for the final-file ``post_tts_file`` observer."""

from __future__ import annotations

import json
import threading
import time


def _wait_for(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def test_post_tts_file_observes_final_delivery_file_without_blocking(monkeypatch, tmp_path):
    """The observer sees the final Opus path, but a slow callback cannot hold up TTS."""
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher
    from tools import tts_tool

    source = tmp_path / "source.mp3"
    delivered = tmp_path / "delivered.ogg"
    source.write_bytes(b"source")
    delivered.write_bytes(b"delivered-audio")
    seen = []
    started = threading.Event()

    def slow_observer(**kwargs):
        started.set()
        time.sleep(0.2)
        seen.append(kwargs)

    shutdown_plugin_stream_hook_dispatcher()
    monkeypatch.setattr(
        "hermes_cli.plugins.iter_hook_callbacks",
        lambda name: (slow_observer,) if name == "post_tts_file" else (),
    )
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge", "edge": {"voice": "Ava"}})
    monkeypatch.setattr(tts_tool, "_split_text_for_tts", lambda text, max_len: [text])
    monkeypatch.setattr(
        tts_tool,
        "_synthesize_chunks",
        lambda *args, **kwargs: ([str(source)], [{"provider": "edge", "voice_compatible": True, "reencoded": True}]),
    )
    monkeypatch.setattr(
        tts_tool,
        "_build_audio_delivery_files",
        lambda *args, **kwargs: ([str(delivered)], False),
    )
    from tools.approval_context import (
        reset_current_observability_context,
        set_current_observability_context,
    )

    tokens = set_current_observability_context(session_id="session-1", tool_call_id="call-1")
    try:
        started_at = time.monotonic()
        result = json.loads(tts_tool.text_to_speech_tool("Hello", output_path=str(tmp_path / "request.mp3")))
        elapsed = time.monotonic() - started_at
        assert result["file_path"] == str(delivered)
        assert elapsed < 0.05
        assert started.wait(timeout=1.0)
        _wait_for(lambda: bool(seen))
    finally:
        reset_current_observability_context(tokens)
        shutdown_plugin_stream_hook_dispatcher()

    [payload] = seen
    assert payload["path"] == str(delivered.resolve())
    assert payload["byte_length"] == len(b"delivered-audio")
    assert payload["provider"] == "edge"
    assert payload["model"] is None
    assert payload["voice_id"] == "Ava"
    assert payload["tool_name"] == "text_to_speech"
    assert payload["tool_call_id"] == "call-1"
    assert payload["session_id"] == "session-1"
    assert payload["reencoded"] is True


def test_post_tts_file_is_a_valid_plugin_hook():
    from hermes_cli.plugins import VALID_HOOKS

    assert "post_tts_file" in VALID_HOOKS
