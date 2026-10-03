"""Profile-context regression for the TUI gateway streaming-TTS worker."""

import queue
import threading

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from tui_gateway import methods_voice


def test_streaming_tts_worker_keeps_routed_profile_context(tmp_path, monkeypatch):
    """The per-turn TTS consumer must inherit the profile scope that spawned it."""
    launch_home = tmp_path / "launch"
    served_home = tmp_path / "served"
    launch_home.mkdir()
    served_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch_home))

    # methods_voice is normally rebound onto server.py globals; provide only the small
    # process-global pieces this focused spawn-boundary regression needs.
    monkeypatch.setattr(methods_voice, "queue", queue, raising=False)
    monkeypatch.setattr(methods_voice, "_voice_tts_enabled", lambda: True)
    monkeypatch.setattr(methods_voice, "_tts_stream_stop", lambda *args, **kwargs: None)
    monkeypatch.setattr(methods_voice, "_arm_barge_listener_if_enabled", lambda: None)

    import tools.tts_tool as tts_tool
    import tools.tts_tool_speaker as speaker

    monkeypatch.setattr(tts_tool, "check_tts_requirements", lambda: True)

    seen = {}
    observed = threading.Event()

    def fake_stream(_text_queue, _stop_event, tts_done_event, **_kwargs):
        seen["home"] = str(get_hermes_home())
        tts_done_event.set()
        observed.set()

    monkeypatch.setattr(speaker, "stream_tts_to_speaker", fake_stream)

    token = set_hermes_home_override(served_home)
    try:
        text_queue = methods_voice._tts_stream_begin()
    finally:
        reset_hermes_home_override(token)

    assert text_queue is not None
    assert observed.wait(1.0), "streaming-TTS worker did not run"
    assert seen["home"] == str(served_home), (
        "streaming-TTS worker must keep the routed profile Context instead of "
        "falling back to the launch HERMES_HOME"
    )
