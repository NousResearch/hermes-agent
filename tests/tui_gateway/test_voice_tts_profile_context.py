"""Profile-context regression for the TUI gateway streaming-TTS worker."""

import queue
import threading

import pytest

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
    assert observed.wait(5.0), "streaming-TTS worker did not run"
    assert seen["home"] == str(served_home), (
        "streaming-TTS worker must keep the routed profile Context instead of "
        "falling back to the launch HERMES_HOME"
    )


@pytest.mark.parametrize("active", [True, False])
def test_tts_lease_worker_keeps_routed_profile_context(tmp_path, monkeypatch, active):
    """Both async lease operations must use the calling profile's home and secrets."""
    from agent.secret_scope import get_secret, reset_secret_scope, set_secret_scope
    from tools import tts_tool_lifecycle as lifecycle

    launch_home = tmp_path / "launch"
    served_home = tmp_path / "served"
    launch_home.mkdir()
    served_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setenv("VOICE_TTS_PROFILE_TEST_TOKEN", "launch-only")

    seen = []
    completed = threading.Event()

    def record_lease(lease):
        seen.append((lease, get_hermes_home(), get_secret("VOICE_TTS_PROFILE_TEST_TOKEN")))
        completed.set()
        return {"leases": 1}

    monkeypatch.setattr(lifecycle, "acquire_tts_lease", record_lease)
    monkeypatch.setattr(lifecycle, "release_tts_lease", record_lease)

    home_token = set_hermes_home_override(served_home)
    secret_token = set_secret_scope(
        {"VOICE_TTS_PROFILE_TEST_TOKEN": "served-only"}, profile_home=str(served_home)
    )
    try:
        methods_voice._tts_lease_async("tui:voice-tts", active)
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)

    assert completed.wait(5.0), "TTS lease worker did not reach the lifecycle boundary"
    assert seen == [("tui:voice-tts", served_home, "served-only")]
