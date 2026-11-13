# Profile context across fallback and speaker thread hops.

import threading

import pytest

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def profile_b(tmp_path, monkeypatch):
    from tui_gateway import server
    monkeypatch.setenv("HERMES_VOICE_TTS", "1")
    monkeypatch.setattr(server, "_arm_barge_listener_if_enabled", lambda: None)
    home = tmp_path / "profiles" / "b"
    home.mkdir(parents=True)
    token = set_hermes_home_override(str(home))
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


def _speaker(monkeypatch):
    """Replace the real speaker with one that records the home it runs under."""
    from hermes_cli import voice
    seen, done = {}, threading.Event()

    def speak_text(text, stop_event=None):
        seen["home"] = get_hermes_home()
        done.set()
    monkeypatch.setattr(voice, "speak_text", speak_text)
    return seen, done


def test_the_voice_fallback_speaks_as_the_turn_profile(profile_b, monkeypatch):
    from types import SimpleNamespace
    from tui_gateway import server
    seen, done = _speaker(monkeypatch)
    server._after_complete_turn("sid", {}, SimpleNamespace(tts_queue=None), "the reply")
    assert done.wait(10)
    assert seen["home"] == profile_b


def test_speaker_keeps_caller_profile(profile_b, monkeypatch):
    from tui_gateway import server
    seen, done = _speaker(monkeypatch)
    server._speak_text_with_barge("the reply")
    assert done.wait(10)
    assert seen["home"] == profile_b


def test_voice_tts_rpc_speaks_as_the_requested_profile(tmp_path, monkeypatch):
    """The registered ``voice.tts`` handler binds ``params.profile`` and both thread hops keep it;
    without a profile it stays on the launch home."""
    from tui_gateway import server
    launch, worker = tmp_path / "launch", tmp_path / "profiles" / "b"
    launch.mkdir()
    worker.mkdir(parents=True)
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_profile_home", lambda name: worker if name == "b" else None)
    monkeypatch.setattr(server, "_arm_barge_listener_if_enabled", lambda: None)
    for params, expected in (({"text": "hi", "profile": "b"}, worker), ({"text": "hi"}, launch)):
        seen, done = _speaker(monkeypatch)
        resp = server._methods["voice.tts"]("rid", params)
        assert resp["result"]["status"] == "speaking"
        assert done.wait(10)
        assert seen["home"] == expected
