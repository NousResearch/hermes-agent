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


@pytest.fixture
def two_profiles(tmp_path, monkeypatch):
    """A profile-B session (own home and ``.env`` secret) in a process launched on another home."""
    from tui_gateway import server
    home = tmp_path / "profiles" / "b"
    home.mkdir(parents=True)
    (home / ".env").write_text("VOICE_PROFILE_PROBE=b\n")
    monkeypatch.setenv("VOICE_PROFILE_PROBE", "launch")
    monkeypatch.setenv("HERMES_VOICE", "1")
    monkeypatch.setenv("HERMES_VOICE_TTS", "1")
    monkeypatch.setattr(server, "_arm_barge_listener_if_enabled", lambda: None)
    return home, {"profile_home": str(home)}


def _recorder():
    from agent.secret_scope import get_secret
    seen, done = [], threading.Event()

    def record(*_a, **_k):
        seen.append((get_hermes_home(), get_secret("VOICE_PROFILE_PROBE")))
        done.set()
    return seen, done, record


@pytest.mark.parametrize("streaming", [False, True])
def test_turn_tts_synthesizes_as_the_turn_profile(two_profiles, monkeypatch, streaming):
    """A turn's streaming TTS consumer and the hop that synthesizes (the sync executor, or the
    prefetch thread that runs a streaming provider's generator) keep the turn's home and secrets."""
    from tools import tts_streaming
    from tools import tts_tool_speaker as speaker
    from tools import tts_tool, voice_mode
    from tui_gateway import server
    home, session = two_profiles
    seen, done, record = _recorder()
    monkeypatch.setattr(tts_tool, "check_tts_requirements", lambda: True)
    monkeypatch.setattr(voice_mode, "play_audio_file", lambda *_a, **_k: None)
    monkeypatch.setattr(speaker, "_play_via_tempfile", lambda *_a, **_k: None)
    monkeypatch.setattr(speaker._StreamerPlayback, "_device_usable", lambda self: False)

    class Streamer:
        sample_rate, channels = 24000, 1

        def stream(self, text):
            record()  # a provider resolves its key on the first next(), not at stream()
            yield from ()

    def synthesize(text, output_path=None, **_k):
        record()
        return output_path
    monkeypatch.setattr(tts_streaming, "resolve_streaming_provider",
                        lambda *_a, **_k: Streamer() if streaming else None)
    monkeypatch.setattr(tts_tool, "text_to_speech_tool", synthesize)
    with server._session_profile_runtime_scope(session):
        text_queue = server._tts_stream_begin()
    finished = server._tts_stream_state["done"]
    try:
        text_queue.put("Hello from profile b. ")
        text_queue.put(None)
        assert done.wait(10)
    finally:
        server._tts_stream_stop(user_barge=False)
        assert finished.wait(10), "the TTS consumer outlived the test"
    assert seen[0] == (home, "b")


@pytest.mark.parametrize("tts_on", [False, True])
def test_voice_toggle_leases_tts_as_the_requested_profile(two_profiles, monkeypatch, tts_on):
    """``voice.toggle {action: tts, profile: b}`` acquires (off -> on) or releases (on -> off) the
    engine lease under B's scope."""
    from tools import tts_tool_lifecycle as lifecycle
    from tui_gateway import server
    home, _session = two_profiles
    seen, done, record = _recorder()
    monkeypatch.setenv("HERMES_VOICE_TTS", "1" if tts_on else "0")
    monkeypatch.setattr(lifecycle, "acquire_tts_lease" if not tts_on else "release_tts_lease", record)
    monkeypatch.setattr(server, "_profile_home", lambda name: home if name == "b" else None)
    resp = server.handle_request({"id": "1", "method": "voice.toggle",
                                  "params": {"action": "tts", "profile": "b"}})
    assert resp.get("result", {}).get("tts") is (not tts_on), resp
    assert done.wait(10)
    for worker in [t for t in threading.enumerate() if t.name == "tts-lease-tui:voice-tts"]:
        worker.join(10)
    assert seen == [(home, "b")]
