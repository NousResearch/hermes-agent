"""GPT-Live standby: the session config carries wake_policy/idle_timeout_s only when enabled."""

from tools import voice_live


def test_standby_off_by_default():
    assert voice_live.standby_settings({}) == {}
    assert voice_live.standby_settings({"standby_idle_seconds": 0}) == {}
    assert voice_live.standby_settings({"standby_idle_seconds": "nope"}) == {}


def test_standby_enabled_and_clamped():
    assert voice_live.standby_settings({"standby_idle_seconds": 30}) == {
        "wake_policy": {"type": "speech"}, "idle_timeout_s": 30}
    assert voice_live.standby_settings({"standby_idle_seconds": 2})["idle_timeout_s"] == 5
    assert voice_live.standby_settings({"standby_idle_seconds": 600})["idle_timeout_s"] == 60


def test_standby_vad_threshold():
    s = voice_live.standby_settings({"standby_idle_seconds": 20, "standby_vad_threshold": 0.7})
    assert s["wake_policy"] == {"type": "speech", "vad_threshold": 0.7}
    s = voice_live.standby_settings({"standby_idle_seconds": 20, "standby_vad_threshold": 4})
    assert s["wake_policy"]["vad_threshold"] == 1.0


def test_build_session_config_includes_standby(monkeypatch):
    monkeypatch.setattr(voice_live, "_live_section", lambda *a: {"model": "gpt-live-1.1", "standby_idle_seconds": 45})
    cfg = voice_live.build_session_config()
    assert cfg["model"] == "gpt-live-1.1"
    assert cfg["wake_policy"] == {"type": "speech"} and cfg["idle_timeout_s"] == 45
    monkeypatch.setattr(voice_live, "_live_section", lambda *a: {})
    assert "wake_policy" not in voice_live.build_session_config()
