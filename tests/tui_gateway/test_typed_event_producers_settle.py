"""The producers that cross the typed event boundary AFTER state already changed must settle.

``server._event_frame`` refuses anything but the registered ``Payload`` class. A producer that still
built a dict would raise there — after ``config.set skin`` had written config.yaml, after the Bot Chat
agent had been rebuilt, or while reporting a failed config-driven model switch — so the caller saw an
error for an effect that had happened. Each test drives the real handler through the real ``_emit`` /
``_broadcast_global_event`` and asserts the event settled and the state is what the reply says.
"""

import pytest
import yaml

from tui_gateway import server
from tui_gateway.contracts.events import ErrorPayload, NoticePayload, SkinPayload


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    server._cfg_cache = server._cfg_sig = server._cfg_path = None
    yield tmp_path / "config.yaml"
    server._cfg_cache = server._cfg_sig = server._cfg_path = None


@pytest.fixture
def broadcasts(monkeypatch):
    """Capture at the wire: the real ``_broadcast_global_event`` builds the frame, we only swallow the write."""
    frames = []
    monkeypatch.setattr(server, "_live_transports", set())
    monkeypatch.setattr(server, "write_json", lambda frame: frames.append(frame))
    return frames


@pytest.fixture
def emitted(monkeypatch):
    frames = []
    monkeypatch.setattr(server, "write_json", lambda frame: frames.append(frame))
    monkeypatch.setattr(server, "_sessions", {})
    return frames


def test_config_set_skin_persists_and_broadcasts_a_typed_payload(config_home, broadcasts, monkeypatch):
    monkeypatch.setattr(server, "resolve_skin", lambda: {"name": "solarized", "colors": {"fg": "#111"}})
    resp = server._methods["config.set"](1, {"key": "skin", "value": "solarized"})

    assert resp.get("error") is None, resp
    assert resp["result"]["key"] == "skin"
    assert yaml.safe_load(config_home.read_text())["display"]["skin"] == "solarized"
    skin = [f for f in broadcasts if f.get("params", {}).get("type") == "skin.changed"]
    assert len(skin) == 1
    assert SkinPayload.model_validate(skin[0]["params"]["payload"]).name == "solarized"


def test_bot_capability_refresh_notice_settles_after_the_rebuild(emitted, monkeypatch):
    sid = "bot-sid"
    agent = type("Agent", (), {"_session_title_hint": "Bot Chat"})()
    session = {"session_key": "k", "agent": agent, "profile_home": None, "bot_caps_seen": "old-fingerprint"}
    server._sessions[sid] = session
    rebuilt = []
    monkeypatch.setattr("tools.bot_mode_probe.capability_fingerprint", lambda home: "new-fingerprint")
    monkeypatch.setattr(server, "_set_session_context", lambda sid, cwd=None: object())
    monkeypatch.setattr(server, "_clear_session_context", lambda tokens: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: "/tmp")
    monkeypatch.setattr(server, "_session_source", lambda session: "tui")
    monkeypatch.setattr(server, "_rebuild_session_agent", lambda *a, **k: rebuilt.append(a) or type("A", (), {})())

    server._sync_bot_capabilities(sid, session)

    assert rebuilt, "the capability change must rebuild the agent"
    notices = [f for f in emitted if f.get("params", {}).get("type") == "notice"]
    assert len(notices) == 1
    assert NoticePayload.model_validate(notices[0]["params"]["payload"]).message.startswith("Capabilities updated")


def test_failed_config_model_switch_reports_through_the_typed_error_event(emitted, monkeypatch):
    sid = "cfg-sid"
    session = {"session_key": "k", "agent": type("Agent", (), {"model": "old", "provider": "p"})()}
    server._sessions[sid] = session
    monkeypatch.setattr(server, "_config_model_target", lambda: ("new-model", "p"))

    def boom(*a, **k):
        raise RuntimeError("provider refused")
    monkeypatch.setattr(server, "_apply_model_switch", boom)

    server._sync_agent_model_with_config(sid, session)  # must not raise

    errors = [f for f in emitted if f.get("params", {}).get("type") == "error"]
    assert len(errors) == 1
    assert "provider refused" in ErrorPayload.model_validate(errors[0]["params"]["payload"]).message
    assert session["config_model_seen"] == ("new-model", "p"), "one attempt per config edit, recorded before the switch"

