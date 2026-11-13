"""The standalone backend serves the Desktop Daybreak switch (`agent.daybreak`) like the native owner.

Desktop shows the switch on catalog-eligible models from any backend, including `hermes serve` over
SSH, so `config.get` / `config.set` must know the key here too, and a flip must reach this profile's
live agents (requests check eligibility themselves: hermes_agent/agent/codex_service_tiers.py).
"""

from __future__ import annotations

from types import SimpleNamespace

import hermes_agent.hermes_yaml as yaml
import pytest

from hermes_agent.tui_gateway import server


@pytest.fixture
def homes(tmp_path, monkeypatch):
    launch, worker = tmp_path / "launch", tmp_path / "profiles" / "code"
    launch.mkdir(parents=True)
    worker.mkdir(parents=True)
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_profile_home", lambda name: worker if (name or "").strip() == "code" else None)
    server._cfg_cache = server._cfg_sig = server._cfg_path = None
    yield launch, worker
    server._cfg_cache = server._cfg_sig = server._cfg_path = None


def _call(method, **params):
    return server._methods[method](1, params)


def test_the_switch_round_trips_through_agent_daybreak(homes):
    launch, _worker = homes
    assert _call("config.get", key="daybreak")["result"] == {"value": False}  # default off

    assert _call("config.set", key="daybreak", value=True)["result"] == {"key": "daybreak", "value": True}
    assert yaml.safe_load((launch / "config.yaml").read_text())["agent"]["daybreak"] is True
    assert _call("config.get", key="daybreak")["result"] == {"value": True}

    assert _call("config.set", key="daybreak", value="false")["result"] == {"key": "daybreak", "value": False}
    assert _call("config.get", key="daybreak")["result"] == {"value": False}


def test_a_non_boolean_is_refused_rather_than_written(homes):
    launch, _worker = homes
    assert _call("config.set", key="daybreak", value="sometimes")["error"]["code"] == 4002
    assert not (launch / "config.yaml").exists()


class _StopAfterDaybreak(Exception):
    pass


def _admit_turn(monkeypatch, session):
    """Run real turn admission (``_prepare_turn_input``, under the session's real profile scope) and stop just
    after the turn's agent is selected and its Daybreak setting synced."""
    from hermes_agent.tools import approval_context
    monkeypatch.setattr(server, "_set_session_context", lambda *a, **k: [])
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    for name in ("_apply_pending_model_switch", "_sync_agent_model_with_config", "_sync_agent_compression_with_config",
                 "_sync_agent_fallback_with_config", "_sync_bot_capabilities"):
        monkeypatch.setattr(server, name, lambda sid, session: None)
    monkeypatch.setattr(server, "_adopt_out_of_band_turns", lambda session: None)

    def stop(*args):
        raise _StopAfterDaybreak()
    monkeypatch.setattr(server, "_stage_first_contact_onboarding_note", stop)
    st = server._TurnRun(agent=None, one_turn_restore=None, terminal_callback=None, receipt_committed=False)
    try:
        with pytest.raises(_StopAfterDaybreak):
            server._prepare_turn_input("sid", session, st, "hello", [])
    finally:
        server._release_turn_scopes("sid", session, st)
    assert approval_context.get_current_session_key(default="") == ""


@pytest.mark.parametrize(("platform", "expected"), [
    ("desktop", True), ("tui", True), ("cli", True), ("telegram", False), ("acp", False), (None, False)])
def test_every_turn_reads_the_switch_for_the_owners_own_surfaces(homes, monkeypatch, platform, expected):
    """Each turn reads ``agent.daybreak`` from the session's profile once the final agent is selected, so a
    flip, a build that raced one, or a compute-host child (same turn path) all send the saved value. A plugin
    session tagged ``telegram`` (or an editor client) built by this backend gets standard requests."""
    import threading
    _launch, worker = homes
    agent = SimpleNamespace(daybreak=not expected, platform=platform)
    session = {"agent": agent, "session_key": "daybreak-key", "profile_home": str(worker),
               "history": [], "history_lock": threading.Lock()}
    config = worker / "config.yaml"

    config.write_text("agent:\n  daybreak: true\n", encoding="utf-8")
    _admit_turn(monkeypatch, session)
    assert agent.daybreak is expected

    config.write_text("agent:\n  daybreak: false\n", encoding="utf-8")  # flipped off between turns
    _admit_turn(monkeypatch, session)
    assert agent.daybreak is False


def test_the_turn_reads_its_own_profile_not_the_launch_one(homes, monkeypatch):
    import threading
    launch, worker = homes
    (launch / "config.yaml").write_text("agent:\n  daybreak: true\n", encoding="utf-8")
    (worker / "config.yaml").write_text("agent:\n  daybreak: false\n", encoding="utf-8")
    agent = SimpleNamespace(daybreak=True, platform="desktop")
    _admit_turn(monkeypatch, {"agent": agent, "session_key": "worker-key", "profile_home": str(worker),
                              "history": [], "history_lock": threading.Lock()})
    assert agent.daybreak is False
