"""Adversarial review probes: retirement-permit accounting + missing-agent path."""

from __future__ import annotations

import threading
import types

import pytest

from agent.turn_authorization import TurnAuthorization
from tui_gateway import server as srv


@pytest.fixture
def fence(monkeypatch):
    from hermes_cli import backend_retirement

    f = backend_retirement.RetirementFence()
    monkeypatch.setattr(backend_retirement, "retirement", f)
    return f


def _session(agent):
    return {
        "agent": agent,
        "session_key": "k",
        "history": [],
        "history_lock": threading.RLock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 80,
        "inflight_turn": None,
        "transport": None,
    }


def _quiet(monkeypatch, emitted):
    monkeypatch.setattr(srv, "_emit", lambda *a, **kw: emitted.append(a))
    monkeypatch.setattr(srv, "_ensure_session_db_row", lambda *a, **kw: True)
    monkeypatch.setattr(srv, "_ensure_active_session_slot", lambda *a, **kw: None)
    monkeypatch.setattr(srv, "_record_turn_marker", lambda *a, **kw: "mk")
    monkeypatch.setattr(srv, "_retire_turn_marker", lambda *a, **kw: None)
    monkeypatch.setattr(srv, "_get_usage", lambda _a: {})
    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda *a, **kw: False)


def test_probe_missing_agent_refusal_permit_and_state(fence, monkeypatch):
    """A turn refused for a missing agent: frame + idle + identity cleared + permit 0."""
    emitted = []
    _quiet(monkeypatch, emitted)
    session = _session(None)
    session["agent_error"] = "boom"
    authorization = TurnAuthorization.from_raw(None)
    with session["history_lock"]:
        session["running"] = True
        turn_id = srv._activate_turn_identity(session)
        session["_active_turn_route"] = "inline"
        session["_active_turn_authorization"] = authorization

    started = srv._run_prompt_submit(
        "rid", "sid", session, "hello", turn_authorization=authorization,
        expected_turn_id=turn_id)

    print("started       =", started)
    print("running       =", session.get("running"))
    print("active id     =", session.get("_active_turn_id"))
    print("permits       =", fence.active_count())
    print("frames        =", [e[0] for e in emitted])
    assert started is False
    assert session["running"] is False
    assert "_active_turn_id" not in session
    assert fence.active_count() == 0, "retirement permit leaked on the missing-agent path"
    assert "message.complete" in [e[0] for e in emitted]


def test_probe_worker_spawn_failure_releases_permit(fence, monkeypatch):
    """spawn_context_thread raising must not strand a process-bound permit."""
    emitted = []
    _quiet(monkeypatch, emitted)
    agent = types.SimpleNamespace(session_id="a", clear_interrupt=lambda: None)
    session = _session(agent)
    authorization = TurnAuthorization.from_raw(None)
    with session["history_lock"]:
        session["running"] = True
        turn_id = srv._activate_turn_identity(session)
        session["_active_turn_route"] = "inline"
        session["_active_turn_authorization"] = authorization

    import agent.memory_provider as mp

    def boom(*a, **kw):
        raise RuntimeError("cannot spawn")

    monkeypatch.setattr(mp, "spawn_context_thread", boom)

    with pytest.raises(RuntimeError):
        srv._run_prompt_submit(
            "rid", "sid", session, "hello", turn_authorization=authorization,
            expected_turn_id=turn_id)

    print("permits after spawn failure =", fence.active_count())
    print("running                      =", session.get("running"))
    assert fence.active_count() == 0, (
        "retirement permit leaked when the turn worker could not be created: "
        "the backend can never retire again (prepare() returns idle=False forever)")


def test_probe_routing_provenance_failure_releases_permit(fence, monkeypatch):
    """A failure between acquire() and the worker start must release the permit."""
    emitted = []
    _quiet(monkeypatch, emitted)
    agent = types.SimpleNamespace(session_id="a", clear_interrupt=lambda: None)
    session = _session(agent)
    authorization = TurnAuthorization.from_raw(None)
    with session["history_lock"]:
        session["running"] = True
        turn_id = srv._activate_turn_identity(session)
        session["_active_turn_route"] = "inline"
        session["_active_turn_authorization"] = authorization

    class _Boom:
        def __enter__(self):
            raise RuntimeError("state registry exploded")

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(srv, "_routing_provenance_db", lambda _s: _Boom())

    with pytest.raises(RuntimeError):
        srv._run_prompt_submit(
            "rid", "sid", session, "hello", turn_authorization=authorization,
            expected_turn_id=turn_id)

    print("permits after routing failure =", fence.active_count())
    assert fence.active_count() == 0, (
        "retirement permit leaked when routing-provenance resolution raised")


def test_probe_leaked_permit_blocks_retirement_forever(fence, monkeypatch):
    """Consequence of the leak: prepare() can never succeed again."""
    emitted = []
    _quiet(monkeypatch, emitted)
    agent = types.SimpleNamespace(session_id="a", clear_interrupt=lambda: None)
    session = _session(agent)
    authorization = TurnAuthorization.from_raw(None)
    with session["history_lock"]:
        session["running"] = True
        turn_id = srv._activate_turn_identity(session)
        session["_active_turn_route"] = "inline"
        session["_active_turn_authorization"] = authorization

    import agent.memory_provider as mp
    monkeypatch.setattr(
        mp, "spawn_context_thread",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("cannot spawn")))
    with pytest.raises(RuntimeError):
        srv._run_prompt_submit(
            "rid", "sid", session, "hello", turn_authorization=authorization,
            expected_turn_id=turn_id)

    # Session is fully idle now; nothing is running anywhere.
    with session["history_lock"]:
        session["running"] = False
    assert fence.active_count() == 0, (
        "retirement permit leaked: active_count() never returns to zero")
    # Called ONCE: prepare() reserves the retirement, so a second call would fail
    # for that reason rather than because of a leak.
    prepared = fence.prepare()
    assert prepared["ok"] is True, (
        "the backend can never retire again: a leaked permit keeps "
        "RetirementFence.prepare() returning idle=False forever")
    fence.cancel(prepared["token"])
