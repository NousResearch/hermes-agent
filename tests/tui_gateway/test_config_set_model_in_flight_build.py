"""config.set model with an explicit provider while session.create's pre-warm agent build races it.

A provider-qualified pick on a session with no agent is recorded as ``model_override``, which the build
reads when it starts. session.create pre-warms that build on a timer thread, so it can start before the
pick is recorded — before config.set arrives, or during the switch — and land on the profile default
while config.set answered success (a desktop plugin opening a chat and picking a model at once; the
model menu too when a pick follows a new chat closely). config.set must settle that build.

The in-flight build is an ``agent_ready`` Event that finishes the build only when the handler waits on
it, so the tests prove the wait happens and need no wall clock.
"""

import threading
import types

from tui_gateway import server

PICK = {"model": "local/model", "provider": "tensorfold"}
DEFAULT = types.SimpleNamespace(model="default/model", provider="default-provider")


class _BuildSettlesWhenAwaited(threading.Event):
    """``agent_ready`` of a running build: it settles (``finish``) the moment the handler waits on it."""

    def __init__(self, finish):
        super().__init__()
        self._finish = finish
        self.waited = False

    def wait(self, timeout=None):
        if not self.is_set():
            self.waited = True
            self._finish()
            self.set()
        return super().wait(timeout)


def _session(finish, *, build_started=True):
    ready = _BuildSettlesWhenAwaited(finish)
    session = {
        "agent": None, "agent_ready": ready, "session_key": "session-key", "history": [],
        "history_lock": threading.Lock(), "history_version": 0, "running": False, "attached_images": [],
        "image_counter": 0, "cols": 80, "slash_worker": None, "show_reasoning": False,
        "tool_progress_mode": "all",
    }
    if build_started:
        session["agent_build_started"] = True
    server._sessions["sid"] = session
    return session, ready


def _record_switches(monkeypatch, session, during_switch=None):
    """Stand-in for _apply_model_switch: no agent yet -> record the pick as the override (as the real
    one does); a live agent -> switch it. ``during_switch`` runs inside the first call."""
    calls = []

    def apply(sid, target, raw, **kw):
        calls.append({"agent": target.get("agent"), "confirmed": kw.get("confirm_expensive_model"),
                      "counted": kw.get("count_switch", True)})
        if during_switch and len(calls) == 1:
            during_switch()
        target["model_override"] = dict(PICK)
        return {"value": PICK["model"], "warning": ""}

    monkeypatch.setattr(server, "_apply_model_switch", apply)
    monkeypatch.setattr(server, "_start_agent_build", lambda *a: calls.append("unexpected build"))
    return calls


def _set_model(value="local/model --provider tensorfold --session"):
    try:
        return server.handle_request(
            {"id": "1", "method": "config.set", "params": {"session_id": "sid", "key": "model", "value": value}})
    finally:
        server._sessions.pop("sid", None)


def test_build_already_running_lands_on_default_then_the_pick_switches_that_live_agent(monkeypatch):
    session, ready = _session(lambda: session.update(agent=DEFAULT))
    calls = _record_switches(monkeypatch, session)

    assert _set_model()["result"]["value"] == "local/model"
    assert ready.waited
    # First the pick is recorded (no agent yet), then the agent the build produced is switched live —
    # its guards already passed, and it is not a second user switch.
    assert calls == [{"agent": None, "confirmed": False, "counted": True},
                     {"agent": DEFAULT, "confirmed": True, "counted": False}]


def test_build_starting_during_the_switch_is_settled_too(monkeypatch):
    """The window seen live: config.set arrives before the pre-warm timer fires, and the build starts
    while the switch resolves — after the entry checks, before the override is recorded."""
    session, ready = _session(lambda: session.update(agent=DEFAULT), build_started=False)
    calls = _record_switches(monkeypatch, session, during_switch=lambda: session.update(agent_build_started=True))

    assert _set_model()["result"]["value"] == "local/model"
    assert ready.waited
    assert [c["agent"] for c in calls] == [None, DEFAULT]


def test_build_that_read_the_pick_is_left_alone(monkeypatch):
    built_on_pick = types.SimpleNamespace(**PICK)
    session, ready = _session(lambda: session.update(agent=built_on_pick))
    calls = _record_switches(monkeypatch, session)

    assert _set_model()["result"]["value"] == "local/model"
    assert ready.waited
    assert len(calls) == 1   # no second switch


def test_failed_prewarm_build_is_rebuilt_on_the_pick(monkeypatch):
    """A broken default makes the racing build fail: rebuild on the recorded pick (the failed-init
    path), rather than answering success for a session with no agent."""
    session, ready = _session(lambda: session.update(agent_error="default provider is not configured"))
    calls = _record_switches(monkeypatch, session)
    rebuilt = types.SimpleNamespace(**PICK)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build",
                        lambda sid, target, failed_ready: calls.append("rebuild")
                        or target.update(agent=rebuilt, agent_error=None))
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda *a, **k: None)

    assert _set_model()["result"]["value"] == "local/model"
    assert ready.waited
    assert calls[1:] == ["rebuild"]


def test_no_build_started_means_no_wait(monkeypatch):
    """Unchanged contract: with no build running the pick rides in as the override, so a broken default
    is never built or waited on."""
    session, ready = _session(lambda: session.update(agent=DEFAULT), build_started=False)
    calls = _record_switches(monkeypatch, session)

    assert _set_model()["result"]["value"] == "local/model"
    assert not ready.waited
    assert len(calls) == 1
