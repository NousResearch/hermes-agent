"""#107924: session.create / cold resume must keep the compute-host gate open.

``dashboard.turn_isolation`` used to lose the first-prompt host route because
``session.create`` / ``_resume_cold`` always armed the 50ms ``_schedule_agent_build``
timer. Once the in-process AIAgent attached, ``_session_uses_compute_host`` stayed
False forever.

This file does NOT stub ``_schedule_agent_build`` — the Detection assertion is that
the gate remains True *after* that timer would have fired.
"""

from __future__ import annotations

import threading
import time

import pytest

from tui_gateway import server


class _DummyAgent:
    """Production-shaped stand-in: ``_attach_built_agent`` stores whatever ``_make_agent`` returns."""

    def __init__(self):
        self.model = "x"
        self.provider = "openrouter"
        self.base_url = ""
        self.api_key = ""
        self._todo_store = None


class FakeSupervisor:
    def __init__(self):
        self.frames = []
        self.callback = None

    def submit_turn(self, frame, *, on_complete=None):
        self.frames.append(frame)
        self.callback = on_complete
        return frame["request_id"]


class _ResumeDB:
    def __init__(self, session_id="cold-1"):
        self.session_id = session_id
        self.rows = {session_id: {"id": session_id, "cwd": "", "message_count": 0, "title": "cold"}}

    def get_session(self, target):
        return self.rows.get(target)

    def get_session_by_title(self, _target):
        return None

    def resolve_resume_session_id(self, target):
        return target

    def reopen_session(self, _target):
        return None

    def get_resume_conversations(self, _target):
        return ([], [])

    def get_ancestor_display_prefix(self, _target):
        return []

    def get_messages_as_conversation(self, _target, **_kwargs):
        return []

    def get_compression_tip(self, target):
        return target

    def assert_resume_safe(self, _target, **_kwargs):
        return None

    def close(self):
        return None


def _isolation_cfg(on: bool) -> dict:
    return {"dashboard": {"turn_isolation": on}}


def _stub_build_side_effects(monkeypatch) -> None:
    """Keep production ``_start_agent_build`` / ``_attach_built_agent`` live."""
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_wire_session_agent", lambda *_a, **_k: False)
    monkeypatch.setattr(server, "_announce_built_agent", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_session_todo_state", lambda _session: None)
    monkeypatch.setattr(server, "_config_model_target", lambda: None)
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_child_run_active", lambda *_a, **_k: False)
    monkeypatch.setattr(server, "_start_session_services", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_probe_config_health", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_set_session_context", lambda *_a, **_k: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_await_resume_history", lambda *_a, **_k: True)
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_run_after_agent_ready", lambda *_a, **_k: None)
    monkeypatch.setattr("tui_gateway.entry.ensure_mcp_discovery_started", lambda: None)


def _flush_prewarm(sid: str) -> dict:
    """Wait until the 50ms timer + build thread would have completed.

    Isolation-on sessions skip the timer and leave ``agent_ready`` unset so
    dispatch-failure fallback can still start an in-process build — do not
    ``ready.wait`` in that case (it would sit out the full timeout).
    """
    session = server._sessions[sid]
    if session.get("_compute_host_active") and session.get("_agent_build_thread") is None:
        time.sleep(0.12)
        return session
    ready = session.get("agent_ready")
    if ready is not None:
        ready.wait(timeout=2.0)
    thread = session.get("_agent_build_thread")
    if thread is not None:
        thread.join(timeout=2.0)
    else:
        time.sleep(0.12)
        ready = session.get("agent_ready")
        if ready is not None:
            ready.wait(timeout=1.0)
        thread = session.get("_agent_build_thread")
        if thread is not None:
            thread.join(timeout=1.0)
    return session


def _create_session() -> str:
    resp = server.handle_request({
        "id": "create",
        "method": "session.create",
        "params": {"cols": 80, "source": "desktop"},
    })
    assert "result" in resp, resp
    return resp["result"]["session_id"]


def _submit(sid: str) -> dict:
    return server.handle_request({
        "id": "submit",
        "method": "prompt.submit",
        "params": {"session_id": sid, "text": "hello"},
    })


@pytest.fixture
def isolation_env(monkeypatch):
    known = set(server._sessions)
    _stub_build_side_effects(monkeypatch)
    monkeypatch.setattr(server, "_make_agent", lambda sid, key, **_kw: _DummyAgent())
    monkeypatch.setattr(server, "_load_cfg", lambda: _isolation_cfg(True))
    monkeypatch.delenv("HERMES_COMPUTE_HOST_CHILD", raising=False)
    yield
    with server._sessions_lock:
        for sid in [s for s in server._sessions if s not in known]:
            server._sessions.pop(sid, None)


def test_session_create_keeps_compute_host_gate_after_prewarm_timer(isolation_env, monkeypatch):
    supervisor = FakeSupervisor()
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: supervisor)
    started: list = []
    real_start = server._start_agent_build

    def _track_start(sid, session):
        started.append(sid)
        return real_start(sid, session)

    monkeypatch.setattr(server, "_start_agent_build", _track_start)

    sid = _create_session()
    session = _flush_prewarm(sid)

    assert server._session_uses_compute_host(session) is True

    started.clear()
    resp = _submit(sid)
    assert resp.get("result", {}).get("turn_isolation") is True, resp
    assert supervisor.frames, "prompt.submit must dispatch to HostSupervisor"
    assert supervisor.frames[0]["type"] == "turn.start"
    assert started == []


def test_session_create_isolation_off_does_not_dispatch(isolation_env, monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: _isolation_cfg(False))
    supervisor = FakeSupervisor()
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: supervisor)

    sid = _create_session()
    session = _flush_prewarm(sid)

    assert server._session_uses_compute_host(session) is False
    assert session.get("agent") is not None

    resp = _submit(sid)
    assert "result" in resp, resp
    assert resp["result"].get("turn_isolation") is not True
    assert supervisor.frames == []


def test_session_create_child_env_fail_open(isolation_env, monkeypatch):
    monkeypatch.setenv("HERMES_COMPUTE_HOST_CHILD", "1")
    supervisor = FakeSupervisor()
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: supervisor)

    sid = _create_session()
    session = _flush_prewarm(sid)

    assert server._session_uses_compute_host(session) is False
    resp = _submit(sid)
    assert resp.get("result", {}).get("turn_isolation") is not True
    assert supervisor.frames == []


def test_cold_resume_keeps_compute_host_gate_after_prewarm_timer(isolation_env, monkeypatch):
    db = _ResumeDB()
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_find_live_session_by_key", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_stored_session_runtime_overrides", lambda _found: {})
    monkeypatch.setattr(server, "_default_session_cwd", lambda *_a, **_k: "/tmp")
    supervisor = FakeSupervisor()
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: supervisor)

    resp = server.handle_request({
        "id": "resume",
        "method": "session.resume",
        "params": {"session_id": "cold-1", "cols": 80, "source": "desktop"},
    })
    assert "result" in resp, resp
    sid = resp["result"]["session_id"]
    session = _flush_prewarm(sid)

    assert server._session_uses_compute_host(session) is True

    submit = _submit(sid)
    assert submit.get("result", {}).get("turn_isolation") is True, submit
    assert supervisor.frames


def test_deferred_resume_skips_inprocess_build_when_isolation_on(isolation_env, monkeypatch):
    db = _ResumeDB("deferred-1")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_find_live_session_by_key", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_stored_session_runtime_overrides", lambda _found: {})
    monkeypatch.setattr(server, "_default_session_cwd", lambda *_a, **_k: "/tmp")
    supervisor = FakeSupervisor()
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: supervisor)

    resp = server.handle_request({
        "id": "resume",
        "method": "session.resume",
        "params": {
            "session_id": "deferred-1", "cols": 80, "source": "desktop",
            "defer_history": True,
        },
    })
    assert "result" in resp, resp
    sid = resp["result"]["session_id"]
    session = server._sessions[sid]
    ready = session.get("resume_history_ready")
    if ready is not None:
        assert ready.wait(timeout=2.0)
    session = _flush_prewarm(sid)

    assert session.get("agent") is None
    assert server._session_uses_compute_host(session) is True
    submit = _submit(sid)
    assert submit.get("result", {}).get("turn_isolation") is True, submit
    assert supervisor.frames


class _BoomSupervisor:
    def __init__(self):
        self.interrupts = []

    def submit_turn(self, frame, *, on_complete=None):
        raise RuntimeError("pipe broken")

    def interrupt(self, sid, *, request_id=None):
        self.interrupts.append((sid, request_id))


def test_session_create_isolation_dispatch_failure_still_starts_inline_build(isolation_env, monkeypatch):
    """Fail-open: a broken HostSupervisor must still start the in-process AIAgent."""
    supervisor = _BoomSupervisor()
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: supervisor)
    monkeypatch.setattr(server, "_persist_session_row_for_submit", lambda *_a, **_k: None)
    started: list = []
    ran_after: list = []
    interrupted: list = []
    build_entered = threading.Event()
    release_build = threading.Event()
    real_start = server._start_agent_build

    def _track_start(sid, session):
        started.append(sid)
        return real_start(sid, session)

    def _blocking_make_agent(*_args, **_kwargs):
        build_entered.set()
        assert release_build.wait(timeout=2.0)
        return _DummyAgent()

    monkeypatch.setattr(server, "_make_agent", _blocking_make_agent)
    monkeypatch.setattr(server, "_start_agent_build", _track_start)
    monkeypatch.setattr(
        server, "_run_after_agent_ready",
        lambda *a, **k: ran_after.append(a[1] if a else None))
    monkeypatch.setattr(
        "agent.interrupt_compat.request_hard_interrupt",
        lambda agent: interrupted.append(agent))

    sid = _create_session()
    session = _flush_prewarm(sid)

    assert server._session_uses_compute_host(session) is True
    assert session.get("agent") is None
    session.update({
        "_metadata_mirror": {"model": "stale-host"},
        "_metadata_mirror_updated_at": 1.0,
        "_metadata_message_count": 7,
        "_compute_host_pending_clarify": {"request_id": "stale"},
    })
    ready = session.get("agent_ready")
    assert ready is not None and not ready.is_set()

    resp = _submit(sid)
    assert "result" in resp, resp
    assert resp["result"].get("status") == "streaming"
    assert resp["result"].get("turn_isolation") is not True
    assert started == [sid], "fallback must invoke _start_agent_build"
    assert ran_after == [sid]
    assert build_entered.wait(timeout=1.0)
    assert session.get("agent") is None, "the ownership assertion must precede agent attach"
    assert session.get("_compute_host_active") is None
    assert session.get("_compute_host_released") is True
    assert server._session_uses_compute_host(session) is False
    for stale_key in (
        "_metadata_mirror", "_metadata_mirror_updated_at", "_metadata_message_count",
        "_compute_host_pending_clarify",
    ):
        assert stale_key not in session
    assert server._arm_isolated_compute_host_session(session) is False
    assert session.get("_compute_host_released") is True
    assert session.get("_compute_host_active") is None
    assert server._session_uses_compute_host(session) is False
    server._apply_compute_host_metadata_mirror(session, {
        "message_count": 99,
        "session_info": {"model": "late-stale-host"},
    })
    assert "_metadata_mirror" not in session
    assert "_metadata_message_count" not in session

    local_interrupt_entered = threading.Event()
    real_sess = server._sess

    def _track_local_sess(params, rid):
        local_interrupt_entered.set()
        return real_sess(params, rid)

    monkeypatch.setattr(server, "_sess", _track_local_sess)
    interrupt_responses: list[dict] = []
    interrupt_thread = threading.Thread(
        target=lambda: interrupt_responses.append(server.handle_request({
            "id": "interrupt",
            "method": "session.interrupt",
            "params": {"session_id": sid},
        })),
    )
    interrupt_thread.start()
    assert local_interrupt_entered.wait(timeout=1.0)
    assert session.get("agent") is None
    assert supervisor.interrupts == []

    release_build.set()
    interrupt_thread.join(timeout=2.0)
    assert not interrupt_thread.is_alive()
    assert ready.wait(timeout=2.0)
    thread = session.get("_agent_build_thread")
    if thread is not None:
        thread.join(timeout=2.0)
    assert isinstance(session.get("agent"), _DummyAgent)
    assert server._session_uses_compute_host(session) is False

    assert len(interrupt_responses) == 1
    interrupt = interrupt_responses[0]
    assert interrupt.get("result", {}).get("status") == "interrupted", interrupt
    assert interrupt["result"].get("turn_isolation") is not True
    assert interrupted == [session["agent"]]
    assert supervisor.interrupts == []
    assert session.get("running") is False


def test_session_create_isolation_dispatch_failure_interrupt_after_inline_build(
    isolation_env, monkeypatch,
):
    """After fallback build completes, interrupt must stay on the local agent."""
    supervisor = _BoomSupervisor()
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: supervisor)
    monkeypatch.setattr(server, "_persist_session_row_for_submit", lambda *_a, **_k: None)
    hold_run = threading.Event()
    monkeypatch.setattr(
        server, "_run_after_agent_ready", lambda *_a, **_k: hold_run.wait(timeout=2.0))
    interrupted: list = []
    monkeypatch.setattr(
        "agent.interrupt_compat.request_hard_interrupt",
        lambda agent: interrupted.append(agent),
    )

    sid = _create_session()
    session = _flush_prewarm(sid)
    resp = _submit(sid)
    assert "result" in resp, resp
    ready = session.get("agent_ready")
    assert ready is not None
    assert ready.wait(timeout=2.0)
    thread = session.get("_agent_build_thread")
    if thread is not None:
        thread.join(timeout=2.0)
    assert isinstance(session.get("agent"), _DummyAgent)
    assert session.get("_compute_host_released") is True
    assert server._session_uses_compute_host(session) is False
    assert session.get("running") is True

    interrupt = server.handle_request({
        "id": "interrupt-after-build",
        "method": "session.interrupt",
        "params": {"session_id": sid},
    })
    hold_run.set()
    run_thread = session.get("_run_thread")
    if run_thread is not None:
        run_thread.join(timeout=2.0)
    assert interrupt.get("result", {}).get("status") == "interrupted", interrupt
    assert interrupt["result"].get("turn_isolation") is not True
    assert interrupted == [session["agent"]]
    assert supervisor.interrupts == []


def test_queued_drain_dispatch_failure_keeps_compute_host_ownership(isolation_env, monkeypatch):
    """A drain reports host dispatch failure; it must not silently switch ownership inline."""
    supervisor = _BoomSupervisor()
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: supervisor)

    sid = _create_session()
    session = server._sessions[sid]
    session["queued_prompt"] = {"text": "continue after crash", "transport": None}
    session["queued_prompts"] = []

    assert server._drain_queued_prompt("drain", sid, session) is True
    assert session.get("running") is False
    assert session.get("_compute_host_active") is True
    assert not session.get("_compute_host_released")
    assert server._session_uses_compute_host(session) is True
