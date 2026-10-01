import json
import logging
import os
import subprocess
import sys
import threading
import time
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.active_sessions import active_session_registry_snapshot
from hermes_cli.browser_connect import ChromeDebugLaunch
from tools import async_delegation as ad
from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport


def _dispatch_sync(req: dict, transport=None) -> dict | None:
    """Run one RPC to completion synchronously, regardless of pool routing.

    Voice RPCs (voice.toggle/record/tts) are in ``_LONG_HANDLERS`` so they run
    on the RPC pool and ``server.dispatch`` returns None â€” the pool worker
    writes the response via the bound transport. These tests exercise the
    handler's business logic, not the routing, so they drive the handler
    inline while preserving the transport-binding semantics ``dispatch``
    applies around a real request.
    """
    token = bind_transport(transport)
    try:
        return server.handle_request(req)
    finally:
        reset_transport(token)


@pytest.fixture(autouse=True)
def _neuter_agent_prewarm_timer(request, monkeypatch):
    """Stub the deferred agent pre-warm timer for every test in this module.

    ``session.create`` and non-eager ``session.resume`` fire a 50 ms
    background ``threading.Timer`` (``_schedule_agent_build``) that calls
    whatever ``server._make_agent`` is patched in AT FIRE TIME. Left live,
    a timer armed by one test outlives it and lands in the NEXT test's
    ``_make_agent`` mock, racily corrupting its captured state (the
    ``'tip' == 'cont_tip'`` flakes in the session_resume tests). Tests that
    exercise the deferred build itself opt back in with
    ``@pytest.mark.real_agent_prewarm``.
    """
    if request.node.get_closest_marker("real_agent_prewarm"):
        yield
        return
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    yield


@pytest.fixture(autouse=True)
def _reap_leaked_notification_pollers():
    """Stop and join notification pollers leaked by each test.

    session.init/create paths start a per-session poller daemon thread. A
    poller left running by one test steals-and-requeues events off the
    PROCESS-GLOBAL process_registry.completion_queue while a later test is
    asserting on it â€” the root cause of the flaky
    test_run_prompt_submit_requeues_all_unstarted_notifications_with_real_threading
    (two CI hits on unrelated PRs, Aug 2026). Set every registered poller's
    stop event (the loop wakes at least every 0.5s), then join with ONE
    small shared budget â€” never per-thread â€” so teardown stays O(seconds)
    for the whole file even when many tests leaked pollers.
    """
    yield
    pollers = [
        (stop, thread)
        for stop, thread in list(server._notification_pollers)
        if thread.is_alive()
    ]
    for stop, _thread in pollers:
        stop.set()
    deadline = time.time() + 3.0
    for _stop, thread in pollers:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        thread.join(timeout=remaining)
    server._notification_pollers[:] = [
        (stop, thread)
        for stop, thread in server._notification_pollers
        if thread.is_alive()
    ]


def test_session_slot_is_claimed_on_first_turn_not_on_create(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("max_concurrent_sessions: 1\n", encoding="utf-8")
    token = set_hermes_home_override(home)

    def _clear_server_sessions():
        for session in list(server._sessions.values()):
            server._teardown_session(session)
        server._sessions.clear()

    try:
        server._cfg_cache = None
        server._cfg_sig = None
        server._cfg_path = None
        _clear_server_sessions()
        monkeypatch.setattr(server, "_start_agent_build", lambda *args, **kwargs: None)
        monkeypatch.setattr(server, "_completion_cwd", lambda params=None: str(tmp_path))

        # Opening a chat must NOT take a slot. Every tile paint and every
        # background reconnect-resume calls session.create, and an unprompted
        # draft has no DB row and is filtered out of the sidebar â€” so a slot
        # held here is invisible to the user while still starving the other
        # surfaces that share this cap.
        first = server._methods["session.create"]("r1", {"cols": 80})
        second = server._methods["session.create"]("r2", {"cols": 80})
        assert "result" in first and "result" in second
        sid = first["result"]["session_id"]
        other = second["result"]["session_id"]
        assert active_session_registry_snapshot() == []

        # The first turn is what claims the slot, and is re-entrant.
        assert server._ensure_active_session_slot(sid, server._sessions[sid]) is None
        assert server._ensure_active_session_slot(sid, server._sessions[sid]) is None
        assert len(active_session_registry_snapshot()) == 1

        blocked = server._ensure_active_session_slot(other, server._sessions[other])
        assert blocked  # refusal reason returned; exact wording not pinned

        closed = server._methods["session.close"]("r3", {"session_id": sid})
        assert closed["result"]["closed"] is True
        assert active_session_registry_snapshot() == []

        assert server._ensure_active_session_slot(other, server._sessions[other]) is None
    finally:
        _clear_server_sessions()
        server._cfg_cache = None
        server._cfg_sig = None
        server._cfg_path = None
        reset_hermes_home_override(token)


def test_session_context_uses_session_cwd(monkeypatch, tmp_path):
    """Desktop/TUI sessions must pin the agent cwd per session.

    The gateway process itself is often launched from apps/desktop in dev, so
    falling back to os.getcwd() makes agents answer from the desktop app folder
    even when the sidebar/session cwd is a real project.
    """
    from agent.runtime_cwd import resolve_agent_cwd

    sid = "cwd-sid"
    session_key = "cwd-key"
    project = tmp_path / "project"
    project.mkdir()
    (project / ".git").mkdir()
    launcher = tmp_path / "apps" / "desktop"
    launcher.mkdir(parents=True)

    server._sessions[sid] = {"session_key": session_key, "cwd": str(project)}
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.chdir(launcher)

    tokens = server._set_session_context(session_key)
    try:
        assert resolve_agent_cwd() == project
    finally:
        server._clear_session_context(tokens)
        server._sessions.pop(sid, None)


def test_handoff_fail_marks_only_inflight_rows(monkeypatch):
    class DbContext:
        def __init__(self, db):
            self.db = db

        def __enter__(self):
            return self.db

        def __exit__(self, *_args):
            return False

    class FakeDb:
        def __init__(self, state):
            self.state = state
            self.failed_with = None

        def get_handoff_state(self, _key):
            return {"state": self.state, "platform": "telegram", "error": None}

        def fail_handoff(self, _key, error):
            self.failed_with = error
            self.state = "failed"

    sid = "rt-handoff"
    server._sessions[sid] = {"session_key": "stored-handoff"}
    try:
        pending = FakeDb("pending")
        monkeypatch.setattr(server, "_session_db", lambda _session: DbContext(pending))
        result = server._methods["handoff.fail"]("r1", {"session_id": sid, "error": "timed out"})
        assert result["result"] == {"failed": True, "state": "failed"}
        assert pending.failed_with == "timed out"

        completed = FakeDb("completed")
        monkeypatch.setattr(server, "_session_db", lambda _session: DbContext(completed))
        result = server._methods["handoff.fail"]("r2", {"session_id": sid, "error": "late timeout"})
        assert result["result"] == {"failed": False, "state": "completed"}
        assert completed.failed_with is None
    finally:
        server._sessions.pop(sid, None)




def test_dashboard_process_isolation_config_coerces_raw_values():
    cfg = {
        "dashboard": {
            "turn_isolation": "yes",
            "compute_host_heartbeat_secs": "30",
            "compute_host_respawn_max": "0",
        }
    }

    assert server._load_dashboard_process_isolation_config(cfg) == {
        "turn_isolation": True,
        "compute_host_heartbeat_secs": 30,
        "compute_host_respawn_max": 0,
    }

    malformed = {"dashboard": "enabled"}
    assert server._load_dashboard_process_isolation_config(malformed) == {
        "turn_isolation": False,
        "compute_host_heartbeat_secs": 15,
        "compute_host_respawn_max": 3,
    }




def test_prompt_submit_dispatches_to_compute_host_when_turn_isolation_enabled(monkeypatch):
    class FakeSupervisor:
        def __init__(self):
            self.frames = []
            self.callback = None

        def submit_turn(self, frame, *, on_complete=None):
            self.frames.append(frame)
            self.callback = on_complete
            return frame["request_id"]

    fake_supervisor = FakeSupervisor()
    seed_history = [{"role": "user", "content": "previous"}]
    server._sessions["iso-sid"] = _session(history=list(seed_history))
    server._sessions["iso-sid"]["agent"] = None
    server._sessions["iso-sid"]["agent_ready"] = threading.Event()
    parent_writes = {"ensure_session": 0, "persist_seed": 0}
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"dashboard": {"turn_isolation": True}},
    )
    monkeypatch.setattr(
        server,
        "_ensure_session_db_row",
        lambda _session: parent_writes.__setitem__(
            "ensure_session", parent_writes["ensure_session"] + 1
        ),
    )
    monkeypatch.setattr(
        server,
        "_persist_branch_seed",
        lambda _session: parent_writes.__setitem__(
            "persist_seed", parent_writes["persist_seed"] + 1
        ),
    )
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: fake_supervisor)

    try:
        resp = server.handle_request(
            {
                "id": "submit",
                "method": "prompt.submit",
                "params": {"session_id": "iso-sid", "text": "hello"},
            }
        )
        assert resp["result"] == {"status": "streaming", "turn_isolation": True}
        assert fake_supervisor.frames[0]["type"] == "turn.start"
        assert fake_supervisor.frames[0]["sid"] == "iso-sid"
        assert fake_supervisor.frames[0]["text"] == "hello"
        assert fake_supervisor.frames[0]["history"] == seed_history
        assert server._sessions["iso-sid"]["history"] == seed_history
        assert parent_writes == {"ensure_session": 0, "persist_seed": 0}
        assert server._sessions["iso-sid"]["running"] is True

        fake_supervisor.callback(
            {
                "type": "turn.end",
                "sid": "iso-sid",
                "request_id": "submit",
                "history_version": 1,
            }
        )
        assert server._sessions["iso-sid"]["running"] is False
        assert server._sessions["iso-sid"]["history_version"] == 1
    finally:
        server._sessions.pop("iso-sid", None)


def test_compute_host_explicit_images_do_not_clear_later_attachment(monkeypatch):
    class _Supervisor:
        def submit_turn(self, _frame, *, on_complete=None):
            session["attached_images"].append("/tmp/c.png")

    session = _session(attached_images=[])
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: _Supervisor())

    response = server._submit_prompt_to_compute_host(
        "r1", "sid", session, "B", image_paths=["/tmp/b.png"]
    )

    assert response["result"]["status"] == "streaming"
    assert session["attached_images"] == ["/tmp/c.png"]


def test_prompt_submit_unknown_session_logs_warning(caplog):
    """A submit against a reaped runtime id must leave a diagnosable trace.

    Regression for #90428: messages sent into a session whose in-memory
    runtime was detached on WS disconnect and orphan-reaped vanished
    silently â€” the 4001 was never logged, so "request arrived and was
    rejected" was indistinguishable from "request never arrived".
    """
    for session in list(server._sessions.values()):
        server._teardown_session(session)
    server._sessions.clear()

    with caplog.at_level(logging.WARNING, logger="tui_gateway.server"):
        resp = _dispatch_sync(
            {
                "id": "r1",
                "method": "prompt.submit",
                "params": {"session_id": "gone-sid", "text": "hello"},
            }
        )

    assert resp["id"] == "r1"
    assert resp["error"]["code"] == 4001
    # The runtime id and method name must be in the warning so a client looping
    # on a stale runtime id is attributable from logs (exact wording not pinned).
    assert any(
        rec.levelno >= logging.WARNING
        and "gone-sid" in rec.getMessage()
        and "prompt.submit" in rec.getMessage()
        for rec in caplog.records
    )


def test_prompt_submit_fails_open_inline_when_compute_host_dispatch_breaks(monkeypatch):
    class _BrokenSupervisor:
        def submit_turn(self, frame, *, on_complete=None):
            if on_complete is not None:
                on_complete(
                    {
                        "type": "turn.error",
                        "request_id": frame["request_id"],
                        "reason": "send_failed",
                        "message": "broken pipe",
                    }
                )
            raise BrokenPipeError("broken pipe")

    class _ImmediateThread:
        def __init__(self, target=None, **_kwargs):
            self._target = target

        def start(self):
            assert self._target is not None
            self._target()

    session = _session(agent=None, agent_ready=threading.Event())
    server._sessions["iso-fallback"] = session
    inline_calls = []
    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: _BrokenSupervisor())
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda _sid, _session: None)
    monkeypatch.setattr(server, "_wait_agent", lambda _session, _rid: None)
    # The deferred inline-fallback thread now waits via the patient variant.
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda _session, _rid, _sid: None)
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda rid, sid, _session, text, **_kwargs: inline_calls.append((rid, sid, text)),
    )
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)

    try:
        resp = server.handle_request(
            {
                "id": "fallback-turn",
                "method": "prompt.submit",
                "params": {"session_id": "iso-fallback", "text": "hello"},
            }
        )
    finally:
        server._sessions.pop("iso-fallback", None)

    assert resp == {
        "jsonrpc": "2.0",
        "id": "fallback-turn",
        "result": {"status": "streaming"},
    }
    assert inline_calls == [("fallback-turn", "iso-fallback", "hello")]
    assert session.get("_compute_host_active") is not True


def test_compute_host_turn_end_updates_metadata_mirror(monkeypatch):
    # _session_info embeds get_update_result(), whose value flips whenever the
    # background update-check thread happens to finish. This test compares two
    # snapshots taken at different times, so pin the value to keep it
    # deterministic regardless of how long the preceding tests ran.
    import hermes_cli.banner as _banner

    monkeypatch.setattr(_banner, "get_update_result", lambda timeout=0.5: None)
    session = _session(
        agent=None,
        agent_ready=threading.Event(),
        history=[{"role": "user", "content": "serving process must not read this"}],
        _compute_host_active=True,
    )
    server._sessions["iso-sid"] = session
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append((event, sid, payload)))

    try:
        server._on_compute_host_turn_done(
            "turn-1",
            "iso-sid",
            session,
            {
                "type": "turn.end",
                "sid": "iso-sid",
                "request_id": "turn-1",
                "session_key": "rotated-session-key",
                "history_version": 4,
                "message_count": 3,
                "session_info": {
                    "model": "host-model",
                    "provider": "host-provider",
                    "system_prompt": "host system prompt",
                    "tools": {"core": ["terminal"]},
                    "usage": {"total": 140, "context_used": 80, "context_max": 1000},
                },
            },
        )

        assert session["session_key"] == "rotated-session-key"
        assert session["history_version"] == 4
        assert session["_metadata_mirror"]["model"] == "host-model"
        info = server._session_info(None, session)
        assert info["model"] == "host-model"
        assert info["provider"] == "host-provider"
        assert info["system_prompt"] == "host system prompt"
        assert info["tools"] == {"core": ["terminal"]}
        assert info["usage"]["total"] == 140
        assert "credential_warning" not in info
        assert emitted[-1] == ("session.info", "iso-sid", info)
    finally:
        server._sessions.pop("iso-sid", None)


def test_compute_host_open_request_survives_activation_and_proxies_locks_and_responses(monkeypatch):
    """A host-owned server request (batch clarify) is mirrored by the parent so `open_requests` replays it;
    `clarify.lock` and the client's response frame are relayed to the child that owns the wait."""
    class _Supervisor:
        def __init__(self):
            self.responses = []

        def respond(self, sid, params, *, timeout=15.0):
            self.responses.append((sid, dict(params), timeout))
            lock = params.get("lock") or {}
            remaining = ["q1"] if lock.get("question_id") == "q0" else []
            return {"type": "respond.ack", "response": {"result": {"status": "ok", "remaining": remaining}}}

    sid = "host-clarify"
    supervisor = _Supervisor()
    session = _session(agent=None, agent_ready=threading.Event(), _compute_host_active=True)
    server._sessions[sid] = session
    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: supervisor)
    monkeypatch.setattr(server, "write_json", lambda _message: True)

    questions = [{"qid": "q0", "question": "First?", "choices": ["a"]}, {"qid": "q1", "question": "Second?", "choices": ["b"]}]
    try:
        server._relay_compute_host_rpc({"jsonrpc": "2.0", "id": "srq-host", "method": "clarify",
                                        "params": {"session_id": sid, "questions": questions}})

        activated = server._live_session_payload(sid, session)
        assert activated["open_requests"] == [{"id": "srq-host", "method": "clarify",
                                               "params": {"session_id": sid, "questions": questions}}]

        response = server.handle_request({"id": "lock-q0", "method": "clarify.lock",
                                          "params": {"request_id": "srq-host", "question_id": "q0", "answer": "a"}})
        assert response["result"] == {"status": "ok", "remaining": ["q1"]}
        assert supervisor.responses == [(sid, {"lock": {"request_id": "srq-host", "question_id": "q0", "answer": "a"}}, 15.0)]
        assert server._live_session_payload(sid, session)["open_requests"][0]["params"]["answers"] == {"q0": "a"}

        # The client's response frame (cancel-all) is relayed to the child and clears the mirror.
        assert server.dispatch({"jsonrpc": "2.0", "id": "srq-host", "result": {}}) is None
        assert supervisor.responses[-1] == (sid, {"frame": {"jsonrpc": "2.0", "id": "srq-host", "result": {}}}, 15.0)
        assert "open_requests" not in server._live_session_payload(sid, session)
    finally:
        server._sessions.pop(sid, None)


def test_compute_host_interrupt_forwards_when_parent_running_mirror_is_stale(monkeypatch):
    """The host, not the parent's mirrored running flag, owns interruption."""
    interrupted = []

    class _Supervisor:
        def interrupt(self, sid, *, request_id=None):
            interrupted.append((sid, request_id))

    sid = "host-stale-running"
    server._sessions[sid] = _session(
        agent=None,
        agent_ready=threading.Event(),
        _compute_host_active=True,
        running=False,
    )
    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: _Supervisor())

    try:
        response = server.handle_request(
            {"id": "interrupt", "method": "session.interrupt", "params": {"session_id": sid}}
        )
        assert response["result"] == {"status": "interrupted", "turn_isolation": True}
        assert interrupted == [(sid, "interrupt-interrupt")]
    finally:
        server._sessions.pop(sid, None)


def test_compute_host_interrupt_skips_lazy_session_with_no_hosted_turn(monkeypatch):
    """A lazy session that never submitted a hosted turn must not spawn a host.

    ``HostSupervisor.interrupt()`` calls ``start()``, so forwarding the
    interrupt unconditionally would launch a compute-host child just to
    deliver an interrupt for a session with no work in it.
    """
    class _Supervisor:
        def interrupt(self, sid, *, request_id=None):  # pragma: no cover - must not run
            raise AssertionError("interrupt must not be forwarded for idle lazy sessions")

    sid = "lazy-idle"
    session = _session(
        agent_ready=threading.Event(),
        running=False,
    )
    session["agent"] = None  # _session() substitutes a namespace for None
    server._sessions[sid] = session
    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: _Supervisor())

    try:
        response = server.handle_request(
            {"id": "interrupt", "method": "session.interrupt", "params": {"session_id": sid}}
        )
        assert response["result"] == {"status": "interrupted", "turn_isolation": True}
    finally:
        server._sessions.pop(sid, None)


def test_slash_exec_compress_flag_on_applies_host_control_mirror(monkeypatch):
    class _ExplodingWorker:
        def __init__(self, *args, **kwargs):
            raise AssertionError("slash worker should not run for isolated /compress")

    class _FakeSupervisor:
        def __init__(self):
            self.controls = []

        def control(self, sid, *, route_name, payload=None, wait=True, timeout=30.0, on_late_ack=None):
            self.controls.append((sid, route_name, dict(payload or {}), wait))
            return {
                "type": "control.ack",
                "sid": sid,
                "request_id": (payload or {}).get("request_id", "control-1"),
                "route_name": route_name,
                "output": "Compressed 4 â†’ 2 messages",
                "session_key": "host-rotated-key",
                "history_version": 9,
                "message_count": 2,
                "session_info": {
                    "model": "host-model",
                    "provider": "host-provider",
                    "usage": {"total": 42},
                },
            }

    fake = _FakeSupervisor()
    session = _session(agent=None, agent_ready=threading.Event(), _compute_host_active=True)
    server._sessions["sid"] = session
    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: fake)
    monkeypatch.setattr(server, "_SlashWorker", _ExplodingWorker)
    monkeypatch.setattr(server, "_compress_session_history", lambda *a, **k: (_ for _ in ()).throw(AssertionError("parent compressed")))
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: (_ for _ in ()).throw(AssertionError("parent identity guard ran")))

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "slash.exec",
                "params": {"command": "compress focus", "session_id": "sid"},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert resp["result"]["output"] == "Compressed 4 â†’ 2 messages"
    assert fake.controls[0][1] == "slash.compress"
    assert fake.controls[0][2]["command"] == "/compress focus"
    assert session["session_key"] == "host-rotated-key"
    assert session["history_version"] == 9
    assert server._session_info(None, session)["model"] == "host-model"


def test_prompt_submit_golden_transcript_matches_flag_off_and_on(monkeypatch):
    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **_kwargs):
            self._target = target

        def start(self):
            assert self._target is not None
            self._target()

    class _Agent:
        model = "gold-model"
        provider = "gold-provider"
        session_id = "session-key"
        session_input_tokens = 10
        session_output_tokens = 5
        session_prompt_tokens = 10
        session_completion_tokens = 5
        session_total_tokens = 15
        session_api_calls = 1
        context_compressor = None

        def clear_interrupt(self):
            return None

        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            if stream_callback is not None:
                stream_callback("hi")
            return {
                "final_response": "hi",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "hi"},
                ],
            }

    fixed_info = {"model": "gold-model", "provider": "gold-provider", "usage": {"total": 15}}
    usage = server._get_usage(_Agent())
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_session_info", lambda _agent, _session=None: dict(fixed_info))
    monkeypatch.setattr(server, "make_stream_renderer", lambda _cols: None)
    monkeypatch.setattr(server, "render_message", lambda _raw, _cols: None)
    fake_title = types.ModuleType("agent.title_generator")
    setattr(fake_title, "maybe_auto_title", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "agent.title_generator", fake_title)

    def run_flag_off():
        events = []
        monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: events.append((event, sid, payload)))
        monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": False}})
        server._sessions["sid"] = _session(
            agent=_Agent(), model_override={"model": "gold-model", "provider": "gold-provider"}
        )
        try:
            response = server.handle_request(
                {"id": "turn-1", "method": "prompt.submit", "params": {"session_id": "sid", "text": "hello"}}
            )
            assert response["result"]["status"] == "streaming"
            return events
        finally:
            server._sessions.pop("sid", None)

    def run_flag_on():
        events = []
        monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: events.append((event, sid, payload)))
        monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})

        class _FakeSupervisor:
            def submit_turn(self, frame, *, on_complete=None):
                sid = frame["sid"]
                server._emit("message.start", sid)
                server._emit("message.delta", sid, {"text": "hi"})
                server._emit("message.complete", sid, {"text": "hi", "usage": usage, "status": "complete"})
                server._emit("session.info", sid, dict(fixed_info))
                if on_complete is not None:
                    on_complete(
                        {
                            "type": "turn.end",
                            "sid": sid,
                            "request_id": frame["request_id"],
                            "session_key": "session-key",
                            "history_version": 1,
                            "message_count": 2,
                            "session_info": dict(fixed_info),
                            "session_info_emitted": True,
                        }
                    )
                return frame["request_id"]

        monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: _FakeSupervisor())
        session = _session(
            agent=None,
            agent_ready=threading.Event(),
            _compute_host_active=True,
            model_override={"model": "gold-model", "provider": "gold-provider"},
        )
        session["agent"] = None
        server._sessions["sid"] = session
        try:
            response = server.handle_request(
                {"id": "turn-1", "method": "prompt.submit", "params": {"session_id": "sid", "text": "hello"}}
            )
            assert response["result"]["status"] == "streaming"
            return events
        finally:
            server._sessions.pop("sid", None)

    assert run_flag_on() == run_flag_off()


def test_session_context_explicit_cwd_for_ephemeral_task(monkeypatch, tmp_path):
    """Background/preview tasks use ephemeral ids absent from `_sessions`, so the
    parent workspace is passed explicitly; it must pin instead of clearing back
    to the gateway launch dir."""
    from agent.runtime_cwd import resolve_agent_cwd

    project = tmp_path / "project"
    project.mkdir()
    launcher = tmp_path / "apps" / "desktop"
    launcher.mkdir(parents=True)

    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.chdir(launcher)

    tokens = server._set_session_context("bg_deadbe", cwd=str(project))
    try:
        assert resolve_agent_cwd() == project
    finally:
        server._clear_session_context(tokens)


def _write_profile_cfg(home: Path, cwd: str | None) -> Path:
    import hermes_yaml as yaml

    home.mkdir(parents=True, exist_ok=True)
    cfg = {"terminal": {"cwd": cwd}} if cwd is not None else {}
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return home


def test_profile_scoped_mcp_discovery_uses_target_home(monkeypatch, tmp_path):
    """MCP discovery must start under the selected profile's HERMES_HOME."""
    from hermes_cli import mcp_startup
    from hermes_constants import get_hermes_home
    from tui_gateway import entry

    profile_home = tmp_path / "profiles" / "sheepyr"
    profile_home.mkdir(parents=True)

    (profile_home / "config.yaml").write_text(
        "mcp_servers:\n"
        "  bluesky_sheepyr:\n"
        "    command: test-command\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "default"))
    token = set_hermes_home_override(str(profile_home))

    seen = []

    monkeypatch.setattr(mcp_startup, "_mcp_discovery_started", set())
    monkeypatch.setattr(mcp_startup, "_mcp_discovery_thread", {})
    # ensure_mcp_discovery_started flips this module global; monkeypatch it so
    # the enablement doesn't leak into sibling tests in this file.
    monkeypatch.setattr(entry, "_mcp_discovery_enabled", False)
    monkeypatch.setattr(
        mcp_startup,
        "_discover_mcp_tools_without_interactive_oauth",
        lambda: seen.append(str(get_hermes_home())),
    )

    try:
        entry.ensure_mcp_discovery_started()
        thread = mcp_startup._current_home_thread()
        assert thread is not None
        thread.join(timeout=2)
    finally:
        reset_hermes_home_override(token)

    assert seen == [str(profile_home)]


def test_profile_scoped_agent_build_starts_mcp_discovery_in_profile_home(
    monkeypatch, tmp_path
):
    """Agent construction must start MCP discovery under the selected profile."""
    import threading
    import uuid

    from hermes_constants import get_hermes_home

    profile_home = tmp_path / "profiles" / "sheepyr"
    profile_home.mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "default"))

    seen = []
    built = threading.Event()

    monkeypatch.setattr(
        server,
        "_make_agent",
        lambda *args, **kwargs: built.set()
        or type("Agent", (), {"model": "test"})(),
    )
    monkeypatch.setattr(
        "tui_gateway.entry.ensure_mcp_discovery_started",
        lambda: seen.append(str(get_hermes_home())),
    )
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_SlashWorker", lambda *args: None)
    monkeypatch.setattr(server, "_attach_worker", lambda *args: None)
    monkeypatch.setattr(server, "_config_model_target", lambda: ("", ""))
    # CI runs this huge file serially under load; a prior session's _build can
    # still be finishing (session.info emit) when the next test starts, so a
    # 2s Event wait flakes. Unique sid + longer bound; still fail closed.
    monkeypatch.setattr(server, "_start_notification_poller", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *a, **k: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)

    ready = threading.Event()
    sid = f"test-sid-{uuid.uuid4().hex[:8]}"
    session = {
        "agent_ready": ready,
        "session_key": f"test-key-{uuid.uuid4().hex[:8]}",
        "profile_home": str(profile_home),
    }

    server._sessions[sid] = session
    try:
        server._start_agent_build(sid, session)
        assert built.wait(timeout=15), "agent build thread never called _make_agent"
        assert ready.wait(timeout=5), "agent_ready never set after build"
    finally:
        server._sessions.pop(sid, None)

    assert seen == [str(profile_home)]


def test_profile_scoped_agent_build_installs_secret_scope(monkeypatch, tmp_path):
    """Agent construction must install the selected profile's secret scope.

    Without it, get_secret() falls through to process os.environ, so a session
    "switched" to profile X resolves credentials from the LAUNCH profile's
    .env (#67605 item 2).
    """
    import threading
    import uuid

    from agent.secret_scope import current_secret_scope

    profile_home = tmp_path / "profiles" / "grace"
    profile_home.mkdir(parents=True)
    (profile_home / ".env").write_text(
        "PROXMOX_TOKEN=grace-secret\n", encoding="utf-8"
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "default"))

    scopes = []
    built = threading.Event()

    def _fake_make_agent(*args, **kwargs):
        scope = current_secret_scope()
        scopes.append(dict(scope) if scope else None)
        built.set()
        return type("Agent", (), {"model": "test"})()

    monkeypatch.setattr(server, "_make_agent", _fake_make_agent)
    monkeypatch.setattr(
        "tui_gateway.entry.ensure_mcp_discovery_started", lambda: None
    )
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_SlashWorker", lambda *args: None)
    monkeypatch.setattr(server, "_attach_worker", lambda *args: None)
    monkeypatch.setattr(server, "_config_model_target", lambda: ("", ""))
    # Same CI flake class as the MCP profile-home test: bound wait + less work
    # on the build thread (no poller / late MCP refresh / session.info emit).
    monkeypatch.setattr(server, "_start_notification_poller", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *a, **k: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)

    ready = threading.Event()
    sid = f"test-secret-sid-{uuid.uuid4().hex[:8]}"
    session = {
        "agent_ready": ready,
        "session_key": f"test-secret-key-{uuid.uuid4().hex[:8]}",
        "profile_home": str(profile_home),
    }

    server._sessions[sid] = session
    try:
        server._start_agent_build(sid, session)
        assert built.wait(timeout=15), "agent build thread never called _make_agent"
        assert ready.wait(timeout=5), "agent_ready never set after build"
    finally:
        server._sessions.pop(sid, None)

    assert scopes == [{"PROXMOX_TOKEN": "grace-secret"}]


def test_profile_configured_cwd_reads_target_profile(tmp_path):
    """A profile's own terminal.cwd is read from its config.yaml."""
    project = tmp_path / "proj"
    project.mkdir()
    home = _write_profile_cfg(tmp_path / "home", str(project))
    assert server._profile_configured_cwd(home) == str(project)


def test_profile_configured_cwd_skips_placeholders_and_missing(tmp_path):
    """Placeholder values, missing config, and bad paths fall through to None."""
    assert server._profile_configured_cwd(None) is None
    assert server._profile_configured_cwd(tmp_path / "nope") is None
    for placeholder in (".", "auto", "cwd", ""):
        home = _write_profile_cfg(tmp_path / placeholder.strip("."), placeholder)
        assert server._profile_configured_cwd(home) is None
    home = _write_profile_cfg(tmp_path / "ghost", str(tmp_path / "does-not-exist"))
    assert server._profile_configured_cwd(home) is None


def test_completion_cwd_prefers_profile_over_stale_env(monkeypatch, tmp_path):
    """Issue #40334: a new session bound to another profile must use THAT
    profile's terminal.cwd, not the launch profile's stale TERMINAL_CWD."""
    profile_b = tmp_path / "ef-design"
    profile_b.mkdir()
    home = _write_profile_cfg(tmp_path / "home-b", str(profile_b))
    stale = tmp_path / "mahjong"
    stale.mkdir()

    monkeypatch.setenv("TERMINAL_CWD", str(stale))
    monkeypatch.setattr(server, "_hermes_home", tmp_path / "launch-home")
    monkeypatch.setattr(server, "_profile_home", lambda name: home if name else None)

    assert server._completion_cwd({"profile": "ef-design"}) == str(profile_b)
    # No profile and no launch config â†’ fallback to the launch env var.
    assert server._completion_cwd({}) == str(stale)


def test_completion_cwd_prefers_launch_config_over_stale_env(monkeypatch, tmp_path):
    """Dashboard /chat's launch-profile in-memory gateway must honor config.

    The embedded Node TUI child gets TERMINAL_CWD from the dashboard PTY bridge,
    but the default-profile chat attaches to the dashboard process's already
    running in-memory gateway. That process may not have TERMINAL_CWD in its own
    environment (or has a stale one), so config.yaml is read directly and wins
    over the process env before falling back to the launch directory.
    """
    configured = tmp_path / "omni"
    configured.mkdir()
    stale = tmp_path / "hermes-agent"
    stale.mkdir()
    launch_home = _write_profile_cfg(tmp_path / "launch-home", str(configured))

    monkeypatch.setenv("TERMINAL_CWD", str(stale))
    monkeypatch.setattr(server, "_hermes_home", launch_home)
    monkeypatch.setattr(server, "_profile_home", lambda _name: None)

    assert server._completion_cwd({}) == str(configured)


def test_default_session_cwd_prefers_launch_config(monkeypatch, tmp_path):
    """A freshly created / resumed session with no explicit cwd lands in the
    configured terminal.cwd, not os.getcwd(), even when the in-memory gateway
    process env carries a stale TERMINAL_CWD."""
    configured = tmp_path / "workspace"
    configured.mkdir()
    stale = tmp_path / "launch-dir"
    stale.mkdir()
    launch_home = _write_profile_cfg(tmp_path / "launch-home", str(configured))

    monkeypatch.setenv("TERMINAL_CWD", str(stale))
    monkeypatch.setattr(server, "_hermes_home", launch_home)

    assert server._default_session_cwd() == str(configured)

    # No launch config â†’ fall back to the process env var.
    (launch_home / "config.yaml").write_text("{}", encoding="utf-8")
    assert server._default_session_cwd() == str(stale)


def test_completion_cwd_explicit_cwd_wins_over_profile(monkeypatch, tmp_path):
    """An explicit client-provided cwd still beats the profile config."""
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    profile_b = tmp_path / "configured"
    profile_b.mkdir()
    home = _write_profile_cfg(tmp_path / "home-c", str(profile_b))

    monkeypatch.setattr(server, "_profile_home", lambda name: home if name else None)
    result = server._completion_cwd(
        {"cwd": str(explicit), "cwd_explicit": True, "profile": "ef-design"}
    )
    assert result == str(explicit)


def test_completion_cwd_profile_overrides_inherited_workspace(monkeypatch, tmp_path):
    """Issue #52589: the desktop seeds a new chat's cwd from its app-global workspace
    (the launch profile's configured directory) â€” that inherited default must NOT
    override the target profile's own ``terminal.cwd``."""
    launch_ws = tmp_path / "workspace"
    launch_ws.mkdir()
    profile_ws = tmp_path / "products"
    profile_ws.mkdir()
    home = _write_profile_cfg(tmp_path / "home-dev", str(profile_ws))

    monkeypatch.setattr(server, "_profile_home", lambda name: home if name else None)
    # No cwd_explicit: the client cwd is the inherited app-global workspace.
    assert (
        server._completion_cwd({"profile": "dev", "cwd": str(launch_ws)}) == str(profile_ws)
    )


def test_completion_cwd_explicit_pick_wins_over_profile(monkeypatch, tmp_path):
    """Issue #52589 regression guard: a deliberate workspace pick (``cwd_explicit``)
    still beats the profile's configured ``terminal.cwd``."""
    explicit = tmp_path / "explicit-lane"
    explicit.mkdir()
    profile_ws = tmp_path / "products"
    profile_ws.mkdir()
    home = _write_profile_cfg(tmp_path / "home-dev", str(profile_ws))

    monkeypatch.setattr(server, "_profile_home", lambda name: home if name else None)
    assert (
        server._completion_cwd(
            {"profile": "dev", "cwd": str(explicit), "cwd_explicit": True}
        )
        == str(explicit)
    )


def test_completion_cwd_inherited_workspace_without_profile_cfg_kept(monkeypatch, tmp_path):
    """An inherited workspace stays when the target profile has NO configured
    terminal.cwd â€” the profile override only applies when one exists (#52589)."""
    launch_ws = tmp_path / "workspace"
    launch_ws.mkdir()
    no_cfg_home = tmp_path / "home-plain"
    no_cfg_home.mkdir()
    (no_cfg_home / "config.yaml").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(server, "_profile_home", lambda name: no_cfg_home if name else None)
    assert server._completion_cwd({"profile": "plain", "cwd": str(launch_ws)}) == str(launch_ws)


def test_terminal_task_cwd_local_backend_uses_session_cwd(monkeypatch, tmp_path):
    """A local terminal backend must keep host-validated session cwd behaviour."""
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    assert server._terminal_task_cwd({"cwd": str(project)}) == str(project)


def test_terminal_task_cwd_ssh_uses_remote_path_unvalidated(monkeypatch):
    """SSH (non-local) backend: the configured remote cwd is used verbatim even
    though it does not exist on the local host. This is the jonbohz fix â€” host
    `isdir()` validation would otherwise discard the remote path and fall back
    to os.getcwd(), running commands against the wrong machine."""
    remote = "/home/jonboh/workspace/proj"  # does not exist on this host
    assert not os.path.isdir(remote)
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_CWD", remote)

    assert server._terminal_task_cwd({"cwd": "/some/host/dir"}) == remote


def test_terminal_task_cwd_ssh_falls_back_to_config(monkeypatch):
    """When TERMINAL_CWD is unset, the SSH path reads terminal.cwd from config."""
    remote = "/home/jonboh/workspace/from-config"
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"terminal": {"cwd": remote}})

    assert server._terminal_task_cwd({"cwd": "/some/host/dir"}) == remote


def test_terminal_task_cwd_ssh_sentinel_cwd_uses_remote_home(monkeypatch):
    """An SSH placeholder must not register the TUI host's session cwd."""
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_CWD", "auto")
    monkeypatch.setattr(server, "_load_cfg", lambda: {"terminal": {"cwd": "."}})

    assert server._terminal_task_cwd({"cwd": "/host/session/dir"}) == "~"


class _ChunkyStdout:
    def __init__(self):
        self.parts: list[str] = []

    def write(self, text: str) -> int:
        for ch in text:
            self.parts.append(ch)
            time.sleep(0.0001)
        return len(text)

    def flush(self) -> None:
        return None


class _BrokenStdout:
    def write(self, text: str) -> int:
        raise BrokenPipeError

    def flush(self) -> None:
        return None


def test_write_json_serializes_concurrent_writes(monkeypatch):
    """Assert StdioTransport holds _stdout_lock across the full stream.write.

    The old char-by-char sleep mock made this test take long enough that
    leftover background write_json calls from earlier cases in this file
    could append an extra line (intermittent ``assert 9 == 8`` on CI/main).
    Match the WS concurrent-send check: count in-flight writes, and only
    assert on frames that carry this test's marker payload.
    """
    marker = "x" * 24
    active = 0
    max_active = 0
    gate = threading.Lock()
    frames: list[str] = []

    class RecordingStdout:
        def write(self, text: str) -> int:
            nonlocal active, max_active
            with gate:
                active += 1
                max_active = max(max_active, active)
            try:
                # Release the GIL while "in write" so a missing outer lock
                # would let another thread bump max_active above 1.
                time.sleep(0.01)
                frames.append(text)
            finally:
                with gate:
                    active -= 1
            return len(text)

        def flush(self) -> None:
            return None

    monkeypatch.setattr(server, "_real_stdout", RecordingStdout())

    barrier = threading.Barrier(8)

    def _worker(seq: int) -> None:
        barrier.wait(timeout=5)
        server.write_json({"seq": seq, "text": marker})

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()

    assert max_active == 1

    ours = []
    for frame in frames:
        assert frame.endswith("\n"), frame
        obj = json.loads(frame)
        if obj.get("text") == marker and "seq" in obj:
            ours.append(obj)

    assert {obj["seq"] for obj in ours} == set(range(8))
    assert len(ours) == 8


def test_write_json_returns_false_on_broken_pipe(monkeypatch):
    monkeypatch.setattr(server, "_real_stdout", _BrokenStdout())

    assert server.write_json({"ok": True}) is False


def test_write_json_unserializable_payload_becomes_error_frame(monkeypatch, caplog):
    """The stdio twin of the WS guard (#92506): an unserializable result must reach the Ink TUI /
    stdio bridge as a JSON-RPC error frame with the original id plus a log line, not kill the pool
    worker silently while the client waits forever."""
    import datetime
    import logging

    out = _ChunkyStdout()
    monkeypatch.setattr(server, "_real_stdout", out)
    with caplog.at_level(logging.ERROR, logger="tui_gateway.transport"):
        assert server.write_json({"jsonrpc": "2.0", "id": "profiles",
                                  "result": {"created": datetime.datetime(2026, 8, 22)}}) is True
    frame = json.loads("".join(out.parts))
    assert frame["id"] == "profiles" and frame["error"]["code"] == -32603
    assert "datetime" in frame["error"]["message"]
    assert any(rec.levelno >= logging.ERROR for rec in caplog.records)


def test_write_json_drops_detached_ws_frames(monkeypatch):
    out = _ChunkyStdout()
    monkeypatch.setattr(server, "_real_stdout", out)
    server._sessions["detached-sid"] = {"transport": server._detached_ws_transport}
    try:
        assert server.write_json({
            "jsonrpc": "2.0",
            "method": "event",
            "params": {"session_id": "detached-sid", "type": "message.delta"},
        }) is False
        assert out.parts == []
    finally:
        server._sessions.pop("detached-sid", None)


def test_usage_ticker_emits_wrapped_usage_payload(monkeypatch):
    # The live ticker must nest the snapshot under a "usage" key, matching the
    # message.complete / session.info payloads the desktop & TUI handlers read
    # as payload.usage. Emitting the bare _get_usage() dict (payload.input/total
    # â€¦) silently drops every live tick on the client side.
    events: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload: events.append((event_type, sid, payload))
    )
    snapshot = {"input": 1200, "total": 1280}
    monkeypatch.setattr(server, "_get_usage", lambda agent: dict(snapshot))

    stop, thread = server._start_usage_ticker("sess-1", object(), interval=0.01)
    # The dedup baseline is sampled synchronously inside _start_usage_ticker,
    # so this mutation is guaranteed to read as the first counter movement.
    snapshot["total"] = 2400
    try:
        deadline = time.time() + 1.0
        while not events and time.time() < deadline:
            time.sleep(0.01)
    finally:
        stop.set()
        thread.join(timeout=2.0)

    assert events, "ticker never emitted"
    event_type, sid, payload = events[0]
    assert event_type == "session.usage"
    assert sid == "sess-1"
    assert payload == {"usage": {"input": 1200, "total": 2400}}


def test_usage_ticker_skips_unchanged_snapshots(monkeypatch):
    # A single long API call leaves the token counters frozen for many
    # intervals; the ticker must emit nothing at all (the client already has
    # the turn-start values from the previous message.complete / session.info).
    # Only a changed snapshot emits.
    events: list[dict] = []
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload: events.append(payload)
    )
    snapshot = {"input": 1200, "total": 1280}
    monkeypatch.setattr(server, "_get_usage", lambda agent: dict(snapshot))

    stop, thread = server._start_usage_ticker("sess-1", object(), interval=0.01)
    try:
        # ~15 ticks with counters frozen at the turn-start baseline: zero frames.
        time.sleep(0.15)
        assert events == []

        # Counters move â†’ the next tick emits the new snapshot.
        snapshot["total"] = 2400
        deadline = time.time() + 1.0
        while not events and time.time() < deadline:
            time.sleep(0.01)
    finally:
        stop.set()
        thread.join(timeout=2.0)

    assert events == [{"usage": {"input": 1200, "total": 2400}}]


def test_usage_ticker_baseline_sampled_before_thread_start(monkeypatch):
    """The dedup baseline must be sampled synchronously in _start_usage_ticker,
    not inside the ticker thread: a late-scheduled thread would otherwise seed
    itself with counters the turn's first API call already bumped, absorbing
    that first growth so it never emits."""
    events: list[dict] = []
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload: events.append(payload)
    )
    snapshot = {"input": 1200, "total": 1280}
    monkeypatch.setattr(server, "_get_usage", lambda agent: dict(snapshot))

    class _SlowStartThread(threading.Thread):
        def start(self):
            # Deterministic stand-in for a scheduler delay: the turn's first
            # API call bumps the counters before the ticker thread ever runs.
            snapshot["total"] = 2400
            super().start()

    monkeypatch.setattr(server, "_RealThread", _SlowStartThread)

    stop, thread = server._start_usage_ticker("sess-1", object(), interval=0.01)
    try:
        deadline = time.time() + 1.0
        while not events and time.time() < deadline:
            time.sleep(0.01)
    finally:
        stop.set()
        thread.join(timeout=2.0)

    # An in-thread seed would have read 2400 as the baseline and stayed
    # silent; the synchronous seed (1280) sees it as the first growth.
    assert events == [{"usage": {"input": 1200, "total": 2400}}]


def test_usage_ticker_stop_join_prevents_late_ticks(monkeypatch):
    """The stop sequence (set + join) must guarantee no session.usage after it
    returns: a tick captured mid-turn but emitted after message.complete would
    roll the client's final usage back to a stale snapshot (clients merge
    payload.usage unconditionally)."""
    events: list[str] = []
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload: events.append(event_type)
    )

    in_snapshot = threading.Event()
    release = threading.Event()
    calls = {"n": 0}

    def _blocking_get_usage(agent):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"total": 0}  # dedup seed
        # First real tick: hold it mid-snapshot so the stop lands while the
        # iteration is already past the stop.wait() gate.
        in_snapshot.set()
        release.wait(2.0)
        return {"total": 999}

    monkeypatch.setattr(server, "_get_usage", _blocking_get_usage)

    stop, thread = server._start_usage_ticker("sess-1", object(), interval=0.01)
    assert in_snapshot.wait(2.0), "ticker never reached a snapshot"

    # Turn ends while the tick is mid-snapshot: run the exact stop sequence
    # _run_prompt_submit uses, then emit message.complete.
    stop.set()
    release.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "ticker thread survived the stop sequence"
    server._emit("message.complete", "sess-1", {})

    # The in-flight tick was dropped (stop re-checked before emit), so nothing
    # can land after â€” let alone overwrite â€” the final usage.
    assert "session.usage" not in events
    assert events[-1] == "message.complete"


def test_run_prompt_submit_never_ticks_after_message_complete(monkeypatch):
    """End-to-end ordering through _run_prompt_submit: live session.usage ticks
    happen strictly before message.complete, never after it."""
    events: list[str] = []
    tick_seen = threading.Event()

    def _record_emit(event_type, sid, payload=None):
        events.append(event_type)
        if event_type == "session.usage":
            tick_seen.set()

    monkeypatch.setattr(server, "_emit", _record_emit)

    counter = {"n": 0}

    def _moving_usage(agent):
        counter["n"] += 1
        return {"total": counter["n"]}  # moves every sample â†’ every tick emits

    monkeypatch.setattr(server, "_get_usage", _moving_usage)
    real_ticker = server._start_usage_ticker
    monkeypatch.setattr(
        server,
        "_start_usage_ticker",
        lambda sid, agent, interval=1.0: real_ticker(sid, agent, interval=0.01),
    )
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)

    class _Agent:
        def run_conversation(
            self,
            prompt,
            conversation_history=None,
            stream_callback=None,
            persist_user_message=None,
        ):
            # Hold the turn open until at least one live tick has fired.
            assert tick_seen.wait(5.0), "no live tick during the turn"
            return {"final_response": "done", "messages": [], "completed": True}

    server._sessions["sid"] = _session(agent=_Agent())
    try:
        server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": "hello"},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert "session.usage" in events
    assert "message.complete" in events
    last_tick = max(i for i, e in enumerate(events) if e == "session.usage")
    assert last_tick < events.index("message.complete")


def test_usage_ticker_unbounded_join_waits_out_blocked_emit(monkeypatch):
    """A tick stalled inside _emit (a transport write can block up to
    _WS_WRITE_TIMEOUT_S = 10s on a stalled event loop) must be waited out by
    the stop sequence, not abandoned: stop.set() + an unbounded join may only
    return after the in-flight emit has fully flushed, so nothing can land
    after message.complete."""
    order: list[str] = []
    in_emit = threading.Event()
    release = threading.Event()

    def _stalled_emit(event_type, sid, payload):
        in_emit.set()
        release.wait(10.0)  # the stalled transport write
        order.append(event_type)

    monkeypatch.setattr(server, "_emit", _stalled_emit)

    counter = {"n": 0}

    def _moving_usage(agent):
        counter["n"] += 1
        return {"total": counter["n"]}  # moves every sample â†’ a tick emits

    monkeypatch.setattr(server, "_get_usage", _moving_usage)

    stop, thread = server._start_usage_ticker("sess-1", object(), interval=0.01)
    assert in_emit.wait(2.0), "no tick got in flight"

    # Run the exact stop sequence _run_prompt_submit uses, on a side thread so
    # the test can observe whether it returns while the emit is still stuck.
    stopped = threading.Event()

    def _stop_sequence():
        stop.set()
        thread.join()
        stopped.set()

    stopper = threading.Thread(target=_stop_sequence, daemon=True)
    stopper.start()

    # While the tick is stalled in the transport write, the stop sequence must
    # NOT complete â€” a timed join returning here is exactly the bug: the
    # caller would proceed to message.complete with the tick still pending.
    assert not stopped.wait(0.2), "stop sequence returned with the tick still in flight"

    release.set()
    assert stopped.wait(2.0), "stop sequence never completed after the emit flushed"
    stopper.join(timeout=2.0)

    # The flushed tick strictly precedes anything the caller emits afterwards.
    order.append("message.complete")
    assert order == ["session.usage", "message.complete"]




def test_tui_verbose_tool_details_fail_closed_when_redaction_fails(monkeypatch):
    redact_module = types.ModuleType("agent.redact")

    def fail_redaction(*_args, **_kwargs):
        raise RuntimeError("redaction unavailable")

    setattr(redact_module, "redact_sensitive_text", fail_redaction)
    monkeypatch.setitem(sys.modules, "agent.redact", redact_module)

    assert server._redact_tui_verbose_text("api_key=secret") == ""
    assert server._tool_args_text({"api_key": "secret"}) == ""
    assert server._tool_result_text("token=secret") == ""


def test_tui_verbose_tool_details_are_capped_before_emit(monkeypatch):
    monkeypatch.setattr(server, "_TUI_VERBOSE_TEXT_MAX_CHARS", 12)
    monkeypatch.setattr(server, "_TUI_VERBOSE_TEXT_MAX_LINES", 2)

    capped = server._cap_tui_verbose_text("one\ntwo\nthree\nfour")

    assert capped.endswith("three\nfour")
    assert "one" not in capped


def test_tui_verbose_default_cap_stays_small(monkeypatch):
    # Regression guard for #34095: the verbose tool text shipped to the TUI is
    # rendered into a persisted, expanded-by-default trail block for the whole
    # session. Raising this cap back toward the old 16KB re-introduces the Ink
    # render-tree blowup that silently OOM-killed the TUI. Keep it small.
    huge = "x" * 40_000
    capped = server._cap_tui_verbose_text(huge)

    assert len(capped) < 2_000


def test_tui_verbose_tool_events_omit_details_when_redaction_fails(monkeypatch):
    redact_module = types.ModuleType("agent.redact")

    def fail_redaction(*_args, **_kwargs):
        raise RuntimeError("redaction unavailable")

    setattr(redact_module, "redact_sensitive_text", fail_redaction)
    monkeypatch.setitem(sys.modules, "agent.redact", redact_module)

    events: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload: events.append((event_type, sid, payload))
    )
    monkeypatch.setitem(
        server._sessions,
        "redaction-test",
        {"tool_progress_mode": "verbose", "tool_started_at": {}},
    )

    server._on_tool_start("redaction-test", "tool-1", "terminal", {"command": "pwd"})
    server._on_tool_complete("redaction-test", "tool-1", "terminal", {"command": "pwd"}, "done")

    assert events[0][0] == "tool.start"
    assert events[1][0] == "tool.complete"
    assert "args_text" not in events[0][2]
    assert "result_text" not in events[1][2]


def test_tui_tool_output_risk_event_exposes_metadata_without_raw_output(monkeypatch):
    events: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload: events.append((event_type, sid, payload))
    )
    monkeypatch.setitem(
        server._sessions,
        "risk-test",
        {"tool_progress_mode": "all"},
    )

    server._on_tool_progress(
        "risk-test",
        "tool.output_risk",
        "web_extract",
        tool_call_id="tool-1",
        risk_metadata={
            "risk": "high",
            "findings": ["prompt_injection"],
            "redacted": False,
        },
    )

    assert events == [(
        "tool.output_risk",
        "risk-test",
        {
            "tool_id": "tool-1",
            "name": "web_extract",
            "risk": "high",
            "findings": ["prompt_injection"],
            "redacted": False,
        },
    )]
    assert "result" not in events[0][2]


def test_tui_clarify_lifecycle_events_emit_when_tool_progress_off(monkeypatch):
    events: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload: events.append((event_type, sid, payload))
    )
    monkeypatch.setitem(
        server._sessions,
        "clarify-off-test",
        {"tool_progress_mode": "off", "tool_started_at": {}},
    )

    args = {"question": "Pick one", "choices": ["A", "B"]}
    result = '{"question":"Pick one","choices_offered":["A","B"],"user_response":"A"}'

    server._on_tool_start("clarify-off-test", "tool-clarify", "clarify", args)
    server._on_tool_complete("clarify-off-test", "tool-clarify", "clarify", args, result)

    assert [event[0] for event in events] == ["tool.start", "tool.complete"]
    assert events[0][2]["name"] == "clarify"
    assert events[0][2]["tool_id"] == "tool-clarify"
    assert events[1][2]["result"]["user_response"] == "A"


def test_tui_non_interactive_tool_lifecycle_stays_hidden_when_tool_progress_off(monkeypatch):
    events: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload: events.append((event_type, sid, payload))
    )
    monkeypatch.setitem(
        server._sessions,
        "terminal-off-test",
        {"tool_progress_mode": "off", "tool_started_at": {}},
    )

    server._on_tool_start("terminal-off-test", "tool-1", "terminal", {"command": "pwd"})
    server._on_tool_complete("terminal-off-test", "tool-1", "terminal", {"command": "pwd"}, "done")

    assert events == []


def test_dispatch_rejects_non_object_request():
    resp = server.dispatch([])

    assert resp["id"] is None
    assert resp["error"]["code"] == -32600


def test_dispatch_rejects_non_object_params():
    resp = server.dispatch({"id": "1", "method": "session.create", "params": []})

    assert resp["id"] == "1"
    assert resp["error"]["code"] == -32602




def test_system_battery_fails_open(monkeypatch):
    def boom():
        raise RuntimeError("no battery subsystem")

    monkeypatch.setitem(
        sys.modules,
        "agent.battery",
        types.SimpleNamespace(read_battery=boom, battery_category=lambda _s: "dim"),
    )

    resp = server.dispatch({"id": "b2", "method": "system.battery", "params": {}})

    assert resp["result"]["available"] is False
    assert resp["result"]["percent"] is None


def test_config_set_battery_toggles_and_persists(monkeypatch):
    writes: dict[str, object] = {}
    monkeypatch.setattr(server, "_load_cfg", lambda: {"display": {"battery": False}})
    monkeypatch.setattr(
        server, "_write_config_key", lambda k, v: writes.__setitem__(k, v)
    )

    resp = server.dispatch(
        {"id": "c1", "method": "config.set", "params": {"key": "battery", "value": ""}}
    )

    assert resp["result"] == {"key": "battery", "value": "on"}
    assert writes == {"display.battery": True}


def test_config_set_battery_explicit_off(monkeypatch):
    writes: dict[str, object] = {}
    monkeypatch.setattr(server, "_load_cfg", lambda: {"display": {"battery": True}})
    monkeypatch.setattr(
        server, "_write_config_key", lambda k, v: writes.__setitem__(k, v)
    )

    resp = server.dispatch(
        {
            "id": "c2",
            "method": "config.set",
            "params": {"key": "battery", "value": "off"},
        }
    )

    assert resp["result"] == {"key": "battery", "value": "off"}
    assert writes == {"display.battery": False}


def test_voice_toggle_returns_configured_record_key(monkeypatch):
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"voice": {"record_key": "ctrl+o"}},
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.voice_mode",
        types.SimpleNamespace(
            check_voice_requirements=lambda: {"available": True, "details": ""}
        ),
    )
    # ``voice.toggle`` action=on mutates ``os.environ["HERMES_VOICE"]``
    # directly (CLI parity, runtime-only flag). Take monkeypatch
    # ownership of the var so the change is reverted at teardown and
    # later tests don't inherit a stale ON state (Copilot round-5
    # review on #19835).
    monkeypatch.setenv("HERMES_VOICE", "0")

    on_resp = _dispatch_sync(
        {"id": "voice-on", "method": "voice.toggle", "params": {"action": "on"}}
    )
    status_resp = _dispatch_sync(
        {"id": "voice-status", "method": "voice.toggle", "params": {"action": "status"}}
    )

    assert on_resp["result"]["record_key"] == "ctrl+o"
    assert status_resp["result"]["record_key"] == "ctrl+o"




def test_voice_toggle_handles_non_dict_voice_cfg(monkeypatch):
    """Round-3 Copilot review regression on #19835.

    ``_load_cfg()`` is raw ``yaml.safe_load()`` output â€” a hand-edited
    ``voice: true`` / ``voice: cmd+b`` / ``voice: null`` leaves ``voice``
    as a bool/str/None, not a dict. Previously ``.get("record_key")``
    on a non-dict broke every ``voice.toggle`` branch. Now it falls
    back to the documented default.
    """
    monkeypatch.setitem(
        sys.modules,
        "tools.voice_mode",
        types.SimpleNamespace(
            check_voice_requirements=lambda: {"available": True, "details": ""}
        ),
    )

    for bad in (True, "cmd+b", None, 42, ["ctrl+b"]):
        monkeypatch.setattr(server, "_load_cfg", lambda b=bad: {"voice": b})

        status_resp = _dispatch_sync(
            {
                "id": "voice-status",
                "method": "voice.toggle",
                "params": {"action": "status"},
            }
        )

        assert (
            status_resp["result"]["record_key"] == "ctrl+b"
        ), f"voice.record_key fell back to default for voice={bad!r}"

    # Round-4 follow-up: the YAML root itself may be a non-dict. A
    # hand-edit that collapses config.yaml to a scalar / list would
    # otherwise crash ``.get("voice")`` before the inner isinstance
    # guard gets a chance to run.
    for bad_root in (True, None, [], "ctrl+b", 42):
        monkeypatch.setattr(server, "_load_cfg", lambda r=bad_root: r)

        status_resp = _dispatch_sync(
            {
                "id": "voice-status-root",
                "method": "voice.toggle",
                "params": {"action": "status"},
            }
        )

        assert (
            status_resp["result"]["record_key"] == "ctrl+b"
        ), f"voice.record_key fell back to default for root={bad_root!r}"


def test_voice_record_start_handles_non_dict_voice_cfg(monkeypatch):
    """Round-7 Copilot review regression on #19835.

    The ``voice.record`` start path previously read
    ``_load_cfg().get("voice", {}).get(...)`` without any shape checks.
    When ``voice`` is a non-dict (bool/scalar/list) ``get`` raises
    AttributeError and the handler returns 5025 instead of falling
    back to the VAD defaults. Now it uses ``_voice_cfg_dict()`` and
    non-numeric silence values are coerced to the documented defaults.
    """
    captured: dict = {}

    def fake_start_continuous(**kwargs):
        captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(
            start_continuous=fake_start_continuous, stop_continuous=lambda: None
        ),
    )
    monkeypatch.setenv("HERMES_VOICE", "1")

    for bad in (True, "cmd+b", None, 42, ["ctrl+b"], {"silence_threshold": "loud"}):
        captured.clear()
        monkeypatch.setattr(server, "_load_cfg", lambda b=bad: {"voice": b})

        resp = _dispatch_sync(
            {
                "id": "voice-record",
                "method": "voice.record",
                "params": {"action": "start"},
            }
        )

        assert (
            "result" in resp
        ), f"voice.record raised for voice={bad!r}: {resp.get('error')}"
        assert resp["result"]["status"] == "recording"
        assert captured["silence_threshold"] == 200
        assert captured["silence_duration"] == 3.0
        assert captured["auto_restart"] is False


    # Round-12 Copilot review regression on #19835: ``bool`` is a subclass
    # of ``int``, so the naive ``isinstance(threshold, (int, float))``
    # guard would forward ``silence_threshold: true`` as ``1`` instead
    # of falling back to the documented 200 default.
    for bad_bool_cfg in (
        {"silence_threshold": True, "silence_duration": False},
        {"silence_threshold": False},
        {"silence_duration": True},
    ):
        captured.clear()
        monkeypatch.setattr(server, "_load_cfg", lambda c=bad_bool_cfg: {"voice": c})

        resp = _dispatch_sync(
            {
                "id": "voice-record-bool",
                "method": "voice.record",
                "params": {"action": "start"},
            }
        )

        assert "result" in resp, f"voice.record raised for bool cfg={bad_bool_cfg!r}"
        assert (
            captured["silence_threshold"] == 200
        ), f"bool silence_threshold leaked through for {bad_bool_cfg!r}"
        assert (
            captured["silence_duration"] == 3.0
        ), f"bool silence_duration leaked through for {bad_bool_cfg!r}"
        assert captured["auto_restart"] is False


def test_prompt_submit_typed_stop_phrase_ends_voice_chat(monkeypatch):
    """Typed bare stop phrase during an active voice chat is consumed at the
    prompt.submit choke point: voice mode flips off, a distinct
    voice.transcript {stop_phrase, typed} event fires, and NO turn starts.
    """
    calls = {"stop_continuous": 0}
    emitted = []
    monkeypatch.setattr(
        server, "_emit", lambda event, sid, payload=None: emitted.append((event, payload))
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.voice_mode",
        types.SimpleNamespace(
            is_voice_stop_phrase=lambda t: t.strip().lower().strip(".!?") == "stop"
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(
            stop_continuous=lambda force_transcribe=False: calls.__setitem__(
                "stop_continuous", calls["stop_continuous"] + 1
            )
        ),
    )
    monkeypatch.setattr(server, "_tts_stream_stop", lambda user_barge=False: None)
    monkeypatch.setenv("HERMES_VOICE", "1")
    monkeypatch.setenv("HERMES_VOICE_TTS", "1")

    resp = server.dispatch(
        {
            "id": "typed-stop",
            "method": "prompt.submit",
            "params": {"session_id": "any-sid", "text": "Stop."},
        }
    )

    assert resp["result"] == {"voice_stopped": True}
    assert os.environ["HERMES_VOICE"] == "0"
    assert os.environ["HERMES_VOICE_TTS"] == "0"
    assert calls["stop_continuous"] == 1
    assert ("voice.transcript", {"stop_phrase": True, "typed": True}) in emitted


def test_prompt_submit_typed_stop_passes_through_when_voice_off(monkeypatch):
    """Outside a voice chat, typed "stop" is a normal message â€” the stop
    matcher must not even be consulted (guard is on voice mode)."""
    monkeypatch.setitem(
        sys.modules,
        "tools.voice_mode",
        types.SimpleNamespace(
            is_voice_stop_phrase=lambda t: (_ for _ in ()).throw(
                AssertionError("stop matcher must not run when voice is off")
            )
        ),
    )
    monkeypatch.setenv("HERMES_VOICE", "0")

    resp = server.dispatch(
        {
            "id": "typed-stop-off",
            "method": "prompt.submit",
            "params": {"session_id": "missing-sid", "text": "stop"},
        }
    )

    # The submit proceeds into normal handling (here: unknown session error),
    # NOT the voice_stopped consumption path.
    assert resp.get("result") != {"voice_stopped": True}




def test_wake_owner_is_sticky_and_routes_detection_to_first_transport(monkeypatch):
    from tools import wake_word

    state = {"owner": None, "callback": None, "paused": False}
    voice_callbacks = {}

    def start_listening(callback, *, owner, config, external_audio=False):
        if state["owner"] is not None and state["owner"] is not owner:
            raise wake_word.WakeWordInUse
        state.update(
            owner=owner,
            callback=callback,
            paused=False,
            external_audio=bool(external_audio),
        )

    def pause_listening(*, owner):
        if state["owner"] is not owner:
            return False
        state["paused"] = True
        return True

    def stop_listening(*, owner):
        if state["owner"] is not owner:
            return False
        state.update(owner=None, callback=None, paused=False)
        return True

    def resume_listening(*, owner):
        if state["owner"] is not owner:
            return False
        state["paused"] = False
        return True

    def start_continuous(**callbacks):
        voice_callbacks.update(callbacks)
        return True

    monkeypatch.setattr(wake_word, "load_wake_word_config", lambda: {
        "enabled": True,
        "phrase": "hey hermes",
        "surface": "auto",
        "start_new_session": True,
    })
    monkeypatch.setattr(wake_word, "check_wake_word_requirements", lambda _cfg: {
        "available": True,
        "phrase": "hey hermes",
        "provider": "test",
        "hint": "",
    })
    monkeypatch.setattr(wake_word, "start_listening", start_listening)
    monkeypatch.setattr(wake_word, "pause_listening", pause_listening)
    monkeypatch.setattr(wake_word, "stop_listening", stop_listening)
    monkeypatch.setattr(wake_word, "owns_listener", lambda owner: state["owner"] is owner)
    monkeypatch.setattr(
        wake_word,
        "is_listening",
        lambda: state["owner"] is not None and not state["paused"],
    )
    monkeypatch.setattr(
        wake_word,
        "resume_listening",
        resume_listening,
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(
            start_continuous=start_continuous,
            stop_continuous=lambda **_kwargs: None,
        ),
    )
    monkeypatch.setenv("HERMES_VOICE", "1")

    first = types.SimpleNamespace(_closed=False)
    second = types.SimpleNamespace(_closed=False)
    emitted = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload: emitted.append(
            (event, sid, payload, server.current_transport())
        ),
    )
    server._wake_owner_transport = None
    server._wake_owner_surface = ""
    try:
        started = _dispatch_sync({
            "id": "wake-1",
            "method": "wake.start",
            "params": {"surface": "gui", "session_id": "first-session"},
        }, transport=first)
        denied = _dispatch_sync({
            "id": "wake-2",
            "method": "wake.start",
            "params": {"surface": "tui", "session_id": "second-session"},
        }, transport=second)
        denied_stop = server.dispatch({
            "id": "wake-stop-2",
            "method": "wake.stop",
            "params": {},
        }, transport=second)
        denied_voice_stop = _dispatch_sync({
            "id": "voice-stop-2",
            "method": "voice.record",
            "params": {"action": "stop"},
        }, transport=second)

        assert started["result"]["started"] is True
        assert denied["result"] == {
            "started": False,
            "reason": "owned",
            "owner_surface": "gui",
        }
        assert denied_stop["result"] == {
            "stopped": False,
            "reason": "not_owner",
            "disabled_persisted": False,
        }
        assert denied_voice_stop["result"] == {
            "status": "busy",
            "reason": "wake_owned",
        }

        state["callback"]()
        assert emitted == [(
            "wake.detected",
            "first-session",
            {"phrase": "hey hermes", "profile": None, "start_new_session": True},
            first,
        )]
        assert state["paused"] is True

        voice_started = _dispatch_sync({
            "id": "voice-start-1",
            "method": "voice.record",
            "params": {"action": "start", "session_id": "first-session"},
        }, transport=first)
        assert voice_started["result"]["status"] == "recording"
        voice_callbacks["on_status"]("idle")
        assert state["paused"] is False

        stopped = server.dispatch({
            "id": "wake-stop-1",
            "method": "wake.stop",
            "params": {},
        }, transport=first)
        assert stopped["result"] == {
            "stopped": True,
            "reason": None,
            "disabled_persisted": False,
        }

        reclaimed = _dispatch_sync({
            "id": "wake-reclaim-2",
            "method": "wake.start",
            "params": {"surface": "tui", "session_id": "second-session"},
        }, transport=second)
        assert reclaimed["result"]["started"] is True
        assert state["owner"] is second

        state["callback"]()
        assert emitted[-1] == (
            "wake.detected",
            "second-session",
            {"phrase": "hey hermes", "profile": None, "start_new_session": True},
            second,
        )

        stopped_again = server.dispatch({
            "id": "wake-stop-2-after-reclaim",
            "method": "wake.stop",
            "params": {},
        }, transport=second)
        assert stopped_again["result"] == {
            "stopped": True,
            "reason": None,
            "disabled_persisted": False,
        }
    finally:
        server._wake_owner_transport = None
        server._wake_owner_surface = ""


def test_wake_toggle_persists_enabled_flag_only_on_explicit_gesture(monkeypatch):
    """The ear toggle / /wake on|off write wake_word.enabled; auto-arm never does."""
    from tools import wake_word

    config = {"enabled": False, "phrase": "hey hermes", "surface": "auto",
              "start_new_session": True}
    persisted = []

    def fake_persist(enabled):
        persisted.append(enabled)
        config["enabled"] = enabled
        return True

    monkeypatch.setattr(server, "_persist_wake_enabled", fake_persist)
    monkeypatch.setattr(wake_word, "load_wake_word_config", lambda: dict(config))
    monkeypatch.setattr(wake_word, "check_wake_word_requirements", lambda _cfg: {
        "available": True,
        "phrase": "hey hermes",
        "provider": "test",
        "hint": "",
    })
    listener = {"owner": None}
    monkeypatch.setattr(
        wake_word, "start_listening",
        lambda callback, *, owner, config, external_audio=False: listener.update(
            owner=owner, external_audio=bool(external_audio)
        ),
    )
    monkeypatch.setattr(
        wake_word, "stop_listening",
        lambda *, owner: listener["owner"] is owner and not listener.update(owner=None),
    )
    monkeypatch.setattr(wake_word, "owns_listener", lambda owner: listener["owner"] is owner)

    transport = types.SimpleNamespace(_closed=False)
    server._wake_owner_transport = None
    server._wake_owner_surface = ""
    try:
        # Passive auto-arm (no persist): refused, config untouched.
        passive = _dispatch_sync({
            "id": "wake-passive",
            "method": "wake.start",
            "params": {"surface": "gui"},
        }, transport=transport)
        assert passive["result"] == {"started": False, "reason": "disabled"}
        assert persisted == []

        # Explicit gesture: enables in config AND arms.
        clicked = _dispatch_sync({
            "id": "wake-click",
            "method": "wake.start",
            "params": {"surface": "gui", "persist": True},
        }, transport=transport)
        assert clicked["result"]["started"] is True
        assert clicked["result"]["enabled_persisted"] is True
        assert persisted == [True]

        # Explicit stop: disables in config.
        stopped = server.dispatch({
            "id": "wake-click-off",
            "method": "wake.stop",
            "params": {"persist": True},
        }, transport=transport)
        assert stopped["result"]["stopped"] is True
        assert stopped["result"]["disabled_persisted"] is True
        assert persisted == [True, False]

        # persist does NOT override an explicit surface scoping.
        config.update(enabled=True, surface="tui")
        scoped = _dispatch_sync({
            "id": "wake-scoped",
            "method": "wake.start",
            "params": {"surface": "gui", "persist": True},
        }, transport=transport)
        assert scoped["result"] == {"started": False, "reason": "disabled_for_surface"}
        assert persisted == [True, False]
    finally:
        server._wake_owner_transport = None
        server._wake_owner_surface = ""




def test_voice_record_start_forwards_max_recording_seconds(monkeypatch):
    """voice.max_recording_seconds must reach start_continuous from the TUI.

    The CLI wiring alone doesn't cover TUI recordings: the gateway forwards
    recorder params explicitly, so a missing kwarg here silently leaves the
    cap dead in the TUI while CLI tests stay green. Semantics mirror the
    silence params: non-numeric / bool / missing falls back to the documented
    120 default, an explicit numeric value <= 0 disables the cap.
    """
    captured: dict = {}

    def fake_start_continuous(**kwargs):
        captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(
            start_continuous=fake_start_continuous, stop_continuous=lambda: None
        ),
    )
    monkeypatch.setenv("HERMES_VOICE", "1")

    for cfg, expected in (
        ({"max_recording_seconds": 45}, 45),        # explicit cap forwarded as-is
        ({"max_recording_seconds": 0}, 0.0),        # explicit 0 = disabled
        ({"max_recording_seconds": -5}, 0.0),       # negative = disabled
        ({}, 120.0),                                # missing = documented default
        ({"max_recording_seconds": True}, 120.0),   # bool must not become 1s cap
        ({"max_recording_seconds": "long"}, 120.0), # garbage = documented default
    ):
        captured.clear()
        monkeypatch.setattr(server, "_load_cfg", lambda c=cfg: {"voice": c})

        resp = _dispatch_sync(
            {
                "id": "voice-record-cap",
                "method": "voice.record",
                "params": {"action": "start"},
            }
        )

        assert "result" in resp, f"voice.record raised for cfg={cfg!r}: {resp.get('error')}"
        assert resp["result"]["status"] == "recording"
        assert (
            captured["max_recording_seconds"] == expected
        ), f"cfg={cfg!r} forwarded {captured.get('max_recording_seconds')!r}, expected {expected!r}"


def test_voice_record_stop_forces_transcription(monkeypatch):
    captured: dict = {}

    def fake_stop_continuous(**kwargs):
        captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(
            start_continuous=lambda **_kwargs: None,
            stop_continuous=fake_stop_continuous,
        ),
    )

    resp = _dispatch_sync(
        {
            "id": "voice-record-stop",
            "method": "voice.record",
            "params": {"action": "stop"},
        }
    )

    assert resp["result"]["status"] == "stopped"
    assert captured["force_transcribe"] is True


def test_voice_record_stop_updates_event_session_id(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(
            start_continuous=lambda **_kwargs: True,
            stop_continuous=lambda **_kwargs: None,
        ),
    )
    monkeypatch.setattr(server, "_voice_event_sid", "old-session")

    resp = _dispatch_sync(
        {
            "id": "voice-record-stop-session",
            "method": "voice.record",
            "params": {"action": "stop", "session_id": "new-session"},
        }
    )

    assert resp["result"]["status"] == "stopped"
    assert server._voice_event_sid == "new-session"


def test_voice_record_start_reports_busy_when_stop_is_in_progress(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(
            start_continuous=lambda **_kwargs: False,
            stop_continuous=lambda **_kwargs: None,
        ),
    )
    monkeypatch.setenv("HERMES_VOICE", "1")
    monkeypatch.setattr(server, "_load_cfg", lambda: {"voice": {}})

    resp = _dispatch_sync(
        {
            "id": "voice-record-busy",
            "method": "voice.record",
            "params": {"action": "start"},
        }
    )

    assert resp["result"]["status"] == "busy"


def test_voice_toggle_tts_branch_also_carries_record_key(monkeypatch):
    """Round-2 Copilot review regression on #19835.

    The ``tts`` branch used to omit ``record_key`` from its response, so a
    TUI client would parse ``r.record_key ?? 'ctrl+b'`` and reset a
    custom binding to the default on every TTS toggle. Every branch of
    ``voice.toggle`` now carries the configured key so frontend state
    stays authoritative.
    """
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"voice": {"record_key": "ctrl+space"}},
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.voice_mode",
        types.SimpleNamespace(
            check_voice_requirements=lambda: {"available": True, "details": ""}
        ),
    )
    monkeypatch.setenv("HERMES_VOICE", "1")
    # setenv (not delenv) â€” the handler writes HERMES_VOICE_TTS directly, and
    # delenv on an absent var registers no teardown, leaking TTS=1 into every
    # later test in the file (which now spins up the streaming TTS pipeline).
    monkeypatch.setenv("HERMES_VOICE_TTS", "0")

    tts_resp = _dispatch_sync(
        {"id": "voice-tts", "method": "voice.toggle", "params": {"action": "tts"}}
    )

    assert tts_resp["result"]["record_key"] == "ctrl+space"
    assert tts_resp["result"]["tts"] is True


def test_load_enabled_toolsets_prefers_tui_env(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "web, terminal, ,memory")

    assert server._load_enabled_toolsets() == ["web", "terminal", "memory"]


def test_load_enabled_toolsets_filters_invalid_tui_env(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "web, nope")
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )

    assert server._load_enabled_toolsets() == ["web"]
    assert "nope" in capsys.readouterr().err


def test_load_enabled_toolsets_accepts_plugin_env_after_discovery(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "plugin_demo")

    import toolsets

    discovered = {"ready": False}
    original_validate = toolsets.validate_toolset

    def fake_validate(name):
        return name == "plugin_demo" and discovered["ready"] or original_validate(name)

    monkeypatch.setattr(toolsets, "validate_toolset", fake_validate)
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(
            discover_plugins=lambda: discovered.update({"ready": True})
        ),
    )

    assert server._load_enabled_toolsets() == ["plugin_demo"]


def test_load_enabled_toolsets_folds_project_into_focus_posture(monkeypatch):
    # Focus-mode coding posture returns before the config fallback, but it's
    # still a GUI-only resolver â€” `project` must come along so the desktop keeps
    # the project tools while sitting in a repo.
    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)

    import agent.coding_context as cc

    monkeypatch.setattr(cc, "coding_selection", lambda **_: ["coding", "figma"])

    assert server._load_enabled_toolsets("tui") == ["coding", "figma", "project"]


def test_load_enabled_toolsets_honors_disabled_project_on_focus_path(monkeypatch):
    """#54433: focus/coding posture must not re-add `project` when disabled."""
    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)

    import agent.coding_context as cc

    monkeypatch.setattr(cc, "coding_selection", lambda **_: ["coding", "figma"])
    monkeypatch.setattr(server, "_load_disabled_toolsets", lambda: ["project"])

    result = server._load_enabled_toolsets("tui")
    assert result == ["coding", "figma"]
    assert "project" not in result


def test_load_enabled_toolsets_honors_disabled_project_on_configured_fallback(
    monkeypatch,
):
    """#54433: configured/fallback path must not re-add disabled `project`."""
    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)

    import agent.coding_context as cc
    import hermes_cli.tools_config as tools_config_mod

    monkeypatch.setattr(cc, "coding_selection", lambda **_: None)
    monkeypatch.setattr(
        tools_config_mod,
        "_get_platform_tools",
        lambda *_args, **_kwargs: {"memory", "web"},
    )
    monkeypatch.setattr(server, "_load_disabled_toolsets", lambda: ["project"])

    result = server._load_enabled_toolsets("tui")
    assert result == ["memory", "web"]
    assert "project" not in result


def test_with_session_toolsets_keeps_desktop_ui_when_project_disabled(monkeypatch):
    """#54433: a disabled name is subtracted from the client-surface fold-in, but
    ``desktop_ui`` â€” the client's own control surface â€” survives the subtraction."""
    monkeypatch.setattr(server, "_load_disabled_toolsets", lambda: ["project"])

    assert server._with_session_toolsets(["memory"], "desktop") == [
        "memory",
        "desktop_ui",
    ]
    # Nothing disabled: the fold-in keeps both client-surface toolsets.
    monkeypatch.setattr(server, "_load_disabled_toolsets", lambda: None)
    assert server._with_session_toolsets(["memory"], "desktop") == [
        "memory",
        "desktop_ui",
        "project",
    ]


def test_load_enabled_toolsets_rejects_disabled_mcp_env(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "mcp-off")
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )

    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "read_raw_config",
        lambda: {"mcp_servers": {"mcp-off": {"enabled": False}}},
    )
    monkeypatch.setattr(
        config_mod, "load_config", lambda: {"platform_toolsets": {"cli": ["memory"]}}
    )

    # Sorted: ["memory", "project"]. `kanban` is a configurable opt-in and is
    # never recovered onto a saved list; `project` is GUI-only, folded in by
    # _load_enabled_toolsets. Toolsets inside their first release
    # (_RECENTLY_SHIPPED_TOOLSETS) are back-filled onto saved lists that never
    # offered them â€” allow those too.
    from hermes_cli.tools_config import _RECENTLY_SHIPPED_TOOLSETS

    result = server._load_enabled_toolsets()
    assert result is not None
    assert {"memory", "project"} <= set(result)
    assert "kanban" not in result
    assert set(result) - {"memory", "project"} <= _RECENTLY_SHIPPED_TOOLSETS
    err = capsys.readouterr().err
    assert "mcp-off" in err


def test_load_enabled_toolsets_falls_back_when_tui_env_invalid(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "nope")
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )

    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config", lambda: {"platform_toolsets": {"cli": ["memory"]}}
    )

    from hermes_cli.tools_config import _RECENTLY_SHIPPED_TOOLSETS

    result = server._load_enabled_toolsets()
    assert result is not None
    assert {"memory", "project"} <= set(result)
    assert "kanban" not in result
    assert set(result) - {"memory", "project"} <= _RECENTLY_SHIPPED_TOOLSETS
    assert capsys.readouterr().err.strip()  # a fallback warning is printed


def test_load_enabled_toolsets_warns_when_config_fallback_fails(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "nope")
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )

    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    assert server._load_enabled_toolsets() is None
    assert capsys.readouterr().err.strip()  # a warning is printed


def test_load_enabled_toolsets_honors_builtin_env_if_config_fails(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "web")

    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    assert server._load_enabled_toolsets() == ["web"]


def test_load_enabled_toolsets_all_env_means_all(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "all")

    assert server._load_enabled_toolsets() is None


def test_load_disabled_toolsets_reads_agent_config(monkeypatch):
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {"agent": {"disabled_toolsets": ["browser"]}},
    )

    assert server._load_disabled_toolsets() == ["browser"]


def test_load_disabled_toolsets_none_when_unset_or_config_fails(monkeypatch):
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: {"agent": {"disabled_toolsets": []}})
    assert server._load_disabled_toolsets() is None

    monkeypatch.setattr(config_mod, "load_config", lambda: {})
    assert server._load_disabled_toolsets() is None

    monkeypatch.setattr(
        config_mod, "load_config", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert server._load_disabled_toolsets() is None




def test_load_enabled_toolsets_reports_disabled_mcp_separately(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "web,mcp-off,nope")
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )

    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "read_raw_config",
        lambda: {"mcp_servers": {"mcp-off": {"enabled": False}}},
    )

    assert server._load_enabled_toolsets() == ["web"]
    err = capsys.readouterr().err
    assert "nope" in err
    assert "mcp-off" in err


def test_history_to_messages_preserves_tool_calls_for_resume_display():
    history = [
        {"role": "user", "content": "first prompt"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "search_files",
                        "arguments": json.dumps({"pattern": "resume"}),
                    },
                }
            ],
        },
        {"role": "tool", "content": "{}", "tool_call_id": "call_1"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second prompt"},
    ]

    assert server._history_to_messages(history) == [
        {"role": "user", "text": "first prompt"},
        {
            "args": {"pattern": "resume"},
            "context": "resume",
            "name": "search_files",
            "role": "tool",
            "tool_call_id": history[2]["tool_call_id"],
        },
        {"role": "assistant", "text": "first answer"},
        {"role": "user", "text": "second prompt"},
    ]


def test_history_to_messages_types_the_failed_turn_boundary_for_resume():
    """Desktop keys the failed-turn boundary on ``display_kind`` (a room poller must not post it
    as the member's reply); rows written before the closer typed it are typed on read."""
    from agent.turn_failure_copy import FAILED_TURN_DISPLAY_KIND, FAILED_TURN_NOTICE, PARTIAL_FAILED_TURN_NOTICE

    history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": FAILED_TURN_NOTICE, "display_kind": FAILED_TURN_DISPLAY_KIND},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": PARTIAL_FAILED_TURN_NOTICE},  # legacy untyped row
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": f"Quoting Hermes: {FAILED_TURN_NOTICE}"},  # a real reply
    ]

    assert [m.get("display_kind") for m in server._history_to_messages(history)] == [
        None, FAILED_TURN_DISPLAY_KIND, None, FAILED_TURN_DISPLAY_KIND, None, None,
    ]


def test_history_to_messages_drops_pure_compaction_scaffolding():
    from agent.context_compressor import (
        HISTORICAL_TASK_HEADING,
        SUMMARY_PREFIX,
        _SUMMARY_END_MARKER,
    )

    summary = (
        f"{SUMMARY_PREFIX}\n\n"
        f"{HISTORICAL_TASK_HEADING}\nold work\n\n"
        f"{_SUMMARY_END_MARKER}"
    )

    assert server._history_to_messages(
        [
            {"role": "user", "content": summary},
            {"role": "assistant", "content": "real answer"},
        ]
    ) == [{"role": "assistant", "text": "real answer"}]


def test_history_to_messages_preserves_live_ask_without_compaction_scaffolding():
    from agent.context_compressor import (
        HISTORICAL_TASK_HEADING,
        SUMMARY_PREFIX,
        _SUMMARY_END_MARKER,
    )

    carrier = (
        f"{SUMMARY_PREFIX}\n\n"
        f"{HISTORICAL_TASK_HEADING}\nold work\n\n"
        f"{_SUMMARY_END_MARKER}\n\n"
        "test the browser controller"
    )

    assert server._history_to_messages(
        [
            {
                "role": "user",
                "content": carrier,
                "tool_calls": [{"id": "stale"}],
                "reasoning": "internal compaction reasoning",
            }
        ]
    ) == [{"role": "user", "text": "test the browser controller"}]


def test_history_to_messages_unwraps_merged_assistant_carrier():
    from agent.context_compressor import (
        HISTORICAL_TASK_HEADING,
        SUMMARY_PREFIX,
        _MERGED_PRIOR_CONTEXT_HEADER,
        _MERGED_SUMMARY_DELIMITER,
        _SUMMARY_END_MARKER,
    )

    carrier = (
        f"{_MERGED_PRIOR_CONTEXT_HEADER}\n"
        "real completed answer\n\n"
        f"{_MERGED_SUMMARY_DELIMITER}\n\n"
        f"{SUMMARY_PREFIX}\n\n"
        f"{HISTORICAL_TASK_HEADING}\nold work\n\n"
        f"{_SUMMARY_END_MARKER}"
    )

    assert server._history_to_messages(
        [
            {
                "role": "assistant",
                "content": carrier,
                "tool_calls": [{"id": "stale"}],
                "reasoning_details": [{"summary": "internal"}],
            }
        ]
    ) == [{"role": "assistant", "text": "real completed answer"}]


def test_history_to_messages_ships_full_tool_args():
    # This is the display projection. `context` is an 80-char preview for
    # collapsed row titles. A renderer that shows the full call (the expanded
    # `$` transcript in the desktop) rebuilds it from `args`. When the
    # projection dropped the args, the preview truncation was permanent.
    long_command = "echo " + "x" * 200
    history = [
        {"role": "user", "content": "run it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "terminal",
                        "arguments": json.dumps({"command": long_command}),
                    },
                }
            ],
        },
        {"role": "tool", "content": "{}", "tool_call_id": "call_1"},
    ]

    rows = server._history_to_messages(history)
    assert rows[1]["args"] == {"command": long_command}
    # The preview stays alongside for the collapsed title.
    assert rows[1]["context"]

    # A tool row with no recorded args keeps the old small shape.
    argless = server._history_to_messages(
        [{"role": "tool", "content": "{}", "tool_call_id": "missing"}]
    )
    assert "args" not in argless[0]


def test_tool_start_ships_full_args(monkeypatch):
    # The desktop rebuilds the expanded row's `$` transcript from args. When
    # only the 80-char `context` preview shipped, the expanded command was
    # truncated until tool.complete. tool.complete already ships full args to
    # every client, so tool.start does too. There is no per-client gate.
    events: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload: events.append((event_type, sid, payload))
    )
    long_command = "echo " + "y" * 200
    monkeypatch.setitem(
        server._sessions,
        "args-test",
        {"source": "desktop", "tool_progress_mode": "all", "tool_started_at": {}},
    )

    server._on_tool_start("args-test", "tool-1", "terminal", {"command": long_command})
    server._on_tool_start("args-test", "tool-2", "terminal", {})

    assert events[0][2]["args"] == {"command": long_command}
    # Empty args stay omitted. Argless tools get no noise key.
    assert "args" not in events[1][2]




def test_history_to_messages_keeps_reasoning_only_assistant_turn():
    # A thinking-only assistant turn (reasoning present, no visible text) is
    # persisted and recallable, but was dropped from the resumed session view
    # as "empty" -- so it vanished while the agent could still recall it from
    # the transcript. Keep it (with reasoning) so the desktop "Thinkingâ€¦"
    # disclosure renders. (#44022)
    history = [
        {"role": "user", "content": "think about this"},
        {"role": "assistant", "content": "", "reasoning": "step-by-step thoughts"},
        {"role": "assistant", "content": "here is the answer"},
    ]

    assert server._history_to_messages(history) == [
        {"role": "user", "text": "think about this"},
        {"role": "assistant", "text": "", "reasoning": "step-by-step thoughts"},
        {"role": "assistant", "text": "here is the answer"},
    ]


def test_history_to_messages_still_drops_empty_assistant_without_reasoning():
    # A genuinely empty assistant turn (no text, no reasoning, no tool calls)
    # remains filtered out -- the fix only spares reasoning-bearing turns.
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "reasoning": ""},
        {"role": "assistant", "content": "   "},
        {"role": "assistant", "content": "real reply"},
    ]

    assert server._history_to_messages(history) == [
        {"role": "user", "text": "hi"},
        {"role": "assistant", "text": "real reply"},
    ]


def test_history_to_messages_renders_multimodal_content():
    # bb/gui preserves image URLs in the resume payload so the desktop
    # renderer's extractEmbeddedImages can pull them back out and display
    # the actual image instead of a placeholder. This also keeps the
    # resume payload in sync with the cached message.
    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look here"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        },
        {"role": "assistant", "content": "saw it"},
    ]

    assert server._history_to_messages(history) == [
        {"role": "user", "text": "look here\ndata:image/png;base64,abc"},
        {"role": "assistant", "text": "saw it"},
    ]


def test_history_to_messages_strips_legacy_discord_triggering_note():
    # Rows written before the gateway persisted the authored text carry the model-facing
    # routing note in user ``content``; this projection heals them for TUI/web resume
    # (the desktop hydration strip is the same rule). Reply pointer and assistant rows are kept.
    from gateway.run_inbound import discord_triggering_note

    note = discord_triggering_note("123")
    history = [
        {"role": "user", "content": f"{note}\n\n[Replying to: hi]\nwhat is up"},
        {"role": "assistant", "content": f"echo: {note}"},
    ]

    assert server._history_to_messages(history) == [
        {"role": "user", "text": "[Replying to: hi]\nwhat is up"},
        {"role": "assistant", "text": f"echo: {note}"},
    ]


def test_history_to_messages_hides_gateway_system_markers():
    # Model-switch / personality notices are persisted as role=user [System: â€¦]
    # rows so strict providers accept them mid-history, but they are model-facing
    # metadata -- never a user turn. They must not render as a user bubble on any
    # surface, and dropping them from the display projection also stops the
    # stored marker from shifting the desktop's user-message ordinals and
    # duplicating the optimistic prompt (#67603).
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {
            "role": "user",
            "content": "[System: The active model for this chat has changed to k3.]",
        },
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
        {
            "role": "user",
            "content": (
                "[System: The user has changed the assistant's personality. "
                "Adopt the new persona going forward.]"
            ),
        },
    ]

    assert server._history_to_messages(history) == [
        {"role": "user", "text": "first question"},
        {"role": "assistant", "text": "first answer"},
        {"role": "user", "text": "second question"},
        {"role": "assistant", "text": "second answer"},
    ]


def test_history_to_messages_drops_display_hidden_scaffolding():
    # A mid-stream steer persists an interrupted-turn checkpoint. When nothing
    # reached the screen the row carries only model-facing scaffolding and is
    # marked display_kind="hidden"; the scaffolded bytes live in the server-only
    # api_content sidecar for provider replay. This projection -- the single
    # display source every client reads -- must drop the row by its declared
    # display_kind, not just the "[System:" string convention, or the raw
    # "[This response was interrupted by a user correction.]" paints as an
    # assistant bubble (and api_content must never ship to a client).
    history = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "[This response was interrupted by a user correction.]",
            "api_content": "[This response was interrupted by a user correction.]",
            "display_kind": "hidden",
        },
        {"role": "user", "content": "i love you"},
        {
            "role": "assistant",
            "content": "Love you too",
            "api_content": (
                "[This response was interrupted by a user correction.]\n\n"
                "Visible response before the interruption:\n\nLove you too"
            ),
        },
    ]

    projected = server._history_to_messages(history)

    assert projected == [
        {"role": "user", "text": "go"},
        {"role": "user", "text": "i love you"},
        {"role": "assistant", "text": "Love you too"},
    ]
    # Server-only sidecar never crosses the wire.
    assert all("api_content" not in m for m in projected)


def test_history_to_messages_projects_a_skill_turn_to_its_invocation():
    # A /skill invocation is persisted EXPANDED: the activation note plus the
    # entire skill body. That payload is model-facing scaffolding -- this
    # projection is the single display source every client reads, so it must
    # hand back the invocation the user typed and never the body. Without it a
    # chat bubble renders the whole skill as if the user had written it.
    scaffolded = (
        '[IMPORTANT: The user has invoked the "work" skill, indicating they '
        "want you to follow its instructions. The full skill content is "
        "loaded below.]\n\n"
        "# /work\n\nSPIN UP A WORKTREE, never the primary checkout.\n\n"
        "The user has provided the following instruction alongside the skill "
        "invocation: fix the title leak"
    )

    history = [
        {"role": "user", "content": scaffolded},
        {"role": "assistant", "content": "on it"},
    ]

    assert server._history_to_messages(history) == [
        {
            "role": "user",
            "text": "/work fix the title leak",
            "display_kind": "skill_invocation",
        },
        {"role": "assistant", "text": "on it"},
    ]


def test_history_to_messages_projects_a_bare_skill_turn_to_the_command():
    scaffolded = (
        '[IMPORTANT: The user has invoked the "work" skill, indicating they '
        "want you to follow its instructions. The full skill content is "
        "loaded below.]\n\n# /work\n\nSPIN UP A WORKTREE."
    )

    assert server._history_to_messages([{"role": "user", "content": scaffolded}]) == [
        {"role": "user", "text": "/work", "display_kind": "skill_invocation"}
    ]


def test_expand_skill_invocation_for_replay_round_trips_the_projection(
    tmp_path, monkeypatch
):
    # Rewind/regenerate replays a turn from what the transcript SHOWS, and a
    # skill turn shows its invocation. Re-running that verbatim would send the
    # agent the literal "/work fix it" instead of the skill, so the server
    # re-expands it â€” the exact inverse of _skill_scaffold_projection, with the
    # body never leaving the server.
    import agent.skill_commands as skill_commands
    import agent.skill_utils as skill_utils
    import tools.skills_tool as skills_tool

    skills_dir = tmp_path / "skills"
    (skills_dir / "worktree-kickoff").mkdir(parents=True)
    (skills_dir / "worktree-kickoff" / "SKILL.md").write_text(
        "---\nname: worktree-kickoff\ndescription: Spin up a worktree\n---\n\n"
        "# kickoff\n\nSPIN UP A WORKTREE, never the primary checkout.\n"
    )
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda *a, **k: [])
    monkeypatch.setattr(skill_commands, "_skill_commands", {})
    monkeypatch.setattr(skill_commands, "_skill_commands_platform", None)
    skill_commands.scan_skill_commands()

    expanded = server._expand_skill_invocation_for_replay(
        "/worktree-kickoff fix it", "task-1"
    )

    assert "SPIN UP A WORKTREE" in expanded
    assert server._skill_scaffold_projection(expanded) == "/worktree-kickoff fix it"


def test_expand_skill_invocation_for_replay_leaves_ordinary_text_alone(monkeypatch):
    import agent.skill_commands as skill_commands
    import agent.skill_utils as skill_utils

    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda *a, **k: [])
    monkeypatch.setattr(skill_commands, "_skill_commands", {})
    monkeypatch.setattr(skill_commands, "_skill_commands_platform", None)

    assert server._expand_skill_invocation_for_replay("just words", "t") == "just words"
    # A core slash command is not a skill â€” nothing to expand.
    assert server._expand_skill_invocation_for_replay("/status", "t") == "/status"


def _two_repo_project_skill_sessions(tmp_path, monkeypatch) -> tuple[Path, Path]:
    """Two trusted repos (``alpha-skill`` / ``beta-skill``) bound to sessions ``sid-a`` / ``sid-b``, in a
    launch shape whose process cwd and TERMINAL_CWD both point at a non-project dir."""
    import agent.skill_commands as skill_commands
    import agent.skill_utils as skill_utils
    import tools.skills_tool as skills_tool

    def repo(name: str, skill: str) -> Path:
        r = tmp_path / name
        (r / ".git").mkdir(parents=True)
        (r / ".hermes" / "skills" / skill).mkdir(parents=True)
        (r / ".hermes" / "skills" / skill / "SKILL.md").write_text(
            f"---\nname: {skill}\ndescription: from {name}\n---\n\n# {skill}\n\nBODY OF {skill.upper()}\n")
        return r

    repo_a, repo_b = repo("proj-a", "alpha-skill"), repo("proj-b", "beta-skill")
    home = tmp_path / "hermes-home"
    (home / "skills").mkdir(parents=True)
    (home / "config.yaml").write_text(
        f"skills:\n  external_dirs: []\n  trusted_project_dirs: ['{repo_a}', '{repo_b}']\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", home / "skills")
    monkeypatch.setattr(skill_utils, "_skills_cfg", lambda: {
        "external_dirs": [], "trusted_project_dirs": [str(repo_a), str(repo_b)]})
    skill_utils._external_dirs_cache_clear()
    monkeypatch.setattr(skill_commands, "_skill_commands", {})
    monkeypatch.setattr(skill_commands, "_skill_commands_platform", None)
    # Launch shape: process cwd and TERMINAL_CWD both point at a non-project dir (the resolved placeholder).
    elsewhere = tmp_path / "home-dir"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("TERMINAL_CWD", str(elsewhere))
    monkeypatch.setattr(server, "_sessions", {
        "sid-a": {"session_key": "key-a", "cwd": str(repo_a)},
        "sid-b": {"session_key": "key-b", "cwd": str(repo_b)}})
    return repo_a, repo_b


def test_command_dispatch_and_catalog_resolve_project_skills_from_the_session_cwd(tmp_path, monkeypatch):
    # TUI/desktop: the RPC thread has no session context and the terminal scope / process env resolve a
    # placeholder ``terminal.cwd`` to $HOME, so project skills of the session's repo never registered and
    # ``/<name>`` died with "not a quick/plugin/bundle/skill command". Two sessions in two trusted repos
    # in ONE process must each catalog and dispatch their own repo's skill (the cached registry is
    # keyed by project root, not just platform + home).
    import agent.skill_utils as skill_utils

    _two_repo_project_skill_sessions(tmp_path, monkeypatch)
    for sid, own, other in (("sid-a", "alpha-skill", "beta-skill"), ("sid-b", "beta-skill", "alpha-skill")):
        catalog = server._methods["commands.catalog"]("c", {"session_id": sid})["result"]
        assert f"/{own}" in catalog["skills"] and f"/{other}" not in catalog["skills"]
        res = server._methods["command.dispatch"]("d", {"name": own, "arg": "go", "session_id": sid})
        assert res["result"]["type"] == "skill", res
        assert f"BODY OF {own.upper()}" in res["result"]["message"]
        miss = server._methods["command.dispatch"]("m", {"name": other, "arg": "", "session_id": sid})
        assert miss["error"]["code"] == 4018
    # Nothing leaks past the RPC: the thread's logical cwd is unbound again.
    assert skill_utils.find_project_root() is None


def test_complete_slash_and_skills_reload_are_bound_to_the_session_cwd(tmp_path, monkeypatch):
    # The '/' popup and /reload-skills ran the registry unbound: the popup never offered a project skill
    # ``command.dispatch`` accepts, and a rescan after that dispatch reported the session's project skills
    # as "Removed" and republished a registry without them.
    import agent.skill_commands as skill_commands

    _two_repo_project_skill_sessions(tmp_path, monkeypatch)
    items = server._methods["complete.slash"]("s", {"text": "/alph", "session_id": "sid-a"})["result"]["items"]
    assert [i["text"] for i in items if i["kind"] == "skill"] == ["alpha-skill"]
    assert server._methods["command.dispatch"]("d", {"name": "alpha-skill", "arg": "", "session_id": "sid-a"})[
        "result"]["type"] == "skill"
    reload = server._methods["skills.reload"]("r", {"session_id": "sid-a"})["result"]
    assert reload["result"]["removed"] == [] and "/alpha-skill" in skill_commands._skill_commands, reload["output"]
    # Another session's reload resolves ITS repo, not the launch env.
    other = server._methods["skills.reload"]("r", {"session_id": "sid-b"})["result"]
    assert {i["name"] for i in other["result"]["added"]} == {"beta-skill"}, other["output"]


def test_history_to_messages_types_a_legacy_auto_continue_row():
    # A crash-interrupted turn used to be typed only AFTER it finished, so a
    # turn killed a second time (or any row written before turn-start typing
    # landed) sits in the DB untyped and painted the raw recovery note as a
    # user bubble. The projection recognizes the synthetic note's fixed
    # prefix so those rows still read as a timeline event.
    history = [
        {"role": "user", "content": "keep going"},
        {"role": "user", "content": server._auto_continue_note("keep going")},
    ]

    projected = server._history_to_messages(history)

    assert projected == [
        {"role": "user", "text": "keep going"},
        {
            "role": "user",
            "text": server._auto_continue_note("keep going"),
            "display_kind": "auto_continue",
        },
    ]


def test_history_to_messages_keeps_real_user_bracket_text():
    # Only role=user rows whose text OPENS with the [System: marker sentinel are
    # bookkeeping notices. A genuine user turn that merely mentions the token is
    # a real message and stays visible.
    history = [
        {"role": "user", "content": "why does [System: ...] show up in my chat?"},
        {"role": "assistant", "content": "it should not"},
    ]

    assert server._history_to_messages(history) == [
        {"role": "user", "text": "why does [System: ...] show up in my chat?"},
        {"role": "assistant", "text": "it should not"},
    ]


@pytest.mark.parametrize("omit_messages", [False, True])
def test_session_resume_uses_parent_lineage_for_display(monkeypatch, omit_messages):
    captured = {}
    target = "tip-omit" if omit_messages else "tip-full"

    class FakeDB:
        def get_session(self, target):
            return {"id": target}

        def reopen_session(self, target):
            captured["reopened"] = target

        def get_resume_conversations(self, session_id):
            return (
                self.get_messages_as_conversation(session_id, repair_alternation=True),
                self.get_messages_as_conversation(session_id, include_ancestors=True),
            )

        def get_ancestor_display_prefix(self, _sid):
            return []

        def get_messages_as_conversation(
            self,
            target,
            include_ancestors=False,
            repair_alternation=False,
            include_row_ids=False,
            **_kwargs,
        ):
            captured.setdefault("history_calls", []).append(
                (target, include_ancestors, include_row_ids)
            )
            return (
                [
                    {"role": "user", "content": "root prompt"},
                    {"role": "assistant", "content": "root answer"},
                ]
                if include_ancestors
                else [{"role": "user", "content": "tip prompt"}]
            )

    monkeypatch.setattr(server, "_get_db", lambda: FakeDB())
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_set_session_context", lambda target: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda tokens: None)
    monkeypatch.setattr(
        server,
        "_make_agent",
        lambda *args, **kwargs: types.SimpleNamespace(model="test"),
    )
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda agent, *a: {"model": "test", "tools": {}, "skills": {}},
    )
    monkeypatch.setattr(
        server, "_init_session", lambda sid, key, agent, history, cols=80, **_kwargs: None
    )
    # The deferred pre-warm timer is neutered module-wide by the autouse
    # _neuter_agent_prewarm_timer fixture; this test only asserts the
    # returned display history.

    params = {"session_id": target}
    if omit_messages:
        params["omit_messages"] = True
    resp = server.handle_request(
        {"id": "1", "method": "session.resume", "params": params}
    )

    expected = [] if omit_messages else [
        {"role": "user", "text": "root prompt"},
        {"role": "assistant", "text": "root answer"},
    ]
    assert resp["result"]["messages"] == expected
    assert resp["result"]["message_count"] == (1 if omit_messages else 2)
    assert resp["result"]["messages_omitted"] is omit_messages




def test_live_visible_history_falls_back_without_db_or_key():
    in_memory = [{"role": "user", "content": "hi"}]
    # No DB handle available.
    assert server._live_visible_history({"session_key": "s"}, None, in_memory) == in_memory

    # DB available but the session has no persist key yet.
    class DB:
        def get_messages_as_conversation(self, *a, **k):  # pragma: no cover - not reached
            raise AssertionError("must not query without a session_key")

    assert server._live_visible_history({}, DB(), in_memory) == in_memory


def test_live_visible_history_falls_back_when_db_empty():
    """A brand-new live session whose first turn hasn't been flushed keeps its
    in-memory history rather than rendering empty."""
    in_memory = [{"role": "user", "content": "fresh turn not flushed yet"}]

    class EmptyDB:
        def get_messages_as_conversation(self, *a, **k):
            return []

    assert server._live_visible_history({"session_key": "s"}, EmptyDB(), in_memory) == in_memory


def test_live_visible_history_falls_back_when_db_raises():
    in_memory = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]

    class BrokenDB:
        def get_messages_as_conversation(self, *a, **k):
            raise RuntimeError("db exploded")

    assert server._live_visible_history({"session_key": "s"}, BrokenDB(), in_memory) == in_memory


def test_live_visible_history_keeps_candidate_and_fresh_tail():
    """The hard case: the persisted candidate (missing from in-memory) AND a
    not-yet-flushed live turn (missing from the DB) must BOTH survive."""
    # Persisted display: has the verification candidate, lags the newest turn.
    db_display = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "long substantive answer",
         "finish_reason": "verification_required"},
        {"role": "assistant", "content": "terse verified reply"},
    ]
    # In-memory model history: candidate collapsed out, but has a fresh turn 2.
    in_memory = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "terse verified reply"},
        {"role": "user", "content": "turn 2 not flushed"},
        {"role": "assistant", "content": "turn 2 reply not flushed"},
    ]

    class DB:
        def get_messages_as_conversation(self, key, include_ancestors=False, repair_alternation=False, **_kwargs):
            return list(db_display)

    result = server._live_visible_history({"session_key": "s1"}, DB(), in_memory)
    assert result == [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "long substantive answer",
         "finish_reason": "verification_required"},
        {"role": "assistant", "content": "terse verified reply"},
        {"role": "user", "content": "turn 2 not flushed"},
        {"role": "assistant", "content": "turn 2 reply not flushed"},
    ]


def test_reconcile_display_with_live_trusts_db_when_tail_absent():
    """If the DB tail isn't in memory (DB ahead / diverged), don't duplicate â€”
    serve the persisted display."""
    db_display = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    in_memory = [{"role": "user", "content": "unrelated"}]
    assert server._reconcile_display_with_live(db_display, in_memory) == db_display
    assert server._reconcile_display_with_live([], in_memory) == in_memory
    assert server._reconcile_display_with_live(db_display, []) == db_display


def test_live_visible_history_matches_eager_resume_with_real_db(tmp_path):
    """E2E cross-builder consistency against a real SessionDB.

    A persisted verification candidate (finish_reason=verification_required)
    is collapsed out of the model history by repair_message_sequence but kept
    in the display lineage (#65919). The warm/live projection
    (_live_visible_history) must equal the eager session.resume display
    projection â€” both keeping the candidate â€” so switching to a live session
    shows the same substantive answer a cold resume would.
    """
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="tui")
    db.append_message("s1", role="user", content="do the thing")
    db.append_message(
        "s1", role="assistant", content="long substantive answer",
        finish_reason="verification_required",
    )
    db.append_message(
        "s1", role="assistant", content="terse verified reply", finish_reason="stop",
    )

    model_history, display_history = db.get_resume_conversations("s1")

    # The divergence #65919 introduced: candidate absent from the model
    # projection, present in the display projection.
    assert not any("long substantive" in (m.get("content") or "") for m in model_history)
    assert any("long substantive" in (m.get("content") or "") for m in display_history)

    # Eager session.resume serves the display projection.
    eager_messages = server._history_to_messages(display_history)
    # Warm/live reuse: in-memory history is the collapsed model projection.
    live_history = server._live_visible_history({"session_key": "s1"}, db, list(model_history))
    # They must agree â€” the candidate survives the warm switch.
    assert server._history_to_messages(live_history) == eager_messages
    assert any(m.get("text") == "long substantive answer" for m in eager_messages)


def test_live_visible_history_keeps_candidate_and_new_flushed_turn_real_db(tmp_path):
    """Real-DB variant of the combined case: a candidate from turn 1 AND a
    fully-flushed turn 2 both appear once."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="tui")
    db.append_message("s1", role="user", content="turn 1")
    db.append_message(
        "s1", role="assistant", content="candidate answer",
        finish_reason="verification_required",
    )
    db.append_message("s1", role="assistant", content="verified reply", finish_reason="stop")
    db.append_message("s1", role="user", content="turn 2")
    db.append_message("s1", role="assistant", content="turn 2 reply", finish_reason="stop")

    model_history, display_history = db.get_resume_conversations("s1")
    live_history = server._live_visible_history({"session_key": "s1"}, db, list(model_history))
    texts = [m.get("text") for m in server._history_to_messages(live_history)]

    assert texts == [
        "turn 1",
        "candidate answer",
        "verified reply",
        "turn 2",
        "turn 2 reply",
    ]


def test_live_session_payload_reads_profile_db_not_launch_db(monkeypatch, tmp_path):
    """Warm/live reuse for a non-launch profile session must open that
    profile's state.db, not the process launch DB.

    App-global remote mode stores verification candidates in the resumed
    profile's DB. ``_live_session_payload`` previously hard-coded
    ``_get_db()`` (launch), so the display projection missed those rows and
    fell back to collapsed in-memory model history â€” while eager
    ``session.resume`` against the same profile still showed them.
    """
    from hermes_state import SessionDB

    launch_home = tmp_path / "launch"
    profile_home = tmp_path / "profile"
    launch_home.mkdir()
    profile_home.mkdir()

    launch_db = SessionDB(db_path=launch_home / "state.db")
    profile_db = SessionDB(db_path=profile_home / "state.db")
    profile_db.create_session("s-profile", source="tui")
    profile_db.append_message("s-profile", role="user", content="do the thing")
    profile_db.append_message(
        "s-profile",
        role="assistant",
        content="long substantive answer",
        finish_reason="verification_required",
    )
    profile_db.append_message(
        "s-profile",
        role="assistant",
        content="terse verified reply",
        finish_reason="stop",
    )
    model_history, display_history = profile_db.get_resume_conversations("s-profile")
    assert not any("long substantive" in (m.get("content") or "") for m in model_history)
    assert any("long substantive" in (m.get("content") or "") for m in display_history)

    session = {
        "session_key": "s-profile",
        "profile_home": str(profile_home),
        "agent": None,
        "history": list(model_history),
        "display_history_prefix": [],
        "history_lock": threading.Lock(),
        "created_at": 1.0,
        "last_active": 1.0,
        "running": False,
    }
    # Launch DB has no row for this session â€” the pre-fix path would miss
    # candidates and fall back to collapsed in-memory history.
    monkeypatch.setattr(server, "_get_db", lambda: launch_db)

    payload = server._live_session_payload("live1", session, touch=False)
    texts = [m.get("text") for m in payload.get("messages") or []]

    assert "long substantive answer" in texts
    assert texts == [m.get("text") for m in server._history_to_messages(display_history)]


def test_lazy_child_watch_resume_serves_candidate_inclusive_display(monkeypatch, tmp_path):
    """The delegated-child watch-window cold resume (lazy=True) must serve the
    verbatim display projection so a persisted verification candidate is not
    collapsed out of the watch window (#65919 sibling of the warm-payload fix).
    """
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("child1", source="tui")
    db.append_message("child1", role="user", content="child prompt")
    db.append_message(
        "child1", role="assistant", content="child substantive answer",
        finish_reason="verification_required",
    )
    db.append_message(
        "child1", role="assistant", content="child terse reply", finish_reason="stop",
    )

    lease = types.SimpleNamespace(session_id="child1", release=lambda: None)

    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(
        server, "_claim_active_session_slot", lambda *a, **k: (lease, None)
    )
    monkeypatch.setattr(
        server, "_deferred_session_record", lambda *a, **k: {"created_at": 123.0}
    )
    monkeypatch.setattr(server, "_claim_or_reuse_live", lambda *a, **k: None)
    monkeypatch.setattr(server, "_child_run_active", lambda *a, **k: False)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *a, **k: None)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "session.resume",
            "params": {"session_id": "child1", "lazy": True},
        }
    )

    assert "error" not in resp, resp
    texts = [m.get("text") for m in resp["result"]["messages"]]
    assert "child substantive answer" in texts
    assert texts == ["child prompt", "child substantive answer", "child terse reply"]


def test_session_resume_deferred_history_acknowledges_and_reuses(monkeypatch):
    history_started = threading.Event()
    release_history = threading.Event()
    build_started = threading.Event()
    history_calls = []
    auto_continue_calls = []
    ancestor = {"role": "assistant", "content": "ancestor"}
    loaded = {"role": "user", "content": "loaded"}

    class FakeDB:
        def get_session(self, target):
            return {"id": target, "message_count": 1200}

        def resolve_resume_session_id(self, target):
            return target

        def reopen_session(self, target):
            assert target == "large-session"

        def get_resume_conversations(self, target):
            history_calls.append(("resume", target))
            history_started.set()
            assert release_history.wait(timeout=2.0)
            return [loaded], [ancestor, loaded]

        def get_ancestor_display_prefix(self, target):
            history_calls.append(("prefix", target))
            return [ancestor]

    monkeypatch.setattr(server, "_get_db", lambda: FakeDB())
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(
        server,
        "_start_agent_build",
        lambda _sid, _session: build_started.set(),
    )
    monkeypatch.setattr(
        server,
        "_maybe_schedule_auto_continue",
        lambda sid, session, stored_id: auto_continue_calls.append(
            (sid, session, stored_id)
        ),
    )

    try:
        first = server._methods["session.resume"](
            "r1",
            {
                "session_id": "large-session",
                "source": "desktop",
                "defer_history": True,
            },
        )

        assert first["result"]["hydrating"] is True
        assert first["result"]["messages"] == []
        assert first["result"]["message_count"] == 1200
        assert history_started.wait(timeout=1.0)

        second = server._methods["session.resume"](
            "r2",
            {"session_id": "large-session", "defer_history": True},
        )
        assert second["result"]["session_id"] == first["result"]["session_id"]
        assert second["result"]["hydrating"] is True
        assert second["result"]["messages"] == []

        release_history.set()
        sid = first["result"]["session_id"]
        assert server._sessions[sid]["resume_history_ready"].wait(timeout=1.0)
        assert build_started.wait(timeout=1.0)
        assert history_calls == [
            ("resume", "large-session"),
            ("prefix", "large-session"),
        ]
        assert server._sessions[sid]["history"] == [loaded]
        assert server._sessions[sid]["display_history_prefix"] == [ancestor]
        assert server._sessions[sid]["resume_message_count"] == 2
        assert auto_continue_calls == [(sid, server._sessions[sid], "large-session")]
    finally:
        release_history.set()
        for sid, session in list(server._sessions.items()):
            if session.get("session_key") == "large-session":
                lease = session.get("active_session_lease")
                if lease is not None:
                    lease.release()
                server._sessions.pop(sid, None)


def test_session_resume_deferred_history_failure_can_retry(monkeypatch):
    first_released = threading.Event()
    build_started = threading.Event()
    attempts = 0

    class FakeDB:
        def get_session(self, target):
            return {"id": target, "message_count": 1}

        def resolve_resume_session_id(self, target):
            return target

        def reopen_session(self, _target):
            pass

        def get_resume_conversations(self, _target):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                first_released.set()
                raise RuntimeError("sqlite read failed")
            loaded = [{"role": "user", "content": "retry loaded"}]
            return loaded, loaded

        def get_ancestor_display_prefix(self, _target):
            return []

    monkeypatch.setattr(server, "_get_db", lambda: FakeDB())
    monkeypatch.setattr(
        server,
        "_claim_active_session_slot",
        lambda *_args, **_kwargs: pytest.fail(
            "resume must not claim a session slot before the first prompt"
        ),
    )
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(
        server,
        "_start_agent_build",
        lambda _sid, _session: build_started.set(),
    )

    try:
        first = server._methods["session.resume"](
            "r1",
            {"session_id": "retry-session", "defer_history": True},
        )
        first_sid = first["result"]["session_id"]
        assert first_released.wait(timeout=1.0)
        assert first_sid not in server._sessions

        second = server._methods["session.resume"](
            "r2",
            {"session_id": "retry-session", "defer_history": True},
        )
        second_sid = second["result"]["session_id"]
        assert second_sid != first_sid
        assert server._sessions[second_sid]["resume_history_ready"].wait(timeout=1.0)
        assert build_started.wait(timeout=1.0)
    finally:
        for sid, session in list(server._sessions.items()):
            if session.get("session_key") == "retry-session":
                lease = session.get("active_session_lease")
                if lease is not None:
                    lease.release()
                server._sessions.pop(sid, None)


def test_session_resume_deferred_history_close_cancels_build(monkeypatch):
    history_started = threading.Event()
    release_history = threading.Event()
    build_started = threading.Event()

    class FakeDB:
        def get_session(self, target):
            return {"id": target, "message_count": 1}

        def resolve_resume_session_id(self, target):
            return target

        def reopen_session(self, _target):
            pass

        def get_resume_conversations(self, _target):
            history_started.set()
            assert release_history.wait(timeout=2.0)
            loaded = [{"role": "user", "content": "late"}]
            return loaded, loaded

        def get_ancestor_display_prefix(self, _target):
            return []

    monkeypatch.setattr(server, "_get_db", lambda: FakeDB())
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(
        server,
        "_start_agent_build",
        lambda _sid, _session: build_started.set(),
    )

    response = {}
    try:
        response = server._methods["session.resume"](
            "r1",
            {"session_id": "cancel-session", "defer_history": True},
        )
        sid = response["result"]["session_id"]
        session = server._sessions[sid]
        assert history_started.wait(timeout=1.0)

        assert server._close_session_by_id(sid, end_reason="tui_close") is True
        assert session["resume_history_ready"].is_set()
        assert session["resume_history_error"] == "session resume cancelled"

        release_history.set()
        time.sleep(0.05)
        assert not build_started.is_set()
        assert sid not in server._sessions
    finally:
        release_history.set()
        server._sessions.pop(response.get("result", {}).get("session_id", ""), None)


def test_session_resume_follows_compression_tip(monkeypatch, tmp_path):
    """Resuming a rotated-out parent id must load the continuation's messages.

    Regression for the desktop "I came back and the reply isn't there" report:
    auto-compression ends the live session and forks a continuation child, so a
    resume on the parent id (the desktop's routed id when the chat was opened
    before it rotated) used to reload the pre-compression transcript and drop
    the response generated after compression. session.resume must follow the
    compression tip via resolve_resume_session_id.
    """
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    base = int(time.time()) - 10_000
    db.create_session("parent_root", source="tui")
    db.append_message(
        "parent_root", role="user", content="pre-compression turn",
        timestamp=base + 10,
    )
    db.end_session("parent_root", "compression")
    db.create_session("cont_tip", source="tui", parent_session_id="parent_root")
    db.append_message(
        "cont_tip", role="assistant", content="post-compression reply",
        timestamp=base + 110,
    )
    conn = db._conn
    assert conn is not None
    conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = 'parent_root'",
        (base, base + 50),
    )
    conn.execute("UPDATE sessions SET started_at = ? WHERE id = 'cont_tip'", (base + 100,))
    conn.commit()

    captured = {}

    def fake_make_agent(sid, key, session_id=None, session_db=None, **kwargs):
        # Record only the FIRST (synchronous, eager) build. A stray background
        # build leaked from an earlier test's deferred resume could otherwise
        # overwrite this with its own session_id and corrupt the assertion.
        captured.setdefault("agent_session_id", session_id)
        return types.SimpleNamespace(model="test", provider="test")

    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_set_session_context", lambda target, cwd=None: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda tokens: None)
    monkeypatch.setattr(server, "_make_agent", fake_make_agent)
    monkeypatch.setattr(
        server, "_session_info", lambda agent, *a: {"model": "test", "tools": {}, "skills": {}}
    )
    monkeypatch.setattr(
        server, "_init_session", lambda sid, key, agent, history, cols=80, **_kwargs: None
    )

    try:
        # eager_build: this asserts the synchronously-built agent binds to the
        # resolved tip (captured["agent_session_id"]); the compression-tip
        # resolution itself runs before the build and is mode-agnostic.
        resp = server.handle_request(
            {"id": "1", "method": "session.resume", "params": {"session_id": "parent_root", "eager_build": True}}
        )
    finally:
        db.close()

    # The agent must bind to the continuation tip, and the returned transcript
    # must include the post-compression reply (which lives only in the tip).
    assert resp["result"]["session_key"] == "cont_tip"
    assert captured["agent_session_id"] == "cont_tip"
    texts = [m.get("text") for m in resp["result"]["messages"]]
    assert "post-compression reply" in texts


def test_session_resume_passes_stored_runtime_to_agent(monkeypatch):
    captured = {}

    class FakeDB:
        def get_session(self, target):
            return {
                "id": target,
                "model": "gpt-5.4",
                "billing_provider": "openai-codex",
                "model_config": '{"reasoning_config":{"enabled":true,"effort":"high"},"service_tier":"priority","base_url":"https://custom.example/v1","api_mode":"chat_completions"}',
            }

        def reopen_session(self, target):
            pass

        def get_resume_conversations(self, session_id):
            return (
                self.get_messages_as_conversation(session_id, repair_alternation=True),
                self.get_messages_as_conversation(session_id, include_ancestors=True),
            )

        def get_ancestor_display_prefix(self, _sid):
            return []

        def get_messages_as_conversation(self, target, include_ancestors=False, repair_alternation=False, **_kwargs):
            return [{"role": "user", "content": "hello"}]

    def fake_make_agent(sid, key, session_id=None, session_db=None, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(model="gpt-5.4", provider="openai-codex")

    monkeypatch.setattr(server, "_get_db", lambda: FakeDB())
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_set_session_context", lambda target, cwd=None: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda tokens: None)
    monkeypatch.setattr(server, "_make_agent", fake_make_agent)
    monkeypatch.setattr(server, "_session_info", lambda agent, *a: {"model": agent.model, "provider": agent.provider})

    def fake_init_session(sid, key, agent, history, cols=80, **_kwargs):
        server._sessions[sid] = {"agent": agent, "session_key": key}

    monkeypatch.setattr(server, "_init_session", fake_init_session)

    # eager_build: this asserts the synchronous build contract (stored runtime
    # overrides reach _make_agent, info comes from _session_info). The deferred
    # default restores the same overrides via _start_agent_build off-thread.
    resp = server.handle_request(
        {"id": "1", "method": "session.resume", "params": {"session_id": "stored-session", "eager_build": True}}
    )

    assert captured["model_override"] == {
        "model": "gpt-5.4",
        "provider": "openai-codex",
        "base_url": "https://custom.example/v1",
        "api_mode": "chat_completions",
    }
    assert captured["provider_override"] == "openai-codex"
    assert captured["reasoning_config_override"] == {"enabled": True, "effort": "high"}
    assert captured["service_tier_override"] == "priority"
    runtime_sid = resp["result"]["session_id"]
    assert server._sessions[runtime_sid]["model_override"] == captured["model_override"]


def test_session_resume_profile_uses_profile_db_cwd(monkeypatch, tmp_path):
    target = "stored-profile-session"
    launch_cwd = tmp_path / "launch"
    profile_cwd = tmp_path / "worker"
    profile_home = tmp_path / "profiles" / "worker"
    launch_cwd.mkdir()
    profile_cwd.mkdir()
    profile_home.mkdir(parents=True)
    captured = {}

    class ProfileDB:
        def get_session(self, _target):
            return {"id": target, "cwd": str(profile_cwd)}

        def get_session_by_title(self, _target):
            return None

        def reopen_session(self, _target):
            captured["reopened"] = _target

        def get_resume_conversations(self, session_id):
            return (
                self.get_messages_as_conversation(session_id, repair_alternation=True),
                self.get_messages_as_conversation(session_id, include_ancestors=True),
            )

        def get_ancestor_display_prefix(self, _sid):
            return []

        def get_messages_as_conversation(self, _target, include_ancestors=False, repair_alternation=False, **_kwargs):
            return [{"role": "user", "content": "hello"}]

        def update_session_cwd(self, *_args):
            raise AssertionError("profile row already has cwd")

    class LaunchDB:
        def get_session(self, _target):
            return {"id": target, "cwd": str(launch_cwd)}

        def update_session_cwd(self, *_args):
            captured["launch_update"] = True

    profile_db = ProfileDB()
    launch_db = LaunchDB()

    class FakeWorker:
        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    def fake_make_agent(sid, key, session_id=None, session_db=None, **kwargs):
        captured["agent_db"] = session_db
        captured["agent_cwd"] = kwargs.get("cwd_override")
        return types.SimpleNamespace(model="test/model")

    monkeypatch.setenv("TERMINAL_CWD", str(launch_cwd))
    monkeypatch.setattr(server, "_profile_home", lambda _profile: profile_home)
    monkeypatch.setattr("hermes_state_registry.acquire", lambda db_path=None: profile_db)
    monkeypatch.setattr(server, "_get_db", lambda: launch_db)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(
        server,
        "_set_session_context",
        lambda target, cwd=None: captured.setdefault("context_cwd", cwd) or [],
    )
    monkeypatch.setattr(server, "_clear_session_context", lambda tokens: None)
    monkeypatch.setattr(server, "_make_agent", fake_make_agent)
    monkeypatch.setattr(server, "_SlashWorker", FakeWorker)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, session=None: {"cwd": session.get("cwd") if session else ""},
    )

    import tools.approval as approval

    monkeypatch.setattr(approval, "register_gateway_notify", lambda key, cb: None)
    monkeypatch.setattr(approval, "load_permanent_allowlist", lambda: None)

    try:
        # eager_build: asserts the synchronous build receives the profile's db
        # (the deferred default builds with the same db via _start_agent_build).
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.resume",
                "params": {"session_id": target, "profile": "worker", "eager_build": True},
            }
        )

        assert "error" not in resp
        sid = resp["result"]["session_id"]
        assert captured["agent_db"] is profile_db
        assert captured["context_cwd"] == str(profile_cwd)
        assert captured["agent_cwd"] == str(profile_cwd)
        assert server._sessions[sid]["cwd"] == str(profile_cwd)
        assert resp["result"]["info"]["cwd"] == str(profile_cwd)
        assert "launch_update" not in captured
    finally:
        server._sessions.clear()


def test_session_cwd_set_profile_session_updates_profile_db(monkeypatch, tmp_path):
    target = "stored-profile-session"
    profile_home = tmp_path / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    new_cwd = tmp_path / "new-workspace"
    new_cwd.mkdir()
    captured = {}

    class ProfileDB:
        def update_session_cwd(self, session_id, cwd, git_branch=None, git_repo_root=None):
            captured["profile_update"] = (session_id, cwd)

        def close(self):
            captured["profile_closed"] = True

    class LaunchDB:
        def update_session_cwd(self, *_args):
            captured["launch_update"] = True

    profile_db = ProfileDB()

    import tools.terminal_tool_lifecycle as terminal_tool_lifecycle

    monkeypatch.setattr("hermes_state_registry.acquire", lambda db_path=None: profile_db)
    monkeypatch.setattr(server, "_get_db", lambda: LaunchDB())
    monkeypatch.setattr(terminal_tool_lifecycle, "cleanup_vm", lambda _key: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)

    session = {"session_key": target, "profile_home": str(profile_home)}
    assert server._set_session_cwd(session, str(new_cwd)) == str(new_cwd)
    assert session["cwd"] == str(new_cwd)
    assert session["explicit_cwd"] is True
    assert captured["profile_update"] == (target, str(new_cwd))
    assert captured["profile_closed"] is True
    assert "launch_update" not in captured


def test_stored_session_runtime_overrides_skips_bare_billing_provider(monkeypatch):
    """A bare billing bucket ("custom"/"auto") must not be restored as the provider
    identity on resume. A custom endpoint that never used `/model` persists only
    `billing_provider="custom"`; restoring that broke `session.resume` with "No LLM provider
    configured" (agent_init treats it as non-routable). ``"openrouter"`` is NOT a bare bucket
    â€” it is a fully routable provider; see #57588. A real provider, or an explicit
    `model_config.provider`, is still restored.
    """
    # Bare "custom" bucket, no explicit model_config.provider: no provider override restored.
    ov = server._stored_session_runtime_overrides({"model": "my-model", "billing_provider": "custom"})
    assert "provider_override" not in ov
    assert ov["model_override"]["provider"] is None

    for bare in ("auto", "custom"):
        ov = server._stored_session_runtime_overrides({"model": "m", "billing_provider": bare})
        assert "provider_override" not in ov

    # A real provider in billing_provider is still restored.
    ov = server._stored_session_runtime_overrides({"model": "m", "billing_provider": "anthropic"})
    assert ov["provider_override"] == "anthropic"
    assert ov["model_override"]["provider"] == "anthropic"

    # An explicit ROUTABLE provider in model_config wins over the bare billing
    # bucket. It must actually resolve in the registry â€” a stale/renamed
    # provider is dropped (see TestStaleProviderNameFallsBack).
    cfg = {
        "custom_providers": [
            {
                "name": "myendpoint",
                "base_url": "https://myendpoint.invalid/v1",
                "api_key": "sk-test",
                "model": "m",
            }
        ]
    }
    import hermes_cli.runtime_provider as rp

    monkeypatch.setattr(rp, "load_config", lambda: cfg)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    ov = server._stored_session_runtime_overrides(
        {"model": "m", "billing_provider": "custom", "model_config": {"provider": "custom:myendpoint"}}
    )
    assert ov["provider_override"] == "custom:myendpoint"
    assert ov["model_override"]["provider"] == "custom:myendpoint"


def test_stored_session_runtime_overrides_restores_explicit_normal_tier():
    overrides = server._stored_session_runtime_overrides(
        {
            "model": "gpt-5.4",
            "model_config": {"service_tier": "normal"},
        }
    )

    assert "service_tier_override" in overrides
    assert overrides["service_tier_override"] == ""


def test_openrouter_session_resume_restores_provider():
    """OpenRouter is a fully routable provider â€” sessions that used OpenRouter must
    restore the "openrouter" provider override on resume, not fall through to whatever
    the current global model is.  (#57588)
    """
    # OpenRouter session with no explicit model_config.provider (the common case
    # for sessions that never used /model): billing_provider="openrouter" should
    # be restored as the provider override.
    ov = server._stored_session_runtime_overrides(
        {"model": "anthropic/claude-opus-4.8", "billing_provider": "openrouter"}
    )
    assert ov["provider_override"] == "openrouter"
    assert ov["model_override"]["provider"] == "openrouter"
    assert ov["model_override"]["model"] == "anthropic/claude-opus-4.8"

    # When an explicit model_config.provider exists, it takes precedence over
    # billing_provider (this path was already correct).
    ov = server._stored_session_runtime_overrides(
        {
            "model": "anthropic/claude-opus-4.8",
            "billing_provider": "openrouter",
            "model_config": {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1"},
        }
    )
    assert ov["provider_override"] == "openrouter"


def test_persist_live_session_runtime_preserves_resume_metadata(monkeypatch):
    updates = {}

    class FakeDB:
        def get_session(self, session_id):
            assert session_id == "stored-session"
            return {"model_config": '{"_branched_from":"root"}'}

        def update_session_meta(self, session_id, model_config_json, model=None):
            updates["meta"] = (session_id, json.loads(model_config_json), model)

    agent = types.SimpleNamespace(
        model="gpt-5.4",
        provider="openai-codex",
        base_url="https://custom.example/v1",
        api_mode="chat_completions",
        reasoning_config={"enabled": True, "effort": "high"},
        service_tier="priority",
        _session_db=FakeDB(),
    )

    server._persist_live_session_runtime({"agent": agent, "session_key": "stored-session"})

    assert "model" not in updates
    assert updates["meta"] == (
        "stored-session",
        {
            "_branched_from": "root",
            "model": "gpt-5.4",
            "provider": "openai-codex",
            "base_url": "https://custom.example/v1",
            "api_mode": "chat_completions",
            "reasoning_config": {"enabled": True, "effort": "high"},
            "service_tier": "priority",
        },
        "gpt-5.4",
    )


def test_persist_live_session_runtime_preserves_explicit_normal_tier():
    updates = {}

    class FakeDB:
        def get_session(self, _session_id):
            return {"model_config": '{"service_tier":"priority"}'}

        def update_session_meta(self, _session_id, model_config_json, model=None):
            updates["config"] = json.loads(model_config_json)

    agent = types.SimpleNamespace(
        model="gpt-5.4",
        provider="openai-codex",
        base_url=None,
        api_mode=None,
        reasoning_config=None,
        service_tier="",
        _session_db=FakeDB(),
    )

    server._persist_live_session_runtime(
        {
            "agent": agent,
            "session_key": "stored-session",
            "create_service_tier_override": "",
        }
    )

    assert updates["config"]["service_tier"] == "normal"


def test_status_callback_emits_kind_and_text():
    with patch("tui_gateway.server._emit") as emit:
        cb = server._agent_cbs("sid")["status_callback"]
        cb("context_pressure", "85% to compaction")

    emit.assert_called_once_with(
        "status.update",
        "sid",
        {"kind": "context_pressure", "text": "85% to compaction"},
    )


def test_status_callback_accepts_single_message_argument():
    with patch("tui_gateway.server._emit") as emit:
        cb = server._agent_cbs("sid")["status_callback"]
        cb("thinking...")

    emit.assert_called_once_with(
        "status.update",
        "sid",
        {"kind": "status", "text": "thinking..."},
    )


def test_resolve_model_uses_inference_model_env(monkeypatch):
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.setenv("HERMES_INFERENCE_MODEL", " anthropic/claude-sonnet-4.6\n")

    assert server._resolve_model() == "anthropic/claude-sonnet-4.6"


def test_resolve_model_strips_config_model(monkeypatch):
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    monkeypatch.setattr(
        server, "_load_cfg", lambda: {"model": {"default": " nous/hermes-test "}}
    )

    assert server._resolve_model() == "nous/hermes-test"


def _sync_test_session(**extra):
    session = {
        "agent": types.SimpleNamespace(model="old/model"),
        "session_key": "session-key",
    }
    session.update(extra)
    return session


def _patch_config_model(monkeypatch, model, provider=""):
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    cfg_model = {"default": model}
    if provider:
        cfg_model["provider"] = provider
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": cfg_model})


def test_config_sync_switches_unpinned_session(monkeypatch):
    _patch_config_model(monkeypatch, "new/model", provider="nous")
    session = _sync_test_session(config_model_seen=("old/model", "nous"))
    calls = []
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda sid, sess, raw, **kw: calls.append((sid, raw, kw)),
    )

    server._sync_agent_model_with_config("sid", session)

    assert calls == [
        (
            "sid",
            "new/model --provider nous",
            {
                "confirm_expensive_model": True,
                "pin_session_override": False,
                "persist_override": False,
            },
        )
    ]
    assert session["config_model_seen"] == ("new/model", "nous")


def test_config_sync_treats_auto_provider_as_unset(monkeypatch):
    _patch_config_model(monkeypatch, "new/model", provider="auto")
    session = _sync_test_session(config_model_seen=("old/model", ""))
    calls = []
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda sid, sess, raw, **kw: calls.append(raw),
    )

    server._sync_agent_model_with_config("sid", session)

    assert calls == ["new/model"]


def test_config_sync_skips_session_pinned_by_model_command(monkeypatch):
    _patch_config_model(monkeypatch, "new/model")
    session = _sync_test_session(
        config_model_seen=("old/model", ""),
        model_override={"model": "pinned/model"},
    )
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda *a, **k: pytest.fail("pinned session must not be switched"),
    )

    server._sync_agent_model_with_config("sid", session)


def test_config_sync_noop_when_config_unchanged(monkeypatch):
    _patch_config_model(monkeypatch, "old/model")
    session = _sync_test_session(config_model_seen=("old/model", ""))
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda *a, **k: pytest.fail("unchanged config must not switch"),
    )

    server._sync_agent_model_with_config("sid", session)


def test_config_sync_adopts_baseline_when_agent_already_on_target(monkeypatch):
    # Branched/resumed sessions reach their first sync with no snapshot but
    # an agent already built from config; that must not trigger a switch.
    _patch_config_model(monkeypatch, "old/model")
    session = _sync_test_session()
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda *a, **k: pytest.fail("agent already on target must not switch"),
    )

    server._sync_agent_model_with_config("sid", session)

    assert session["config_model_seen"] == ("old/model", "")


def test_config_sync_switches_when_only_provider_differs(monkeypatch):
    _patch_config_model(monkeypatch, "old/model", provider="nous")
    session = _sync_test_session(config_model_seen=("old/model", ""))
    calls = []
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda sid, sess, raw, **kw: calls.append(raw),
    )

    server._sync_agent_model_with_config("sid", session)

    assert calls == ["old/model --provider nous"]


def test_config_sync_failure_emits_error_once_per_edit(monkeypatch):
    _patch_config_model(monkeypatch, "broken/model")
    session = _sync_test_session(config_model_seen=("old/model", ""))

    def boom(*a, **k):
        raise ValueError("no such model")

    monkeypatch.setattr(server, "_apply_model_switch", boom)
    emits = []
    monkeypatch.setattr(
        server, "_emit", lambda ev, sid, payload: emits.append((ev, payload))
    )

    server._sync_agent_model_with_config("sid", session)
    server._sync_agent_model_with_config("sid", session)

    assert len(emits) == 1
    assert emits[0][0] == "error"
    assert "broken/model" in emits[0][1]["message"]


def test_config_sync_config_wins_over_env_seed(monkeypatch):
    # Hosted instances set HERMES_INFERENCE_MODEL as a provision-time seed;
    # the per-turn sync must follow config.yaml edits, not stay pinned to it.
    monkeypatch.setenv("HERMES_INFERENCE_MODEL", "seed/model")
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"default": "new/model"}})
    session = _sync_test_session(config_model_seen=("seed/model", ""))
    calls = []
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda sid, sess, raw, **kw: calls.append(raw),
    )

    server._sync_agent_model_with_config("sid", session)

    assert calls == ["new/model"]
    assert session["config_model_seen"] == ("new/model", "")


def test_config_sync_ignores_env_seed_without_config_model(monkeypatch):
    # `hermes --tui -m <model>` sets HERMES_MODEL/HERMES_INFERENCE_MODEL as a
    # launch-scoped seed. When config.yaml has NO model.default (typical
    # custom-provider-only setup), the sync must NOT adopt the env seed as a
    # config target â€” doing so replayed the -m flag as a /model switch and
    # (with persist_switch_by_default=True) wrote it into config.yaml
    # permanently.
    monkeypatch.setenv("HERMES_MODEL", "one-shot/model")
    monkeypatch.setenv("HERMES_INFERENCE_MODEL", "one-shot/model")
    monkeypatch.setattr(
        server, "_load_cfg", lambda: {"model": {"provider": "custom:mylocal"}}
    )
    session = _sync_test_session()
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda *a, **k: pytest.fail("env seed must not trigger a config sync switch"),
    )

    server._sync_agent_model_with_config("sid", session)




def test_apply_model_switch_persist_override_false_never_persists(monkeypatch):
    # Internal callers (config sync, /moa one-shot + restore) pass
    # persist_override=False; even with persist_switch_by_default=True the
    # switch must not write config.yaml.
    import types as _types

    result = _types.SimpleNamespace(
        success=True,
        new_model="new/model",
        target_provider="nous",
        base_url="",
        api_key="key",
        api_mode="chat_completions",
        warning_message="",
        model_info=None,
        error_message="",
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model", lambda **kw: result
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.resolve_persist_behavior",
        lambda *a: pytest.fail("persist_override must bypass resolve_persist_behavior"),
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.persist_model_selection",
        lambda _r: pytest.fail("persist_override=False must not persist"),
    )
    monkeypatch.setattr(
        "hermes_cli.model_cost_guard.expensive_model_warning",
        lambda *a, **k: None,
    )
    session = {"agent": None}

    out = server._apply_model_switch(
        "sid", session, "new/model --provider nous", persist_override=False
    )

    assert out["value"] == "new/model"
    assert session["model_override"]["model"] == "new/model"


def test_startup_runtime_uses_tui_provider_env(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL", "nous/hermes-test")
    monkeypatch.setenv("HERMES_TUI_PROVIDER", "nous")
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)

    assert server._resolve_startup_runtime() == ("nous/hermes-test", "nous")


def test_startup_runtime_does_not_treat_inference_provider_as_explicit(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL", "nous/hermes-test")
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "nous")
    monkeypatch.setattr(
        "hermes_cli.models.detect_static_provider_for_model",
        lambda model, provider: None,
    )

    assert server._resolve_startup_runtime() == ("nous/hermes-test", None)




def test_load_fallback_model_merges_chain_providers_first(monkeypatch):
    # Parity with HermesCLI / gateway: fallback_providers stays first and keeps
    # its order, with any distinct legacy fallback_model entry merged in after
    # (deduped on provider/model/base_url).
    fallback_chain = [
        {"provider": "openrouter", "model": "openai/gpt-5.5"},
        {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    ]
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "fallback_model": {"provider": "legacy", "model": "legacy-model"},
            "fallback_providers": fallback_chain,
        },
    )

    assert server._load_fallback_model() == [
        {"provider": "openrouter", "model": "openai/gpt-5.5"},
        {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        {"provider": "legacy", "model": "legacy-model"},
    ]


def test_make_agent_passes_configured_fallback_chain(monkeypatch):
    captured = {}
    fallback_chain = [
        {"provider": "openrouter", "model": "openai/gpt-5.5"},
    ]

    def fake_agent(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(model=kwargs.get("model"))

    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "model": {"default": "gpt-5.5", "provider": "openai-codex"},
            "fallback_providers": fallback_chain,
        },
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, target_model=None: {
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "token",
            "api_mode": "codex_responses",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr("run_agent.AIAgent", fake_agent)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: ["file"])
    monkeypatch.setattr(server, "_get_db", lambda: None)

    agent = server._make_agent("sid", "session-key")

    assert agent.model == "gpt-5.5"
    assert captured["fallback_model"] == fallback_chain
    assert captured["platform"] == "tui"


def test_make_agent_forwards_agent_disabled_toolsets(monkeypatch):
    """``agent.disabled_toolsets`` must reach AIAgent in gateway sessions too: only the AIAgent
    filter strips a toolset out of composite defaults (``hermes-cli``), which the gateway's
    enabled-list resolver can't reach into (#44499)."""
    captured = _capture_make_agent_kwargs(monkeypatch)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: ["file"])
    monkeypatch.setattr(server, "_load_disabled_toolsets", lambda: ["browser"])

    server._make_agent("sid", "session-key")

    assert captured["disabled_toolsets"] == ["browser"]


def test_make_agent_disabled_toolsets_none_by_default(monkeypatch):
    captured = _capture_make_agent_kwargs(monkeypatch)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: ["file"])
    monkeypatch.setattr(server, "_load_disabled_toolsets", lambda: None)

    server._make_agent("sid", "session-key")

    assert captured["disabled_toolsets"] is None


def _capture_make_agent_kwargs(monkeypatch) -> dict:
    """Stub AIAgent so ``server._make_agent`` records the kwargs it was built with."""
    captured = {}

    def fake_agent(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(model=kwargs.get("model"))

    _setup_make_agent_mocks(monkeypatch, {})
    monkeypatch.setattr("run_agent.AIAgent", fake_agent)
    return captured


@pytest.mark.parametrize("identity, user_id", [
    ({"user_id": "oidc|abc123", "provider": "oidc"}, "oidc:oidc|abc123"),
    ({"user_id": "alice", "provider": "basic"}, "basic:alice"),
    ({"user_id": "alice", "provider": "oidc"}, "oidc:alice"),
    (None, None),
    ({"user_id": "server-internal", "provider": "server-internal"}, None),
    ({"user_id": "", "provider": "oidc"}, None),
    ({"user_id": "abc", "provider": ""}, None),
])
def test_make_agent_passes_the_authenticated_dashboard_user_as_user_id(monkeypatch, identity, user_id):
    """WSTransport.auth_identity reaches the agent as ``user_id`` prefixed with the login provider, so a
    basic-auth ``alice`` and an OIDC ``alice`` stay two memory peers. The legacy token, stdio and the PTY
    child's server-internal credential name no human, so the agent gets no user id for them."""
    captured = _capture_make_agent_kwargs(monkeypatch)
    transport = types.SimpleNamespace(auth_identity=identity)
    monkeypatch.setitem(server._sessions, "sid-auth", {"session_key": "k", "transport": transport})

    server._make_agent("sid-auth", "k")

    assert captured["user_id"] == user_id


def test_make_agent_passes_no_user_id_for_an_unknown_session(monkeypatch):
    captured = _capture_make_agent_kwargs(monkeypatch)
    server._sessions.pop("sid-missing", None)

    server._make_agent("sid-missing", "k")

    assert captured["user_id"] is None


class _LoginSocket:
    """A live WS peer carrying the identity the upgrade auth minted."""

    def __init__(self, user_id="alice", provider="basic"):
        self.auth_identity = {"provider": provider, "user_id": user_id}
        self._closed = False

    def write(self, frame):
        return True


def _login_session(monkeypatch, sid, key, transport, **extra) -> dict:
    """A record the way session.create stamps it: the login lives on the record, not the transport slot."""
    record = {"session_key": key, "transport": transport,
              "auth_user_id": server._transport_auth_user_id(transport), **extra}
    monkeypatch.setitem(server._sessions, sid, record)
    return record


def test_make_agent_keeps_the_login_after_a_second_window_attaches_and_detaches(monkeypatch):
    """A pop-out turns the transport slot into a FanoutTransport, which names no login. The rebuild must read the
    login the record was created with, before and after the pop-out goes away."""
    captured = _capture_make_agent_kwargs(monkeypatch)
    first, popout = _LoginSocket(), _LoginSocket()
    record = _login_session(monkeypatch, "sid-popout", "k", first)

    assert server._attach_session_transport(record, popout)
    server._make_agent("sid-popout", "k")
    assert captured["user_id"] == "basic:alice"

    server._detach_session_transport(record, popout)
    server._make_agent("sid-popout", "k")
    assert captured["user_id"] == "basic:alice"


def test_build_branch_agent_carries_the_parent_login(monkeypatch, tmp_path):
    """The branch agent is built before its record exists, so the parent's login is passed to the build and
    copied onto the new record."""
    captured = _capture_make_agent_kwargs(monkeypatch)
    socket = _LoginSocket()
    parent = _login_session(monkeypatch, "sid-parent", "parent-key", socket, cwd=str(tmp_path), source="desktop")

    def fake_init_session(sid, key, agent, history, **kwargs):
        monkeypatch.setitem(server._sessions, sid, {"session_key": key, "transport": server._stdio_transport})

    monkeypatch.setattr(server, "_set_session_context", lambda key, cwd=None: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda tokens: None)
    monkeypatch.setattr(server, "_init_session", fake_init_session)
    monkeypatch.setattr(server, "_transfer_db_to_agent", lambda *args: False)
    token = bind_transport(socket)
    try:
        server._build_branch_agent(parent, "sid-branch", "branch-key", [], "desktop")
    finally:
        reset_transport(token)

    assert captured["user_id"] == "basic:alice"
    assert server._sessions["sid-branch"]["auth_user_id"] == "basic:alice"


def test_deferred_session_record_stamps_the_creating_login():
    token = bind_transport(_LoginSocket("carol", "oidc"))
    try:
        record = server._deferred_session_record(
            "deferred-key", cols=80, cwd="/tmp", history=[], lease=None)
    finally:
        reset_transport(token)

    assert record["auth_user_id"] == "oidc:carol"
    assert server._session_auth_user_id(record) == "oidc:carol"


def test_compute_host_turn_frame_carries_the_session_login(monkeypatch):
    record = _login_session(monkeypatch, "sid-host", "host-key", _LoginSocket(), history=[],
                            history_lock=threading.Lock(), cwd="/tmp", cols=80)
    monkeypatch.setattr(server, "_session_cwd", lambda session: "/tmp")

    frame = server._compute_host_turn_frame("rid", "sid-host", record, "hello")

    assert frame["auth_user_id"] == "basic:alice"


def test_attaching_a_different_login_keeps_the_creator_and_warns_once(monkeypatch):
    """Ownership is not enforced. The record keeps the creator's login, and the attach is logged once with both
    ids rather than on every prompt the second client sends."""
    creator, other = _LoginSocket("alice"), _LoginSocket("bob")
    record = _login_session(monkeypatch, "sid-shared", "shared-key", creator)

    assert server._attach_session_transport(record, other)
    assert server._attach_session_transport(record, other)

    assert server._session_auth_user_id(record) == "basic:alice"




def test_background_agent_kwargs_preserves_full_fallback_chain(monkeypatch):
    chain = [
        {"provider": "openrouter", "model": "openai/gpt-5.5"},
        {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    ]
    agent = types.SimpleNamespace(
        model="gpt-5.5",
        provider="openai-codex",
        _fallback_chain=chain,
    )
    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_turns": 25})
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: ["file"])
    monkeypatch.setattr(server, "_get_db", lambda: None)

    kwargs = server._background_agent_kwargs(agent, "task-id")

    assert kwargs["fallback_model"] == chain


def test_background_agent_kwargs_preserves_empty_fallback_chain(monkeypatch):
    agent = types.SimpleNamespace(
        model="gpt-5.5",
        provider="anthropic",
        _fallback_chain=[],
    )
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "max_turns": 25,
            "fallback_providers": [
                {"provider": "openrouter", "model": "openai/gpt-5.5"},
            ],
        },
    )
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: ["file"])
    monkeypatch.setattr(server, "_get_db", lambda: None)

    kwargs = server._background_agent_kwargs(agent, "task-id")

    assert kwargs["fallback_model"] == []


def test_background_agent_kwargs_forwards_agent_disabled_toolsets(monkeypatch):
    agent = types.SimpleNamespace(
        model="gpt-5.5",
        provider="anthropic",
        _fallback_chain=[],
        disabled_toolsets=["browser"],
    )
    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_turns": 25})
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: ["file"])
    monkeypatch.setattr(server, "_get_db", lambda: None)

    kwargs = server._background_agent_kwargs(agent, "task-id")

    assert kwargs["disabled_toolsets"] == ["browser"]


def test_background_agent_kwargs_falls_back_to_config_disabled_toolsets(monkeypatch):
    agent = types.SimpleNamespace(
        model="gpt-5.5",
        provider="anthropic",
        _fallback_chain=[],
    )
    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_turns": 25})
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: ["file"])
    monkeypatch.setattr(server, "_load_disabled_toolsets", lambda: ["browser"])
    monkeypatch.setattr(server, "_get_db", lambda: None)

    kwargs = server._background_agent_kwargs(agent, "task-id")

    assert kwargs["disabled_toolsets"] == ["browser"]


def test_startup_runtime_resolves_short_alias_without_network(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL", "sonnet")
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"provider": "auto"}})
    monkeypatch.setattr(
        "hermes_cli.models.fetch_openrouter_models",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("network lookup should not run")
        ),
    )

    model, provider = server._resolve_startup_runtime()

    assert provider == "anthropic"
    assert model.startswith("claude-sonnet")


def test_startup_runtime_does_not_call_network_detector(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL", "sonnet")
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"provider": "auto"}})
    monkeypatch.setattr(
        "hermes_cli.models.detect_provider_for_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("network detector called")
        ),
    )

    model, provider = server._resolve_startup_runtime()

    assert model
    assert provider in {None, "anthropic"}


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        **extra,
    }


def test_session_close_commits_memory_and_fires_finalize_hook(monkeypatch):
    calls = {"hooks": []}

    agent = types.SimpleNamespace(session_id="session-key")
    agent.commit_memory_session = lambda history: calls.setdefault("history", history)
    server._sessions["sid"] = _session(
        agent=agent, history=[{"role": "user", "content": "hello"}]
    )
    monkeypatch.setattr(
        server,
        "_notify_session_boundary",
        lambda event, session_id, *_args: calls["hooks"].append((event, session_id)),
    )

    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.close", "params": {"session_id": "sid"}}
        )
        assert resp["result"]["closed"] is True
        assert calls["history"] == [{"role": "user", "content": "hello"}]
        assert ("on_session_finalize", "session-key") in calls["hooks"]
    finally:
        server._sessions.pop("sid", None)


def test_session_close_releases_resume_lock_before_slow_teardown(monkeypatch):
    """One slow session finalizer must not stall unrelated session.resume RPCs."""
    teardown_started = threading.Event()
    release_teardown = threading.Event()
    response = {}

    def _slow_teardown(_session, *, end_reason="tui_close"):
        assert end_reason == "tui_close"
        teardown_started.set()
        assert release_teardown.wait(timeout=2.0)

    monkeypatch.setattr(server, "_teardown_session", _slow_teardown)
    server._sessions["slow-close"] = _session()

    def _close():
        response.update(
            server.handle_request(
                {
                    "id": "close",
                    "method": "session.close",
                    "params": {"session_id": "slow-close"},
                }
            )
        )

    thread = threading.Thread(target=_close)
    thread.start()
    acquired = False
    try:
        assert teardown_started.wait(timeout=1.0)
        assert "slow-close" not in server._sessions
        acquired = server._session_resume_lock.acquire(timeout=0.2)
        assert acquired, "slow teardown kept the global resume lock held"
    finally:
        if acquired:
            server._session_resume_lock.release()
        release_teardown.set()
        thread.join(timeout=2.0)
        server._sessions.pop("slow-close", None)

    assert not thread.is_alive()
    assert response["result"] == {"closed": True}


def test_session_close_settles_active_turn_before_teardown(monkeypatch):
    """Close must not tear down agent resources while their turn is unwinding."""
    turn_started = threading.Event()
    release_turn = threading.Event()
    teardown_started = threading.Event()
    response = {}

    def _turn():
        turn_started.set()
        assert release_turn.wait(timeout=2.0)

    def _teardown(_session, *, end_reason="tui_close"):
        if end_reason == "tui_close":
            teardown_started.set()

    session = _session()
    run_thread = threading.Thread(target=_turn)
    session["_run_thread"] = run_thread
    server._sessions["settle-close"] = session
    monkeypatch.setattr(server, "_teardown_session", _teardown)
    monkeypatch.setattr(
        server, "_TURN_SETTLE_BEFORE_CLOSE_SECONDS", 1.0, raising=False
    )

    close_thread = threading.Thread(
        target=lambda: response.update(
            server.handle_request(
                {
                    "id": "close",
                    "method": "session.close",
                    "params": {"session_id": "settle-close"},
                }
            )
        )
    )
    run_thread.start()
    close_thread.start()
    try:
        assert turn_started.wait(timeout=1.0)
        assert not teardown_started.wait(timeout=0.1)
        release_turn.set()
        close_thread.join(timeout=2.0)
    finally:
        release_turn.set()
        run_thread.join(timeout=2.0)
        close_thread.join(timeout=2.0)
        server._sessions.pop("settle-close", None)

    assert not close_thread.is_alive()
    assert teardown_started.is_set()
    assert response["result"] == {"closed": True}


def test_ws_orphan_reap_interrupts_isolated_turn_then_reaps(monkeypatch):
    callbacks = []
    interrupted = []
    torn_down = []

    class _Timer:
        def __init__(self, _delay, callback):
            callbacks.append(callback)
            self.daemon = False

        def start(self):
            return None

    class _Supervisor:
        def interrupt(self, sid, *, request_id=None):
            interrupted.append((sid, request_id))

    session = _session(
        agent=None,
        agent_ready=threading.Event(),
        transport=server._detached_ws_transport,
        running=True,
        _compute_host_active=True,
        history=[{"role": "assistant", "content": "partial"}],
        queued_prompt={"text": "must not run"},
    )
    server._sessions["isolated-sid"] = session
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(
        server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}}
    )
    monkeypatch.setattr(
        server, "_get_compute_host_supervisor", lambda _cfg=None: _Supervisor()
    )
    monkeypatch.setattr(
        server,
        "_teardown_popped_session",
        lambda claimed, *, end_reason: torn_down.append((claimed, end_reason)) or True,
    )

    try:
        server._schedule_ws_orphan_reap("isolated-sid")
        callbacks.pop(0)()

        assert interrupted == [("isolated-sid", "client-gone-isolated-sid")]
        assert session["_turn_cancel_requested"] is True
        assert session["queued_prompt"] is None
        assert session["history"] == [{"role": "assistant", "content": "partial"}]
        assert len(callbacks) == 1

        callbacks.pop(0)()

        assert interrupted == [("isolated-sid", "client-gone-isolated-sid")]
        assert len(callbacks) == 1

        session["running"] = False
        callbacks.pop(0)()

        assert "isolated-sid" not in server._sessions
        assert torn_down == [(session, "ws_orphan_reap")]
    finally:
        server._sessions.pop("isolated-sid", None)


def test_ws_orphan_reap_spares_turn_reattached_within_grace(monkeypatch):
    callbacks = []
    interrupted = []

    class _Timer:
        def __init__(self, _delay, callback):
            callbacks.append(callback)

        def start(self):
            return None

    class _LiveThread:
        def is_alive(self):
            return True

    class _LiveTransport:
        def write(self, *_args, **_kwargs):
            return True

    disconnecting_transport = _LiveTransport()
    session = _session(
        agent=types.SimpleNamespace(
            interrupt=lambda: interrupted.append("interrupted")
        ),
        transport=disconnecting_transport,
        running=True,
        _run_thread=_LiveThread(),
    )
    server._sessions["reattached-sid"] = session
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})

    try:
        server._close_sessions_for_transport(disconnecting_transport)
        assert session["transport"] is server._detached_ws_transport

        session["transport"] = _LiveTransport()
        callbacks.pop(0)()

        assert interrupted == []
        assert "reattached-sid" in server._sessions
        assert callbacks == []
    finally:
        server._sessions.pop("reattached-sid", None)


def test_session_resume_does_not_rebind_after_client_gone_interrupt_claim(monkeypatch):
    class _DB:
        def get_session(self, session_id):
            assert session_id == "stored-sid"
            return {"id": session_id, "cwd": "/tmp"}

        def resolve_resume_session_id(self, session_id):
            return session_id

    live_transport = object()
    session = _session(
        session_key="stored-sid",
        transport=server._detached_ws_transport,
        running=True,
        _client_gone_interrupt_requested=True,
    )
    server._sessions["live-sid"] = session
    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "current_transport", lambda: live_transport)

    try:
        response = server.handle_request(
            {
                "id": "resume-after-claim",
                "method": "session.resume",
                "params": {"session_id": "stored-sid"},
            }
        )

        assert response is not None
        assert response["error"]["code"] == 4009
        assert session["transport"] is server._detached_ws_transport
    finally:
        server._sessions.pop("live-sid", None)


def test_ws_orphan_reap_defers_running_turn_for_active_delegation(monkeypatch):
    callbacks = []
    interrupted = []
    delegation_active = iter((True, False, False))

    class _Timer:
        def __init__(self, _delay, callback):
            callbacks.append(callback)

        def start(self):
            return None

    class _LiveThread:
        def is_alive(self):
            return True

    def _interrupt():
        interrupted.append("interrupted")
        session["running"] = False

    session = _session(
        agent=types.SimpleNamespace(interrupt=_interrupt),
        transport=server._detached_ws_transport,
        running=True,
        _run_thread=_LiveThread(),
    )
    server._sessions["delegating-turn"] = session
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(
        server,
        "_session_has_active_delegations",
        lambda *_args, **_kwargs: next(delegation_active),
    )
    monkeypatch.setattr(server, "_teardown_popped_session", lambda *_args, **_kwargs: True)

    try:
        server._schedule_ws_orphan_reap("delegating-turn")
        callbacks.pop(0)()

        assert interrupted == []
        assert len(callbacks) == 1

        callbacks.pop(0)()

        assert interrupted == ["interrupted"]
        assert len(callbacks) == 1

        callbacks.pop(0)()
        assert "delegating-turn" not in server._sessions
    finally:
        server._sessions.pop("delegating-turn", None)


def test_ws_orphan_reap_interrupts_in_process_turn(monkeypatch):
    callbacks = []
    interrupted = []

    class _Timer:
        def __init__(self, _delay, callback):
            callbacks.append(callback)

        def start(self):
            return None

    class _LiveThread:
        def is_alive(self):
            return True

    def _interrupt():
        interrupted.append("interrupted")
        session["running"] = False

    session = _session(
        agent=types.SimpleNamespace(interrupt=_interrupt),
        transport=server._detached_ws_transport,
        running=True,
        _run_thread=_LiveThread(),
    )
    server._sessions["inline-sid"] = session
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})

    try:
        server._schedule_ws_orphan_reap("inline-sid")
        callbacks.pop(0)()

        assert interrupted == ["interrupted"]
        assert session["_turn_cancel_requested"] is True
        assert len(callbacks) == 1
    finally:
        server._sessions.pop("inline-sid", None)


def test_ws_disconnect_running_sidecar_still_closes_without_orphan_timer(monkeypatch):
    closed = []
    scheduled = []
    transport = object()
    server._sessions["sidecar-sid"] = _session(
        transport=transport,
        running=True,
        close_on_disconnect=True,
    )
    monkeypatch.setattr(
        server,
        "_teardown_popped_session",
        lambda session, *, end_reason: closed.append((session["_sid"], end_reason)) or True,
    )
    monkeypatch.setattr(
        server, "_schedule_ws_orphan_reap", lambda sid: scheduled.append(sid)
    )

    try:
        reaped, detached = server._close_sessions_for_transport(transport)

        assert (reaped, detached) == (1, 0)
        assert closed == [("sidecar-sid", "ws_disconnect")]
        assert scheduled == []
    finally:
        server._sessions.pop("sidecar-sid", None)




def test_ws_orphan_reap_releases_resume_lock_before_slow_teardown(monkeypatch):
    """Grace reaping claims under the lock but finalizes after releasing it."""
    scheduled = {}
    teardown_started = threading.Event()
    release_teardown = threading.Event()

    class _Timer:
        def __init__(self, _delay, callback):
            scheduled["callback"] = callback

        def start(self):
            return None

    def _slow_teardown(_session, *, end_reason="tui_close"):
        assert end_reason == "ws_orphan_reap"
        teardown_started.set()
        assert release_teardown.wait(timeout=10.0)

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_teardown_session", _slow_teardown)
    server._sessions["slow-orphan"] = _session(
        transport=server._detached_ws_transport,
        running=False,
    )

    server._schedule_ws_orphan_reap("slow-orphan")
    thread = threading.Thread(target=scheduled["callback"])
    thread.start()
    acquired = False
    try:
        assert teardown_started.wait(timeout=10.0)
        assert "slow-orphan" not in server._sessions
        acquired = server._session_resume_lock.acquire(timeout=2.0)
        assert acquired, "orphan teardown kept the global resume lock held"
    finally:
        if acquired:
            server._session_resume_lock.release()
        release_teardown.set()
        thread.join(timeout=10.0)
        server._sessions.pop("slow-orphan", None)

    assert not thread.is_alive()


def test_ws_orphan_reap_reschedules_while_mid_turn_then_reaps(monkeypatch):
    """A detached session that is still running must keep the reap timer (#85578)."""
    callbacks = []
    torn_down = []

    class _Timer:
        def __init__(self, _delay, callback):
            callbacks.append(callback)
            self.daemon = False

        def start(self):
            return None

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(
        server,
        "_teardown_session",
        lambda session, *, end_reason="tui_close": torn_down.append(
            (session, end_reason)
        ),
    )
    live = _session(
        transport=server._detached_ws_transport,
        running=True,
    )
    server._sessions["midturn-sid"] = live

    try:
        server._schedule_ws_orphan_reap("midturn-sid")
        callbacks.pop(0)()

        assert "midturn-sid" in server._sessions
        assert len(callbacks) == 1
        assert torn_down == []

        live["running"] = False
        callbacks.pop(0)()

        assert "midturn-sid" not in server._sessions
        assert len(torn_down) == 1
        assert torn_down[0][1] == "ws_orphan_reap"
    finally:
        server._sessions.pop("midturn-sid", None)


def test_ws_orphan_reap_waits_for_active_delegation_then_reaps(monkeypatch):
    from tools import async_delegation

    callbacks = []
    torn_down = []
    delegation_id = "deleg_ws_orphan_reap_test"

    class _Timer:
        def __init__(self, _delay, callback):
            callbacks.append(callback)
            self.daemon = False

        def start(self):
            return None

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(
        server,
        "_teardown_session",
        lambda session, *, end_reason="tui_close": torn_down.append(
            (session, end_reason)
        ),
    )
    server._sessions["delegating-sid"] = _session(
        transport=server._detached_ws_transport,
        running=False,
    )
    with async_delegation._records_lock:
        async_delegation._records[delegation_id] = {
            "status": "running",
            "origin_ui_session_id": "delegating-sid",
        }

    try:
        server._schedule_ws_orphan_reap("delegating-sid")
        callbacks.pop(0)()

        assert "delegating-sid" in server._sessions
        assert len(callbacks) == 1
        assert torn_down == []

        with async_delegation._records_lock:
            async_delegation._records[delegation_id]["status"] = "completed"
        callbacks.pop(0)()

        assert "delegating-sid" not in server._sessions
        assert len(torn_down) == 1
        assert torn_down[0][1] == "ws_orphan_reap"
    finally:
        server._sessions.pop("delegating-sid", None)
        with async_delegation._records_lock:
            async_delegation._records.pop(delegation_id, None)


def test_ws_orphan_reap_retries_when_delegation_lookup_fails(monkeypatch):
    from tools import async_delegation

    callbacks = []
    torn_down = []

    class _Timer:
        def __init__(self, _delay, callback):
            callbacks.append(callback)
            self.daemon = False

        def start(self):
            return None

    def _raise_lookup_error(*_args, **_kwargs):
        raise RuntimeError("delegation registry unavailable")

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(
        async_delegation, "has_live_for_session", _raise_lookup_error
    )
    monkeypatch.setattr(
        server,
        "_teardown_session",
        lambda session, *, end_reason="tui_close": torn_down.append(
            (session, end_reason)
        ),
    )
    server._sessions["lookup-error-sid"] = _session(
        transport=server._detached_ws_transport,
        running=False,
    )

    try:
        server._schedule_ws_orphan_reap("lookup-error-sid")
        callbacks.pop(0)()

        assert "lookup-error-sid" in server._sessions
        assert len(callbacks) == 1
        assert torn_down == []
    finally:
        server._sessions.pop("lookup-error-sid", None)


def test_finalize_session_closes_slash_worker(monkeypatch):
    """_finalize_session closes the slash_worker subprocess itself.

    Regression for #38095: the worker cleanup used to live only in the
    callers (_teardown_session / _shutdown_sessions), so any code path that
    finalized a session without going through them leaked the worker. Folding
    close() into the single _finalized-guarded chokepoint makes the cleanup
    defense-in-depth and idempotent.
    """
    closed = {"count": 0}

    class _FakeWorker:
        def close(self):
            closed["count"] += 1

    monkeypatch.setattr(server, "_notify_session_boundary", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)

    session = _session(slash_worker=_FakeWorker())

    server._finalize_session(session)
    assert closed["count"] == 1
    assert session.get("_finalized") is True

    # Idempotent: a second finalize (or a follow-up teardown) must not
    # re-close the worker â€” the _finalized guard short-circuits.
    server._finalize_session(session)
    server._teardown_session(session)
    assert closed["count"] == 1


def test_close_transport_rebinds_session_to_remaining_viewer(monkeypatch):
    """Closing a pop-out window's transport must leave the session with the
    still-open window instead of stranding it on the drop sentinel (#83716).

    The rebind #83716 added is gone; multi-client fan-out subsumes it. Both
    windows are attached to the slot at once, so the pop-out is a fan-out peer
    rather than a viewer waiting to be promoted, and closing it detaches that
    peer while retaining the surviving ordered mailbox. This pins the same
    guarantee through the mechanism that replaced the rebind: the session is
    not parked, not reaped, not handed to the orphan reaper, and the surviving
    window keeps receiving frames.
    """
    reap_calls = []
    monkeypatch.setattr(server, "_schedule_ws_orphan_reap", lambda sid: reap_calls.append(sid))

    class _LiveTransport:
        def __init__(self):
            self.frames = []
            self.received = threading.Event()

        def write(self, obj=None, *a, **k):
            self.frames.append(obj)
            self.received.set()
            return True

    main = _LiveTransport()
    popout = _LiveTransport()
    session = _session(transport=None, running=False)
    # Build the state the way production does: every window that resumes goes
    # through _live_session_payload, which attaches it into the slot and then
    # stamps it into the viewers registry.
    server._attach_session_transport(session, main)
    server._attach_session_transport(session, popout)
    session["viewers"] = {main: 100.0, popout: 200.0}
    server._sessions["multi-sid"] = session
    assert isinstance(session["transport"], server.FanoutTransport)

    try:
        reaped, detached = server._close_sessions_for_transport(popout)

        assert reaped == 0 and detached == 0
        assert server._session_transport_contains(session, main)
        assert not server._session_transport_contains(session, popout)
        assert "multi-sid" not in reap_calls
        assert server._ws_session_is_orphaned(session) is False

        # And it is still a working stream, not just a surviving reference.
        server._emit("message.delta", "multi-sid", {"text": "still here"})
        assert main.received.wait(timeout=5)
        assert [(f.get("params") or {}).get("type") for f in main.frames] == [
            "message.delta"
        ]
        assert popout.frames == []
    finally:
        # The fake slot must not outlive the test: _sessions is module state and
        # later sweeps would walk it.
        server._sessions.pop("multi-sid", None)


def test_close_transport_detaches_when_no_viewers_remain(monkeypatch):
    """The last viewer closing still lands the session on the drop sentinel
    and schedules the grace reap (unchanged single-window behavior)."""
    reap_calls = []
    monkeypatch.setattr(server, "_schedule_ws_orphan_reap", lambda sid: reap_calls.append(sid))

    class _LiveTransport:
        def write(self, *a, **k):
            return True

    only = _LiveTransport()
    session = _session(transport=only, running=False)
    session["viewers"] = {only: 100.0}
    server._sessions["solo-sid"] = session

    reaped, detached = server._close_sessions_for_transport(only)

    assert reaped == 0 and detached == 1
    assert session["transport"] is server._detached_ws_transport
    assert reap_calls == ["solo-sid"]


def test_close_transport_skips_dead_remaining_viewers(monkeypatch):
    """A viewer whose socket is already dead must not hold the session open.

    #83716's rebind refused to hand the session to a dead viewer; fan-out
    membership keeps that filter through _transport_is_live_peer, which is what
    decides whether anything survives the departing client. Both windows are
    ATTACHED here, which is the state production builds â€” a viewer that was
    never attached leaves the slot single-client and exercises the ordinary park
    path instead of this one.
    """
    reap_calls = []
    monkeypatch.setattr(server, "_schedule_ws_orphan_reap", lambda sid: reap_calls.append(sid))

    class _LiveTransport:
        def write(self, *a, **k):
            return True

    dead = _LiveTransport()
    popout = _LiveTransport()
    session = _session(transport=None, running=False)
    server._attach_session_transport(session, dead)
    server._attach_session_transport(session, popout)
    session["viewers"] = {dead: 100.0, popout: 200.0}
    assert isinstance(session["transport"], server.FanoutTransport)
    # The socket goes away without a disconnect reaching the gateway; the latch
    # _transport_is_dead reads is the only trace it leaves behind.
    dead._closed = True
    server._sessions["dead-viewer-sid"] = session

    try:
        reaped, detached = server._close_sessions_for_transport(popout)

        assert reaped == 0 and detached == 1
        assert session["transport"] is server._detached_ws_transport
        assert reap_calls == ["dead-viewer-sid"]
    finally:
        server._sessions.pop("dead-viewer-sid", None)


def test_live_session_payload_registers_transport_as_viewer():
    """Resume/activate through _live_session_payload must register the caller
    as a viewer so the disconnect path has something to re-bind to (#83716)."""
    class _LiveTransport:
        def write(self, *a, **k):
            return True

    t = _LiveTransport()
    session = _session(transport=server._detached_ws_transport, running=False)
    server._live_session_payload("viewer-sid", session, transport=t)

    assert session["transport"] is t
    assert t in session.get("viewers", {})


def test_ws_orphan_reap_spares_reattached_session(monkeypatch):
    """A session that rebinds a live transport is NOT considered orphaned."""

    class _LiveTransport:
        def write(self, *a, **k):
            return True

    # Reattached: transport is a live (non-stdio) transport.
    reattached = _session(transport=_LiveTransport(), running=False)
    assert server._ws_session_is_orphaned(reattached) is False

    # Mid-turn sessions are also spared even if detached.
    mid_turn = _session(transport=server._detached_ws_transport, running=True)
    assert server._ws_session_is_orphaned(mid_turn) is False

    # Already finalized sessions are spared (idempotency).
    done = _session(
        transport=server._detached_ws_transport,
        running=False,
        _finalized=True,
    )
    assert server._ws_session_is_orphaned(done) is False


def test_resume_rebind_cancels_pending_ws_orphan_reap(monkeypatch):
    """Re-binding a live transport via _live_session_payload must cancel the
    pending ws-orphan reap Timer (storm killer, part 1)."""
    cancelled = []

    class _Timer:
        def __init__(self, _delay, fn):
            self.fn = fn

        def start(self):
            return None

        def cancel(self):
            cancelled.append(self)

    class _LiveTransport:
        def write(self, *a, **k):
            return True

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    session = _session(transport=server._detached_ws_transport, running=False)
    server._sessions["cancel-sid"] = session

    try:
        server._schedule_ws_orphan_reap("cancel-sid")
        assert "cancel-sid" in server._pending_ws_reaps

        server._live_session_payload("cancel-sid", session, transport=_LiveTransport())

        assert "cancel-sid" not in server._pending_ws_reaps
        assert len(cancelled) == 1
    finally:
        server._sessions.pop("cancel-sid", None)
        server._pending_ws_reaps.pop("cancel-sid", None)


def test_claim_or_reuse_live_winner_cancels_pending_reap(monkeypatch):
    """The winner's pending reap is cancelled only once guarded reuse is accepted."""
    cancelled = []

    class _Timer:
        def __init__(self, _delay, fn):
            self.fn = fn

        def start(self):
            return None

        def cancel(self):
            cancelled.append(self)

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    agent = types.SimpleNamespace(session_id="stored-claim")
    winner = _session(
        agent=agent,
        session_key="stored-claim",
        transport=server._detached_ws_transport,
        running=False,
    )
    server._sessions["winner-sid"] = winner

    try:
        server._schedule_ws_orphan_reap("winner-sid")
        assert "winner-sid" in server._pending_ws_reaps

        live = server._claim_or_reuse_live(
            "fresh-sid", "stored-claim", _session(), None
        )

        assert live == ("winner-sid", winner)
        assert "winner-sid" in server._pending_ws_reaps
        assert cancelled == []
        assert winner["transport"] is server._detached_ws_transport

        transport = object()
        monkeypatch.setattr(server, "current_transport", lambda: transport)
        ctx = server._Resume(1, {"omit_messages": True}, "stored-claim")
        response = server._resume_reuse_live(ctx, *live)

        assert response["result"]["session_id"] == "winner-sid"
        assert winner["transport"] is transport
        assert "winner-sid" not in server._pending_ws_reaps
        assert len(cancelled) == 1
    finally:
        server._sessions.pop("winner-sid", None)
        server._sessions.pop("fresh-sid", None)
        server._pending_ws_reaps.pop("winner-sid", None)


def test_superseded_runtime_finalized_without_reclaimed_broadcast(monkeypatch):
    """When a resume mints a fresh runtime for a stored id whose prior runtime
    is sentinel-parked, the old record is finalized quietly with end_reason
    superseded_by_resume â€” no session.reclaimed broadcast, and its pending
    reap Timer is cancelled (storm killer, part 2)."""
    cancelled = []
    broadcasts = []
    torn_down = []

    class _Timer:
        def __init__(self, _delay, fn):
            self.fn = fn

        def start(self):
            return None

        def cancel(self):
            cancelled.append(self)

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(
        server,
        "_broadcast_global_event",
        lambda event, payload=None: broadcasts.append((event, payload)),
    )
    monkeypatch.setattr(
        server,
        "_teardown_popped_session",
        lambda popped, *, end_reason: torn_down.append((popped, end_reason)) or True,
    )
    monkeypatch.setattr(server, "_register_session_cwd", lambda _s: None)

    old_agent = types.SimpleNamespace(session_id="stored-super")
    old = _session(
        agent=old_agent,
        session_key="stored-super",
        transport=server._detached_ws_transport,
        running=False,
    )
    server._sessions["old-sid"] = old
    fresh = _session(session_key="stored-super")

    try:
        server._schedule_ws_orphan_reap("old-sid")
        assert "old-sid" in server._pending_ws_reaps

        # The old runtime looks live to _find_live_session_by_key, so make it
        # invisible the way a real mint path would (its client is gone and the
        # resume slow path only mints after the fast path found no live match:
        # mark it finalized-for-lookup via a different stored key is wrong â€”
        # instead simulate the mint race by removing it from lookup).
        old["_finalized"] = False
        monkeypatch.setattr(server, "_find_live_session_by_key", lambda _k, *_a: None)

        result = server._claim_or_reuse_live("new-sid", "stored-super", fresh, None)

        assert result is None
        assert server._sessions.get("new-sid") is fresh
        assert "old-sid" not in server._sessions
        assert "old-sid" not in server._pending_ws_reaps
        assert len(cancelled) == 1
        assert torn_down == [(old, "superseded_by_resume")]
        assert broadcasts == []  # no session.reclaimed storm
    finally:
        server._sessions.pop("old-sid", None)
        server._sessions.pop("new-sid", None)
        server._pending_ws_reaps.pop("old-sid", None)
        server._pending_ws_reaps.pop("new-sid", None)




def test_lazy_unpersisted_resume_rebinds_transport_and_cancels_reap(monkeypatch):
    """The lazy/unpersisted resume branch (no state.db row yet â€” every fresh
    Bot Chat) must ALSO rebind the transport and cancel the pending reap when
    it hands back a sentinel-parked live record. Found by live WS E2E after
    the #93361 merge: the unit-covered paths (_live_session_payload,
    _reuse_live_response, _claim_or_reuse_live) all cancelled, but this branch
    returned the record while leaving it on the drop sentinel with the reap
    Timer armed â€” the storm survived for unpersisted sessions (storm killer,
    part 3)."""
    cancelled = []

    class _Timer:
        def __init__(self, _delay, fn):
            self.fn = fn

        def start(self):
            return None

        def cancel(self):
            cancelled.append(self)

    class _LiveTransport:
        def write(self, *a, **k):
            return True

    live_transport = _LiveTransport()
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "current_transport", lambda: live_transport)
    # get_session/get_session_by_title miss -> forces the unpersisted branch
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: types.SimpleNamespace(
            get_session=lambda _t: None,
            get_session_by_title=lambda _t: None,
        ),
    )

    session = _session(
        session_key="stored-lazy",
        transport=server._detached_ws_transport,
        running=False,
        history=[],
        profile_home=None,
    )
    server._sessions["lazy-sid"] = session

    try:
        server._schedule_ws_orphan_reap("lazy-sid")
        assert "lazy-sid" in server._pending_ws_reaps

        resp = _dispatch_sync(
            {
                "id": "lz1",
                "method": "session.resume",
                "params": {"session_id": "stored-lazy", "omit_messages": True},
            },
            transport=live_transport,
        )

        assert resp is not None and resp["result"]["session_id"] == "lazy-sid"
        assert session["transport"] is live_transport
        assert "lazy-sid" not in server._pending_ws_reaps
        assert len(cancelled) == 1
    finally:
        server._sessions.pop("lazy-sid", None)
        server._pending_ws_reaps.pop("lazy-sid", None)


def test_ws_orphan_reap_still_fires_when_never_resumed(monkeypatch):
    """Nobody re-resumes: the reap fires normally and unregisters its Timer."""
    callbacks = []
    torn_down = []

    class _Timer:
        def __init__(self, _delay, fn):
            callbacks.append(fn)

        def start(self):
            return None

        def cancel(self):
            return None

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(
        server,
        "_teardown_popped_session",
        lambda popped, *, end_reason: torn_down.append((popped, end_reason)) or True,
    )
    session = _session(transport=server._detached_ws_transport, running=False)
    server._sessions["lonely-sid"] = session

    try:
        server._schedule_ws_orphan_reap("lonely-sid")
        assert "lonely-sid" in server._pending_ws_reaps
        callbacks.pop(0)()

        assert "lonely-sid" not in server._sessions
        assert "lonely-sid" not in server._pending_ws_reaps
        assert torn_down == [(session, "ws_orphan_reap")]
    finally:
        server._sessions.pop("lonely-sid", None)
        server._pending_ws_reaps.pop("lonely-sid", None)


def test_ws_orphan_reap_spares_detached_session_with_running_async_delegation(monkeypatch):
    """A detached desktop session with live background delegation is parked.

    Regression for Desktop session switches / transient WS detaches: the parent
    turn is idle, but a background delegate_task still owns the session's
    return address. Reaping immediately interrupts the child and turns its
    completion into an unowned orphan.
    """
    timers = []
    closed = []

    class _Timer:
        def __init__(self, _delay, fn):
            self.fn = fn
            timers.append(self)

        def start(self):
            return None

    class _DB:
        def get_session(self, _session_id):
            return {"id": "sess_bg", "source": "desktop"}

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(
        server,
        "_teardown_popped_session",
        lambda session, *, end_reason="tui_close": (
            closed.append((session["_sid"], end_reason)) if session is not None else None
        ),
    )

    server._sessions["bg-sid"] = _session(
        transport=server._detached_ws_transport,
        running=False,
        session_key="sess_bg",
    )
    ad._reset_for_tests()
    try:
        with ad._records_lock:
            ad._records["deleg_bg"] = {
                "delegation_id": "deleg_bg",
                "status": "running",
                "session_key": "sess_bg",
                "origin_ui_session_id": "bg-sid",
                "interrupt_fn": lambda: None,
            }

        server._schedule_ws_orphan_reap("bg-sid")
        assert len(timers) == 1

        timers.pop(0).fn()

        assert closed == []
        assert "bg-sid" in server._sessions
        assert len(timers) == 1

        with ad._records_lock:
            ad._records["deleg_bg"]["status"] = "finalizing"
            ad._records["deleg_bg"]["interrupt_fn"] = None

        timers.pop(0).fn()

        assert closed == []
        assert "bg-sid" in server._sessions
        assert len(timers) == 1

        with ad._records_lock:
            ad._records["deleg_bg"]["status"] = "completed"

        timers.pop(0).fn()

        assert closed == [("bg-sid", "ws_orphan_reap")]
    finally:
        ad._reset_for_tests()
        server._sessions.pop("bg-sid", None)


def test_ws_orphan_reap_disabled_when_grace_zero(monkeypatch):
    """Grace=0 disables the reaper entirely (pre-fix park-forever behaviour)."""
    fired = {"timer": False}

    class _Timer:
        def __init__(self, *a, **k):
            fired["timer"] = True

        def start(self):
            pass

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.0)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    server._schedule_ws_orphan_reap("any-sid")
    assert fired["timer"] is False


def test_ws_orphan_reap_defers_running_turn_with_fresh_activity(monkeypatch):
    """#98028/#100325: a client-absent turn whose activity clock is fresh is
    NOT interrupted â€” it keeps running detached and the reaper re-polls at the
    grace interval. Once the clock goes stale the wedged-turn interrupt fires,
    and after the turn settles the session is reaped as before."""
    callbacks = []
    delays = []
    interrupted = []
    torn_down = []

    class _Timer:
        def __init__(self, delay, callback):
            delays.append(delay)
            callbacks.append(callback)
            self.daemon = False

        def start(self):
            return None

    activity = {"seconds_since_activity": 1.0}
    agent = types.SimpleNamespace(
        get_activity_summary=lambda: dict(activity),
        interrupt=lambda message=None: interrupted.append("interrupted"),
    )

    class _DeadThread:
        def is_alive(self):
            return False

    session = _session(
        agent=agent,
        transport=server._detached_ws_transport,
        running=True,
        _run_thread=_DeadThread(),
    )
    server._sessions["fresh-sid"] = session
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server, "_WS_ORPHAN_ACTIVITY_STALE_S", 300.0)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(
        server,
        "_teardown_popped_session",
        lambda claimed, *, end_reason: torn_down.append((claimed, end_reason)) or True,
    )

    try:
        server._schedule_ws_orphan_reap("fresh-sid")

        # Two grace cycles with fresh activity: no interrupt, reschedule at
        # the GRACE interval (not the 1s interrupt-settle poll).
        for _ in range(2):
            callbacks.pop(0)()
            assert interrupted == []
            assert not session.get("_client_gone_interrupt_requested")
            assert delays[-1] == server._WS_ORPHAN_REAP_GRACE_S
            assert "fresh-sid" in server._sessions

        # Activity goes stale (turn wedged) -> interrupt fires on next poll.
        activity["seconds_since_activity"] = 301.0
        callbacks.pop(0)()
        assert interrupted == ["interrupted"]
        assert session["_client_gone_interrupt_requested"] is True

        # Turn settles -> reap proceeds exactly as today.
        session["running"] = False
        callbacks.pop(0)()
        assert "fresh-sid" not in server._sessions
        assert torn_down == [(session, "ws_orphan_reap")]
    finally:
        server._sessions.pop("fresh-sid", None)


def test_ws_orphan_activity_gate_zero_restores_interrupt_at_grace(monkeypatch):
    """ws_orphan_activity_stale_s=0 opts out: fresh activity no longer defers
    the client-gone interrupt (pre-#98028 behaviour)."""
    callbacks = []
    interrupted = []

    class _Timer:
        def __init__(self, _delay, callback):
            callbacks.append(callback)
            self.daemon = False

        def start(self):
            return None

    class _LiveThread:
        def is_alive(self):
            return True

    agent = types.SimpleNamespace(
        get_activity_summary=lambda: {"seconds_since_activity": 0.5},
        interrupt=lambda message=None: interrupted.append("interrupted"),
    )
    session = _session(
        agent=agent,
        transport=server._detached_ws_transport,
        running=True,
        _run_thread=_LiveThread(),
    )
    server._sessions["optout-sid"] = session
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server, "_WS_ORPHAN_ACTIVITY_STALE_S", 0.0)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})

    try:
        server._schedule_ws_orphan_reap("optout-sid")
        callbacks.pop(0)()
        assert interrupted == ["interrupted"]
        assert session["_client_gone_interrupt_requested"] is True
    finally:
        server._sessions.pop("optout-sid", None)


def test_ws_orphan_activity_gate_unreadable_summary_stays_eligible(monkeypatch):
    """A broken/opaque activity summary must fail CLOSED (not fresh): the
    wedged-turn interrupt-at-grace safety net is preserved."""

    def _boom():
        raise RuntimeError("summary unavailable")

    agent = types.SimpleNamespace(get_activity_summary=_boom)
    monkeypatch.setattr(server, "_WS_ORPHAN_ACTIVITY_STALE_S", 300.0)
    assert server._ws_orphan_turn_activity_is_fresh({"agent": agent}) is False
    # No agent / no summary method: same conservative answer.
    assert server._ws_orphan_turn_activity_is_fresh({"agent": None}) is False
    assert (
        server._ws_orphan_turn_activity_is_fresh(
            {"agent": types.SimpleNamespace()}
        )
        is False
    )
    # Never-stamped clock (None) is not fresh either.
    agent2 = types.SimpleNamespace(
        get_activity_summary=lambda: {"seconds_since_activity": None}
    )
    assert server._ws_orphan_turn_activity_is_fresh({"agent": agent2}) is False


def test_init_session_fires_reset_hook(monkeypatch):
    hooks = []

    class _FakeWorker:
        def __init__(self, key, model, profile_home=None):
            self.key = key

        def close(self):
            return None

    monkeypatch.setattr(server, "_SlashWorker", _FakeWorker)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        server,
        "_notify_session_boundary",
        lambda event, session_id, *_args: hooks.append((event, session_id)),
    )

    import tools.approval as _approval

    monkeypatch.setattr(_approval, "register_gateway_notify", lambda key, cb: None)
    monkeypatch.setattr(_approval, "load_permanent_allowlist", lambda: None)

    sid = "sid"
    try:
        server._init_session(
            sid,
            "session-key",
            types.SimpleNamespace(model="x"),
            history=[],
            cols=80,
        )
        assert ("on_session_reset", "session-key") in hooks
    finally:
        server._sessions.pop(sid, None)


def test_session_title_creates_row_and_sets_immediately_when_not_ready(monkeypatch):
    """An explicit /title before the first message must persist NOW, not queue.

    Regression: the desktop deferred the DB row to the first prompt, so a
    /title typed before any message only stashed ``pending_title`` and relied
    on a post-turn apply block. When that turn never landed under the session
    key, the title was silently lost and the sidebar fell back to the message
    preview. The handler now creates the row up front (mirroring the messaging
    gateway) so an explicit /title takes effect immediately.
    """
    state = {"row": None, "title": None, "ensured": False}

    class _FakeDB:
        def get_session_title(self, _key):
            return state["title"]

        def get_session(self, _key):
            return state["row"]

        def set_session_title(self, _key, title):
            # Mirrors SessionDB: UPDATE affects 0 rows until the row exists.
            if state["row"] is None:
                return False
            state["title"] = title
            return True

    fake_db = _FakeDB()

    def _fake_ensure_row(_session):
        # The real _ensure_session_db_row does an INSERT OR IGNORE.
        state["ensured"] = True
        state["row"] = {"id": "session-key", "title": None}

    import contextlib

    @contextlib.contextmanager
    def _fake_session_db(_session):
        yield fake_db

    server._sessions["sid"] = _session(pending_title=None)
    monkeypatch.setattr(server, "_get_db", lambda: fake_db)
    monkeypatch.setattr(server, "_ensure_session_db_row", _fake_ensure_row)
    monkeypatch.setattr(server, "_session_db", _fake_session_db)
    try:
        set_resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "my-custom-name"},
            }
        )

        # No longer queued â€” the row is created and the title set immediately.
        assert set_resp["result"]["pending"] is False
        assert set_resp["result"]["title"] == "my-custom-name"
        assert state["ensured"] is True, "the row must be created up front"
        assert state["title"] == "my-custom-name"
        assert server._sessions["sid"]["pending_title"] is None

        # A subsequent read reflects the persisted title.
        get_resp = server.handle_request(
            {"id": "2", "method": "session.title", "params": {"session_id": "sid"}}
        )
        assert get_resp["result"]["title"] == "my-custom-name"
    finally:
        server._sessions.pop("sid", None)


def test_session_title_falls_back_to_queue_when_row_create_fails(monkeypatch):
    """If row creation can't take (DB down / racing writer), keep the queue.

    The post-turn apply block is still the recovery path, so a /title that
    can't persist up front must not be dropped â€” it falls back to
    ``pending_title`` exactly as before.
    """

    class _FakeDB:
        def get_session_title(self, _key):
            return None

        def get_session(self, _key):
            return None

        def set_session_title(self, _key, _title):
            return False

    fake_db = _FakeDB()

    def _fake_ensure_row(_session):
        # Simulate a persist that didn't take â€” row still absent.
        pass

    import contextlib

    @contextlib.contextmanager
    def _fake_session_db(_session):
        yield fake_db

    server._sessions["sid"] = _session(pending_title=None)
    monkeypatch.setattr(server, "_get_db", lambda: fake_db)
    monkeypatch.setattr(server, "_ensure_session_db_row", _fake_ensure_row)
    monkeypatch.setattr(server, "_session_db", _fake_session_db)
    try:
        set_resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "queued title"},
            }
        )

        assert set_resp["result"]["pending"] is True
        assert set_resp["result"]["title"] == "queued title"
        assert server._sessions["sid"]["pending_title"] == "queued title"

        get_resp = server.handle_request(
            {"id": "2", "method": "session.title", "params": {"session_id": "sid"}}
        )
        assert get_resp["result"]["title"] == "queued title"
    finally:
        server._sessions.pop("sid", None)


def test_notification_event_routing_by_session_key(monkeypatch):
    """Background-process events surface only in the session that owns them."""
    mine = _session(session_key="mine")
    other = _session(session_key="other")
    monkeypatch.setattr(server, "_sessions", {"a": mine, "b": other})

    # My own event â†’ handle it.
    assert server._notification_event_belongs_elsewhere("a", mine, {"session_key": "mine"}) is False
    # Global/system event with no owner â†’ handle it.
    assert server._notification_event_belongs_elsewhere("a", mine, {"session_key": ""}) is False
    assert server._notification_event_belongs_elsewhere("a", mine, {}) is False
    # Owned by another *live* session â†’ defer to that session's poller.
    assert server._notification_event_belongs_elsewhere("a", mine, {"session_key": "other"}) is True
    # Owner is gone (not in _sessions) â†’ handle as fallback so it isn't lost.
    assert server._notification_event_belongs_elsewhere("a", mine, {"session_key": "ghost"}) is False


def test_async_delegation_event_prefers_origin_ui_session(monkeypatch):
    """Detached subagent completions return to the commissioning TUI tab.

    Regression: when the durable session key was stale/orphaned, whichever
    desktop poller woke first could consume the async result and inject it into
    an unrelated session.
    """
    mine = _session(session_key="current-key")
    other = _session(session_key="unrelated-key")
    monkeypatch.setattr(server, "_sessions", {"origin-sid": mine, "other-sid": other})
    monkeypatch.setattr(server, "_get_db", lambda: None)
    evt = {
        "type": "async_delegation",
        "session_key": "stale-or-rotated-key",
        "origin_ui_session_id": "origin-sid",
    }

    assert server._notification_event_belongs_elsewhere("other-sid", other, evt) is True
    assert server._notification_event_belongs_elsewhere("origin-sid", mine, evt) is False


def test_notification_event_follows_compression_continuation(monkeypatch):
    """Events keyed to a compressed parent route to the live continuation."""
    old_parent = _session(session_key="old-parent")
    live_tip = _session(session_key="new-tip")
    monkeypatch.setattr(server, "_sessions", {"old-sid": old_parent, "tip-sid": live_tip})

    class _DB:
        def resolve_resume_session_id(self, session_id):
            return "new-tip" if session_id == "old-parent" else session_id

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    evt = {"type": "async_delegation", "session_key": "old-parent"}

    assert server._notification_event_belongs_elsewhere("old-sid", old_parent, evt) is True
    assert server._notification_event_belongs_elsewhere("tip-sid", live_tip, evt) is False
    # A third session must leave it alone for the continuation's poller.
    third = _session(session_key="third")
    monkeypatch.setattr(
        server,
        "_sessions",
        {"old-sid": old_parent, "tip-sid": live_tip, "third-sid": third},
    )
    assert server._notification_event_belongs_elsewhere("third-sid", third, evt) is True


def test_finalized_origin_ui_session_falls_back_to_live_continuation(monkeypatch):
    """A closed origin tab must not steal its resumed continuation's result."""
    finalized_origin = _session(session_key="old-parent", _finalized=True)
    live_tip = _session(session_key="new-tip")
    monkeypatch.setattr(
        server,
        "_sessions",
        {"origin-sid": finalized_origin, "tip-sid": live_tip},
    )

    class _DB:
        def resolve_resume_session_id(self, session_id):
            return "new-tip" if session_id == "old-parent" else session_id

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    evt = {
        "type": "async_delegation",
        "session_key": "old-parent",
        "origin_ui_session_id": "origin-sid",
    }

    assert server._notification_event_belongs_elsewhere("origin-sid", finalized_origin, evt) is True
    assert server._notification_event_belongs_elsewhere("tip-sid", live_tip, evt) is False


def test_prompt_submit_rejects_negative_truncate_ordinal(monkeypatch):
    """A negative truncate_before_user_ordinal must be rejected, not honoured.

    The handler validates the upper bound (`ordinal >= len(user_indices)`) but a
    negative ordinal would otherwise slip through and hit Python negative
    indexing: `user_indices[-1]` selects the LAST user turn, truncating history
    to everything before it and persisting that loss via replace_messages â€” an
    unrecoverable overwrite of the session DB. Reject it on the safe 4018 path
    and leave the in-memory history and the DB untouched.
    """
    replaced = []

    class _FakeDB:
        def replace_messages(
            self,
            key,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            replaced.append((key, list(messages)))

    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "done"},
    ]
    server._sessions["trunc-sid"] = _session(history=list(history))
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    # If the guard ever lets a negative ordinal through, these would run and the
    # session would be marked busy; failing here makes that regression loud.
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )
    monkeypatch.setattr(
        server, "_start_inflight_turn", lambda *a, **k: pytest.fail("must not start a turn")
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "trunc-sid",
                    "text": "next",
                    "truncate_before_user_ordinal": -1,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp["error"]["code"] == 4018
        # History and the DB are left exactly as they were â€” no silent loss.
        assert server._sessions["trunc-sid"]["history"] == history
        assert server._sessions["trunc-sid"]["running"] is False
        assert replaced == []
    finally:
        server._sessions.pop("trunc-sid", None)


def test_prompt_submit_refuses_boolean_ordinal(monkeypatch):
    """A JSON `true` ordinal must return 4004, not coerce to turn 1.

    bool is an int subclass, so `int(True) == 1`: a client bug that sends
    `truncate_before_user_ordinal: true` with confirm_truncate would aim a
    confirmed rewind at the SECOND user turn and hard-truncate everything
    after the first â€” the same silent-loss class as #82756.
    """
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply 2"},
    ]
    server._sessions["bool-trunc-sid"] = _session(history=list(history))
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )
    monkeypatch.setattr(
        server, "_start_inflight_turn", lambda *a, **k: pytest.fail("must not start a turn")
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "bool-trunc-sid",
                    "text": "new turn",
                    "truncate_before_user_ordinal": True,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is not None
        assert resp["error"]["code"] == 4004
        assert server._sessions["bool-trunc-sid"]["history"] == history
    finally:
        server._sessions.pop("bool-trunc-sid", None)


def test_prompt_submit_refuses_confirm_truncate_without_target(monkeypatch):
    """confirm_truncate with no ordinal is leaked rewind state â€” fail fast.

    The desktop auto-attaches confirm_truncate whenever it builds truncation
    params (#82756); a bare flag on an ordinary submit means the client's
    composer state is corrupted. Refusing loudly surfaces the client bug
    instead of quietly ignoring the flag.
    """
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply 1"},
    ]
    server._sessions["bare-confirm-sid"] = _session(history=list(history))
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )
    monkeypatch.setattr(
        server, "_start_inflight_turn", lambda *a, **k: pytest.fail("must not start a turn")
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "bare-confirm-sid",
                    "text": "new turn",
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is not None
        assert resp["error"]["code"] == 4004
        assert server._sessions["bare-confirm-sid"]["history"] == history
    finally:
        server._sessions.pop("bare-confirm-sid", None)


def test_prompt_submit_refuses_unconfirmed_nonempty_truncation(monkeypatch):
    """An ordinal without confirm_truncate must not drop the session tail.

    #80763: a desktop client carried a leftover truncate_before_user_ordinal
    into an ORDINARY submit. The request was indistinguishable from a real
    rewind â€” in-range ordinal, non-empty result â€” so the empty-truncation guard
    never fired and replace_messages() DELETEd 244 durable rows (296 -> 52).
    Intent has to be stated: refuse on 4029 and leave memory and DB untouched.
    """
    replaced = []

    class _FakeDB:
        def replace_messages(
            self,
            key,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            replaced.append((key, list(messages)))

    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "third"},
        {"role": "assistant", "content": "sure"},
    ]
    server._sessions["unconfirmed-trunc-sid"] = _session(history=list(history))
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )
    monkeypatch.setattr(
        server, "_start_inflight_turn", lambda *a, **k: pytest.fail("must not start a turn")
    )

    def _submit(**extra):
        return server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "unconfirmed-trunc-sid",
                    "text": "an ordinary typed message",
                    "truncate_before_user_ordinal": 2,
                    **extra,
                },
            }
        )

    try:
        resp = _submit()
        assert resp["error"]["code"] == 4029
        assert "confirm_truncate" in resp["error"]["message"]
        # Explicit falsey values must not satisfy the opt-in either.
        for falsey in (False, 0, "", "false", "no"):
            assert _submit(confirm_truncate=falsey)["error"]["code"] == 4029, falsey
        # confirm_empty_truncate is a different gate â€” it must not stand in for
        # rewind intent on a cut that leaves the transcript non-empty.
        assert _submit(confirm_empty_truncate=True)["error"]["code"] == 4029
        session = server._sessions["unconfirmed-trunc-sid"]
        assert session["history"] == history
        assert session["history_version"] == 0
        assert session["running"] is False
        assert replaced == []
    finally:
        server._sessions.pop("unconfirmed-trunc-sid", None)


def test_prompt_submit_truncates_by_message_id(monkeypatch):
    """#82756: truncate_before_message_id resolves target message and cuts history accurately."""
    replaced = []

    class _FakeDB:
        def replace_messages(
            self,
            key,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            replaced.append((key, list(messages)))

    history = [
        {"id": "msg-1", "role": "user", "content": "first"},
        {"role": "assistant", "content": "reply 1"},
        {"id": "msg-2", "role": "user", "content": "second"},
        {"role": "assistant", "content": "reply 2"},
    ]
    server._sessions["msg-id-trunc-sid"] = _session(history=list(history))
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: None
    )
    monkeypatch.setattr(
        server, "_start_inflight_turn", lambda *a, **k: None
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "msg-id-trunc-sid",
                    "text": "new turn",
                    "truncate_before_message_id": "msg-2",
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("result") is not None
        assert len(replaced) == 1
        assert replaced[0][1] == history[:2]
    finally:
        server._sessions.pop("msg-id-trunc-sid", None)


def test_prompt_submit_truncation_falls_back_to_sid_when_session_key_null(monkeypatch):
    """#81904: a NULL session_key must not FK-fail the truncation persist.

    CLI-origin sessions resumed in the Desktop have no session_key; the
    truncation path used to call replace_messages(None, ...), whose reinsert
    violated the messages.session_id FK ("FOREIGN KEY constraint failed" â†’
    "Restore failed"). The persist must key off the session id instead â€”
    for CLI-origin rows the durable sessions.id IS the requested sid.
    """
    replaced = []

    class _FakeDB:
        def replace_messages(
            self,
            key,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            replaced.append((key, list(messages)))

    history = [
        {"_row_id": 101, "role": "user", "content": "first"},
        {"_row_id": 102, "role": "assistant", "content": "reply 1"},
        {"_row_id": 103, "role": "user", "content": "second"},
        {"_row_id": 104, "role": "assistant", "content": "reply 2"},
    ]
    server._sessions["null-key-trunc-sid"] = _session(
        history=list(history), session_key=None
    )
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_start_inflight_turn", lambda *a, **k: None)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "null-key-trunc-sid",
                    "text": "new turn",
                    "truncate_before_row_id": 103,
                    "truncate_before_user_ordinal": 1,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("result") is not None
        assert len(replaced) == 1
        # Keyed by the session id, never None (the FK-violating value).
        assert replaced[0][0] == "null-key-trunc-sid"
        assert replaced[0][1] == history[:2]
    finally:
        server._sessions.pop("null-key-trunc-sid", None)


def test_prompt_submit_refuses_ordinal_and_message_id_mismatch(monkeypatch):
    """#82756: A mismatch between truncate_before_user_ordinal and truncate_before_message_id must return 4030."""
    history = [
        {"id": "msg-1", "role": "user", "content": "first"},
        {"role": "assistant", "content": "reply 1"},
        {"id": "msg-2", "role": "user", "content": "second"},
        {"role": "assistant", "content": "reply 2"},
    ]
    server._sessions["mismatch-trunc-sid"] = _session(history=list(history))
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )
    monkeypatch.setattr(
        server, "_start_inflight_turn", lambda *a, **k: pytest.fail("must not start a turn")
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "mismatch-trunc-sid",
                    "text": "new turn",
                    "truncate_before_message_id": "msg-2",  # ordinal index 1
                    "truncate_before_user_ordinal": 0,      # mismatch (stale 0)
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is not None
        assert resp["error"]["code"] == 4030
    finally:
        server._sessions.pop("mismatch-trunc-sid", None)


def test_prompt_submit_refuses_ordinal_only_when_history_has_row_ids(monkeypatch):
    """A durable session must not trust an ordinal without a row-id target."""
    replaced = []

    class _FakeDB:
        def replace_messages(
            self,
            key,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            replaced.append((key, list(messages)))

    history = [
        {"_row_id": 101, "role": "user", "content": "first"},
        {"_row_id": 102, "role": "assistant", "content": "reply 1"},
        {"_row_id": 103, "role": "user", "content": "second"},
        {"_row_id": 104, "role": "assistant", "content": "reply 2"},
    ]
    server._sessions["ordinal-only-durable-sid"] = _session(history=list(history))
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )
    monkeypatch.setattr(
        server, "_start_inflight_turn", lambda *a, **k: pytest.fail("must not start a turn")
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "ordinal-only-durable-sid",
                    "text": "retry",
                    "truncate_before_user_ordinal": 1,
                    "confirm_truncate": True,
                },
            }
        )

        assert resp["error"]["code"] == 4004
        assert "truncate_before_row_id" in resp["error"]["message"]
        assert server._sessions["ordinal-only-durable-sid"]["history"] == history
        assert server._sessions["ordinal-only-durable-sid"]["running"] is False
        assert replaced == []
    finally:
        server._sessions.pop("ordinal-only-durable-sid", None)


def test_prompt_submit_refuses_ordinal_only_when_durable_history_is_unstamped(monkeypatch):
    """Durability comes from state.db, not optional stamps on the live copy."""
    replaced = []
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply 2"},
    ]

    class _FakeDB:
        def get_messages_as_conversation(self, key, **kwargs):
            assert key == "session-key"
            assert kwargs["include_row_ids"] is True
            return [dict(message, _row_id=100 + index) for index, message in enumerate(history)]

        def replace_messages(self, key, messages, active_only=False, archive_dropped=False):
            replaced.append((key, list(messages)))

    server._sessions["unstamped-durable-sid"] = _session(history=list(history))
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )
    monkeypatch.setattr(
        server, "_start_inflight_turn", lambda *a, **k: pytest.fail("must not start a turn")
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "unstamped-durable-sid",
                    "text": "retry",
                    "truncate_before_user_ordinal": 1,
                    "confirm_truncate": True,
                },
            }
        )

        assert resp["error"]["code"] == 4004
        assert "truncate_before_row_id" in resp["error"]["message"]
        assert server._sessions["unstamped-durable-sid"]["history"] == history
        assert replaced == []
    finally:
        server._sessions.pop("unstamped-durable-sid", None)


def test_prompt_submit_truncates_by_row_id(monkeypatch):
    """#82959: prompt.submit with truncate_before_row_id must cut at the target row id."""
    replaced = []

    class _FakeDB:
        def replace_messages(
            self,
            key,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            replaced.append((key, list(messages)))

    history = [
        {"_row_id": 101, "role": "user", "content": "first"},
        {"_row_id": 102, "role": "assistant", "content": "reply 1"},
        {"_row_id": 103, "role": "user", "content": "second"},
        {"_row_id": 104, "role": "assistant", "content": "reply 2"},
    ]
    sess = _session(history=list(history))
    server._sessions["row-id-trunc-sid"] = sess
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    started = []
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: started.append(k)
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "row-id-trunc-sid",
                    "text": "new turn",
                    "truncate_before_row_id": 103,
                    "truncate_before_user_ordinal": 1,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is None
        assert len(sess["history"]) == 2
        assert sess["history"][-1]["content"] == "reply 1"
        assert len(replaced) == 1
        assert replaced[0][1] == history[:2]
    finally:
        server._sessions.pop("row-id-trunc-sid", None)


def test_prompt_submit_truncates_by_string_row_id(monkeypatch):
    """#82959: String row IDs in history match correctly against integer truncate_before_row_id."""
    replaced = []

    class _FakeDB:
        def replace_messages(
            self,
            key,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            replaced.append((key, list(messages)))

    history = [
        {"_row_id": "101", "role": "user", "content": "first"},
        {"_row_id": "102", "role": "assistant", "content": "reply 1"},
        {"_row_id": "103", "role": "user", "content": "second"},
        {"_row_id": "104", "role": "assistant", "content": "reply 2"},
    ]
    sess = _session(history=list(history))
    server._sessions["str-row-id-trunc-sid"] = sess
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "str-row-id-trunc-sid",
                    "text": "new turn",
                    "truncate_before_row_id": 103,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is None
        assert len(sess["history"]) == 2
    finally:
        server._sessions.pop("str-row-id-trunc-sid", None)


def test_prompt_submit_refuses_ordinal_and_row_id_mismatch(monkeypatch):
    """#82959: A mismatch between truncate_before_user_ordinal and truncate_before_row_id must return 4030."""
    history = [
        {"_row_id": 201, "role": "user", "content": "first"},
        {"_row_id": 202, "role": "assistant", "content": "reply 1"},
        {"_row_id": 203, "role": "user", "content": "second"},
        {"_row_id": 204, "role": "assistant", "content": "reply 2"},
    ]
    server._sessions["row-mismatch-sid"] = _session(history=list(history))
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "row-mismatch-sid",
                    "text": "new turn",
                    "truncate_before_row_id": 203,  # user turn ordinal 1
                    "truncate_before_user_ordinal": 0,  # mismatch (stale 0)
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is not None
        assert resp["error"]["code"] == 4030
    finally:
        server._sessions.pop("row-mismatch-sid", None)


def test_prompt_submit_refuses_boolean_row_id(monkeypatch):
    """Boolean truncate_before_row_id must return 4004."""
    history = [
        {"_row_id": 301, "role": "user", "content": "first"},
        {"_row_id": 302, "role": "assistant", "content": "reply 1"},
    ]
    server._sessions["bool-row-sid"] = _session(history=list(history))
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "bool-row-sid",
                    "text": "new turn",
                    "truncate_before_row_id": True,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is not None
        assert resp["error"]["code"] == 4004
    finally:
        server._sessions.pop("bool-row-sid", None)


def test_prompt_submit_row_id_not_found(monkeypatch):
    """Unknown truncate_before_row_id must return 4018."""
    history = [
        {"_row_id": 401, "role": "user", "content": "first"},
    ]
    server._sessions["missing-row-sid"] = _session(history=list(history))
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "missing-row-sid",
                    "text": "new turn",
                    "truncate_before_row_id": 999,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is not None
        assert resp["error"]["code"] == 4018
    finally:
        server._sessions.pop("missing-row-sid", None)


def test_prompt_submit_row_id_ignores_platform_id_fallback(monkeypatch):
    """truncate_before_row_id must not match string platform IDs."""
    history = [
        {"id": "999", "role": "user", "content": "first"},
        {"role": "assistant", "content": "reply 1"},
    ]
    server._sessions["string-id-sid"] = _session(history=list(history))
    try:
        resp = server.handle_request({
            "id": "1",
            "method": "prompt.submit",
            "params": {
                "session_id": "string-id-sid",
                "text": "new turn",
                "truncate_before_row_id": 999,
                "confirm_truncate": True,
            }
        })
        assert resp.get("error") is not None
        assert resp["error"]["code"] == 4018
    finally:
        server._sessions.pop("string-id-sid", None)


def test_prompt_submit_refuses_empty_truncation_without_confirm(monkeypatch):
    """A confirmed rewind still must not wipe a non-empty transcript by accident.

    Ordinal 0 cuts at the first user message (history[:0] == []) and
    replace_messages() would DELETE every durable row. Even a submit that
    declares rewind intent needs the second opt-in for that edge.
    """
    replaced = []

    class _FakeDB:
        def replace_messages(
            self,
            key,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            replaced.append((key, list(messages)))

    history = [
        {"_row_id": 101, "role": "user", "content": "first"},
        {"_row_id": 102, "role": "assistant", "content": "ok"},
        {"_row_id": 103, "role": "user", "content": "second"},
        {"_row_id": 104, "role": "assistant", "content": "done"},
    ]
    server._sessions["empty-trunc-sid"] = _session(history=list(history))
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )
    monkeypatch.setattr(
        server, "_start_inflight_turn", lambda *a, **k: pytest.fail("must not start a turn")
    )

    try:
        # Missing confirm â†’ refuse.
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "empty-trunc-sid",
                    "text": "fresh typed message",
                    "truncate_before_row_id": 101,
                    "truncate_before_user_ordinal": 0,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp["error"]["code"] == 4028
        assert "confirm_empty_truncate" in resp["error"]["message"]
        # Explicit falsey values must not satisfy the opt-in either.
        for falsey in (False, 0, "", "false", "no"):
            resp = server.handle_request(
                {
                    "id": "1",
                    "method": "prompt.submit",
                    "params": {
                        "session_id": "empty-trunc-sid",
                        "text": "fresh typed message",
                        "truncate_before_row_id": 101,
                        "truncate_before_user_ordinal": 0,
                        "confirm_truncate": True,
                        "confirm_empty_truncate": falsey,
                    },
                }
            )
            assert resp["error"]["code"] == 4028, falsey
        assert server._sessions["empty-trunc-sid"]["history"] == history
        assert server._sessions["empty-trunc-sid"]["running"] is False
        assert server._sessions["empty-trunc-sid"]["history_version"] == 0
        assert replaced == []
    finally:
        server._sessions.pop("empty-trunc-sid", None)


def test_prompt_submit_empty_truncation_allowed_with_confirm(monkeypatch):
    """Intentional restore/regenerate of the first user turn may wipe history."""

    seen = {}
    replaced = []

    class _Agent:
        def run_conversation(
            self, prompt, conversation_history=None, stream_callback=None, **_kwargs
        ):
            seen["prompt"] = prompt
            seen["history"] = conversation_history
            return {
                "final_response": "regenerated",
                "messages": [
                    *(conversation_history or []),
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "regenerated"},
                ],
            }

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **_thread_options):
            self._target = target

        def start(self):
            self._target()

    class _FakeDB:
        def replace_messages(
            self,
            key,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            replaced.append((key, list(messages)))

    history = [
        {"_row_id": 101, "role": "user", "content": "first"},
        {"_row_id": 102, "role": "assistant", "content": "ok"},
        {"_row_id": 103, "role": "user", "content": "second"},
        {"_row_id": 104, "role": "assistant", "content": "done"},
    ]
    server._sessions["confirm-empty-sid"] = _session(
        agent=_Agent(), history=list(history)
    )

    try:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_get_usage", lambda _a: {})
        monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
        monkeypatch.setattr(server, "_emit", lambda *a: None)
        monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())

        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "confirm-empty-sid",
                    "text": "first",
                    "truncate_before_row_id": 101,
                    "truncate_before_user_ordinal": 0,
                    "confirm_truncate": True,
                    "confirm_empty_truncate": True,
                },
            }
        )
        assert resp.get("result"), f"got error: {resp.get('error')}"
        assert seen["prompt"] == "first"
        assert seen["history"] == []
        assert replaced == [("session-key", [])]
        assert server._sessions["confirm-empty-sid"]["history"] == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "regenerated"},
        ]
    finally:
        server._sessions.pop("confirm-empty-sid", None)


class _StopAfterOneNotificationPoll:
    def __init__(self):
        self._checks = 0

    def is_set(self):
        self._checks += 1
        return self._checks > 1


def test_notification_poller_live_loop_requeues_foreign_completion_for_owner(
    monkeypatch,
):
    """A foreign live-loop dequeue is handed back to its proven owner."""
    import queue as _queue_mod

    from tools.process_registry import process_registry

    delivered = {"a": [], "b": []}
    emitted = []
    session_a = _session(session_key="session-a-live-handoff")
    session_b = _session(session_key="session-b-live-handoff")
    event = {
        "type": "completion",
        "session_id": "proc-live-handoff",
        "session_key": "session-a-live-handoff",
        "command": "echo owner",
        "exit_code": 0,
        "output": "owner",
    }
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    isolated_queue.put(event)
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **_kwargs: emitted.append(args))

    def _deliver(_rid, sid, session, text, **_kw):
        delivered["a" if sid == "sid-a-live-handoff" else "b"].append(text)
        session["running"] = False

    monkeypatch.setattr(server, "_run_prompt_submit", _deliver)
    server._sessions.update(
        {
            "sid-a-live-handoff": session_a,
            "sid-b-live-handoff": session_b,
        }
    )
    process_registry._completion_consumed.discard(event["session_id"])

    try:
        server._notification_poller_loop(
            _StopAfterOneNotificationPoll(), "sid-b-live-handoff", session_b
        )

        assert delivered["b"] == []
        assert emitted == []
        assert isolated_queue.qsize() == 1
        assert isolated_queue.queue[0] is event

        server._notification_poller_loop(
            _StopAfterOneNotificationPoll(), "sid-a-live-handoff", session_a
        )

        assert len(delivered["a"]) == 1
        assert "proc-live-handoff" in delivered["a"][0]
        assert delivered["b"] == []
        assert isolated_queue.empty()
    finally:
        server._sessions.pop("sid-a-live-handoff", None)
        server._sessions.pop("sid-b-live-handoff", None)
        process_registry._completion_consumed.discard(event["session_id"])
        while not isolated_queue.empty():
            isolated_queue.get_nowait()


def test_completion_ownership_lineage_lookup_failure_fails_closed(monkeypatch):
    """A provenance lookup failure cannot turn an addressed event into ours."""
    import queue as _queue_mod

    from tools.process_registry import process_registry

    class _BrokenDB:
        def resolve_resume_session_id(self, _session_key):
            raise RuntimeError("lineage database unavailable")

    session = _session(session_key="unrelated-live-session")
    event = {
        "type": "completion",
        "session_id": "proc-unknown-lineage",
        "session_key": "unknown-parent",
        "command": "echo unknown",
        "exit_code": 0,
        "output": "unknown",
    }
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    isolated_queue.put(event)
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    monkeypatch.setattr(server, "_get_db", lambda: _BrokenDB())

    drained = process_registry.drain_notifications(
        session_key="unrelated-live-session",
        owns_event=lambda candidate: server._session_owns_notification_event(
            "sid-unrelated-live", session, candidate
        ),
    )

    assert drained == []
    assert isolated_queue.qsize() == 1
    assert isolated_queue.get_nowait() is event


@pytest.mark.parametrize(
    "routing",
    [
        {"session_key": "missing-owner-key"},
        {"origin_ui_session_id": "missing-owner-sid"},
    ],
)
def test_notification_poller_live_loop_drops_addressed_orphan(
    monkeypatch, routing
):
    """A live poll never injects an addressed event whose owner is gone."""
    import queue as _queue_mod

    from tools.process_registry import process_registry

    delivered = []
    emitted = []
    session = _session(session_key="unrelated-live-key")
    event = {
        "type": "completion",
        "session_id": "proc-live-orphan",
        "command": "echo orphan",
        "exit_code": 0,
        "output": "orphan",
        **routing,
    }
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    isolated_queue.put(event)
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **_kwargs: emitted.append(args))
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda _rid, _sid, _session, text, **_kw: delivered.append(text),
    )
    server._sessions["sid-live-orphan"] = session
    process_registry._completion_consumed.discard(event["session_id"])

    try:
        server._notification_poller_loop(
            _StopAfterOneNotificationPoll(), "sid-live-orphan", session
        )

        assert delivered == []
        assert emitted == []
        assert isolated_queue.empty()
    finally:
        server._sessions.pop("sid-live-orphan", None)
        process_registry._completion_consumed.discard(event["session_id"])
        while not isolated_queue.empty():
            isolated_queue.get_nowait()


@pytest.mark.parametrize(
    "routing",
    [
        {"session_key": "session-b"},
        {"origin_ui_session_id": "sid_gone"},
    ],
)
def test_notification_poller_drops_orphaned_events(monkeypatch, routing):
    """Addressed completions whose owner is gone are dropped, not hijacked."""
    import queue as _queue_mod

    from tools.process_registry import process_registry

    emitted = []
    delivered = []
    sess = _session(session_key="session-a")
    server._sessions["sid_a"] = sess
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: emitted.append(a))
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda _rid, _sid, _session, text, **_kw: delivered.append(text),
    )
    monkeypatch.setattr(server, "_get_db", lambda: None)

    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    process_registry._completion_consumed.discard("proc_ghost")
    isolated_queue.put(
        {
            "type": "completion",
            "session_id": "proc_ghost",
            "command": "echo from ghost",
            "exit_code": 0,
            "output": "ghost output",
            **routing,
        }
    )

    stop = threading.Event()
    stop.set()

    try:
        server._notification_poller_loop(stop, "sid_a", sess)

        assert [a for a in emitted if a[0] == "status.update"] == []
        assert delivered == []
    finally:
        server._sessions.pop("sid_a", None)
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


@pytest.mark.parametrize(
    ("routing", "resolved_key"),
    [
        ({"session_key": "session-a"}, None),
        (
            {
                "session_key": "stale-durable-key",
                "origin_ui_session_id": "sid_a",
            },
            None,
        ),
        ({"session_key": "old-parent-key"}, "session-a"),
    ],
)
def test_notification_poller_delivers_owned_events(
    monkeypatch, routing, resolved_key
):
    """Direct, UI-origin, and compression-lineage owners are delivered."""
    import queue as _queue_mod

    from tools.process_registry import process_registry

    class _CompressionDB:
        def resolve_resume_session_id(self, key):
            return resolved_key if key == "old-parent-key" and resolved_key else key

    delivered = []
    emitted = []
    sess = _session(session_key="session-a")
    server._sessions["sid_a"] = sess
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: emitted.append(a))
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda _rid, _sid, _session, text, **_kw: delivered.append(text),
    )
    monkeypatch.setattr(server, "_get_db", lambda: _CompressionDB())

    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    process_registry._completion_consumed.discard("proc_mine")
    isolated_queue.put(
        {
            "type": "completion",
            "session_id": "proc_mine",
            "command": "echo mine",
            "exit_code": 0,
            "output": "mine",
            **routing,
        }
    )

    stop = threading.Event()
    stop.set()

    try:
        server._notification_poller_loop(stop, "sid_a", sess)

        status_calls = [a for a in emitted if a[0] == "status.update"]
        assert len(status_calls) == 1
        assert status_calls[0][2]["kind"] == "process"
        assert len(delivered) == 1
        assert "proc_mine" in delivered[0]
    finally:
        server._sessions.pop("sid_a", None)
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


def _configure_immediate_prompt_run(
    monkeypatch, tmp_path, *, immediate_threads=True
):
    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **_kwargs):
            self._target = target

        def start(self):
            if self._target is not None:
                self._target()

        def is_alive(self):
            return False

    if immediate_threads:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda _cols: None)
    monkeypatch.setattr(server, "render_message", lambda _raw, _cols: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_args: None)
    monkeypatch.setattr(server, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_set_session_context", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(
        server, "_sync_session_key_after_compress", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_args: False)
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)
    monkeypatch.setattr(server, "_get_db", lambda: None)


def test_run_prompt_submit_binds_exact_steer_authority_and_resets_contextvars(
    monkeypatch, tmp_path
):
    """The turn thread commissions children with this session generation only."""
    from tools.delegate_tool import _capture_gateway_steer_authority
    from tui_gateway.transport import (
        bind_transport,
        current_transport,
        reset_transport,
    )

    class _Transport:
        def write(self, _obj):
            return True

        def close(self):
            return None

    observed = {}
    owner_transport = _Transport()
    previous_transport = _Transport()
    previous_record = {"session_key": "previous-generation"}

    class _CapturingAgent(_RecordingAgent):
        def run_conversation(self, prompt, **kwargs):
            authority = _capture_gateway_steer_authority("sid-owner")
            observed["transport"] = authority[0]
            observed["record"] = authority[1]
            return super().run_conversation(prompt, **kwargs)

    _configure_immediate_prompt_run(monkeypatch, tmp_path)
    session = _session(
        session_key="session-owner",
        agent=_CapturingAgent([]),
        running=True,
        transport=owner_transport,
    )
    server._sessions["sid-owner"] = session
    transport_token = bind_transport(previous_transport)
    record_token = server._current_runtime_session_record.set(previous_record)
    try:
        server._run_prompt_submit("rid-owner", "sid-owner", session, "commission")

        assert observed == {"transport": owner_transport, "record": session}
        assert current_transport() is previous_transport
        assert server._current_runtime_session_record.get() is previous_record
    finally:
        server._current_runtime_session_record.reset(record_token)
        reset_transport(transport_token)
        server._sessions.pop("sid-owner", None)


class _RecordingAgent:
    model = "test-model"
    provider = "test-provider"

    def __init__(self, turns):
        self._turns = turns

    def clear_interrupt(self):
        return None

    def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
        self._turns.append(prompt)
        return {"final_response": "", "messages": []}


def test_run_prompt_submit_rejects_worker_when_close_wins_publication(
    monkeypatch, tmp_path
):
    """A close claimed during message.start must prevent the worker from running."""
    _configure_immediate_prompt_run(monkeypatch, tmp_path, immediate_threads=False)
    emit_entered = threading.Event()
    release_emit = threading.Event()
    dispatch_results = []
    turns = []
    popped = []
    sid = "close-wins-publication"
    session = _session(
        session_key="close-wins-publication-key",
        agent=_RecordingAgent(turns),
        running=True,
    )

    def _blocking_emit(event, *_args, **_kwargs):
        if event == "message.start":
            emit_entered.set()
            assert release_emit.wait(timeout=2.0)

    monkeypatch.setattr(server, "_emit", _blocking_emit)
    server._sessions[sid] = session
    dispatch_thread = threading.Thread(
        target=lambda: dispatch_results.append(
            server._run_prompt_submit("rid", sid, session, "turn")
        )
    )

    try:
        dispatch_thread.start()
        assert emit_entered.wait(timeout=1.0)
        popped.append(server._pop_session_by_id(sid))
        assert popped == [session]
        release_emit.set()
        dispatch_thread.join(timeout=2.0)
    finally:
        release_emit.set()
        dispatch_thread.join(timeout=2.0)
        run_thread = session.get("_run_thread")
        if run_thread is not None and run_thread.is_alive():
            run_thread.join(timeout=2.0)
        server._sessions.pop(sid, None)

    assert dispatch_results == [False]
    assert session["running"] is False
    assert turns == []


@pytest.mark.parametrize("exit_code", [0, 7])
def test_run_prompt_submit_requeues_foreign_completion(
    monkeypatch, tmp_path, exit_code
):
    import queue as _queue_mod

    from tools.process_registry import process_registry

    _configure_immediate_prompt_run(monkeypatch, tmp_path)
    turns = []
    session_a = _session(session_key="session-a")
    session_b = _session(
        session_key="session-b",
        agent=_RecordingAgent(turns),
        running=True,
    )
    event = {
        "type": "completion",
        "session_id": f"proc_foreign_{exit_code}",
        "session_key": "session-a",
        "command": "safe-test-command",
        "exit_code": exit_code,
        "output": "foreign",
    }
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    isolated_queue.put(event)
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    server._sessions["sid_a"] = session_a
    server._sessions["sid_b"] = session_b

    try:
        server._run_prompt_submit("rid-b", "sid_b", session_b, "session-b-turn")

        assert turns == ["session-b-turn"]
        assert isolated_queue.get_nowait() == event
        assert isolated_queue.empty()
    finally:
        server._sessions.pop("sid_a", None)
        server._sessions.pop("sid_b", None)
        process_registry._completion_consumed.discard(event["session_id"])


def test_run_prompt_submit_delivers_completion_observed_by_poll(monkeypatch, tmp_path):
    import queue as _queue_mod

    from tools.process_registry import process_registry

    _configure_immediate_prompt_run(monkeypatch, tmp_path)
    turns = []
    session = _session(
        session_key="session-a",
        agent=_RecordingAgent(turns),
        running=True,
    )
    event = {
        "type": "completion",
        "session_id": "proc_polled",
        "session_key": "session-a",
        "command": "safe-test-command",
        "exit_code": 0,
        "output": "observed but not consumed",
    }
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    isolated_queue.put(event)
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    process_registry._completion_consumed.discard(event["session_id"])
    process_registry._poll_observed.add(event["session_id"])
    server._sessions["sid_a"] = session

    try:
        server._run_prompt_submit("rid-a", "sid_a", session, "session-a-turn")

        assert turns[0] == "session-a-turn"
        assert len(turns) == 2
        assert "proc_polled" in turns[1]
        assert isolated_queue.empty()
    finally:
        server._sessions.pop("sid_a", None)
        process_registry._completion_consumed.discard(event["session_id"])
        process_registry._poll_observed.discard(event["session_id"])


def test_run_prompt_submit_requeues_all_unstarted_notifications_with_real_threading(
    monkeypatch, tmp_path
):
    import queue as _queue_mod

    from tools.process_registry import process_registry

    _configure_immediate_prompt_run(
        monkeypatch, tmp_path, immediate_threads=False
    )
    real_thread_class = threading.Thread
    threads = []
    nested_started = threading.Event()
    release_nested = threading.Event()
    turns = []

    def _recording_thread(*args, **kwargs):
        thread = real_thread_class(*args, **kwargs)
        threads.append(thread)
        return thread

    class _BlockingNotificationAgent(_RecordingAgent):
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            turns.append(prompt)
            if "proc_batch_1" in prompt:
                nested_started.set()
                if not release_nested.wait(timeout=5):
                    raise TimeoutError("notification turn was not released")
            return {"final_response": "", "messages": []}

    monkeypatch.setattr(server.threading, "Thread", _recording_thread)
    session = _session(
        session_key="session-a",
        agent=_BlockingNotificationAgent(turns),
        running=True,
    )
    events = [
        {
            "type": "completion",
            "session_id": f"proc_batch_{index}",
            "session_key": "session-a",
            "command": "safe-test-command",
            "exit_code": 0,
            "output": f"owned-{index}",
        }
        for index in range(1, 4)
    ]
    # Consecutive completions share one turn (#104671); a watch_match is a turn
    # barrier, so it is the in-flight turn behind which batch_2/batch_3 must survive.
    events[0].update(type="watch_match", pattern="owned-1")
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    for event in events:
        isolated_queue.put(event)
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    server._sessions["sid_a"] = session

    try:
        server._run_prompt_submit("rid-a", "sid_a", session, "session-a-turn")

        assert nested_started.wait(timeout=5)
        threads[0].join(timeout=5)
        assert not threads[0].is_alive()
        # Membership, not order: the completion_queue is process-global, and
        # notification pollers leaked by earlier session.init tests in this
        # file legitimately steal-and-requeue foreign-session events (see
        # _notification_poller_loop's belongs-elsewhere branch), rotating the
        # queue. The requeue contract is that batch_2 and batch_3 both remain
        # queued (never consumed) while batch_1's turn is in flight â€” so drain
        # with a deadline (an event may be transiently held by a poller
        # mid-cycle) and assert exactly {batch_2, batch_3} come back.
        queued: dict = {}
        deadline = time.time() + 5.0
        while time.time() < deadline and set(queued) != {
            "proc_batch_2",
            "proc_batch_3",
        }:
            try:
                evt = isolated_queue.get(timeout=0.1)
            except _queue_mod.Empty:
                continue
            queued[evt["session_id"]] = evt
        assert set(queued) == {"proc_batch_2", "proc_batch_3"}
    finally:
        release_nested.set()
        for thread in threads:
            thread.join(timeout=5)
        server._sessions.pop("sid_a", None)
        while not isolated_queue.empty():
            isolated_queue.get_nowait()
        for event in events:
            process_registry._completion_consumed.discard(event["session_id"])
            process_registry._poll_observed.discard(event["session_id"])


def test_run_prompt_submit_delivers_completion_owned_through_compression_lineage(
    monkeypatch, tmp_path
):
    import queue as _queue_mod

    from tools.process_registry import process_registry

    class _CompressionDB:
        def resolve_resume_session_id(self, key):
            return "new-child-key" if key == "old-parent-key" else key

    _configure_immediate_prompt_run(monkeypatch, tmp_path)
    monkeypatch.setattr(server, "_get_db", lambda: _CompressionDB())
    ownership_checks = []
    original_owns_event = server._session_owns_notification_event

    def _record_ownership_check(sid, checked_session, checked_event):
        ownership_checks.append(checked_event["session_id"])
        return original_owns_event(sid, checked_session, checked_event)

    monkeypatch.setattr(
        server, "_session_owns_notification_event", _record_ownership_check
    )
    turns = []
    session = _session(
        session_key="new-child-key",
        agent=_RecordingAgent(turns),
        running=True,
    )
    event = {
        "type": "completion",
        "session_id": "proc_precompression",
        "session_key": "old-parent-key",
        "command": "safe-test-command",
        "exit_code": 0,
        "output": "owned",
    }
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    isolated_queue.put(event)
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    server._sessions["sid_b"] = session

    try:
        server._run_prompt_submit("rid-b", "sid_b", session, "session-b-turn")

        assert turns[0] == "session-b-turn"
        assert len(turns) == 2
        assert "proc_precompression" in turns[1]
        assert ownership_checks == ["proc_precompression"]
        assert isolated_queue.empty()
    finally:
        server._sessions.pop("sid_b", None)
        process_registry._completion_consumed.discard(event["session_id"])


def test_run_prompt_submit_prefers_origin_ui_session_id(monkeypatch, tmp_path):
    import queue as _queue_mod

    from tools.process_registry import process_registry

    _configure_immediate_prompt_run(monkeypatch, tmp_path)
    ownership_checks = []
    original_owns_event = server._session_owns_notification_event

    def _record_ownership_check(sid, checked_session, checked_event):
        ownership_checks.append(checked_event["session_id"])
        return original_owns_event(sid, checked_session, checked_event)

    monkeypatch.setattr(
        server, "_session_owns_notification_event", _record_ownership_check
    )
    turns = []
    session = _session(
        session_key="current-key",
        agent=_RecordingAgent(turns),
        running=True,
    )
    event = {
        "type": "completion",
        "session_id": "proc_origin_owned",
        "session_key": "stale-durable-key",
        "origin_ui_session_id": "sid_b",
        "command": "safe-test-command",
        "exit_code": 0,
        "output": "owned",
    }
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    isolated_queue.put(event)
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    server._sessions["sid_b"] = session

    try:
        server._run_prompt_submit("rid-b", "sid_b", session, "session-b-turn")

        assert turns[0] == "session-b-turn"
        assert len(turns) == 2
        assert "proc_origin_owned" in turns[1]
        assert ownership_checks == ["proc_origin_owned"]
        assert isolated_queue.empty()
    finally:
        server._sessions.pop("sid_b", None)
        process_registry._completion_consumed.discard(event["session_id"])



    """session.create must NOT eagerly write a DB row.

    Every TUI/desktop launch opens a session here just to paint the composer;
    eagerly creating a row left an empty "Untitled" session behind for every
    launch the user never typed into. The row is created lazily on first prompt.
    """
    created = []

    class _FakeDB:
        def create_session(self, *args, **kwargs):
            created.append((args, kwargs))

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(
        server.threading,
        "Timer",
        lambda *a, **k: types.SimpleNamespace(daemon=False, start=lambda: None),
    )

    resp = server.handle_request(
        {"id": "1", "method": "session.create", "params": {"cols": 80}}
    )
    sid = resp["result"]["session_id"]
    try:
        assert resp["result"]["stored_session_id"]
        assert created == [], "session.create should not persist an empty DB row"
    finally:
        server._sessions.pop(sid, None)


def test_ensure_session_db_row_persists_explicit_cwd(monkeypatch, tmp_path):
    """An explicitly chosen workspace is persisted as the session cwd."""
    created = []

    class _FakeDB:
        def create_session(self, key, source=None, model=None, model_config=None, parent_session_id=None, cwd=None, profile_name=None, user_id=None):
            created.append(
                {"key": key, "source": source, "model": model, "model_config": model_config, "cwd": cwd}
            )

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)

    server._ensure_session_db_row({"session_key": "k1", "cwd": str(tmp_path), "explicit_cwd": True})

    assert created == [
        {"key": "k1", "source": "tui", "model": "test-model", "model_config": None, "cwd": str(tmp_path)}
    ]


def test_ensure_session_db_row_persists_session_source(monkeypatch):
    created = []

    class _FakeDB:
        def create_session(self, key, source=None, model=None, model_config=None, parent_session_id=None, cwd=None, profile_name=None, user_id=None):
            created.append(
                {"key": key, "source": source, "model": model, "model_config": model_config, "cwd": cwd}
            )

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")

    server._ensure_session_db_row({"session_key": "k1", "source": "tool"})

    assert created == [
        {"key": "k1", "source": "tool", "model": "test-model", "model_config": None, "cwd": None}
    ]


def test_ensure_session_db_row_records_a_terminal_workspace(monkeypatch, tmp_path):
    """A terminal session's directory IS its workspace, so the row records it.

    The user cd'd there before running hermes. Leaving it null stranded the row
    with no cwd and no git_repo_root, so the sidebar could never place the
    session under its project.
    """
    created = []

    class _FakeDB:
        def create_session(self, key, source=None, model=None, model_config=None, parent_session_id=None, cwd=None, profile_name=None, user_id=None):
            created.append(
                {"key": key, "source": source, "model": model, "model_config": model_config, "cwd": cwd}
            )

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)

    server._ensure_session_db_row({"session_key": "k1", "cwd": str(tmp_path)})

    assert created == [
        {"key": "k1", "source": "tui", "model": "test-model", "model_config": None, "cwd": str(tmp_path)}
    ]


def test_ensure_session_db_row_defaults_desktop_to_no_workspace(monkeypatch, tmp_path):
    """The desktop launches from wherever the bundle was opened, so an unpicked
    cwd is an artifact â€” those chats stay null and group under "No workspace"."""
    created = []

    class _FakeDB:
        def create_session(self, key, source=None, model=None, model_config=None, parent_session_id=None, cwd=None, profile_name=None, user_id=None):
            created.append(
                {"key": key, "source": source, "model": model, "model_config": model_config, "cwd": cwd}
            )

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")

    server._ensure_session_db_row({"session_key": "k1", "source": "desktop", "cwd": str(tmp_path)})

    assert created == [
        {"key": "k1", "source": "desktop", "model": "test-model", "model_config": None, "cwd": None}
    ]


def test_ensure_session_db_row_persists_session_model_override(monkeypatch):
    """The session's composer pick (model + effort + fast) must own the DB row.

    Regression for the "switched to gpt-5.5, reconnect snapped back to opus"
    bug: the row was created with the global default and won the INSERT-OR-IGNORE
    race, so resume rebuilt from the global model and silently reverted the
    chat. The override model + a model_config carrying provider/reasoning/
    service_tier must be persisted so session.resume restores all three.
    """
    created = []

    class _FakeDB:
        def create_session(self, key, source=None, model=None, model_config=None, parent_session_id=None, cwd=None, profile_name=None, user_id=None):
            created.append(
                {"key": key, "model": model, "model_config": model_config, "cwd": cwd}
            )

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_resolve_model", lambda: "global/default")

    server._ensure_session_db_row(
        {
            "session_key": "k1",
            "model_override": {"model": "openai/gpt-5.5", "provider": "openrouter"},
            "create_reasoning_override": {"effort": "high"},
            "create_service_tier_override": "priority",
        }
    )

    assert len(created) == 1
    row = created[0]
    assert row["model"] == "openai/gpt-5.5"
    assert row["model_config"]["model"] == "openai/gpt-5.5"
    assert row["model_config"]["provider"] == "openrouter"
    assert row["model_config"]["reasoning_config"] == {"effort": "high"}
    assert row["model_config"]["service_tier"] == "priority"


def test_ensure_session_db_row_no_override_uses_global(monkeypatch):
    """A chat that made no explicit pick falls back to the global model and
    writes no model_config (so it tracks the profile default)."""
    created = []

    class _FakeDB:
        def create_session(self, key, source=None, model=None, model_config=None, parent_session_id=None, cwd=None, profile_name=None, user_id=None):
            created.append({"model": model, "model_config": model_config})

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_resolve_model", lambda: "global/default")

    server._ensure_session_db_row({"session_key": "k1", "model_override": None})

    assert created == [{"model": "global/default", "model_config": None}]


def test_ensure_session_db_row_stamps_profile_name(monkeypatch, tmp_path):
    """A profile session's row carries its owning profile_name, so unified
    multi-profile aggregation never has to guess from which state.db file the
    row happened to be read (the cross-profile session-jump bug)."""
    profile_home = tmp_path / "profiles" / "mlperf"
    profile_home.mkdir(parents=True)
    created = []

    class _ProfileDB:
        def __init__(self, db_path=None):
            created.append({"db_path": db_path})

        def create_session(self, key, **kwargs):
            created[-1].update({"key": key, "profile_name": kwargs.get("profile_name")})

        def close(self):
            pass

    monkeypatch.setattr("hermes_state_registry.acquire", _ProfileDB)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")

    server._ensure_session_db_row(
        {"session_key": "k1", "profile_home": str(profile_home)}
    )

    assert created and created[0]["key"] == "k1"
    assert created[0]["profile_name"] == "mlperf"
    assert created[0]["db_path"] == profile_home / "state.db"


def test_ensure_session_db_row_stamps_launch_profile_name(monkeypatch):
    """A launch-profile session row is stamped with the ACTUAL profile name,
    never NULL. NULL-as-launch-profile rows vanish from the desktop sidebar
    (profile-keyed matching) and break @session:<profile>/<id> deep links, and
    the #94724 one-shot backfill cannot keep repairing rows minted after it
    ran (#99222)."""
    created = []

    class _FakeDB:
        def create_session(self, key, **kwargs):
            created.append({"key": key, "profile_name": kwargs.get("profile_name")})

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")

    server._ensure_session_db_row({"session_key": "k1"})

    assert created and created[0]["key"] == "k1"
    assert created[0]["profile_name"] == "default"


def test_session_title_clears_pending_after_persist(monkeypatch):
    class _FakeDB:
        def __init__(self):
            self.title = "old"

        def get_session_title(self, _key):
            return self.title

        def get_session(self, _key):
            return {"id": _key, "title": self.title}

        def set_session_title(self, _key, title):
            self.title = title
            return True

    db = _FakeDB()
    emitted = []
    server._sessions["sid"] = _session(pending_title="stale")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "fresh"},
            }
        )

        assert resp["result"]["pending"] is False
        assert resp["result"]["title"] == "fresh"
        assert server._sessions["sid"]["pending_title"] is None
        assert emitted[-1][0:2] == ("session.info", "sid")
        assert emitted[-1][2]["title"] == "fresh"
    finally:
        server._sessions.pop("sid", None)


def test_session_title_does_not_queue_noop_when_row_exists(monkeypatch):
    class _FakeDB:
        def __init__(self):
            self.title = "same title"

        def get_session_title(self, _key):
            return self.title

        def get_session(self, _key):
            return {"id": _key, "title": self.title}

        def set_session_title(self, _key, _title):
            # Simulate sqlite UPDATE rowcount==0 for no-op update.
            return False

    server._sessions["sid"] = _session(pending_title="stale")
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "same title"},
            }
        )

        assert resp["result"]["pending"] is False
        assert resp["result"]["title"] == "same title"
        assert server._sessions["sid"]["pending_title"] is None
    finally:
        server._sessions.pop("sid", None)


def test_session_title_get_falls_back_to_pending_when_db_read_throws(monkeypatch):
    class _FakeDB:
        def get_session_title(self, _key):
            raise RuntimeError("db temporarily locked")

    server._sessions["sid"] = _session(pending_title="queued title")
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.title", "params": {"session_id": "sid"}}
        )
        assert resp["result"]["title"] == "queued title"
    finally:
        server._sessions.pop("sid", None)


def test_session_title_get_retries_persist_for_pending_title(monkeypatch):
    class _FakeDB:
        def __init__(self):
            self.title = ""

        def get_session_title(self, _key):
            return self.title

        def set_session_title(self, _key, title):
            self.title = title
            return True

        def get_session(self, _key):
            return {"id": _key, "title": self.title}

    db = _FakeDB()
    server._sessions["sid"] = _session(pending_title="queued title")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.title", "params": {"session_id": "sid"}}
        )
        assert resp["result"]["title"] == "queued title"
        assert server._sessions["sid"]["pending_title"] is None
    finally:
        server._sessions.pop("sid", None)


def test_session_title_get_retries_pending_even_when_db_has_title(monkeypatch):
    class _FakeDB:
        def __init__(self):
            self.title = "auto title"

        def get_session_title(self, _key):
            return self.title

        def set_session_title(self, _key, title):
            self.title = title
            return True

        def get_session(self, _key):
            return {"id": _key, "title": self.title}

    db = _FakeDB()
    server._sessions["sid"] = _session(pending_title="queued title")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.title", "params": {"session_id": "sid"}}
        )
        assert resp["result"]["title"] == "queued title"
        assert server._sessions["sid"]["pending_title"] is None
    finally:
        server._sessions.pop("sid", None)


def test_session_title_rejects_empty_title_with_specific_error_code(monkeypatch):
    class _FakeDB:
        def get_session_title(self, _key):
            return ""

    server._sessions["sid"] = _session()
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "   "},
            }
        )
        assert "error" in resp
        assert resp["error"]["code"] == 4021
    finally:
        server._sessions.pop("sid", None)


def test_session_title_set_maps_valueerror_to_user_error(monkeypatch):
    class _FakeDB:
        def get_session_title(self, _key):
            return ""

        def get_session(self, _key):
            return {"id": _key}

        def set_session_title(self, _key, _title):
            raise ValueError("Title already in use")

    server._sessions["sid"] = _session()
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "dup"},
            }
        )
        assert "error" in resp
        assert resp["error"]["code"] == 4022
        assert "already in use" in resp["error"]["message"]
    finally:
        server._sessions.pop("sid", None)


def test_session_title_set_errors_when_row_lookup_fails_after_noop(monkeypatch):
    class _FakeDB:
        def get_session_title(self, _key):
            return ""

        def get_session(self, _key):
            raise RuntimeError("row lookup failed")

        def set_session_title(self, _key, _title):
            return False

    server._sessions["sid"] = _session()
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "fresh"},
            }
        )
        assert "error" in resp
        assert resp["error"]["code"] == 5007
        assert "row lookup failed" in resp["error"]["message"]
    finally:
        server._sessions.pop("sid", None)


def test_session_create_drops_pending_title_on_valueerror(monkeypatch):
    """When set_session_title raises ValueError during post-message title flush,
    pending_title should be dropped (non-retryable). Updated for post-#18370
    lazy session creation where title is applied post-first-message.
    """

    class _Agent:
        session_id = "test-session"
        model = "x"
        provider = "openrouter"
        base_url = ""
        api_key = ""
        _cached_system_prompt = ""

        def run_conversation(self, prompt, **kw):
            return {
                "final_response": "ok",
                "messages": [{"role": "assistant", "content": "ok"}],
            }

    class _FakeDB:
        def set_session_title(self, _key, _title):
            raise ValueError("Title already in use")

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target

        def start(self):
            self._target()

    agent = _Agent()
    session = {
        "agent": agent,
        "session_key": "test-session",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "pending_title": "duplicate title",
    }

    server._sessions["sid"] = session
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setattr(
        server, "_sync_session_key_after_compress", lambda *a, **kw: None
    )
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)

    try:
        server.handle_request(
            {"id": "1", "method": "prompt.submit", "params": {"session_id": "sid", "text": "hello"}}
        )
        assert session["pending_title"] is None
    finally:
        server._sessions.pop("sid", None)


def test_config_set_yolo_toggles_session_scope():
    from tools.approval import clear_session, is_session_yolo_enabled

    server._sessions["sid"] = _session()
    try:
        resp_on = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "yolo"},
            }
        )
        assert resp_on["result"]["value"] == "1"
        assert is_session_yolo_enabled("session-key") is True

        resp_off = server.handle_request(
            {
                "id": "2",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "yolo"},
            }
        )
        assert resp_off["result"]["value"] == "0"
        assert is_session_yolo_enabled("session-key") is False
    finally:
        clear_session("session-key")
        server._sessions.clear()


def test_config_set_yolo_stale_session_id_is_refused_not_process_scoped(monkeypatch):
    """A runtime id the backend no longer holds must answer 4001 so the client resumes, not flip
    the process HERMES_YOLO_MODE that every child spawned afterwards inherits."""
    monkeypatch.setenv("HERMES_YOLO_MODE", "0")  # setenv, not delenv: undo must also drop a leaked "1"

    with patch.dict(server._sessions, {}, clear=True):
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "reaped-sid", "key": "yolo", "value": "1"},
            }
        )

    assert resp.get("error", {}).get("code") == 4001, resp
    assert os.environ["HERMES_YOLO_MODE"] == "0"


def test_config_set_yolo_global_scope_writes_approvals_mode(tmp_path, monkeypatch):
    """Shift+click the desktop zap -> scope="global" flips persistent approvals.mode."""
    import hermes_yaml as yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"approvals": {"mode": "manual"}}))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    resp_on = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "yolo", "scope": "global"},
        }
    )
    assert resp_on["result"]["value"] == "1"
    assert resp_on["result"]["scope"] == "global"
    assert yaml.safe_load(cfg_path.read_text())["approvals"]["mode"] == "off"

    resp_off = server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {"key": "yolo", "scope": "global"},
        }
    )
    assert resp_off["result"]["value"] == "0"
    assert yaml.safe_load(cfg_path.read_text())["approvals"]["mode"] == "manual"


def test_config_get_approval_mode_uses_smart_default_when_key_is_missing(
    tmp_path, monkeypatch
):
    import hermes_yaml as yaml

    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    # Point the canonical resolver (load_config â†’ env HERMES_HOME) at the
    # temp home too, so the smart default is asserted against THIS config
    # rather than whatever the developer's real ~/.hermes happens to hold.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"approvals": {"timeout": 15}})
    )

    response = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "approvals.mode"}}
    )
    assert response["result"]["value"] == "smart"


def test_config_get_approval_mode_fails_safe_to_manual_for_invalid_explicit_value(
    tmp_path, monkeypatch
):
    import hermes_yaml as yaml

    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    # _load_approval_mode delegates to the canonical resolver in
    # tools.approval, which reads via hermes_cli.config.load_config â€”
    # that path resolves HERMES_HOME from the environment, not
    # server._hermes_home.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"approvals": {"mode": "sometimes"}})
    )

    response = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "approvals.mode"}}
    )
    assert response["result"]["value"] == "manual"


def test_config_get_approval_mode_normalizes_yaml_off(tmp_path, monkeypatch):
    import hermes_yaml as yaml

    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    # See fail-safe test above: the canonical resolver reads via
    # load_config, which resolves HERMES_HOME from the environment.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"approvals": {"mode": False}})
    )

    response = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "approvals.mode"}}
    )
    assert response["result"]["value"] == "off"


def test_config_set_approval_mode_persists_three_way_value_and_emits_live_status(
    tmp_path, monkeypatch
):
    import hermes_yaml as yaml

    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    # config.set writes via server._hermes_home, but the post-write
    # session.info emit resolves the effective mode through the canonical
    # tools.approval resolver (load_config â†’ env HERMES_HOME).
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    server._sessions["sid"] = {"agent": object(), "session_key": "profile-session"}

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"key": "approvals.mode", "value": "manual"},
            }
        )
    finally:
        server._sessions.clear()

    assert resp["result"] == {"key": "approvals.mode", "value": "manual"}
    assert yaml.safe_load((tmp_path / "config.yaml").read_text())["approvals"]["mode"] == "manual"
    assert emitted and emitted[0][0:2] == ("session.info", "sid")
    assert emitted[0][2]["approval_mode"] == "manual"


def test_pet_gallery_quoted_false_enabled_reports_disabled(tmp_path, monkeypatch):
    """display.pet.enabled: "false" (quoted) must report enabled=False.

    The old check was bool(value) â€” bool('false') is True, so a hand-edited
    quoted YAML value kept the petdex mascot enabled against the operator's
    explicit intent.
    """
    import hermes_yaml as yaml

    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"display": {"pet": {"enabled": "false"}}})
    )

    response = server.handle_request(
        {"id": "1", "method": "pet.gallery", "params": {}}
    )
    assert response["result"]["enabled"] is False


def test_pet_info_known_revision_elides_spritesheet(monkeypatch):
    """pet.info with a matching knownRevision must not resend the sheet bytes.

    The spritesheet payload is multi-MB; resending it on every backstop
    refresh stalls the WS write loop (#54730). A caller passing the revision
    it already holds gets metadata plus spritesheetUnchanged instead.
    """

    class _FakePet:
        slug = "codex"
        display_name = "Codex"
        exists = True
        spritesheet = None

    payload = {
        "slug": "codex",
        "displayName": "Codex",
        "mime": "image/png",
        "spritesheetBase64": "A" * 1024,
        "spritesheetRevision": "123:456",
        "frameW": 192,
        "frameH": 208,
        "scale": 0.33,
    }

    monkeypatch.setattr(server, "_pet_active_selection", lambda: (True, _FakePet(), 0.33))
    monkeypatch.setattr(server, "_pet_sprite_payload", lambda pet, *, scale: dict(payload))

    # Matching revision: bytes elided, unchanged marker set.
    resp = server.handle_request(
        {"id": "1", "method": "pet.info", "params": {"knownRevision": "123:456"}}
    )
    assert resp["result"]["enabled"] is True
    assert "spritesheetBase64" not in resp["result"]
    assert resp["result"]["spritesheetUnchanged"] is True
    assert resp["result"]["spritesheetRevision"] == "123:456"

    # Stale revision: full payload still flows.
    resp = server.handle_request(
        {"id": "2", "method": "pet.info", "params": {"knownRevision": "999:999"}}
    )
    assert resp["result"]["spritesheetBase64"] == "A" * 1024
    assert "spritesheetUnchanged" not in resp["result"]

    # No revision (legacy callers): full payload.
    resp = server.handle_request({"id": "3", "method": "pet.info", "params": {}})
    assert resp["result"]["spritesheetBase64"] == "A" * 1024




def test_config_set_approval_mode_rejects_unknown_value():
    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "approvals.mode", "value": "sometimes"},
        }
    )

    assert resp["error"]["code"] == 4002


def test_config_set_yolo_global_scope_honors_explicit_value(tmp_path, monkeypatch):
    """An explicit value pins global approvals.mode regardless of prior state."""
    import hermes_yaml as yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"approvals": {"mode": "manual"}}))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "yolo", "scope": "global", "value": "1"},
        }
    )
    assert resp["result"]["value"] == "1"
    assert yaml.safe_load(cfg_path.read_text())["approvals"]["mode"] == "off"

    # Setting it on again is idempotent â€” stays off.
    resp_again = server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {"key": "yolo", "scope": "global", "value": "1"},
        }
    )
    assert resp_again["result"]["value"] == "1"
    assert yaml.safe_load(cfg_path.read_text())["approvals"]["mode"] == "off"


def test_config_set_fast_updates_live_agent_session_scoped(monkeypatch):
    """A session-targeted fast toggle updates the live agent + pins the
    per-session override, and NEVER writes global config â€” the desktop's
    per-model presets call this on every model pick, and a global write
    flipped the tier for every other session/profile (the "switch one
    session, switches everywhere" class)."""
    writes = []
    emits = []
    agent = types.SimpleNamespace(
        model="openai/gpt-5.4",
        request_overrides={"foo": "bar", "speed": "slow"},
        service_tier=None,
    )
    session = _session(agent=agent)
    server._sessions["sid"] = session

    monkeypatch.setattr(
        server, "_write_config_key", lambda path, value: writes.append((path, value))
    )
    monkeypatch.setattr(server, "_session_info", lambda _agent, *a: {"model": "x"})
    monkeypatch.setattr(server, "_emit", lambda *args: emits.append(args))
    monkeypatch.setattr(
        "hermes_cli.models.resolve_fast_mode_overrides",
        lambda _model_id, **_route: {"service_tier": "priority"},
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "fast", "value": "fast"},
            }
        )
        assert resp["result"]["value"] == "fast"
        assert agent.service_tier == "priority"
        assert agent.request_overrides == {
            "foo": "bar",
            "service_tier": "priority",
        }
        assert session["create_service_tier_override"] == "priority"
        assert writes == []
        assert ("session.info", "sid", {"model": "x"}) in emits

        resp_normal = server.handle_request(
            {
                "id": "2",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "fast", "value": "normal"},
            }
        )
        assert resp_normal["result"]["value"] == "normal"
        assert agent.service_tier is None
        assert agent.request_overrides == {"foo": "bar"}
        # "" (not absent) so a rebuild pins normal instead of falling back to
        # the global default.
        assert session["create_service_tier_override"] == ""
        assert writes == []
    finally:
        server._sessions.pop("sid", None)


def test_config_set_fast_status_is_non_mutating(monkeypatch):
    writes = []
    emits = []
    agent = types.SimpleNamespace(service_tier="priority")
    server._sessions["sid"] = _session(agent=agent)

    monkeypatch.setattr(
        server, "_write_config_key", lambda path, value: writes.append((path, value))
    )
    monkeypatch.setattr(server, "_emit", lambda *args: emits.append(args))

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "fast", "value": "status"},
            }
        )
        assert resp["result"]["value"] == "fast"
        assert writes == []
        assert emits == []
    finally:
        server._sessions.pop("sid", None)


def test_config_set_fast_rejects_unsupported_model(monkeypatch):
    writes = []
    agent = types.SimpleNamespace(
        model="unsupported-model",
        request_overrides={},
        service_tier=None,
    )
    server._sessions["sid"] = _session(agent=agent)

    monkeypatch.setattr(
        server, "_write_config_key", lambda path, value: writes.append((path, value))
    )
    monkeypatch.setattr(
        "hermes_cli.models.resolve_fast_mode_overrides",
        lambda _model_id, **_route: None,
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "fast", "value": "fast"},
            }
        )
        assert resp["error"]["code"] == 4002
        assert agent.service_tier is None
        assert agent.request_overrides == {}
        assert writes == []
    finally:
        server._sessions.pop("sid", None)


def test_config_set_fast_rejects_missing_model(monkeypatch):
    writes = []
    agent = types.SimpleNamespace(
        model="",
        request_overrides={},
        service_tier=None,
    )
    server._sessions["sid"] = _session(agent=agent)

    monkeypatch.setattr(
        server, "_write_config_key", lambda path, value: writes.append((path, value))
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "fast", "value": "fast"},
            }
        )
        assert resp["error"]["code"] == 4002
        assert agent.service_tier is None
        assert agent.request_overrides == {}
        assert writes == []
    finally:
        server._sessions.pop("sid", None)


def test_config_busy_get_and_set(monkeypatch):
    writes = []

    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"display": {"busy_input_mode": "steer"}},
    )
    monkeypatch.setattr(
        server, "_write_config_key", lambda path, value: writes.append((path, value))
    )

    get_resp = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "busy"}}
    )
    assert get_resp["result"]["value"] == "steer"

    set_resp = server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {"key": "busy", "value": "interrupt"},
        }
    )
    assert set_resp["result"]["value"] == "interrupt"
    assert ("display.busy_input_mode", "interrupt") in writes


def test_config_set_yolo_process_scope_treats_false_like_env_as_disabled(monkeypatch):
    monkeypatch.setenv("HERMES_YOLO_MODE", "false")

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "yolo"},
        }
    )

    assert resp["result"]["value"] == "1"
    assert os.environ.get("HERMES_YOLO_MODE") == "1"


def test_config_get_statusbar_survives_non_dict_display(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"display": "broken"})

    resp = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "statusbar"}}
    )

    assert resp["result"]["value"] == "top"


def test_config_get_busy_survives_non_dict_display(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"display": "broken"})

    resp = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "busy"}}
    )

    assert resp["result"]["value"] == "interrupt"


def test_config_set_statusbar_survives_non_dict_display(tmp_path, monkeypatch):
    import hermes_yaml as yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"display": "broken"}))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "statusbar", "value": "bottom"},
        }
    )

    assert resp["result"]["value"] == "bottom"
    saved = yaml.safe_load(cfg_path.read_text())
    assert saved["display"]["tui_statusbar"] == "bottom"


def test_config_set_details_mode_pins_all_sections(tmp_path, monkeypatch):
    import hermes_yaml as yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {"display": {"sections": {"tools": "expanded", "activity": "hidden"}}}
        )
    )
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "details_mode", "value": "collapsed"},
        }
    )

    assert resp["result"] == {"key": "details_mode", "value": "collapsed"}
    saved = yaml.safe_load(cfg_path.read_text())
    assert saved["display"]["details_mode"] == "collapsed"
    assert saved["display"]["sections"] == {
        "thinking": "collapsed",
        "tools": "collapsed",
        "subagents": "collapsed",
        "activity": "collapsed",
    }


def test_config_set_section_writes_per_section_override(tmp_path, monkeypatch):
    import hermes_yaml as yaml

    cfg_path = tmp_path / "config.yaml"
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "details_mode.activity", "value": "hidden"},
        }
    )

    assert resp["result"] == {"key": "details_mode.activity", "value": "hidden"}
    saved = yaml.safe_load(cfg_path.read_text())
    assert saved["display"]["sections"] == {"activity": "hidden"}


def test_config_set_section_clears_override_on_empty_value(tmp_path, monkeypatch):
    import hermes_yaml as yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {"display": {"sections": {"activity": "hidden", "tools": "expanded"}}}
        )
    )
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "details_mode.activity", "value": ""},
        }
    )

    assert resp["result"] == {"key": "details_mode.activity", "value": ""}
    saved = yaml.safe_load(cfg_path.read_text())
    assert saved["display"]["sections"] == {"tools": "expanded"}


def test_config_set_section_rejects_unknown_section_or_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    bad_section = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "details_mode.bogus", "value": "hidden"},
        }
    )
    assert bad_section["error"]["code"] == 4002

    bad_mode = server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {"key": "details_mode.tools", "value": "maximised"},
        }
    )
    assert bad_mode["error"]["code"] == 4002


def test_config_mouse_uses_documented_key_with_legacy_fallback(monkeypatch):
    cfg = {"display": {"tui_mouse": False}}
    writes = []

    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr(
        server, "_write_config_key", lambda path, value: writes.append((path, value))
    )

    get_legacy = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "mouse"}}
    )
    assert get_legacy["result"]["value"] == "off"

    set_toggle = server.handle_request(
        {"id": "2", "method": "config.set", "params": {"key": "mouse"}}
    )
    # /mouse (no arg) toggles between 'all' and 'off'. Starting from
    # tui_mouse: False (â†’ 'off'), the toggle flips to 'all'.
    assert set_toggle["result"] == {"key": "mouse", "value": "all"}
    assert writes == [("display.mouse_tracking", "all")]

    cfg["display"] = {"mouse_tracking": 0, "tui_mouse": True}
    get_canonical = server.handle_request(
        {"id": "3", "method": "config.get", "params": {"key": "mouse"}}
    )
    assert get_canonical["result"]["value"] == "off"

    cfg["display"] = {"mouse_tracking": None, "tui_mouse": False}
    get_null = server.handle_request(
        {"id": "4", "method": "config.get", "params": {"key": "mouse"}}
    )
    # mouse_tracking present-but-None defers neither to tui_mouse nor to
    # the legacy off bucket: it falls through to the 'all' default.
    assert get_null["result"]["value"] == "all"


def test_config_mouse_accepts_preset_strings_and_aliases(monkeypatch):
    cfg = {"display": {"mouse_tracking": "all"}}
    writes = []

    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr(
        server, "_write_config_key", lambda path, value: writes.append((path, value))
    )

    # Direct preset.
    set_wheel = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "mouse", "value": "wheel"},
        }
    )
    assert set_wheel["result"] == {"key": "mouse", "value": "wheel"}
    assert writes[-1] == ("display.mouse_tracking", "wheel")

    # Alias for buttons.
    set_click = server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {"key": "mouse", "value": "click"},
        }
    )
    assert set_click["result"] == {"key": "mouse", "value": "buttons"}
    assert writes[-1] == ("display.mouse_tracking", "buttons")

    # Unknown value â†’ 4002.
    bad = server.handle_request(
        {
            "id": "3",
            "method": "config.set",
            "params": {"key": "mouse", "value": "rainbows"},
        }
    )
    assert bad["error"]["code"] == 4002


def test_enable_gateway_prompts_sets_gateway_env(monkeypatch):
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    server._enable_gateway_prompts()

    assert server.os.environ["HERMES_GATEWAY_SESSION"] == "1"
    assert server.os.environ["HERMES_EXEC_ASK"] == "1"
    assert server.os.environ["HERMES_INTERACTIVE"] == "1"




def test_setup_status_answers_from_the_bootstrap_record_once_it_exists(monkeypatch):
    """Under ``hermes serve`` the boot bootstrap owns the free-tier identity; ``setup.status`` reports
    its record (blocking for it while it is in flight) instead of re-probing, so a client's first poll
    sees the identity that exists rather than racing the mint."""
    import threading
    from hermes_cli import free_tier_bootstrap as fb
    fb.reset_for_tests()
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured",
                        lambda **_kw: pytest.fail("setup.status must read the record, not re-probe"))
    release = threading.Event()

    def slow_bootstrap():
        release.wait(5)
        with fb._lock:
            fb._record = fb.SetupRecord(provider_configured=True, inference_provider="nous", free_tier=True,
                                        has_identity=True, other_providers=False)
            fb._done.set()
    with fb._lock:
        fb._started = True
    threading.Thread(target=slow_bootstrap, daemon=True).start()
    try:
        release.set()
        resp = server.handle_request({"id": "1", "method": "setup.status", "params": {}})
        assert resp["result"]["provider_configured"] is True
        assert resp["result"]["ready"] is True and resp["result"]["free_tier"] is True
        assert resp["result"]["inference_provider"] == "nous"
    finally:
        fb.reset_for_tests()


def test_invalid_params_and_unknown_method_name_the_version_skew_fix():
    """The only signal of a TUI/backend version mismatch; the lead phrases stay for clients."""
    resp = server.handle_request({"id": "1", "method": "no.such.method", "params": {}})
    assert resp["error"]["code"] == -32601
    assert resp["error"]["message"].startswith("unknown method: no.such.method")

    resp = server.handle_request(
        {"id": "2", "method": "session.status", "params": {"session_id": "x", "turn_author": "y"}})
    assert resp["error"]["code"] == 4000
    assert resp["error"]["message"].startswith("invalid params for session.status: turn_author")


def test_probe_credentials_emits_exact_empty_key_warning():
    agent = types.SimpleNamespace(api_key="", provider="openrouter")

    warning = server._probe_credentials(agent)
    assert warning and "openrouter" in warning


def test_probe_credentials_allows_keyless_custom_runtime():
    agent = types.SimpleNamespace(api_key="no-key-required", provider="custom")

    assert server._probe_credentials(agent) == ""


def test_setup_runtime_check_rejects_empty_runtime_key(monkeypatch):
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda **_kw: True)
    monkeypatch.setattr(server, "_resolve_startup_runtime", lambda: ("openrouter/test-model", None))
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **_kw: {
            "provider": "openrouter",
            "api_key": "",
            "source": "env/config",
        },
    )

    resp = server.handle_request({"id": "1", "method": "setup.runtime_check", "params": {}})

    result = resp["result"]
    assert result["ok"] is False
    assert (result["provider"], result["model"], result["source"]) == (
        "openrouter", "openrouter/test-model", "env/config"
    )
    assert result["error"]


def test_setup_runtime_check_allows_no_key_custom_runtime(monkeypatch):
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda **_kw: True)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **_kw: {
            "provider": "custom",
            "api_key": "no-key-required",
            "source": "env/config",
        },
    )

    resp = server.handle_request({"id": "1", "method": "setup.runtime_check", "params": {}})

    assert resp["result"]["ok"] is True
    assert resp["result"]["provider"] == "custom"


def test_setup_runtime_check_rejects_implicit_bedrock_when_unconfigured(monkeypatch):
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda **_kw: False)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **_kw: {
            "provider": "bedrock",
            "api_key": "aws-sdk",
            "source": "iam-role",
        },
    )

    resp = server.handle_request({"id": "1", "method": "setup.runtime_check", "params": {}})

    assert resp["result"]["ok"] is False
    assert resp["result"]["provider"] == "bedrock"


def test_setup_runtime_check_honors_requested_provider(monkeypatch):
    """Onboarding must be able to validate the provider the user just connected."""
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda **_kw: True)

    def fake_resolve(requested=None, **kwargs):
        if requested == "nous":
            return {
                "provider": "nous",
                "api_key": "invoke-jwt",
                "source": "portal",
            }
        return {
            "provider": "anthropic",
            "api_key": "",
            "source": "config",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        fake_resolve,
    )

    scoped = server.handle_request(
        {"id": "1", "method": "setup.runtime_check", "params": {"provider": "nous"}}
    )
    assert scoped["result"]["ok"] is True
    assert scoped["result"]["provider"] == "nous"

    default = server.handle_request({"id": "1", "method": "setup.runtime_check", "params": {}})
    assert default["result"]["ok"] is False
    assert default["result"]["provider"] == "anthropic"


def test_setup_runtime_check_agrees_with_session_fallback_chain(monkeypatch):
    """#111775: with the primary blocked and a complete fallback entry, the probe answers what
    ``_make_agent`` would build (fallback provider + model); an explicit ``provider`` stays strict
    so another provider's fallback cannot mask a failed connection."""
    from hermes_cli.auth import AuthError
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda **_kw: True)
    monkeypatch.setattr(server, "_resolve_startup_runtime", lambda: ("claude-sonnet-4-5", None))
    monkeypatch.setattr(server, "_load_fallback_model",
                        lambda: [{"provider": "openrouter", "model": "openai/gpt-4.1-mini", "api_key": "sk-or-fb"}])

    def fake_resolve(*, requested=None, target_model=None, explicit_api_key=None, **_kw):
        if requested == "openrouter":
            return {"provider": "openrouter", "api_key": explicit_api_key, "source": "explicit"}
        raise AuthError("No Anthropic credentials found.", provider="anthropic")

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve)

    default = server.handle_request({"id": "1", "method": "setup.runtime_check", "params": {}})
    assert default["result"]["ok"] is True
    assert (default["result"]["provider"], default["result"]["model"]) == ("openrouter", "openai/gpt-4.1-mini")

    strict = server.handle_request(
        {"id": "2", "method": "setup.runtime_check", "params": {"provider": "anthropic"}})
    assert strict["result"]["ok"] is False
    assert "Anthropic" in strict["result"]["error"]


def test_setup_runtime_check_reports_target_model_on_credential_failure(monkeypatch):
    """#111775: the probe names the model session creation would use, never ``model: null``."""
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda **_kw: True)
    monkeypatch.setattr(server, "_resolve_startup_runtime", lambda: ("z-ai/glm-5.2", None))
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda *, requested=None, target_model=None: {
            "provider": "zai", "api_key": "", "source": "env/config"
        },
    )

    resp = server.handle_request({"id": "1", "method": "setup.runtime_check", "params": {}})

    assert resp["result"]["ok"] is False
    assert resp["result"]["model"] == "z-ai/glm-5.2"

def test_setup_runtime_check_scopes_launch_profile_in_multiplex_backend(monkeypatch, tmp_path):
    """The launch profile needs a scope too when its Codex route reads an override."""
    from agent import secret_scope
    from tui_gateway import launch_profile_policy

    launch_home = tmp_path / ".hermes"
    launch_home.mkdir()
    monkeypatch.setenv("HERMES_CODEX_BASE_URL", "https://codex.launch.test/v1")
    monkeypatch.setattr(server, "_hermes_home", launch_home)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda **_kw: True)
    monkeypatch.setattr(server, "_resolve_startup_runtime", lambda: ("gpt-5.3-codex", None))

    def resolve_codex(requested=None, **_kwargs):
        assert requested == "openai-codex"
        return {
            "provider": "openai-codex",
            "api_key": "codex-oauth-token",
            "base_url": secret_scope.get_secret("HERMES_CODEX_BASE_URL"),
            "source": "credential-pool",
        }

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve_codex)
    secret_scope.set_multiplex_active(True)
    try:
        response = server.handle_request(
            {"id": "1", "method": "setup.runtime_check", "params": {"provider": "openai-codex"}}
        )
    finally:
        secret_scope.set_multiplex_active(False)

    assert response["result"] == {
        "ok": True,
        "provider": "openai-codex",
        "model": "gpt-5.3-codex",
        "source": "credential-pool",
        "free_tier": False,
    }


def test_setup_readiness_scopes_to_requested_profile(monkeypatch, tmp_path):
    """#94071: the Desktop preflights a freshly created bot on its target
    backend. ``profile`` binds THAT profile's home + .env â€” launch-process
    credentials must not make an unconfigured bot look ready, and the bot's
    own .env must be what the strict check sees."""
    from agent import secret_scope
    from hermes_constants import get_hermes_home

    bot_home = tmp_path / "profiles" / "bot"
    bot_home.mkdir(parents=True)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-launch-profile-secret-0000")
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "bot")
    monkeypatch.setattr(server, "_profile_home", lambda profile: bot_home if profile == "bot" else None)
    seen = {}

    def fake_resolve(requested=None, **kwargs):
        seen["home"] = Path(str(get_hermes_home())).resolve()
        seen["secret"] = secret_scope.get_secret("OPENROUTER_API_KEY")
        return {"provider": "openrouter", "api_key": seen["secret"] or "", "source": "env"}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve)

    secret_scope.set_multiplex_active(True)
    try:
        status = server.handle_request(
            {"id": "1", "method": "setup.status", "params": {"profile": "bot"}}
        )
        assert status["result"] == {"provider_configured": False, "profile": "bot"}

        (bot_home / ".env").write_text("OPENROUTER_API_KEY=sk-or-bot-profile-secret-00001\n")
        status = server.handle_request(
            {"id": "2", "method": "setup.status", "params": {"profile": "bot"}}
        )
        runtime = server.handle_request(
            {"id": "3", "method": "setup.runtime_check", "params": {"profile": "bot"}}
        )
    finally:
        secret_scope.set_multiplex_active(False)

    assert status["result"] == {"provider_configured": True, "profile": "bot"}
    assert runtime["result"]["ok"] is True
    assert runtime["result"]["profile"] == "bot"
    assert seen == {"home": bot_home.resolve(), "secret": "sk-or-bot-profile-secret-00001"}
    assert Path(str(get_hermes_home())).resolve() != bot_home.resolve()


def test_setup_readiness_unknown_profile_never_answers_for_launch_profile(monkeypatch):
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda **_kw: True)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **kw: {"provider": "openrouter", "api_key": "sk-or-launch-0000000000", "source": "env"},
    )
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: False)

    for method in ("setup.status", "setup.runtime_check"):
        resp = server.handle_request({"id": "1", "method": method, "params": {"profile": "ghost"}})
        assert resp["result"]["ok"] is False
        assert resp["result"]["profile"] == "ghost"




def test_complete_slash_returns_plain_string_fields():
    # prompt_toolkit hands us FormattedText (a list subclass) for
    # display/display_meta; the TUI's CompletionItem contract is plain
    # strings, and shipping the raw list trips Ink's row layout into
    # 1-char truncation of the next column (/goal â†’ /goa).
    resp = server.handle_request(
        {"id": "1", "method": "complete.slash", "params": {"text": "/g"}}
    )

    items = resp["result"]["items"]
    goal = next((it for it in items if it["text"] == "goal"), None)
    assert goal is not None
    assert isinstance(goal["display"], str), goal["display"]
    assert isinstance(goal["meta"], str), goal["meta"]
    assert goal["display"] == "/goal"
    for item in items:
        assert isinstance(item["display"], str), item
        assert isinstance(item["meta"], str), item






def test_complete_slash_details_args():
    resp_root = server.handle_request(
        {"id": "0", "method": "complete.slash", "params": {"text": "/details"}}
    )
    resp_section = server.handle_request(
        {"id": "1", "method": "complete.slash", "params": {"text": "/details t"}}
    )
    resp_mode = server.handle_request(
        {
            "id": "2",
            "method": "complete.slash",
            "params": {"text": "/details thinking e"},
        }
    )

    assert resp_root["result"]["replace_from"] == len("/details")
    assert any(item["text"] == " thinking" for item in resp_root["result"]["items"])
    assert any(item["text"] == "thinking" for item in resp_section["result"]["items"])
    assert any(item["text"] == "expanded" for item in resp_mode["result"]["items"])




_SLASH_FILLER_COUNT = 60


def _slash_skill_fixtures(monkeypatch):
    """Stub a skill install big enough that a flat cap would truncate it."""
    filler = {f"/filler-{i:03d}": 0 for i in range(_SLASH_FILLER_COUNT)}
    usage = {"work": 297, "research": 84, "clean": 12}

    monkeypatch.setattr(
        server,
        "_skill_usage_lookup",
        lambda: (
            lambda name: usage.get(name, 0),
            lambda name: "bundled" if name.startswith("unused-") else "local",
        ),
    )
    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: {
            "/work": {"description": "Fresh worktree"},
            "/research": {"description": "Look it up"},
            "/clean": {"description": "Polish the diff"},
            "/unused-bundled": {"description": "Shipped, never opened"},
            **{cmd: {"description": "Filler"} for cmd in filler},
        },
    )
    monkeypatch.setattr("agent.skill_bundles.get_skill_bundles", lambda: {})


def _slash_completions(text: str) -> list[dict]:
    resp = server.handle_request(
        {"id": "1", "method": "complete.slash", "params": {"text": text}}
    )
    return resp["result"]["items"]


def test_complete_slash_offers_skills_alongside_commands(monkeypatch):
    """A bare `/` must reach the skills, not just the registry.

    The completer emits every registry command before the first skill, so one
    flat cap spent every row on commands and no skill was reachable at all.
    """
    _slash_skill_fixtures(monkeypatch)

    kinds = {item["kind"] for item in _slash_completions("/")}

    assert kinds == {"command", "skill"}


def test_complete_slash_ranks_skills_by_recorded_usage(monkeypatch):
    """The skills someone actually invokes lead the ones they never opened."""
    _slash_skill_fixtures(monkeypatch)

    skills = [
        item["text"].strip() for item in _slash_completions("/") if item["kind"] == "skill"
    ]

    assert skills[:3] == ["work", "research", "clean"]


def test_complete_slash_prunes_unused_builtins_only_while_browsing(monkeypatch):
    """A bare `/` is browsing and may prune; a typed query is a search.

    A search that hides a match is broken, so the never-opened bundled skill
    disappears from `/` and comes straight back the moment it is typed for.
    """
    _slash_skill_fixtures(monkeypatch)

    browsing = {item["text"].strip() for item in _slash_completions("/")}
    searching = {item["text"].strip() for item in _slash_completions("/unused")}

    assert "unused-bundled" not in browsing
    assert "unused-bundled" in searching


def test_complete_slash_leaves_argument_stages_alone(monkeypatch):
    """Ranking applies to the command token, never to a command's own args.

    `/details c` completes that command's modes; a skill named /clean also
    starts with a `c` and must not be offered as one of them.
    """
    _slash_skill_fixtures(monkeypatch)

    items = _slash_completions("/details c")

    assert [item["text"] for item in items] == ["collapsed", "cycle"]


def test_config_get_reasoning_renders_dict_form_custom_tier(tmp_path, monkeypatch):
    """`agent.reasoning_effort: {enabled: true, effort: thinking}` (a provider's bespoke tier)
    must read back as the tier name, not `str(dict)`."""
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "agent:\n  reasoning_effort:\n    enabled: true\n    effort: thinking\n", encoding="utf-8"
    )

    resp = server.handle_request({"id": "1", "method": "config.get", "params": {"key": "reasoning"}})

    assert resp["result"]["value"] == "thinking"


def test_config_set_reasoning_updates_live_session_and_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("agent:\n  reasoning_effort: medium\n", encoding="utf-8")
    agent = types.SimpleNamespace(reasoning_config=None)
    server._sessions["sid"] = _session(agent=agent)

    resp_effort = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {
                "session_id": "sid",
                "key": "reasoning",
                "value": "low",
            },
        }
    )
    assert resp_effort["result"]["value"] == "low"
    assert agent.reasoning_config == {"enabled": True, "effort": "low"}
    assert server._sessions["sid"]["create_reasoning_override"] == {"enabled": True, "effort": "low"}
    assert server._load_cfg()["agent"]["reasoning_effort"] == "medium"

    resp_status = server.handle_request(
        {
            "id": "5",
            "method": "config.get",
            "params": {"session_id": "sid", "key": "reasoning"},
        }
    )
    assert resp_status["result"]["value"] == "low"

    resp_global_status = server.handle_request(
        {"id": "6", "method": "config.get", "params": {"key": "reasoning"}}
    )
    assert resp_global_status["result"]["value"] == "medium"

    del server._sessions["sid"]["create_reasoning_override"]
    agent.reasoning_config = {"enabled": True, "effort": "high"}
    resp_agent_status = server.handle_request(
        {
            "id": "7",
            "method": "config.get",
            "params": {"session_id": "sid", "key": "reasoning"},
        }
    )
    assert resp_agent_status["result"]["value"] == "high"

    resp_show = server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {"session_id": "sid", "key": "reasoning", "value": "show"},
        }
    )
    assert resp_show["result"]["value"] == "show"
    assert server._sessions["sid"]["show_reasoning"] is True
    assert server._load_cfg()["display"]["sections"]["thinking"] == "expanded"

    resp_hide = server.handle_request(
        {
            "id": "3",
            "method": "config.set",
            "params": {"session_id": "sid", "key": "reasoning", "value": "hide"},
        }
    )
    assert resp_hide["result"]["value"] == "hide"
    assert server._sessions["sid"]["show_reasoning"] is False
    assert server._load_cfg()["display"]["sections"]["thinking"] == "hidden"

    # /reasoning full | clamp â€” parity with the classic CLI reasoning_full
    # toggle. In the TUI these map to the thinking section's expand/collapse
    # rendering (no fixed 10-line recap exists here).
    resp_full = server.handle_request(
        {
            "id": "4",
            "method": "config.set",
            "params": {"session_id": "sid", "key": "reasoning", "value": "full"},
        }
    )
    assert resp_full["result"]["value"] == "full"
    cfg_full = server._load_cfg()
    assert cfg_full["display"]["reasoning_full"] is True
    assert cfg_full["display"]["sections"]["thinking"] == "expanded"

    resp_clamp = server.handle_request(
        {
            "id": "5",
            "method": "config.set",
            "params": {"session_id": "sid", "key": "reasoning", "value": "clamp"},
        }
    )
    assert resp_clamp["result"]["value"] == "clamp"
    cfg_clamp = server._load_cfg()
    assert cfg_clamp["display"]["reasoning_full"] is False
    assert cfg_clamp["display"]["sections"]["thinking"] == "collapsed"


def test_config_set_reasoning_global_scope_clears_session_override(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("agent:\n  reasoning_effort: medium\n", encoding="utf-8")
    agent = types.SimpleNamespace(reasoning_config=None)
    server._sessions["sid"] = _session(agent=agent)
    server._sessions["sid"]["create_reasoning_override"] = {"enabled": True, "effort": "low"}

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {
                "session_id": "sid",
                "key": "reasoning",
                "value": "high",
                "scope": "global",
            },
        }
    )

    assert resp["result"]["value"] == "high"
    assert server._load_cfg()["agent"]["reasoning_effort"] == "high"
    assert "create_reasoning_override" not in server._sessions["sid"]

    status = server.handle_request(
        {"id": "2", "method": "config.get", "params": {"session_id": "sid", "key": "reasoning"}}
    )
    assert status["result"]["value"] == "high"


def test_config_set_verbose_updates_session_mode_and_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    agent = types.SimpleNamespace(verbose_logging=False)
    server._sessions["sid"] = _session(agent=agent)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"session_id": "sid", "key": "verbose", "value": "cycle"},
        }
    )

    assert resp["result"]["value"] == "verbose"
    assert server._sessions["sid"]["tool_progress_mode"] == "verbose"
    assert agent.verbose_logging is True



def test_config_set_model_waits_for_lazy_agent_before_switch(monkeypatch):
    """A model switch against a lazy-created live session must apply to the
    real agent, not just process env, before the prompt is dispatched.
    """

    agent_ready = threading.Event()
    agent = types.SimpleNamespace(model="old/model", provider="old-provider")
    session = _session(agent=agent)
    session["agent"] = None
    session["agent_ready"] = agent_ready
    server._sessions["sid"] = session
    calls = []

    def fake_start(sid, target):
        calls.append(("start", sid))
        target["agent"] = agent
        agent_ready.set()

    def fake_apply(sid, target, raw, **kwargs):
        calls.append(("apply", sid, target.get("agent"), raw))
        if target.get("agent") is not agent:
            raise AssertionError("model switch ran before lazy agent was ready")
        return {"value": "new/model", "warning": ""}

    monkeypatch.setattr(server, "_start_agent_build", fake_start)
    monkeypatch.setattr(server, "_apply_model_switch", fake_apply)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "model", "value": "new/model"},
            }
        )

        assert resp["result"]["value"] == "new/model"
        assert calls == [("start", "sid"), ("apply", "sid", agent, "new/model")]
    finally:
        server._sessions.pop("sid", None)



def test_config_set_model_requires_confirmation_for_expensive_model(monkeypatch):
    class _Agent:
        provider = "openrouter"
        model = "old/model"
        base_url = ""
        api_key = "sk-or"
        switched = False

        def switch_model(self, **_kwargs):
            self.switched = True

    result = types.SimpleNamespace(
        success=True,
        new_model="openai/gpt-5.5-pro",
        target_provider="openrouter",
        api_key="sk-or",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        warning_message="",
        model_info=types.SimpleNamespace(
            has_cost_data=lambda: True,
            cost_input=25.0,
            cost_output=125.0,
        ),
    )

    agent = _Agent()
    server._sessions["sid"] = _session(agent=agent)
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model", lambda **_kwargs: result
    )
    monkeypatch.setattr(server, "_restart_slash_worker", lambda sid, session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {
                "session_id": "sid",
                "key": "model",
                "value": "openai/gpt-5.5-pro --provider openrouter",
            },
        }
    )

    assert resp["result"]["confirm_required"] is True
    assert agent.switched is False

    confirmed = server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {
                "session_id": "sid",
                "key": "model",
                "value": "openai/gpt-5.5-pro --provider openrouter",
                "confirm_expensive_model": True,
            },
        }
    )

    assert confirmed["result"]["confirm_required"] is False
    assert confirmed["result"]["value"] == "openai/gpt-5.5-pro"
    assert agent.switched is True


def test_config_set_model_global_persists(monkeypatch):
    class _Agent:
        provider = "openrouter"
        model = "old/model"
        base_url = ""
        api_key = "sk-old"

        def switch_model(self, **kwargs):
            return None

    result = types.SimpleNamespace(
        success=True,
        new_model="anthropic/claude-sonnet-4.6",
        target_provider="anthropic",
        api_key="sk-new",
        base_url="https://api.anthropic.com",
        api_mode="anthropic_messages",
        warning_message="",
    )
    seen = {}
    saved_values = {}

    def _switch_model(**kwargs):
        seen.update(kwargs)
        return result

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _switch_model)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda sid, session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    # persist_model_selection uses targeted per-key writes (#48305) so it
    # preserves sibling model.* keys instead of rewriting the whole block.
    monkeypatch.setattr("utils.atomic_roundtrip_yaml_update", lambda path, key, value: saved_values.__setitem__(key, value))

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {
                "session_id": "sid",
                "key": "model",
                "value": "anthropic/claude-sonnet-4.6 --global",
            },
        }
    )

    assert resp["result"]["value"] == "anthropic/claude-sonnet-4.6"
    assert seen["is_global"] is True
    assert saved_values["model.default"] == "anthropic/claude-sonnet-4.6"
    assert saved_values["model.provider"] == "anthropic"
    assert saved_values["model.base_url"] == "https://api.anthropic.com"


def test_config_set_model_explicit_provider_skips_broken_default_init(monkeypatch):
    seen = {"build": 0, "wait": 0, "requested": []}
    session = _session()
    session["agent"] = None
    server._sessions["sid"] = session
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"default": "broken/model", "provider": "openrouter"}})
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: seen.__setitem__("build", seen["build"] + 1))
    monkeypatch.setattr(server, "_wait_agent", lambda *_args: seen.__setitem__("wait", seen["wait"] + 1))
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *args, **kwargs: None)

    def fake_runtime_provider(*, requested=None, target_model=None, **_kwargs):
        seen["requested"].append((requested, target_model))
        if requested is None:
            raise RuntimeError("broken default provider should not be initialized")
        if requested == "anthropic":
            return {
                "api_key": "sk-anthropic",
                "api_mode": "anthropic_messages",
                "base_url": "https://api.anthropic.com",
            }
        raise RuntimeError(f"unexpected provider {requested}")

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_runtime_provider)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {
                    "session_id": "sid",
                    "key": "model",
                    "value": "claude-sonnet-4.6 --provider anthropic",
                },
            }
        )

        assert resp["result"]["value"] == "claude-sonnet-4-6"
        assert seen["build"] == 0
        assert seen["wait"] == 0
        assert seen["requested"] == [("anthropic", "claude-sonnet-4.6")]
        assert session["model_override"]["provider"] == "anthropic"
        assert session["model_override"]["model"] == "claude-sonnet-4-6"
    finally:
        server._sessions.pop("sid", None)


@pytest.mark.parametrize(
    ("provider_flag", "failure_text"),
    [
        (" --provider custom:new-provider", "Unknown provider 'removed-provider'"),
        ("", ""),
    ],
)
def test_config_set_model_recovers_failed_profile_resume_after_build_completes(
    monkeypatch, tmp_path, provider_flag, failure_text
):
    """Recovery waits for the real failed build and uses its owning profile.

    Both the failed and replacement generations cross the real deferred-build
    boundary. Provider resolution is the only model-switch leaf replaced.
    """
    from agent.secret_scope import current_secret_scope
    from hermes_constants import get_hermes_home

    launch_url = "https://launch.example/v1"
    profile_url = "https://profile.example/v1"
    launch_home = tmp_path / "launch"
    profile_home = tmp_path / "profiles" / "work"
    launch_home.mkdir()
    profile_home.mkdir(parents=True)
    (launch_home / "config.yaml").write_text(
        "model:\n"
        "  default: launch/model\n"
        "  provider: custom:new-provider\n"
        "providers:\n"
        "  new-provider:\n"
        f"    base_url: {launch_url}\n"
        "    key_env: LAUNCH_API_KEY\n",
        encoding="utf-8",
    )
    (launch_home / ".env").write_text(
        "LAUNCH_API_KEY=launch-secret\n", encoding="utf-8"
    )
    (profile_home / "config.yaml").write_text(
        "model:\n"
        "  default: old/model\n"
        "  provider: custom:new-provider\n"
        "providers:\n"
        "  new-provider:\n"
        f"    base_url: {profile_url}\n"
        "    key_env: PROFILE_API_KEY\n",
        encoding="utf-8",
    )
    (profile_home / ".env").write_text(
        "PROFILE_API_KEY=profile-secret\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)

    class ControlledReady:
        def __init__(self):
            self._event = threading.Event()
            self.wait_entered = threading.Event()

        def wait(self, timeout=None):
            self.wait_entered.set()
            return self._event.wait(timeout)

        def is_set(self):
            return self._event.is_set()

        def set(self):
            self._event.set()

    old_ready = ControlledReady()
    reasoning = {"effort": "high"}
    old_override = {"model": "old/model", "provider": "removed-provider"}
    session = _session(
        agent_ready=old_ready,
        agent_error=None,
        model_override=old_override,
        resume_runtime_overrides={
            "model_override": old_override,
            "provider_override": "removed-provider",
            "reasoning_config_override": reasoning,
        },
        profile_home=str(profile_home),
    )
    session["agent"] = None
    server._sessions["sid"] = session
    old_finally_entered = threading.Event()
    release_old_finally = threading.Event()
    switch_called = threading.Event()
    seen = {"switch": None, "build": None, "persisted": []}
    make_calls = 0

    def fake_switch_model(**kwargs):
        provider = kwargs["user_providers"]["new-provider"]
        secrets = dict(current_secret_scope() or {})
        api_key = secrets[provider["key_env"]]
        seen["switch"] = {
            "home": get_hermes_home(),
            "secrets": secrets,
            "base_url": provider["base_url"],
            "api_key": api_key,
            "current_provider": kwargs["current_provider"],
            "current_api_key": kwargs["current_api_key"],
        }
        switch_called.set()
        return types.SimpleNamespace(
            success=True,
            new_model="new/model",
            target_provider="custom:new-provider",
            api_key=api_key,
            base_url=provider["base_url"],
            api_mode="chat_completions",
            warning_message="",
            model_info=None,
            error_message="",
        )

    class FakeDb:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_session(self, _key):
            return {"model_config": {}}

        def update_session_meta(self, key, model_config, model):
            seen["persisted"].append(
                {
                    "key": key,
                    "model": model,
                    "config": json.loads(model_config),
                }
            )

        def close(self):
            pass

    def fake_make_agent(_sid, _key, **kwargs):
        nonlocal make_calls
        make_calls += 1
        if make_calls == 1:
            raise RuntimeError(failure_text)
        override = kwargs["model_override"]
        seen["build"] = {
            "home": get_hermes_home(),
            "secrets": dict(current_secret_scope() or {}),
            "overrides": kwargs,
        }
        return types.SimpleNamespace(
            model=override["model"],
            provider="custom",
            base_url=override["base_url"],
            api_key=override["api_key"],
            api_mode=override["api_mode"],
            reasoning_config=kwargs.get("reasoning_config_override"),
            service_tier=None,
            _session_db=kwargs.get("session_db"),
        )

    real_transfer = server._transfer_db_to_agent

    def barrier_transfer(agent, db):
        if agent is None and not old_finally_entered.is_set():
            old_finally_entered.set()
            assert release_old_finally.wait(timeout=10)
        return real_transfer(agent, db)

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", fake_switch_model)
    monkeypatch.setattr(
        "hermes_cli.model_selection_guards.combined_selection_warning",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("hermes_state_registry.acquire", FakeDb)
    monkeypatch.setattr(server, "_make_agent", fake_make_agent)
    monkeypatch.setattr(server, "_transfer_db_to_agent", barrier_transfer)
    monkeypatch.setattr(
        "tui_gateway.entry.ensure_mcp_discovery_started", lambda: None
    )
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_start_notification_poller", lambda *a, **k: None)
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *a, **k: None)
    monkeypatch.setattr(server, "_session_info", lambda *a, **k: {})
    monkeypatch.setattr(server, "_probe_config_health", lambda *_args: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *a, **k: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)

    response = {}

    def run_request():
        try:
            response["value"] = server.handle_request(
                {
                    "id": "1",
                    "method": "config.set",
                    "params": {
                        "session_id": "sid",
                        "key": "model",
                        "value": f"new/model{provider_flag}",
                    },
                }
            )
        except BaseException as exc:
            response["error"] = exc

    request_thread = threading.Thread(target=run_request)
    old_build_thread = None
    try:
        server._start_agent_build("sid", session)
        old_build_thread = session["_agent_build_thread"]
        assert old_finally_entered.wait(timeout=10)
        assert session["agent_error"] == failure_text
        assert not old_ready.is_set()

        request_thread.start()
        assert old_ready.wait_entered.wait(timeout=2), (
            "model recovery did not wait for the failed build generation"
        )
        assert not switch_called.is_set()
        release_old_finally.set()
        request_thread.join(timeout=10)

        assert not request_thread.is_alive()
        assert "error" not in response
        assert response["value"]["result"]["value"] == "new/model"
        assert make_calls == 2
        assert seen["switch"] == {
            "home": profile_home,
            "secrets": {"PROFILE_API_KEY": "profile-secret"},
            "base_url": profile_url,
            "api_key": "profile-secret",
            "current_provider": (
                "custom:new-provider" if provider_flag else "custom"
            ),
            "current_api_key": "profile-secret" if not provider_flag else "",
        }
        assert seen["build"]["home"] == profile_home
        assert seen["build"]["secrets"] == {
            "PROFILE_API_KEY": "profile-secret"
        }
        overrides = seen["build"]["overrides"]
        assert overrides["model_override"] == session["model_override"]
        assert overrides["provider_override"] == "custom:new-provider"
        assert overrides["reasoning_config_override"] == reasoning
        assert session["agent_error"] is None
        assert session["agent"].model == "new/model"
        assert session["agent"].base_url == profile_url
        assert session["agent"].api_key == "profile-secret"
        assert seen["persisted"] == [
            {
                "key": "session-key",
                "model": "new/model",
                "config": {
                    "model": "new/model",
                    "provider": "custom:new-provider",
                    "base_url": profile_url,
                    "api_mode": "chat_completions",
                    "reasoning_config": reasoning,
                },
            }
        ]
    finally:
        release_old_finally.set()
        old_ready.set()
        request_thread.join(timeout=10)
        if old_build_thread is not None:
            old_build_thread.join(timeout=10)
        new_build_thread = session.get("_agent_build_thread")
        if new_build_thread is not None:
            new_build_thread.join(timeout=10)
        server._sessions.pop("sid", None)


def test_config_set_model_explicit_provider_surfaces_selected_provider_errors(monkeypatch):
    seen = {"build": 0, "wait": 0}
    session = _session()
    session["agent"] = None
    server._sessions["sid"] = session
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"default": "broken/model", "provider": "openrouter"}})
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: seen.__setitem__("build", seen["build"] + 1))
    monkeypatch.setattr(server, "_wait_agent", lambda *_args: seen.__setitem__("wait", seen["wait"] + 1))

    def fake_runtime_provider(*, requested=None, **_kwargs):
        if requested is None:
            raise RuntimeError("broken default provider should not be initialized")
        if requested == "anthropic":
            raise RuntimeError("missing anthropic API key")
        raise RuntimeError(f"unexpected provider {requested}")

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_runtime_provider)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {
                    "session_id": "sid",
                    "key": "model",
                    "value": "claude-sonnet-4.6 --provider anthropic",
                },
            }
        )

        assert resp["error"]["code"] == 5001
        assert "anthropic" in resp["error"]["message"].lower()
        assert "missing anthropic api key" in resp["error"]["message"].lower()
        assert seen["build"] == 0
        assert seen["wait"] == 0
    finally:
        server._sessions.pop("sid", None)


def test_config_set_model_does_not_leak_inference_provider_env(monkeypatch):
    """A /model switch must NOT mutate process-global env vars. The desktop /
    dashboard tui_gateway backend hosts every same-profile session in one
    process; writing HERMES_INFERENCE_PROVIDER on a switch leaked the new
    provider into every other live session's next agent rebuild. The switch
    must instead record a per-session override and leave shared env untouched.

    (Was test_config_set_model_syncs_inference_provider_env, which asserted the
    leaky env-sync contract that caused the cross-session contamination bug.)
    """

    class _Agent:
        provider = "openrouter"
        model = "old/model"
        base_url = ""
        api_key = "sk-or"

        def switch_model(self, **_kwargs):
            return None

    result = types.SimpleNamespace(
        success=True,
        new_model="claude-sonnet-4.6",
        target_provider="anthropic",
        api_key="sk-ant",
        base_url="https://api.anthropic.com",
        api_mode="anthropic_messages",
        warning_message="",
    )

    session = _session(agent=_Agent())
    server._sessions["sid"] = session
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openrouter")
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model", lambda **_kwargs: result
    )
    monkeypatch.setattr(server, "_restart_slash_worker", lambda sid, session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

    try:
        server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {
                    "session_id": "sid",
                    "key": "model",
                    "value": "claude-sonnet-4.6 --provider anthropic",
                },
            }
        )

        # Shared process env is UNCHANGED (the contamination vector is gone).
        assert os.environ["HERMES_INFERENCE_PROVIDER"] == "openrouter"
        # The switch was recorded as a per-session override instead.
        assert session["model_override"]["provider"] == "anthropic"
        assert session["model_override"]["model"] == "claude-sonnet-4.6"
    finally:
        server._sessions.clear()


def test_config_set_model_records_per_session_override_not_env(monkeypatch):
    """Regression for #16857 via the per-session override (not env vars):
    /model must record the user's explicit provider on the session so a later
    /new (which rebuilds via _make_agent honoring model_override) honours that
    choice â€” WITHOUT writing process-global env vars that would leak into
    sibling sessions.

    (Was test_config_set_model_syncs_tui_provider_unconditionally.)
    """

    class _Agent:
        provider = "openrouter"
        model = "old/model"
        base_url = ""
        api_key = "sk-or"

        def switch_model(self, **_kwargs):
            return None

    result = types.SimpleNamespace(
        success=True,
        new_model="deepseek-v4-pro",
        target_provider="custom:xuanji",
        api_key="sk-xuanji",
        base_url="https://xuanji.example/v1",
        api_mode="chat_completions",
        warning_message="",
    )

    session = _session(agent=_Agent())
    server._sessions["sid"] = session
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model", lambda **_kwargs: result
    )
    monkeypatch.setattr(server, "_restart_slash_worker", lambda sid, session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

    try:
        server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {
                    "session_id": "sid",
                    "key": "model",
                    "value": "deepseek-v4-pro --provider custom:xuanji",
                },
            }
        )

        # No process-global env mutation.
        assert "HERMES_TUI_PROVIDER" not in os.environ
        assert "HERMES_INFERENCE_PROVIDER" not in os.environ
        # The user's explicit provider + resolved endpoint live on the session,
        # carried into the next /new rebuild by _make_agent.
        override = session["model_override"]
        assert override["provider"] == "custom:xuanji"
        assert override["model"] == "deepseek-v4-pro"
        assert override["base_url"] == "https://xuanji.example/v1"
        assert override["api_key"] == "sk-xuanji"
        assert override["api_mode"] == "chat_completions"
    finally:
        server._sessions.clear()


def test_config_set_model_switches_agent_without_touching_env(monkeypatch):
    """A /model switch mutates the target session's agent in place and records
    a per-session override; it does NOT write HERMES_MODEL / HERMES_TUI_PROVIDER
    etc. into the shared process environment.

    (Was test_config_set_model_syncs_tui_provider_env.)
    """

    class Agent:
        model = "gpt-5.3-codex"
        provider = "openai-codex"
        base_url = ""
        api_key = ""
        session_id = "sid"
        _cached_system_prompt = "Model: gpt-5.3-codex\nProvider: openai-codex"

        def switch_model(self, **kwargs):
            self.model = kwargs["new_model"]
            self.provider = kwargs["new_provider"]

        def _build_system_prompt(self, _system_message=None):
            return f"Model: {self.model}\nProvider: {self.provider}"

    class SessionDB:
        def __init__(self):
            self.model_config = None
            self.system_prompt = None
            self.messages = []

        def get_session(self, _session_id):
            return {"model_config": self.model_config}

        def update_session_meta(self, _session_id, model_config_json, _model=None):
            self.model_config = model_config_json

        def update_system_prompt(self, _session_id, system_prompt):
            self.system_prompt = system_prompt

        def append_message(self, session_id, role, content=None, **_kwargs):
            self.messages.append(
                {"session_id": session_id, "role": role, "content": content}
            )

    agent = Agent()
    db = SessionDB()
    agent._session_db = db
    session = _session(agent=agent)
    server._sessions["sid"] = session
    monkeypatch.setenv("HERMES_TUI_PROVIDER", "openai-codex")
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda sid, session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

    def fake_switch_model(**kwargs):
        return types.SimpleNamespace(
            success=True,
            new_model="anthropic/claude-sonnet-4.6",
            target_provider="anthropic",
            api_key="key",
            base_url="https://api.anthropic.com",
            api_mode="anthropic_messages",
            warning_message="",
        )

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", fake_switch_model)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {
                    "session_id": "sid",
                    "key": "model",
                    "value": "anthropic/claude-sonnet-4.6 --provider anthropic",
                },
            }
        )

        assert resp["result"]["value"] == "anthropic/claude-sonnet-4.6"
        # Agent switched in place...
        assert agent.model == "anthropic/claude-sonnet-4.6"
        assert agent.provider == "anthropic"
        # ...override recorded on the session...
        assert session["model_override"]["model"] == "anthropic/claude-sonnet-4.6"
        assert session["model_override"]["provider"] == "anthropic"
        # ...the persisted prompt snapshot tracks the new runtime identity too.
        # Without this, the next turn restored the old system prompt from the DB:
        # API calls went to the new model, but "what model are you?" still read
        # "Model: old/model" from the stored prompt.
        assert db.system_prompt == (
            "Model: anthropic/claude-sonnet-4.6\nProvider: anthropic"
        )
        assert agent._cached_system_prompt == db.system_prompt
        assert session["history"][-1]["role"] == "user"
        assert "changed to anthropic/claude-sonnet-4.6" in session["history"][-1]["content"]
        assert db.messages[-1] == {
            "session_id": "session-key",
            "role": "user",
            "content": session["history"][-1]["content"],
        }
        # ...and the shared process env was NOT touched.
        assert os.environ["HERMES_TUI_PROVIDER"] == "openai-codex"
        assert "HERMES_MODEL" not in os.environ
        assert "HERMES_INFERENCE_MODEL" not in os.environ
    finally:
        server._sessions.clear()


def test_config_set_model_once_keeps_env_and_records_restore(monkeypatch):
    class Agent:
        model = "old/model"
        provider = "openrouter"
        base_url = "https://openrouter.ai/api/v1"
        api_key = "sk-old"
        api_mode = "chat_completions"

        def switch_model(self, **kwargs):
            self.model = kwargs["new_model"]
            self.provider = kwargs["new_provider"]
            self.api_key = kwargs["api_key"]
            self.base_url = kwargs["base_url"]
            self.api_mode = kwargs["api_mode"]

    result = types.SimpleNamespace(
        success=True,
        new_model="claude-sonnet-4.6",
        target_provider="anthropic",
        api_key="sk-ant",
        base_url="https://api.anthropic.com",
        api_mode="anthropic_messages",
        warning_message="",
    )
    seen = {}
    agent = Agent()
    session = _session(agent=agent)
    server._sessions["sid"] = session
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openrouter")
    monkeypatch.setenv("HERMES_MODEL", "old/model")
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kwargs: seen.update(kwargs) or result,
    )
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {
                    "session_id": "sid",
                    "key": "model",
                    "value": "claude-sonnet-4.6 --provider anthropic --once",
                },
            }
        )

        assert resp["result"]["scope"] == "once"
        assert seen["is_global"] is False
        assert agent.model == "claude-sonnet-4.6"
        assert session["one_turn_model_restore"]["model"] == "old/model"
        assert os.environ["HERMES_INFERENCE_PROVIDER"] == "openrouter"
        assert os.environ["HERMES_MODEL"] == "old/model"
    finally:
        server._sessions.clear()


def test_config_set_model_once_requires_live_session(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_: (_ for _ in ()).throw(AssertionError("switch should not run")),
    )

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {
                "key": "model",
                "value": "claude-sonnet-4.6 --provider anthropic --once",
            },
        }
    )

    assert resp["error"]["code"] == 5001


def test_config_set_model_sessionless_rejected(monkeypatch):
    """Sessionless config.set model must 4001 before _apply_model_switch.

    Missing session_id and a stale session_id miss both take the sessionless
    branch; unscoped values and legacy --global must be rejected the same way
    so a Desktop client cannot persist model.default before session.create.
    """
    called = {"n": 0}

    def boom(*a, **k):
        called["n"] += 1
        raise AssertionError("_apply_model_switch must not run")

    monkeypatch.setattr(server, "_apply_model_switch", boom)
    for value in ["some-model", "some-model --provider openai-codex --global"]:
        resp = server.handle_request({
            "id": "1", "method": "config.set",
            "params": {"key": "model", "value": value},
        })
        assert resp["error"]["code"] == 4001
        assert called["n"] == 0

    resp = server.handle_request({
        "id": "1", "method": "config.set",
        "params": {"session_id": "missing-sid", "key": "model", "value": "some-model --global"},
    })
    assert resp["error"]["code"] == 4001
    assert called["n"] == 0


def test_config_set_model_live_session_still_applies_switch(monkeypatch):
    """CONTROL: a live session still reaches _apply_model_switch, including --global."""
    called = {"raw": []}

    def fake_apply(sid, session, raw, **_kwargs):
        called["raw"].append(raw)
        return {"value": "some-model", "warning": "", "scope": "global"}

    server._sessions["sid"] = _session()
    monkeypatch.setattr(server, "_apply_model_switch", fake_apply)
    try:
        resp = server.handle_request({
            "id": "1", "method": "config.set",
            "params": {
                "session_id": "sid",
                "key": "model",
                "value": "some-model --provider openai-codex --global",
            },
        })
        assert "error" not in resp
        assert called["raw"] == ["some-model --provider openai-codex --global"]
    finally:
        server._sessions.pop("sid", None)


def test_config_set_model_session_switch_clears_pending_once_restore(monkeypatch):
    class Agent:
        model = "temp/model"
        provider = "anthropic"
        base_url = "https://api.anthropic.com"
        api_key = "sk-temp"
        api_mode = "anthropic_messages"

        def switch_model(self, **kwargs):
            self.model = kwargs["new_model"]
            self.provider = kwargs["new_provider"]
            self.api_key = kwargs["api_key"]
            self.base_url = kwargs["base_url"]
            self.api_mode = kwargs["api_mode"]

    result = types.SimpleNamespace(
        success=True,
        new_model="new/model",
        target_provider="openrouter",
        api_key="sk-new",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        warning_message="",
    )
    session = _session(agent=Agent())
    session["one_turn_model_restore"] = {"model": "old/model"}
    server._sessions["sid"] = session
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kwargs: result)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {
                    "session_id": "sid",
                    "key": "model",
                    "value": "new/model --provider openrouter --session",
                },
            }
        )

        assert resp["result"]["scope"] == "session"
        assert "one_turn_model_restore" not in session
    finally:
        server._sessions.clear()


def test_restore_agent_model_runtime_falls_back_to_switch_model():
    class Agent:
        model = "temp/model"
        provider = "anthropic"
        base_url = "https://api.anthropic.com"
        api_key = "sk-temp"
        api_mode = "anthropic_messages"

        def switch_model(self, **kwargs):
            self.model = kwargs["new_model"]
            self.provider = kwargs["new_provider"]
            self.api_key = kwargs["api_key"]
            self.base_url = kwargs["base_url"]
            self.api_mode = kwargs["api_mode"]

    agent = Agent()

    server._restore_agent_model_runtime(
        agent,
        {
            "model": "old/model",
            "provider": "openrouter",
            "api_key": "sk-old",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
        },
    )

    assert agent.model == "old/model"
    assert agent.provider == "openrouter"
    assert agent.base_url == "https://openrouter.ai/api/v1"


def test_config_set_personality_rejects_unknown_name(monkeypatch):
    monkeypatch.setattr(
        server,
        "_available_personalities",
        lambda cfg=None: {"helpful": "You are helpful."},
    )
    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "personality", "value": "bogus"},
        }
    )

    assert "error" in resp


def test_config_set_personality_preserves_history_and_returns_info(monkeypatch):
    agent = types.SimpleNamespace(
        ephemeral_system_prompt=None, _cached_system_prompt="old"
    )
    session = _session(
        agent=agent,
        history=[{"role": "user", "text": "hi"}],
        history_version=4,
    )
    emits = []
    writes = []

    server._sessions["sid"] = session
    monkeypatch.setattr(
        server,
        "_available_personalities",
        lambda cfg=None: {"helpful": "You are helpful."},
    )
    monkeypatch.setattr(
        server, "_session_info", lambda agent, *a: {"model": getattr(agent, "model", "?")}
    )
    monkeypatch.setattr(server, "_emit", lambda *args: emits.append(args))
    # Persistence now flows through the single owner (hermes_cli.personality),
    # never _write_config_key / agent.system_prompt.
    import hermes_cli.personality as personality_mod

    monkeypatch.setattr(
        personality_mod,
        "persist_personality",
        lambda name: writes.append(("display.personality", name)) or True,
    )
    monkeypatch.setattr(
        server,
        "_write_config_key",
      ë}wÙ¼­zÊ&ŠÛ^u•ÍÑ¥½¸‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐ…¹ÍÝ•È‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹ÅÕ•ÍÑ¥½¸‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹…¹ÍÝ•È‰ô°(€€€€€€€¡…¹‘½™˜°(€€€t(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸ (€€€€€€€…•¹ÐõÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡}¡•­Á½¥¹Ñ}µÈõ}5È ¤¤°(€€€€€€€¡¥ÍÑ½Éäõ±¥ÍÐ¡¡¥ÍÑ½Éä¤°(€€€€€€€Í•ÍÍ¥½¹}­•äôˆˆ°(€€€€¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰É½±±‰…¬¹É•ÍÑ½É”ˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰¡…Í ˆè€‰…‰ŒÄÈÌ‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤((€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÍÕ•ÍÌ‰t¥ÌQÉÕ”(€€€€€€€€ŒQÉÕ¹…Ñ¥½¸±…¹‘Ì½¸€‰Í•½¹ÅÕ•ÍÑ¥½¸ˆ°¹½ÐÑ¡”¡…¹‘½™˜É½Ü¸(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰¡¥ÍÑ½Éå}É•µ½Ù•‰t€ôô€Ì€€ŒÄÈ€¬„È€¬¡…¹‘½™˜(€€€€€€€É•µ…¥¹¥¹œ€ôÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰ul‰¡¥ÍÑ½Éä‰t(€€€€€€€…ÍÍ•ÉÐmµl‰½¹Ñ•¹Ð‰t™½È´¥¸É•µ…¥¹¥¹t€ôôl‰™¥ÉÍÐÅÕ•ÍÑ¥½¸ˆ°€‰™¥ÉÍÐ…¹ÍÝ•È‰t(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(((ŒƒŠRŠR Í•ÍÍ¥½¸¹ÍÑ••ÈƒŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠR (()‘•˜Ñ•ÍÑ}É½±±‰…­}É•ÍÑ½É•}ÁÉ•Í•ÉÙ•Í}½µÁ½Í¥Ñ•}…ÉÉ¥•É}Í…™™½±¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰¡•­Á½¥¹ÐÉ•ÍÑ½É”‘É½ÁÌÑ¡”±¥Ù”…Í¬‰ÕÐ­••ÁÌ½µÁ…Ñ•½¹Ñ•áÐ¸ˆˆˆ(€€€™É½´…•¹Ð¹½¹Ñ•áÑ}½µÁÉ•ÍÍ½È¥µÁ½ÉÐ€ (€€€€€€€!%MQ=I%1}QM-}!%9°(€€€€€€€MU55Ie}AI%`°(€€€€€€€}MU55Ie}9}5I-H°(€€€€¤(€€€™É½´¡•Éµ•Í}ÍÑ…Ñ”¥µÁ½ÉÐM•ÍÍ¥½¹((€€€±…ÍÌ}5Èè(€€€€€€€•¹…‰±•€ôQÉÕ”(€€€€€€€Õ¹ÍÕÁÁ½ÉÑ•‘}‰…­•¹‘}É•…Í½¸€ôÍÑ…Ñ¥µ•Ñ¡½¡±…µ‰‘„Ñ…Í­}¥ô‰‘•™…Õ±Ðˆè9½¹”¤((€€€€€€€‘•˜±¥ÍÑ}¡•­Á½¥¹ÑÌ¡Í•±˜°Ý¤è(€€€€€€€€€€€É•ÑÕÉ¸mì‰¡…Í ˆè€‰…‰ŒÄÈÌ‰õt((€€€€€€€‘•˜É•ÍÑ½É”¡Í•±˜°Ý°Ñ…É•Ð°™¥±•}Á…Ñ õ9½¹”¤è(€€€€€€€€€€€É•ÑÕÉ¸ì‰ÍÕ•ÍÌˆèQÉÕ”°€‰µ•ÍÍ…”ˆè€‰É•ÍÑ½É•‰ô((€€€…ÉÉ¥•È€ôì(€€€€€€€€‰É½±”ˆè€‰ÕÍ•Èˆ°(€€€€€€€€‰½¹Ñ•¹Ðˆè€ (€€€€€€€€€€€˜‰íMU55Ie}AI%aõq¹í!%MQ=I%1}QM-}!%9õq¹½±Ñ…Í­q¹q¸ˆ(€€€€€€€€€€€˜‰í}MU55Ie}9}5I-Iõq¹q¹I0M,ˆ(€€€€€€€€¤°(€€€ô(€€€‘ˆ€ôM•ÍÍ¥½¹¡‘‰}Á…Ñ õÑµÁ}Á…Ñ €¼€‰ÍÑ…Ñ”¹‘ˆˆ¤(€€€‘ˆ¹É•…Ñ•}Í•ÍÍ¥½¸ ‰É½±±‰…¬µ…ÉÉ¥•Èˆ°Í½ÕÉ”ô‰ÑÕ¤ˆ¤(€€€‘ˆ¹…ÁÁ•¹‘}µ•ÍÍ…” ‰É½±±‰…¬µ…ÉÉ¥•Èˆ°€‰ÕÍ•Èˆ°…ÉÉ¥•Él‰½¹Ñ•¹Ð‰t¤(€€€‘ˆ¹…ÁÁ•¹‘}µ•ÍÍ…” ‰É½±±‰…¬µ…ÉÉ¥•Èˆ°€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰…¹ÍÝ•Èˆ¤(€€€‘ÕÉ…‰±”€ô‘ˆ¹•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸ ‰É½±±‰…¬µ…ÉÉ¥•Èˆ¤(€€€…•¹Ð€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€}¡•­Á½¥¹Ñ}µÈõ}5È ¤°(€€€€€€€}Í•ÍÍ¥½¹}µ•ÍÍ…•Ìõ±¥ÍÐ¡‘ÕÉ…‰±”¤°(€€€€€€€}±…ÍÑ}™±ÕÍ¡•‘}‘‰}¥‘àõ±•¸¡‘ÕÉ…‰±”¤°(€€€€€€€}‘‰}™±ÕÍ¡}Í…¹}ÁÉ•™¥àõ±¥ÍÐ¡‘ÕÉ…‰±”¤°(€€€€¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸ (€€€€€€€…•¹Ðõ…•¹Ð°(€€€€€€€¡¥ÍÑ½Éäõ±¥ÍÐ¡‘ÕÉ…‰±”¤°(€€€€€€€Í•ÍÍ¥½¹}­•äô‰É½±±‰…¬µ…ÉÉ¥•Èˆ°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è‘ˆ¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰É½±±‰…¬¹É•ÍÑ½É”ˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰¡…Í ˆè€‰…‰ŒÄÈÌ‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤((€€€€€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÀ°É•ÍÀ(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÍÕ•ÍÌ‰t¥ÌQÉÕ”(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰¡¥ÍÑ½Éå}É•µ½Ù•‰t€ôô€È(€€€€€€€É•µ…¥¹¥¹œ€ôÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰ul‰¡¥ÍÑ½Éä‰t(€€€€€€€…ÍÍ•ÉÐ±•¸¡É•µ…¥¹¥¹œ¤€ôô€Ä(€€€€€€€…ÍÍ•ÉÐÉ•µ…¥¹¥¹lÁul‰‘¥ÍÁ±…å}­¥¹‰t€ôô€‰¡¥‘‘•¸ˆ(€€€€€€€…ÍÍ•ÉÐMU55Ie}AI%`¥¸É•µ…¥¹¥¹lÁul‰½¹Ñ•¹Ð‰t(€€€€€€€…ÍÍ•ÉÐ€‰I0M,ˆ¹½Ð¥¸É•µ…¥¹¥¹lÁul‰½¹Ñ•¹Ð‰t(€€€€€€€½±€ô‘ˆ¹•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸ (€€€€€€€€€€€€‰É½±±‰…¬µ…ÉÉ¥•Èˆ°¥¹±Õ‘•}É½Ý}¥‘ÌõQÉÕ”(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐ±•¸¡½±¤€ôô€Ä(€€€€€€€…ÍÍ•ÉÐ½±‘lÁul‰½¹Ñ•¹Ð‰t€ôôÉ•µ…¥¹¥¹lÁul‰½¹Ñ•¹Ð‰t(€€€€€€€…ÍÍ•ÉÐ½±‘lÁul‰‘¥ÍÁ±…å}­¥¹‰t€ôô€‰¡¥‘‘•¸ˆ(€€€€€€€…ÍÍ•ÉÐ½±‘lÁul‰}É½Ý}¥‰t€ôôÉ•µ…¥¹¥¹lÁul‰}É½Ý}¥‰t(€€€€€€€…ÍÍ•ÉÐ…•¹Ð¹}Í•ÍÍ¥½¹}µ•ÍÍ…•Ì€ôôÉ•µ…¥¹¥¹œ(€€€€€€€…ÍÍ•ÉÐ…•¹Ð¹}±…ÍÑ}™±ÕÍ¡•‘}‘‰}¥‘à€ôô€Ä(€€€€€€€…ÍÍ•ÉÐ…•¹Ð¹}‘‰}™±ÕÍ¡}Í…¹}ÁÉ•™¥à€ôôÉ•µ…¥¹¥¹œ(€€€€€€€¥¹…Ñ¥Ù”€ô‘ˆ¹•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸ (€€€€€€€€€€€€‰É½±±‰…¬µ…ÉÉ¥•Èˆ°¥¹±Õ‘•}¥¹…Ñ¥Ù”õQÉÕ”(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐ…¹ä ‰I0M,ˆ¥¸ÍÑÈ¡µ•ÍÍ…”¹•Ð ‰½¹Ñ•¹Ðˆ¤¤™½Èµ•ÍÍ…”¥¸¥¹…Ñ¥Ù”¤(€€€€€€€…ÍÍ•ÉÐ…¹ä¡µ•ÍÍ…”¹•Ð ‰½¹Ñ•¹Ðˆ¤€ôô€‰…¹ÍÝ•Èˆ™½Èµ•ÍÍ…”¥¸¥¹…Ñ¥Ù”¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(€€€€€€€‘ˆ¹±½Í” ¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}ÍÑ••É}…±±Í}…•¹Ñ}ÍÑ••É}Ý¡•¹}…•¹Ñ}ÍÕÁÁ½ÉÑÍ}¥Ð ¤è(€€€€ˆˆ‰Q¡”QU$IAµ•Ñ¡½µÕÍÐ…±°…•¹Ð¹ÍÑ••È¡Ñ•áÐ¤…¹É•ÑÕÉ¸„(€€€ÅÕ•Õ•ÍÑ…ÑÕÌÝ¥Ñ¡½ÕÐÑ½Õ¡¥¹œ¥¹Ñ•ÉÉÕÁÐÍÑ…Ñ”¸(€€€€ˆˆˆ(€€€…±±Ì€ôíô((€€€±…ÍÌ}•¹Ðè(€€€€€€€‘•˜ÍÑ••È¡Í•±˜°Ñ•áÐ¤è(€€€€€€€€€€€…±±Íl‰ÍÑ••É}Ñ•áÐ‰t€ôÑ•áÐ(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€€€€€‘•˜¥¹Ñ•ÉÉÕÁÐ¡Í•±˜°€©…ÉÌ°€¨©­Ý…ÉÌ¤è(€€€€€€€€€€€…±±Íl‰¥¹Ñ•ÉÉÕÁÑ}…±±•‰t€ôQÉÕ”((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸¡…•¹Ðõ}•¹Ð ¤°ÉÕ¹¹¥¹œõQÉÕ”¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹ÍÑ••Èˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰…±Í¼¡•¬…ÕÑ ¹±½œ‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤((€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÀ°É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÍÑ…ÑÕÌ‰t€ôô€‰ÅÕ•Õ•ˆ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Ñ•áÐ‰t€ôô€‰…±Í¼¡•¬…ÕÑ ¹±½œˆ(€€€…ÍÍ•ÉÐ…±±Íl‰ÍÑ••É}Ñ•áÐ‰t€ôô€‰…±Í¼¡•¬…ÕÑ ¹±½œˆ(€€€…ÍÍ•ÉÐ€‰¥¹Ñ•ÉÉÕÁÑ}…±±•ˆ¹½Ð¥¸…±±Ì€€ŒµÕÍÐ9=P¥¹Ñ•ÉÉÕÁÐ(()±…ÍÌ}I•½É‘¥¹MÑ••É•¹Ðè(€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜¤è(€€€€€€€Í•±˜¹ÍÑ••É•€ômt((€€€‘•˜ÍÑ••È¡Í•±˜°Ñ•áÐ¤è(€€€€€€€Í•±˜¹ÍÑ••É•¹…ÁÁ•¹¡Ñ•áÐ¤(€€€€€€€É•ÑÕÉ¸QÉÕ”(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}ÍÑ••É}½¹}¥‘±•}Í•ÍÍ¥½¹}¥Í}É•©•Ñ•‘}¹½Ñ}Á…É­• ¤è(€€€€ˆˆˆŒØÐÔÜàèÝ¥Ñ ¹¼±¥Ù”ÑÕÉ¸„ÍÑ••È¡…Ì¹¼Ñ½½°…±°Ñ¼É¥‘”¸•ÁÑ¥¹œ¥ÐÁ…É­•Ñ¡”Ñ•áÐ¥¸(€€€Ñ¡”…•¹ÐÌÁ•¹‘¥¹œµÍÑ••ÈÍ±½Ð°Ý¡•É”Ñ¡”¹•áÐÑÕÉ¸ÌÁÉ”µA$‘É…¥¸ÍÁ±¥•¥Ð…™Ñ•È…¸=1Ñ½½°(€€€É½Ü¸%ÐµÕÍÐ½µ”‰…¬€É•©•Ñ•œ€¡±¥•¹ÑÌÑ¡•¸Í•¹¥Ð…Ì„¹½Éµ…°ÁÉ½µÁÐ¤…¹¹•Ù•ÈÉ•… (€€€…•¹Ð¹ÍÑ••È ¤¸ˆˆˆ(€€€…•¹Ð€ô}I•½É‘¥¹MÑ••É•¹Ð ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸¡…•¹Ðõ…•¹Ð°ÉÕ¹¹¥¹œõ…±Í”¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹ÍÑ••Èˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰¡•¬Ñ¡”±½Ì‰õô(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÍÑ…ÑÕÌ‰t€ôô€‰É•©•Ñ•ˆ°É•ÍÀ(€€€…ÍÍ•ÉÐ…•¹Ð¹ÍÑ••É•€ôômt(()‘•˜Ñ•ÍÑ}ÍÑ••É}Í±…Í¡}½¹}¥‘±•}Í•ÍÍ¥½¹}Í•¹‘Í}…Í}¹•áÑ}ÑÕÉ¸ ¤è(€€€€ˆˆˆŒØÐÔÜàè¥‘±”€½ÍÑ••È€ñÑ•áÐù€Ù¥„½µµ…¹¹‘¥ÍÁ…Ñ µÕÍÐ¼½ÕÐ…Ì„¹½Éµ…°¹•áÐµÑÕÉ¸µ•ÍÍ…”(€€€Ý¥Ñ „¹½Ñ¥”Í…å¥¹œÍ¼°¹½Ð±…¥´€‰MÑ••ÈÅÕ•Õ•ˆÝ¡¥±”ÍÑ…Í¡¥¹œÑ¡”Ñ•áÐ½¸Ñ¡”…•¹Ð¸ˆˆˆ(€€€…•¹Ð€ô}I•½É‘¥¹MÑ••É•¹Ð ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸¡…•¹Ðõ…•¹Ð°ÉÕ¹¹¥¹œõ…±Í”¤(€€€ÑÉäè(€€€€€€€É•Ì€ôÍ•ÉÙ•È¹}µ•Ñ¡½‘Íl‰½µµ…¹¹‘¥ÍÁ…Ñ ‰t (€€€€€€€€€€€€ˆÄˆ°ì‰¹…µ”ˆè€‰ÍÑ••Èˆ°€‰…Éœˆè€‰¡•¬Ñ¡”±½Ìˆ°€‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥‰ô¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤((€€€É•ÍÕ±Ð€ôÉ•Íl‰É•ÍÕ±Ð‰t(€€€…ÍÍ•ÉÐÉ•ÍÕ±Ñl‰ÑåÁ”‰t€ôô€‰Í•¹ˆ(€€€…ÍÍ•ÉÐÉ•ÍÕ±Ñl‰µ•ÍÍ…”‰t€ôô€‰¡•¬Ñ¡”±½Ìˆ(€€€…ÍÍ•ÉÐÉ•ÍÕ±Ð¹•Ð ‰¹½Ñ¥”ˆ¤(€€€…ÍÍ•ÉÐ…•¹Ð¹ÍÑ••É•€ôômt(()‘•˜Ñ•ÍÑ}ÍÑ••É}Í±…Í¡}‘ÕÉ¥¹}±¥Ù•}ÑÕÉ¹}ÍÑ¥±±}ÍÑ••ÉÌ ¤è(€€€…•¹Ð€ô}I•½É‘¥¹MÑ••É•¹Ð ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸¡…•¹Ðõ…•¹Ð°ÉÕ¹¹¥¹œõQÉÕ”¤(€€€ÑÉäè(€€€€€€€É•Ì€ôÍ•ÉÙ•È¹}µ•Ñ¡½‘Íl‰½µµ…¹¹‘¥ÍÁ…Ñ ‰t (€€€€€€€€€€€€ˆÄˆ°ì‰¹…µ”ˆè€‰ÍÑ••Èˆ°€‰…Éœˆè€‰¡•¬Ñ¡”±½Ìˆ°€‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥‰ô¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤((€€€…ÍÍ•ÉÐÉ•Íl‰É•ÍÕ±Ð‰ul‰ÑåÁ”‰t€ôô€‰•á•Œˆ(€€€…ÍÍ•ÉÐ…•¹Ð¹ÍÑ••É•€ôôl‰¡•¬Ñ¡”±½Ì‰t(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}ÍÑ••É}É•©•ÑÍ}•µÁÑå}Ñ•áÐ ¤è(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸ (€€€€€€€…•¹ÐõÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡ÍÑ••Èõ±…µ‰‘„ÐèQÉÕ”¤(€€€€¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹ÍÑ••Èˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€ˆ€€€‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤((€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¥¸É•ÍÀ°É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÀÈ(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}ÍÑ••É}•ÉÉ½ÉÍ}Ý¡•¹}…•¹Ñ}¡…Í}¹½}ÍÑ••É}µ•Ñ¡½ ¤è(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸¡…•¹ÐõÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” ¤¤€€Œ¹¼ÍÑ••È ¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹ÍÑ••Èˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰¡¤‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤((€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¥¸É•ÍÀ°É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÄÀ(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}É•‘¥É•Ñ}…±±Í}…Á…‰±•}½É•}…•¹Ð¡µ½¹­•åÁ…Ñ ¤è(€€€…±±Ì€ômt(€€€…•¹Ð€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€}ÍÕÁÁ½ÉÑÍ}…Ñ¥Ù•}ÑÕÉ¹}É•‘¥É•ÐõQÉÕ”°(€€€€€€€É•‘¥É•Ðõ±…µ‰‘„Ñ•áÐè…±±Ì¹…ÁÁ•¹¡Ñ•áÐ¤½ÈQÉÕ”°(€€€€¤(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡…•¹Ðõ…•¹Ð¤(€€€Í•ÍÍ¥½¹l‰¥¹™±¥¡Ñ}ÑÕÉ¸‰t€ôì‰ÕÍ•Èˆè€‰½É¥¥¹…°É•ÅÕ•ÍÐˆ°€‰…ÍÍ¥ÍÑ…¹Ðˆè€‰Á…ÉÑ¥…°É•Á±ä‰ô(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ôÍ•ÍÍ¥½¸(€€€ÑÉäè(€€€€€€€‰•™½É”€ôÍ•ÍÍ¥½¸¹•Ð ‰±…ÍÑ}…Ñ¥Ù”ˆ¤(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹É•‘¥É•Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰ÕÍ”A½ÍÑÉ•Ì‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰t€ôôì(€€€€€€€€‰ÍÑ…ÑÕÌˆè€‰É•‘¥É•Ñ•ˆ°(€€€€€€€€‰Ñ•áÐˆè€‰ÕÍ”A½ÍÑÉ•Ìˆ°(€€€ô(€€€…ÍÍ•ÉÐ…±±Ì€ôôl‰ÕÍ”A½ÍÑÉ•Ì‰t(€€€€ŒQ¡”½ÉÉ•Ñ¥½¸¥ÌÉ•½É‘•…±½¹Í¥‘”Ñ¡”ÁÉ½µÁÐÑ¡…ÐÍÑ…ÉÑ•Ñ¡”ÑÕÉ¸°(€€€€Œ¹•Ù•È½Ù•È¥ÐƒŠPÉ•ÍÕµ”µÕÍÐ‰”…‰±”Ñ¼É•‰Õ¥±‰½Ñ ‰Õ‰‰±•Ì¸(€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰¥¹™±¥¡Ñ}ÑÕÉ¸‰ul‰ÕÍ•È‰t€ôô€‰½É¥¥¹…°É•ÅÕ•ÍÐˆ(€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰¥¹™±¥¡Ñ}ÑÕÉ¸‰ul‰½ÉÉ•Ñ¥½¹Ì‰t€ôôl‰ÕÍ”A½ÍÑÉ•Ì‰t(€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¸¹•Ð ‰±…ÍÑ}…Ñ¥Ù”ˆ¤¥Ì¹½Ð9½¹”(€€€…ÍÍ•ÉÐ‰•™½É”¥Ì9½¹”½ÈÍ•ÍÍ¥½¹l‰±…ÍÑ}…Ñ¥Ù”‰t€øô‰•™½É”(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}É•‘¥É•Ñ}ÉÁ}‘É½ÁÍ}ÅÕ•Õ•‘}‘ÕÁ±¥…Ñ•}½™}¥¹™±¥¡Ñ}ÕÍ•È ¤è(€€€€ˆˆˆŒàÐÐÄÜè•Í­Ñ½ÀÍ•ÍÍ¥½¸¹É•‘¥É•Ñ€µÕÍÐÁÕÉ”ÍÑ…±”Í•±˜µ‘ÕÁ±¥…Ñ•Ì¸((€€€AÉ½‘ÕÑ¥½¸Á…Ñ èÉ•¹‘•É•ÈÍÑ••ÉÌÙ¥„Í•ÍÍ¥½¸¹É•‘¥É•Ñ€€¡¹½Ð(€€€ÁÉ½µÁÐ¹ÍÕ‰µ¥Ñ€¤¸Í•±˜µ½Áä½˜Ñ¡”±¥Ù”½É¥¥¹…°ÕÍ•ÈÑ•áÐ…±É•…‘ä¥¸(€€€Ñ¡”Í•ÉÙ•ÈÅÕ•Õ”µÕÍÐ¹½ÐÍÕÉÙ¥Ù”„ÍÕ•ÍÍ™Õ°É•‘¥É•ÐƒŠP½Ñ¡•ÉÝ¥Í”(€€€Á½ÍÐµÑÕÉ¸}‘É…¥¹}ÅÕ•Õ•‘}ÁÉ½µÁÑ€É•ÍÑ…ÉÑÌÁÉ½µÁÐ@…™Ñ•ÈD¥Ì¡…¹‘±•¸(€€€U¹É•±…Ñ•¹•áÐµÑÕÉ¸•¹Ù•±½Á•ÌÍÑ…ä¸(€€€€ˆˆˆ(€€€½É¥¥¹…°€ô€‰‘••ÁÍ••¬É•±•…Í•„¹•Ü™±…Í µ½‘•°ƒŠP$¡…¹•…±°Í•ÑÑ¥¹ÌÑ¼™±…Í ˆ(€€€…•¹Ð€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€}ÍÕÁÁ½ÉÑÍ}…Ñ¥Ù•}ÑÕÉ¹}É•‘¥É•ÐõQÉÕ”°(€€€€€€€É•‘¥É•Ðõ±…µ‰‘„Ñ•áÐèQÉÕ”°(€€€€¤(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡…•¹Ðõ…•¹Ð°ÉÕ¹¹¥¹œõQÉÕ”¤(€€€Í•ÍÍ¥½¹l‰¥¹™±¥¡Ñ}ÑÕÉ¸‰t€ôì(€€€€€€€€‰ÕÍ•Èˆè½É¥¥¹…°°(€€€€€€€€‰…ÍÍ¥ÍÑ…¹Ðˆè€‰Á…ÉÑ¥…°ˆ°(€€€€€€€€‰ÍÑÉ•…µ¥¹œˆèQÉÕ”°(€€€€€€€€‰•ÉÉ½Èˆè€ˆˆ°(€€€ô(€€€Í•ÍÍ¥½¹l‰ÅÕ•Õ•‘}ÁÉ½µÁÐ‰t€ôì‰Ñ•áÐˆè½É¥¥¹…°°€‰ÑÉ…¹ÍÁ½ÉÐˆè€‰ÝÌ´Ä‰ô(€€€Í•ÍÍ¥½¹l‰ÅÕ•Õ•‘}ÁÉ½µÁÑÌ‰t€ôl(€€€€€€€ì‰Ñ•áÐˆè½É¥¥¹…°°€‰ÑÉ…¹ÍÁ½ÉÐˆè€‰ÝÌ´Ä‰ô°(€€€€€€€ì‰Ñ•áÐˆè€‰Õ¹É•±…Ñ•±…Ñ•ÈÑ…Í¬ˆ°€‰ÑÉ…¹ÍÁ½ÉÐˆè€‰ÝÌ´Ä‰ô°(€€€t(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ôÍ•ÍÍ¥½¸(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹É•‘¥É•Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰Ý¡…Ð…‰½ÕÐÑ¡”ÁÉ¥¥¹œ¥¹ÍÑ•…üˆ°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÍÑ…ÑÕÌ‰t€ôô€‰É•‘¥É•Ñ•ˆ(€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰¥¹™±¥¡Ñ}ÑÕÉ¸‰ul‰ÕÍ•È‰t€ôô½É¥¥¹…°(€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰¥¹™±¥¡Ñ}ÑÕÉ¸‰ul‰½ÉÉ•Ñ¥½¹Ì‰t€ôôl(€€€€€€€€‰Ý¡…Ð…‰½ÕÐÑ¡”ÁÉ¥¥¹œ¥¹ÍÑ•…üˆ(€€€t(€€€€ŒM•±˜µ‘ÕÁ±¥…Ñ•Ì½˜Ñ¡”±¥Ù”½É¥¥¹…°…É”½¹”ì±•¥Ñ¥µ…Ñ”™½±±½ÜµÕÀ­•ÁÐ¸(€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¸¹•Ð ‰ÅÕ•Õ•‘}ÁÉ½µÁÐˆ¤€ôôì(€€€€€€€€‰Ñ•áÐˆè€‰Õ¹É•±…Ñ•±…Ñ•ÈÑ…Í¬ˆ°(€€€€€€€€‰ÑÉ…¹ÍÁ½ÉÐˆè€‰ÝÌ´Äˆ°(€€€ô(€€€…ÍÍ•ÉÐ¹½ÐÍ•ÍÍ¥½¸¹•Ð ‰ÅÕ•Õ•‘}ÁÉ½µÁÑÌˆ¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}É•‘¥É•Ñ}‰Õ¥±‘}Ý¥¹‘½Ý}ÍÉÕ‰Í}ÍÑ…±•}Á}Ý¡•¹}ÅÕ•Õ¥¹}Ä ¤è(€€€€ˆˆˆŒàÐÐÄÜè‰Õ¥±µÝ¥¹‘½ÜÅÕ•Õ”½˜DµÕÍÐ¹½Ð±•…Ù”@…¡•…½˜D¸ˆˆˆ(€€€½É¥¥¹…°€ô€‰±¥Ù”½É¥¥¹…°@ˆ(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡ÉÕ¹¹¥¹œõQÉÕ”¤(€€€Í•ÍÍ¥½¹l‰…•¹Ð‰t€ô9½¹”€€Œ…Íå¹Œ…•¹Ð‰Õ¥±Ý¥¹‘½Ü(€€€Í•ÍÍ¥½¹l‰¥¹™±¥¡Ñ}ÑÕÉ¸‰t€ôì(€€€€€€€€‰ÕÍ•Èˆè½É¥¥¹…°°(€€€€€€€€‰…ÍÍ¥ÍÑ…¹Ðˆè€ˆˆ°(€€€€€€€€‰ÍÑÉ•…µ¥¹œˆèQÉÕ”°(€€€€€€€€‰•ÉÉ½Èˆè€ˆˆ°(€€€ô(€€€Í•ÍÍ¥½¹l‰ÅÕ•Õ•‘}ÁÉ½µÁÐ‰t€ôì‰Ñ•áÐˆè½É¥¥¹…°°€‰ÑÉ…¹ÍÁ½ÉÐˆè€‰ÝÌ´Ä‰ô(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ôÍ•ÍÍ¥½¸(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹É•‘¥É•Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰½ÉÉ•Ñ¥½¸D‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰t€ôôì‰ÍÑ…ÑÕÌˆè€‰ÅÕ•Õ•ˆ°€‰Ñ•áÐˆè€‰½ÉÉ•Ñ¥½¸D‰ô(€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰ÅÕ•Õ•‘}ÁÉ½µÁÐ‰ul‰Ñ•áÐ‰t€ôô€‰½ÉÉ•Ñ¥½¸Dˆ(€€€…ÍÍ•ÉÐ¹½ÐÍ•ÍÍ¥½¸¹•Ð ‰ÅÕ•Õ•‘}ÁÉ½µÁÑÌˆ¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}É•‘¥É•Ñ}É•½É‘Í}½ÉÉ•Ñ¥½¹}Ý¥Ñ¡½ÕÑ}•É…Í¥¹}ÁÉ½µÁÐ ¤è(€€€€ˆˆ‰É•‘¥É•ÐµÕÍÐ¹½Ð½Ù•ÉÝÉ¥Ñ”Ñ¡”ÑÕÉ¸Ì½É¥¥¹…°ÕÍ•ÈÑ•áÐ¸((€€€Q¡”¥¹™±¥¡ÐÍ¹…ÁÍ¡½Ð¥ÌÑ¡”½¹±äÑ¡¥¹œÍ•ÍÍ¥½¸¹É•ÍÕµ”…¸É•Á±…ä°Í¼(€€€½Ù•ÉÝÉ¥Ñ¥¹œÕÍ•É€•É…Í•Ñ¡”ÁÉ½µÁÐÑ¡…ÐÍÑ…ÉÑ•Ñ¡”ÑÕÉ¸…¹Ñ¡”(€€€±¥•¹ÐÉ•Á…¥¹Ñ•Ñ¡”Ñ¡É•…Ý¥Ñ Ñ¡”ÕÍ•ÈÌµ•ÍÍ…”µ¥ÍÍ¥¹œ¸(€€€€ˆˆˆ(€€€Í•ÍÍ¥½¸€ôíô(€€€Í•ÉÙ•È¹}ÍÑ…ÉÑ}¥¹™±¥¡Ñ}ÑÕÉ¸¡Í•ÍÍ¥½¸°€‰É•µ½Ù”Ñ¡”Í•ÍÍ¥½¸½Õ¹ÑÌˆ¤(€€€Í•ÉÙ•È¹}…ÁÁ•¹‘}¥¹™±¥¡Ñ}‘•±Ñ„¡Í•ÍÍ¥½¸°€‰5½Ù¥¹œ¸ˆ¤(€€€Í•ÉÙ•È¹}É•½É‘}¥¹™±¥¡Ñ}½ÉÉ•Ñ¥½¸¡Í•ÍÍ¥½¸°€‰¡ÕÉÉäÕÀˆ¤(€€€Í•ÉÙ•È¹}É•½É‘}¥¹™±¥¡Ñ}½ÉÉ•Ñ¥½¸¡Í•ÍÍ¥½¸°€‰…¹Ñ¡”Ý½É­ÑÉ•”½¹•Ìˆ¤((€€€Í¹…ÁÍ¡½Ð€ôÍ•ÉÙ•È¹}¥¹™±¥¡Ñ}Í¹…ÁÍ¡½Ð¡Í•ÍÍ¥½¸¤(€€€…ÍÍ•ÉÐÍ¹…ÁÍ¡½Ð¥Ì¹½Ð9½¹”((€€€…ÍÍ•ÉÐÍ¹…ÁÍ¡½Ñl‰ÕÍ•È‰t€ôô€‰É•µ½Ù”Ñ¡”Í•ÍÍ¥½¸½Õ¹ÑÌˆ(€€€…ÍÍ•ÉÐÍ¹…ÁÍ¡½Ñl‰½ÉÉ•Ñ¥½¹Ì‰t€ôôl‰¡ÕÉÉäÕÀˆ°€‰…¹Ñ¡”Ý½É­ÑÉ•”½¹•Ì‰t(()‘•˜Ñ•ÍÑ}¥¹™±¥¡Ñ}Í¹…ÁÍ¡½Ñ}…ÉÉ¥•Í}…ÉÉ¥Ù…±}½É‘•É}½™™Í•ÑÌ ¤è(€€€€ˆˆ‰… ½ÉÉ•Ñ¥½¸É•½É‘Ì¡½ÜµÕ …ÍÍ¥ÍÑ…¹ÐÑ•áÐ¡……±É•…‘äÍÑÉ•…µ•¸((€€€I•ÍÕµ¥¹œ±¥•¹ÑÌÉ•‰Õ¥±II%Y0½É‘•È™É½´Ñ¡•Í”‰½Õ¹‘…É¥•ÌèÑ¡”(€€€½ÉÉ•Ñ¥½¸‰Õ‰‰±”±…¹‘Ì…™Ñ•ÈÑ¡”½ÕÑÁÕÐÑ¡”ÕÍ•È¡……±É•…‘äÍ••¸…¹(€€€‰•™½É”Ñ¡”½ÕÑÁÕÐ¥ÐÉ•‘¥É•Ñ•€ ŒÜÌÜäÌ¤°¥¹ÍÑ•…½˜…‰½Ù”Ñ¡”Ý¡½±”(€€€É•Á±ä¸(€€€€ˆˆˆ(€€€Í•ÍÍ¥½¸€ôíô(€€€Í•ÉÙ•È¹}ÍÑ…ÉÑ}¥¹™±¥¡Ñ}ÑÕÉ¸¡Í•ÍÍ¥½¸°€‰É•µ½Ù”Ñ¡”Í•ÍÍ¥½¸½Õ¹ÑÌˆ¤(€€€Í•ÉÙ•È¹}…ÁÁ•¹‘}¥¹™±¥¡Ñ}‘•±Ñ„¡Í•ÍÍ¥½¸°€‰5½Ù¥¹œ¸ˆ¤(€€€Í•ÉÙ•È¹}É•½É‘}¥¹™±¥¡Ñ}½ÉÉ•Ñ¥½¸¡Í•ÍÍ¥½¸°€‰¡ÕÉÉäÕÀˆ¤(€€€Í•ÉÙ•È¹}…ÁÁ•¹‘}¥¹™±¥¡Ñ}‘•±Ñ„¡Í•ÍÍ¥½¸°€‰MÑ¥±°¸ˆ¤(€€€Í•ÉÙ•È¹}É•½É‘}¥¹™±¥¡Ñ}½ÉÉ•Ñ¥½¸¡Í•ÍÍ¥½¸°€‰…¹Ñ¡”Ý½É­ÑÉ•”½¹•Ìˆ¤(€€€Í•ÉÙ•È¹}…ÁÁ•¹‘}¥¹™±¥¡Ñ}‘•±Ñ„¡Í•ÍÍ¥½¸°€‰½¹”Í½½¸¸ˆ¤((€€€Í¹…ÁÍ¡½Ð€ôÍ•ÉÙ•È¹}¥¹™±¥¡Ñ}Í¹…ÁÍ¡½Ð¡Í•ÍÍ¥½¸¤(€€€…ÍÍ•ÉÐÍ¹…ÁÍ¡½Ð¥Ì¹½Ð9½¹”((€€€…ÍÍ•ÉÐÍ¹…ÁÍ¡½Ñl‰½ÉÉ•Ñ¥½¹Ì‰t€ôôl‰¡ÕÉÉäÕÀˆ°€‰…¹Ñ¡”Ý½É­ÑÉ•”½¹•Ì‰t(€€€…ÍÍ•ÉÐÍ¹…ÁÍ¡½Ñl‰½ÉÉ•Ñ¥½¹}½™™Í•ÑÌ‰t€ôôm±•¸ ‰5½Ù¥¹œ¸ˆ¤°±•¸ ‰5½Ù¥¹œ¹MÑ¥±°¸ˆ¥t(()‘•˜Ñ•ÍÑ}ÑÕÉ¹}…‘µ¥ÍÍ¥½¹}…ÉÉ¥•Í}Íå¹Ñ¡•Ñ¥}‘¥ÍÁ±…å}µ•Ñ…‘…Ñ…}¥¹Ñ½}¥¹™±¥¡Ñ}Í¹…ÁÍ¡½Ð¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰É•½¹¹•ÐµÕÍÐÉ•Ñ…¥¸Ñ¡”ÑåÁ•Íå¹Ñ¡•Ñ¥ŒÕÍ•È‰Õ‰‰±”€ ŒÄÄÈÄÐÐ¤¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•¹ÍÕÉ•}…Ñ¥Ù•}Í•ÍÍ¥½¹}Í±½Ðˆ°±…µ‰‘„€©}…ÉÌè9½¹”¤(€€€…•¹Ð€ô5½¬ ¤(€€€Í•ÍÍ¥½¸€ôì‰…•¹Ðˆè…•¹Ð°€‰…ÑÑ…¡•‘}¥µ…•Ìˆèmt°€‰¡¥ÍÑ½Éå}±½¬ˆèÑ¡É•…‘¥¹œ¹I1½¬ ¥ô(€€€‘¥ÍÁ±…å}µ•Ñ…‘…Ñ„€ôì‰‘¥ÍÁ±…å}Ñ•áÐˆè€‰¥¹¥Í¡•Íå¹¥¹œÑ¡”Ý½É­ÍÁ…”‰ô((€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}…‘µ¥Ñ}ÁÉ½µÁÑ}ÑÕÉ¸ (€€€€€€€€‰Í¥ˆ°Í•ÍÍ¥½¸°€‰ÁÉ½•ÍÌ½µÁ±•Ñ•ˆ°9½¹”°9½¹”°€‰ÁÉ½•ÍÍ}½µÁ±•Ñ”ˆ°‘¥ÍÁ±…å}µ•Ñ…‘…Ñ„°(€€€€¤€ôô€¡mt°…•¹Ð¤((€€€Í¹…ÁÍ¡½Ð€ôÍ•ÉÙ•È¹}¥¹™±¥¡Ñ}Í¹…ÁÍ¡½Ð¡Í•ÍÍ¥½¸¤((€€€…ÍÍ•ÉÐÍ¹…ÁÍ¡½Ð¥Ì¹½Ð9½¹”(€€€…ÍÍ•ÉÐÍ¹…ÁÍ¡½Ñl‰‘¥ÍÁ±…å}­¥¹‰t€ôô€‰ÁÉ½•ÍÍ}½µÁ±•Ñ”ˆ(€€€…ÍÍ•ÉÐÍ¹…ÁÍ¡½Ñl‰‘¥ÍÁ±…å}µ•Ñ…‘…Ñ„‰t€ôôì‰‘¥ÍÁ±…å}Ñ•áÐˆè€‰¥¹¥Í¡•Íå¹¥¹œÑ¡”Ý½É­ÍÁ…”‰ô(()‘•˜Ñ•ÍÑ}¥¹™±¥¡Ñ}Í¹…ÁÍ¡½Ñ}½µ¥ÑÍ}½™™Í•ÑÍ}Ý¡•¹}¹½Ñ}™Õ±±å}É•½É‘• ¤è(€€€€ˆˆ‰ÁÉ”µÕÁÉ…‘”¥¸µµ•µ½ÉäÑÕÉ¸µ…ä…ÉÉä½ÉÉ•Ñ¥½¹ÌÝ¥Ñ¡½ÕÐ½™™Í•ÑÌ¸((€€€Q¡”Á…É…±±•°±¥ÍÐ¥Ì½¹±äÍ•¹ÐÝ¡•¸•Ù•Éä½ÉÉ•Ñ¥½¸¡…Ì½¹”°Í¼±¥•¹ÑÌ(€€€…¸ÑÉÕÍÐÑ¡”Á…¥É¥¹œ…¹½±‘•ÈÍ¹…ÁÍ¡½ÑÌ‘•É…‘”Ñ¼Ñ¡”¹¼µ½™™Í•ÐÁ…Ñ ¸(€€€€ˆˆˆ(€€€Í•ÍÍ¥½¸€ôíô(€€€Í•ÉÙ•È¹}ÍÑ…ÉÑ}¥¹™±¥¡Ñ}ÑÕÉ¸¡Í•ÍÍ¥½¸°€‰ÁÉ½µÁÐˆ¤(€€€ÑÕÉ¸€ôÍ•ÍÍ¥½¹l‰¥¹™±¥¡Ñ}ÑÕÉ¸‰t(€€€ÑÕÉ¹l‰½ÉÉ•Ñ¥½¹Ì‰t€ôl‰±•…ä½ÉÉ•Ñ¥½¸‰t((€€€Í¹…ÁÍ¡½Ð€ôÍ•ÉÙ•È¹}¥¹™±¥¡Ñ}Í¹…ÁÍ¡½Ð¡Í•ÍÍ¥½¸¤(€€€…ÍÍ•ÉÐÍ¹…ÁÍ¡½Ð¥Ì¹½Ð9½¹”((€€€…ÍÍ•ÉÐÍ¹…ÁÍ¡½Ñl‰½ÉÉ•Ñ¥½¹Ì‰t€ôôl‰±•…ä½ÉÉ•Ñ¥½¸‰t(€€€…ÍÍ•ÉÐ€‰½ÉÉ•Ñ¥½¹}½™™Í•ÑÌˆ¹½Ð¥¸Í¹…ÁÍ¡½Ð(()‘•˜Ñ•ÍÑ}¥¹™±¥¡Ñ}Í¹…ÁÍ¡½Ñ}½µ¥ÑÍ}½ÉÉ•Ñ¥½¹Í}Ý¡•¹}¹½¹•}É•½É‘• ¤è(€€€Í•ÍÍ¥½¸€ôíô(€€€Í•ÉÙ•È¹}ÍÑ…ÉÑ}¥¹™±¥¡Ñ}ÑÕÉ¸¡Í•ÍÍ¥½¸°€‰©ÕÍÐÑ¡”ÁÉ½µÁÐˆ¤((€€€Í¹…ÁÍ¡½Ð€ôÍ•ÉÙ•È¹}¥¹™±¥¡Ñ}Í¹…ÁÍ¡½Ð¡Í•ÍÍ¥½¸¤(€€€…ÍÍ•ÉÐÍ¹…ÁÍ¡½Ð¥Ì¹½Ð9½¹”(€€€…ÍÍ•ÉÐ€‰½ÉÉ•Ñ¥½¹Ìˆ¹½Ð¥¸Í¹…ÁÍ¡½Ð(()‘•˜Ñ•ÍÑ}¹•Ý}ÑÕÉ¹}‘½•Í}¹½Ñ}¥¹¡•É¥Ñ}ÁÉ¥½É}ÑÕÉ¹}½ÉÉ•Ñ¥½¹Ì ¤è(€€€Í•ÍÍ¥½¸€ôíô(€€€Í•ÉÙ•È¹}ÍÑ…ÉÑ}¥¹™±¥¡Ñ}ÑÕÉ¸¡Í•ÍÍ¥½¸°€‰™¥ÉÍÐÁÉ½µÁÐˆ¤(€€€Í•ÉÙ•È¹}É•½É‘}¥¹™±¥¡Ñ}½ÉÉ•Ñ¥½¸¡Í•ÍÍ¥½¸°€‰™¥ÉÍÐ½ÉÉ•Ñ¥½¸ˆ¤(€€€Í•ÉÙ•È¹}ÍÑ…ÉÑ}¥¹™±¥¡Ñ}ÑÕÉ¸¡Í•ÍÍ¥½¸°€‰Í•½¹ÁÉ½µÁÐˆ¤((€€€Í¹…ÁÍ¡½Ð€ôÍ•ÉÙ•È¹}¥¹™±¥¡Ñ}Í¹…ÁÍ¡½Ð¡Í•ÍÍ¥½¸¤(€€€…ÍÍ•ÉÐÍ¹…ÁÍ¡½Ð¥Ì¹½Ð9½¹”((€€€…ÍÍ•ÉÐÍ¹…ÁÍ¡½Ñl‰ÕÍ•È‰t€ôô€‰Í•½¹ÁÉ½µÁÐˆ(€€€…ÍÍ•ÉÐ€‰½ÉÉ•Ñ¥½¹Ìˆ¹½Ð¥¸Í¹…ÁÍ¡½Ð(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}É•‘¥É•Ñ}ÅÕ•Õ•Í}‘ÕÉ¥¹}…•¹Ñ}‰Õ¥±‘}Ý¥¹‘½Ü¡µ½¹­•åÁ…Ñ ¤è(€€€€Œ™É•Í ÑÕÉ¸™±¥ÁÌÉÕ¹¹¥¹œõQÉÕ”…¹‰Õ¥±‘ÌÑ¡”…•¹Ð…Íå¹¡É½¹½ÕÍ±ä°Í¼(€€€€ŒÍ•ÍÍ¥½¹l‰…•¹Ð‰t¥Ì‰É¥•™±ä9½¹”¸½ÉÉ•Ñ¥½¸±…¹‘¥¹œ¡•É”µÕÍÐÅÕ•Õ”(€€€€Œ€¡±½ÍÍ±•ÍÌ°É•…¡•ÌÑ¡”µ½‘•°¹•áÐÑÕÉ¸¤°¹½Ð¡…ÉµÉ•©•Ð…ÌÕ¹ÍÕÁÁ½ÉÑ•¸(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡ÉÕ¹¹¥¹œõQÉÕ”¤(€€€Í•ÍÍ¥½¹l‰…•¹Ð‰t€ô9½¹”(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ôÍ•ÍÍ¥½¸(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹É•‘¥É•Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰Ý…¥Ð°ÕÍ”ME1¥Ñ”‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰t€ôôì‰ÍÑ…ÑÕÌˆè€‰ÅÕ•Õ•ˆ°€‰Ñ•áÐˆè€‰Ý…¥Ð°ÕÍ”ME1¥Ñ”‰ô(€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰ÅÕ•Õ•‘}ÁÉ½µÁÐ‰ul‰Ñ•áÐ‰t€ôô€‰Ý…¥Ð°ÕÍ”ME1¥Ñ”ˆ(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}É•‘¥É•Ñ}É•©•ÑÍ}Ý¡•¹}¥‘±•}Ý¥Ñ¡½ÕÑ}…•¹Ð¡µ½¹­•åÁ…Ñ ¤è(€€€€Œ9¼±¥Ù”ÑÕÉ¸…¹¹¼…•¹Ðè¹½Ñ¡¥¹œÑ¼É•‘¥É•Ð°…¹Ý”µÕÍÐ¹½ÐÅÕ•Õ”„(€€€€ŒÁ¡…¹Ñ½´ÑÕÉ¸ƒŠP­••ÀÑ¡”•áÁ±¥¥ÐÕ¹ÍÕÁÁ½ÉÑ•É•©•Ñ¥½¸¸(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡ÉÕ¹¹¥¹œõ…±Í”¤(€€€Í•ÍÍ¥½¹l‰…•¹Ð‰t€ô9½¹”(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ôÍ•ÍÍ¥½¸(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹É•‘¥É•Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰¡¤‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÄÀ(€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¸¹•Ð ‰ÅÕ•Õ•‘}ÁÉ½µÁÐˆ¤¥Ì9½¹”(((()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}¥¹™½}¥¹±Õ‘•Í}Í•ÍÍ¥½¹}Ñ¥Ñ±”¡µ½¹­•åÁ…Ñ ¤è(€€€±…ÍÌ}…­•è(€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¹}Ñ¥Ñ±”¡Í•±˜°­•ä¤è(€€€€€€€€€€€…ÍÍ•ÉÐ­•ä€ôô€‰Í•ÍÍ¥½¸µ­•äˆ(€€€€€€€€€€€É•ÑÕÉ¸€‰…Í¡‰½…ÉÑ¥Ñ±”ˆ((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è}…­• ¤¤((€€€¥¹™¼€ôÍ•ÉÙ•È¹}Í•ÍÍ¥½¹}¥¹™¼ (€€€€€€€ÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡Ñ½½±Ìõmt°µ½‘•°ô‰Ñ•ÍÐ½µ½‘•°ˆ°ÁÉ½Ù¥‘•Èô‰½Á•¹…¤µ½‘•àˆ¤°(€€€€€€€ì‰Í•ÍÍ¥½¹}­•äˆè€‰Í•ÍÍ¥½¸µ­•äˆ°€‰¡¥ÍÑ½Éäˆèmuô°(€€€€¤((€€€…ÍÍ•ÉÐ¥¹™½l‰Ñ¥Ñ±”‰t€ôô€‰…Í¡‰½…ÉÑ¥Ñ±”ˆ(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}¥¹™½}É•Á½ÉÑÍ}Á•¹‘¥¹}µ½‘•±}ÍÝ¥Ñ ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰µ½‘•°ÅÕ•Õ•µ¥µÑÕÉ¸Í¡½ÝÌ…ÌÑ¡”Í•ÍÍ¥½¸Ìµ½‘•°¥¸Í•ÍÍ¥½¸¹¥¹™¼°Í¼(€€€Ñ¡”•¹µ½˜µÑÕÉ¸Í•ÑÑ±”‘½•Í¸Ð‰±¥ÀÑ¡”U$‰…¬Ñ¼Ñ¡”ÍÑ¥±°µ±¥Ù”½±µ½‘•°(€€€‰•™½É”Ñ¡”ÍÝ¥Ñ …ÁÁ±¥•Ì…ÐÑ¡”¹•áÐÑÕÉ¸ÍÑ…ÉÐ¸ˆˆˆ(€€€…•¹Ð€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡Ñ½½±Ìõmt°µ½‘•°ô‰½±½µ½‘•°ˆ°ÁÉ½Ù¥‘•Èô‰½Á•¹…¤ˆ¤(€€€Í•ÍÍ¥½¸€ôì(€€€€€€€€‰Í•ÍÍ¥½¹}­•äˆè€ˆˆ°(€€€€€€€€‰¡¥ÍÑ½Éäˆèmt°(€€€€€€€€‰Á•¹‘¥¹}µ½‘•±}ÍÝ¥Ñ ˆèì(€€€€€€€€€€€€‰É…Üˆè€‰¹•Ü½µ½‘•°€´µÁÉ½Ù¥‘•È…¹Ñ¡É½Á¥Œˆ°(€€€€€€€€€€€€‰‘¥ÍÁ±…å}µ½‘•°ˆè€‰¹•Ü½µ½‘•°ˆ°(€€€€€€€€€€€€‰‘¥ÍÁ±…å}ÁÉ½Ù¥‘•Èˆè€‰…¹Ñ¡É½Á¥Œˆ°(€€€€€€€ô°(€€€ô((€€€¥¹™¼€ôÍ•ÉÙ•È¹}Í•ÍÍ¥½¹}¥¹™¼¡…•¹Ð°Í•ÍÍ¥½¸¤(€€€…ÍÍ•ÉÐ¥¹™½l‰µ½‘•°‰t€ôô€‰¹•Ü½µ½‘•°ˆ(€€€…ÍÍ•ÉÐ¥¹™½l‰ÁÉ½Ù¥‘•È‰t€ôô€‰…¹Ñ¡É½Á¥Œˆ((€€€€Œ]¥Ñ ¹½Ñ¡¥¹œÅÕ•Õ•Ñ¡”±¥Ù”…•¹Ðµ½‘•°Ý¥¹Ì°…Ì‰•™½É”¸(€€€Í•ÍÍ¥½¸¹Á½À ‰Á•¹‘¥¹}µ½‘•±}ÍÝ¥Ñ ˆ¤(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹}¥¹™¼¡…•¹Ð°Í•ÍÍ¥½¸¥l‰µ½‘•°‰t€ôô€‰½±½µ½‘•°ˆ(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}¥¹™½}¥¹±Õ‘•Í}ÑÕÉ¹}ÍÑ…ÉÑ•‘}…Ð ¤è(€€€…•¹Ð€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡Ñ½½±Ìõmt°µ½‘•°ôˆˆ°ÁÉ½Ù¥‘•Èôˆˆ¤(€€€Í•ÍÍ¥½¸€ôì(€€€€€€€€‰¡¥ÍÑ½Éäˆèmt°(€€€€€€€€‰¥¹™±¥¡Ñ}ÑÕÉ¸ˆèì‰ÍÑ…ÉÑ•‘}…Ðˆè€Å|ÜÀÁ|ÀÀÁ|ÄÈÌ¸Õô°(€€€€€€€€‰ÉÕ¹¹¥¹œˆèQÉÕ”°(€€€ô((€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹}¥¹™¼¡…•¹Ð°Í•ÍÍ¥½¸¥l‰ÑÕÉ¹}ÍÑ…ÉÑ•‘}…Ð‰t€ôô€Å|ÜÀÁ|ÀÀÁ|ÄÈÌ¸Ô((€€€Í•ÍÍ¥½¹l‰¥¹™±¥¡Ñ}ÑÕÉ¸‰t€ô9½¹”(€€€Í•ÍÍ¥½¹l‰ÉÕ¹¹¥¹œ‰t€ô…±Í”(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹}¥¹™¼¡…•¹Ð°Í•ÍÍ¥½¸¥l‰ÑÕÉ¹}ÍÑ…ÉÑ•‘}…Ð‰t¥Ì9½¹”(((Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(Œ!¥ÍÑ½ÉäµµÕÑ…Ñ¥¹œ½µµ…¹‘ÌµÕÍÐÉ•©•ÐÝ¡¥±”Í•ÍÍ¥½¸¹ÉÕ¹¹¥¹œ¥ÌQÉÕ”¸(Œ]¥Ñ¡½ÕÐÑ¡•Í”Õ…É‘Ì°ÁÉ½µÁÐ¹ÍÕ‰µ¥ÐÌÁ½ÍÐµÉÕ¸¡¥ÍÑ½ÉäÝÉ¥Ñ”•¥Ñ¡•È(Œ±½‰‰•ÉÌÑ¡”µÕÑ…Ñ¥½¸€¡Ù•ÉÍ¥½¸µ…Ñ¡•Ì¤½ÈÍ¥±•¹Ñ±ä‘É½ÁÌÑ¡”…•¹ÐÌ(Œ½ÕÑÁÕÐ€¡Ù•ÉÍ¥½¸µ¥Íµ…Ñ ¤ƒŠP‰½Ñ ÁÉ½‘Õ”U$ð´ù‰…­•¹ÍÑ…Ñ”‘•Íå¹Œ¸(Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}Õ¹‘½}É•©•ÑÍ}Ý¡¥±•}ÉÕ¹¹¥¹œ ¤è(€€€€ˆˆ‰¥à™½ÈQU$Í¥±•¹Ðµ‘É½À€ŒÄè€½Õ¹‘¼µÕÍÐ¹½ÐµÕÑ…Ñ”¡¥ÍÑ½Éä(€€€Ý¡¥±”Ñ¡”…•¹Ð¥Ìµ¥µÑÕÉ¸ƒŠPÝ½Õ±•¥Ñ¡•È±½‰‰•ÈÑ¡”Õ¹‘¼½È(€€€…ÕÍ”ÁÉ½µÁÐ¹ÍÕ‰µ¥ÐÑ¼Í¥±•¹Ñ±ä‘É½ÀÑ¡”…•¹ÐÌÉ•ÍÁ½¹Í”¸ˆˆˆ(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸ (€€€€€€€ÉÕ¹¹¥¹œõQÉÕ”°(€€€€€€€¡¥ÍÑ½Éäõl(€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰¡¤‰ô°(€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰¡•±±¼‰ô°(€€€€€€€t°(€€€€¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹Õ¹‘¼ˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥‰õô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰•ÉÉ½Èˆ¤°€‰Í•ÍÍ¥½¸¹Õ¹‘¼Í¡½Õ±É•©•ÐÝ¡¥±”ÉÕ¹¹¥¹œˆ(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÀä(€€€€€€€€Œ!¥ÍÑ½ÉäµÕÍÐ‰”Õ¹¡…¹•(€€€€€€€…ÍÍ•ÉÐ±•¸¡Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰ul‰¡¥ÍÑ½Éä‰t¤€ôô€È(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}Õ¹‘½}…±±½Ý•‘}Ý¡•¹}¥‘±” ¤è(€€€€ˆˆ‰I•É•ÍÍ¥½¸Õ…ÉèÝ¡•¸¹½ÐÉÕ¹¹¥¹œ°€½Õ¹‘¼ÍÑ¥±°Ý½É­Ì¸ˆˆˆ(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸ (€€€€€€€ÉÕ¹¹¥¹œõ…±Í”°(€€€€€€€Í•ÍÍ¥½¹}­•äôˆˆ°(€€€€€€€¡¥ÍÑ½Éäõl(€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰¡¤‰ô°(€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰¡•±±¼‰ô°(€€€€€€€t°(€€€€¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹Õ¹‘¼ˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥‰õô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÉ•ÍÀ¹•Ð •ÉÉ½Èœ¥ôˆ(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰É•µ½Ù•‰t€ôô€È(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰ul‰¡¥ÍÑ½Éä‰t€ôômt(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}½µÁÉ•ÍÍ}É•©•ÑÍ}Ý¡¥±•}ÉÕ¹¹¥¹œ¡µ½¹­•åÁ…Ñ ¤è(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸¡ÉÕ¹¹¥¹œõQÉÕ”¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹½µÁÉ•ÍÌˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥‰õô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰•ÉÉ½Èˆ¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÀä(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}É½±±‰…­}É•ÍÑ½É•}É•©•ÑÍ}™Õ±±}¡¥ÍÑ½Éå}Ý¡¥±•}ÉÕ¹¹¥¹œ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰Õ±°µ¡¥ÍÑ½ÉäÉ½±±‰…¬µÕÍÐÉ•©•Ðì™¥±”µÍ½Á•É½±±‰…¬ÍÑ¥±°…±±½Ý•¸ˆˆˆ(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸¡ÉÕ¹¹¥¹œõQÉÕ”¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰É½±±‰…¬¹É•ÍÑ½É”ˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰¡…Í ˆè€‰…‰Œ‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰•ÉÉ½Èˆ¤°€‰™Õ±°µ¡¥ÍÑ½ÉäÉ½±±‰…¬Í¡½Õ±É•©•ÐÝ¡¥±”ÉÕ¹¹¥¹œˆ(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÀä(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}¡¥ÍÑ½Éå}Ù•ÉÍ¥½¹}µ¥Íµ…Ñ¡}ÍÕÉ™…•Í}Ý…É¹¥¹œ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰¥à™½ÈQU$Í¥±•¹Ðµ‘É½À€ŒÈèÑ¡”‘•™•¹Í¥Ù”‰…­ÍÑ½À…ÐÁÉ½µÁÐ¹ÍÕ‰µ¥Ð(€€€µÕÍÐ…ÑÑ… „€Ý…É¹¥¹œœÑ¼µ•ÍÍ…”¹½µÁ±•Ñ”Ý¡•¸¡¥ÍÑ½ÉäÝ…Ì(€€€µÕÑ…Ñ••áÑ•É¹…±±ä‘ÕÉ¥¹œÑ¡”ÑÕÉ¸€¡¥¹ÍÑ•…½˜Í¥±•¹Ñ±ä‘É½ÁÁ¥¹œ(€€€Ñ¡”…•¹ÐÌ½ÕÑÁÕÐ¤¸ˆˆˆ(€€€€Œ•¹Ð‰ÕµÁÌ¡¥ÍÑ½Éå}Ù•ÉÍ¥½¸¥ÑÍ•±˜µ¥µÉÕ¸Ñ¼Í¥µÕ±…Ñ”…¸•áÑ•É¹…°(€€€€ŒµÕÑ…Ñ¥½¸Í±¥ÁÁ¥¹œÁ…ÍÐÑ¡”Õ…É‘Ì¸(€€€Í•ÍÍ¥½¹}É•˜€ôì‰Ìˆè9½¹•ô((€€€±…ÍÌ}I…å•¹Ðè(€€€€€€€‘•˜ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°ÁÉ½µÁÐ°½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éäõ9½¹”°ÍÑÉ•…µ}…±±‰…¬õ9½¹”°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€€ŒM¥µÕ±…Ñ”èÍ½µ•Ñ¡¥¹œ•áÑ•É¹…°‰ÕµÁ•¡¥ÍÑ½Éå}Ù•ÉÍ¥½¸(€€€€€€€€€€€€ŒÝ¡¥±”Ý”Ý•É”ÉÕ¹¹¥¹œ¸(€€€€€€€€€€€Ý¥Ñ Í•ÍÍ¥½¹}É•™l‰Ì‰ul‰¡¥ÍÑ½Éå}±½¬‰tè(€€€€€€€€€€€€€€€Í•ÍÍ¥½¹}É•™l‰Ì‰ul‰¡¥ÍÑ½Éå}Ù•ÉÍ¥½¸‰t€¬ô€Ä(€€€€€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€€€€€‰™¥¹…±}É•ÍÁ½¹Í”ˆè€‰…•¹ÐÉ•Á±äˆ°(€€€€€€€€€€€€€€€€‰µ•ÍÍ…•Ìˆèmì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰…•¹ÐÉ•Á±ä‰õt°(€€€€€€€€€€€ô((€€€±…ÍÌ}%µµ•‘¥…Ñ•Q¡É•…è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©}Ñ¡É•…‘}½ÁÑ¥½¹Ì¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð€ôÑ…É•Ð((€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð ¤((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸¡…•¹Ðõ}I…å•¹Ð ¤¤(€€€Í•ÍÍ¥½¹}É•™l‰Ì‰t€ôÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t(€€€•µ¥ÑÌè±¥ÍÑmÑÕÁ±•t€ômt(€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}%µµ•‘¥…Ñ•Q¡É•…¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}ÕÍ…”ˆ°±…µ‰‘„}„èíô¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰É•¹‘•É}µ•ÍÍ…”ˆ°±…µ‰‘„}Ð°}Œè€ˆˆ¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„è•µ¥ÑÌ¹…ÁÁ•¹¡„¤¤((€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰¡¤‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÉ•ÍÀ¹•Ð •ÉÉ½Èœ¥ôˆ((€€€€€€€€Œ!¥ÍÑ½ÉäÍ¡½Õ±9=P½¹Ñ…¥¸Ñ¡”…•¹ÐÌ½ÕÑÁÕÐ€¡Ù•ÉÍ¥½¸µ¥Íµ…Ñ ¤(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰ul‰¡¥ÍÑ½Éä‰t€ôômt((€€€€€€€€Œµ•ÍÍ…”¹½µÁ±•Ñ”µÕÍÐ…ÉÉä„€Ý…É¹¥¹œœÍ¼Ñ¡”U$€¼½Á•É…Ñ½È(€€€€€€€€Œ­¹½ÝÌÑ¡”½ÕÑÁÕÐÝ…Ì¹½ÐÁ•ÉÍ¥ÍÑ•¸(€€€€€€€½µÁ±•Ñ•}…±±Ì€ôm„™½È„¥¸•µ¥ÑÌ¥˜…lÁt€ôô€‰µ•ÍÍ…”¹½µÁ±•Ñ”‰t(€€€€€€€…ÍÍ•ÉÐ±•¸¡½µÁ±•Ñ•}…±±Ì¤€ôô€Ä(€€€€€€€|°|°Á…å±½…€ô½µÁ±•Ñ•}…±±ÍlÁt(€€€€€€€…ÍÍ•ÉÐ€‰Ý…É¹¥¹œˆ¥¸Á…å±½…°€ (€€€€€€€€€€€€‰µ•ÍÍ…”¹½µÁ±•Ñ”µÕÍÐ¥¹±Õ‘”„€Ý…É¹¥¹œœ™¥•±½¸€ˆ(€€€€€€€€€€€€‰¡¥ÍÑ½Éå}Ù•ÉÍ¥½¸µ¥Íµ…Ñ ƒŠP½Ñ¡•ÉÝ¥Í”Ñ¡”U$Í¥±•¹Ñ±ä€ˆ(€€€€€€€€€€€€‰Í¡½ÝÌ½ÕÑÁÕÐÑ¡…ÐÝ…Ì¹•Ù•ÈÁ•ÉÍ¥ÍÑ•ˆ(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐ€ (€€€€€€€€€€€€‰¹½ÐÍ…Ù•ˆ¥¸Á…å±½…‘l‰Ý…É¹¥¹œ‰t¹±½Ý•È ¤(€€€€€€€€€€€½È€‰¡…¹•ˆ¥¸Á…å±½…‘l‰Ý…É¹¥¹œ‰t¹±½Ý•È ¤(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}µ•É•Í}½¹}µ½‘•±}ÍÝ¥Ñ¡}µ…É­•È¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆˆŒÜØàÜÀèÝ¡•¸„µ½‘•°µÍÝ¥Ñ µ…É­•È¥ÌÑ¡”½¹±ä¡¥ÍÑ½ÉäµÕÑ…Ñ¥½¸‘ÕÉ¥¹œ(€€€„ÑÕÉ¸°Ñ¡”…•¹ÐÌ½ÕÑÁÕÐµÕÍÐ‰”µ•É•¥¹Ñ¼Ñ¡”ÕÉÉ•¹Ð¡¥ÍÑ½Éä€¡Ý¡¥ (€€€¹½Ü½¹Ñ…¥¹ÌÑ¡”µ…É­•È¤¥¹ÍÑ•…½˜‰•¥¹œ‘¥Í…É‘•¸((€€€Q¡¥ÌÑ•ÍÐ½Ù•ÉÌ	=Q …Í•Ìè(€€€€´9¼ÁÉ¥½Èµ…É­•È¥¸ÑÕÉ¸µÍÑ…ÉÐ¡¥ÍÑ½Éä€¡™¥ÉÍÐÍÝ¥Ñ ¥¸„Í•ÍÍ¥½¸¤(€€€€´AÉ¥½Èµ…É­•È•á¥ÍÑ•€¡•Ù•ÉäÍÕ‰Í•ÅÕ•¹ÐÍÝ¥Ñ ƒŠPÑ¡”½É¥¥¹…°AH€ŒÜÜÈÜÐ(€€€€€™¥àÝ…Ì‘•…½‘”¡•É”‰•…ÕÍ”}…ÁÁ•¹‘}µ½‘•±}ÍÝ¥Ñ¡}µ…É­•ÈÍÑÉ¥ÁÌÑ¡”(€€€€€½±µ…É­•È‰•™½É”…ÁÁ•¹‘¥¹œÑ¡”¹•Ü½¹”°ÁÉ½‘Õ¥¹œ„¹•Ðµé•É¼±•¹Ñ (€€€€€‘•±Ñ„Ñ¡…ÐÑ¡”Á½Í¥Ñ¥½¹…°Í±¥”µ¥ÍÍ•¤¸(€€€€ˆˆˆ(€€€™É½´ÑÕ¥}…Ñ•Ý…ä¹Í•ÉÙ•È¥µÁ½ÉÐ}5=1}M]%Q!}5I-I}AI%`((€€€Í•ÍÍ¥½¹}É•˜€ôì‰Ìˆè9½¹•ô((€€€‘•˜}µ…­•}µ…É­•È¡µ½‘•°èÍÑÈ¤€´ø‘¥Ðè(€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€‰É½±”ˆè€‰ÕÍ•Èˆ°(€€€€€€€€€€€€‰½¹Ñ•¹Ðˆè˜‰í}5=1}M]%Q!}5I-I}AI%aõíµ½‘•±ô¹tˆ°(€€€€€€€€€€€€‰‘¥ÍÁ±…å}­¥¹ˆè€‰µ½‘•±}ÍÝ¥Ñ ˆ°(€€€€€€€ô((€€€±…ÍÌ}5…É­•É•¹Ðè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°¹•Ý}¡¥ÍÑ½Éå}ÍÑ…Ñ”è±¥ÍÐ¤è(€€€€€€€€€€€Í•±˜¹}¹•Ý}¡¥ÍÑ½Éå}ÍÑ…Ñ”€ô¹•Ý}¡¥ÍÑ½Éå}ÍÑ…Ñ”((€€€€€€€‘•˜ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°ÁÉ½µÁÐ°½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éäõ9½¹”°ÍÑÉ•…µ}…±±‰…¬õ9½¹”°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€€ŒM¥µÕ±…Ñ”}…ÁÁ•¹‘}µ½‘•±}ÍÝ¥Ñ¡}µ…É­•ÈèÍÑÉ¥ÀÁÉ¥½Èµ…É­•ÉÌ°…ÁÁ•¹¹•Ü½¹”¸(€€€€€€€€€€€Ý¥Ñ Í•ÍÍ¥½¹}É•™l‰Ì‰ul‰¡¥ÍÑ½Éå}±½¬‰tè(€€€€€€€€€€€€€€€¡¥ÍÐ€ôÍ•ÍÍ¥½¹}É•™l‰Ì‰ul‰¡¥ÍÑ½Éä‰t(€€€€€€€€€€€€€€€¡¥ÍÑlét€ôm ™½È ¥¸¡¥ÍÐ¥˜¹½Ð}¥Í}µ…É­•È¡ ¥t(€€€€€€€€€€€€€€€¡¥ÍÐ¹…ÁÁ•¹¡}µ…­•}µ…É­•È ‰¹•Üµµ½‘•°ˆ¤¤(€€€€€€€€€€€€€€€Í•ÍÍ¥½¹}É•™l‰Ì‰ul‰¡¥ÍÑ½Éå}Ù•ÉÍ¥½¸‰t€¬ô€Ä(€€€€€€€€€€€€ŒÉ•ÍÕ±Ñl‰µ•ÍÍ…•Ì‰t€ô½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éä€¬ÕÍ•ÈµÍœ€¬…ÍÍ¥ÍÑ…¹ÐÉ•Á±ä(€€€€€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€€€€€‰™¥¹…±}É•ÍÁ½¹Í”ˆè€‰…•¹ÐÉ•Á±äˆ°(€€€€€€€€€€€€€€€€‰µ•ÍÍ…•Ìˆè±¥ÍÐ¡½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éä¤€¬l(€€€€€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰¡¤‰ô°(€€€€€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰…•¹ÐÉ•Á±ä‰ô°(€€€€€€€€€€€€€€€t°(€€€€€€€€€€€ô((€€€‘•˜}¥Í}µ…É­•È¡•¹ÑÉä¤€´ø‰½½°è(€€€€€€€™É½´ÑÕ¥}…Ñ•Ý…ä¹Í•ÉÙ•È¥µÁ½ÉÐ}¥Í}µ½‘•±}ÍÝ¥Ñ¡}µ…É­•È(€€€€€€€É•ÑÕÉ¸}¥Í}µ½‘•±}ÍÝ¥Ñ¡}µ…É­•È¡•¹ÑÉä¤((€€€±…ÍÌ}%µµ•‘¥…Ñ•Q¡É•…è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©}Ñ¡É•…‘}½ÁÑ¥½¹Ì¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð€ôÑ…É•Ð((€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð ¤((€€€€ŒQ•ÍÐ‰½Ñ è¹¼ÁÉ¥½Èµ…É­•È°…¹ÁÉ¥½Èµ…É­•ÈÁÉ•Í•¹Ð(€€€™½È±…‰•°°ÁÉ¥½É}¡¥ÍÑ½Éä¥¸l(€€€€€€€€ ‰¹¼ÁÉ¥½Èµ…É­•Èˆ°mì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰¡•±±¼‰õt¤°(€€€€€€€€ ‰Ý¥Ñ ÁÉ¥½Èµ…É­•Èˆ°l(€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰¡•±±¼‰ô°(€€€€€€€€€€€}µ…­•}µ…É­•È ‰½±µµ½‘•°ˆ¤°(€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰¡¤Ñ¡•É”‰ô°(€€€€€€€t¤°(€€€tè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸ (€€€€€€€€€€€…•¹Ðõ}5…É­•É•¹Ð¡mt¤°(€€€€€€€€€€€¡¥ÍÑ½Éäõ±¥ÍÐ¡ÁÉ¥½É}¡¥ÍÑ½Éä¤°(€€€€€€€€¤(€€€€€€€Í•ÍÍ¥½¹}É•™l‰Ì‰t€ôÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t(€€€€€€€•µ¥ÑÌè±¥ÍÑmÑÕÁ±•t€ômt(€€€€€€€ÑÉäè(€€€€€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}%µµ•‘¥…Ñ•Q¡É•…¤(€€€€€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}ÕÍ…”ˆ°±…µ‰‘„}„èíô¤(€€€€€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰É•¹‘•É}µ•ÍÍ…”ˆ°±…µ‰‘„}Ð°}Œè€ˆˆ¤(€€€€€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„è•µ¥ÑÌ¹…ÁÁ•¹¡„¤¤((€€€€€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰¡¤‰ô°(€€€€€€€€€€€€€€€ô(€€€€€€€€€€€€¤(€€€€€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰mí±…‰•±õt½Ð•ÉÉ½ÈèíÉ•ÍÀ¹•Ð •ÉÉ½Èœ¥ôˆ((€€€€€€€€€€€™¥¹…±}¡¥ÍÑ½Éä€ôÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰ul‰¡¥ÍÑ½Éä‰t((€€€€€€€€€€€€ŒQ¡”…•¹ÐÌ¹•Üµ•ÍÍ…•ÌµÕÍÐ‰”ÁÉ•Í•¹Ð¥¸Ñ¡”Á•ÉÍ¥ÍÑ•¡¥ÍÑ½Éä¸(€€€€€€€€€€€…ÍÍ¥ÍÑ…¹Ñ}µÍÌ€ôl(€€€€€€€€€€€€€€€”™½È”¥¸™¥¹…±}¡¥ÍÑ½Éä(€€€€€€€€€€€€€€€¥˜¥Í¥¹ÍÑ…¹”¡”°‘¥Ð¤…¹”¹•Ð ‰É½±”ˆ¤€ôô€‰…ÍÍ¥ÍÑ…¹Ðˆ(€€€€€€€€€€€€€€€…¹”¹•Ð ‰½¹Ñ•¹Ðˆ¤€ôô€‰…•¹ÐÉ•Á±äˆ(€€€€€€€€€€€t(€€€€€€€€€€€…ÍÍ•ÉÐ±•¸¡…ÍÍ¥ÍÑ…¹Ñ}µÍÌ¤€ôô€Ä°€ (€€€€€€€€€€€€€€€˜‰mí±…‰•±õt…•¹Ð½ÕÑÁÕÐÝ…Ì¹½Ðµ•É•¥¹Ñ¼¡¥ÍÑ½Éä€ˆ(€€€€€€€€€€€€€€€˜ˆ¡½Ðí±•¸¡…ÍÍ¥ÍÑ…¹Ñ}µÍÌ¥ô…ÍÍ¥ÍÑ…¹Ð€…•¹ÐÉ•Á±äœµ•ÍÍ…•Ì¤ˆ(€€€€€€€€€€€€¤((€€€€€€€€€€€€ŒQ¡”µ½‘•°µÍÝ¥Ñ µ…É­•ÈµÕÍÐ‰”ÁÉ•Í•¹Ð¸(€€€€€€€€€€€µ…É­•ÉÌ€ôm”™½È”¥¸™¥¹…±}¡¥ÍÑ½Éä¥˜}¥Í}µ…É­•È¡”¥t(€€€€€€€€€€€…ÍÍ•ÉÐ±•¸¡µ…É­•ÉÌ¤€ôô€Ä°€ (€€€€€€€€€€€€€€€˜‰mí±…‰•±õt•áÁ•Ñ••á…Ñ±ä€Äµ½‘•°µÍÝ¥Ñ µ…É­•È°½Ðí±•¸¡µ…É­•ÉÌ¥ôˆ(€€€€€€€€€€€€¤(€€€€€€€€€€€…ÍÍ•ÉÐ€‰¹•Üµµ½‘•°ˆ¥¸µ…É­•ÉÍlÁul‰½¹Ñ•¹Ð‰t((€€€€€€€€€€€€Œ9¼Ý…É¹¥¹œÍ¡½Õ±‰”ÍÕÉ™…•ƒŠPÑ¡”µ•É”ÍÕ••‘•¸(€€€€€€€€€€€½µÁ±•Ñ•}…±±Ì€ôm„™½È„¥¸•µ¥ÑÌ¥˜…lÁt€ôô€‰µ•ÍÍ…”¹½µÁ±•Ñ”‰t(€€€€€€€€€€€…ÍÍ•ÉÐ±•¸¡½µÁ±•Ñ•}…±±Ì¤€ôô€Ä(€€€€€€€€€€€|°|°Á…å±½…€ô½µÁ±•Ñ•}…±±ÍlÁt(€€€€€€€€€€€…ÍÍ•ÉÐ€‰Ý…É¹¥¹œˆ¹½Ð¥¸Á…å±½…°€ (€€€€€€€€€€€€€€€˜‰mí±…‰•±õtµ•É”Á…Ñ Í¡½Õ±¹½ÐÍÕÉ™…”„Ý…É¹¥¹œˆ(€€€€€€€€€€€€¤(€€€€€€€™¥¹…±±äè(€€€€€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}µ•É•Í}½¹}Á•ÉÍ½¹…±¥Ñå}Á¥Ù½Ñ}µ…É­•È¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰Á•ÉÍ½¹…±¥ÑäÁ¥Ù½Ð¥¹©•Ñ•µ¥µÑÕÉ¸µÕÍÐµ•É”±¥­”„µ½‘•°ÍÝ¥Ñ ¸((€€€€½Á•ÉÍ½¹…±¥Ñå€…ÁÁ±¥•Ì¥µµ•‘¥…Ñ•±äƒŠPÑ¡•É”¥Ì¹¼‘•™•ÉÉ•ÅÕ•Õ”™½È¥ÐÑ¡”(€€€Ý…äÁ•¹‘¥¹}µ½‘•±}ÍÝ¥Ñ¡€‘•™•ÉÌ„µ¥µÑÕÉ¸µ½‘•°¡…¹”ƒŠPÍ¼¡½½Í¥¹œ„(€€€Á•ÉÍ½¹…±¥ÑäÝ¡¥±”„ÑÕÉ¸¥ÌÉÕ¹¹¥¹œ‰ÕµÁÌ¡¥ÍÑ½Éå}Ù•ÉÍ¥½¹€™É½´Ñ¡”IA(€€€Ñ¡É•…¸Q¡”µ¥µÑÕÉ¸É•½¹¥±¥…Ñ¥½¸½¹±äÉ•½¹¥é•Ñ¡”µ½‘•°µÍÝ¥Ñ (€€€µ…É­•È°Í¼Ñ¡”Á¥Ù½ÐÉ•……Ì„•¹Õ¥¹”‘•Íå¹Œ…¹Ñ¡”™¥¹¥Í¡•ÑÕÉ¸Ý…Ì(€€€‘É½ÁÁ•™É½´Í•ÍÍ¥½¸¡¥ÍÑ½ÉäèÑ¡”ÕÍ•ÈÍ…ÜÑ¡”É•Á±ä…¹¥ÐÝ…Ì¹•Ù•È(€€€ÍÑ½É•€ ŒàÈÜÔØ¤¸(€€€€ˆˆˆ(€€€Í•ÍÍ¥½¹}É•˜è‘¥ÑmÍÑÈ°‘¥Ðð9½¹•t€ôì‰Ìˆè9½¹•ô((€€€±…ÍÌ}A¥Ù½Ñ•¹Ðè(€€€€€€€‘•˜ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸ (€€€€€€€€€€€Í•±˜°ÁÉ½µÁÐ°½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éäõ9½¹”°ÍÑÉ•…µ}…±±‰…¬õ9½¹”°€¨©}­Ý…ÉÌ(€€€€€€€€¤è(€€€€€€€€€€€€ŒI•…°¥¹©•Ñ¥½¸Á½¥¹Ð°µ¥µÑÕÉ¸°•á…Ñ±ä…ÌÑ¡”Á•ÉÍ½¹…±¥ÑäIA(€€€€€€€€€€€€ŒÝ½Õ±É•… ¥Ð™É½´Ñ¡”½Ñ¡•ÈÑ¡É•…¸(€€€€€€€€€€€Í•ÉÙ•È¹}…ÁÁ±å}Á•ÉÍ½¹…±¥Ñå}Ñ½}Í•ÍÍ¥½¸ (€€€€€€€€€€€€€€€€‰Í¥ˆ°Í•ÍÍ¥½¹}É•™l‰Ì‰t°€‰¹ÍÝ•ÈÑ•ÉÍ•±ä¸ˆ°€‰Ñ•ÉÍ”ˆ(€€€€€€€€€€€€¤(€€€€€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€€€€€‰™¥¹…±}É•ÍÁ½¹Í”ˆè€‰…•¹ÐÉ•Á±äˆ°(€€€€€€€€€€€€€€€€‰µ•ÍÍ…•Ìˆè±¥ÍÐ¡½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éä¤(€€€€€€€€€€€€€€€€¬l(€€€€€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰¡¤‰ô°(€€€€€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰…•¹ÐÉ•Á±ä‰ô°(€€€€€€€€€€€€€€€t°(€€€€€€€€€€€ô((€€€±…ÍÌ}%µµ•‘¥…Ñ•Q¡É•…è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©}Ñ¡É•…‘}½ÁÑ¥½¹Ì¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð€ôÑ…É•Ð((€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð ¤((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸ (€€€€€€€…•¹Ðõ}A¥Ù½Ñ•¹Ð ¤°(€€€€€€€¡¥ÍÑ½Éäõmì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰¡•±±¼‰õt°(€€€€¤(€€€Í•ÍÍ¥½¹}É•™l‰Ì‰t€ôÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t(€€€•µ¥ÑÌè±¥ÍÑmÑÕÁ±•t€ômt(€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}%µµ•‘¥…Ñ•Q¡É•…¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}ÕÍ…”ˆ°±…µ‰‘„}„èíô¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰É•¹‘•É}µ•ÍÍ…”ˆ°±…µ‰‘„}Ð°}Œè€ˆˆ¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„€©„°€¨©¬èíô¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„è•µ¥ÑÌ¹…ÁÁ•¹¡„¤¤((€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰¡¤‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÉ•ÍÀ¹•Ð •ÉÉ½Èœ¥ôˆ((€€€€€€€™¥¹…±}¡¥ÍÑ½Éä€ôÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰ul‰¡¥ÍÑ½Éä‰t((€€€€€€€…ÍÍ¥ÍÑ…¹Ñ}µÍÌ€ôl(€€€€€€€€€€€”(€€€€€€€€€€€™½È”¥¸™¥¹…±}¡¥ÍÑ½Éä(€€€€€€€€€€€¥˜¥Í¥¹ÍÑ…¹”¡”°‘¥Ð¤(€€€€€€€€€€€…¹”¹•Ð ‰É½±”ˆ¤€ôô€‰…ÍÍ¥ÍÑ…¹Ðˆ(€€€€€€€€€€€…¹”¹•Ð ‰½¹Ñ•¹Ðˆ¤€ôô€‰…•¹ÐÉ•Á±äˆ(€€€€€€€t(€€€€€€€…ÍÍ•ÉÐ±•¸¡…ÍÍ¥ÍÑ…¹Ñ}µÍÌ¤€ôô€Ä°€ (€€€€€€€€€€€€‰Ñ¡”Á•ÉÍ½¹…±¥ÑäÁ¥Ù½Ð‘¥Í…É‘•Ñ¡”™¥¹¥Í¡•ÑÕÉ¸¥¹ÍÑ•…½˜€ˆ(€€€€€€€€€€€˜‰µ•É¥¹œ¥Ð€¡½Ðí±•¸¡…ÍÍ¥ÍÑ…¹Ñ}µÍÌ¥ô…ÍÍ¥ÍÑ…¹ÐÉ•Á±¥•Ì¤ˆ(€€€€€€€€¤((€€€€€€€Á¥Ù½ÑÌ€ôl(€€€€€€€€€€€”(€€€€€€€€€€€™½È”¥¸™¥¹…±}¡¥ÍÑ½Éä(€€€€€€€€€€€¥˜¥Í¥¹ÍÑ…¹”¡”°‘¥Ð¤…¹”¹•Ð ‰‘¥ÍÁ±…å}­¥¹ˆ¤€ôô€‰Á•ÉÍ½¹…±¥Ñå}ÍÝ¥Ñ ˆ(€€€€€€€t(€€€€€€€…ÍÍ•ÉÐ±•¸¡Á¥Ù½ÑÌ¤€ôô€Ä°˜‰•áÁ•Ñ••á…Ñ±ä€ÄÁ¥Ù½Ð°½Ðí±•¸¡Á¥Ù½ÑÌ¥ôˆ((€€€€€€€½µÁ±•Ñ•}…±±Ì€ôm„™½È„¥¸•µ¥ÑÌ¥˜…lÁt€ôô€‰µ•ÍÍ…”¹½µÁ±•Ñ”‰t(€€€€€€€…ÍÍ•ÉÐ±•¸¡½µÁ±•Ñ•}…±±Ì¤€ôô€Ä(€€€€€€€|°|°Á…å±½…€ô½µÁ±•Ñ•}…±±ÍlÁt(€€€€€€€…ÍÍ•ÉÐ€‰Ý…É¹¥¹œˆ¹½Ð¥¸Á…å±½…°€‰µ•É”Á…Ñ Í¡½Õ±¹½ÐÍÕÉ™…”„Ý…É¹¥¹œˆ(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}Í…¹¥Ñ¥é•Í}‰É…­•Ñ•‘}Á…ÍÑ•}‰•™½É•}…•¹Ð¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰ÁÉ½µÁÐ¹ÍÕ‰µ¥ÐµÕÍÐÍ…¹¥Ñ¥é”½ÉÉÕÁÑ•ÕÍ•ÈÑ•áÐ‰•™½É”ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¸ˆˆˆ(€€€…ÁÑÕÉ•è‘¥ÑmÍÑÈ°ÍÑÉt€ôíô((€€€±…ÍÌ}•¹Ðè(€€€€€€€‘•˜ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°ÁÉ½µÁÐ°½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éäõ9½¹”°ÍÑÉ•…µ}…±±‰…¬õ9½¹”°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€…ÁÑÕÉ•‘l‰ÁÉ½µÁÐ‰t€ôÁÉ½µÁÐ(€€€€€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€€€€€‰™¥¹…±}É•ÍÁ½¹Í”ˆè€‰½¬ˆ°(€€€€€€€€€€€€€€€€‰µ•ÍÍ…•Ìˆèmì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰½¬‰õt°(€€€€€€€€€€€ô((€€€±…ÍÌ}%µµ•‘¥…Ñ•Q¡É•…è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©­Ü¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð€ôÑ…É•Ð((€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð ¤((€€€½ÉÉÕÁÑ•€ô€‰¡•±±½lˆ€¬€‰ùmm”ˆ€¨€à(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸¡…•¹Ðõ}•¹Ð ¤¤(€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}%µµ•‘¥…Ñ•Q¡É•…¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}ÕÍ…”ˆ°±…µ‰‘„}„èíô¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰É•¹‘•É}µ•ÍÍ…”ˆ°±…µ‰‘„}Ð°}Œè€ˆˆ¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•¹ÍÕÉ•}Í•ÍÍ¥½¹}‘‰}É½Üˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Á•ÉÍ¥ÍÑ}‰É…¹¡}Í••ˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤((€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè½ÉÉÕÁÑ•‘ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÉ•ÍÀ¹•Ð •ÉÉ½Èœ¥ôˆ(€€€€€€€…ÍÍ•ÉÐ…ÁÑÕÉ•‘l‰ÁÉ½µÁÐ‰t€ôô€‰¡•±±¼ˆ(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}¡¥ÍÑ½Éå}Ù•ÉÍ¥½¹}µ…Ñ¡}Á•ÉÍ¥ÍÑÍ}¹½Éµ…±±ä¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰I•É•ÍÍ¥½¸Õ…ÉèÑ¡”‰…­ÍÑ½À‘½•Ì¹½Ð…™™•ÐÑ¡”¡…ÁÁäÁ…Ñ ¸ˆˆˆ((€€€±…ÍÌ}•¹Ðè(€€€€€€€‘•˜ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°ÁÉ½µÁÐ°½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éäõ9½¹”°ÍÑÉ•…µ}…±±‰…¬õ9½¹”°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€€€€€‰™¥¹…±}É•ÍÁ½¹Í”ˆè€‰É•Á±äˆ°(€€€€€€€€€€€€€€€€‰µ•ÍÍ…•Ìˆèmì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä‰õt°(€€€€€€€€€€€ô((€€€±…ÍÌ}%µµ•‘¥…Ñ•Q¡É•…è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©}Ñ¡É•…‘}½ÁÑ¥½¹Ì¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð€ôÑ…É•Ð((€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð ¤((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸¡…•¹Ðõ}•¹Ð ¤¤(€€€•µ¥ÑÌè±¥ÍÑmÑÕÁ±•t€ômt(€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}%µµ•‘¥…Ñ•Q¡É•…¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}ÕÍ…”ˆ°±…µ‰‘„}„èíô¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰É•¹‘•É}µ•ÍÍ…”ˆ°±…µ‰‘„}Ð°}Œè€ˆˆ¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„è•µ¥ÑÌ¹…ÁÁ•¹¡„¤¤((€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰¡¤‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰É•ÍÕ±Ðˆ¤((€€€€€€€€Œ!¥ÍÑ½ÉäÝ…ÌÝÉ¥ÑÑ•¸(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰ul‰¡¥ÍÑ½Éä‰t€ôôl(€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä‰ô(€€€€€€€t(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰ul‰¡¥ÍÑ½Éå}Ù•ÉÍ¥½¸‰t€ôô€Ä((€€€€€€€€Œ9¼Ý…É¹¥¹œÍ¡½Õ±‰”…ÑÑ…¡•(€€€€€€€½µÁ±•Ñ•}…±±Ì€ôm„™½È„¥¸•µ¥ÑÌ¥˜…lÁt€ôô€‰µ•ÍÍ…”¹½µÁ±•Ñ”‰t(€€€€€€€…ÍÍ•ÉÐ±•¸¡½µÁ±•Ñ•}…±±Ì¤€ôô€Ä(€€€€€€€|°|°Á…å±½…€ô½µÁ±•Ñ•}…±±ÍlÁt(€€€€€€€…ÍÍ•ÉÐ€‰Ý…É¹¥¹œˆ¹½Ð¥¸Á…å±½…(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}Í¹…ÁÍ¡½ÑÍ}¡¥ÍÑ½Éå}…™Ñ•É}Á•¹‘¥¹}µ½‘•±}ÍÝ¥Ñ ¡µ½¹­•åÁ…Ñ ¤è(€€€µ…É­•È€ôì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰mµ½‘•°ÍÝ¥Ñ¡•‘t‰ô(€€€Í••¸€ôíô((€€€±…ÍÌ}•¹Ðè(€€€€€€€‘•˜ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°ÁÉ½µÁÐ°½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éäõ9½¹”°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€Í••¹l‰¡¥ÍÑ½Éä‰t€ô½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éä(€€€€€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€€€€€‰™¥¹…±}É•ÍÁ½¹Í”ˆè€‰É•Á±äˆ°(€€€€€€€€€€€€€€€€‰µ•ÍÍ…•Ìˆèl(€€€€€€€€€€€€€€€€€€€€¨¡½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éä½Èmt¤°(€€€€€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹ÐˆèÁÉ½µÁÑô°(€€€€€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä‰ô°(€€€€€€€€€€€€€€€t°(€€€€€€€€€€€ô((€€€±…ÍÌ}%µµ•‘¥…Ñ•Q¡É•…è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð€ôÑ…É•Ð((€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð ¤((€€€‘•˜}…ÁÁ±å}Á•¹‘¥¹œ¡}Í¥°Í•ÍÍ¥½¸¤è(€€€€€€€Ý¥Ñ Í•ÍÍ¥½¹l‰¡¥ÍÑ½Éå}±½¬‰tè(€€€€€€€€€€€Í•ÍÍ¥½¹l‰¡¥ÍÑ½Éä‰t¹…ÁÁ•¹¡µ…É­•È¤(€€€€€€€€€€€Í•ÍÍ¥½¹l‰¡¥ÍÑ½Éå}Ù•ÉÍ¥½¸‰t€¬ô€Ä((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸¡…•¹Ðõ}•¹Ð ¤¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰ul‰Á•¹‘¥¹}µ½‘•±}ÍÝ¥Ñ ‰t€ôì‰É…Üˆè€‰¹•Üµµ½‘•°‰ô(€€€•µ¥ÑÌ€ômt(€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}%µµ•‘¥…Ñ•Q¡É•…¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}…ÁÁ±å}Á•¹‘¥¹}µ½‘•±}ÍÝ¥Ñ ˆ°}…ÁÁ±å}Á•¹‘¥¹œ¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Íå¹}…•¹Ñ}µ½‘•±}Ý¥Ñ¡}½¹™¥œˆ°±…µ‰‘„€©}„è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}ÕÍ…”ˆ°±…µ‰‘„}„èíô¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰É•¹‘•É}µ•ÍÍ…”ˆ°±…µ‰‘„€©}„è€ˆˆ¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„è•µ¥ÑÌ¹…ÁÁ•¹¡„¤¤((€€€€€€€Í•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰¡¤‰õô(€€€€€€€€¤((€€€€€€€…ÍÍ•ÉÐÍ••¹l‰¡¥ÍÑ½Éä‰t€ôômµ…É­•Ét(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰ul‰¡¥ÍÑ½Éä‰ul´Åt€ôôì(€€€€€€€€€€€€‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±äˆ(€€€€€€€ô(€€€€€€€½µÁ±•Ñ”€ôm„™½È„¥¸•µ¥ÑÌ¥˜…lÁt€ôô€‰µ•ÍÍ…”¹½µÁ±•Ñ”‰t(€€€€€€€…ÍÍ•ÉÐ€‰Ý…É¹¥¹œˆ¹½Ð¥¸½µÁ±•Ñ•lÁulÉt(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}…¹}ÑÉÕ¹…Ñ•}‰•™½É•}ÕÍ•É}½É‘¥¹…°¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰•Í­Ñ½ÀÕÍ•Èµµ•ÍÍ…”•‘¥ÑÌÍ¡½Õ±É•ÍÑ…ÉÐÑ¡”ÑÕÉ¸™É½´Ñ¡”•‘¥Ñ•ÕÍ•È¸ˆˆˆ((€€€Í••¸€ôíô((€€€±…ÍÌ}•¹Ðè(€€€€€€€‘•˜ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°ÁÉ½µÁÐ°½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éäõ9½¹”°ÍÑÉ•…µ}…±±‰…¬õ9½¹”°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€Í••¹l‰ÁÉ½µÁÐ‰t€ôÁÉ½µÁÐ(€€€€€€€€€€€Í••¹l‰¡¥ÍÑ½Éä‰t€ô½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éä(€€€€€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€€€€€‰™¥¹…±}É•ÍÁ½¹Í”ˆè€‰•‘¥Ñ•É•Á±äˆ°(€€€€€€€€€€€€€€€€‰µ•ÍÍ…•Ìˆèl(€€€€€€€€€€€€€€€€€€€€¨¡½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éä½Èmt¤°(€€€€€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹ÐˆèÁÉ½µÁÑô°(€€€€€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰•‘¥Ñ•É•Á±ä‰ô°(€€€€€€€€€€€€€€€t°(€€€€€€€€€€€ô((€€€±…ÍÌ}%µµ•‘¥…Ñ•Q¡É•…è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©}Ñ¡É•…‘}½ÁÑ¥½¹Ì¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð€ôÑ…É•Ð((€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð ¤((€€€½É¥¥¹…±}¡¥ÍÑ½Éä€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐ‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐÉ•Á±ä‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹É•Á±ä‰ô°(€€€t(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸¡…•¹Ðõ}•¹Ð ¤°¡¥ÍÑ½Éäõ½É¥¥¹…±}¡¥ÍÑ½Éä¤((€€€±…ÍÌ}MÑÕ‰ˆè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹É•Á±…•€ômt((€€€€€€€‘•˜•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°€©}…ÉÌ°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€É•ÑÕÉ¸mt((€€€€€€€‘•˜É•Á±…•}µ•ÍÍ…•Ì (€€€€€€€€€€€Í•±˜°(€€€€€€€€€€€Í•ÍÍ¥½¹}¥°(€€€€€€€€€€€µ•ÍÍ…•Ì°(€€€€€€€€€€€…Ñ¥Ù•}½¹±äõ…±Í”°(€€€€€€€€€€€…É¡¥Ù•}‘É½ÁÁ•õ…±Í”°(€€€€€€€€€€€É•©•Ñ}…Ñ¥Ù•}ÑÕÉ¹}±•…Í”õ…±Í”°(€€€€€€€€¤è(€€€€€€€€€€€Í•±˜¹É•Á±…•¹…ÁÁ•¹ ¡Í•ÍÍ¥½¹}¥°±¥ÍÐ¡µ•ÍÍ…•Ì¤¤¤((€€€ÍÑÕ‰}‘ˆ€ô}MÑÕ‰ˆ ¤((€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}%µµ•‘¥…Ñ•Q¡É•…¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}ÕÍ…”ˆ°±…µ‰‘„}„èíô¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰É•¹‘•É}µ•ÍÍ…”ˆ°±…µ‰‘„}Ð°}Œè€ˆˆ¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„èÍÑÕ‰}‘ˆ¤((€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰•‘¥Ñ•Í•½¹ˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}ÕÍ•É}½É‘¥¹…°ˆè€Ä°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÉ•ÍÀ¹•Ð •ÉÉ½Èœ¥ôˆ((€€€€€€€…ÍÍ•ÉÐÍ••¹l‰ÁÉ½µÁÐ‰t€ôô€‰•‘¥Ñ•Í•½¹ˆ(€€€€€€€…ÍÍ•ÉÐÍ••¹l‰¡¥ÍÑ½Éä‰t€ôô½É¥¥¹…±}¡¥ÍÑ½ÉålèÉt(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰ul‰¡¥ÍÑ½Éä‰t€ôôl(€€€€€€€€€€€€©½É¥¥¹…±}¡¥ÍÑ½ÉålèÉt°(€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰•‘¥Ñ•Í•½¹‰ô°(€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰•‘¥Ñ•É•Á±ä‰ô°(€€€€€€€t(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰ul‰¡¥ÍÑ½Éå}Ù•ÉÍ¥½¸‰t€ôô€È(€€€€€€€…ÍÍ•ÉÐÍÑÕ‰}‘ˆ¹É•Á±…•€ôôl ‰Í•ÍÍ¥½¸µ­•äˆ°½É¥¥¹…±}¡¥ÍÑ½ÉålèÉt¥t(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}ÑÉÕ¹…Ñ¥½¹}Í¥¹…±Í}‰ÕÍå}¥¹ÍÑ•…‘}½™}ÅÕ•Õ•¥¹œ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆˆŒÄÄÌäÐÈè•‘¥Ñ¥¹œ„µ•ÍÍ…”Ý¡¥±”Ñ¡”ÑÕÉ¸¥Ð‰•±½¹ÌÑ¼¥ÌÍÑ¥±°ÉÕ¹¹¥¹œ((€€€µÕÍÐ¹½Ð‰”Í¥±•¹Ñ±ä…‰Í½É‰•…Ì„ÍÑ••É•½ÉÉ•Ñ¥½¸½È„Á±…¥¸™½±±½ÜµÕÀ(€€€ÅÕ•Õ•Ñ¼ÉÕ¸QHÑ¡”±¥Ù”ÑÕÉ¸ƒŠP‰½Ñ Ý½Õ±‘É½ÀÑ¡”ÑÉÕ¹…Ñ¥½¸…¹(€€€±•…Ù”Ñ¡”½É¥¥¹…°€¡Õ¸µ•‘¥Ñ•¤ÑÕÉ¸ÌÉ•Á±ä¥¹Ñ…Ð°Ý¡¥ É•…‘ÌÑ¼Ñ¡”(€€€ÕÍ•È…Ì€‰µä•‘¥ÐÝ…ÌÉ•©•Ñ•ˆ¸%ÐµÕÍÐÍÕÉ™…”…ÌÑ¡”Í…µ”€‰Í•ÍÍ¥½¸‰ÕÍäˆ(€€€€ ÐÀÀä¤Ñ¡”‘•Í­Ñ½À±¥•¹ÐÌ•‘¥ÐÁ…Ñ …±É•…‘äÉ•ÑÉ¥•ÌÑ¡É½Õ €¡¥¹Ñ•ÉÉÕÁÐ°(€€€Ñ¡•¸Á½±°¤Õ¹Ñ¥°Ñ¡”ÑÕÉ¸…ÑÕ…±±ä±•…ÉÌ¸(€€€€ˆˆˆ(€€€…•¹Ð€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” ¤€€Œ¹¼É•‘¥É•Ð½ÍÑ••ÈÍÕÁÁ½ÉÐ€´øÝ½Õ±™…±°Ñ¼ÅÕ•Õ”­¥¹Ñ•ÉÉÕÁÐ(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸¡…•¹Ðõ…•¹Ð°ÉÕ¹¹¥¹œõQÉÕ”¤((€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰•‘¥Ñ•Ý¡¥±”Ñ¡¥¹­¥¹œˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}ÕÍ•É}½É‘¥¹…°ˆè€À°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}•µÁÑå}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤((€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰•ÉÉ½Èˆ°íô¤¹•Ð ‰½‘”ˆ¤€ôô€ÐÀÀä(€€€€€€€€ŒQ¡”•‘¥ÐµÕÍÐ¹½Ð¡…Ù”‰••¸Í¥±•¹Ñ±ä…•ÁÑ•…Ì„ÅÕ•Õ•™½±±½ÜµÕÀ¸(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t¹•Ð ‰ÅÕ•Õ•‘}ÁÉ½µÁÐˆ¤¥Ì9½¹”(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰ul‰ÉÕ¹¹¥¹œ‰t¥ÌQÉÕ”(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}ÑÉÕ¹…Ñ¥½¹}É•™ÕÍ•Í}É•‘¥É•Ñ}½™}±¥Ù•}ÑÕÉ¸ ¤è(€€€€ˆˆ‰Q¡”É•‘¥É•Ðµ…Á…‰±”…•¹Ð€¡Ñ¡”•Í­Ñ½ÀÌ¹½Éµ…°…Í”¤¥ÌÑ¡”‰É…¹ Ñ¡…ÐÍ¥±•¹Ñ±ä(€€€…‰Í½É‰•Ñ¡”•‘¥Ðè‰ÕÍå}¥¹ÁÕÑ}µ½‘”õ¥¹Ñ•ÉÉÕÁÑ€ÑÕÉ¹•Ñ¡”É•Ý¥¹¥¹Ñ¼„µ¥µÑÕÉ¸(€€€É•‘¥É•Ð…¹±•™ÐÑ¡”Õ¸µ•‘¥Ñ•ÑÉ…¹ÍÉ¥ÁÐ¥¸Á±…”¸ˆˆˆ(€€€É•‘¥É•Ñ•€ômt(€€€…•¹Ð€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€}ÍÕÁÁ½ÉÑÍ}…Ñ¥Ù•}ÑÕÉ¹}É•‘¥É•ÐõQÉÕ”°É•‘¥É•Ðõ±…µ‰‘„Ñ•áÐèÉ•‘¥É•Ñ•¹…ÁÁ•¹¡Ñ•áÐ¤½ÈQÉÕ”¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸¡…•¹Ðõ…•¹Ð°ÉÕ¹¹¥¹œõQÉÕ”°¡¥ÍÑ½Éäõmì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰½É¥¥¹…°‰õt¤((€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ¡ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰•‘¥Ñ•Ý¡¥±”Ñ¡¥¹­¥¹œˆ°€‰ÑÉÕ¹…Ñ•}‰•™½É•}ÕÍ•É}½É‘¥¹…°ˆè€À°(€€€€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°€‰½¹™¥Éµ}•µÁÑå}ÑÉÕ¹…Ñ”ˆèQÉÕ•õô¤((€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰•ÉÉ½Èˆ°íô¤¹•Ð ‰½‘”ˆ¤€ôô€ÐÀÀä(€€€€€€€…ÍÍ•ÉÐÉ•‘¥É•Ñ•€ôômt(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰ul‰¡¥ÍÑ½Éä‰t€ôômì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰½É¥¥¹…°‰õt(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}É•™ÕÍ•Í}ÑÕÉ¹}Ý¡•¹}ÑÉÕ¹…Ñ•}Á•ÉÍ¥ÍÑ}™…¥±Ì¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰%˜É•Á±…•}µ•ÍÍ…•Ì™…¥±Ì‘ÕÉ¥¹œ•‘¥Ð½É••¹•É…Ñ”ÑÉÕ¹…Ñ”°‘¼¹½ÐÉÕ¸Ñ¡”ÑÕÉ¸¸((€€€5•µ½Éäµ™¥ÉÍÐ€¬™…¥°µ½Á•¸±•™ÐÍ•ÍÍ¥½¹l¡¥ÍÑ½ÉätÍ¡½ÉÐÝ¡¥±”ÍÑ…Ñ”¹‘ˆ­•ÁÐ(€€€Ñ¡”½±Ñ…¥°¸Q¡”…•¹Ð™±ÕÍ Ñ¡•¸…ÁÁ•¹‘ÌÑ¡”¹•Ü•á¡…¹”½¸Ñ½À½˜Ñ¡”(€€€€Õ¹‘½¹”œÑÕÉ¹ÌƒŠP‘ÕÉ…‰±”é½µ‰¥”¡¥ÍÑ½Éä¸]É¥Ñ”™¥ÉÍÐì½¸™…¥±ÕÉ”±•…Ù”(€€€µ•µ½Éä…¹Õ¹¡…¹•…¹É•ÑÕÉ¸€ÔÀÀà¸(€€€€ˆˆˆ(€€€½É¥¥¹…±}¡¥ÍÑ½Éä€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐ‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐÉ•Á±ä‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹É•Á±ä‰ô°(€€€t(€€€Í•ÍÌ€ô}Í•ÍÍ¥½¸¡¡¥ÍÑ½Éäõ±¥ÍÐ¡½É¥¥¹…±}¡¥ÍÑ½Éä¤¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰ÑÉÕ¹Œµ™…¥°µÍ¥‰t€ôÍ•ÍÌ((€€€±…ÍÌ}…¥±ˆè(€€€€€€€‘•˜•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°€©}…ÉÌ°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€É•ÑÕÉ¸mt((€€€€€€€‘•˜É•Á±…•}µ•ÍÍ…•Ì (€€€€€€€€€€€Í•±˜°(€€€€€€€€€€€Í•ÍÍ¥½¹}¥°(€€€€€€€€€€€µ•ÍÍ…•Ì°(€€€€€€€€€€€…Ñ¥Ù•}½¹±äõ…±Í”°(€€€€€€€€€€€…É¡¥Ù•}‘É½ÁÁ•õ…±Í”°(€€€€€€€€€€€É•©•Ñ}…Ñ¥Ù•}ÑÕÉ¹}±•…Í”õ…±Í”°(€€€€€€€€¤è(€€€€€€€€€€€É…¥Í”=MÉÉ½È ‰‘¥Í¬™Õ±°ˆ¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è}…¥±ˆ ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}¥¹™±¥¡Ñ}ÑÕÉ¸ˆ°±…µ‰‘„€©„°€¨©¬èÁåÑ•ÍÐ¹™…¥° ‰µÕÍÐ¹½ÐÍÑ…ÉÐ„ÑÕÉ¸ˆ¤(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„€©„°€¨©¬èÁåÑ•ÍÐ¹™…¥° ‰µÕÍÐ¹½ÐÍÑ…ÉÐ„ÑÕÉ¸ˆ¤(€€€€¤((€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰ÑÉÕ¹Œµ™…¥°µÍ¥ˆ°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰•‘¥Ñ•Í•½¹ˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}ÕÍ•É}½É‘¥¹…°ˆè€Ä°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¥¸É•ÍÀ(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÔÀÀà(€€€€€€€€Œ5•µ½Éä±•™Ð¥¹Ñ…ÐƒŠPÍ…µ”±¥ÍÐ½¹Ñ•¹ÑÌ…Ì‰•™½É”Ñ¡”É•™ÕÍ•ÕÐ¸(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍl‰¡¥ÍÑ½Éä‰t€ôô½É¥¥¹…±}¡¥ÍÑ½Éä(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍl‰¡¥ÍÑ½Éå}Ù•ÉÍ¥½¸‰t€ôô€À(€€€€€€€…ÍÍ•ÉÐÍ•ÍÌ¹•Ð ‰ÉÕ¹¹¥¹œˆ¤¥Ì¹½ÐQÉÕ”(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰ÑÉÕ¹Œµ™…¥°µÍ¥ˆ°9½¹”¤(((Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(ŒÍ•ÍÍ¥½¸¹¥¹Ñ•ÉÉÕÁÐµÕÍÐ½¹±ä…¹•°Á•¹‘¥¹œÁÉ½µÁÑÌ½Ý¹•‰äÑ¡”…±±¥¹œ(ŒÍ•ÍÍ¥½¸ƒŠP¥ÐµÕÍÐ¹½Ð‰±…ÍÐµÉ•Í½±Ù”±…É¥™ä½ÍÕ‘¼½Í•É•ÐÁÉ½µÁÑÌ½¸(ŒÕ¹É•±…Ñ•Í•ÍÍ¥½¹ÌÍ¡…É¥¹œÑ¡”Í…µ”ÑÕ¥}…Ñ•Ý…äÁÉ½•ÍÌ¸€]¥Ñ¡½ÕÐ(ŒÍ•ÍÍ¥½¸Í½Á¥¹œÑ¡”½Ñ¡•ÈÍ•ÍÍ¥½¹ÌœÁÉ½µÁÑÌÍ¥±•¹Ñ±äÉ•Í½±Ù”Ñ¼•µÁÑä(ŒÍÑÉ¥¹Ì°Õ¹‰±½­¥¹œÑ¡•¥È…•¹ÐÑ¡É•…‘Ì…Ì¥˜Ñ¡”ÕÍ•È…¹•±±•¸(Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}ÑÉÕ¹…Ñ•}½É‘¥¹…±}Í­¥ÁÍ}‘¥ÍÁ±…å}­¥¹‘}É½ÝÌ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰ÑÉÕ¹…Ñ•}‰•™½É•}ÕÍ•É}½É‘¥¹…°µÕÍÐ½Õ¹Ð½¹±äÉ•…°ÕÍ•ÈÑÕÉ¹Ì¸((€€€‘¥ÍÁ±…å}­¥¹Ñ¥µ•±¥¹”É½ÝÌ€¡µ½‘•±}ÍÝ¥Ñ °…Íå¹}‘•±•…Ñ¥½¹}½µÁ±•Ñ”°ƒŠ˜¤(€€€…É”É½±”õÕÍ•È‰ÕÐ¹¼±¥•¹Ð½Õ¹ÑÌÑ¡•´…ÌÕÍ•ÈÑÕÉ¹Ì¸]¥Ñ¡½ÕÐÑ¡”(€€€™¥±Ñ•È°„ÑÉ…¥±¥¹œµ…É­•ÈÍ¡¥™ÑÌÑ¡”½É‘¥¹…°Í¼Ñ¡”ÝÉ½¹œµ•ÍÍ…”¥Ì(€€€Ñ…É•Ñ•™½ÈÑÉÕ¹…Ñ¥½¸¸(€€€€ˆˆˆ((€€€Í••¸€ôíô((€€€±…ÍÌ}•¹Ðè(€€€€€€€‘•˜ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°ÁÉ½µÁÐ°½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éäõ9½¹”°ÍÑÉ•…µ}…±±‰…¬õ9½¹”°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€Í••¹l‰ÁÉ½µÁÐ‰t€ôÁÉ½µÁÐ(€€€€€€€€€€€Í••¹l‰¡¥ÍÑ½Éä‰t€ô½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éä(€€€€€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€€€€€‰™¥¹…±}É•ÍÁ½¹Í”ˆè€‰É•Á±äˆ°(€€€€€€€€€€€€€€€€‰µ•ÍÍ…•Ìˆèl(€€€€€€€€€€€€€€€€€€€€¨¡½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éä½Èmt¤°(€€€€€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹ÐˆèÁÉ½µÁÑô°(€€€€€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä‰ô°(€€€€€€€€€€€€€€€t°(€€€€€€€€€€€ô((€€€±…ÍÌ}%µµ•‘¥…Ñ•Q¡É•…è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©}Ñ¡É•…‘}½ÁÑ¥½¹Ì¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð€ôÑ…É•Ð((€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð ¤((€€€½É¥¥¹…±}¡¥ÍÑ½Éä€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐ‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐÉ•Á±ä‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹É•Á±ä‰ô°(€€€€€€€ì(€€€€€€€€€€€€‰É½±”ˆè€‰ÕÍ•Èˆ°(€€€€€€€€€€€€‰½¹Ñ•¹Ðˆè€‰‰…­É½Õ¹…•¹Ð™¥¹¥Í¡•ˆ°(€€€€€€€€€€€€‰‘¥ÍÁ±…å}­¥¹ˆè€‰…Íå¹}‘•±•…Ñ¥½¹}½µÁ±•Ñ”ˆ°(€€€€€€€ô°(€€€t(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸¡…•¹Ðõ}•¹Ð ¤°¡¥ÍÑ½Éäõ½É¥¥¹…±}¡¥ÍÑ½Éä¤((€€€±…ÍÌ}MÑÕ‰ˆè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹É•Á±…•€ômt((€€€€€€€‘•˜•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°€©}…ÉÌ°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€É•ÑÕÉ¸mt((€€€€€€€‘•˜É•Á±…•}µ•ÍÍ…•Ì (€€€€€€€€€€€Í•±˜°(€€€€€€€€€€€Í•ÍÍ¥½¹}¥°(€€€€€€€€€€€µ•ÍÍ…•Ì°(€€€€€€€€€€€…Ñ¥Ù•}½¹±äõ…±Í”°(€€€€€€€€€€€…É¡¥Ù•}‘É½ÁÁ•õ…±Í”°(€€€€€€€€€€€É•©•Ñ}…Ñ¥Ù•}ÑÕÉ¹}±•…Í”õ…±Í”°(€€€€€€€€¤è(€€€€€€€€€€€Í•±˜¹É•Á±…•¹…ÁÁ•¹ ¡Í•ÍÍ¥½¹}¥°±¥ÍÐ¡µ•ÍÍ…•Ì¤¤¤((€€€ÍÑÕ‰}‘ˆ€ô}MÑÕ‰ˆ ¤((€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}%µµ•‘¥…Ñ•Q¡É•…¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}ÕÍ…”ˆ°±…µ‰‘„}„èíô¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰É•¹‘•É}µ•ÍÍ…”ˆ°±…µ‰‘„}Ð°}Œè€ˆˆ¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„èÍÑÕ‰}‘ˆ¤((€€€€€€€€Œ½É‘¥¹…°ôÄµ•…¹Ì€‰ÑÉÕ¹…Ñ”‰•™½É”Ñ¡”€É¹µ™É½´µ±…ÍÐÉ•…°ÕÍ•ÈÑÕÉ¸ˆ(€€€€€€€€ŒÝ¡¥ ¥Ì€‰™¥ÉÍÐˆ¸Q¡”‘¥ÍÁ±…å}­¥¹µ…É­•ÈµÕÍÐ9=PÍ¡¥™ÐÑ¡”½É‘¥¹…°¸(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰•‘¥Ñ•™¥ÉÍÐˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}ÕÍ•É}½É‘¥¹…°ˆè€Ä°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÉ•ÍÀ¹•Ð •ÉÉ½Èœ¥ôˆ((€€€€€€€€Œ]¥Ñ ‘¥ÍÁ±…å}­¥¹™¥±Ñ•ÈèÕÍ•É}¥¹‘¥•Ì€ôlÀ°€Ét€¡¥¹‘¥•Ì½˜€‰™¥ÉÍÐˆ…¹€‰Í•½¹ˆ¤¸(€€€€€€€€Œ½É‘¥¹…°ôÄƒŠHÕÍ•É}¥¹‘¥•ÍlÅt€ô€È°ÑÉÕ¹…Ñ•€ô¡¥ÍÑ½ÉålèÉt€ôm™¥ÉÍÐ°™¥ÉÍÐÉ•Á±åt¸(€€€€€€€€Œ]¥Ñ¡½ÕÐÑ¡”™¥±Ñ•ÈèÕÍ•É}¥¹‘¥•Ì€ôlÀ°€È°€Ñt€¡¥¹±Õ‘•ÌÑ¡”µ…É­•È¤°(€€€€€€€€Œ½É‘¥¹…°ôÄƒŠHÕÍ•É}¥¹‘¥•ÍlÅt€ô€È°Í…µ”É•ÍÕ±Ð‰ä±Õ¬ƒŠP‰ÕÐ½É‘¥¹…°ôÀ(€€€€€€€€ŒÝ½Õ±ÑÉÕ¹…Ñ”Ñ¼¡¥ÍÑ½ÉålèÁtÙÌ¡¥ÍÑ½ÉålèÁt°…¹¡¥¡•È½É‘¥¹…±ÌÍ¡¥™Ð¸(€€€€€€€…ÍÍ•ÉÐÍ••¹l‰¡¥ÍÑ½Éä‰t€ôô½É¥¥¹…±}¡¥ÍÑ½ÉålèÉt°€ (€€€€€€€€€€€˜‰áÁ•Ñ•ÑÉÕ¹…Ñ¥½¸Ñ¼™¥ÉÍÐ€Èµ•ÍÍ…•Ì°½ÐíÍ••¹l¡¥ÍÑ½Éäuôˆ(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÍÑÕ‰}‘ˆ¹É•Á±…•€ôôl ‰Í•ÍÍ¥½¸µ­•äˆ°½É¥¥¹…±}¡¥ÍÑ½ÉålèÉt¥t°€ (€€€€€€€€€€€˜‰áÁ•Ñ•É•Á±…”Ý¥Ñ ™¥ÉÍÐ€Èµ•ÍÍ…•Ì°½ÐíÍÑÕ‰}‘ˆ¹É•Á±…•‘ôˆ(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}ÑÉÕ¹…Ñ•}ÑÉ…¹Í±…Ñ•Í}‘¥ÍÁ±…å}ÁÉ•™¥á}½É‘¥¹…°¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰Õ±°µ±¥¹•…”•Í­Ñ½À½É‘¥¹…±ÌµÕÍÐÑÉÕ¹…Ñ”Ñ¡”Ñ¥ÀÍ•µ•¹Ð€ ŒàÈÐØÈ¤¸((€€€™Ñ•È½µÁÉ•ÍÍ¥½¸°Í•ÍÍ¥½¹l‰¡¥ÍÑ½Éä‰t¥ÌÑ¡”Ñ¥ÀÝ¡¥±”‘¥ÍÁ±…å}¡¥ÍÑ½Éå}ÁÉ•™¥à(€€€ÍÑ¥±°¡½±‘Ì…¹•ÍÑ½ÈÕÍ•ÈÑÕÉ¹ÌÑ¡”U$Í¡½ÝÌ¸±¥•¹Ð½É‘¥¹…°Ñ¡…Ð¥¹±Õ‘•Ì(€€€Ñ¡½Í”…¹•ÍÑ½ÉÌµÕÍÐµ…À¥¹Ñ¼Ñ¡”Ñ¥À°¹½Ð€ÐÀÄà¸(€€€€ˆˆˆ((€€€Í••¸€ôíô((€€€±…ÍÌ}•¹Ðè(€€€€€€€‘•˜ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°ÁÉ½µÁÐ°½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éäõ9½¹”°ÍÑÉ•…µ}…±±‰…¬õ9½¹”°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€Í••¹l‰ÁÉ½µÁÐ‰t€ôÁÉ½µÁÐ(€€€€€€€€€€€Í••¹l‰¡¥ÍÑ½Éä‰t€ô½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éä(€€€€€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€€€€€‰™¥¹…±}É•ÍÁ½¹Í”ˆè€‰•‘¥Ñ•É•Á±äˆ°(€€€€€€€€€€€€€€€€‰µ•ÍÍ…•Ìˆèl(€€€€€€€€€€€€€€€€€€€€¨¡½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éä½Èmt¤°(€€€€€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹ÐˆèÁÉ½µÁÑô°(€€€€€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰•‘¥Ñ•É•Á±ä‰ô°(€€€€€€€€€€€€€€€t°(€€€€€€€€€€€ô((€€€±…ÍÌ}%µµ•‘¥…Ñ•Q¡É•…è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©}Ñ¡É•…‘}½ÁÑ¥½¹Ì¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð€ôÑ…É•Ð((€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð ¤((€€€Ñ¥Á}¡¥ÍÑ½Éä€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Á½ÍÐµ½µÁÉ•ÍÌ‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Á½ÍÐµ½µÁÉ•ÍÌ‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä‰ô°(€€€t(€€€‘¥ÍÁ±…å}ÁÉ•™¥à€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰ÁÉ”µ½µÁÉ•ÍÌ€Ä‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰ÁÉ”É•Á±ä€Ä‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰ÁÉ”µ½µÁÉ•ÍÌ€È‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰ÁÉ”É•Á±ä€È‰ô°(€€€t(€€€€Œ•Í­Ñ½À±¥¹•…”èÁÉ”ÄôÀ°ÁÉ”ÈôÄ°Á½ÍÑôÈ°Á½ÍÑôÌ(€€€‘•Í­Ñ½Á}½É‘¥¹…±}™½É}Á½ÍÑ}ˆ€ô€Ì((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸ (€€€€€€€…•¹Ðõ}•¹Ð ¤°(€€€€€€€¡¥ÍÑ½ÉäõÑ¥Á}¡¥ÍÑ½Éä°(€€€€€€€‘¥ÍÁ±…å}¡¥ÍÑ½Éå}ÁÉ•™¥àõ‘¥ÍÁ±…å}ÁÉ•™¥à°(€€€€¤((€€€±…ÍÌ}MÑÕ‰ˆè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹É•Á±…•€ômt((€€€€€€€‘•˜•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°€©}…ÉÌ°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€€ŒµÁÑä‘ÕÉ…‰±”ÑÉ…¹ÍÉ¥ÁÐèÁÉ½Ù•ÌÑ¡¥Ì•Á¡•µ•É…°µÍÑå±”Í•ÍÍ¥½¸(€€€€€€€€€€€€Œµ…äÑ…­”Ñ¡”½É‘¥¹…°µ½¹±äÁ…Ñ Á…ÍÐÑ¡”‘ÕÉ…‰¥±¥Ñä…Ñ”¸(€€€€€€€€€€€É•ÑÕÉ¸mt((€€€€€€€‘•˜É•Á±…•}µ•ÍÍ…•Ì (€€€€€€€€€€€Í•±˜°(€€€€€€€€€€€Í•ÍÍ¥½¹}¥°(€€€€€€€€€€€µ•ÍÍ…•Ì°(€€€€€€€€€€€…Ñ¥Ù•}½¹±äõ…±Í”°(€€€€€€€€€€€…É¡¥Ù•}‘É½ÁÁ•õ…±Í”°(€€€€€€€€€€€É•©•Ñ}…Ñ¥Ù•}ÑÕÉ¹}±•…Í”õ…±Í”°(€€€€€€€€¤è(€€€€€€€€€€€…ÍÍ•ÉÐÉ•©•Ñ}…Ñ¥Ù•}ÑÕÉ¹}±•…Í”¥ÌQÉÕ”(€€€€€€€€€€€Í•±˜¹É•Á±…•¹…ÁÁ•¹ ¡Í•ÍÍ¥½¹}¥°±¥ÍÐ¡µ•ÍÍ…•Ì¤¤¤((€€€ÍÑÕ‰}‘ˆ€ô}MÑÕ‰ˆ ¤((€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}%µµ•‘¥…Ñ•Q¡É•…¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}ÕÍ…”ˆ°±…µ‰‘„}„èíô¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰É•¹‘•É}µ•ÍÍ…”ˆ°±…µ‰‘„}Ð°}Œè€ˆˆ¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„èÍÑÕ‰}‘ˆ¤((€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰•‘¥Ñ•Á½ÍÐˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}ÕÍ•É}½É‘¥¹…°ˆè‘•Í­Ñ½Á}½É‘¥¹…±}™½É}Á½ÍÑ}ˆ°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÉ•ÍÀ¹•Ð •ÉÉ½Èœ¥ôˆ(€€€€€€€…ÍÍ•ÉÐÍ••¹l‰ÁÉ½µÁÐ‰t€ôô€‰•‘¥Ñ•Á½ÍÐˆ(€€€€€€€…ÍÍ•ÉÐÍ••¹l‰¡¥ÍÑ½Éä‰t€ôôÑ¥Á}¡¥ÍÑ½ÉålèÉt(€€€€€€€…ÍÍ•ÉÐÍÑÕ‰}‘ˆ¹É•Á±…•€ôôl ‰Í•ÍÍ¥½¸µ­•äˆ°Ñ¥Á}¡¥ÍÑ½ÉålèÉt¥t(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}ÑÉÕ¹…Ñ•}½½É}¥¹±Õ‘•Í}ÍÑÉÕÑÕÉ•‘}ÕÍ•É}ÑÕÉ¹}½Õ¹Ð¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆˆÐÀÄà…™Ñ•È½µÁ…Ñ¥½¸µÕÍÐ…ÉÉäÉ•½Ù•Éä™¥•±‘Ì™½È•Í­Ñ½À€ ŒàÈÐØÈ¤¸ˆˆˆ((€€€Ñ¥Á}¡¥ÍÑ½Éä€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Á½ÍÐµ½µÁÉ•ÍÌ‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä‰ô°(€€€t(€€€‘¥ÍÁ±…å}ÁÉ•™¥à€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰ÁÉ”µ½µÁÉ•ÍÌ€Ä‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰ÁÉ”É•Á±ä€Ä‰ô°(€€€t(€€€€Œ=É‘¥¹…°€ÀÁ½¥¹ÑÌ…ÐÑ¡”…¹•ÍÑ½ÈÁÉ•™¥àƒŠP¹½Ð•‘¥Ñ…‰±”™É½´Ñ¡”Ñ¥À¸(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸ (€€€€€€€¡¥ÍÑ½ÉäõÑ¥Á}¡¥ÍÑ½Éä°(€€€€€€€‘¥ÍÁ±…å}¡¥ÍÑ½Éå}ÁÉ•™¥àõ‘¥ÍÁ±…å}ÁÉ•™¥à°(€€€€¤((€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}¥¹™±¥¡Ñ}ÑÕÉ¸ˆ°±…µ‰‘„€©„°€¨©¬èÁåÑ•ÍÐ¹™…¥° ‰µÕÍÐ¹½ÐÍÑ…ÉÐ„ÑÕÉ¸ˆ¤(€€€€€€€€¤(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰•‘¥Ð…¹•ÍÑ½Èˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}ÕÍ•É}½É‘¥¹…°ˆè€À°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€•ÉÈ€ôÉ•ÍÀ¹•Ð ‰•ÉÉ½Èˆ¤½Èíô(€€€€€€€…ÍÍ•ÉÐ•ÉÈ¹•Ð ‰½‘”ˆ¤€ôô€ÐÀÄà(€€€€€€€‘…Ñ„€ô•ÉÈ¹•Ð ‰‘…Ñ„ˆ¤½Èíô(€€€€€€€…ÍÍ•ÉÐ‘…Ñ„¹•Ð ‰ÕÍ•É}ÑÕÉ¹}½Õ¹Ðˆ¤€ôô€Ä(€€€€€€€…ÍÍ•ÉÐ‘…Ñ„¹•Ð ‰½É‘¥¹…°ˆ¤€ôô€À(€€€€€€€…ÍÍ•ÉÐ‘…Ñ„¹•Ð ‰Í•µ•¹Ñ}½É‘¥¹…°ˆ¤€ôô€´Ä(€€€€€€€…ÍÍ•ÉÐ‘…Ñ„¹•Ð ‰ÁÉ•™¥á}ÕÍ•É}½Õ¹Ðˆ¤€ôô€Ä(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}É½Ý}¥‘}…•ÁÑÍ}™Õ±±}±¥¹•…•}½É‘¥¹…°¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰•Í­Ñ½ÀÍ•¹‘ÌÉ½Ý%€¬„™Õ±°µ±¥¹•…”½É‘¥¹…°ì‰½Ñ µÕÍÐ…É•”€ ŒàÈÐØÈ¤¸((€€€™Ñ•È½µÁÉ•ÍÍ¥½¸Ñ¡”±¥•¹ÐÌÙ¥Í¥‰±”µÕÍ•È½É‘¥¹…°½Õ¹ÑÌÑ¡”…¹•ÍÑ½È(€€€ÁÉ•™¥àÑÕÉ¹ÌÝ¡¥±”Ñ¡”…Ñ•Ý…äÉ•Í½±Ù•ÌÑ¡”É½Ü¥Ñ¥ÀµÉ•±…Ñ¥Ù”¸Q¡”(€€€É•½¹¥±”É½ÍÌµ¡•¬µÕÍÐÑÉ•…ÐÑ¥Á}½É‘¥¹…°€¬ÁÉ•™¥á}ÕÍ•É}½Õ¹Ñ€…Ì(€€€…É••µ•¹Ð°¹½Ð€ŒàÈÜÔØ‘É¥™ÐƒŠP…¹Ñ¡”ÕÐÍÑ…åÌ…¥µ•‰äÑ¡”É½Ü¥¸(€€€€ˆˆˆ(€€€™É½´…•¹Ð¹½¹Ñ•áÑ}½µÁÉ•ÍÍ½È¥µÁ½ÉÐ€ (€€€€€€€!%MQ=I%1}QM-}!%9°(€€€€€€€MU55Ie}AI%`°(€€€€€€€}MU55Ie}9}5I-H°(€€€€¤((€€€Ñ¥Á}¡¥ÍÑ½Éä€ôl(€€€€€€€ì‰}É½Ý}¥ˆè€ÔÀÄ°€‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Á½ÍÐµ½µÁÉ•ÍÌ‰ô°(€€€€€€€ì‰}É½Ý}¥ˆè€ÔÀÈ°€‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä‰ô°(€€€€€€€ì‰}É½Ý}¥ˆè€ÔÀÌ°€‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Á½ÍÐµ½µÁÉ•ÍÌ‰ô°(€€€€€€€ì‰}É½Ý}¥ˆè€ÔÀÐ°€‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä‰ô°(€€€t(€€€‘¥ÍÁ±…å}ÁÉ•™¥à€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰ÁÉ”µ½µÁÉ•ÍÌ€Ä‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰ÁÉ”É•Á±ä€Ä‰ô°(€€€€€€€ì(€€€€€€€€€€€€Œ1•…äÁÕÉ”¡…¹‘½™™Ì‘¥¹½Ð…±Ý…åÌ…ÉÉä‘¥ÍÁ±…å}­¥¹õ¡¥‘‘•¸¸(€€€€€€€€€€€€ŒQ¡•ä…É”Á¡åÍ¥…±±äÕÍ•ÈÉ½ÝÌ‰ÕÐ¹½ÐÙ¥Í¥‰±”½ÕÍ•Èµ½É¥¥¹…Ñ•(€€€€€€€€€€€€ŒÑÕÉ¹Ì°Í¼Ñ¡•äµÕÍÐ¹½ÐÍ¡¥™ÐÑ¡”•Í­Ñ½À±¥¹•…”½É‘¥¹…°¸(€€€€€€€€€€€€‰É½±”ˆè€‰ÕÍ•Èˆ°(€€€€€€€€€€€€‰½¹Ñ•¹Ðˆè€ (€€€€€€€€€€€€€€€˜‰íMU55Ie}AI%aõq¹í!%MQ=I%1}QM-}!%9õq¹½±Ñ…Í­q¹q¸ˆ(€€€€€€€€€€€€€€€˜‰í}MU55Ie}9}5I-Iôˆ(€€€€€€€€€€€€¤°(€€€€€€€ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰ÁÉ”µ½µÁÉ•ÍÌ€È‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰ÁÉ”É•Á±ä€È‰ô°(€€€t((€€€É•Á±…•€ômt((€€€±…ÍÌ}…­•è(€€€€€€€‘•˜É•Á±…•}µ•ÍÍ…•Ì (€€€€€€€€€€€Í•±˜°(€€€€€€€€€€€­•ä°(€€€€€€€€€€€µ•ÍÍ…•Ì°(€€€€€€€€€€€…Ñ¥Ù•}½¹±äõ…±Í”°(€€€€€€€€€€€…É¡¥Ù•}‘É½ÁÁ•õ…±Í”°(€€€€€€€€€€€É•©•Ñ}…Ñ¥Ù•}ÑÕÉ¹}±•…Í”õ…±Í”°(€€€€€€€€¤è(€€€€€€€€€€€…ÍÍ•ÉÐÉ•©•Ñ}…Ñ¥Ù•}ÑÕÉ¹}±•…Í”¥ÌQÉÕ”(€€€€€€€€€€€É•Á±…•¹…ÁÁ•¹ ¡­•ä°±¥ÍÐ¡µ•ÍÍ…•Ì¤¤¤((€€€Í•ÍÌ€ô}Í•ÍÍ¥½¸¡¡¥ÍÑ½Éäõ±¥ÍÐ¡Ñ¥Á}¡¥ÍÑ½Éä¤°‘¥ÍÁ±…å}¡¥ÍÑ½Éå}ÁÉ•™¥àõ‘¥ÍÁ±…å}ÁÉ•™¥à¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰±¥¹•…”µÉ½ÜµÍ¥‰t€ôÍ•ÍÌ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è}…­• ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤((€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰±¥¹•…”µÉ½ÜµÍ¥ˆ°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰•‘¥Ñ•Á½ÍÐˆ°(€€€€€€€€€€€€€€€€€€€€ŒI½Ü¥É•Í½±Ù•ÌÑ¼Ñ¥À½É‘¥¹…°€ÄìÑ¡”±¥•¹Ð½Õ¹Ñ•Ñ¡”(€€€€€€€€€€€€€€€€€€€€Œ€È…¹•ÍÑ½ÈÕÍ•ÈÑÕÉ¹Ì°Í¼¥ÑÌ±¥¹•…”½É‘¥¹…°¥Ì€Ì¸(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}É½Ý}¥ˆè€ÔÀÌ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}ÕÍ•É}½É‘¥¹…°ˆè€Ì°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰•ÉÉ½Èˆ¤¥Ì9½¹”°˜‰½Ð•ÉÉ½ÈèíÉ•ÍÀ¹•Ð •ÉÉ½Èœ¥ôˆ(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍl‰¡¥ÍÑ½Éä‰t€ôôÑ¥Á}¡¥ÍÑ½ÉålèÉt(€€€™¥¹…±±äè(€€€€€€€€ŒI•±•…Í”Ñ¡”Í±½ÐÑ¡”™¥ÉÍÐÑÕÉ¸±…¥µ•°…Ì}™¥¹…±¥é•}Í•ÍÍ¥½¸‘½•Ì¥¸(€€€€€€€€ŒÁÉ½‘ÕÑ¥½¸¸A½ÁÁ¥¹œ…±½¹”±•…­ÌÑ¡”±•…Í”°…¹Ñ¡”Í•½¹Í•ÍÍ¥½¸‰•±½Ü(€€€€€€€€ŒÕÍ•ÌÑ¡”Í…µ”Í•ÍÍ¥½¹}­•ä€´´Í¼Ý¥Ñ¡½ÕÐÑ¡¥ÌÑ¡”Ñ•ÍÐ™•¹•Ì¥ÑÍ•±˜½ÕÐ(€€€€€€€€Œ½˜¥ÑÌ½Ý¸­•ä…¹¹•Ù•ÈÉ•…¡•ÌÑ¡”µ¥Íµ…Ñ ¥Ð¥Ì¡•­¥¹œ¸(€€€€€€€Í•ÉÙ•È¹}É•±•…Í•}…Ñ¥Ù•}Í•ÍÍ¥½¹}Í±½Ð¡Í•ÍÌ¤(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰±¥¹•…”µÉ½ÜµÍ¥ˆ°9½¹”¤((€€€€Œ•¹Õ¥¹•±äÍÑ…±”½É‘¥¹…°€¡µ…Ñ¡•Ì¹•¥Ñ¡•ÈÑ¡”Ñ¥ÀÍÁ…”¹½ÈÑ¡”(€€€€Œ±¥¹•…”ÍÁ…”¤µÕÍÐÍÑ¥±°É•™ÕÍ”Ý¥Ñ Ñ¡”€ŒàÈÜÔØµ¥Íµ…Ñ ¸(€€€Í•ÍÌÈ€ô}Í•ÍÍ¥½¸¡¡¥ÍÑ½Éäõ±¥ÍÐ¡Ñ¥Á}¡¥ÍÑ½Éä¤°‘¥ÍÁ±…å}¡¥ÍÑ½Éå}ÁÉ•™¥àõ‘¥ÍÁ±…å}ÁÉ•™¥à¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰±¥¹•…”µÉ½ÜµÍ¥´È‰t€ôÍ•ÍÌÈ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„€©„°€¨©¬èÁåÑ•ÍÐ¹™…¥° ‰µÕÍÐ¹½ÐÍÑ…ÉÐ„ÑÕÉ¸ˆ¤(€€€€¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÈˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰±¥¹•…”µÉ½ÜµÍ¥´Èˆ°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰•‘¥Ñ•Á½ÍÐˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}É½Ý}¥ˆè€ÔÀÌ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}ÕÍ•É}½É‘¥¹…°ˆè€È°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰•ÉÉ½Èˆ¤¥Ì¹½Ð9½¹”(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÌÀ(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}É•±•…Í•}…Ñ¥Ù•}Í•ÍÍ¥½¹}Í±½Ð¡Í•ÍÌÈ¤(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰±¥¹•…”µÉ½ÜµÍ¥´Èˆ°9½¹”¤(((Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(ŒÍ•ÍÍ¥½¸¹¥¹Ñ•ÉÉÕÁÐµÕÍÐ½¹±ä…¹•°Á•¹‘¥¹œÁÉ½µÁÑÌ½Ý¹•‰äÑ¡”…±±¥¹œ(ŒÍ•ÍÍ¥½¸ƒŠP¥ÐµÕÍÐ¹½Ð‰±…ÍÐµÉ•Í½±Ù”±…É¥™ä½ÍÕ‘¼½Í•É•ÐÁÉ½µÁÑÌ½¸(ŒÕ¹É•±…Ñ•Í•ÍÍ¥½¹ÌÍ¡…É¥¹œÑ¡”Í…µ”ÑÕ¥}…Ñ•Ý…äÁÉ½•ÍÌ¸€]¥Ñ¡½ÕÐ(ŒÍ•ÍÍ¥½¸Í½Á¥¹œÑ¡”½Ñ¡•ÈÍ•ÍÍ¥½¹ÌœÁÉ½µÁÑÌÍ¥±•¹Ñ±äÉ•Í½±Ù”Ñ¼•µÁÑä(ŒÍÑÉ¥¹Ì°Õ¹‰±½­¥¹œÑ¡•¥È…•¹ÐÑ¡É•…‘Ì…Ì¥˜Ñ¡”ÕÍ•È…¹•±±•¸(Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(()‘•˜}½Á•¹}É•ÅÕ•ÍÐ¡Í¥°µ•Ñ¡½ô‰±…É¥™äˆ¤è(€€€™É½´ÑÕ¥}…Ñ•Ý…ä¥µÁ½ÉÐÍ•ÉÙ•É}É•ÅÕ•ÍÑÌ(€€€É•Ä€ôÍ•ÉÙ•É}É•ÅÕ•ÍÑÌ¹M•ÉÙ•ÉI•ÅÕ•ÍÐ¡Í¥°µ•Ñ¡½°íô¤(€€€Ý¥Ñ Í•ÉÙ•É}É•ÅÕ•ÍÑÌ¹}±½¬è(€€€€€€€Í•ÉÙ•É}É•ÅÕ•ÍÑÌ¹}½Á•¹mÉ•Ä¹¥‘t€ôÉ•Ä(€€€É•ÑÕÉ¸É•Ä(()‘•˜Ñ•ÍÑ}¥¹Ñ•ÉÉÕÁÑ}½¹±å}±•…ÉÍ}½Ý¹}Í•ÍÍ¥½¹}Á•¹‘¥¹œ ¤è(€€€€ˆˆ‰Í•ÍÍ¥½¸¹¥¹Ñ•ÉÉÕÁÐ½¸Í•ÍÍ¥½¸Ý¥Ñ¡‘É…ÝÌÌ½Á•¸Í•ÉÙ•ËŠI±¥•¹ÐÉ•ÅÕ•ÍÑÌ€¡„É•ÅÕ•ÍÐ¹…¹•°•… ¤(€€€…¹µÕÍÐ9=PÑ½Õ Í•ÍÍ¥½¸ÌƒŠP½Ñ¡•ÉÝ¥Í”Ì±…É¥™ä½ÍÕ‘¼½Í•É•ÐÁÉ½µÁÐÍ¥±•¹Ñ±äÉ•Í½±Ù•Ì…Ì¥˜(€€€Ñ¡”ÕÍ•È…¹•±±•¥Ð¸ˆˆˆ(€€€¥µÁ½ÉÐÑåÁ•Ì(€€€™É½´ÑÕ¥}…Ñ•Ý…ä¥µÁ½ÉÐÍ•ÉÙ•É}É•ÅÕ•ÍÑÌ((€€€Í•ÍÍ¥½¹}„€ô}Í•ÍÍ¥½¸ ¤(€€€Í•ÍÍ¥½¹}…l‰…•¹Ð‰t€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡¥¹Ñ•ÉÉÕÁÐõ±…µ‰‘„è9½¹”¤(€€€Í•ÍÍ¥½¹}ˆ€ô}Í•ÍÍ¥½¸ ¤(€€€Í•ÍÍ¥½¹}‰l‰…•¹Ð‰t€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡¥¹Ñ•ÉÉÕÁÐõ±…µ‰‘„è9½¹”¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‘}„‰t€ôÍ•ÍÍ¥½¹}„(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‘}ˆ‰t€ôÍ•ÍÍ¥½¹}ˆ(€€€É•Å}„Ä°É•Å}„È°É•Å}ˆ€ô}½Á•¹}É•ÅÕ•ÍÐ ‰Í¥‘}„ˆ¤°}½Á•¹}É•ÅÕ•ÍÐ ‰Í¥‘}„ˆ°€‰ÍÕ‘¼ˆ¤°}½Á•¹}É•ÅÕ•ÍÐ ‰Í¥‘}ˆˆ¤((€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ¡ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹¥¹Ñ•ÉÉÕÁÐˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥‘}„‰õô¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÉ•ÍÀ¹•Ð •ÉÉ½Èœ¥ôˆ((€€€€€€€…ÍÍ•ÉÐÉ•Å}„Ä¹•Ù•¹Ð¹¥Í}Í•Ð ¤…¹É•Å}„È¹•Ù•¹Ð¹¥Í}Í•Ð ¤…¹¹½ÐÉ•Å}„Ä¹…¹ÍÝ•É•(€€€€€€€…ÍÍ•ÉÐ¹½ÐÉ•Å}ˆ¹•Ù•¹Ð¹¥Í}Í•Ð ¤°€ (€€€€€€€€€€€€‰I%Q%0èÍ•ÍÍ¥½¸¹¥¹Ñ•ÉÉÕÁÐ½¸Í¥‘}„É•±•…Í•„ÁÉ½µÁÐ‰•±½¹¥¹œÑ¼Í¥‘}ˆˆ¤(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•É}É•ÅÕ•ÍÑÌ¹½Á•¹}É•ÅÕ•ÍÑÌ ‰Í¥‘}„ˆ¤€ôômt(€€€€€€€…ÍÍ•ÉÐmÉl‰¥‰t™½ÈÈ¥¸Í•ÉÙ•É}É•ÅÕ•ÍÑÌ¹½Á•¹}É•ÅÕ•ÍÑÌ ‰Í¥‘}ˆˆ¥t€ôômÉ•Å}ˆ¹¥‘t(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥‘}„ˆ°9½¹”¤(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥‘}ˆˆ°9½¹”¤(€€€€€€€Í•ÉÙ•É}É•ÅÕ•ÍÑÌ¹É•Í•Ñ}™½É}Ñ•ÍÑÌ ¤(()‘•˜Ñ•ÍÑ}ÉÕ¹}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}É•¥ÍÑ•ÉÍ}ÑÕÉ¹}Ñ¡É•…‘}™½É}¥¹Ñ•ÉÉÕÁÐ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰}ÉÕ¹}ÁÉ½µÁÑ}ÍÕ‰µ¥ÐµÕÍÐ•áÁ½Í”Ñ¡”…ÑÕ…°ÑÕÉ¸Ñ¡É•…Ñ¼Í•ÍÍ¥½¸¹¥¹Ñ•ÉÉÕÁÐ¸((€€€ÁÉ½µÁÐ¹ÍÕ‰µ¥ÐÌ½ÕÑ•ÈÝÉ…ÁÁ•È½¹±äÝ…¥ÑÌ™½È…•¹Ð¥¹¥Ñ¥…±¥é…Ñ¥½¸°Ñ¡•¸(€€€}ÉÕ¹}ÁÉ½µÁÑ}ÍÕ‰µ¥ÐÍÑ…ÉÑÌÑ¡”É•…°½¹Ù•ÉÍ…Ñ¥½¸Ñ¡É•…¸%˜Ñ¡”Í•ÍÍ¥½¸­••ÁÌ(€€€Ñ¡”ÝÉ…ÁÁ•ÈÑ¡É•…¡…¹‘±”°ÍÑ½À½•ÍŒÍ••Ì„‘•…Ñ¡É•……¹¹•Ù•È…±±Ì(€€€…•¹Ð¹¥¹Ñ•ÉÉÕÁÐ ¤½¸Ñ¡”±¥Ù”ÑÕÉ¸¸(€€€€ˆˆˆ(€€€…±±Ì€ôì‰¥¹Ñ•ÉÉÕÁÑ•ˆè…±Í”°€‰ÍÑ…ÉÑ•ˆè…±Í•ô((€€€±…ÍÌ}…­•Q¡É•…è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©}Ñ¡É•…‘}½ÁÑ¥½¹Ì¤è(€€€€€€€€€€€Í•±˜¹Ñ…É•Ð€ôÑ…É•Ð((€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€…±±Íl‰ÍÑ…ÉÑ•‰t€ôQÉÕ”((€€€€€€€‘•˜¥Í}…±¥Ù”¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€…•¹Ð€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€¥¹Ñ•ÉÉÕÁÐõ±…µ‰‘„è…±±Ì¹}}Í•Ñ¥Ñ•µ}| ‰¥¹Ñ•ÉÉÕÁÑ•ˆ°QÉÕ”¤°(€€€€€€€ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸õ±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌèíô°(€€€€¤(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡…•¹Ðõ…•¹Ð°ÉÕ¹¹¥¹œõQÉÕ”¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ôÍ•ÍÍ¥½¸((€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}…­•Q¡É•…¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌè9½¹”¤((€€€€€€€Í•ÉÙ•È¹}ÉÕ¹}ÁÉ½µÁÑ}ÍÕ‰µ¥Ð ˆÄˆ°€‰Í¥ˆ°Í•ÍÍ¥½¸°€‰¡•±±¼ˆ¤((€€€€€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¸¹•Ð ‰}ÉÕ¹}Ñ¡É•…ˆ¤¥Ì¹½Ð9½¹”(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÈˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹¥¹Ñ•ÉÉÕÁÐˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥‰õô(€€€€€€€€¤((€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÉ•ÍÀ¹•Ð •ÉÉ½Èœ¥ôˆ(€€€€€€€…ÍÍ•ÉÐ…±±Íl‰¥¹Ñ•ÉÉÕÁÑ•‰t¥ÌQÉÕ”(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}¥¹Ñ•ÉÉÕÁÑ}‘É½ÁÍ}ÅÕ•Õ•‘}ÁÉ½µÁÑ}™½É}Í•ÍÍ¥½¸ ¤è(€€€€ˆˆ‰áÁ±¥¥ÐÍÑ½À…¹•±Ì„ÅÕ•Õ•¹•áÐÑÕÉ¸¥¹ÍÑ•…½˜…ÕÑ¼µ‘É…¥¹¥¹œ¥Ð¸ˆˆˆ(€€€…±±Ì€ôì‰¥¹Ñ•ÉÉÕÁÑ•ˆè…±Í•ô((€€€±…ÍÌ}1¥Ù•Q¡É•…è(€€€€€€€‘•˜¥Í}…±¥Ù”¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸ (€€€€€€€…•¹ÐõÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€€€€€¥¹Ñ•ÉÉÕÁÐõ±…µ‰‘„è…±±Ì¹}}Í•Ñ¥Ñ•µ}| ‰¥¹Ñ•ÉÉÕÁÑ•ˆ°QÉÕ”¤(€€€€€€€€¤°(€€€€€€€ÉÕ¹¹¥¹œõQÉÕ”°(€€€€€€€ÅÕ•Õ•‘}ÁÉ½µÁÐõì‰Ñ•áÐˆè€‰¹•áÐÁÉ½µÁÐˆ°€‰ÑÉ…¹ÍÁ½ÉÐˆè9½¹•ô°(€€€€€€€ÅÕ•Õ•‘}ÁÉ½µÁÑÌõmì‰Ñ•áÐˆè€‰±…Ñ•ÈÁÉ½µÁÐˆ°€‰¥µ…•}Á…Ñ¡Ìˆèlˆ½ÑµÀ½±…Ñ•È¹Á¹œ‰t°€‰ÑÉ…¹ÍÁ½ÉÐˆè9½¹•õt°(€€€€€€€}ÉÕ¹}Ñ¡É•…õ}1¥Ù•Q¡É•… ¤°(€€€€¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ôÍ•ÍÍ¥½¸((€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹¥¹Ñ•ÉÉÕÁÐˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥‰õô(€€€€€€€€¤((€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÉ•ÍÀ¹•Ð •ÉÉ½Èœ¥ôˆ(€€€€€€€…ÍÍ•ÉÐ…±±Íl‰¥¹Ñ•ÉÉÕÁÑ•‰t¥ÌQÉÕ”(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¸¹•Ð ‰ÅÕ•Õ•‘}ÁÉ½µÁÐˆ¤¥Ì9½¹”(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¸¹•Ð ‰ÅÕ•Õ•‘}ÁÉ½µÁÑÌˆ¤¥Ì9½¹”(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}¥¹Ñ•ÉÉÕÁÑ}‰•™½É•}…•¹Ñ}É•…‘å}ÁÉ•Ù•¹ÑÍ}±…Ñ•}ÑÕÉ¹}ÍÑ…ÉÐ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰MÑ½À‘ÕÉ¥¹œ±…éä…•¹ÐÍÑ…ÉÑÕÀµÕÍÐ¹½ÐÍÑ…ÉÐÑ¡”ÑÕÉ¸…™Ñ•È¥¹¥Ð™¥¹¥Í¡•Ì¸ˆˆˆ(€€€Ñ¡É•…‘Ì€ômt(€€€…±±Ì€ôì‰ÉÕ¹}ÁÉ½µÁÐˆè€Áô((€€€±…ÍÌ}…­•Q¡É•…è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©}Ñ¡É•…‘}½ÁÑ¥½¹Ì¤è(€€€€€€€€€€€Í•±˜¹Ñ…É•Ð€ôÑ…É•Ð(€€€€€€€€€€€Ñ¡É•…‘Ì¹…ÁÁ•¹¡Í•±˜¤((€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸9½¹”((€€€€€€€‘•˜¥Í}…±¥Ù”¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸ ¤(€€€Í•ÍÍ¥½¹l‰…•¹Ð‰t€ô9½¹”(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ôÍ•ÍÍ¥½¸((€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}…­•Q¡É•…¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌè9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•¹ÍÕÉ•}Í•ÍÍ¥½¹}‘‰}É½Üˆ°±…µ‰‘„Í•ÍÍ¥½¸è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Á•ÉÍ¥ÍÑ}‰É…¹¡}Í••ˆ°±…µ‰‘„Í•ÍÍ¥½¸è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„Í¥°Í•ÍÍ¥½¸è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Ý…¥Ñ}…•¹Ðˆ°±…µ‰‘„Í•ÍÍ¥½¸°É¥è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€Í•ÉÙ•È°(€€€€€€€€€€€€‰}ÉÕ¹}ÁÉ½µÁÑ}ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌè…±±Ì¹}}Í•Ñ¥Ñ•µ}| (€€€€€€€€€€€€€€€€‰ÉÕ¹}ÁÉ½µÁÐˆ°…±±Íl‰ÉÕ¹}ÁÉ½µÁÐ‰t€¬€Ä(€€€€€€€€€€€€¤°(€€€€€€€€¤((€€€€€€€ÍÕ‰µ¥Ð€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰¡•±±¼‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÍÕ‰µ¥Ð¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÍÕ‰µ¥Ð¹•Ð •ÉÉ½Èœ¥ôˆ(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰ÉÕ¹¹¥¹œ‰t¥ÌQÉÕ”(€€€€€€€…ÍÍ•ÉÐ±•¸¡Ñ¡É•…‘Ì¤€ôô€Ä((€€€€€€€ÍÑ½À€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÈˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹¥¹Ñ•ÉÉÕÁÐˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥‰õô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÍÑ½À¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÍÑ½À¹•Ð •ÉÉ½Èœ¥ôˆ((€€€€€€€Ñ¡É•…‘ÍlÁt¹Ñ…É•Ð ¤((€€€€€€€…ÍÍ•ÉÐ…±±Íl‰ÉÕ¹}ÁÉ½µÁÐ‰t€ôô€À(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰ÉÕ¹¹¥¹œ‰t¥Ì…±Í”(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¸¹•Ð ‰¥¹™±¥¡Ñ}ÑÕÉ¸ˆ¤¥Ì9½¹”(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}…¹•±±•‘}ÑÕÉ¹}‰•™½É•}…•¹Ñ}É•…‘å}•µ¥ÑÍ}•ÉÉ½É}•Ù•¹Ð¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰ÑÕÉ¸…¹•±±•‘ÕÉ¥¹œ±…éä…•¹ÐÍÑ…ÉÑÕÀµÕÍÐÍÕÉ™…”…¸•ÉÉ½È•Ù•¹Ð¸((€€€M¥‰±¥¹œ½˜Ñ•ÍÑ}¥¹Ñ•ÉÉÕÁÑ}‰•™½É•}…•¹Ñ}É•…‘å}ÁÉ•Ù•¹ÑÍ}±…Ñ•}ÑÕÉ¹}ÍÑ…ÉÐèÑ¡…Ð(€€€Ñ•ÍÐ½¹±ä…ÍÍ•ÉÑÌ}ÉÕ¹}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ€¥ÌÍ­¥ÁÁ•°µ½­¥¹œ}•µ¥Ñ€Ñ¼„(€€€¹¼µ½ÀÍ¼¥Ð…¹¹½Ð…Ñ „Í¥±•¹Ð‘É½À¸Q¡¥ÌÑ•ÍÐ…ÁÑÕÉ•Ì}•µ¥Ñ€…¹(€€€…ÍÍ•ÉÑÌÑ¡”±¥•¹ÐÉ••¥Ù•Ì…¸•ÉÉ½É€•Ù•¹ÐÝ¥Ñ „¡Õµ…¸µÉ•…‘…‰±”µ•ÍÍ…”°(€€€Í¼Ñ¡”•Í­Ñ½À½µÁ½Í•È…¸Í¡½Ü™••‘‰…¬¥¹ÍÑ•…½˜¡…¹¥¹œ½¸„(€€€ì‰ÍÑ…ÑÕÌˆè‰ÍÑÉ•…µ¥¹œ‰õ€É•Á±äÑ¡…Ð¹•Ù•ÈÁÉ½‘Õ•Ì„ÑÕÉ¸€¡¥ÍÍÕ”€ŒØÌÀÜà(€€€Í•ÉÙ•ÈµÍ¥‘”¡…±˜¤¸(€€€€ˆˆˆ(€€€Ñ¡É•…‘Ì€ômt(€€€•µ¥ÑÑ•€ômt(€€€…±±Ì€ôì‰ÉÕ¹}ÁÉ½µÁÐˆè€Áô((€€€±…ÍÌ}…­•Q¡É•…è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©}Ñ¡É•…‘}½ÁÑ¥½¹Ì¤è(€€€€€€€€€€€Í•±˜¹Ñ…É•Ð€ôÑ…É•Ð(€€€€€€€€€€€Ñ¡É•…‘Ì¹…ÁÁ•¹¡Í•±˜¤((€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸9½¹”((€€€€€€€‘•˜¥Í}…±¥Ù”¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸ ¤(€€€Í•ÍÍ¥½¹l‰…•¹Ð‰t€ô9½¹”(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ôÍ•ÍÍ¥½¸((€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}…­•Q¡É•…¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌè•µ¥ÑÑ•¹…ÁÁ•¹¡…ÉÌ¤¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•¹ÍÕÉ•}Í•ÍÍ¥½¹}‘‰}É½Üˆ°±…µ‰‘„Í•ÍÍ¥½¸è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Á•ÉÍ¥ÍÑ}‰É…¹¡}Í••ˆ°±…µ‰‘„Í•ÍÍ¥½¸è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„Í¥°Í•ÍÍ¥½¸è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Ý…¥Ñ}…•¹Ðˆ°±…µ‰‘„Í•ÍÍ¥½¸°É¥è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€Í•ÉÙ•È°(€€€€€€€€€€€€‰}ÉÕ¹}ÁÉ½µÁÑ}ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌè…±±Ì¹}}Í•Ñ¥Ñ•µ}| (€€€€€€€€€€€€€€€€‰ÉÕ¹}ÁÉ½µÁÐˆ°…±±Íl‰ÉÕ¹}ÁÉ½µÁÐ‰t€¬€Ä(€€€€€€€€€€€€¤°(€€€€€€€€¤((€€€€€€€ÍÕ‰µ¥Ð€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰¡•±±¼‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÍÕ‰µ¥Ð¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÍÕ‰µ¥Ð¹•Ð •ÉÉ½Èœ¥ôˆ(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰ÉÕ¹¹¥¹œ‰t¥ÌQÉÕ”((€€€€€€€€ŒUÍ•È¡¥ÑÌMÑ½ÀÝ¡¥±”Ñ¡”…•¹Ð¥ÌÍÑ¥±°‰Õ¥±‘¥¹œ¸(€€€€€€€ÍÑ½À€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÈˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹¥¹Ñ•ÉÉÕÁÐˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥‰õô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÍÑ½À¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÍÑ½À¹•Ð •ÉÉ½Èœ¥ôˆ(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¸¹•Ð ‰}ÑÕÉ¹}…¹•±}É•ÅÕ•ÍÑ•ˆ¤¥ÌQÉÕ”((€€€€€€€€ŒQ¡”‘•™•ÉÉ•ÉÕ¸Ñ¡É•…¹½ÜÝ…­•ÌÕÀìÝ¥Ñ¡½ÕÐÑ¡”•µ¥Ð¥ÐÝ½Õ±‰…¥°(€€€€€€€€ŒÍ¥±•¹Ñ±ä…¹Ñ¡”•Í­Ñ½ÀÝ½Õ±¹•Ù•È±•…É¸Ñ¡”ÑÕÉ¸Ý…Ì‘É½ÁÁ•¸(€€€€€€€Ñ¡É•…‘ÍlÁt¹Ñ…É•Ð ¤((€€€€€€€…ÍÍ•ÉÐ…±±Íl‰ÉÕ¹}ÁÉ½µÁÐ‰t€ôô€À(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰ÉÕ¹¹¥¹œ‰t¥Ì…±Í”(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¸¹•Ð ‰¥¹™±¥¡Ñ}ÑÕÉ¸ˆ¤¥Ì9½¹”(€€€€€€€€Œá…Ñ±ä½¹”•ÉÉ½È•Ù•¹Ð…‘‘É•ÍÍ•Ñ¼Ñ¡¥ÌÍ•ÍÍ¥½¸¸(€€€€€€€•ÉÉ½É}•Ù•¹ÑÌ€ôm”™½È”¥¸•µ¥ÑÑ•¥˜”…¹±•¸¡”¤€øô€È…¹•lÁt€ôô€‰•ÉÉ½Èˆ…¹•lÅt€ôô€‰Í¥‰t(€€€€€€€…ÍÍ•ÉÐ±•¸¡•ÉÉ½É}•Ù•¹ÑÌ¤€ôô€Ä°˜‰•áÁ•Ñ•½¹”•ÉÉ½È•Ù•¹Ð°½Ðèí•µ¥ÑÑ•‘ôˆ(€€€€€€€µÍœ€ô•ÉÉ½É}•Ù•¹ÑÍlÁulÉt¹•Ð ‰µ•ÍÍ…”ˆ°€ˆˆ¤(€€€€€€€…ÍÍ•ÉÐ€‰…¹•±±•ˆ¥¸µÍœ¹±½Ý•È ¤°˜‰Õ¹•áÁ•Ñ•µ•ÍÍ…”èíµÍôˆ(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}¹½Ñ}ÉÕ¹¹¥¹}‰•™½É•}…•¹Ñ}É•…‘å}•µ¥ÑÍ}•ÉÉ½É}•Ù•¹Ð¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰]¡•¸ÉÕ¹¹¥¹€¥Ì±•…É•‰äÍ½µ•Ñ¡¥¹œ½Ñ¡•ÈÑ¡…¸…¸•áÁ±¥¥Ð¥¹Ñ•ÉÉÕÁÐ(€€€€¡”¹œ¸„½¹ÕÉÉ•¹ÐÍ•ÍÍ¥½¸¹É•…Ñ”É…”Ñ¡…ÐÉ•Í•ÑÌÑ¡”™±…œ¤°Ñ¡”‘•™•ÉÉ•(€€€ÉÕ¸Ñ¡É•…µÕÍÐÍÑ¥±°•µ¥Ð…¸•ÉÉ½È•Ù•¹ÐÉ…Ñ¡•ÈÑ¡…¸‘¥Í…ÁÁ•…É¥¹œÍ¥±•¹Ñ±ä¸(€€€€ˆˆˆ(€€€Ñ¡É•…‘Ì€ômt(€€€•µ¥ÑÑ•€ômt(€€€…±±Ì€ôì‰ÉÕ¹}ÁÉ½µÁÐˆè€Áô((€€€±…ÍÌ}…­•Q¡É•…è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©}Ñ¡É•…‘}½ÁÑ¥½¹Ì¤è(€€€€€€€€€€€Í•±˜¹Ñ…É•Ð€ôÑ…É•Ð(€€€€€€€€€€€Ñ¡É•…‘Ì¹…ÁÁ•¹¡Í•±˜¤((€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸9½¹”((€€€€€€€‘•˜¥Í}…±¥Ù”¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸ ¤(€€€Í•ÍÍ¥½¹l‰…•¹Ð‰t€ô9½¹”(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ôÍ•ÍÍ¥½¸((€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}…­•Q¡É•…¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌè•µ¥ÑÑ•¹…ÁÁ•¹¡…ÉÌ¤¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•¹ÍÕÉ•}Í•ÍÍ¥½¹}‘‰}É½Üˆ°±…µ‰‘„Í•ÍÍ¥½¸è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Á•ÉÍ¥ÍÑ}‰É…¹¡}Í••ˆ°±…µ‰‘„Í•ÍÍ¥½¸è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„Í¥°Í•ÍÍ¥½¸è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Ý…¥Ñ}…•¹Ðˆ°±…µ‰‘„Í•ÍÍ¥½¸°É¥è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€Í•ÉÙ•È°(€€€€€€€€€€€€‰}ÉÕ¹}ÁÉ½µÁÑ}ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌè…±±Ì¹}}Í•Ñ¥Ñ•µ}| (€€€€€€€€€€€€€€€€‰ÉÕ¹}ÁÉ½µÁÐˆ°…±±Íl‰ÉÕ¹}ÁÉ½µÁÐ‰t€¬€Ä(€€€€€€€€€€€€¤°(€€€€€€€€¤((€€€€€€€ÍÕ‰µ¥Ð€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰¡•±±¼‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÍÕ‰µ¥Ð¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÍÕ‰µ¥Ð¹•Ð •ÉÉ½Èœ¥ôˆ(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰ÉÕ¹¹¥¹œ‰t¥ÌQÉÕ”((€€€€€€€€ŒM¥µÕ±…Ñ”„½¹ÕÉÉ•¹ÐÁ…Ñ ±•…É¥¹œÉÕ¹¹¥¹€Ý¥Ñ¡½ÕÐÍ•ÑÑ¥¹œÑ¡”(€€€€€€€€Œ…¹•°™±…œ€¡Ñ¡”½Ñ¡•È‰É…¹ ½˜Ñ¡”Õ…É¤¸(€€€€€€€Ý¥Ñ Í•ÍÍ¥½¹l‰¡¥ÍÑ½Éå}±½¬‰tè(€€€€€€€€€€€Í•ÍÍ¥½¹l‰ÉÕ¹¹¥¹œ‰t€ô…±Í”((€€€€€€€Ñ¡É•…‘ÍlÁt¹Ñ…É•Ð ¤((€€€€€€€…ÍÍ•ÉÐ…±±Íl‰ÉÕ¹}ÁÉ½µÁÐ‰t€ôô€À(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¸¹•Ð ‰¥¹™±¥¡Ñ}ÑÕÉ¸ˆ¤¥Ì9½¹”(€€€€€€€•ÉÉ½É}•Ù•¹ÑÌ€ôm”™½È”¥¸•µ¥ÑÑ•¥˜”…¹±•¸¡”¤€øô€È…¹•lÁt€ôô€‰•ÉÉ½Èˆ…¹•lÅt€ôô€‰Í¥‰t(€€€€€€€…ÍÍ•ÉÐ±•¸¡•ÉÉ½É}•Ù•¹ÑÌ¤€ôô€Ä°˜‰•áÁ•Ñ•½¹”•ÉÉ½È•Ù•¹Ð°½Ðèí•µ¥ÑÑ•‘ôˆ(€€€€€€€µÍœ€ô•ÉÉ½É}•Ù•¹ÑÍlÁulÉt¹•Ð ‰µ•ÍÍ…”ˆ°€ˆˆ¤(€€€€€€€…ÍÍ•ÉÐµÍœ¹ÍÑÉ¥À ¤°˜‰•µÁÑä•ÉÉ½Èµ•ÍÍ…”èí•µ¥ÑÑ•‘ôˆ(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}Í±½Ý}…•¹Ñ}‰Õ¥±‘}‘•±¥Ù•ÉÍ}ÁÉ½µÁÑ}¥¹ÍÑ•…‘}½™}Ñ¥µ¥¹}½ÕÐ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆˆŒØÌÀÜàÍ•ÉÙ•ÈµÍ¥‘”¡…±˜è„‘•™•ÉÉ•‰Õ¥±Í±½Ý•ÈÑ¡…¸Ñ¡”½±€ÌÁÌ(€€€}Ý…¥Ñ}…•¹Ñ€±¥™˜µÕÍÐ9=P•…ÐÑ¡”™¥ÉÍÐµ•ÍÍ…”¸Q¡”Á…Ñ¥•¹ÐÝ…¥Ð(€€€­••ÁÌÑ¡”Á•¹‘¥¹œÁÉ½µÁÐ…ÑÑ…¡•…¹‘•±¥Ù•ÉÌ¥Ð…ÌÍ½½¸…ÌÑ¡”(€€€ÍÑ¥±°µÉÕ¹¹¥¹œ‰Õ¥±½µÁ±•Ñ•Ì¸ˆˆˆ(€€€Ñ¡É•…‘Ì€ômt(€€€•µ¥ÑÑ•€ômt(€€€…±±Ì€ôì‰ÉÕ¹}ÁÉ½µÁÐˆè€Áô((€€€±…ÍÌ}…­•Q¡É•…è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©}Ñ¡É•…‘}½ÁÑ¥½¹Ì¤è(€€€€€€€€€€€Í•±˜¹Ñ…É•Ð€ôÑ…É•Ð(€€€€€€€€€€€Ñ¡É•…‘Ì¹…ÁÁ•¹¡Í•±˜¤((€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸9½¹”((€€€€€€€‘•˜¥Í}…±¥Ù”¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€É•…‘ä€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡…•¹Ñ}É•…‘äõÉ•…‘ä¤(€€€Í•ÍÍ¥½¹l‰…•¹Ð‰t€ô9½¹”(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ôÍ•ÍÍ¥½¸((€€€€ŒQ¡”‰Õ¥±€‰½µÁ±•Ñ•Ìˆ½¹±ä…™Ñ•ÈÑ¡”Ý…¥Ð±½½À¡…Ì…±É•…‘ä½¹”Ñ¡É½Õ (€€€€ŒÍ•Ù•É…°•µÁÑäÍ±¥•ÌƒŠP¤¹”¸Ý•±°Á…ÍÐÝ¡…Ð„Í¥¹±”™¥á•µÑ¥µ•½ÕÐÝ…¥Ð(€€€€ŒÍ±¥”Ý½Õ±Ñ½±•É…Ñ”¸(€€€Í±¥•Ì€ôì‰¸ˆè€Áô((€€€±…ÍÌ}M±½ÝI•…‘äè(€€€€€€€‘•˜Ý…¥Ð¡Í•±˜°Ñ¥µ•½ÕÐõ9½¹”¤è(€€€€€€€€€€€Í±¥•Íl‰¸‰t€¬ô€Ä(€€€€€€€€€€€¥˜Í±¥•Íl‰¸‰t€øô€Ìè(€€€€€€€€€€€€€€€É•…‘ä¹Í•Ð ¤(€€€€€€€€€€€€€€€Í•ÍÍ¥½¹l‰…•¹Ð‰t€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” ¤(€€€€€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”(€€€€€€€€€€€É•ÑÕÉ¸…±Í”((€€€€€€€‘•˜¥Í}Í•Ð¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸É•…‘ä¹¥Í}Í•Ð ¤((€€€Í•ÍÍ¥½¹l‰…•¹Ñ}É•…‘ä‰t€ô}M±½ÝI•…‘ä ¤((€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}…­•Q¡É•…¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌè•µ¥ÑÑ•¹…ÁÁ•¹¡…ÉÌ¤¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•¹ÍÕÉ•}Í•ÍÍ¥½¹}‘‰}É½Üˆ°±…µ‰‘„Í•ÍÍ¥½¸è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Á•ÉÍ¥ÍÑ}‰É…¹¡}Í••ˆ°±…µ‰‘„Í•ÍÍ¥½¸è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„Í¥°Í•ÍÍ¥½¸è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€Í•ÉÙ•È°(€€€€€€€€€€€€‰}ÉÕ¹}ÁÉ½µÁÑ}ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌè…±±Ì¹}}Í•Ñ¥Ñ•µ}| (€€€€€€€€€€€€€€€€‰ÉÕ¹}ÁÉ½µÁÐˆ°…±±Íl‰ÉÕ¹}ÁÉ½µÁÐ‰t€¬€Ä(€€€€€€€€€€€€¤°(€€€€€€€€¤((€€€€€€€ÍÕ‰µ¥Ð€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰™¥ÉÍÐµ•ÍÍ…”‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÍÕ‰µ¥Ð¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÍÕ‰µ¥Ð¹•Ð •ÉÉ½Èœ¥ôˆ((€€€€€€€Ñ¡É•…‘ÍlÁt¹Ñ…É•Ð ¤((€€€€€€€€ŒQ¡”µ•ÍÍ…”Ý…Ì1%YI°¹½Ð‘É½ÁÁ•°…¹¹¼•ÉÉ½È•Ù•¹Ð™¥É•¸(€€€€€€€…ÍÍ•ÉÐ…±±Íl‰ÉÕ¹}ÁÉ½µÁÐ‰t€ôô€Ä(€€€€€€€•ÉÉ½É}•Ù•¹ÑÌ€ôm”™½È”¥¸•µ¥ÑÑ•¥˜”…¹•lÁt€ôô€‰•ÉÉ½È‰t(€€€€€€€…ÍÍ•ÉÐ¹½Ð•ÉÉ½É}•Ù•¹ÑÌ°˜‰Õ¹•áÁ•Ñ••ÉÉ½È•Ù•¹ÑÌèí•ÉÉ½É}•Ù•¹ÑÍôˆ(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}Í±½Ý}…•¹Ñ}‰Õ¥±‘}•µ¥ÑÍ}­•å•‘}ÁÉ½É•ÍÍ}¹½Ñ¥”¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰A…ÍÐÑ¡”Í±½ÜÑ¡É•Í¡½±Ñ¡”Á…Ñ¥•¹ÐÝ…¥ÐµÕÍÐÑ•±°Ñ¡”ÕÍ•È½¹”(€€€€¡­•å•¹½Ñ¥™¥…Ñ¥½¸¹Í¡½Ü¤…¹±•…ÈÑ¡”¹½Ñ¥”Ý¡•¸Ñ¡”‰Õ¥±±…¹‘ÌƒŠP(€€€„±½¹œÝ…¥Ð¥Ì…•ÁÑ…‰±”°„Í¥±•¹Ð½¹”¥Ì¹½Ð¸ˆˆˆ(€€€Ñ¡É•…‘Ì€ômt(€€€•µ¥ÑÑ•€ômt(€€€…±±Ì€ôì‰ÉÕ¹}ÁÉ½µÁÐˆè€Áô((€€€±…ÍÌ}…­•Q¡É•…è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©}Ñ¡É•…‘}½ÁÑ¥½¹Ì¤è(€€€€€€€€€€€Í•±˜¹Ñ…É•Ð€ôÑ…É•Ð(€€€€€€€€€€€Ñ¡É•…‘Ì¹…ÁÁ•¹¡Í•±˜¤((€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸9½¹”((€€€€€€€‘•˜¥Í}…±¥Ù”¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€É•…‘ä€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡…•¹Ñ}É•…‘äõÉ•…‘ä¤(€€€Í•ÍÍ¥½¹l‰…•¹Ð‰t€ô9½¹”(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ôÍ•ÍÍ¥½¸((€€€Í±¥•Ì€ôì‰¸ˆè€Áô((€€€±…ÍÌ}M±½ÝI•…‘äè(€€€€€€€‘•˜Ý…¥Ð¡Í•±˜°Ñ¥µ•½ÕÐõ9½¹”¤è(€€€€€€€€€€€Í±¥•Íl‰¸‰t€¬ô€Ä(€€€€€€€€€€€¥˜Í±¥•Íl‰¸‰t€øô€Ìè(€€€€€€€€€€€€€€€É•…‘ä¹Í•Ð ¤(€€€€€€€€€€€€€€€Í•ÍÍ¥½¹l‰…•¹Ð‰t€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” ¤(€€€€€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”(€€€€€€€€€€€É•ÑÕÉ¸…±Í”((€€€€€€€‘•˜¥Í}Í•Ð¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸É•…‘ä¹¥Í}Í•Ð ¤((€€€Í•ÍÍ¥½¹l‰…•¹Ñ}É•…‘ä‰t€ô}M±½ÝI•…‘ä ¤((€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}…­•Q¡É•…¤(€€€€€€€€ŒÙ•ÉäÝ…¥ÐÍ±¥”±…¹‘ÌÁ…ÍÐÑ¡”Í±½ÜÑ¡É•Í¡½±¸(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}9Q}	U%1}M1=]}9=Q%}QHˆ°€À¸À¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌè•µ¥ÑÑ•¹…ÁÁ•¹¡…ÉÌ¤¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•¹ÍÕÉ•}Í•ÍÍ¥½¹}‘‰}É½Üˆ°±…µ‰‘„Í•ÍÍ¥½¸è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Á•ÉÍ¥ÍÑ}‰É…¹¡}Í••ˆ°±…µ‰‘„Í•ÍÍ¥½¸è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„Í¥°Í•ÍÍ¥½¸è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€Í•ÉÙ•È°(€€€€€€€€€€€€‰}ÉÕ¹}ÁÉ½µÁÑ}ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌè…±±Ì¹}}Í•Ñ¥Ñ•µ}| (€€€€€€€€€€€€€€€€‰ÉÕ¹}ÁÉ½µÁÐˆ°…±±Íl‰ÉÕ¹}ÁÉ½µÁÐ‰t€¬€Ä(€€€€€€€€€€€€¤°(€€€€€€€€¤((€€€€€€€ÍÕ‰µ¥Ð€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰™¥ÉÍÐµ•ÍÍ…”‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÍÕ‰µ¥Ð¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÍÕ‰µ¥Ð¹•Ð •ÉÉ½Èœ¥ôˆ((€€€€€€€Ñ¡É•…‘ÍlÁt¹Ñ…É•Ð ¤((€€€€€€€…ÍÍ•ÉÐ…±±Íl‰ÉÕ¹}ÁÉ½µÁÐ‰t€ôô€Ä(€€€€€€€Í¡½ÝÌ€ôm”™½È”¥¸•µ¥ÑÑ•¥˜”…¹•lÁt€ôô€‰¹½Ñ¥™¥…Ñ¥½¸¹Í¡½Üˆ…¹•lÅt€ôô€‰Í¥‰t(€€€€€€€±•…ÉÌ€ôm”™½È”¥¸•µ¥ÑÑ•¥˜”…¹•lÁt€ôô€‰¹½Ñ¥™¥…Ñ¥½¸¹±•…Èˆ…¹•lÅt€ôô€‰Í¥‰t(€€€€€€€€Œá…Ñ±ä½¹”­•å•¹½Ñ¥”°É•Á±…•µ¥¸µÁ±…”Í•µ…¹Ñ¥Ì°Ñ¡•¸±•…É•¸(€€€€€€€…ÍÍ•ÉÐ±•¸¡Í¡½ÝÌ¤€ôô€Ä°˜‰•áÁ•Ñ•½¹”Í±½Üµ‰Õ¥±¹½Ñ¥”°½ÐèíÍ¡½ÝÍôˆ(€€€€€€€…ÍÍ•ÉÐÍ¡½ÝÍlÁulÉt¹•Ð ‰­•äˆ¤€ôôÍ•ÉÙ•È¹}9Q}	U%1}M1=]}9=Q%}-d(€€€€€€€…ÍÍ•ÉÐ±•¸¡±•…ÉÌ¤€ôô€Ä…¹±•…ÉÍlÁulÉt¹•Ð ‰­•äˆ¤€ôôÍ•ÉÙ•È¹}9Q}	U%1}M1=]}9=Q%}-d(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}…•¹Ñ}‰Õ¥±‘}™…¥±ÕÉ•}ÍÕÉ™…•Í}•ÉÉ½É}…¹‘}‘É½ÁÍ}ÑÕÉ¸¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰]¡•¸Ñ¡”‰Õ¥±¥ÑÍ•±˜%1L€¡…•¹Ñ}•ÉÉ½ÈÍ•ÐÝ¡•¸É•…‘ä™¥É•Ì¤°Ñ¡”(€€€ÁÉ½µÁÐµÕÍÐ¹½ÐÉÕ¸…¹Ñ¡”™…¥±ÕÉ”µÕÍÐÉ•… Ñ¡”±¥•¹Ð…Ì„Ù¥Í¥‰±”(€€€•ÉÉ½È•Ù•¹ÐƒŠP¹•Ù•È„Í¥±•¹Ð‘É½À¸((€€€ÁÉ½µÁÐ¹ÍÕ‰µ¥ÐÉ•ÑÉ¥•Ì„½µÁ±•Ñ•™…¥±•‰Õ¥±½¹”€¡™É•Í ÁÉ½Ù¥‘•È(€€€É•Í½±ÕÑ¥½¸Õ¸µÝ•‘•ÌÍ•ÍÍ¥½¹ÌÝ¡½Í”™…¥±ÕÉ”…ÕÍ”Ý…Ì™¥á•¤°Í¼Ñ¡”(€€€‰Õ¥±ÍÑÕˆ¡•É”¥Ì„™…¥Ñ¡™Õ°™…¥±¥¹œ‰Õ¥±è¥ÐÍ•ÑÌ…•¹Ñ}•ÉÉ½È…¹(€€€™¥É•ÌÑ¡”Í•ÍÍ¥½¸ÌUII9PÉ•…‘ä•Ù•¹Ð€¡Ñ¡”É•ÑÉä¥¹ÍÑ…±±Ì„¹•Ü½¹”¤¸(€€€¹¼µ½ÀÍÑÕˆÝ½Õ±±•…Ù”Ñ¡…Ð•Ù•¹ÐÕ¹Í•Ð…¹¡…¹œÑ¡”Á…Ñ¥•¹ÐÝ…¥Ð¸ˆˆˆ(€€€Ñ¡É•…‘Ì€ômt(€€€•µ¥ÑÑ•€ômt(€€€…±±Ì€ôì‰ÉÕ¹}ÁÉ½µÁÐˆè€Áô((€€€±…ÍÌ}…­•Q¡É•…è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©}Ñ¡É•…‘}½ÁÑ¥½¹Ì¤è(€€€€€€€€€€€Í•±˜¹Ñ…É•Ð€ôÑ…É•Ð(€€€€€€€€€€€Ñ¡É•…‘Ì¹…ÁÁ•¹¡Í•±˜¤((€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸9½¹”((€€€€€€€‘•˜¥Í}…±¥Ù”¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€É•…‘ä€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤(€€€É•…‘ä¹Í•Ð ¤€€Œ‰Õ¥±™¥¹¥Í¡•¸¸¸(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡…•¹Ñ}É•…‘äõÉ•…‘ä¤(€€€Í•ÍÍ¥½¹l‰…•¹Ð‰t€ô9½¹”(€€€Í•ÍÍ¥½¹l‰…•¹Ñ}•ÉÉ½È‰t€ô€‰9¼114ÁÉ½Ù¥‘•È½¹™¥ÕÉ•ˆ€€Œ€¸¸¹‰ÕÐ™…¥±•(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ôÍ•ÍÍ¥½¸((€€€‘•˜}™…¥±¥¹}‰Õ¥±¡Í¥°Í•ÍÍ¥½¸¤è(€€€€€€€Í•ÍÍ¥½¹l‰…•¹Ñ}•ÉÉ½È‰t€ô€‰9¼114ÁÉ½Ù¥‘•È½¹™¥ÕÉ•ˆ(€€€€€€€Í•ÍÍ¥½¹l‰…•¹Ñ}É•…‘ä‰t¹Í•Ð ¤((€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}…­•Q¡É•…¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌè•µ¥ÑÑ•¹…ÁÁ•¹¡…ÉÌ¤¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•¹ÍÕÉ•}Í•ÍÍ¥½¹}‘‰}É½Üˆ°±…µ‰‘„Í•ÍÍ¥½¸è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Á•ÉÍ¥ÍÑ}‰É…¹¡}Í••ˆ°±…µ‰‘„Í•ÍÍ¥½¸è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°}™…¥±¥¹}‰Õ¥±¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€Í•ÉÙ•È°(€€€€€€€€€€€€‰}ÉÕ¹}ÁÉ½µÁÑ}ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌè…±±Ì¹}}Í•Ñ¥Ñ•µ}| (€€€€€€€€€€€€€€€€‰ÉÕ¹}ÁÉ½µÁÐˆ°…±±Íl‰ÉÕ¹}ÁÉ½µÁÐ‰t€¬€Ä(€€€€€€€€€€€€¤°(€€€€€€€€¤((€€€€€€€ÍÕ‰µ¥Ð€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰™¥ÉÍÐµ•ÍÍ…”‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÍÕ‰µ¥Ð¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÍÕ‰µ¥Ð¹•Ð •ÉÉ½Èœ¥ôˆ((€€€€€€€Ñ¡É•…‘ÍlÁt¹Ñ…É•Ð ¤((€€€€€€€…ÍÍ•ÉÐ…±±Íl‰ÉÕ¹}ÁÉ½µÁÐ‰t€ôô€À(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰ÉÕ¹¹¥¹œ‰t¥Ì…±Í”(€€€€€€€€Œ€ŒÜÄÄàÐÕÁÉ…‘•™…¥±ÕÉ”‘•±¥Ù•Éä™É½´„‰…É”€‰•ÉÉ½Èˆ•Ù•¹ÐÑ¼„(€€€€€€€€ŒÑ•Éµ¥¹…°µ•ÍÍ…”¹½µÁ±•Ñ”™É…µ”€¡ÍÑ…ÑÕÌõ•ÉÉ½È°É•½Ù•É…‰±”¤Í¼(€€€€€€€€Œ™…¥±•ÑÕÉ¹Ì…É”É•Ñ…¥¹•…ÌÉ•Á±…å…‰±”¥¹™±¥¡ÐÍ¹…ÁÍ¡½ÑÌ¸Q¡”(€€€€€€€€Œ½¹ÑÉ…ÐÑ¡¥ÌÑ•ÍÐÁ¥¹Ì¥ÌÕ¹¡…¹•èÑ¡”‰Õ¥±™…¥±ÕÉ”µÕÍÐÉ•… (€€€€€€€€ŒÑ¡”±¥•¹ÐY%M%	1dƒŠP¹•Ù•È„Í¥±•¹Ð‘É½À¸(€€€€€€€™…¥±ÕÉ•}™É…µ•Ì€ôl(€€€€€€€€€€€”(€€€€€€€€€€€™½È”¥¸•µ¥ÑÑ•(€€€€€€€€€€€¥˜”(€€€€€€€€€€€…¹•lÁt¥¸€ ‰•ÉÉ½Èˆ°€‰µ•ÍÍ…”¹½µÁ±•Ñ”ˆ¤(€€€€€€€€€€€…¹•lÅt€ôô€‰Í¥ˆ(€€€€€€€€€€€…¹€ (€€€€€€€€€€€€€€€€‰9¼114ÁÉ½Ù¥‘•È½¹™¥ÕÉ•ˆ¥¸ÍÑÈ¡•lÉt¹•Ð ‰µ•ÍÍ…”ˆ°€ˆˆ¤¤(€€€€€€€€€€€€€€€½È€‰9¼114ÁÉ½Ù¥‘•È½¹™¥ÕÉ•ˆ¥¸ÍÑÈ¡•lÉt¹•Ð ‰•ÉÉ½Èˆ°€ˆˆ¤¤(€€€€€€€€€€€€€€€½È€‰9¼114ÁÉ½Ù¥‘•È½¹™¥ÕÉ•ˆ¥¸ÍÑÈ¡•lÉt¹•Ð ‰Ñ•áÐˆ°€ˆˆ¤¤(€€€€€€€€€€€€¤(€€€€€€€t(€€€€€€€…ÍÍ•ÉÐ±•¸¡™…¥±ÕÉ•}™É…µ•Ì¤€ôô€Ä°˜‰•áÁ•Ñ•½¹”Ù¥Í¥‰±”™…¥±ÕÉ”™É…µ”°½Ðèí•µ¥ÑÑ•‘ôˆ(€€€€€€€™É…µ”€ô™…¥±ÕÉ•}™É…µ•ÍlÁt(€€€€€€€¥˜™É…µ•lÁt€ôô€‰µ•ÍÍ…”¹½µÁ±•Ñ”ˆè(€€€€€€€€€€€…ÍÍ•ÉÐ™É…µ•lÉt¹•Ð ‰ÍÑ…ÑÕÌˆ¤€ôô€‰•ÉÉ½Èˆ(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}‘•…‘}‰Õ¥±‘}Ñ¡É•…‘}™…¥±Í}™…ÍÑ}¹½Ñ}™Õ±±}…À¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰‰Õ¥±Ñ¡É•…Ñ¡…Ð‘¥•Ý¥Ñ¡½ÕÐÍ•ÑÑ¥¹œ…•¹Ñ}É•…‘äµ•…¹ÌÑ¡”‰Õ¥±(€€€‘¥•¡…ÉƒŠPÑ¡”Ý…¥Ñ•ÈµÕÍÐ™…¥°ÁÉ½µÁÑ±äÝ¥Ñ „Ù¥Í¥‰±”•ÉÉ½È¥¹ÍÑ•…½˜(€€€Í¥ÑÑ¥¹œ½ÕÐÑ¡”™Õ±°Ý…¥Ð…À½¸„½ÉÁÍ”¸ˆˆˆ(€€€•µ¥ÑÑ•€ômt((€€€±…ÍÌ}•…‘Q¡É•…è(€€€€€€€‘•˜¥Í}…±¥Ù”¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸…±Í”((€€€É•…‘ä€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤€€Œ¹•Ù•ÈÍ•Ð(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡…•¹Ñ}É•…‘äõÉ•…‘ä¤(€€€Í•ÍÍ¥½¹l‰…•¹Ð‰t€ô9½¹”(€€€Í•ÍÍ¥½¹l‰ÉÕ¹¹¥¹œ‰t€ôQÉÕ”(€€€Í•ÍÍ¥½¹l‰}…•¹Ñ}‰Õ¥±‘}Ñ¡É•…‰t€ô}•…‘Q¡É•… ¤(€€€Í•ÍÍ¥½¹l‰…•¹Ñ}•ÉÉ½È‰t€ô€‰…•¹Ð¥¹¥Ð™…¥±•è‰½½´ˆ(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ôÍ•ÍÍ¥½¸((€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌè•µ¥ÑÑ•¹…ÁÁ•¹¡…ÉÌ¤¤(€€€€€€€€ŒM¡½ÉÐÍ±¥•ÌÍ¼Ñ¡”Ñ•ÍÐ¥Ì™…ÍÐìÑ¡”‘•…µÑ¡É•…¡•¬™¥É•Ì½¸Ñ¡”(€€€€€€€€Œ™¥ÉÍÐ•µÁÑäÍ±¥”°™…È‰•±½ÜÑ¡”…À¸(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}9Q}	U%1}]%Q}M1%ˆ°€À¸ÀÄ¤((€€€€€€€ÍÑ…ÉÐ€ôÑ¥µ”¹µ½¹½Ñ½¹¥Œ ¤(€€€€€€€•ÉÈ€ôÍ•ÉÙ•È¹}Ý…¥Ñ}…•¹Ñ}™½É}ÁÉ½µÁÐ¡Í•ÍÍ¥½¸°€‰É¥´Äˆ°€‰Í¥ˆ¤(€€€€€€€•±…ÁÍ•€ôÑ¥µ”¹µ½¹½Ñ½¹¥Œ ¤€´ÍÑ…ÉÐ((€€€€€€€…ÍÍ•ÉÐ•ÉÈ¥Ì¹½Ð9½¹”(€€€€€€€…ÍÍ•ÉÐ€‰‰½½´ˆ¥¸€¡•ÉÈ¹•Ð ‰•ÉÉ½Èˆ¤½Èíô¤¹•Ð ‰µ•ÍÍ…”ˆ°€ˆˆ¤(€€€€€€€…ÍÍ•ÉÐ•±…ÁÍ•€ð€Ô¸À°˜‰‘•…µÑ¡É•…‘•Ñ•Ñ¥½¸Ñ½½¬í•±…ÁÍ•è¸Å™õÌˆ(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}Ý…¥Ñ}…•¹Ñ}™½É}ÁÉ½µÁÑ}¡½¹½ÉÍ}…¹•±}µ¥‘}Ý…¥Ð¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰…¹•°…ÉÉ¥Ù¥¹œ‘ÕÉ¥¹œÑ¡”Á…Ñ¥•¹ÐÝ…¥ÐµÕÍÐ•¹¥ÐÁÉ½µÁÑ±ä…¹(€€€É•ÑÕÉ¸9½¹”€¡Ñ¡”…±±•ÈÌ…¹•°‰É…¹ ½Ý¹ÌÑ¡”ÕÍ•ÈµÙ¥Í¥‰±”•Ù•¹Ð¤¸ˆˆˆ(€€€É•…‘ä€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤€€Œ¹•Ù•ÈÍ•Ð(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡…•¹Ñ}É•…‘äõÉ•…‘ä¤(€€€Í•ÍÍ¥½¹l‰…•¹Ð‰t€ô9½¹”(€€€Í•ÍÍ¥½¹l‰ÉÕ¹¹¥¹œ‰t€ôQÉÕ”(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ôÍ•ÍÍ¥½¸((€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}9Q}	U%1}]%Q}M1%ˆ°€À¸ÀÄ¤((€€€€€€€‘•˜…¹•±}Í½½¸ ¤è(€€€€€€€€€€€Ñ¥µ”¹Í±••À À¸ÀÔ¤(€€€€€€€€€€€Ý¥Ñ Í•ÍÍ¥½¹l‰¡¥ÍÑ½Éå}±½¬‰tè(€€€€€€€€€€€€€€€Í•ÍÍ¥½¹l‰}ÑÕÉ¹}…¹•±}É•ÅÕ•ÍÑ•‰t€ôQÉÕ”((€€€€€€€…¹•±±•È€ôÑ¡É•…‘¥¹œ¹Q¡É•…¡Ñ…É•Ðõ…¹•±}Í½½¸¤(€€€€€€€…¹•±±•È¹ÍÑ…ÉÐ ¤(€€€€€€€ÍÑ…ÉÐ€ôÑ¥µ”¹µ½¹½Ñ½¹¥Œ ¤(€€€€€€€•ÉÈ€ôÍ•ÉÙ•È¹}Ý…¥Ñ}…•¹Ñ}™½É}ÁÉ½µÁÐ¡Í•ÍÍ¥½¸°€‰É¥´Äˆ°€‰Í¥ˆ¤(€€€€€€€•±…ÁÍ•€ôÑ¥µ”¹µ½¹½Ñ½¹¥Œ ¤€´ÍÑ…ÉÐ(€€€€€€€…¹•±±•È¹©½¥¸ ¤((€€€€€€€…ÍÍ•ÉÐ•ÉÈ¥Ì9½¹”(€€€€€€€…ÍÍ•ÉÐ•±…ÁÍ•€ð€Ô¸À°˜‰…¹•°¡½¹½É•½¹±ä…™Ñ•Èí•±…ÁÍ•è¸Å™õÌˆ(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}…•¹Ñ}‰Õ¥±‘}Ý…¥Ñ}…Á}½¹™¥}½Ù•ÉÉ¥‘”¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰…•¹Ð¹‰Õ¥±‘}Ý…¥Ñ}Ñ¥µ•½ÕÐ¥¸½¹™¥œ¹å…µ°½Ù•ÉÉ¥‘•ÌÑ¡”‘•™…Õ±Ð…Àì(€€€¥¹Ù…±¥½…‰Í•¹ÐÙ…±Õ•Ì™…±°‰…¬Ñ¼€ØÀÁÌ¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}™œˆ°±…µ‰‘„èì‰…•¹Ðˆèì‰‰Õ¥±‘}Ý…¥Ñ}Ñ¥µ•½ÕÐˆè€äÁõô¤(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}…•¹Ñ}‰Õ¥±‘}Ý…¥Ñ}…À ¤€ôô€äÀ¸À((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}™œˆ°±…µ‰‘„èì‰…•¹Ðˆèíõô¤(€€€‘•™…Õ±Ñ}…À€ôÍ•ÉÙ•È¹}…•¹Ñ}‰Õ¥±‘}Ý…¥Ñ}…À ¤(€€€…ÍÍ•ÉÐ‘•™…Õ±Ñ}…À€ø€À…¹‘•™…Õ±Ñ}…À€„ô€äÀ¸À((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}™œˆ°±…µ‰‘„èì‰…•¹Ðˆèì‰‰Õ¥±‘}Ý…¥Ñ}Ñ¥µ•½ÕÐˆè€Áõô¤(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}…•¹Ñ}‰Õ¥±‘}Ý…¥Ñ}…À ¤€ôô‘•™…Õ±Ñ}…À((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}™œˆ°±…µ‰‘„èì‰…•¹Ðˆèì‰‰Õ¥±‘}Ý…¥Ñ}Ñ¥µ•½ÕÐˆè€‰¹½¹Í•¹Í”‰õô¤(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}…•¹Ñ}‰Õ¥±‘}Ý…¥Ñ}…À ¤€ôô‘•™…Õ±Ñ}…À(()‘•˜Ñ•ÍÑ}Ý…¥Ñ}…•¹Ñ}™½É}ÁÉ½µÁÑ}•áÁ¥É•Í}…Ñ}…À¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰•¹Õ¥¹•±ä¡Õ¹œ‰Õ¥±€¡Ñ¡É•……±¥Ù”°¹•Ù•ÈÉ•…‘ä¤ÍÑ¥±°™…¥±Ì…ÐÑ¡”(€€€‰½Õ¹‘•…ÀÝ¥Ñ „µ•ÍÍ…”Ñ¡…ÐÑ•±±ÌÑ¡”ÕÍ•ÈÑ¡•¥ÈÑ•áÐÝ…Ì¹½ÐÍ•¹Ð¸ˆˆˆ(€€€±…ÍÌ}±¥Ù•Q¡É•…è(€€€€€€€‘•˜¥Í}…±¥Ù”¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€É•…‘ä€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤€€Œ¹•Ù•ÈÍ•Ð(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡…•¹Ñ}É•…‘äõÉ•…‘ä¤(€€€Í•ÍÍ¥½¹l‰…•¹Ð‰t€ô9½¹”(€€€Í•ÍÍ¥½¹l‰ÉÕ¹¹¥¹œ‰t€ôQÉÕ”(€€€Í•ÍÍ¥½¹l‰}…•¹Ñ}‰Õ¥±‘}Ñ¡É•…‰t€ô}±¥Ù•Q¡É•… ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ôÍ•ÍÍ¥½¸((€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}9Q}	U%1}]%Q}M1%ˆ°€À¸ÀÄ¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}…•¹Ñ}‰Õ¥±‘}Ý…¥Ñ}…Àˆ°±…µ‰‘„è€À¸ÀÔ¤((€€€€€€€•ÉÈ€ôÍ•ÉÙ•È¹}Ý…¥Ñ}…•¹Ñ}™½É}ÁÉ½µÁÐ¡Í•ÍÍ¥½¸°€‰É¥´Äˆ°€‰Í¥ˆ¤((€€€€€€€…ÍÍ•ÉÐ•ÉÈ¥Ì¹½Ð9½¹”(€€€€€€€µ•ÍÍ…”€ô€¡•ÉÈ¹•Ð ‰•ÉÉ½Èˆ¤½Èíô¤¹•Ð ‰µ•ÍÍ…”ˆ°€ˆˆ¤(€€€€€€€…ÍÍ•ÉÐ€‰Ñ¥µ•½ÕÐˆ¥¸µ•ÍÍ…”…¹€‰Ý…Ì¹½ÐÍ•¹Ðˆ¥¸µ•ÍÍ…”(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}±•…É}Á•¹‘¥¹}Ý¥Ñ¡½ÕÑ}Í¥‘}±•…ÉÍ}…±° ¤è(€€€€ˆˆ‰}±•…É}Á•¹‘¥¹œ¡9½¹”¤¥ÌÑ¡”ÁÉ½•ÍÌµ•á¥ÐÁ…Ñ ƒŠP•Ù•Éä½Á•¸É•ÅÕ•ÍÐ¥ÌÝ¥Ñ¡‘É…Ý¸°…¹„É•ÍÁ½¹Í”™½È(€€€„Ý¥Ñ¡‘É…Ý¸¥¥Ì‘É½ÁÁ•ÅÕ¥•Ñ±ä€¡¹¼•ÉÉ½È™É…µ”‰…¬Ñ¼Ñ¡”±¥•¹Ð¤¸ˆˆˆ(€€€™É½´ÑÕ¥}…Ñ•Ý…ä¥µÁ½ÉÐÍ•ÉÙ•É}É•ÅÕ•ÍÑÌ(€€€É•ÅÌ€ôm}½Á•¹}É•ÅÕ•ÍÐ ‰Í¥µàˆ¤°}½Á•¹}É•ÅÕ•ÍÐ ‰Í¥µäˆ°€‰ÍÕ‘¼ˆ¥t(€€€ÑÉäè(€€€€€€€Í•ÉÙ•È¹}±•…É}Á•¹‘¥¹œ¡9½¹”¤(€€€€€€€…ÍÍ•ÉÐ…±°¡È¹•Ù•¹Ð¹¥Í}Í•Ð ¤…¹¹½ÐÈ¹…¹ÍÝ•É•™½ÈÈ¥¸É•ÅÌ¤(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹‘¥ÍÁ…Ñ ¡ì‰©Í½¹ÉÁŒˆè€ˆÈ¸Àˆ°€‰¥ˆèÉ•ÅÍlÁt¹¥°€‰É•ÍÕ±Ðˆèì‰…¹ÍÝ•Èˆè€‰±…Ñ”‰õô¤¥Ì9½¹”(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•É}É•ÅÕ•ÍÑÌ¹É•Í•Ñ}™½É}Ñ•ÍÑÌ ¤(((Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(Œ€½µ½‘•°ÍÝ¥Ñ …¹½Ñ¡•È…•¹ÐµµÕÑ…Ñ¥¹œ½µµ…¹‘ÌµÕÍÐÉ•©•ÐÝ¡¥±”Ñ¡”(ŒÍ•ÍÍ¥½¸¥ÌÉÕ¹¹¥¹œ¸€…•¹Ð¹ÍÝ¥Ñ¡}µ½‘•° ¤µÕÑ…Ñ•ÌÍ•±˜¹µ½‘•°°Í•±˜¹ÁÉ½Ù¥‘•È°(ŒÍ•±˜¹‰…Í•}ÕÉ°°Í•±˜¹±¥•¹Ð•ÑŒ¸¥¸Á±…”ƒŠPÑ¡”Ý½É­•ÈÑ¡É•…ÉÕ¹¹¥¹œ(Œ…•¹Ð¹ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¥ÌÉ•…‘¥¹œÑ¡½Í”½¸•Ù•Éä¥Ñ•É…Ñ¥½¸¸€M¼„µ¥µÑÕÉ¸(Œ½¹™¥œ¹Í•Ðµ½‘•°µÕÍÐ9=PÍÝ¥Ñ ¥¸Á±…”ì¥¹ÍÑ•…¥ÐÅÕ•Õ•ÌÑ¡”Á¥¬(Œ€¡Í•ÍÍ¥½¹l‰Á•¹‘¥¹}µ½‘•±}ÍÝ¥Ñ ‰t¤…¹}…ÁÁ±å}Á•¹‘¥¹}µ½‘•±}ÍÝ¥Ñ …ÁÁ±¥•Ì¥Ð(Œ½¸Ñ¡”ÑÕÉ¸Ñ¡É•……ÐÑ¡”¹•áÐÑÕÉ¸ÍÑ…ÉÐ°Ý¡•É”¹½Ñ¡¥¹œ¥Ì¥¸™±¥¡Ð¸(Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(()‘•˜Ñ•ÍÑ}½¹™¥}Í•Ñ}µ½‘•±}‘•™•ÉÍ}Ý¡¥±•}ÉÕ¹¹¥¹œ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆˆ½µ½‘•°Ù¥„½¹™¥œ¹Í•ÐÅÕ•Õ•ÌÑ¡”Á¥¬‘ÕÉ¥¹œ…¸¥¸µ™±¥¡ÐÑÕÉ¸¥¹ÍÑ•…(€€€½˜É•©•Ñ¥¹œ½ÈÉ…¥¹œÑ¡”Ý½É­•ÈÑ¡É•…¸ˆˆˆ(€€€Í••¸€ôì‰…±±•ˆè…±Í•ô((€€€‘•˜}™…­•}…ÁÁ±ä¡Í¥°Í•ÍÍ¥½¸°É…Ü°€¨©}­Ý…ÉÌ¤è(€€€€€€€Í••¹l‰…±±•‰t€ôQÉÕ”(€€€€€€€É•ÑÕÉ¸ì‰Ù…±Õ”ˆèÉ…Ü°€‰Ý…É¹¥¹œˆè€ˆ‰ô((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}…ÁÁ±å}µ½‘•±}ÍÝ¥Ñ ˆ°}™…­•}…ÁÁ±ä¤((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸¡ÉÕ¹¹¥¹œõQÉÕ”¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰½¹™¥œ¹Í•Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°(€€€€€€€€€€€€€€€€€€€€‰­•äˆè€‰µ½‘•°ˆ°(€€€€€€€€€€€€€€€€€€€€‰Ù…±Õ”ˆè€‰…¹Ñ¡É½Á¥Œ½±…Õ‘”µÍ½¹¹•Ð´Ð¸Øˆ°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐ¹½ÐÉ•ÍÀ¹•Ð ‰•ÉÉ½Èˆ¤(€€€€€€€É•ÍÕ±Ð€ôÉ•ÍÁl‰É•ÍÕ±Ð‰t(€€€€€€€…ÍÍ•ÉÐÉ•ÍÕ±Ñl‰‘•™•ÉÉ•‰t¥ÌQÉÕ”(€€€€€€€…ÍÍ•ÉÐÉ•ÍÕ±Ñl‰Ù…±Õ”‰t€ôô€‰…¹Ñ¡É½Á¥Œ½±…Õ‘”µÍ½¹¹•Ð´Ð¸Øˆ(€€€€€€€…ÍÍ•ÉÐ¹½ÐÍ••¹l‰…±±•‰t°€ (€€€€€€€€€€€€‰}…ÁÁ±å}µ½‘•±}ÍÝ¥Ñ É…¸µ¥µÑÕÉ¸ƒŠPÝ½Õ±É…”Ñ¡”Ý½É­•ÈÑ¡É•…€ˆ(€€€€€€€€€€€€‰É•…‘¥¹œ…•¹Ð¹µ½‘•°€¼…•¹Ð¹±¥•¹Ðì¥ÐµÕÍÐ‘•™•ÈÑ¼ÑÕÉ¸ÍÑ…ÉÐˆ(€€€€€€€€¤(€€€€€€€Á•¹‘¥¹œ€ôÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t¹•Ð ‰Á•¹‘¥¹}µ½‘•±}ÍÝ¥Ñ ˆ¤(€€€€€€€…ÍÍ•ÉÐÁ•¹‘¥¹œ…¹Á•¹‘¥¹l‰É…Ü‰t€ôô€‰…¹Ñ¡É½Á¥Œ½±…Õ‘”µÍ½¹¹•Ð´Ð¸Øˆ(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}…ÁÁ±å}Á•¹‘¥¹}µ½‘•±}ÍÝ¥Ñ¡}ÉÕ¹Í}ÅÕ•Õ•‘}Á¥¬¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰Q¡”ÅÕ•Õ•Á¥¬¥Ì½¹ÍÕµ•½¹”°½¸Ñ¡”ÑÕÉ¸Ñ¡É•…°Ù¥„(€€€}…ÁÁ±å}µ½‘•±}ÍÝ¥Ñ ƒŠP…¹±•…É•Í¼¥Ð…¸ÐÉ”µ™¥É”¹•áÐÑÕÉ¸¸ˆˆˆ(€€€…±±Ì€ômt((€€€‘•˜}™…­•}…ÁÁ±ä¡Í¥°Í•ÍÍ¥½¸°É…Ü°€¨©­Ý…ÉÌ¤è(€€€€€€€…±±Ì¹…ÁÁ•¹¡É…Ü¤(€€€€€€€É•ÑÕÉ¸ì‰Ù…±Õ”ˆèÉ…Ü°€‰Ý…É¹¥¹œˆè€ˆˆ°€‰½¹™¥Éµ}É•ÅÕ¥É•ˆè…±Í•ô((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}…ÁÁ±å}µ½‘•±}ÍÝ¥Ñ ˆ°}™…­•}…ÁÁ±ä¤((€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡ÉÕ¹¹¥¹œõ…±Í”¤(€€€Í•ÍÍ¥½¹l‰…•¹Ð‰t€ô½‰©•Ð ¤(€€€Í•ÍÍ¥½¹l‰Á•¹‘¥¹}µ½‘•±}ÍÝ¥Ñ ‰t€ôì(€€€€€€€€‰É…Üˆè€‰…¹Ñ¡É½Á¥Œ½±…Õ‘”µÍ½¹¹•Ð´Ð¸Øˆ°(€€€€€€€€‰½¹™¥Éµ}•áÁ•¹Í¥Ù•}µ½‘•°ˆè…±Í”°(€€€ô((€€€Í•ÉÙ•È¹}…ÁÁ±å}Á•¹‘¥¹}µ½‘•±}ÍÝ¥Ñ  ‰Í¥ˆ°Í•ÍÍ¥½¸¤(€€€…ÍÍ•ÉÐ…±±Ì€ôôl‰…¹Ñ¡É½Á¥Œ½±…Õ‘”µÍ½¹¹•Ð´Ð¸Ø‰t(€€€…ÍÍ•ÉÐ€‰Á•¹‘¥¹}µ½‘•±}ÍÝ¥Ñ ˆ¹½Ð¥¸Í•ÍÍ¥½¸((€€€€Œ%‘•µÁ½Ñ•¹Ðè„Í•½¹ÑÕÉ¸ÍÑ…ÉÐÝ¥Ñ ¹½Ñ¡¥¹œÅÕ•Õ•¥Ì„¹¼µ½À¸(€€€Í•ÉÙ•È¹}…ÁÁ±å}Á•¹‘¥¹}µ½‘•±}ÍÝ¥Ñ  ‰Í¥ˆ°Í•ÍÍ¥½¸¤(€€€…ÍÍ•ÉÐ…±±Ì€ôôl‰…¹Ñ¡É½Á¥Œ½±…Õ‘”µÍ½¹¹•Ð´Ð¸Ø‰t(((()‘•˜Ñ•ÍÑ}µ¥ÉÉ½É}Í±…Í¡}Í¥‘•}•™™•ÑÍ}É•©•ÑÍ}µÕÑ…Ñ¥¹}½µµ…¹‘Í}Ý¡¥±•}ÉÕ¹¹¥¹œ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰M±…Í Ý½É­•ÈÁ…ÍÍÑ¡É½Õ €¡”¹œ¸€½µ½‘•°°€½Á•ÉÍ½¹…±¥Ñä°€½ÁÉ½µÁÐ°(€€€€½½µÁÉ•ÍÌ¤µÕÍÐÉ•©•Ð‘ÕÉ¥¹œ…¸¥¸µ™±¥¡ÐÑÕÉ¸¸€M…µ”É…”…Ì(€€€½¹™¥œ¹Í•ÐƒŠPµÕÑ…Ñ•Ì±¥Ù”…•¹ÐÍÑ…Ñ”Ý¡¥±”ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¥Ì(€€€É•…‘¥¹œ¥Ð¸ˆˆˆ(€€€¥µÁ½ÉÐÑåÁ•Ì((€€€…ÁÁ±¥•€ôì‰µ½‘•°ˆè…±Í”°€‰½µÁÉ•ÍÌˆè…±Í•ô((€€€‘•˜}™…­•}…ÁÁ±å}µ½‘•°¡Í¥°Í•ÍÍ¥½¸°…Éœ¤è(€€€€€€€…ÁÁ±¥•‘l‰µ½‘•°‰t€ôQÉÕ”(€€€€€€€É•ÑÕÉ¸ì‰Ù…±Õ”ˆè…Éœ°€‰Ý…É¹¥¹œˆè€ˆ‰ô((€€€‘•˜}™…­•}½µÁÉ•ÍÌ¡Í•ÍÍ¥½¸°™½ÕÌ¤è(€€€€€€€…ÁÁ±¥•‘l‰½µÁÉ•ÍÌ‰t€ôQÉÕ”(€€€€€€€É•ÑÕÉ¸€ À°íô¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}…ÁÁ±å}µ½‘•±}ÍÝ¥Ñ ˆ°}™…­•}…ÁÁ±å}µ½‘•°¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}½µÁÉ•ÍÍ}Í•ÍÍ¥½¹}¡¥ÍÑ½Éäˆ°}™…­•}½µÁÉ•ÍÌ¤((€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡ÉÕ¹¹¥¹œõQÉÕ”¤(€€€Í•ÍÍ¥½¹l‰…•¹Ð‰t€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡µ½‘•°ô‰àˆ¤((€€€™½Èµ°•áÁ•Ñ•‘}¹…µ”¥¸l(€€€€€€€€ ˆ½µ½‘•°¹•Ü½µ½‘•°ˆ°€‰µ½‘•°ˆ¤°(€€€€€€€€ ˆ½Á•ÉÍ½¹…±¥Ñä‘•™…Õ±Ðˆ°€‰Á•ÉÍ½¹…±¥Ñäˆ¤°(€€€€€€€€ ˆ½ÁÉ½µÁÐˆ°€‰ÁÉ½µÁÐˆ¤°(€€€€€€€€ ˆ½½µÁÉ•ÍÌˆ°€‰½µÁÉ•ÍÌˆ¤°(€€€tè(€€€€€€€Ý…É¹¥¹œ€ôÍ•ÉÙ•È¹}µ¥ÉÉ½É}Í±…Í¡}Í¥‘•}•™™•ÑÌ ‰Í¥ˆ°Í•ÍÍ¥½¸°µ¤(€€€€€€€…ÍÍ•ÉÐ€ (€€€€€€€€€€€€‰Í•ÍÍ¥½¸‰ÕÍäˆ¥¸Ý…É¹¥¹œ(€€€€€€€€¤°˜‰íµ‘ôÍ¡½Õ±¡…Ù”É•ÑÕÉ¹•‰ÕÍäÝ…É¹¥¹œ°½ÐèíÝ…É¹¥¹œ…Éôˆ(€€€€€€€…ÍÍ•ÉÐ˜ˆ½í•áÁ•Ñ•‘}¹…µ•ôˆ¥¸Ý…É¹¥¹œ((€€€€Œ9½¹”½˜Ñ¡”µÕÑ…Ñ¥¹œÍ¥‘”µ•™™•Ð¡•±Á•ÉÌÍ¡½Õ±¡…Ù”™¥É•¸(€€€…ÍÍ•ÉÐ¹½Ð…ÁÁ±¥•‘l‰µ½‘•°‰t°€‰µ½‘•°ÍÝ¥Ñ ™¥É•‘•ÍÁ¥Ñ”ÉÕ¹¹¥¹œÍ•ÍÍ¥½¸ˆ(€€€…ÍÍ•ÉÐ¹½Ð…ÁÁ±¥•‘l‰½µÁÉ•ÍÌ‰t°€‰½µÁÉ•ÍÌ™¥É•‘•ÍÁ¥Ñ”ÉÕ¹¹¥¹œÍ•ÍÍ¥½¸ˆ(()‘•˜Ñ•ÍÑ}µ¥ÉÉ½É}Í±…Í¡}Í¥‘•}•™™•ÑÍ}…±±½Ý•‘}Ý¡•¹}¥‘±”¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰I•É•ÍÍ¥½¸Õ…Éè¥‘±”Í•ÍÍ¥½¸ÍÑ¥±°ÉÕ¹ÌÑ¡”Í¥‘”•™™•ÑÌ¸ˆˆˆ(€€€¥µÁ½ÉÐÑåÁ•Ì((€€€…ÁÁ±¥•€ôì‰µ½‘•°ˆè…±Í•ô((€€€‘•˜}™…­•}…ÁÁ±å}µ½‘•°¡Í¥°Í•ÍÍ¥½¸°…Éœ¤è(€€€€€€€…ÁÁ±¥•‘l‰µ½‘•°‰t€ôQÉÕ”(€€€€€€€É•ÑÕÉ¸ì‰Ù…±Õ”ˆè…Éœ°€‰Ý…É¹¥¹œˆè€ˆ‰ô((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}…ÁÁ±å}µ½‘•±}ÍÝ¥Ñ ˆ°}™…­•}…ÁÁ±å}µ½‘•°¤((€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡ÉÕ¹¹¥¹œõ…±Í”¤(€€€Í•ÍÍ¥½¹l‰…•¹Ð‰t€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡µ½‘•°ô‰àˆ¤((€€€Ý…É¹¥¹œ€ôÍ•ÉÙ•È¹}µ¥ÉÉ½É}Í±…Í¡}Í¥‘•}•™™•ÑÌ ‰Í¥ˆ°Í•ÍÍ¥½¸°€ˆ½µ½‘•°™½¼ˆ¤(€€€€ŒM¡½Õ±9=P½¹Ñ…¥¸€‰Í•ÍÍ¥½¸‰ÕÍäˆƒŠPÑ¡”ÍÝ¥Ñ Ý•¹ÐÑ¡É½Õ ¸(€€€…ÍÍ•ÉÐ€‰Í•ÍÍ¥½¸‰ÕÍäˆ¹½Ð¥¸Ý…É¹¥¹œ(€€€…ÍÍ•ÉÐ…ÁÁ±¥•‘l‰µ½‘•°‰t(()‘•˜Ñ•ÍÑ}µ¥ÉÉ½É}Í±…Í¡}½µÁÉ•ÍÍ}‘½•Í}¹½Ñ}ÁÉ•±½­}¡¥ÍÑ½Éä¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰I•É•ÍÍ¥½¸Õ…Éè€½½µÁÉ•ÍÌÍ¥‘”•™™•ÐµÕÍÐ¹½Ð¡½±¡¥ÍÑ½Éå}±½¬(€€€Ý¡•¸…±±¥¹œ}½µÁÉ•ÍÍ}Í•ÍÍ¥½¹}¡¥ÍÑ½Éä€¡Ñ¡”¡•±Á•ÈÍ¹…ÁÍ¡½ÑÌÕ¹‘•È(€€€Ñ¡”Í…µ”¹½¸µÉ••¹ÑÉ…¹Ð±½¬¥¹Ñ•É¹…±±ä¤¸%Ð…±Í¼É•ÑÕÉ¹Ì„‰•™½É”½…™Ñ•È(€€€ÍÕµµ…ÉäÍÑÉ¥¹œ€ ŒÐØØàØ¤¸ˆˆˆ(€€€¥µÁ½ÉÐÑåÁ•Ì((€€€Í••¸€ôì‰½µÁÉ•ÍÌˆè…±Í”°€‰Íå¹Œˆè…±Í•ô(€€€•µ¥ÑÑ•€ômt((€€€‘•˜}™…­•}½µÁÉ•ÍÌ¡Í•ÍÍ¥½¸°™½ÕÍ}Ñ½Á¥Œõ9½¹”°€¨©}­Ü¤è(€€€€€€€Í••¹l‰½µÁÉ•ÍÌ‰t€ôQÉÕ”(€€€€€€€…ÍÍ•ÉÐ¹½ÐÍ•ÍÍ¥½¹l‰¡¥ÍÑ½Éå}±½¬‰t¹±½­• ¤(€€€€€€€€ŒM¥µÕ±…Ñ”„É•…°½µÁ…Ñ¥½¸Í¡É¥¹­¥¹œÑ¡”ÑÉ…¹ÍÉ¥ÁÐ¸(€€€€€€€Í•ÍÍ¥½¹l‰¡¥ÍÑ½Éä‰t€ômì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰ÍÕµµ…Éä‰õt(€€€€€€€É•ÑÕÉ¸€ Ä°ì‰Ñ½Ñ…°ˆè€Áô¤((€€€‘•˜}™…­•}Íå¹Œ¡}Í¥°}Í•ÍÍ¥½¸¤è(€€€€€€€Í••¹l‰Íå¹Œ‰t€ôQÉÕ”((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}½µÁÉ•ÍÍ}Í•ÍÍ¥½¹}¡¥ÍÑ½Éäˆ°}™…­•}½µÁÉ•ÍÌ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Íå¹}Í•ÍÍ¥½¹}­•å}…™Ñ•É}½µÁÉ•ÍÌˆ°}™…­•}Íå¹Œ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„}…•¹Ð°€©„èì‰µ½‘•°ˆè€‰à‰ô¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©…ÉÌè•µ¥ÑÑ•¹…ÁÁ•¹¡…ÉÌ¤¤((€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡ÉÕ¹¹¥¹œõ…±Í”¤(€€€Í•ÍÍ¥½¹l‰¡¥ÍÑ½Éä‰t€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè˜‰µí¥ô‰ô™½È¤¥¸É…¹” Ø¤(€€€t(€€€Í•ÍÍ¥½¹l‰…•¹Ð‰t€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡µ½‘•°ô‰àˆ°}…¡•‘}ÍåÍÑ•µ}ÁÉ½µÁÐôˆˆ°Ñ½½±Ìõ9½¹”¤((€€€Ý…É¹¥¹œ€ôÍ•ÉÙ•È¹}µ¥ÉÉ½É}Í±…Í¡}Í¥‘•}•™™•ÑÌ ‰Í¥ˆ°Í•ÍÍ¥½¸°€ˆ½½µÁÉ•ÍÌˆ¤((€€€€Œ9½ÜÉ•ÑÕÉ¹Ì„‰•™½É”½…™Ñ•ÈÍÕµµ…Éä€¡Ý…Ì€ˆˆ‰•™½É”€ŒÐØØàØ¤¸(€€€…ÍÍ•ÉÐÍ••¹l‰½µÁÉ•ÍÌ‰t(€€€…ÍÍ•ÉÐÍ••¹l‰Íå¹Œ‰t(€€€…ÍÍ•ÉÐ€ ‰Í•ÍÍ¥½¸¹¥¹™¼ˆ°€‰Í¥ˆ°ì‰µ½‘•°ˆè€‰à‰ô¤¥¸•µ¥ÑÑ•(€€€…ÍÍ•ÉÐÝ…É¹¥¹œ€€Œ‰•™½É”½…™Ñ•ÈÍÕµµ…ÉäÉ•ÑÕÉ¹•€ ŒÐØØàØ¤(()}AIQ%1}-}!%MQ=Id€ôl(€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰µÍœÄ‰ô°(€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•ÍÀÄ‰ô°(€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰µÍœÈ‰ô°(€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•ÍÀÈ‰ô°(€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰­••ÀÑ¡¥Ì‰ô°(€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰­••ÀÑ¡¥ÌÑ½¼‰ô°)t)}AIQ%1}=5AIMM}!€ôl(€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰mÍÕµµ…Éåt‰ô°(€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰½¬‰ô°)t(()‘•˜}Á…ÉÑ¥…±}½µÁÉ•ÍÍ}…•¹Ð¡½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±Ì¤è(€€€€ˆˆ‰•¹ÐÍÑÕˆÝ¡½Í”}½µÁÉ•ÍÍ}½¹Ñ•áÐÉ•½É‘Ì€¡¡¥ÍÑ½Éä°™½ÕÍ}Ñ½Á¥Œ¤¸ˆˆˆ(€€€…•¹Ð€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€}…¡•‘}ÍåÍÑ•µ}ÁÉ½µÁÐõ9½¹”°(€€€€€€€Ñ½½±Ìõ9½¹”°(€€€€€€€Í•ÍÍ¥½¹}¥ô‰ÌÄˆ°(€€€€€€€½¹Ñ•áÑ}½µÁÉ•ÍÍ½Èõ9½¹”°€€Œ­••À}•Ñ}ÕÍ…”½¸Ñ¡”Í¥µÁ±”Á…Ñ (€€€€¤((€€€‘•˜}™…­•}½µÁÉ•ÍÍ}½¹Ñ•áÐ¡¡¥ÍÑ½Éä°ÍåÌ°…ÁÁÉ½á}Ñ½­•¹ÌôÀ°™½ÕÍ}Ñ½Á¥Œõ9½¹”°€¨©­Ü¤è(€€€€€€€½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±Ì¹…ÁÁ•¹ ¡±¥ÍÐ¡¡¥ÍÑ½Éä¤°™½ÕÍ}Ñ½Á¥Œ¤¤(€€€€€€€É•ÑÕÉ¸±¥ÍÐ¡}AIQ%1}=5AIMM}!¤°íô((€€€…•¹Ð¹}½µÁÉ•ÍÍ}½¹Ñ•áÐ€ô}™…­•}½µÁÉ•ÍÍ}½¹Ñ•áÐ(€€€É•ÑÕÉ¸…•¹Ð(()‘•˜Ñ•ÍÑ}½µÁÉ•ÍÍ}Í•ÍÍ¥½¹}¡¥ÍÑ½Éå}¡•É•}ÑÉ¥•ÉÍ}Á…ÉÑ¥…±}½µÁÉ•ÍÌ ¤è(€€€€ˆˆˆ½½µÁÉ•ÍÌ¡•É”m9tµÕÍÐÍÁ±¥Ð¡¥ÍÑ½Éä¥¹Ñ¼¡•…½Ñ…¥°…¹É•©½¥¸…™Ñ•È(€€€½µÁÉ•ÍÍ¥½¸ƒŠPÑ¡”Á…ÉÑ¥…±}½µÁÉ•ÍÌµ½‘Õ±”¥ÌÕÍ•°¹½Ð™Õ±°½µÁÉ•ÍÌ¸((€€€	•™½É”Ñ¡¥Ì™¥à°€½½µÁÉ•ÍÌ¡•É”€ÌÁ…ÍÍ•€‰¡•É”€Ìˆ…Ì™½ÕÍ}Ñ½Á¥ŒÑ¼Ñ¡”(€€€™Õ±°½µÁÉ•ÍÌ°Í¥±•¹Ñ±ä¥¹½É¥¹œÑ¡”‰½Õ¹‘…Éä¥¹Ñ•¹Ð¸Q¡”Á…ÉÍ¥¹œ±¥Ù•Ì(€€€¥¸}½µÁÉ•ÍÍ}Í•ÍÍ¥½¹}¡¥ÍÑ½ÉäƒŠPÑ¡”¡½­”Á½¥¹Ð•Ù•Éäµ…¹Õ…°µ½µÁÉ•ÍÌ(€€€É½ÕÑ”€¡Í•ÍÍ¥½¸¹½µÁÉ•ÍÌIA°½µµ…¹¹‘¥ÍÁ…Ñ °Í±…Í µ•á•Œµ¥ÉÉ½È¤(€€€½¹Ù•É•Ì½¸ƒŠPÍ¼€¡•É”m9tœÝ½É­Ì•Ù•ÉåÝ¡•É”€ ŒÌÔÔÌÌ¤¸(€€€€ˆˆˆ(€€€½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±Ì€ômt(€€€…•¹Ð€ô}Á…ÉÑ¥…±}½µÁÉ•ÍÍ}…•¹Ð¡½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±Ì¤((€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡…•¹Ðõ…•¹Ð¤(€€€Í•ÍÍ¥½¹l‰¡¥ÍÑ½Éä‰t€ô±¥ÍÐ¡}AIQ%1}-}!%MQ=Id¤(€€€Í•ÍÍ¥½¹l‰¡¥ÍÑ½Éå}Ù•ÉÍ¥½¸‰t€ô€Ü((€€€É•µ½Ù•°}ÕÍ…”€ôÍ•ÉÙ•È¹}½µÁÉ•ÍÍ}Í•ÍÍ¥½¹}¡¥ÍÑ½Éä¡Í•ÍÍ¥½¸°€‰¡•É”€Äˆ¤((€€€€Œ…•¹Ð¹}½µÁÉ•ÍÍ}½¹Ñ•áÐµÕÍÐ¡…Ù”‰••¸…±±•Ý¥Ñ Ñ¡”!½¹±ä(€€€…ÍÍ•ÉÐ±•¸¡½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±Ì¤€ôô€Ä(€€€¡•…‘}Á…ÍÍ•°™½ÕÍ}Á…ÍÍ•€ô½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±ÍlÁt(€€€…ÍÍ•ÉÐ¡•…‘}Á…ÍÍ•€ôô}AIQ%1}-}!%MQ=Ielè´Ét(€€€…ÍÍ•ÉÐ™½ÕÍ}Á…ÍÍ•¥Ì9½¹”€€ŒÁ…ÉÑ¥…°½µÁÉ•ÍÌ¡…Ì¹¼™½ÕÌÑ½Á¥Œ(€€€€ŒM•ÍÍ¥½¸¡¥ÍÑ½ÉäµÕÍÐ¹½Ü½¹Ñ…¥¸Ñ¡”É•©½¥¹•ÑÉ…¹ÍÉ¥ÁÐè½µÁÉ•ÍÍ•(€€€€Œ¡•…€¬Ñ¡”±…ÍÐ•á¡…¹”Ù•É‰…Ñ¥´¸(€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰¡¥ÍÑ½Éä‰t€ôô}AIQ%1}=5AIMM}!€¬}AIQ%1}-}!%MQ=Iel´Èét(€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰¡¥ÍÑ½Éå}Ù•ÉÍ¥½¸‰t€ôô€à(€€€…ÍÍ•ÉÐÉ•µ½Ù•€ôô±•¸¡}AIQ%1}-}!%MQ=Id¤€´±•¸¡Í•ÍÍ¥½¹l‰¡¥ÍÑ½Éä‰t¤(()‘•˜Ñ•ÍÑ}½µÁÉ•ÍÍ}Í•ÍÍ¥½¹}¡¥ÍÑ½Éå}¡•É•}™…±±Í}‰…­}½¹}‘••¹•É…Ñ•}ÍÁ±¥Ð ¤è(€€€€ˆˆˆ½½µÁÉ•ÍÌ¡•É”Ý¥Ñ ­••Á}±…ÍÐ€øô•á¡…¹•ÌÁÉ½‘Õ•Ì…¸•µÁÑäÑ…¥°ƒŠP(€€€µÕÍÐ™…±°‰…¬Ñ¼™Õ±°½µÁÉ•ÍÍ¥½¸€¡Ý¡½±”¡¥ÍÑ½Éä°¹¼É•©½¥¹•Ñ…¥°¤¸ˆˆˆ(€€€½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±Ì€ômt(€€€…•¹Ð€ô}Á…ÉÑ¥…±}½µÁÉ•ÍÍ}…•¹Ð¡½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±Ì¤((€€€€Œ€Ðµ•ÍÍ…•Ì€ô€È•á¡…¹•Ìì­••Á}±…ÍÐôÔ±•…Ù•Ì¹½Ñ¡¥¹œÑ¼½µÁÉ•ÍÌ¸(€€€Í¡½ÉÑ}¡¥ÍÑ½Éä€ô}AIQ%1}-}!%MQ=IelèÑt(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡…•¹Ðõ…•¹Ð¤(€€€Í•ÍÍ¥½¹l‰¡¥ÍÑ½Éä‰t€ô±¥ÍÐ¡Í¡½ÉÑ}¡¥ÍÑ½Éä¤((€€€Í•ÉÙ•È¹}½µÁÉ•ÍÍ}Í•ÍÍ¥½¹}¡¥ÍÑ½Éä¡Í•ÍÍ¥½¸°€‰¡•É”€Ôˆ¤((€€€€Œ••¹•É…Ñ”ÍÁ±¥ÐƒŠH™Õ±°½µÁÉ•ÍÌ½˜Ñ¡”Ý¡½±”¡¥ÍÑ½Éä°™½ÕÍ}Ñ½Á¥Œõ9½¹”(€€€…ÍÍ•ÉÐ±•¸¡½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±Ì¤€ôô€Ä(€€€¡•…‘}Á…ÍÍ•°™½ÕÍ}Á…ÍÍ•€ô½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±ÍlÁt(€€€…ÍÍ•ÉÐ¡•…‘}Á…ÍÍ•€ôôÍ¡½ÉÑ}¡¥ÍÑ½Éä(€€€…ÍÍ•ÉÐ™½ÕÍ}Á…ÍÍ•¥Ì9½¹”(€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰¡¥ÍÑ½Éä‰t€ôô}AIQ%1}=5AIMM}!(()‘•˜Ñ•ÍÑ}½µÁÉ•ÍÍ}Í•ÍÍ¥½¹}¡¥ÍÑ½Éå}Á±…¥¹}™½ÕÍ}Ñ½Á¥}¹½Ñ}Á…ÉÍ•‘}…Í}Á…ÉÑ¥…° ¤è(€€€€ˆˆˆ½½µÁÉ•ÍÌµäÑ½Á¥ŒµÕÍÐÍÑ¥±°‘¼™Õ±°½µÁÉ•ÍÌÝ¥Ñ ™½ÕÍ}Ñ½Á¥ŒÍ•Ð¸ˆˆˆ(€€€½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±Ì€ômt(€€€…•¹Ð€ô}Á…ÉÑ¥…±}½µÁÉ•ÍÍ}…•¹Ð¡½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±Ì¤((€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡…•¹Ðõ…•¹Ð¤(€€€Í•ÍÍ¥½¹l‰¡¥ÍÑ½Éä‰t€ô±¥ÍÐ¡}AIQ%1}-}!%MQ=Id¤((€€€Í•ÉÙ•È¹}½µÁÉ•ÍÍ}Í•ÍÍ¥½¹}¡¥ÍÑ½Éä¡Í•ÍÍ¥½¸°€‰µäÑ½Á¥Œˆ¤((€€€…ÍÍ•ÉÐ±•¸¡½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±Ì¤€ôô€Ä(€€€¡•…‘}Á…ÍÍ•°™½ÕÍ}Á…ÍÍ•€ô½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±ÍlÁt(€€€…ÍÍ•ÉÐ¡•…‘}Á…ÍÍ•€ôô}AIQ%1}-}!%MQ=Id€€Œ™Õ±°¡¥ÍÑ½Éä°¹¼ÍÁ±¥Ð(€€€…ÍÍ•ÉÐ™½ÕÍ}Á…ÍÍ•€ôô€‰µäÑ½Á¥Œˆ(€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰¡¥ÍÑ½Éä‰t€ôô}AIQ%1}=5AIMM}!(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}½µÁÉ•ÍÍ}ÉÁ}¡½¹½ÉÍ}¡•É•}…ÉÕµ•¹Ð¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰I½ÕÑ”€Ä¼ÌèÑ¡”Í•ÍÍ¥½¸¹½µÁÉ•ÍÌIAµÕÍÐ¡½¹½È€¡•É”m9tœ¸ˆˆˆ(€€€½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±Ì€ômt(€€€…•¹Ð€ô}Á…ÉÑ¥…±}½µÁÉ•ÍÍ}…•¹Ð¡½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±Ì¤(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡…•¹Ðõ…•¹Ð¤(€€€Í•ÍÍ¥½¹l‰¡¥ÍÑ½Éä‰t€ô±¥ÍÐ¡}AIQ%1}-}!%MQ=Id¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ôÍ•ÍÍ¥½¸((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„€©}„°€¨©}­Üèì‰µ½‘•°ˆè€‰à‰ô¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Íå¹}Í•ÍÍ¥½¹}­•å}…™Ñ•É}½µÁÉ•ÍÌˆ°±…µ‰‘„€©„°€¨©­Üè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©…ÉÌè9½¹”¤((€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹½µÁÉ•ÍÌˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰™½ÕÍ}Ñ½Á¥Œˆè€‰¡•É”€Ä‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÍÑ…ÑÕÌ‰t€ôô€‰½µÁÉ•ÍÍ•ˆ(€€€…ÍÍ•ÉÐ±•¸¡½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±Ì¤€ôô€Ä(€€€¡•…‘}Á…ÍÍ•°™½ÕÍ}Á…ÍÍ•€ô½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±ÍlÁt(€€€…ÍÍ•ÉÐ¡•…‘}Á…ÍÍ•€ôô}AIQ%1}-}!%MQ=Ielè´Ét(€€€…ÍÍ•ÉÐ™½ÕÍ}Á…ÍÍ•¥Ì9½¹”(€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰¡¥ÍÑ½Éä‰t€ôô}AIQ%1}=5AIMM}!€¬}AIQ%1}-}!%MQ=Iel´Èét(()‘•˜Ñ•ÍÑ}½µµ…¹‘}‘¥ÍÁ…Ñ¡}½µÁÉ•ÍÍ}¡½¹½ÉÍ}¡•É•}…ÉÕµ•¹Ð¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰I½ÕÑ”€È¼Ìè½µµ…¹¹‘¥ÍÁ…Ñ €½½µÁÉ•ÍÌµÕÍÐ¡½¹½È€¡•É”m9tœ¸ˆˆˆ(€€€½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±Ì€ômt(€€€…•¹Ð€ô}Á…ÉÑ¥…±}½µÁÉ•ÍÍ}…•¹Ð¡½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±Ì¤(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡…•¹Ðõ…•¹Ð¤(€€€Í•ÍÍ¥½¹l‰¡¥ÍÑ½Éä‰t€ô±¥ÍÐ¡}AIQ%1}-}!%MQ=Id¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ôÍ•ÍÍ¥½¸((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}ÕÍ•Í}½µÁÕÑ•}¡½ÍÐˆ°±…µ‰‘„€©}„°€¨©}­Üè…±Í”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„€©}„°€¨©}­Üèì‰µ½‘•°ˆè€‰à‰ô¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Íå¹}Í•ÍÍ¥½¹}­•å}…™Ñ•É}½µÁÉ•ÍÌˆ°±…µ‰‘„€©„°€¨©­Üè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©…ÉÌè9½¹”¤((€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰½µµ…¹¹‘¥ÍÁ…Ñ ˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰¹…µ”ˆè€‰½µÁÉ•ÍÌˆ°€‰…Éœˆè€‰¡•É”€Ä‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÑåÁ”‰t€ôô€‰•á•Œˆ(€€€…ÍÍ•ÉÐ±•¸¡½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±Ì¤€ôô€Ä(€€€¡•…‘}Á…ÍÍ•°™½ÕÍ}Á…ÍÍ•€ô½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±ÍlÁt(€€€…ÍÍ•ÉÐ¡•…‘}Á…ÍÍ•€ôô}AIQ%1}-}!%MQ=Ielè´Ét(€€€…ÍÍ•ÉÐ™½ÕÍ}Á…ÍÍ•¥Ì9½¹”(€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰¡¥ÍÑ½Éä‰t€ôô}AIQ%1}=5AIMM}!€¬}AIQ%1}-}!%MQ=Iel´Èét(()‘•˜Ñ•ÍÑ}µ¥ÉÉ½É}Í±…Í¡}½µÁÉ•ÍÍ}¡½¹½ÉÍ}¡•É•}…ÉÕµ•¹Ð¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰I½ÕÑ”€Ì¼ÌèÑ¡”Í±…Í µ•á•Œµ¥ÉÉ½ÈµÕÍÐ¡½¹½È€¡•É”m9tœ¸ˆˆˆ(€€€½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±Ì€ômt(€€€…•¹Ð€ô}Á…ÉÑ¥…±}½µÁÉ•ÍÍ}…•¹Ð¡½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±Ì¤(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡…•¹Ðõ…•¹Ð¤(€€€Í•ÍÍ¥½¹l‰¡¥ÍÑ½Éä‰t€ô±¥ÍÐ¡}AIQ%1}-}!%MQ=Id¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„€©}„°€¨©}­Üèì‰µ½‘•°ˆè€‰à‰ô¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Íå¹}Í•ÍÍ¥½¹}­•å}…™Ñ•É}½µÁÉ•ÍÌˆ°±…µ‰‘„€©„°€¨©­Üè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©…ÉÌè9½¹”¤((€€€Ý…É¹¥¹œ€ôÍ•ÉÙ•È¹}µ¥ÉÉ½É}Í±…Í¡}Í¥‘•}•™™•ÑÌ ‰Í¥ˆ°Í•ÍÍ¥½¸°€ˆ½½µÁÉ•ÍÌ¡•É”€Äˆ¤((€€€…ÍÍ•ÉÐ€‰½µÁÉ•ÍÍ•èˆ¥¸Ý…É¹¥¹œ(€€€…ÍÍ•ÉÐ±•¸¡½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±Ì¤€ôô€Ä(€€€¡•…‘}Á…ÍÍ•°™½ÕÍ}Á…ÍÍ•€ô½µÁÉ•ÍÍ}½¹Ñ•áÑ}…±±ÍlÁt(€€€…ÍÍ•ÉÐ¡•…‘}Á…ÍÍ•€ôô}AIQ%1}-}!%MQ=Ielè´Ét(€€€…ÍÍ•ÉÐ™½ÕÍ}Á…ÍÍ•¥Ì9½¹”(€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰¡¥ÍÑ½Éä‰t€ôô}AIQ%1}=5AIMM}!€¬}AIQ%1}-}!%MQ=Iel´Èét(((Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(ŒÍ•ÍÍ¥½¸¹É•…Ñ”€¼Í•ÍÍ¥½¸¹±½Í”É…”è™…ÍÐ€½¹•Ü¡ÕÉ¸µÕÍÐ¹½Ð½ÉÁ¡…¸Ñ¡”(Œ±½‰…°…ÁÁÉ½Ù…°µ¹½Ñ¥™äÉ•¥ÍÑÉ…Ñ¥½¸¸€¡M±…Í Ý½É­•ÉÌ…É”¹¼±½¹•ÈÁÉ”µÝ…Éµ•(Œ‰äÑ¡”‰Õ¥±Ñ¡É•…ƒŠPÍ±…Í ¹•á•ŒÍÁ…Ý¹ÌÑ¡•´½¸‘•µ…¹ƒŠPÍ¼Ñ¡”‰Õ¥±Ñ¡É•…(ŒµÕÍÐ1M<¹•Ù•È½¹ÍÑÉÕÐ½¹”¡•É”¸¤(Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(()ÁåÑ•ÍÐ¹µ…É¬¹É•…±}…•¹Ñ}ÁÉ•Ý…É´)‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}É•…Ñ•}±½Í•}É…•}‘½•Í}¹½Ñ}½ÉÁ¡…¹}Ý½É­•È¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰I•É•ÍÍ¥½¸Õ…Éè¥˜Í•ÍÍ¥½¸¹±½Í”ÉÕ¹ÌÝ¡¥±”Í•ÍÍ¥½¸¹É•…Ñ”Ì(€€€}‰Õ¥±Ñ¡É•…¥ÌÍÑ¥±°½¹ÍÑÉÕÑ¥¹œÑ¡”…•¹Ð°Ñ¡”‰Õ¥±Ñ¡É•…(€€€µÕÍÐ‘•Ñ•ÐÑ¡”½ÉÁ¡…¸…¹Õ¹É•¥ÍÑ•ÈÑ¡”¹½Ñ¥™äÉ•¥ÍÑÉ…Ñ¥½¸¥ÐÌ(€€€…‰½ÕÐÑ¼¥¹ÍÑ…±°¸€%ÐµÕÍÐ…±Í¼¹•Ù•ÈÁÉ”µÝ…É´„Í±…Í Ý½É­•È€¡•… (€€€Ý½É­•È™½É­ÌÑ¡”™Õ±°ÍÑ‘¥¼5@™±••ÐìÍÁ…Ý¸¥Ì½¸µ‘•µ…¹¥¸(€€€Í±…Í ¹•á•Œ¤ƒŠP„Ý½É­•È½¹ÍÑÉÕÑ•¡•É”Ý½Õ±‰”„É•É•ÍÍ¥½¸¸ˆˆˆ(€€€¥µÁ½ÉÐÑ¡É•…‘¥¹œ((€€€É•…Ñ•‘}Ý½É­•ÉÌè±¥ÍÑmÍÑÉt€ômt(€€€±½Í•‘}Ý½É­•ÉÌè±¥ÍÑmÍÑÉt€ômt(€€€Õ¹É•¥ÍÑ•É•‘}­•åÌè±¥ÍÑmÍÑÉt€ômt((€€€±…ÍÌ}…­•]½É­•Èè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°­•ä°µ½‘•°°ÁÉ½™¥±•}¡½µ”õ9½¹”¤è(€€€€€€€€€€€Í•±˜¹­•ä€ô­•ä(€€€€€€€€€€€Í•±˜¹}±½Í•€ô…±Í”(€€€€€€€€€€€É•…Ñ•‘}Ý½É­•ÉÌ¹…ÁÁ•¹¡­•ä¤((€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹}±½Í•€ôQÉÕ”(€€€€€€€€€€€±½Í•‘}Ý½É­•ÉÌ¹…ÁÁ•¹¡Í•±˜¹­•ä¤((€€€±…ÍÌ}…­••¹Ðè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹µ½‘•°€ô€‰àˆ(€€€€€€€€€€€Í•±˜¹ÁÉ½Ù¥‘•È€ô€‰½Á•¹É½ÕÑ•Èˆ(€€€€€€€€€€€Í•±˜¹‰…Í•}ÕÉ°€ô€ˆˆ(€€€€€€€€€€€Í•±˜¹…Á¥}­•ä€ô€ˆˆ((€€€€Œ5…­”}‰Õ¥±‰±½¬Õ¹Ñ¥°Ý”É•±•…Í”¥ÐƒŠPÍ¥µÕ±…Ñ•ÌÍ±½Ü…•¹Ð¥¹¥Ð¸(€€€€Œ±Í¼Í¥¹…°Ý¡•¸}‰Õ¥±…ÑÕ…±±äÉ•…¡•Ì}µ…­•}…•¹ÐÍ¼Ñ¡”Ñ•ÍÐ(€€€€Œ…¸±½Í”Ñ¡”Í•ÍÍ¥½¸…ÐÑ¡”É¥¡Ðµ½µ•¹ÐèÍ•ÍÍ¥½¸¹É•…Ñ”¹½Ü(€€€€Œ‘•™•ÉÌ}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±‰•¡¥¹„€ÔÁµÌÑ¥µ•È€¡Í•”Ñ¡”(€€€€Œ}‘•™•ÉÉ•‘}‰Õ¥±‘€Á…Ñ ¥¸µ•Ñ¡½ ‰Í•ÍÍ¥½¸¹É•…Ñ”ˆ¤¤°Í¼±½Í¥¹œ(€€€€Œ‰•™½É”Ñ¡”‰Õ¥±Ñ¡É•…¡…Ì•Ù•¸ÍÑ…ÉÑ•Ý½Õ±Í­¥ÀÑ¡”½ÉÁ¡…¸(€€€€Œ‘•Ñ•Ñ¥½¸•¹Ñ¥É•±ä…¹Ñ¡”Ñ•ÍÐÝ½Õ±É…”„¹½¸µ•Ù•¹Ð¸(€€€‰Õ¥±‘}ÍÑ…ÉÑ•€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤(€€€É•±•…Í•}‰Õ¥±€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤(€€€‰Õ¥±‘}•¹Ñ•É•€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤((€€€‘•˜}Í±½Ý}µ…­•}…•¹Ð¡Í¥°­•ä°Í•ÍÍ¥½¹}¥õ9½¹”°Í•ÍÍ¥½¹}‘ˆõ9½¹”°€¨©}­Ý…ÉÌ¤è(€€€€€€€‰Õ¥±‘}ÍÑ…ÉÑ•¹Í•Ð ¤(€€€€€€€‰Õ¥±‘}•¹Ñ•É•¹Í•Ð ¤(€€€€€€€É•±•…Í•}‰Õ¥±¹Ý…¥Ð¡Ñ¥µ•½ÕÐôÌ¸À¤(€€€€€€€É•ÑÕÉ¸}…­••¹Ð ¤((€€€€ŒMÑÕˆ•Ù•ÉåÑ¡¥¹œ}‰Õ¥±Ñ½Õ¡•Ì(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}µ…­•}…•¹Ðˆ°}Í±½Ý}µ…­•}…•¹Ð¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}M±…Í¡]½É­•Èˆ°}…­•]½É­•È¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°(€€€€€€€€‰}•Ñ}‘ˆˆ°(€€€€€€€±…µ‰‘„èÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡É•…Ñ•}Í•ÍÍ¥½¸õ±…µ‰‘„€©„°€¨©­Üè9½¹”¤°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„}„°€©„Èèì‰µ½‘•°ˆè€‰à‰ô¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÁÉ½‰•}É•‘•¹Ñ¥…±Ìˆ°±…µ‰‘„}„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Ý¥É•}…±±‰…­Ìˆ°±…µ‰‘„}Í¥è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„°€¨©­Üè9½¹”¤((€€€€ŒM¡¥´É•¥ÍÑ•È½Õ¹É•¥ÍÑ•ÈÑ¼½‰Í•ÉÙ”±•…­Ì(€€€¥µÁ½ÉÐÑ½½±Ì¹…ÁÁÉ½Ù…°…Ì}…ÁÁÉ½Ù…°((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡}…ÁÁÉ½Ù…°°€‰É•¥ÍÑ•É}…Ñ•Ý…å}¹½Ñ¥™äˆ°±…µ‰‘„­•ä°ˆè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€}…ÁÁÉ½Ù…°°(€€€€€€€€‰Õ¹É•¥ÍÑ•É}…Ñ•Ý…å}¹½Ñ¥™äˆ°(€€€€€€€±…µ‰‘„­•äèÕ¹É•¥ÍÑ•É•‘}­•åÌ¹…ÁÁ•¹¡­•ä¤°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡}…ÁÁÉ½Ù…°°€‰±½…‘}Á•Éµ…¹•¹Ñ}…±±½Ý±¥ÍÐˆ°±…µ‰‘„è9½¹”¤((€€€€ŒMÑ…ÉÐèÍ•ÍÍ¥½¸¹É•…Ñ”ÍÁ…Ý¹Ì}‰Õ¥±Ñ¡É•…°É•ÑÕÉ¹ÌÍå¹¡É½¹½ÕÍ±ä(€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹É•…Ñ”ˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì‰½±Ìˆè€àÁô°(€€€€€€€ô(€€€€¤(€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÉ•ÍÀ¹•Ð •ÉÉ½Èœ¥ôˆ(€€€Í¥€ôÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰t(€€€½Ý¹}­•ä€ôÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÍÑ½É•‘}Í•ÍÍ¥½¹}¥‰t(€€€…ÍÍ•ÉÐ‰Õ¥±‘}•¹Ñ•É•¹Ý…¥Ð¡Ñ¥µ•½ÕÐôÄ¸À¤°€‰‘•™•ÉÉ•‰Õ¥±‘¥¹½ÐÍÑ…ÉÐˆ((€€€€Œ]…¥ÐÕ¹Ñ¥°Ñ¡”€¡‘•™•ÉÉ•¤‰Õ¥±Ñ¡É•…¡…Ì…ÑÕ…±±ä•¹Ñ•É•(€€€€Œ}µ…­•}…•¹ÐƒŠP½Ñ¡•ÉÝ¥Í”Í•ÍÍ¥½¸¹±½Í”Á½ÁÌ}Í•ÍÍ¥½¹ÍmÍ¥‘t‰•™½É”(€€€€Œ}‰Õ¥±•Ù•ÈÉÕ¹Ì°}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±¹•Ù•È…±±Ì}‰Õ¥±°…¹Ý”(€€€€Œ¹•Ù•È•á•É¥Í”Ñ¡”½ÉÁ¡…¸µ±•…¹ÕÀÁ…Ñ ¸(€€€…ÍÍ•ÉÐ‰Õ¥±‘}ÍÑ…ÉÑ•¹Ý…¥Ð¡Ñ¥µ•½ÕÐôÈ¸À¤°€‰‰Õ¥±Ñ¡É•…¹•Ù•È•¹Ñ•É•}µ…­•}…•¹Ðˆ((€€€€Œ	Õ¥±Ñ¡É•…¥Ì‰±½­•¥¸}Í±½Ý}µ…­•}…•¹Ð¸€±½Í”Ñ¡”Í•ÍÍ¥½¸(€€€€Œ9=\ƒŠPÑ¡¥ÌÁ½ÁÌ}Í•ÍÍ¥½¹ÍmÍ¥‘t‰•™½É”}‰Õ¥±…¸¥¹ÍÑ…±°Ñ¡”(€€€€ŒÝ½É­•È½¹½Ñ¥™ä¸(€€€±½Í•}É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÈˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹±½Í”ˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆèÍ¥‘ô°(€€€€€€€ô(€€€€¤(€€€…ÍÍ•ÉÐ±½Í•}É•ÍÀ¹•Ð ‰É•ÍÕ±Ðˆ°íô¤¹•Ð ‰±½Í•ˆ¤¥ÌQÉÕ”((€€€€ŒÐÑ¡¥ÌÁ½¥¹ÐÍ•ÍÍ¥½¸¹±½Í”Í…ÜÍ±…Í¡}Ý½É­•Èõ9½¹”€¡¹•Ù•È•…•É±ä(€€€€Œ¥¹ÍÑ…±±•¤Í¼¥Ð¡…¹½Ñ¡¥¹œÑ¼±½Í”¸€I•±•…Í”Ñ¡”‰Õ¥±Ñ¡É•…(€€€€Œ…¹±•Ð¥Ð™¥¹¥Í ƒŠP¥ÐÍ¡½Õ±‘•Ñ•ÐÑ¡”½ÉÁ¡…¸…¹Õ¹É•¥ÍÑ•È(€€€€ŒÑ¡”¹½Ñ¥™ä°Ý¥Ñ¡½ÕÐ•Ù•È¡…Ù¥¹œ½¹ÍÑÉÕÑ•„Ý½É­•È¸(€€€É•±•…Í•}‰Õ¥±¹Í•Ð ¤((€€€€Œ¥Ù”Ñ¡”‰Õ¥±Ñ¡É•…„µ½µ•¹ÐÑ¼ÉÕ¸Ñ¡É½Õ ¥ÑÌ™¥¹…±±ä¸(€€€™½È|¥¸É…¹” ÄÀÀ¤è(€€€€€€€¥˜½Ý¹}­•ä¥¸Õ¹É•¥ÍÑ•É•‘}­•åÌè(€€€€€€€€€€€‰É•…¬(€€€€€€€¥µÁ½ÉÐÑ¥µ”((€€€€€€€Ñ¥µ”¹Í±••À À¸ÀÈ¤((€€€…ÍÍ•ÉÐÉ•…Ñ•‘}Ý½É­•ÉÌ€ôômt°€ (€€€€€€€˜‰‰Õ¥±Ñ¡É•…ÁÉ”µÝ…Éµ•„Í±…Í Ý½É­•È€¡ÍÁ…Ý¸µÕÍÐÍÑ…ä½¸µ‘•µ…¹€ˆ(€€€€€€€˜‰¥¸Í±…Í ¹•á•Œ¤ƒŠPÉ•…Ñ•‘}Ý½É­•ÉÌõíÉ•…Ñ•‘}Ý½É­•ÉÍôˆ(€€€€¤(€€€€Œ9½Ñ¥™äµ…ä‰”Õ¹É•¥ÍÑ•É•‰ä‰½Ñ Í•ÍÍ¥½¸¹±½Í”€¡Õ¹½¹‘¥Ñ¥½¹…°¤(€€€€Œ…¹Ñ¡”½ÉÁ¡…¸µ±•…¹ÕÀÁ…Ñ ìÑ¡”­•äÕ…É…¹Ñ•”¥ÌÑ¡…ÐQ!%LÍ•ÍÍ¥½¸Ì(€€€€Œ­•ä•ÑÌÕ¹É•¥ÍÑ•É•€¡…¹äÁÉ¥½È±½Í”…±É•…‘äÁ½ÁÁ•Ñ¡”…±±‰…¬ìÑ¡”(€€€€Œ‘ÕÁ±¥…Ñ”¥Ì„¹¼µ½À¤¸5…Ñ ½¸½ÕÈ½Ý¸­•ä°¹½ÐÑ¡”±½‰…°½Õ¹ÐèÑ¡”(€€€€ŒÉ•¥ÍÑÉä¥ÌÁÉ½•ÍÌµÝ¥‘”…¹„±•…­•}‰Õ¥±Ñ¡É•…™É½´…¹½Ñ¡•È(€€€€ŒÍ•ÍÍ¥½¸¹É•…Ñ”Ñ•ÍÐ…¸…ÁÁ•¹„™½É•¥¸­•ä¡•É”…¹™…±Í•±äÍ…Ñ¥Í™ä(€€€€Œ„‰…É”€øô€Å€¸(€€€…ÍÍ•ÉÐ½Ý¹}­•ä¥¸Õ¹É•¥ÍÑ•É•‘}­•åÌ°€ (€€€€€€€˜‰½ÉÁ¡…¸¹½Ñ¥™äÉ•¥ÍÑÉ…Ñ¥½¸Ý…Ì¹½ÐÕ¹É•¥ÍÑ•É•ƒŠP€ˆ(€€€€€€€˜‰í½Ý¹}­•åô¹½Ð¥¸Õ¹É•¥ÍÑ•É•‘}­•åÌõíÕ¹É•¥ÍÑ•É•‘}­•åÍôˆ(€€€€¤(()ÁåÑ•ÍÐ¹µ…É¬¹É•…±}…•¹Ñ}ÁÉ•Ý…É´)‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}É•…Ñ•}¹½}É…•}­••ÁÍ}Ý½É­•É}…±¥Ù”¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰I•É•ÍÍ¥½¸Õ…ÉèÝ¡•¸Í•ÍÍ¥½¸¹±½Í”‘½•Ì9=PÉ…”°Ñ¡”‰Õ¥±(€€€Ñ¡É•…µÕÍÐ¥¹ÍÑ…±°Ñ¡”¹½Ñ¥™ä¹½Éµ…±±ä…¹±•…Ù”¥Ð…±½¹”€¡¹¼(€€€½Ù•Èµ•…•È±•…¹ÕÀ¤ƒŠP…¹µÕÍÐ¹½ÐÁÉ”µÝ…É´„Í±…Í Ý½É­•È€¡ÍÁ…Ý¸(€€€¥Ì½¸µ‘•µ…¹¥¸Í±…Í ¹•á•Œ¤¸ˆˆˆ(€€€±½Í•‘}Ý½É­•ÉÌè±¥ÍÑmÍÑÉt€ômt(€€€Õ¹É•¥ÍÑ•É•‘}­•åÌè±¥ÍÑmÍÑÉt€ômt((€€€±…ÍÌ}…­•]½É­•Èè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°­•ä°µ½‘•°°ÁÉ½™¥±•}¡½µ”õ9½¹”¤è(€€€€€€€€€€€Í•±˜¹­•ä€ô­•ä((€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€±½Í•‘}Ý½É­•ÉÌ¹…ÁÁ•¹¡Í•±˜¹­•ä¤((€€€±…ÍÌ}…­••¹Ðè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹µ½‘•°€ô€‰àˆ(€€€€€€€€€€€Í•±˜¹ÁÉ½Ù¥‘•È€ô€‰½Á•¹É½ÕÑ•Èˆ(€€€€€€€€€€€Í•±˜¹‰…Í•}ÕÉ°€ô€ˆˆ(€€€€€€€€€€€Í•±˜¹…Á¥}­•ä€ô€ˆˆ((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}µ…­•}…•¹Ðˆ°±…µ‰‘„Í¥°­•ä°Í•ÍÍ¥½¹}‘ˆõ9½¹”°€¨©}­Ý…ÉÌè}…­••¹Ð ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}M±…Í¡]½É­•Èˆ°}…­•]½É­•È¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°(€€€€€€€€‰}•Ñ}‘ˆˆ°(€€€€€€€±…µ‰‘„èÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡É•…Ñ•}Í•ÍÍ¥½¸õ±…µ‰‘„€©„°€¨©­Üè9½¹”¤°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„}„°€©„Èèì‰µ½‘•°ˆè€‰à‰ô¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÁÉ½‰•}É•‘•¹Ñ¥…±Ìˆ°±…µ‰‘„}„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Ý¥É•}…±±‰…­Ìˆ°±…µ‰‘„}Í¥è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„°€¨©­Üè9½¹”¤((€€€¥µÁ½ÉÐÑ½½±Ì¹…ÁÁÉ½Ù…°…Ì}…ÁÁÉ½Ù…°((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡}…ÁÁÉ½Ù…°°€‰É•¥ÍÑ•É}…Ñ•Ý…å}¹½Ñ¥™äˆ°±…µ‰‘„­•ä°ˆè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€}…ÁÁÉ½Ù…°°(€€€€€€€€‰Õ¹É•¥ÍÑ•É}…Ñ•Ý…å}¹½Ñ¥™äˆ°(€€€€€€€±…µ‰‘„­•äèÕ¹É•¥ÍÑ•É•‘}­•åÌ¹…ÁÁ•¹¡­•ä¤°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡}…ÁÁÉ½Ù…°°€‰±½…‘}Á•Éµ…¹•¹Ñ}…±±½Ý±¥ÍÐˆ°±…µ‰‘„è9½¹”¤((€€€€Œ%Í½±…Ñ”™É½´Í¥‰±¥¹œµÑ•ÍÐ±•…­…”è‘…•µ½¸‰Õ¥±Ñ¡É•…‘Ì™É½´ÁÉ¥½È(€€€€ŒÍ•ÍÍ¥½¸¹É•…Ñ”Ñ•ÍÑÌ¥¸Ñ¡”Í…µ”Í¡…ÉÁÉ½•ÍÌµÕÑ…Ñ”Ñ¡”Í¡…É•(€€€€ŒÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Í€‘¥ÐÕ¹‘•È}Í•ÍÍ¥½¹Í}±½­€…¹…¸É•Á±…”½Á½À(€€€€Œ•¹ÑÉ¥•Ìµ¥µÉÕ¸°Ý¡¥ Ý½Õ±™±¥ÀÑ¡¥Ì‰Õ¥±Ñ¡É•…ÌÉ•Á±…•‘€¡•¬(€€€€ŒÑ¼QÉÕ”…¹ÑÉ¥•È„ÍÁÕÉ¥½ÕÌÕ¹É•¥ÍÑ•È¸M¹…ÁÍ¡½Ð°±•…È°…¹É•ÍÑ½É”(€€€€ŒÍ¼Ñ¡¥ÌÑ•ÍÐÍ••Ì½¹±ä¥ÑÌ½Ý¸Í•ÍÍ¥½¸É•…É‘±•ÍÌ½˜Í¡…É½µÁ½Í¥Ñ¥½¸¸(€€€}Í…Ù•‘}Í•ÍÍ¥½¹Ì€ô‘¥Ð¡Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤((€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹É•…Ñ”ˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰½±Ìˆè€àÁô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€Í¥€ôÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰t((€€€€€€€€Œ]…¥Ð™½ÈÑ¡”‰Õ¥±Ñ¼™¥¹¥Í €¡É•…‘ä•Ù•¹Ð¥¹Í¥‘”Í•ÍÍ¥½¸‘¥Ð¤¸(€€€€€€€Í•ÍÍ¥½¸€ôÍ•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÍ¥‘t(€€€€€€€‰Õ¥±Ð€ôÍ•ÍÍ¥½¹l‰…•¹Ñ}É•…‘ä‰t¹Ý…¥Ð¡Ñ¥µ•½ÕÐôÄÀ¸À¤(€€€€€€€…ÍÍ•ÉÐ‰Õ¥±Ð°€‰…•¹Ð‰Õ¥±‘¥¹½Ð½µÁ±•Ñ”Ý¥Ñ¡¥¸Ñ¥µ•½ÕÐˆ((€€€€€€€€Œ	Õ¥±™¥¹¥Í¡•Ý¥Ñ¡½ÕÐ„±½Í”É…”ƒŠP¹½Ñ¡¥¹œÍ¡½Õ±¡…Ù”‰••¸(€€€€€€€€Œ±•…¹•ÕÀ‰äÑ¡”½ÉÁ¡…¸¡•¬¸€M½Á”Ñ¡”…ÍÍ•ÉÑ¥½¹ÌÑ¼Q!%L(€€€€€€€€ŒÑ•ÍÐÌ½Ý¸Í•ÍÍ¥½¹}­•äè„‘…•µ½¸‰Õ¥±Ñ¡É•…±•…­•™É½´„ÁÉ¥½È(€€€€€€€€ŒÍ•ÍÍ¥½¸¹É•…Ñ”Ñ•ÍÐ¥¸Ñ¡”Í…µ”Í¡…ÉÁÉ½•ÍÌ…¸™¥É”±½Í”¼(€€€€€€€€ŒÕ¹É•¥ÍÑ•È……¥¹ÍÐ¥ÑÌ½Ý¸€¡™½É•¥¸¤­•ä…™Ñ•ÈÝ”Ù”Á…Ñ¡•Ñ¡”(€€€€€€€€Œ±½‰…°¡½½­Ì°Á½±±ÕÑ¥¹œÑ¡•Í”±¥ÍÑÌ¸€¥±Ñ•É¥¹œ‰äÑ¡¥ÌÍ•ÍÍ¥½¸Ì(€€€€€€€€Œ­•ä­••ÁÌÑ¡”É•É•ÍÍ¥½¸¥¹Ñ•¹Ð€¡Ñ¡¥ÌÍ•ÍÍ¥½¸ÌÝ½É­•È½¹½Ñ¥™äµÕÍÐ(€€€€€€€€ŒÍÕÉÙ¥Ù”¤Ý¡¥±”µ…­¥¹œÑ¡”Ñ•ÍÐ¥µµÕ¹”Ñ¼Í¡…É½µÁ½Í¥Ñ¥½¸¸(€€€€€€€€Œ€¡™±…­äÕ¹‘•È€µ¨€àè™½É•¥¸­•ä”¹œ¸€ÈÀÈØÀØÈå|ÈÄÀÈÀá}Ñ˜ÔÐÔ¤(€€€€€€€½Ý¹}­•ä€ôÍ•ÍÍ¥½¹l‰Í•ÍÍ¥½¹}­•ä‰t(€€€€€€€½Ý¹}±½Í•€ôm¬™½È¬¥¸±½Í•‘}Ý½É­•ÉÌ¥˜¬€ôô½Ý¹}­•åt(€€€€€€€½Ý¹}Õ¹É•¥ÍÑ•É•€ôm¬™½È¬¥¸Õ¹É•¥ÍÑ•É•‘}­•åÌ¥˜¬€ôô½Ý¹}­•åt(€€€€€€€…ÍÍ•ÉÐ€ (€€€€€€€€€€€½Ý¹}±½Í•€ôômt(€€€€€€€€¤°˜‰‰Õ¥±Ñ¡É•…±½Í•¥ÑÌ½Ý¸Ý½É­•È‘•ÍÁ¥Ñ”¹¼É…”èí½Ý¹}±½Í•‘ôˆ(€€€€€€€…ÍÍ•ÉÐ€ (€€€€€€€€€€€½Ý¹}Õ¹É•¥ÍÑ•É•€ôômt(€€€€€€€€¤°˜‰‰Õ¥±Ñ¡É•…Õ¹É•¥ÍÑ•É•¥ÑÌ½Ý¸¹½Ñ¥™ä‘•ÍÁ¥Ñ”¹¼É…”èí½Ý¹}Õ¹É•¥ÍÑ•É•‘ôˆ((€€€€€€€€Œ9¼ÁÉ”µÝ…Éµ•Ý½É­•ÈèÍ±…Í ¹•á•ŒÍÁ…Ý¹Ì½¸‘•µ…¹°Í¼„™É•Í (€€€€€€€€ŒÍ•ÍÍ¥½¸Ñ¡…Ð¡…Í¸ÐÉÕ¸„Ý½É­•ÈµÉ½ÕÑ•½µµ…¹…ÉÉ¥•Ì9½¹”¸(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¸¹•Ð ‰Í±…Í¡}Ý½É­•Èˆ¤¥Ì9½¹”(€€€™¥¹…±±äè(€€€€€€€€Œ±•…¹ÕÀ€¬É•ÍÑ½É”Í¥‰±¥¹œÍ•ÍÍ¥½¹ÌÝ”Í¹…ÁÍ¡½ÑÑ•¸(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹ÕÁ‘…Ñ”¡}Í…Ù•‘}Í•ÍÍ¥½¹Ì¤(()‘•˜Ñ•ÍÑ}•Ñ}‘‰}‘•É…‘•Í}±•…¹±å}Ý¡•¹}Í•ÍÍ¥½¹‘‰}¥¹¥Ñ}™…¥±Ì¡µ½¹­•åÁ…Ñ ¤è(€€€™…­•}µ½€ôÑåÁ•Ì¹5½‘Õ±•QåÁ” ‰¡•Éµ•Í}ÍÑ…Ñ”ˆ¤((€€€±…ÍÌ}	É½­•¹M•ÍÍ¥½¹è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜¤è(€€€€€€€€€€€É…¥Í”IÕ¹Ñ¥µ•ÉÉ½È ‰±½­¥¹œÁÉ½Ñ½½°ˆ¤((€€€™…­•}µ½¹M•ÍÍ¥½¹€ô}	É½­•¹M•ÍÍ¥½¹((€€€‘•˜}‰É½­•¹}Í¡…É•¡}‘‰}Á…Ñ õ9½¹”¤è(€€€€€€€É…¥Í”IÕ¹Ñ¥µ•ÉÉ½È ‰±½­¥¹œÁÉ½Ñ½½°ˆ¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ¥Ñ•´¡ÍåÌ¹µ½‘Õ±•Ì°€‰¡•Éµ•Í}ÍÑ…Ñ”ˆ°™…­•}µ½¤(€€€™…­•}É•¥ÍÑÉä€ôÑåÁ•Ì¹5½‘Õ±•QåÁ” ‰¡•Éµ•Í}ÍÑ…Ñ•}É•¥ÍÑÉäˆ¤(€€€™…­•}É•¥ÍÑÉä¹…ÅÕ¥É”€ô}‰É½­•¹}Í¡…É•(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ¥Ñ•´¡ÍåÌ¹µ½‘Õ±•Ì°€‰¡•Éµ•Í}ÍÑ…Ñ•}É•¥ÍÑÉäˆ°™…­•}É•¥ÍÑÉä¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}‘ˆˆ°9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}‘‰}•ÉÉ½Èˆ°9½¹”¤((€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}•Ñ}‘ˆ ¤¥Ì9½¹”(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}‘‰}•ÉÉ½È€ôô€‰±½­¥¹œÁÉ½Ñ½½°ˆ(()‘•˜Ñ•ÍÑ}•¹ÍÕÉ•}Í•ÍÍ¥½¹}‘‰}É½Ý}™…±Í•}Ý¡•¹}ÍÑ½É•}Õ¹…Ù…¥±…‰±”¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰MÑ½É”Õ¹…Ù…¥±…‰±”ƒŠH…±Í”°Í¼ÁÉ½µÁÐ¹ÍÕ‰µ¥Ð…¸™…¥°Ñ¡”Í•¹±½Õ‘±ä(€€€¥¹ÍÑ•…½˜ÍÑÉ•…µ¥¹œ¥¹Ñ¼„ÍÑ½É”Ñ¡…ÐÝ¥±°¹•Ù•ÈÍ…Ù”¥Ð€ ŒäàäÈÐ¤¸ˆˆˆ(€€€™…­•}µ½€ôÑåÁ•Ì¹5½‘Õ±•QåÁ” ‰¡•Éµ•Í}ÍÑ…Ñ”ˆ¤((€€€±…ÍÌ}	É½­•¹M•ÍÍ¥½¹è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜¤è(€€€€€€€€€€€É…¥Í”IÕ¹Ñ¥µ•ÉÉ½È ‰ÕÑ˜´à‰½½´ˆ¤((€€€™…­•}µ½¹M•ÍÍ¥½¹€ô}	É½­•¹M•ÍÍ¥½¹((€€€‘•˜}‰É½­•¹}Í¡…É•¡}‘‰}Á…Ñ õ9½¹”¤è(€€€€€€€É…¥Í”IÕ¹Ñ¥µ•ÉÉ½È ‰ÕÑ˜´à‰½½´ˆ¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ¥Ñ•´¡ÍåÌ¹µ½‘Õ±•Ì°€‰¡•Éµ•Í}ÍÑ…Ñ”ˆ°™…­•}µ½¤(€€€™…­•}É•¥ÍÑÉä€ôÑåÁ•Ì¹5½‘Õ±•QåÁ” ‰¡•Éµ•Í}ÍÑ…Ñ•}É•¥ÍÑÉäˆ¤(€€€™…­•}É•¥ÍÑÉä¹…ÅÕ¥É”€ô}‰É½­•¹}Í¡…É•(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ¥Ñ•´¡ÍåÌ¹µ½‘Õ±•Ì°€‰¡•Éµ•Í}ÍÑ…Ñ•}É•¥ÍÑÉäˆ°™…­•}É•¥ÍÑÉä¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}‘ˆˆ°9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}‘‰}•ÉÉ½Èˆ°9½¹”¤((€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}•¹ÍÕÉ•}Í•ÍÍ¥½¹}‘‰}É½Ü¡ì‰Í•ÍÍ¥½¹}­•äˆè€‰¬Ä‰ô¤¥Ì…±Í”(()‘•˜Ñ•ÍÑ}•¹ÍÕÉ•}Í•ÍÍ¥½¹}‘‰}É½Ý}ÑÉÕ•}Ý¡•¹}É½Ý}Á•ÉÍ¥ÍÑ•¡µ½¹­•åÁ…Ñ ¤è(€€€É•…Ñ•€ômt((€€€±…ÍÌ}…­•è(€€€€€€€‘•˜É•…Ñ•}Í•ÍÍ¥½¸¡Í•±˜°­•ä°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€É•…Ñ•¹…ÁÁ•¹¡­•ä¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è}…­• ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}É•Í½±Ù•}µ½‘•°ˆ°±…µ‰‘„è€‰Ñ•ÍÐµµ½‘•°ˆ¤((€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}•¹ÍÕÉ•}Í•ÍÍ¥½¹}‘‰}É½Ü¡ì‰Í•ÍÍ¥½¹}­•äˆè€‰¬Äˆ°€‰Ýˆè€ˆ½ÑµÀ‰ô¤¥ÌQÉÕ”(€€€…ÍÍ•ÉÐÉ•…Ñ•€ôôl‰¬Ä‰t(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}™…¥±Í}±½Õ‘±å}Ý¡•¹}ÍÑ½É•}Õ¹…Ù…¥±…‰±”¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰Í•¹Ý¥Ñ ¹¼Á•ÉÍ¥ÍÑ…‰±”ÍÑ½É”µÕÍÐ™…¥°Ñ¡”IAÝ¥Ñ „É•…°•ÉÉ½È(€€€€¡‘•Í­Ñ½Àµ…ÁÌ¥ÐÑ¼„Ñ½…ÍÐ¤¥¹ÍÑ•…½˜ÍÑÉ•…µ¥¹œÑ¡”µ•ÍÍ…”¥¹Ñ¼„(€€€ÍÑ½É”Ñ¡…ÐÝ¥±°¹•Ù•ÈÍ…Ù”¥Ð€ ŒäàäÈÐ¤¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}™œˆ°±…µ‰‘„èì‰‘…Í¡‰½…Éˆèíõô¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}‘‰}•ÉÉ½Èˆ°€‰ÕÑ˜´à‘•½‘”™…¥±ÕÉ”ˆ¤((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰±½ÍÐµÍ¥‰t€ô}Í•ÍÍ¥½¸ ¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€‰±½ÍÐˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰±½ÍÐµÍ¥ˆ°€‰Ñ•áÐˆè€‰Ý¥±°Ù…¹¥Í ‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰±½ÍÐµÍ¥ˆ°9½¹”¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÔÀÜÈ(€€€µÍœ€ôÉ•ÍÁl‰•ÉÉ½È‰ul‰µ•ÍÍ…”‰t(€€€…ÍÍ•ÉÐ€‰ÕÑ˜´à‘•½‘”™…¥±ÕÉ”ˆ¹½Ð¥¸µÍœ€€ŒÉ…Ü…ÕÍ”É¥‘•Ì‘…Ñ„¹‘•Ñ…¥±Í€°¹•Ù•ÈÑ¡”±•…(€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰‘…Ñ„‰ul‰½‘”‰t€ôô€‰ÍÑ½É…•}Õ¹…Ù…¥±…‰±”ˆ(€€€…ÍÍ•ÉÐ€‰ÕÑ˜´à‘•½‘”™…¥±ÕÉ”ˆ¥¸É•ÍÁl‰•ÉÉ½È‰ul‰‘…Ñ„‰ul‰‘•Ñ…¥±Ì‰t(()ÁåÑ•ÍÐ¹µ…É¬¹Á…É…µ•ÑÉ¥é” (€€€€ ‰Á•ÉÍ¥ÍÑ}µ½‘”ˆ°€‰•áÁ•Ñ•‘}½‘”ˆ¤°(€€€l(€€€€€€€€ ‰‘¥Í­}™Õ±°ˆ°€ÔÀÜÀ¤°(€€€€€€€€ ‰•¹•É¥}Á•ÉÍ¥ÍÐˆ°€ÔÀÜÄ¤°(€€€€€€€€ ‰ÍÑ½É•}Õ¹…Ù…¥±…‰±”ˆ°€ÔÀÜÈ¤°(€€€t°(€€€¥‘ÌõlˆÔÀÜÀµ‘¥Í¬µ™Õ±°ˆ°€ˆÔÀÜÄµ•¹•É¥Œˆ°€ˆÔÀÜÈµÍÑ½É”µÕ¹…Ù…¥±…‰±”‰t°(¤)‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}Á•ÉÍ¥ÍÑ}™…¥±ÕÉ•Í}Õ¹Ý¥¹‘}±…¥µ•‘}ÑÕÉ¸ (€€€µ½¹­•åÁ…Ñ °Á•ÉÍ¥ÍÑ}µ½‘”°•áÁ•Ñ•‘}½‘”°(¤è(€€€€ˆˆ‰™Ñ•È}±½­}¥¹}ÍÕ‰µ¥Ñ}ÑÕÉ¸±…¥µÌÑ¡”ÑÕÉ¸°Á•ÉÍ¥ÍÐ™…¥±ÕÉ•Ì€ÔÀÜÀ¼ÔÀÜÄ¼ÔÀÜÈ(€€€µÕÍÐÕ¹Ý¥¹ÉÕ¹¹¥¹œ½¥¹™±¥¡ÐÍ¼Ñ¡”Í•ÍÍ¥½¸¥Ì¹½Ð±•™ÐÝ½É­¥¹œÝ¥Ñ ¹¼Ñ¡É•…€ ŒÄÀØäàÜ¤¸ˆˆˆ(€€€¥µÁ½ÉÐ•ÉÉ¹¼((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}™œˆ°±…µ‰‘„èì‰‘…Í¡‰½…Éˆèíõô¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•¹ÍÕÉ•}…Ñ¥Ù•}Í•ÍÍ¥½¹}Í±½Ðˆ°±…µ‰‘„€©}„°€¨©}¬è9½¹”¤((€€€¥˜Á•ÉÍ¥ÍÑ}µ½‘”€ôô€‰ÍÑ½É•}Õ¹…Ù…¥±…‰±”ˆè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}‘‰}•ÉÉ½Èˆ°€‰ÕÑ˜´à‘•½‘”™…¥±ÕÉ”ˆ¤(€€€•±¥˜Á•ÉÍ¥ÍÑ}µ½‘”€ôô€‰‘¥Í­}™Õ±°ˆè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•¹ÍÕÉ•}Í•ÍÍ¥½¹}‘‰}É½Üˆ°±…µ‰‘„}Í•ÍÍ¥½¸èQÉÕ”¤((€€€€€€€‘•˜}É…¥Í•}‘¥Í­}™Õ±°¡}Í•ÍÍ¥½¸¤è(€€€€€€€€€€€É…¥Í”=MÉÉ½È¡•ÉÉ¹¼¹9=MA°€‰9¼ÍÁ…”±•™Ð½¸‘•Ù¥”ˆ¤((€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Á•ÉÍ¥ÍÑ}‰É…¹¡}Í••ˆ°}É…¥Í•}‘¥Í­}™Õ±°¤(€€€•±Í”è(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•¹ÍÕÉ•}Í•ÍÍ¥½¹}‘‰}É½Üˆ°±…µ‰‘„}Í•ÍÍ¥½¸èQÉÕ”¤((€€€€€€€‘•˜}É…¥Í•}•¹•É¥Œ¡}Í•ÍÍ¥½¸¤è(€€€€€€€€€€€É…¥Í”IÕ¹Ñ¥µ•ÉÉ½È ‰Á•ÉÍ¥ÍÐ‰½½´ˆ¤((€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Á•ÉÍ¥ÍÑ}‰É…¹¡}Í••ˆ°}É…¥Í•}•¹•É¥Œ¤((€€€Í¥€ô˜‰Õ¹Ý¥¹µíÁ•ÉÍ¥ÍÑ}µ½‘•ôˆ(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸ ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÍ¥‘t€ôÍ•ÍÍ¥½¸(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€‰Õ¹Ý¥¹ˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆèÍ¥°€‰Ñ•áÐˆè€‰±…¥´Ñ¡•¸™…¥°‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô•áÁ•Ñ•‘}½‘”(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰ÉÕ¹¹¥¹œ‰t¥Ì…±Í”(€€€€€€€…ÍÍ•ÉÐ¹½ÐÍ•ÍÍ¥½¸¹•Ð ‰¥¹™±¥¡Ñ}ÑÕÉ¸ˆ¤(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¸¹•Ð ‰}ÉÕ¹}Ñ¡É•…ˆ¤¥Ì9½¹”(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹}±¥Ù•}ÍÑ…ÑÕÌ¡Í¥°Í•ÍÍ¥½¸¤€ôô€‰¥‘±”ˆ((€€€€€€€¥˜•áÁ•Ñ•‘}½‘”€ôô€ÔÀÜÈè(€€€€€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•¹ÍÕÉ•}Í•ÍÍ¥½¹}‘‰}É½Üˆ°±…µ‰‘„}Í•ÍÍ¥½¸èQÉÕ”¤(€€€€€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Á•ÉÍ¥ÍÑ}‰É…¹¡}Í••ˆ°±…µ‰‘„}Í•ÍÍ¥½¸è9½¹”¤(€€€€€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤((€€€€€€€€€€€±…ÍÌ}I•½É‘Q¡É•…è(€€€€€€€€€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©­Ü¤è(€€€€€€€€€€€€€€€€€€€Í•±˜¹Ñ…É•Ð€ôÑ…É•Ð((€€€€€€€€€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€€€€€€€€€É•ÑÕÉ¸9½¹”((€€€€€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}I•½É‘Q¡É•…¤(€€€€€€€€€€€É•½Ù•É•€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€€€€€‰¥ˆè€‰É•½Ù•Èˆ°(€€€€€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆèÍ¥°€‰Ñ•áÐˆè€‰É•ÑÉä…™Ñ•ÈÍÑ½É”É•½Ù•É•‰ô°(€€€€€€€€€€€€€€€ô(€€€€€€€€€€€€¤(€€€€€€€€€€€…ÍÍ•ÉÐÉ•½Ù•É•¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÉ•½Ù•É•¹•Ð •ÉÉ½Èœ¥ôˆ(€€€€€€€€€€€…ÍÍ•ÉÐÉ•½Ù•É•‘l‰É•ÍÕ±Ð‰ul‰ÍÑ…ÑÕÌ‰t€ôô€‰ÍÑÉ•…µ¥¹œˆ(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡Í¥°9½¹”¤(()ÁåÑ•ÍÐ¹µ…É¬¹É•…±}…•¹Ñ}ÁÉ•Ý…É´)‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}É•…Ñ•}½¹Ñ¥¹Õ•Í}Ý¡•¹}ÍÑ…Ñ•}‘‰}¥Í}Õ¹…Ù…¥±…‰±”¡µ½¹­•åÁ…Ñ ¤è(€€€±…ÍÌ}…­•]½É­•Èè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°­•ä°µ½‘•°°ÁÉ½™¥±•}¡½µ”õ9½¹”¤è(€€€€€€€€€€€Í•±˜¹­•ä€ô­•ä((€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸9½¹”((€€€±…ÍÌ}…­••¹Ðè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹µ½‘•°€ô€‰àˆ(€€€€€€€€€€€Í•±˜¹ÁÉ½Ù¥‘•È€ô€‰½Á•¹É½ÕÑ•Èˆ(€€€€€€€€€€€Í•±˜¹‰…Í•}ÕÉ°€ô€ˆˆ(€€€€€€€€€€€Í•±˜¹…Á¥}­•ä€ô€ˆˆ((€€€•µ¥ÑÌ€ômt((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}µ…­•}…•¹Ðˆ°±…µ‰‘„Í¥°­•ä°Í•ÍÍ¥½¹}‘ˆõ9½¹”°€¨©}­Ý…ÉÌè}…­••¹Ð ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}M±…Í¡]½É­•Èˆ°}…­•]½É­•È¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„}„°€©„Èèì‰µ½‘•°ˆè€‰à‰ô¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÁÉ½‰•}É•‘•¹Ñ¥…±Ìˆ°±…µ‰‘„}„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Ý¥É•}…±±‰…­Ìˆ°±…µ‰‘„}Í¥è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„°€¨©­Üè•µ¥ÑÌ¹…ÁÁ•¹¡„¤¤((€€€¥µÁ½ÉÐÑ½½±Ì¹…ÁÁÉ½Ù…°…Ì}…ÁÁÉ½Ù…°((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡}…ÁÁÉ½Ù…°°€‰É•¥ÍÑ•É}…Ñ•Ý…å}¹½Ñ¥™äˆ°±…µ‰‘„­•ä°ˆè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡}…ÁÁÉ½Ù…°°€‰±½…‘}Á•Éµ…¹•¹Ñ}…±±½Ý±¥ÍÐˆ°±…µ‰‘„è9½¹”¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹É•…Ñ”ˆ°€‰Á…É…µÌˆèì‰½±Ìˆè€àÁõô(€€€€¤(€€€Í¥€ôÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰t(€€€Í•ÍÍ¥½¸€ôÍ•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÍ¥‘t(€€€Í•ÍÍ¥½¹l‰…•¹Ñ}É•…‘ä‰t¹Ý…¥Ð¡Ñ¥µ•½ÕÐôÈ¸À¤((€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰…•¹Ñ}•ÉÉ½È‰t¥Ì9½¹”(€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰…•¹Ð‰t¥Ì¹½Ð9½¹”(€€€…ÍÍ•ÉÐ¹½Ð…¹ä¡…ÉÌ…¹…ÉÍlÁt€ôô€‰•ÉÉ½Èˆ™½È…ÉÌ¥¸•µ¥ÑÌ¤((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡Í¥°9½¹”¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}É•…Ñ•}±…éå}¥¹™½}É•Á½ÉÑÍ}‘•Í­Ñ½Á}½¹ÑÉ…Ð¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰Q¡”±…éäÍ•ÍÍ¥½¸¹É•…Ñ”¥¹™¼Á…å±½…µÕÍÐ…ÉÉä‘•Í­Ñ½Á}½¹ÑÉ…Ð°•±Í”(€€€Ñ¡”‘•Í­Ñ½ÀU$É•…‘Ì¥Ð…ÌÕ¹‘•™¥¹•…¹™…±Í•±äÝ…É¹Ì€‰	…­•¹½ÕÐ½˜(€€€‘…Ñ”ˆ½¸•Ù•Éä±…Õ¹ •Ù•¸……¥¹ÍÐ„ÕÉÉ•¹Ð‰…­•¹¸ˆˆˆ((€€€±…ÍÌ}…­•]½É­•Èè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°­•ä°µ½‘•°°ÁÉ½™¥±•}¡½µ”õ9½¹”¤è(€€€€€€€€€€€Í•±˜¹­•ä€ô­•ä((€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸9½¹”((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}M±…Í¡]½É­•Èˆ°}…­•]½É­•È¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„°€¨©­Üè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„€©„°€¨©­Üè9½¹”¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹É•…Ñ”ˆ°€‰Á…É…µÌˆèì‰½±Ìˆè€àÁõô(€€€€¤(€€€¥¹™¼€ôÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰¥¹™¼‰t((€€€…ÍÍ•ÉÐ¥¹™½l‰‘•Í­Ñ½Á}½¹ÑÉ…Ð‰t€ôôÍ•ÉÙ•È¹M-Q=A}	-9}=9QIP((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡É•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰t°9½¹”¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}…Ñ¥Ù…Ñ•}±…éå}¥¹™½}É•Á½ÉÑÍ}‘•Í­Ñ½Á}½¹ÑÉ…Ð ¤è(€€€€ˆˆ‰Ñ¥Ù…Ñ¥¹œ…¸…±É•…‘äµ±¥Ù”€©±…éä¨Í•ÍÍ¥½¸€¡…•¹Ð¹½Ð‰Õ¥±Ðå•Ð¤µÕÍÐ(€€€ÍÑ¥±°…‘Ù•ÉÑ¥Í”‘•Í­Ñ½Á}½¹ÑÉ…Ð¸}±¥Ù•}Í•ÍÍ¥½¹}Á…å±½…™…±±Ì‰…¬Ñ¼(€€€}™…±±‰…­}Í•ÍÍ¥½¹}¥¹™¼Ý¡¥±”Í•ÍÍ¥½¹l‰…•¹Ð‰t¥Ì9½¹”ìÑ¡”‘•Í­Ñ½ÀÉ•…‘Ì„(€€€µ¥ÍÍ¥¹œ™¥•±…Ì½¹ÑÉ…Ð€À…¹™…±Í•±äÝ…É¹Ì€‰	…­•¹½ÕÐ½˜‘…Ñ”ˆ……¥¹ÍÐ(€€€„ÕÉÉ•¹Ð‰…­•¹€ ŒØàÌäÈ¤¸Q¡”Í¥‰±¥¹œÍ•ÍÍ¥½¸¹É•…Ñ”Á…Ñ Ý…Ì™¥á•¥¸(€€€€ŒÌØÄÄÈìÑ¡¥ÌÁ¥¹ÌÑ¡”Í•ÍÍ¥½¸¹…Ñ¥Ù…Ñ”Á…Ñ ¸ˆˆˆ(€€€¥µÁ½ÉÐÑ¡É•…‘¥¹œ((€€€Í¥€ô€‰±…éäµ…Ñ¥Ù…Ñ”µ½¹ÑÉ…Ðˆ(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÍ¥‘t€ôì(€€€€€€€€‰…•¹Ðˆè9½¹”°(€€€€€€€€‰É•…Ñ•‘}…Ðˆè€ÄÈÌ¸À°(€€€€€€€€‰¡¥ÍÑ½Éäˆèmt°(€€€€€€€€‰¡¥ÍÑ½Éå}±½¬ˆèÑ¡É•…‘¥¹œ¹I1½¬ ¤°(€€€€€€€€‰±…ÍÑ}…Ñ¥Ù”ˆè€ÄÈÌ¸À°(€€€€€€€€‰ÉÕ¹¹¥¹œˆè…±Í”°(€€€€€€€€‰Í•ÍÍ¥½¹}­•äˆèÍ¥°(€€€€€€€€‰ÑÉ…¹ÍÁ½ÉÐˆèÍ•ÉÙ•È¹}ÍÑ‘¥½}ÑÉ…¹ÍÁ½ÉÐ°(€€€ô(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€‰…Ñ¥Ù…Ñ”µ±…éäˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹…Ñ¥Ù…Ñ”ˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆèÍ¥‘ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€¥¹™¼€ôÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰¥¹™¼‰t(€€€€€€€…ÍÍ•ÉÐ¥¹™½l‰±…éä‰t¥ÌQÉÕ”(€€€€€€€…ÍÍ•ÉÐ¥¹™½l‰‘•Í­Ñ½Á}½¹ÑÉ…Ð‰t€ôôÍ•ÉÙ•È¹M-Q=A}	-9}=9QIP(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡Í¥°9½¹”¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}±¥ÍÑ}É•ÑÕÉ¹Í}±•…¹}•ÉÉ½É}Ý¡•¹}ÍÑ…Ñ•}‘‰}¥Í}Õ¹…Ù…¥±…‰±”¡µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}‘‰}•ÉÉ½Èˆ°€‰±½­¥¹œÁÉ½Ñ½½°ˆ¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ¡ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹±¥ÍÐˆ°€‰Á…É…µÌˆèíõô¤((€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¥¸É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰‘…Ñ„‰ul‰½‘”‰t€ôô€‰ÍÑ½É…•}Õ¹…Ù…¥±…‰±”ˆ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰‘…Ñ„‰ul‰‘•Ñ…¥±Ì‰t€ôô€‰±½­¥¹œÁÉ½Ñ½½°ˆ(((Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(ŒÍ•ÍÍ¥½¸¹‘•±•Ñ”ƒŠPQU$É•ÍÕµ”Á¥­•È‘€­•ä(Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}‘•±•Ñ•}É•ÅÕ¥É•Í}Í•ÍÍ¥½¹}¥¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰µÁÑä€¼µ¥ÍÍ¥¹œÍ•ÍÍ¥½¹}¥¥Ì„€ÐÀÀØ±¥•¹Ð•ÉÉ½È€¡¹¼…±°¤¸ˆˆˆ(€€€…±±•è±¥ÍÑmÑÕÁ±•t€ômt((€€€±…ÍÌ}è(€€€€€€€‘•˜‘•±•Ñ•}Í•ÍÍ¥½¸¡Í•±˜°€©„°€¨©­Ü¤è(€€€€€€€€€€€…±±•¹…ÁÁ•¹ ¡„°­Ü¤¤(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è} ¤¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ¡ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹‘•±•Ñ”ˆ°€‰Á…É…µÌˆèíõô¤(€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¥¸É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÀØ(€€€…ÍÍ•ÉÐ…±±•€ôômt(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}‘•±•Ñ•}É•ÑÕÉ¹Í}‘‰}Õ¹…Ù…¥±…‰±•}Ý¡•¹}¹½}‘ˆ¡µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}‘‰}•ÉÉ½Èˆ°€‰±½­•ˆ¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹‘•±•Ñ”ˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰…‰Œ‰õô(€€€€¤((€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¥¸É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÔÀÌØ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰‘…Ñ„‰ul‰½‘”‰t€ôô€‰ÍÑ½É…•}±½­•ˆ(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}‘•±•Ñ•}É•™ÕÍ•Í}…Ñ¥Ù•}Í•ÍÍ¥½¸¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰…¹¹½Ð‘•±•Ñ”„Í•ÍÍ¥½¸ÕÉÉ•¹Ñ±ä‰½Õ¹Ñ¼„±¥Ù”QU$Í•ÍÍ¥½¸¸ˆˆˆ(€€€…±±•è±¥ÍÑmÍÑÉt€ômt((€€€±…ÍÌ}è(€€€€€€€‘•˜‘•±•Ñ•}Í•ÍÍ¥½¸¡Í•±˜°Í¥°Í•ÍÍ¥½¹Í}‘¥Èõ9½¹”¤è(€€€€€€€€€€€…±±•¹…ÁÁ•¹¡Í¥¤(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è} ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ¥Ñ•´¡Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì°€‰±¥Ù”ˆ°ì‰Í•ÍÍ¥½¹}­•äˆè€‰­•äµ±¥Ù”‰ô¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹‘•±•Ñ”ˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰­•äµ±¥Ù”‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰±¥Ù”ˆ°9½¹”¤((€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¥¸É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÈÌ(€€€…ÍÍ•ÉÐ…±±•€ôômt°€‰‘•±•Ñ•}Í•ÍÍ¥½¸µÕÍÐ¹½Ð‰”…±±•™½È…Ñ¥Ù”Í•ÍÍ¥½¹Ìˆ(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}‘•±•Ñ•}™…¥±Í}±½Í•‘}Ý¡•¹}…Ñ¥Ù•}Í¹…ÁÍ¡½Ñ}É…¥Í•Ì¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰½¹ÕÉÉ•¹Ð}Í•ÍÍ¥½¹Í€µÕÑ…Ñ¥½¸™É½´…¹½Ñ¡•ÈIAÑ¡É•……¸É…¥Í”(€€€IÕ¹Ñ¥µ•ÉÉ½Èè‘¥Ñ¥½¹…Éä¡…¹•Í¥é”‘ÕÉ¥¹œ¥Ñ•É…Ñ¥½¹€¸€]¡•¸Ñ¡”(€€€¡…¹‘±•È…¸Ð•¹Õµ•É…Ñ”…Ñ¥Ù”Í•ÍÍ¥½¹ÌÍ…™•±ä¥ÐµÕÍÐÉ•™ÕÍ”Ñ¡”(€€€‘•±•Ñ”€¡™…¥°±½Í•¤É…Ñ¡•ÈÑ¡…¸™…±°Ñ¡É½Õ …¹…±±½Ü¥Ð¸ˆˆˆ((€€€±…ÍÌ}è(€€€€€€€‘•˜‘•±•Ñ•}Í•ÍÍ¥½¸¡Í•±˜°€©„°€¨©­Ü¤è(€€€€€€€€€€€É…¥Í”ÍÍ•ÉÑ¥½¹ÉÉ½È ‰‘•±•Ñ”µÕÍÐ¹½ÐÉÕ¸Ý¡•¸…Ñ¥Ù”Í¹…ÁÍ¡½Ð™…¥±Ìˆ¤((€€€±…ÍÌ}áÁ±½‘¥¹¥Ðè(€€€€€€€‘•˜Ù…±Õ•Ì¡Í•±˜¤è(€€€€€€€€€€€É…¥Í”IÕ¹Ñ¥µ•ÉÉ½È ‰‘¥Ñ¥½¹…Éä¡…¹•Í¥é”‘ÕÉ¥¹œ¥Ñ•É…Ñ¥½¸ˆ¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è} ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹Ìˆ°}áÁ±½‘¥¹¥Ð ¤¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹‘•±•Ñ”ˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰à‰õô(€€€€¤((€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¥¸É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÔÀÌØ(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}‘•±•Ñ•}É•ÑÕÉ¹Í|ÐÀÀÝ}Ý¡•¹}µ¥ÍÍ¥¹œ¡µ½¹­•åÁ…Ñ ¤è(€€€±…ÍÌ}è(€€€€€€€‘•˜‘•±•Ñ•}Í•ÍÍ¥½¸¡Í•±˜°Í¥°Í•ÍÍ¥½¹Í}‘¥Èõ9½¹”¤è(€€€€€€€€€€€É•ÑÕÉ¸…±Í”((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è} ¤¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹‘•±•Ñ”ˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰¡½ÍÐ‰õô(€€€€¤((€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¥¸É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÀÜ(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}‘•±•Ñ•}ÁÉ½Á……Ñ•Í}‘‰}•á•ÁÑ¥½¸¡µ½¹­•åÁ…Ñ ¤è(€€€±…ÍÌ}è(€€€€€€€‘•˜‘•±•Ñ•}Í•ÍÍ¥½¸¡Í•±˜°Í¥°Í•ÍÍ¥½¹Í}‘¥Èõ9½¹”¤è(€€€€€€€€€€€É…¥Í”IÕ¹Ñ¥µ•ÉÉ½È ‰‘¥Í¬™Õ±°ˆ¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è} ¤¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹‘•±•Ñ”ˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰à‰õô(€€€€¤((€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¥¸É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÔÀÌØ(€€€…ÍÍ•ÉÐ€‰‘¥Í¬™Õ±°ˆ¥¸É•ÍÁl‰•ÉÉ½È‰ul‰µ•ÍÍ…”‰t(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}‘•±•Ñ•}ÍÕ•ÍÍ}É•ÑÕÉ¹Í}‘•±•Ñ•‘}¥¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰!…ÁÁäÁ…Ñ ƒŠP‘•±•Ñ”ÍÕ••‘Ì°É•ÍÁ½¹Í”…ÉÉ¥•ÌÑ¡”‘•±•Ñ•¥(€€€…¹Ñ¡”½¸µ‘¥Í¬Í•ÍÍ¥½¹Ì‘¥È¥Ì™½ÉÝ…É‘•Í¼ÑÉ…¹ÍÉ¥ÁÐ™¥±•Ì•Ð(€€€±•…¹•ÕÀ…±½¹Í¥‘”Ñ¡”É½Ü¸ˆˆˆ(€€€…ÁÑÕÉ•è‘¥Ð€ôíô((€€€±…ÍÌ}è(€€€€€€€‘•˜‘•±•Ñ•}Í•ÍÍ¥½¸¡Í•±˜°Í¥°Í•ÍÍ¥½¹Í}‘¥Èõ9½¹”¤è(€€€€€€€€€€€…ÁÑÕÉ•‘l‰Í¥‰t€ôÍ¥(€€€€€€€€€€€…ÁÑÕÉ•‘l‰Í•ÍÍ¥½¹Í}‘¥È‰t€ôÍ•ÍÍ¥½¹Í}‘¥È(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è} ¤¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹‘•±•Ñ”ˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰½±´Ä‰õô(€€€€¤((€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÀ°É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰t€ôôì‰‘•±•Ñ•ˆè€‰½±´Ä‰ô(€€€…ÍÍ•ÉÐ…ÁÑÕÉ•‘l‰Í¥‰t€ôô€‰½±´Äˆ(€€€€ŒÍ•ÍÍ¥½¹Í}‘¥ÈµÕÍÐ‰”™½ÉÝ…É‘•Í¼ÑÉ…¹ÍÉ¥ÁÐ™¥±•Ì•Ð±•…¹•ÕÀ(€€€€ŒÑ½¼ƒŠP¹½Ð©ÕÍÐÑ¡”ME1¥Ñ”É½Ü¸€Q¡”…ÕÑ½ÕÍ”}¥Í½±…Ñ•}¡•Éµ•Í}¡½µ”(€€€€Œ™¥áÑÕÉ”Á¥¹Ì!I5M}!=5Ñ¼„Ñ•µÀ‘¥ÈìÑ¡”¡…¹‘±•ÈÍ¡½Õ±…ÁÁ•¹(€€€€Œ€½Í•ÍÍ¥½¹ÌÑ¼¥Ð¸(€€€…ÍÍ•ÉÐ…ÁÑÕÉ•‘l‰Í•ÍÍ¥½¹Í}‘¥È‰t¥Ì¹½Ð9½¹”(€€€…ÍÍ•ÉÐÍÑÈ¡…ÁÑÕÉ•‘l‰Í•ÍÍ¥½¹Í}‘¥È‰t¤¹•¹‘ÍÝ¥Ñ  ‰Í•ÍÍ¥½¹Ìˆ¤(((((Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(ŒÍ•ÍÍ¥½¸¸¨ÁÉ½™¥±”Í½Á¥¹œ€¡…ÁÀµ±½‰…°É•µ½Ñ”µ½‘”¤ƒŠP€ŒØÈÔÀÌ(Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}±¥ÍÑ}¡½¹½ÉÍ}Á…É…µÍ}ÁÉ½™¥±•}½Á•¹Í}ÁÉ½™¥±•}‘ˆ¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰%ÍÍÕ”€ŒØÈÔÀÌèÍ•ÍÍ¥½¸¹±¥ÍÐµÕÍÐÉ•…Ñ¡”ÁÉ½™¥±”ÌÍÑ…Ñ”¹‘ˆ°¹½Ð±…Õ¹ ¸ˆˆˆ(€€€ÁÉ½™¥±•}¡½µ”€ôÑµÁ}Á…Ñ €¼€‰ÁÉ½™¥±•Ìˆ€¼€‰µ±Á•É˜ˆ(€€€ÁÉ½™¥±•}¡½µ”¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”¤(€€€€¡ÁÉ½™¥±•}¡½µ”€¼€‰ÍÑ…Ñ”¹‘ˆˆ¤¹ÝÉ¥Ñ•}‰åÑ•Ì¡ˆˆˆ¤(€€€Í••¸è‘¥Ð€ôíô((€€€±…ÍÌ1…Õ¹¡è(€€€€€€€‘•˜±¥ÍÑ}Í•ÍÍ¥½¹Í}É¥ ¡Í•±˜°€¨©­Ý…ÉÌ¤è(€€€€€€€€€€€Í••¹l‰±…Õ¹ ‰t€ôQÉÕ”(€€€€€€€€€€€É•ÑÕÉ¸mì‰¥ˆè€‰±…Õ¹ ´Äˆ°€‰Í½ÕÉ”ˆè€‰ÑÕ¤ˆ°€‰Ñ¥Ñ±”ˆè€‰0‰õt((€€€±…ÍÌAÉ½™¥±•è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°‘‰}Á…Ñ õ9½¹”¤è(€€€€€€€€€€€Í••¹l‰‘‰}Á…Ñ ‰t€ô‘‰}Á…Ñ ((€€€€€€€‘•˜±¥ÍÑ}Í•ÍÍ¥½¹Í}É¥ ¡Í•±˜°€¨©­Ý…ÉÌ¤è(€€€€€€€€€€€Í••¹l‰ÁÉ½™¥±”‰t€ôQÉÕ”(€€€€€€€€€€€É•ÑÕÉ¸l(€€€€€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€€€€€‰¥ˆè€‰µ°´Äˆ°(€€€€€€€€€€€€€€€€€€€€‰Í½ÕÉ”ˆè€‰ÑÕ¤ˆ°(€€€€€€€€€€€€€€€€€€€€‰Ñ¥Ñ±”ˆè€‰4ˆ°(€€€€€€€€€€€€€€€€€€€€‰ÁÉ•Ù¥•Üˆè€ˆˆ°(€€€€€€€€€€€€€€€€€€€€‰ÍÑ…ÉÑ•‘}…Ðˆè€Ä°(€€€€€€€€€€€€€€€€€€€€‰µ•ÍÍ…•}½Õ¹Ðˆè€Ä°(€€€€€€€€€€€€€€€ô(€€€€€€€€€€€t((€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€Í••¹l‰±½Í•‰t€ôQÉÕ”((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÁÉ½™¥±•}¡½µ”ˆ°±…µ‰‘„ÀèÁÉ½™¥±•}¡½µ”¥˜À€ôô€‰µ±Á•É˜ˆ•±Í”9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è1…Õ¹¡ ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€‰¡•Éµ•Í}±¤¹Ý•‰}Í•ÉÙ•É}Í•ÍÍ¥½¹Ì¹}½Á•¹}Í•ÍÍ¥½¹}‘‰}…Ñ}Á…Ñ ˆ°(€€€€€€€±…µ‰‘„‘‰}Á…Ñ °€¨°É•…‘}½¹±äèAÉ½™¥±•¡‘‰}Á…Ñ õ‘‰}Á…Ñ ¤°(€€€€¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹±¥ÍÐˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì‰ÁÉ½™¥±”ˆè€‰µ±Á•É˜ˆ°€‰±¥µ¥Ðˆè€Õô°(€€€€€€€ô(€€€€¤(€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÀ°É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹Ì‰ulÁul‰¥‰t€ôô€‰µ°´Äˆ(€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰ÁÉ½™¥±”ˆ¤¥ÌQÉÕ”(€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰±…Õ¹ ˆ¤¥Ì9½¹”(€€€…ÍÍ•ÉÐÍÑÈ¡Í••¸¹•Ð ‰‘‰}Á…Ñ ˆ¤¤¹•¹‘ÍÝ¥Ñ  ‰ÍÑ…Ñ”¹‘ˆˆ¤(€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰±½Í•ˆ¤¥ÌQÉÕ”(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}µ½ÍÑ}É••¹Ñ}¡½¹½ÉÍ}Á…É…µÍ}ÁÉ½™¥±”¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰%ÍÍÕ”€ŒØÈÔÀÌèÍ•ÍÍ¥½¸¹µ½ÍÑ}É••¹ÐµÕÍÐ¹½ÐÉ•ÑÕÉ¸Ñ¡”±…Õ¹ ÁÉ½™¥±”Ñ¥À¸ˆˆˆ(€€€ÁÉ½™¥±•}¡½µ”€ôÑµÁ}Á…Ñ €¼€‰ÁÉ½™¥±•Ìˆ€¼€‰µ±Á•É˜ˆ(€€€ÁÉ½™¥±•}¡½µ”¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”¤((€€€±…ÍÌ1…Õ¹¡è(€€€€€€€‘•˜±¥ÍÑ}Í•ÍÍ¥½¹Í}É¥ ¡Í•±˜°€¨©­Ý…ÉÌ¤è(€€€€€€€€€€€É•ÑÕÉ¸mì‰¥ˆè€‰±…Õ¹ µÑ¥Àˆ°€‰Í½ÕÉ”ˆè€‰ÑÕ¤ˆ°€‰Ñ¥Ñ±”ˆè€‰0ˆ°€‰ÍÑ…ÉÑ•‘}…Ðˆè€åõt((€€€±…ÍÌAÉ½™¥±•Èè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°‘‰}Á…Ñ õ9½¹”¤è(€€€€€€€€€€€Í•±˜¹‘‰}Á…Ñ €ô‘‰}Á…Ñ ((€€€€€€€‘•˜±¥ÍÑ}Í•ÍÍ¥½¹Í}É¥ ¡Í•±˜°€¨©­Ý…ÉÌ¤è(€€€€€€€€€€€É•ÑÕÉ¸l(€€€€€€€€€€€€€€€ì‰¥ˆè€‰Ñ½½°µ¹½¥Í”ˆ°€‰Í½ÕÉ”ˆè€‰Ñ½½°ˆ°€‰Ñ¥Ñ±”ˆè€‰Ðˆ°€‰ÍÑ…ÉÑ•‘}…Ðˆè€åô°(€€€€€€€€€€€€€€€ì‰¥ˆè€‰µ°µÑ¥Àˆ°€‰Í½ÕÉ”ˆè€‰‘•Í­Ñ½Àˆ°€‰Ñ¥Ñ±”ˆè€‰4ˆ°€‰ÍÑ…ÉÑ•‘}…Ðˆè€Íô°(€€€€€€€€€€€t((€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€Á…ÍÌ((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÁÉ½™¥±•}¡½µ”ˆ°±…µ‰‘„ÀèÁÉ½™¥±•}¡½µ”¥˜À€ôô€‰µ±Á•É˜ˆ•±Í”9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è1…Õ¹¡ ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€‰¡•Éµ•Í}±¤¹Ý•‰}Í•ÉÙ•É}Í•ÍÍ¥½¹Ì¹}½Á•¹}Í•ÍÍ¥½¹}‘‰}…Ñ}Á…Ñ ˆ°(€€€€€€€±…µ‰‘„‘‰}Á…Ñ °€¨°É•…‘}½¹±äèAÉ½™¥±•È¡‘‰}Á…Ñ õ‘‰}Á…Ñ ¤°(€€€€¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹µ½ÍÑ}É••¹Ðˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì‰ÁÉ½™¥±”ˆè€‰µ±Á•É˜‰ô°(€€€€€€€ô(€€€€¤(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰t€ôô€‰µ°µÑ¥Àˆ(()‘•˜Ñ•ÍÑ}¡…¹‘½™™}É•ÅÕ•ÍÑ}ÕÍ•Í}Í•ÍÍ¥½¹}ÁÉ½™¥±•}¡½µ”¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰!…¹‘½™˜Ù…±¥‘…Ñ¥½¸µÕÍÐÉ•…Ñ¡”½Ý¹¥¹œÍ•ÍÍ¥½¸Ì…Ñ•Ý…ä½¹™¥œ¸ˆˆˆ(€€€¥µÁ½ÉÐ½¹Ñ•áÑ±¥ˆ((€€€™É½´…Ñ•Ý…ä¹½¹™¥œ¥µÁ½ÉÐ…Ñ•Ý…å½¹™¥œ°!½µ•¡…¹¹•°°A±…Ñ™½É´°A±…Ñ™½Éµ½¹™¥œ(€€€™É½´¡•Éµ•Í}±¤¹½¹™¥œ¥µÁ½ÉÐ•Ñ}¡•Éµ•Í}¡½µ”(€€€™É½´ÑÕ¥}…Ñ•Ý…ä¥µÁ½ÉÐµ•Ñ¡½‘Í}Í•ÍÍ¥½¸((€€€µ•Ñ¡½‘Í}Í•ÍÍ¥½¸¹É•¥ÍÑ•È¡Í•ÉÙ•È¤(€€€ÁÉ½™¥±•}¡½µ”€ôÑµÁ}Á…Ñ €¼€‰ÁÉ½™¥±•Ìˆ€¼€‰½‘•Èˆ(€€€ÁÉ½™¥±•}¡½µ”¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”¤(€€€Í••¹}¡½µ•Ì€ômt((€€€‘•˜±½…‘}½¹™¥œ ¤è(€€€€€€€¡½µ”€ô•Ñ}¡•Éµ•Í}¡½µ” ¤(€€€€€€€Í••¹}¡½µ•Ì¹…ÁÁ•¹¡¡½µ”¤(€€€€€€€½¹™¥œ€ô…Ñ•Ý…å½¹™¥œ ¤(€€€€€€€¥˜¡½µ”€ôôÁÉ½™¥±•}¡½µ”è(€€€€€€€€€€€½¹™¥œ¹Á±…Ñ™½ÉµÍmA±…Ñ™½É´¹%M=It€ôA±…Ñ™½Éµ½¹™¥œ (€€€€€€€€€€€€€€€•¹…‰±•õQÉÕ”°(€€€€€€€€€€€€€€€¡½µ•}¡…¹¹•°õ!½µ•¡…¹¹•° (€€€€€€€€€€€€€€€€€€€Á±…Ñ™½É´õA±…Ñ™½É´¹%M=I°(€€€€€€€€€€€€€€€€€€€¡…Ñ}¥ô‰‘¥Í½Éµ¡½µ”ˆ°(€€€€€€€€€€€€€€€€€€€¹…µ”ô‰!•Éµ•Ì€¼€¡…Ðµ½‘¥¹œˆ°(€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€¤(€€€€€€€É•ÑÕÉ¸½¹™¥œ((€€€±…ÍÌAÉ½™¥±•è(€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¸¡Í•±˜°}­•ä¤è(€€€€€€€€€€€É•ÑÕÉ¸ì‰¥ˆè}­•åô((€€€€€€€‘•˜É•ÅÕ•ÍÑ}¡…¹‘½™˜¡Í•±˜°}­•ä°Á±…Ñ™½É´¤è(€€€€€€€€€€€É•ÑÕÉ¸Á±…Ñ™½É´€ôô€‰‘¥Í½Éˆ((€€€½¹Ñ•áÑ±¥ˆ¹½¹Ñ•áÑµ…¹…•È(€€€‘•˜ÁÉ½™¥±•}‘ˆ¡}Í•ÍÍ¥½¸¤è(€€€€€€€å¥•±AÉ½™¥±• ¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰…Ñ•Ý…ä¹½¹™¥œ¹±½…‘}…Ñ•Ý…å}½¹™¥œˆ°±½…‘}½¹™¥œ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•¹ÍÕÉ•}Í•ÍÍ¥½¹}‘‰}É½Üˆ°±…µ‰‘„}Í•ÍÍ¥½¸è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}‘ˆˆ°ÁÉ½™¥±•}‘ˆ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰¡…¹‘½™˜µÁÉ½™¥±”‰t€ôì(€€€€€€€€‰ÉÕ¹¹¥¹œˆè…±Í”°(€€€€€€€€‰Í•ÍÍ¥½¹}­•äˆè€‰‘•Í­Ñ½Àµ½‘•ÈµÍ•ÍÍ¥½¸ˆ°(€€€€€€€€‰ÁÉ½™¥±•}¡½µ”ˆèÍÑÈ¡ÁÉ½™¥±•}¡½µ”¤°(€€€ô(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰¡…¹‘½™˜¹É•ÅÕ•ÍÐˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰¡…¹‘½™˜µÁÉ½™¥±”ˆ°(€€€€€€€€€€€€€€€€€€€€‰Á±…Ñ™½É´ˆè€‰‘¥Í½Éˆ°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰¡…¹‘½™˜µÁÉ½™¥±”ˆ°9½¹”¤((€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÀ°É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÅÕ•Õ•‰t¥ÌQÉÕ”(€€€…ÍÍ•ÉÐÍ••¹}¡½µ•Ì€ôômÁÉ½™¥±•}¡½µ•t(€€€…ÍÍ•ÉÐ•Ñ}¡•Éµ•Í}¡½µ” ¤€„ôÁÉ½™¥±•}¡½µ”(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}É•…Ñ•}É•Á½ÉÑÍ}É•ÅÕ•ÍÑ•‘}ÁÉ½™¥±•}¹…µ”¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰%ÍÍÕ”€ŒØÈÔÀÌèÍ•ÍÍ¥½¸¹É•…Ñ”¥¹™¼¹ÁÉ½™¥±•}¹…µ”µÕÍÐ¹½Ð…±Ý…åÌ‰”±…Õ¹ ¸ˆˆˆ(€€€ÁÉ½™¥±•}¡½µ”€ôÑµÁ}Á…Ñ €¼€‰ÁÉ½™¥±•Ìˆ€¼€‰µ±Á•É˜ˆ(€€€ÁÉ½™¥±•}¡½µ”¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”¤((€€€‘•˜}±•…È ¤è(€€€€€€€™½ÈÍ•ÍÍ¥½¸¥¸±¥ÍÐ¡Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Ù…±Õ•Ì ¤¤è(€€€€€€€€€€€Í•ÉÙ•È¹}Ñ•…É‘½Ý¹}Í•ÍÍ¥½¸¡Í•ÍÍ¥½¸¤(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í¡•‘Õ±•}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í¡•‘Õ±•}Í•ÍÍ¥½¹}…Á}•¹™½É•µ•¹Ðˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}½µÁ±•Ñ¥½¹}Ýˆ°±…µ‰‘„Á…É…µÌõ9½¹”èÍÑÈ¡ÑµÁ}Á…Ñ ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÁÉ½™¥±•}¡½µ”ˆ°±…µ‰‘„ÀèÁÉ½™¥±•}¡½µ”¥˜À€ôô€‰µ±Á•É˜ˆ•±Í”9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÕÉÉ•¹Ñ}ÁÉ½™¥±•}¹…µ”ˆ°±…µ‰‘„è€‰‘•™…Õ±Ðˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±…¥µ}…Ñ¥Ù•}Í•ÍÍ¥½¹}Í±½Ðˆ°±…µ‰‘„€©„°€¨©¬è€¡9½¹”°9½¹”¤¤(€€€}±•…È ¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹}µ•Ñ¡½‘Íl‰Í•ÍÍ¥½¸¹É•…Ñ”‰t ‰ÈÄˆ°ì‰ÁÉ½™¥±”ˆè€‰µ±Á•É˜ˆ°€‰½±Ìˆè€àÁô¤(€€€€€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÀ°É•ÍÀ(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰¥¹™¼‰ul‰ÁÉ½™¥±•}¹…µ”‰t€ôô€‰µ±Á•É˜ˆ(€€€€€€€Í¥€ôÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰t(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÍ¥‘ul‰ÁÉ½™¥±•}¡½µ”‰t€ôôÍÑÈ¡ÁÉ½™¥±•}¡½µ”¤(€€€™¥¹…±±äè(€€€€€€€}±•…È ¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}‘•±•Ñ•}¡½¹½ÉÍ}Á…É…µÍ}ÁÉ½™¥±•}Í•ÍÍ¥½¹Í}‘¥È¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰%ÍÍÕ”€ŒØÈÔÀÌè‘•±•Ñ”µÕÍÐÑ…É•ÐÑ¡”ÁÉ½™¥±”ÍÑ…Ñ”¹‘ˆ€¬Í•ÍÍ¥½¹Ì‘¥È¸ˆˆˆ(€€€ÁÉ½™¥±•}¡½µ”€ôÑµÁ}Á…Ñ €¼€‰ÁÉ½™¥±•Ìˆ€¼€‰µ±Á•É˜ˆ(€€€€¡ÁÉ½™¥±•}¡½µ”€¼€‰Í•ÍÍ¥½¹Ìˆ¤¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”¤(€€€…ÁÑÕÉ•è‘¥Ð€ôíô((€€€±…ÍÌAÉ½™¥±•è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°‘‰}Á…Ñ õ9½¹”¤è(€€€€€€€€€€€…ÁÑÕÉ•‘l‰‘‰}Á…Ñ ‰t€ô‘‰}Á…Ñ ((€€€€€€€‘•˜‘•±•Ñ•}Í•ÍÍ¥½¸¡Í•±˜°Í¥°Í•ÍÍ¥½¹Í}‘¥Èõ9½¹”¤è(€€€€€€€€€€€…ÁÑÕÉ•‘l‰Í¥‰t€ôÍ¥(€€€€€€€€€€€…ÁÑÕÉ•‘l‰Í•ÍÍ¥½¹Í}‘¥È‰t€ôÍ•ÍÍ¥½¹Í}‘¥È(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€…ÁÑÕÉ•‘l‰±½Í•‰t€ôQÉÕ”((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÁÉ½™¥±•}¡½µ”ˆ°±…µ‰‘„ÀèÁÉ½™¥±•}¡½µ”¥˜À€ôô€‰µ±Á•É˜ˆ•±Í”9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰¡•Éµ•Í}ÍÑ…Ñ•}É•¥ÍÑÉä¹…ÅÕ¥É”ˆ°AÉ½™¥±•¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹‘•±•Ñ”ˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰½±µµ°ˆ°€‰ÁÉ½™¥±”ˆè€‰µ±Á•É˜‰ô°(€€€€€€€ô(€€€€¤(€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÀ°É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰t€ôôì‰‘•±•Ñ•ˆè€‰½±µµ°‰ô(€€€…ÍÍ•ÉÐÍÑÈ¡…ÁÑÕÉ•‘l‰‘‰}Á…Ñ ‰t¤¹•¹‘ÍÝ¥Ñ  ‰ÍÑ…Ñ”¹‘ˆˆ¤(€€€…ÍÍ•ÉÐA…Ñ ¡…ÁÑÕÉ•‘l‰Í•ÍÍ¥½¹Í}‘¥È‰t¤€ôôÁÉ½™¥±•}¡½µ”€¼€‰Í•ÍÍ¥½¹Ìˆ(€€€…ÍÍ•ÉÐ…ÁÑÕÉ•¹•Ð ‰±½Í•ˆ¤¥ÌQÉÕ”(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}Ñ¥Ñ±•}ÕÍ•Í}Í•ÍÍ¥½¹}ÁÉ½™¥±•}‘‰}¹½Ñ}±…Õ¹ ¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰Í•ÍÍ¥½¸¹Ñ¥Ñ±”½¸„¹½¸µ±…Õ¹ ÁÉ½™¥±”Í•ÍÍ¥½¸µÕÍÐ¹½ÐÑ½Õ ±…Õ¹ ¸ˆˆˆ(€€€ÁÉ½™¥±•}¡½µ”€ôÑµÁ}Á…Ñ €¼€‰ÁÉ½™¥±•Ìˆ€¼€‰µ±Á•É˜ˆ(€€€ÁÉ½™¥±•}¡½µ”¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”¤(€€€Í••¸è‘¥Ð€ôíô((€€€±…ÍÌ1…Õ¹¡è(€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¹}Ñ¥Ñ±”¡Í•±˜°}­•ä¤è(€€€€€€€€€€€Í••¹l‰±…Õ¹¡}É•…‰t€ôQÉÕ”(€€€€€€€€€€€É•ÑÕÉ¸€‰™É½´µ±…Õ¹ ˆ((€€€€€€€‘•˜Í•Ñ}Í•ÍÍ¥½¹}Ñ¥Ñ±”¡Í•±˜°}­•ä°}Ñ¥Ñ±”¤è(€€€€€€€€€€€Í••¹l‰±…Õ¹¡}ÝÉ¥Ñ”‰t€ôQÉÕ”(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¸¡Í•±˜°}­•ä¤è(€€€€€€€€€€€É•ÑÕÉ¸ì‰¥ˆè}­•ä°€‰Ñ¥Ñ±”ˆè€‰™É½´µ±…Õ¹ ‰ô((€€€±…ÍÌAÉ½™¥±•è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°‘‰}Á…Ñ õ9½¹”¤è(€€€€€€€€€€€Í•±˜¹‘‰}Á…Ñ €ô‘‰}Á…Ñ (€€€€€€€€€€€Í••¹l‰‘‰}Á…Ñ ‰t€ô‘‰}Á…Ñ ((€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¹}Ñ¥Ñ±”¡Í•±˜°}­•ä¤è(€€€€€€€€€€€É•ÑÕÉ¸Í••¸¹•Ð ‰Ñ¥Ñ±”ˆ¤((€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¸¡Í•±˜°}­•ä¤è(€€€€€€€€€€€¥˜€‰Ñ¥Ñ±”ˆ¥¸Í••¸è(€€€€€€€€€€€€€€€É•ÑÕÉ¸ì‰¥ˆè}­•ä°€‰Ñ¥Ñ±”ˆèÍ••¹l‰Ñ¥Ñ±”‰uô(€€€€€€€€€€€É•ÑÕÉ¸9½¹”((€€€€€€€‘•˜Í•Ñ}Í•ÍÍ¥½¹}Ñ¥Ñ±”¡Í•±˜°}­•ä°Ñ¥Ñ±”¤è(€€€€€€€€€€€Í••¹l‰Ñ¥Ñ±”‰t€ôÑ¥Ñ±”(€€€€€€€€€€€Í••¹l‰ÁÉ½™¥±•}ÝÉ¥Ñ”‰t€ôQÉÕ”(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€Í••¹l‰±½Í•‰t€ôQÉÕ”((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ôì(€€€€€€€€‰Í•ÍÍ¥½¹}­•äˆè€‰µ°µÍ•ÍÌˆ°(€€€€€€€€‰¡¥ÍÑ½Éäˆèmt°(€€€€€€€€‰¡¥ÍÑ½Éå}±½¬ˆè}}¥µÁ½ÉÑ}| ‰Ñ¡É•…‘¥¹œˆ¤¹1½¬ ¤°(€€€€€€€€‰ÉÕ¹¹¥¹œˆè…±Í”°(€€€€€€€€‰Á•¹‘¥¹}Ñ¥Ñ±”ˆè9½¹”°(€€€€€€€€‰ÁÉ½™¥±•}¡½µ”ˆèÍÑÈ¡ÁÉ½™¥±•}¡½µ”¤°(€€€€€€€€‰…•¹Ðˆè9½¹”°(€€€€€€€€‰É•…Ñ•‘}…Ðˆè€Ä¸À°(€€€€€€€€‰±…ÍÑ}…Ñ¥Ù”ˆè€Ä¸À°(€€€ô(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è1…Õ¹¡ ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰¡•Éµ•Í}ÍÑ…Ñ•}É•¥ÍÑÉä¹…ÅÕ¥É”ˆ°AÉ½™¥±•¤(€€€ÑÉäè(€€€€€€€Í•Ñ}É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹Ñ¥Ñ±”ˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ¥Ñ±”ˆè€‰ÁÉ½™¥±”µÑ¥Ñ±”‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸Í•Ñ}É•ÍÀ°Í•Ñ}É•ÍÀ(€€€€€€€…ÍÍ•ÉÐÍ•Ñ}É•ÍÁl‰É•ÍÕ±Ð‰ul‰Ñ¥Ñ±”‰t€ôô€‰ÁÉ½™¥±”µÑ¥Ñ±”ˆ(€€€€€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰ÁÉ½™¥±•}ÝÉ¥Ñ”ˆ¤¥ÌQÉÕ”(€€€€€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰±…Õ¹¡}ÝÉ¥Ñ”ˆ¤¥Ì9½¹”(€€€€€€€…ÍÍ•ÉÐÍÑÈ¡Í••¸¹•Ð ‰‘‰}Á…Ñ ˆ¤¤¹•¹‘ÍÝ¥Ñ  ‰ÍÑ…Ñ”¹‘ˆˆ¤((€€€€€€€•Ñ}É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÈˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹Ñ¥Ñ±”ˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥‰õô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐ•Ñ}É•ÍÁl‰É•ÍÕ±Ð‰ul‰Ñ¥Ñ±”‰t€ôô€‰ÁÉ½™¥±”µÑ¥Ñ±”ˆ(€€€€€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰±…Õ¹¡}É•…ˆ¤¥Ì9½¹”(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}¡¥ÍÑ½Éå}ÕÍ•Í}Í•ÍÍ¥½¹}ÁÉ½™¥±•}‘ˆ¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰Í•ÍÍ¥½¸¹¡¥ÍÑ½ÉäµÕÍÐÉ•…‘ÕÉ…‰±”µ•ÍÍ…•Ì™É½´Ñ¡”ÁÉ½™¥±”ÍÑ…Ñ”¹‘ˆ¸ˆˆˆ(€€€ÁÉ½™¥±•}¡½µ”€ôÑµÁ}Á…Ñ €¼€‰ÁÉ½™¥±•Ìˆ€¼€‰µ±Á•É˜ˆ(€€€ÁÉ½™¥±•}¡½µ”¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”¤(€€€Í••¸è‘¥Ð€ôíô((€€€±…ÍÌ1…Õ¹¡è(€€€€€€€‘•˜•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°}­•ä°¥¹±Õ‘•}…¹•ÍÑ½ÉÌõQÉÕ”°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€Í••¹l‰±…Õ¹ ‰t€ôQÉÕ”(€€€€€€€€€€€É•ÑÕÉ¸mì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰±…Õ¹ ‰õt((€€€±…ÍÌAÉ½™¥±•è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°‘‰}Á…Ñ õ9½¹”¤è(€€€€€€€€€€€Í••¹l‰‘‰}Á…Ñ ‰t€ô‘‰}Á…Ñ ((€€€€€€€‘•˜•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°}­•ä°¥¹±Õ‘•}…¹•ÍÑ½ÉÌõQÉÕ”°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€Í••¹l‰ÁÉ½™¥±”‰t€ôQÉÕ”(€€€€€€€€€€€É•ÑÕÉ¸mì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰™É½´µÁÉ½™¥±”‰õt((€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€Í••¹l‰±½Í•‰t€ôQÉÕ”((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ôì(€€€€€€€€‰Í•ÍÍ¥½¹}­•äˆè€‰µ°µÍ•ÍÌˆ°(€€€€€€€€‰¡¥ÍÑ½Éäˆèmì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰µ•´‰õt°(€€€€€€€€‰¡¥ÍÑ½Éå}±½¬ˆè}}¥µÁ½ÉÑ}| ‰Ñ¡É•…‘¥¹œˆ¤¹1½¬ ¤°(€€€€€€€€‰ÉÕ¹¹¥¹œˆè…±Í”°(€€€€€€€€‰ÁÉ½™¥±•}¡½µ”ˆèÍÑÈ¡ÁÉ½™¥±•}¡½µ”¤°(€€€€€€€€‰…•¹Ðˆè9½¹”°(€€€€€€€€‰É•…Ñ•‘}…Ðˆè€Ä¸À°(€€€€€€€€‰±…ÍÑ}…Ñ¥Ù”ˆè€Ä¸À°(€€€ô(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è1…Õ¹¡ ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰¡•Éµ•Í}ÍÑ…Ñ•}É•¥ÍÑÉä¹…ÅÕ¥É”ˆ°AÉ½™¥±•¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹¡¥ÍÑ½Éäˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥‰õô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÀ°É•ÍÀ(€€€€€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰ÁÉ½™¥±”ˆ¤¥ÌQÉÕ”(€€€€€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰±…Õ¹ ˆ¤¥Ì9½¹”(€€€€€€€€Œ½Õ¹Ð½µ•Ì™É½´ÁÉ½™¥±”µ‰…­•½¹Ù•ÉÍ…Ñ¥½¸€ ÄµÍœ¤°¹½Ð‰…É”µ•´±¥ÍÐ…±½¹”¸(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰½Õ¹Ð‰t€ôô€Ä(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}¡¥ÍÑ½Éå}Í¡¥ÁÍ}‘ÕÉ…‰±•}É½Ý}¥‘Ì¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰Í•ÍÍ¥½¸¹¡¥ÍÑ½ÉäµÕÍÐÉ•ÅÕ•ÍÐÉ½Üµ¥ÍÑ…µÁÌƒŠP±¥•¹ÑÌÉ•Í½±Ù”ÑÉÕ¹…Ñ¥½¸(€€€Ñ…É•ÑÌ‰ä½¹Ñ•¹Ð……¥¹ÍÐÑ¡¥ÌÁÉ½©•Ñ¥½¸€ ŒàÜÀÔä±¥•¹Ð¡…±˜¤¸ˆˆˆ(€€€Í••¸è‘¥Ð€ôíô((€€€±…ÍÌ}ˆè(€€€€€€€‘•˜•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°}­•ä°¥¹±Õ‘•}…¹•ÍÑ½ÉÌõ…±Í”°¥¹±Õ‘•}É½Ý}¥‘Ìõ…±Í”°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€Í••¹l‰¥¹±Õ‘•}É½Ý}¥‘Ì‰t€ô¥¹±Õ‘•}É½Ý}¥‘Ì((€€€€€€€€€€€É•ÑÕÉ¸l(€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰¡•±±¼ˆ°€‰}É½Ý}¥ˆè€ÐÅô°(€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰¡¤‰ô°(€€€€€€€€€€€t((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰É½Ý¥µ¡¥ÍÐµÍ¥‰t€ôì(€€€€€€€€‰Í•ÍÍ¥½¹}­•äˆè€‰É½Ý¥µÍ•ÍÌˆ°(€€€€€€€€‰¡¥ÍÑ½Éäˆèmt°(€€€€€€€€‰¡¥ÍÑ½Éå}±½¬ˆè}}¥µÁ½ÉÑ}| ‰Ñ¡É•…‘¥¹œˆ¤¹1½¬ ¤°(€€€€€€€€‰ÉÕ¹¹¥¹œˆè…±Í”°(€€€€€€€€‰…•¹Ðˆè9½¹”°(€€€€€€€€‰É•…Ñ•‘}…Ðˆè€Ä¸À°(€€€€€€€€‰±…ÍÑ}…Ñ¥Ù”ˆè€Ä¸À°(€€€ô(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è}ˆ ¤¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹¡¥ÍÑ½Éäˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰É½Ý¥µ¡¥ÍÐµÍ¥‰õô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÀ°É•ÍÀ(€€€€€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰¥¹±Õ‘•}É½Ý}¥‘Ìˆ¤¥ÌQÉÕ”(€€€€€€€ÕÍ•É}É½ÝÌ€ôm´™½È´¥¸É•ÍÁl‰É•ÍÕ±Ð‰ul‰µ•ÍÍ…•Ì‰t¥˜´¹•Ð ‰É½±”ˆ¤€ôô€‰ÕÍ•È‰t(€€€€€€€…ÍÍ•ÉÐÕÍ•É}É½ÝÌ…¹ÕÍ•É}É½ÝÍlÁt¹•Ð ‰É½Ý}¥ˆ¤€ôô€ÐÄ(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰É½Ý¥µ¡¥ÍÐµÍ¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}ÍÑ…ÑÕÍ}ÕÍ•Í}Í•ÍÍ¥½¹}ÁÉ½™¥±•}‘ˆ¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰Í•ÍÍ¥½¸¹ÍÑ…ÑÕÌµÕÍÐ±½…µ•Ñ„™É½´Ñ¡”Í•ÍÍ¥½¸ÁÉ½™¥±”ÍÑ…Ñ”¹‘ˆ¸ˆˆˆ(€€€ÁÉ½™¥±•}¡½µ”€ôÑµÁ}Á…Ñ €¼€‰ÁÉ½™¥±•Ìˆ€¼€‰µ±Á•É˜ˆ(€€€ÁÉ½™¥±•}¡½µ”¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”¤(€€€Í••¸è‘¥Ð€ôíô((€€€±…ÍÌ1…Õ¹¡è(€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¸¡Í•±˜°}­•ä¤è(€€€€€€€€€€€Í••¹l‰±…Õ¹ ‰t€ôQÉÕ”(€€€€€€€€€€€É•ÑÕÉ¸ì‰¥ˆè}­•ä°€‰Ñ¥Ñ±”ˆè€‰±…Õ¹ µÑ¥Ñ±”ˆ°€‰ÍÑ…ÉÑ•‘}…Ðˆè€Åô((€€€±…ÍÌAÉ½™¥±•è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°‘‰}Á…Ñ õ9½¹”¤è(€€€€€€€€€€€Í••¹l‰‘‰}Á…Ñ ‰t€ô‘‰}Á…Ñ ((€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¸¡Í•±˜°}­•ä¤è(€€€€€€€€€€€Í••¹l‰ÁÉ½™¥±”‰t€ôQÉÕ”(€€€€€€€€€€€É•ÑÕÉ¸ì‰¥ˆè}­•ä°€‰Ñ¥Ñ±”ˆè€‰ÁÉ½™¥±”µÑ¥Ñ±”ˆ°€‰ÍÑ…ÉÑ•‘}…Ðˆè€ÐÉô((€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€Í••¹l‰±½Í•‰t€ôQÉÕ”((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ôì(€€€€€€€€‰Í•ÍÍ¥½¹}­•äˆè€‰µ°µÍ•ÍÌˆ°(€€€€€€€€‰¡¥ÍÑ½Éäˆèmt°(€€€€€€€€‰¡¥ÍÑ½Éå}±½¬ˆè}}¥µÁ½ÉÑ}| ‰Ñ¡É•…‘¥¹œˆ¤¹1½¬ ¤°(€€€€€€€€‰ÉÕ¹¹¥¹œˆè…±Í”°(€€€€€€€€‰ÁÉ½™¥±•}¡½µ”ˆèÍÑÈ¡ÁÉ½™¥±•}¡½µ”¤°(€€€€€€€€‰…•¹Ðˆè9½¹”°(€€€€€€€€‰É•…Ñ•‘}…Ðˆè€Ä¸À°(€€€€€€€€‰±…ÍÑ}…Ñ¥Ù”ˆè€Ä¸À°(€€€ô(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è1…Õ¹¡ ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰¡•Éµ•Í}ÍÑ…Ñ•}É•¥ÍÑÉä¹…ÅÕ¥É”ˆ°AÉ½™¥±•¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹ÍÑ…ÑÕÌˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥‰õô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÀ°É•ÍÀ(€€€€€€€…ÍÍ•ÉÐ€‰ÁÉ½™¥±”µÑ¥Ñ±”ˆ¥¸É•ÍÁl‰É•ÍÕ±Ð‰ul‰½ÕÑÁÕÐ‰t(€€€€€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰ÁÉ½™¥±”ˆ¤¥ÌQÉÕ”(€€€€€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰±…Õ¹ ˆ¤¥Ì9½¹”(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}Ñ•…É‘½Ý¹}•¹‘Í}Í•ÍÍ¥½¹}¥¹}ÁÉ½™¥±•}‘ˆ¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰}Ñ•…É‘½Ý¹}Í•ÍÍ¥½¸µÕÍÐ•¹‘}Í•ÍÍ¥½¸½¸Ñ¡”ÁÉ½™¥±”ÍÑ½É”°¹½Ð±…Õ¹ ¸ˆˆˆ(€€€ÁÉ½™¥±•}¡½µ”€ôÑµÁ}Á…Ñ €¼€‰ÁÉ½™¥±•Ìˆ€¼€‰µ±Á•É˜ˆ(€€€ÁÉ½™¥±•}¡½µ”¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”¤(€€€Í••¸è‘¥Ð€ôíô((€€€±…ÍÌ1…Õ¹¡è(€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¸¡Í•±˜°}­•ä¤è(€€€€€€€€€€€Í••¹l‰±…Õ¹ ‰t€ôQÉÕ”(€€€€€€€€€€€É•ÑÕÉ¸ì‰¥ˆè}­•ä°€‰Í½ÕÉ”ˆè€‰ÑÕ¤‰ô((€€€€€€€‘•˜•¹‘}Í•ÍÍ¥½¸¡Í•±˜°}­•ä°}É•…Í½¸¤è(€€€€€€€€€€€Í••¹l‰±…Õ¹¡}•¹‰t€ôQÉÕ”((€€€±…ÍÌAÉ½™¥±•è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°‘‰}Á…Ñ õ9½¹”¤è(€€€€€€€€€€€Í••¹l‰‘‰}Á…Ñ ‰t€ô‘‰}Á…Ñ ((€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¸¡Í•±˜°}­•ä¤è(€€€€€€€€€€€Í••¹l‰ÁÉ½™¥±”‰t€ôQÉÕ”(€€€€€€€€€€€É•ÑÕÉ¸ì‰¥ˆè}­•ä°€‰Í½ÕÉ”ˆè€‰ÑÕ¤‰ô((€€€€€€€‘•˜•¹‘}Í•ÍÍ¥½¸¡Í•±˜°­•ä°É•…Í½¸¤è(€€€€€€€€€€€Í••¹l‰•¹‘•‰t€ô€¡­•ä°É•…Í½¸¤((€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€Í••¹l‰±½Í•‰t€ôQÉÕ”((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è1…Õ¹¡ ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰¡•Éµ•Í}ÍÑ…Ñ•}É•¥ÍÑÉä¹…ÅÕ¥É”ˆ°AÉ½™¥±•¤(€€€Í•ÍÍ¥½¸€ôì(€€€€€€€€‰Í•ÍÍ¥½¹}­•äˆè€‰µ°µÍ•ÍÌˆ°(€€€€€€€€‰ÁÉ½™¥±•}¡½µ”ˆèÍÑÈ¡ÁÉ½™¥±•}¡½µ”¤°(€€€€€€€€‰…•¹Ðˆè9½¹”°(€€€€€€€€‰¡¥ÍÑ½Éäˆèmt°(€€€€€€€€‰Í½ÕÉ”ˆè€‰ÑÕ¤ˆ°(€€€ô(€€€Í•ÉÙ•È¹}Ñ•…É‘½Ý¹}Í•ÍÍ¥½¸¡Í•ÍÍ¥½¸°•¹‘}É•…Í½¸ô‰±½Í•ˆ¤(€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰•¹‘•ˆ¤€ôô€ ‰µ°µÍ•ÍÌˆ°€‰±½Í•ˆ¤(€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰±…Õ¹¡}•¹ˆ¤¥Ì9½¹”(€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰±…Õ¹ ˆ¤¥Ì9½¹”(€€€…ÍÍ•ÉÐÍÑÈ¡Í••¸¹•Ð ‰‘‰}Á…Ñ ˆ¤¤¹•¹‘ÍÝ¥Ñ  ‰ÍÑ…Ñ”¹‘ˆˆ¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}‰É…¹¡}ÝÉ¥Ñ•Í}Ñ½}Á…É•¹Ñ}ÁÉ½™¥±•}‘ˆ¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰Í•ÍÍ¥½¸¹‰É…¹ µÕÍÐ½Áä¡¥ÍÑ½Éä¥¹Ñ¼Ñ¡”Á…É•¹ÐÌÁÉ½™¥±”ÍÑ…Ñ”¹‘ˆ¸ˆˆˆ(€€€ÁÉ½™¥±•}¡½µ”€ôÑµÁ}Á…Ñ €¼€‰ÁÉ½™¥±•Ìˆ€¼€‰µ±Á•É˜ˆ(€€€ÁÉ½™¥±•}¡½µ”¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}!=5ˆ°ÍÑÈ¡ÑµÁ}Á…Ñ ¤¤(€€€Í••¸è‘¥Ð€ôì‰µÍÌˆèmuô((€€€±…ÍÌ1…Õ¹¡è(€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¹}Ñ¥Ñ±”¡Í•±˜°}­•ä¤è(€€€€€€€€€€€Í••¹l‰±…Õ¹ ‰t€ôQÉÕ”(€€€€€€€€€€€É•ÑÕÉ¸€‰0ˆ((€€€€€€€‘•˜É•…Ñ•}Í•ÍÍ¥½¸¡Í•±˜°€©„°€¨©¬¤è(€€€€€€€€€€€Í••¹l‰±…Õ¹¡}É•…Ñ”‰t€ôQÉÕ”((€€€€€€€‘•˜…ÁÁ•¹‘}µ•ÍÍ…”¡Í•±˜°€¨©¬¤è(€€€€€€€€€€€Í••¹l‰±…Õ¹¡}µÍœ‰t€ôQÉÕ”((€€€€€€€‘•˜Í•Ñ}Í•ÍÍ¥½¹}Ñ¥Ñ±”¡Í•±˜°€©„°€¨©¬¤è(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€±…ÍÌAÉ½™¥±•è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°‘‰}Á…Ñ õ9½¹”¤è(€€€€€€€€€€€Í••¹l‰‘‰}Á…Ñ ‰t€ô‘‰}Á…Ñ (€€€€€€€€€€€Í••¸¹Í•Ñ‘•™…Õ±Ð ‰¥¹¥ÑÌˆ°€À¤(€€€€€€€€€€€Í••¹l‰¥¹¥ÑÌ‰t€¬ô€Ä((€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¹}Ñ¥Ñ±”¡Í•±˜°}­•ä¤è(€€€€€€€€€€€É•ÑÕÉ¸€‰Á…É•¹Ðˆ((€€€€€€€‘•˜•Ñ}¹•áÑ}Ñ¥Ñ±•}¥¹}±¥¹•…”¡Í•±˜°ÕÉÉ•¹Ð¤è(€€€€€€€€€€€É•ÑÕÉ¸˜‰íÕÉÉ•¹Ñô€¡‰É…¹ ¤ˆ((€€€€€€€‘•˜É•…Ñ•}Í•ÍÍ¥½¸¡Í•±˜°¹•Ý}­•ä°€¨©­Ý…ÉÌ¤è(€€€€€€€€€€€Í••¹l‰É•…Ñ•‰t€ô¹•Ý}­•ä(€€€€€€€€€€€Í••¹l‰Á…É•¹Ð‰t€ô­Ý…ÉÌ¹•Ð ‰Á…É•¹Ñ}Í•ÍÍ¥½¹}¥ˆ¤(€€€€€€€€€€€Í••¹l‰ÁÉ½™¥±•}¹…µ”‰t€ô­Ý…ÉÌ¹•Ð ‰ÁÉ½™¥±•}¹…µ”ˆ¤((€€€€€€€‘•˜…ÁÁ•¹‘}µ•ÍÍ…”¡Í•±˜°€¨©­Ý…ÉÌ¤è(€€€€€€€€€€€Í••¹l‰µÍÌ‰t¹…ÁÁ•¹¡­Ý…ÉÌ¤((€€€€€€€‘•˜…ÁÁ•¹‘}µ•ÍÍ…•Í}‰…Ñ ¡Í•±˜°Í•ÍÍ¥½¹}¥°µ•ÍÍ…•Ì°€¨©­Ý…ÉÌ¤è(€€€€€€€€€€€™½È´¥¸µ•ÍÍ…•Ìè(€€€€€€€€€€€€€€€Í••¹l‰µÍÌ‰t¹…ÁÁ•¹¡‘¥Ð¡´°Í•ÍÍ¥½¹}¥õÍ•ÍÍ¥½¹}¥¤¤(€€€€€€€€€€€É•ÑÕÉ¸±¥ÍÐ¡É…¹” Ä°±•¸¡µ•ÍÍ…•Ì¤€¬€Ä¤¤((€€€€€€€‘•˜Í•Ñ}Í•ÍÍ¥½¹}Ñ¥Ñ±”¡Í•±˜°­•ä°Ñ¥Ñ±”¤è(€€€€€€€€€€€Í••¹l‰Ñ¥Ñ±”‰t€ô€¡­•ä°Ñ¥Ñ±”¤(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¸¡Í•±˜°­•ä¤è(€€€€€€€€€€€É•ÑÕÉ¸ì‰¥ˆè­•ä°€‰ÝˆèÍÑÈ¡ÑµÁ}Á…Ñ ¥ô((€€€€€€€‘•˜ÕÁ‘…Ñ•}Í•ÍÍ¥½¹}Ý¡Í•±˜°€©„°€¨©¬¤è(€€€€€€€€€€€É•ÑÕÉ¸9½¹”((€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€Í••¹l‰±½Í•‰t€ôQÉÕ”((€€€±…ÍÌ…­••¹Ðè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹µ½‘•°€ô€‰Ñ•ÍÐµµ½‘•°ˆ(€€€€€€€€€€€Í•±˜¹Í•ÍÍ¥½¹}¥€ô9½¹”((€€€Á…É•¹Ð€ôì(€€€€€€€€‰Í•ÍÍ¥½¹}­•äˆè€‰Á…É•¹Ðµ­•äˆ°(€€€€€€€€‰¡¥ÍÑ½Éäˆèmì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰¡¤‰õt°(€€€€€€€€‰¡¥ÍÑ½Éå}±½¬ˆè}}¥µÁ½ÉÑ}| ‰Ñ¡É•…‘¥¹œˆ¤¹1½¬ ¤°(€€€€€€€€‰ÉÕ¹¹¥¹œˆè…±Í”°(€€€€€€€€‰½±Ìˆè€àÀ°(€€€€€€€€‰ÁÉ½™¥±•}¡½µ”ˆèÍÑÈ¡ÁÉ½™¥±•}¡½µ”¤°(€€€€€€€€‰Í½ÕÉ”ˆè€‰ÑÕ¤ˆ°(€€€€€€€€‰…•¹Ðˆè…­••¹Ð ¤°(€€€€€€€€‰É•…Ñ•‘}…Ðˆè€Ä¸À°(€€€€€€€€‰±…ÍÑ}…Ñ¥Ù”ˆè€Ä¸À°(€€€€€€€€‰ÝˆèÍÑÈ¡ÑµÁ}Á…Ñ ¤°(€€€ô(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Á…É•¹Ð‰t€ôÁ…É•¹Ð(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è1…Õ¹¡ ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰¡•Éµ•Í}ÍÑ…Ñ•}É•¥ÍÑÉä¹…ÅÕ¥É”ˆ°AÉ½™¥±•¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±…¥µ}…Ñ¥Ù•}Í•ÍÍ¥½¹}Í±½Ðˆ°±…µ‰‘„€©„°€¨©¬è€¡9½¹”°9½¹”¤¤((€€€‘•˜}™…­•}µ…­•}…•¹Ð ©„°€¨©¬¤è(€€€€€€€Í••¹l‰…•¹Ñ}Í•ÍÍ¥½¹}‘ˆ‰t€ô¬¹•Ð ‰Í•ÍÍ¥½¹}‘ˆˆ¤(€€€€€€€Í••¹l‰…•¹Ñ}Ý‰t€ô¬¹•Ð ‰Ý‘}½Ù•ÉÉ¥‘”ˆ¤(€€€€€€€É•ÑÕÉ¸…­••¹Ð ¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}µ…­•}…•¹Ðˆ°}™…­•}µ…­•}…•¹Ð¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•Ñ}Í•ÍÍ¥½¹}½¹Ñ•áÐˆ°±…µ‰‘„€©„°€¨©¬èíô¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±•…É}Í•ÍÍ¥½¹}½¹Ñ•áÐˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}É•Í½±Ù•}µ½‘•°ˆ°±…µ‰‘„è€‰Ñ•ÍÐµµ½‘•°ˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}Ýˆ°±…µ‰‘„ÌèÍÑÈ¡ÑµÁ}Á…Ñ ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}É•¥ÍÑ•É}Í•ÍÍ¥½¹}Ýˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}…ÑÑ…¡}Ý½É­•Èˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹‰É…¹¡}Ý¡½±”ˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Á…É•¹Ðˆ°€‰¹…µ”ˆè€‰™½É­•‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÀ°É•ÍÀ(€€€€€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰É•…Ñ•ˆ¤(€€€€€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰Á…É•¹Ðˆ¤€ôô€‰Á…É•¹Ðµ­•äˆ(€€€€€€€€ŒQ¡”‰É…¹ É½Ü¥ÌÍ•±˜µ‘•ÍÉ¥‰¥¹œèÍÑ…µÁ•Ý¥Ñ Ñ¡”Á…É•¹ÐÌ½Ý¹¥¹œ(€€€€€€€€ŒÁÉ½™¥±”°¹½Ð±•™Ð9U10™½È…É•…Ñ½ÉÌÑ¼µ¥ÌµÑ…œ…Ì€‰‘•™…Õ±Ðˆ¸(€€€€€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰ÁÉ½™¥±•}¹…µ”ˆ¤€ôô€‰µ±Á•É˜ˆ(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰t¹•Ð ‰µ•ÍÍ…•Í}½µ¥ÑÑ•ˆ¤¥ÌQÉÕ”(€€€€€€€…ÍÍ•ÉÐ€‰µ•ÍÍ…•Ìˆ¹½Ð¥¸É•ÍÁl‰É•ÍÕ±Ð‰t(€€€€€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰Ñ¥Ñ±”ˆ¤€ôô€¡Í••¹l‰É•…Ñ•‰t°€‰™½É­•ˆ¤(€€€€€€€…ÍÍ•ÉÐ±•¸¡Í••¹l‰µÍÌ‰t¤€ôô€Ä(€€€€€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰±…Õ¹ ˆ¤¥Ì9½¹”(€€€€€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰±…Õ¹¡}É•…Ñ”ˆ¤¥Ì9½¹”(€€€€€€€¡¥±‘}Í¥€ôÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰t(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Ím¡¥±‘}Í¥‘ul‰ÁÉ½™¥±•}¡½µ”‰t€ôôÍÑÈ¡ÁÉ½™¥±•}¡½µ”¤(€€€€€€€€ŒQ¡”‰É…¹¡•9PµÕÍÐ‰”‰½Õ¹Ñ¼Ñ¡”Á…É•¹ÐÁÉ½™¥±”ÌÍÑ…Ñ”¹‘ˆƒŠP(€€€€€€€€Œ¹½Ð©ÕÍÐÑ¡”É½Ü¸=Ñ¡•ÉÝ¥Í”¥ÑÌ½Ý¸™±ÕÍ¡•Ì€¡…¹„±…Ñ•È½µÁÉ•ÍÍ¥½¸(€€€€€€€€ŒÉ½Ñ…Ñ¥½¸¤±…¹½¸Ñ¡”±…Õ¹ ‘ˆ°ÍÁ±¥ÑÑ¥¹œÑ¡”±¥¹•…”……¥¸¸(€€€€€€€…ÍÍ•ÉÐ¥Í¥¹ÍÑ…¹”¡Í••¸¹•Ð ‰…•¹Ñ}Í•ÍÍ¥½¹}‘ˆˆ¤°AÉ½™¥±•¤(€€€€€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰…•¹Ñ}Ýˆ¤€ôôÍÑÈ¡ÑµÁ}Á…Ñ ¤(€€€™¥¹…±±äè(€€€€€€€™½È¬¥¸±¥ÍÐ¡Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¤è(€€€€€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡¬°9½¹”¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}É•…Ñ•}Á•ÉÍ¥ÍÑÍ}Í••‘•‘}‰É…¹¡}¡¥±¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰‘•Í­Ñ½À‰É…¹ €¡Í•ÍÍ¥½¸¹É•…Ñ”Ý¥Ñ Á…É•¹Ñ}Í•ÍÍ¥½¹}¥€¬Í••‘•(€€€µ•ÍÍ…•Ì¤µÕÍÐÁ•ÉÍ¥ÍÐ¥ÑÌÉ½Ü€¬ÑÉ…¹ÍÉ¥ÁÐ¥µµ•‘¥…Ñ•±ä€ ŒäÌäÔä¤¸((€€€Q¡”É•¹‘•É•ÈÉ”µ™•Ñ¡•ÌÑ¡”™É•Í ¡¥±Ù¥„IMP…¹‘•™•É}¡¥ÍÑ½Éä(€€€¡å‘É…Ñ¥½¸É¥¡Ð…™Ñ•ÈÉ•…Ñ”ì‰½Ñ É•…Ñ¡”¸¸Õ¹Á•ÉÍ¥ÍÑ•¡¥±(€€€€ÐÀÑÌ½¡å‘É…Ñ•Ì•µÁÑä°Ñ¡”±¥•¹Ð™…¥°µ±…Ñ É•™ÕÍ•ÌÑ¼‰¥¹¥Ð°…¹Ñ¡”(€€€ÕÍ•È•ÑÌ…¸¥¹™¥¹¥Ñ”ÍÁ¥¹¹•ÈÝ¡½Í”½ÁÑ¥µ¥ÍÑ¥ŒÉ½ÜÙ…¹¥Í¡•Ì½¸É•ÍÑ…ÉÐ¸(€€€€ˆˆˆ((€€€±…ÍÌ}…­••¹Ðè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹µ½‘•°€ô€‰Ñ•ÍÐµµ½‘•°ˆ((€€€Í••¸è‘¥Ð€ôíô((€€€±…ÍÌ}…­•è(€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¹}Ñ¥Ñ±”¡Í•±˜°­•ä¤è(€€€€€€€€€€€Í••¹l‰Á…É•¹Ñ}Ñ¥Ñ±”‰t€ô­•ä(€€€€€€€€€€€É•ÑÕÉ¸€‰5äA…É•¹ÐM•ÍÍ¥½¸ˆ((€€€€€€€‘•˜•Ñ}¹•áÑ}Ñ¥Ñ±•}¥¹}±¥¹•…”¡Í•±˜°ÕÉÉ•¹Ð¤è(€€€€€€€€€€€É•ÑÕÉ¸˜‰íÕÉÉ•¹Ñô€ŒÈˆ((€€€€€€€‘•˜É•…Ñ•}Í•ÍÍ¥½¸¡Í•±˜°­•ä°€¨©­Ý…ÉÌ¤è(€€€€€€€€€€€Í••¹l‰É•…Ñ•‰t€ô­•ä(€€€€€€€€€€€Í••¹l‰Á…É•¹Ð‰t€ô­Ý…ÉÌ¹•Ð ‰Á…É•¹Ñ}Í•ÍÍ¥½¹}¥ˆ¤(€€€€€€€€€€€Í••¹l‰‰É…¹¡•‘}™É½´‰t€ô€¡­Ý…ÉÌ¹•Ð ‰µ½‘•±}½¹™¥œˆ¤½Èíô¤¹•Ð ‰}‰É…¹¡•‘}™É½´ˆ¤((€€€€€€€‘•˜…ÁÁ•¹‘}µ•ÍÍ…•Í}‰…Ñ ¡Í•±˜°Í•ÍÍ¥½¹}¥°µ•ÍÍ…•Ì°€¨©­Ý…ÉÌ¤è(€€€€€€€€€€€Í••¹l‰µ•ÍÍ…•Ì‰t€ô±¥ÍÐ¡µ•ÍÍ…•Ì¤((€€€€€€€‘•˜Í•Ñ}…ÕÑ½}Ñ¥Ñ±”¡Í•±˜°­•ä°Ñ¥Ñ±”°€¨°Í½ÕÉ”¤è(€€€€€€€€€€€Í••¹l‰Ñ¥Ñ±”‰t€ôÑ¥Ñ±”(€€€€€€€€€€€Í••¹l‰Ñ¥Ñ±•}Í½ÕÉ”‰t€ôÍ½ÕÉ”(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è}…­• ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}µ…­•}…•¹Ðˆ°±…µ‰‘„Í¥°­•ä°Í•ÍÍ¥½¹}‘ˆõ9½¹”°€¨©}­Üè}…­••¹Ð ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}M±…Í¡]½É­•Èˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„}„°€©„Èèì‰µ½‘•°ˆè€‰à‰ô¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÁÉ½‰•}É•‘•¹Ñ¥…±Ìˆ°±…µ‰‘„}„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Ý¥É•}…±±‰…­Ìˆ°±…µ‰‘„}Í¥è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„°€¨©­Üè9½¹”¤((€€€¥µÁ½ÉÐÑ½½±Ì¹…ÁÁÉ½Ù…°…Ì}…ÁÁÉ½Ù…°((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡}…ÁÁÉ½Ù…°°€‰É•¥ÍÑ•É}…Ñ•Ý…å}¹½Ñ¥™äˆ°±…µ‰‘„­•ä°ˆè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡}…ÁÁÉ½Ù…°°€‰±½…‘}Á•Éµ…¹•¹Ñ}…±±½Ý±¥ÍÐˆ°±…µ‰‘„è9½¹”¤((€€€Í••‘•€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰¡•±±¼™É½´Á…É•¹Ð‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰Á…É•¹ÐÉ•Á±ä‰ô°(€€€t((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹É•…Ñ”ˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€‰½±Ìˆè€äØ°(€€€€€€€€€€€€€€€€‰Í½ÕÉ”ˆè€‰‘•Í­Ñ½Àˆ°(€€€€€€€€€€€€€€€€‰Á…É•¹Ñ}Í•ÍÍ¥½¹}¥ˆè€ˆÈÀÈØÀàÈÍ|ÀàÐÄÄÍ|Ù‘”ÈÄÄˆ°(€€€€€€€€€€€€€€€€‰µ•ÍÍ…•ÌˆèÍ••‘•°(€€€€€€€€€€€ô°(€€€€€€€ô(€€€€¤((€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÀ°É•ÍÀ(€€€­•ä€ôÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÍÑ½É•‘}Í•ÍÍ¥½¹}¥‰t((€€€€ŒI½ÜÁ•ÉÍ¥ÍÑ•ÕÀ™É½¹ÐÝ¥Ñ ±¥¹•…”±¥¹­…”…¹„±¥¹•…”Ñ¥Ñ±”ƒŠP(€€€€Œ¹½Ð‘•™•ÉÉ•Ñ¼Ñ¡”™¥ÉÍÐÁÉ½µÁÐ¸(€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰É•…Ñ•ˆ¤€ôô­•ä(€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰Á…É•¹Ðˆ¤€ôô€ˆÈÀÈØÀàÈÍ|ÀàÐÄÄÍ|Ù‘”ÈÄÄˆ(€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰‰É…¹¡•‘}™É½´ˆ¤€ôô€ˆÈÀÈØÀàÈÍ|ÀàÐÄÄÍ|Ù‘”ÈÄÄˆ(€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰Ñ¥Ñ±”ˆ¤€ôô€‰5äA…É•¹ÐM•ÍÍ¥½¸€ŒÈˆ(€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰Ñ¥Ñ±•}Í½ÕÉ”ˆ¤€ôô€‰‘•É¥Ù•ˆ((€€€€ŒM••‘•ÑÉ…¹ÍÉ¥ÁÐ½Á¥•¥¹Ñ¼Ñ¡”‘ÕÉ…‰±”É½ÜÍ¼IMPÁÉ•™•Ñ …¹(€€€€Œ‘•™•É}¡¥ÍÑ½Éä¡å‘É…Ñ¥½¸‰½Ñ ™¥¹¥Ð¥µµ•‘¥…Ñ•±ä¸(€€€…ÍÍ•ÉÐ±•¸¡Í••¸¹•Ð ‰µ•ÍÍ…•Ìˆ¤½Èmt¤€ôô€È(€€€…ÍÍ•ÉÐÍ••¹l‰µ•ÍÍ…•Ì‰ulÁul‰½¹Ñ•¹Ð‰t€ôô€‰¡•±±¼™É½´Á…É•¹Ðˆ((€€€€ŒQ¡”±¥Ù”É•½É¹¼±½¹•ÈÅÕ•Õ•ÌÑ¡”Ñ¥Ñ±”ƒŠPÑ¡”…±É•…‘ä¡½±‘Ì¥Ð¸(€€€ÉÕ¹Ñ¥µ•}Í¥€ôÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰t(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÉÕ¹Ñ¥µ•}Í¥‘ul‰Á•¹‘¥¹}Ñ¥Ñ±”‰t¥Ì9½¹”((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡ÉÕ¹Ñ¥µ•}Í¥°9½¹”¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}‰É…¹¡}ÍÑ½É•‘}½Á¥•Í}Á…É•¹Ñ}¡¥ÍÑ½Éå}Ý¥Ñ¡½ÕÑ}É•ÑÕÉ¹¥¹}ÑÉ…¹ÍÉ¥ÁÐ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰]¡½±”µÍ•ÍÍ¥½¸‘•Í­Ñ½À‰É…¹¡•ÌÉ•…Ñ¡”Á…É•¹Ð¥¸Ñ¡”…Ñ•Ý…ä°¹½Ð¥¸Ñ¡”É•¹‘•É•È¸ˆˆˆ((€€€±…ÍÌ}M½Á”è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°‘ˆ¤è(€€€€€€€€€€€Í•±˜¹‘ˆ€ô‘ˆ((€€€€€€€‘•˜}}•¹Ñ•É}|¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸Í•±˜¹‘ˆ((€€€€€€€‘•˜}}•á¥Ñ}|¡Í•±˜°€©}…ÉÌ¤è(€€€€€€€€€€€É•ÑÕÉ¸…±Í”((€€€±…ÍÌ}…­•è(€€€€€€€‘•˜•Ñ}É•ÍÕµ•}½¹Ù•ÉÍ…Ñ¥½¹Ì¡Í•±˜°­•ä¤è(€€€€€€€€€€€…ÍÍ•ÉÐ­•ä€ôô€‰Á…É•¹Ðˆ(€€€€€€€€€€€É•ÑÕÉ¸mt°l(€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐÅÕ•ÍÑ¥½¸ˆ°€‰Ñ¥µ•ÍÑ…µÀˆè€Åô°(€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐ…¹ÍÝ•Èˆ°€‰Ñ¥µ•ÍÑ…µÀˆè€Éô°(€€€€€€€€€€€t((€€€±…ÍÌ}…­••¹Ðè(€€€€€€€µ½‘•°€ô€‰Ñ•ÍÐµµ½‘•°ˆ((€€€Í••¸è‘¥Ð€ôíô(€€€‘ˆ€ô}…­• ¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÁÉ½™¥±•}‘ˆˆ°±…µ‰‘„}Á…É…µÌè}M½Á”¡‘ˆ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°(€€€€€€€€‰}Í••‘}‰É…¹¡}É½Üˆ°(€€€€€€€±…µ‰‘„}É•½É°}­•ä°}Á…É•¹Ð°¡¥ÍÑ½Éä°€©}…ÉÌèÍ••¸¹ÕÁ‘…Ñ”¡¡¥ÍÑ½Éäõ±¥ÍÐ¡¡¥ÍÑ½Éä¤¤°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}µ…­•}…•¹Ðˆ°±…µ‰‘„€©}…ÉÌ°€¨©}­Ý…ÉÌè}…­••¹Ð ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„€©}…ÉÌèì‰µ½‘•°ˆè€‰Ñ•ÍÐµµ½‘•°‰ô¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹‰É…¹¡}ÍÑ½É•ˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€‰½±Ìˆè€äØ°(€€€€€€€€€€€€€€€€‰Á…É•¹Ñ}Í•ÍÍ¥½¹}¥ˆè€‰Á…É•¹Ðˆ°(€€€€€€€€€€€€€€€€‰Í½ÕÉ”ˆè€‰‘•Í­Ñ½Àˆ°(€€€€€€€€€€€ô°(€€€€€€€ô(€€€€¤((€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÀ°É•ÍÀ(€€€…ÍÍ•ÉÐmµ•ÍÍ…•l‰½¹Ñ•¹Ð‰t™½Èµ•ÍÍ…”¥¸Í••¹l‰¡¥ÍÑ½Éä‰ut€ôôl‰™¥ÉÍÐÅÕ•ÍÑ¥½¸ˆ°€‰™¥ÉÍÐ…¹ÍÝ•È‰t(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰µ•ÍÍ…•}½Õ¹Ð‰t€ôô€È(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰t¹•Ð ‰µ•ÍÍ…•Í}½µ¥ÑÑ•ˆ¤¥ÌQÉÕ”(€€€…ÍÍ•ÉÐ€‰µ•ÍÍ…•Ìˆ¹½Ð¥¸É•ÍÁl‰É•ÍÕ±Ð‰t(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}É•…Ñ•}‰É…¹¡}Í••‘}™…¥±ÕÉ•}‘½•Í}¹½Ñ}‰É•…­}É•…Ñ”¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰	•ÍÐµ•™™½ÉÐÁ•ÉÍ¥ÍÑ•¹”è„‰É½­•¸µÕÍÐ¹½Ð™…¥°Í•ÍÍ¥½¸¹É•…Ñ”¸ˆˆˆ((€€€±…ÍÌ}…­••¹Ðè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹µ½‘•°€ô€‰Ñ•ÍÐµµ½‘•°ˆ((€€€±…ÍÌ}	É½­•¹è(€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¹}Ñ¥Ñ±”¡Í•±˜°­•ä¤è(€€€€€€€€€€€É…¥Í”IÕ¹Ñ¥µ•ÉÉ½È ‰‘ˆ‘½Ý¸ˆ¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è}	É½­•¹ ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}µ…­•}…•¹Ðˆ°±…µ‰‘„Í¥°­•ä°Í•ÍÍ¥½¹}‘ˆõ9½¹”°€¨©}­Üè}…­••¹Ð ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}M±…Í¡]½É­•Èˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„}„°€©„Èèì‰µ½‘•°ˆè€‰à‰ô¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÁÉ½‰•}É•‘•¹Ñ¥…±Ìˆ°±…µ‰‘„}„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Ý¥É•}…±±‰…­Ìˆ°±…µ‰‘„}Í¥è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„°€¨©­Üè9½¹”¤((€€€¥µÁ½ÉÐÑ½½±Ì¹…ÁÁÉ½Ù…°…Ì}…ÁÁÉ½Ù…°((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡}…ÁÁÉ½Ù…°°€‰É•¥ÍÑ•É}…Ñ•Ý…å}¹½Ñ¥™äˆ°±…µ‰‘„­•ä°ˆè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡}…ÁÁÉ½Ù…°°€‰±½…‘}Á•Éµ…¹•¹Ñ}…±±½Ý±¥ÍÐˆ°±…µ‰‘„è9½¹”¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹É•…Ñ”ˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€‰Í½ÕÉ”ˆè€‰‘•Í­Ñ½Àˆ°(€€€€€€€€€€€€€€€€‰Á…É•¹Ñ}Í•ÍÍ¥½¹}¥ˆè€‰Á…É•¹Ð´Äˆ°(€€€€€€€€€€€€€€€€‰µ•ÍÍ…•Ìˆèmì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í••‰õt°(€€€€€€€€€€€ô°(€€€€€€€ô(€€€€¤((€€€€ŒÉ•…Ñ”¥ÑÍ•±˜ÍÑ¥±°ÍÕ••‘ÌƒŠPÑ¡”±…éä™¥ÉÍÐµÁÉ½µÁÐÁ…Ñ É•µ…¥¹Ì…Ì(€€€€ŒÑ¡”™…±±‰…¬™½ÈÑ¡”Í••¸(€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÀ((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡É•ÍÁl‰É•ÍÕ±Ð‰ul‰ÍÑ½É•‘}Í•ÍÍ¥½¹}¥‰t°9½¹”¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}É•…Ñ•}Í••‘}™…¥±ÕÉ•}…™Ñ•É}É½Ý}½µÁ•¹Í…Ñ•Ì¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰A…ÉÑ¥…°µ™…¥±ÕÉ”½µÁ•¹Í…Ñ¥½¸€ ŒäÌäÔäÉ•Ù¥•Ü¤è¥˜Ñ¡”É½Ü½µµ¥ÑÌ‰ÕÐ(€€€Ñ¡”ÑÉ…¹ÍÉ¥ÁÐ½Áä™…¥±Ì°Ñ¡”©ÕÍÐµÉ•…Ñ•¡¥±¥Ì1QÍ¼Ñ¡”±…éä(€€€™¥ÉÍÐµÁÉ½µÁÐ™…±±‰…¬…¸É•ÑÉä±•…¹±ä¸]¥Ñ¡½ÕÐÑ¡¥Ì°„‘ÕÉ…‰±”•µÁÑä(€€€É½Ü‘•™•…ÑÌ}•¹ÍÕÉ•}Í•ÍÍ¥½¹}‘‰}É½ÜÌ%9MIP=H%9=I…¹Ñ¡”É•¹‘•É•È(€€€™…¥°µ±…Ñ¡•Ì½¸„ÑÉ…¹ÍÉ¥ÁÐµ±•ÍÌÍ•ÍÍ¥½¸……¥¸¸ˆˆˆ((€€€±…ÍÌ}…­••¹Ðè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹µ½‘•°€ô€‰Ñ•ÍÐµµ½‘•°ˆ((€€€Í••¸è‘¥Ð€ôíô((€€€±…ÍÌ}…­•è(€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¹}Ñ¥Ñ±”¡Í•±˜°­•ä¤è(€€€€€€€€€€€É•ÑÕÉ¸€‰A…É•¹Ðˆ((€€€€€€€‘•˜•Ñ}¹•áÑ}Ñ¥Ñ±•}¥¹}±¥¹•…”¡Í•±˜°ÕÉÉ•¹Ð¤è(€€€€€€€€€€€É•ÑÕÉ¸˜‰íÕÉÉ•¹Ñô€ŒÈˆ((€€€€€€€‘•˜É•…Ñ•}Í•ÍÍ¥½¸¡Í•±˜°­•ä°€¨©­Ý…ÉÌ¤è(€€€€€€€€€€€Í••¹l‰É•…Ñ•‰t€ô­•ä((€€€€€€€‘•˜…ÁÁ•¹‘}µ•ÍÍ…•Í}‰…Ñ ¡Í•±˜°Í•ÍÍ¥½¹}¥°µ•ÍÍ…•Ì°€¨©­Ý…ÉÌ¤è(€€€€€€€€€€€É…¥Í”IÕ¹Ñ¥µ•ÉÉ½È ‰ÑÉ…¹ÍÉ¥ÁÐÝÉ¥Ñ”™…¥±•ˆ¤((€€€€€€€‘•˜‘•±•Ñ•}Í•ÍÍ¥½¸¡Í•±˜°Í•ÍÍ¥½¹}¥¤è(€€€€€€€€€€€Í••¹l‰‘•±•Ñ•‰t€ôÍ•ÍÍ¥½¹}¥(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è}…­• ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}µ…­•}…•¹Ðˆ°±…µ‰‘„Í¥°­•ä°Í•ÍÍ¥½¹}‘ˆõ9½¹”°€¨©}­Üè}…­••¹Ð ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}M±…Í¡]½É­•Èˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„}„°€©„Èèì‰µ½‘•°ˆè€‰à‰ô¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÁÉ½‰•}É•‘•¹Ñ¥…±Ìˆ°±…µ‰‘„}„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Ý¥É•}…±±‰…­Ìˆ°±…µ‰‘„}Í¥è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„°€¨©­Üè9½¹”¤((€€€¥µÁ½ÉÐÑ½½±Ì¹…ÁÁÉ½Ù…°…Ì}…ÁÁÉ½Ù…°((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡}…ÁÁÉ½Ù…°°€‰É•¥ÍÑ•É}…Ñ•Ý…å}¹½Ñ¥™äˆ°±…µ‰‘„­•ä°ˆè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡}…ÁÁÉ½Ù…°°€‰±½…‘}Á•Éµ…¹•¹Ñ}…±±½Ý±¥ÍÐˆ°±…µ‰‘„è9½¹”¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹É•…Ñ”ˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€‰Í½ÕÉ”ˆè€‰‘•Í­Ñ½Àˆ°(€€€€€€€€€€€€€€€€‰Á…É•¹Ñ}Í•ÍÍ¥½¹}¥ˆè€‰Á…É•¹Ð´Äˆ°(€€€€€€€€€€€€€€€€‰Ñ¥Ñ±”ˆè€‰5ä	É…¹ ˆ°(€€€€€€€€€€€€€€€€‰µ•ÍÍ…•Ìˆèmì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í••‰õt°(€€€€€€€€€€€ô°(€€€€€€€ô(€€€€¤((€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÀ(€€€­•ä€ôÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÍÑ½É•‘}Í•ÍÍ¥½¹}¥‰t(€€€€ŒQ¡”¡…±˜µÝÉ¥ÑÑ•¸¡¥±Ý…ÌÉ½±±•‰…¬ƒŠP¹¼‘ÕÉ…‰±”•µÁÑäÉ½Ü±•™ÐÑ¼(€€€€ŒÍ¡…‘½ÜÑ¡”±…éäÍ••Á…Ñ ¸(€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰‘•±•Ñ•ˆ¤€ôô­•ä(€€€€ŒÁ•¹‘¥¹}Ñ¥Ñ±”ÍÕÉÙ¥Ù•è¥ÐÍÑ¥±°±…¹‘ÌÙ¥„Ñ¡”±…éäÁ½ÍÐµÑÕÉ¸…ÁÁ±ä¸(€€€ÉÕ¹Ñ¥µ•}Í¥€ôÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰t(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÉÕ¹Ñ¥µ•}Í¥‘ul‰Á•¹‘¥¹}Ñ¥Ñ±”‰t€ôô€‰5ä	É…¹ ˆ((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡ÉÕ¹Ñ¥µ•}Í¥°9½¹”¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}É•…Ñ•}Í••‘}‘¥Í­}™Õ±±}­••ÁÍ}É½Ý}™½É}É•ÑÉä¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰¥Í¬µ™Õ±°¥Ì9=P½µÁ•¹Í…Ñ•èÑ¡”É½ÜÍÑ…åÌ€¡‘•±•Ñ¥¹œ‘…Ñ„½¸„™Õ±°(€€€‘¥Í¬…¸µ…­”Ñ¡¥¹ÌÝ½ÉÍ”¤°É•…Ñ”ÍÑ¥±°ÍÕ••‘Ì°…¹Ñ¡”™…¥±ÕÉ”¥Ì(€€€½‰Í•ÉÙ…‰±”…ÐÝ…É¹¥¹œ±•Ù•°€ ŒäÌäÔäÉ•Ù¥•Ü¤¸ˆˆˆ((€€€¥µÁ½ÉÐ±½¥¹œ…Ì}±½¥¹œ((€€€±…ÍÌ}…­••¹Ðè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹µ½‘•°€ô€‰Ñ•ÍÐµµ½‘•°ˆ((€€€±…ÍÌ}…­•è(€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¹}Ñ¥Ñ±”¡Í•±˜°­•ä¤è(€€€€€€€€€€€É•ÑÕÉ¸€‰A…É•¹Ðˆ((€€€€€€€‘•˜•Ñ}¹•áÑ}Ñ¥Ñ±•}¥¹}±¥¹•…”¡Í•±˜°ÕÉÉ•¹Ð¤è(€€€€€€€€€€€É•ÑÕÉ¸˜‰íÕÉÉ•¹Ñô€ŒÈˆ((€€€€€€€‘•˜É•…Ñ•}Í•ÍÍ¥½¸¡Í•±˜°­•ä°€¨©­Ý…ÉÌ¤è(€€€€€€€€€€€Á…ÍÌ((€€€€€€€‘•˜…ÁÁ•¹‘}µ•ÍÍ…•Í}‰…Ñ ¡Í•±˜°Í•ÍÍ¥½¹}¥°µ•ÍÍ…•Ì°€¨©­Ý…ÉÌ¤è(€€€€€€€€€€€É…¥Í”=MÉÉ½È Èà°€‰9¼ÍÁ…”±•™Ð½¸‘•Ù¥”ˆ¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è}…­• ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}µ…­•}…•¹Ðˆ°±…µ‰‘„Í¥°­•ä°Í•ÍÍ¥½¹}‘ˆõ9½¹”°€¨©}­Üè}…­••¹Ð ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}M±…Í¡]½É­•Èˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„}„°€©„Èèì‰µ½‘•°ˆè€‰à‰ô¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÁÉ½‰•}É•‘•¹Ñ¥…±Ìˆ°±…µ‰‘„}„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Ý¥É•}…±±‰…­Ìˆ°±…µ‰‘„}Í¥è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„°€¨©­Üè9½¹”¤((€€€¥µÁ½ÉÐÑ½½±Ì¹…ÁÁÉ½Ù…°…Ì}…ÁÁÉ½Ù…°((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡}…ÁÁÉ½Ù…°°€‰É•¥ÍÑ•É}…Ñ•Ý…å}¹½Ñ¥™äˆ°±…µ‰‘„­•ä°ˆè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡}…ÁÁÉ½Ù…°°€‰±½…‘}Á•Éµ…¹•¹Ñ}…±±½Ý±¥ÍÐˆ°±…µ‰‘„è9½¹”¤((€€€É•½É‘Ìè±¥ÍÐ€ômt((€€€±…ÍÌ}…ÁÑÕÉ”¡}±½¥¹œ¹!…¹‘±•È¤è(€€€€€€€‘•˜•µ¥Ð¡Í•±˜°É•½É¤è(€€€€€€€€€€€É•½É‘Ì¹…ÁÁ•¹¡É•½É¤((€€€¡…¹‘±•È€ô}…ÁÑÕÉ”¡±•Ù•°õ}±½¥¹œ¹]I9%9¤(€€€É½½Ð€ô}±½¥¹œ¹•Ñ1½•È ¤(€€€É½½Ð¹…‘‘!…¹‘±•È¡¡…¹‘±•È¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹É•…Ñ”ˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í½ÕÉ”ˆè€‰‘•Í­Ñ½Àˆ°(€€€€€€€€€€€€€€€€€€€€‰Á…É•¹Ñ}Í•ÍÍ¥½¹}¥ˆè€‰Á…É•¹Ð´Äˆ°(€€€€€€€€€€€€€€€€€€€€‰µ•ÍÍ…•Ìˆèmì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í••‰õt°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€É½½Ð¹É•µ½Ù•!…¹‘±•È¡¡…¹‘±•È¤((€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÀ(€€€€ŒQ¡”™…¥±ÕÉ”ÍÕÉ™…•…Ð]I9%9€¡½‰Í•ÉÙ…‰±”¤°¹½Ð‰ÕÉ¥•…Ð‘•‰Õœ¸(€€€Ý…É¹¥¹Ì€ômÈ™½ÈÈ¥¸É•½É‘Ì¥˜È¹±•Ù•±¹¼€øô}±½¥¹œ¹]I9%9t(€€€…ÍÍ•ÉÐÝ…É¹¥¹Ì((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡É•ÍÁl‰É•ÍÕ±Ð‰ul‰ÍÑ½É•‘}Í•ÍÍ¥½¹}¥‰t°9½¹”¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}É•…Ñ•}Ý¥Ñ¡½ÕÑ}Á…É•¹Ñ}ÍÑ¥±±}‘•™•ÉÍ}É½Ü¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰A±…¥¸‘É…™ÑÌ­••ÀÑ¡”±…éäµÉ½Ü½¹ÑÉ…Ðè¹¼Á…É•¹Ð€¬¹¼•áÁ±¥¥Ð‰É…¹ (€€€¥¹Ñ•¹Ðµ•…¹Ì¹¼•…•ÈÁ•ÉÍ¥ÍÑ•¹”€¡Ñ¡”½É¥¥¹…°‘É…™Ðµ¡å¥•¹”¥¹Ù…É¥…¹Ð¤¸ˆˆˆ((€€€±…ÍÌ}…­••¹Ðè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹µ½‘•°€ô€‰Ñ•ÍÐµµ½‘•°ˆ((€€€…±±Ìè‘¥Ð€ôì‰É•…Ñ”ˆè€Áô((€€€±…ÍÌ}…­•è(€€€€€€€‘•˜É•…Ñ•}Í•ÍÍ¥½¸¡Í•±˜°€©„°€¨©¬¤è(€€€€€€€€€€€…±±Íl‰É•…Ñ”‰t€¬ô€Ä((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è}…­• ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}µ…­•}…•¹Ðˆ°±…µ‰‘„Í¥°­•ä°Í•ÍÍ¥½¹}‘ˆõ9½¹”°€¨©}­Üè}…­••¹Ð ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}M±…Í¡]½É­•Èˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„}„°€©„Èèì‰µ½‘•°ˆè€‰à‰ô¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÁÉ½‰•}É•‘•¹Ñ¥…±Ìˆ°±…µ‰‘„}„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Ý¥É•}…±±‰…­Ìˆ°±…µ‰‘„}Í¥è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„°€¨©­Üè9½¹”¤((€€€¥µÁ½ÉÐÑ½½±Ì¹…ÁÁÉ½Ù…°…Ì}…ÁÁÉ½Ù…°((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡}…ÁÁÉ½Ù…°°€‰É•¥ÍÑ•É}…Ñ•Ý…å}¹½Ñ¥™äˆ°±…µ‰‘„­•ä°ˆè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡}…ÁÁÉ½Ù…°°€‰±½…‘}Á•Éµ…¹•¹Ñ}…±±½Ý±¥ÍÐˆ°±…µ‰‘„è9½¹”¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹É•…Ñ”ˆ°€‰Á…É…µÌˆèì‰½±Ìˆè€àÁõô(€€€€¤(€€€Í¥€ôÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰t(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÍ¥‘ul‰…•¹Ñ}É•…‘ä‰t¹Ý…¥Ð¡Ñ¥µ•½ÕÐôÈ¸À¤((€€€…ÍÍ•ÉÐ…±±Íl‰É•…Ñ”‰t€ôô€À°€‰Á±…¥¸‘É…™ÑÌµÕÍÐ¹½ÐÁ•ÉÍ¥ÍÐ•…•É±äˆ((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡Í¥°9½¹”¤()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}‰É…¹¡}¥¹ÍÑ…±±Í}Á…É•¹Ñ}ÁÉ½™¥±•}Í•É•Ñ}Í½Á”¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰Q¡”‰É…¹¡•…•¹ÐµÕÍÐ‰”‰Õ¥±ÐÕ¹‘•ÈÑ¡”Á…É•¹ÐÁÉ½™¥±”ÌÍ•É•ÑÌ¸((€€€Í•ÍÍ¥½¸¹‰É…¹ …±É•…‘ä‰¥¹‘ÌÑ¡”Á…É•¹ÐÌ!I5M}!=5…¹ÍÑ…Ñ”¹‘ˆ°‰ÕÐÑ¡”(€€€Í•É•ÐÍ½Á”¥ÌÝ¡…Ðµ…­•Ì•Ñ}Í•É•Ð ¤É•Í½±Ù”Ñ¡…ÐÁÉ½™¥±”Ì€¹•¹Ø¸]¥Ñ¡½ÕÐ(€€€¥ÐÑ¡”‰Õ¥±™…±±ÌÑ¡É½Õ Ñ¼ÁÉ½•ÍÌ½Ì¹•¹Ù¥É½¸ƒŠPÑ¡”1U9 ÁÉ½™¥±”Ì(€€€É•‘•¹Ñ¥…±ÌƒŠPÝ¡¥ ¥Ì•á…Ñ±äÑ¡”É½ÍÌµÁÉ½™¥±”É•Í½±ÕÑ¥½¸€ŒØÜØÀÔ™¥á•™½È(€€€Í•ÍÍ¥½¸¹É•…Ñ”€¼Í•ÍÍ¥½¸¹É•ÍÕµ”¸(€€€€ˆˆˆ(€€€¥µÁ½ÉÐÑ¡É•…‘¥¹œ((€€€™É½´…•¹Ð¹Í•É•Ñ}Í½Á”¥µÁ½ÉÐÕÉÉ•¹Ñ}Í•É•Ñ}Í½Á”((€€€ÁÉ½™¥±•}¡½µ”€ôÑµÁ}Á…Ñ €¼€‰ÁÉ½™¥±•Ìˆ€¼€‰µ±Á•É˜ˆ(€€€ÁÉ½™¥±•}¡½µ”¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}!=5ˆ°ÍÑÈ¡ÑµÁ}Á…Ñ ¤¤(€€€€¡ÁÉ½™¥±•}¡½µ”€¼€ˆ¹•¹Øˆ¤¹ÝÉ¥Ñ•}Ñ•áÐ (€€€€€€€€‰AI=a5=a}Q=-8õµ±Á•É˜µÍ•É•Ñq¸ˆ°•¹½‘¥¹œô‰ÕÑ˜´àˆ(€€€€¤(€€€Í••¸è‘¥Ð€ôì‰µÍÌˆèmuô((€€€±…ÍÌAÉ½™¥±•è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°‘‰}Á…Ñ õ9½¹”¤è(€€€€€€€€€€€Á…ÍÌ((€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¹}Ñ¥Ñ±”¡Í•±˜°}­•ä¤è(€€€€€€€€€€€É•ÑÕÉ¸€‰Á…É•¹Ðˆ((€€€€€€€‘•˜•Ñ}¹•áÑ}Ñ¥Ñ±•}¥¹}±¥¹•…”¡Í•±˜°ÕÉÉ•¹Ð¤è(€€€€€€€€€€€É•ÑÕÉ¸˜‰íÕÉÉ•¹Ñô€¡‰É…¹ ¤ˆ((€€€€€€€‘•˜É•…Ñ•}Í•ÍÍ¥½¸¡Í•±˜°¹•Ý}­•ä°€¨©­Ý…ÉÌ¤è(€€€€€€€€€€€Í••¹l‰É•…Ñ•‰t€ô¹•Ý}­•ä((€€€€€€€‘•˜…ÁÁ•¹‘}µ•ÍÍ…”¡Í•±˜°€¨©­Ý…ÉÌ¤è(€€€€€€€€€€€Í••¹l‰µÍÌ‰t¹…ÁÁ•¹¡­Ý…ÉÌ¤((€€€€€€€‘•˜…ÁÁ•¹‘}µ•ÍÍ…•Í}‰…Ñ ¡Í•±˜°Í•ÍÍ¥½¹}¥°µ•ÍÍ…•Ì°€¨©­Ý…ÉÌ¤è(€€€€€€€€€€€™½È´¥¸µ•ÍÍ…•Ìè(€€€€€€€€€€€€€€€Í••¹l‰µÍÌ‰t¹…ÁÁ•¹¡‘¥Ð¡´°Í•ÍÍ¥½¹}¥õÍ•ÍÍ¥½¹}¥¤¤(€€€€€€€€€€€É•ÑÕÉ¸±¥ÍÐ¡É…¹” Ä°±•¸¡µ•ÍÍ…•Ì¤€¬€Ä¤¤((€€€€€€€‘•˜Í•Ñ}Í•ÍÍ¥½¹}Ñ¥Ñ±”¡Í•±˜°­•ä°Ñ¥Ñ±”¤è(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¸¡Í•±˜°­•ä¤è(€€€€€€€€€€€É•ÑÕÉ¸ì‰¥ˆè­•ä°€‰ÝˆèÍÑÈ¡ÑµÁ}Á…Ñ ¥ô((€€€€€€€‘•˜ÕÁ‘…Ñ•}Í•ÍÍ¥½¹}Ý¡Í•±˜°€©„°€¨©¬¤è(€€€€€€€€€€€É•ÑÕÉ¸9½¹”((€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸9½¹”((€€€±…ÍÌ…­••¹Ðè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹µ½‘•°€ô€‰Ñ•ÍÐµµ½‘•°ˆ(€€€€€€€€€€€Í•±˜¹Í•ÍÍ¥½¹}¥€ô9½¹”((€€€Á…É•¹Ð€ôì(€€€€€€€€‰Í•ÍÍ¥½¹}­•äˆè€‰Á…É•¹Ðµ­•äˆ°(€€€€€€€€‰¡¥ÍÑ½Éäˆèmì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰¡¤‰õt°(€€€€€€€€‰¡¥ÍÑ½Éå}±½¬ˆèÑ¡É•…‘¥¹œ¹1½¬ ¤°(€€€€€€€€‰ÉÕ¹¹¥¹œˆè…±Í”°(€€€€€€€€‰½±Ìˆè€àÀ°(€€€€€€€€‰ÁÉ½™¥±•}¡½µ”ˆèÍÑÈ¡ÁÉ½™¥±•}¡½µ”¤°(€€€€€€€€‰Í½ÕÉ”ˆè€‰ÑÕ¤ˆ°(€€€€€€€€‰…•¹Ðˆè…­••¹Ð ¤°(€€€€€€€€‰É•…Ñ•‘}…Ðˆè€Ä¸À°(€€€€€€€€‰±…ÍÑ}…Ñ¥Ù”ˆè€Ä¸À°(€€€€€€€€‰ÝˆèÍÑÈ¡ÑµÁ}Á…Ñ ¤°(€€€ô(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Á…É•¹Ð‰t€ôÁ…É•¹Ð(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„èAÉ½™¥±• ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰¡•Éµ•Í}ÍÑ…Ñ•}É•¥ÍÑÉä¹…ÅÕ¥É”ˆ°AÉ½™¥±•¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±…¥µ}…Ñ¥Ù•}Í•ÍÍ¥½¹}Í±½Ðˆ°±…µ‰‘„€©„°€¨©¬è€¡9½¹”°9½¹”¤¤((€€€‘•˜}™…­•}µ…­•}…•¹Ð ©„°€¨©¬¤è(€€€€€€€Í½Á”€ôÕÉÉ•¹Ñ}Í•É•Ñ}Í½Á” ¤(€€€€€€€Í••¹l‰Í½Á”‰t€ô‘¥Ð¡Í½Á”¤¥˜Í½Á”•±Í”9½¹”(€€€€€€€É•ÑÕÉ¸…­••¹Ð ¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}µ…­•}…•¹Ðˆ°}™…­•}µ…­•}…•¹Ð¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•Ñ}Í•ÍÍ¥½¹}½¹Ñ•áÐˆ°±…µ‰‘„€©„°€¨©¬èíô¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±•…É}Í•ÍÍ¥½¹}½¹Ñ•áÐˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}É•Í½±Ù•}µ½‘•°ˆ°±…µ‰‘„è€‰Ñ•ÍÐµµ½‘•°ˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}Ýˆ°±…µ‰‘„ÌèÍÑÈ¡ÑµÁ}Á…Ñ ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}É•¥ÍÑ•É}Í•ÍÍ¥½¹}Ýˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}…ÑÑ…¡}Ý½É­•Èˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹‰É…¹ ˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Á…É•¹Ðˆ°€‰¹…µ”ˆè€‰™½É­•‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÀ°É•ÍÀ(€€€€€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰Í½Á”ˆ¤€ôôì‰AI=a5=a}Q=-8ˆè€‰µ±Á•É˜µÍ•É•Ð‰ô(€€€™¥¹…±±äè(€€€€€€€™½È¬¥¸±¥ÍÐ¡Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¤è(€€€€€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡¬°9½¹”¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}‰É…¹¡}ÕÍ•Í}Á•ÉÍ¥ÍÑ•‘}‘¥ÍÁ±…å}¡¥ÍÑ½Éå}…™Ñ•É}½µÁ…Ñ¥½¸¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰±¥Ù”‰É…¹ µÕÍÐ½ÁäÑ¡”½µÁ±•Ñ”Ù¥Í¥‰±”ÑÉ…¹ÍÉ¥ÁÐ°¹½ÐÑ¡”½µÁ…Ñ•µ½‘•°Ñ…¥°¸ˆˆˆ(€€€ÁÉ½™¥±•}¡½µ”€ôÑµÁ}Á…Ñ €¼€‰ÁÉ½™¥±•Ìˆ€¼€‰µ±Á•É˜ˆ(€€€ÁÉ½™¥±•}¡½µ”¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}!=5ˆ°ÍÑÈ¡ÑµÁ}Á…Ñ ¤¤(€€€Í••¸è‘¥Ð€ôì‰µÍÌˆèmuô((€€€‘¥ÍÁ±…å}¡¥ÍÑ½Éä€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐÅÕ•ÍÑ¥½¸ˆ°€‰Ñ¥µ•ÍÑ…µÀˆè€Ä¸Áô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐ…¹ÍÝ•Èˆ°€‰Ñ¥µ•ÍÑ…µÀˆè€È¸Áô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€ˆˆ°€‰Ñ½½±}…±±Ìˆèmì‰¥ˆè€‰…±°´Ä‰õuô°(€€€€€€€ì‰É½±”ˆè€‰Ñ½½°ˆ°€‰½¹Ñ•¹Ðˆè€‰Ñ½½°½ÕÑÁÕÐˆ°€‰Ñ½½±}…±±}¥ˆè€‰…±°´Ä‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹ÅÕ•ÍÑ¥½¸ˆ°€‰Ñ¥µ•ÍÑ…µÀˆè€Ì¸Áô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹…¹ÍÝ•Èˆ°€‰Ñ¥µ•ÍÑ…µÀˆè€Ð¸Áô°(€€€t((€€€±…ÍÌ1…Õ¹¡è(€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¹}Ñ¥Ñ±”¡Í•±˜°}­•ä¤è(€€€€€€€€€€€É•ÑÕÉ¸€‰±…Õ¹ ˆ((€€€±…ÍÌAÉ½™¥±•è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°‘‰}Á…Ñ õ9½¹”¤è(€€€€€€€€€€€Í••¸¹Í•Ñ‘•™…Õ±Ð ‰¥¹¥ÑÌˆ°€À¤(€€€€€€€€€€€Í••¹l‰¥¹¥ÑÌ‰t€¬ô€Ä((€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¹}Ñ¥Ñ±”¡Í•±˜°}­•ä¤è(€€€€€€€€€€€É•ÑÕÉ¸€‰Á…É•¹Ðˆ((€€€€€€€‘•˜•Ñ}¹•áÑ}Ñ¥Ñ±•}¥¹}±¥¹•…”¡Í•±˜°ÕÉÉ•¹Ð¤è(€€€€€€€€€€€É•ÑÕÉ¸˜‰íÕÉÉ•¹Ñô€¡‰É…¹ ¤ˆ((€€€€€€€‘•˜•Ñ}É•ÍÕµ•}½¹Ù•ÉÍ…Ñ¥½¹Ì¡Í•±˜°­•ä¤è(€€€€€€€€€€€…ÍÍ•ÉÐ­•ä€ôô€‰Á…É•¹Ðµ­•äˆ(€€€€€€€€€€€€ŒQ¡”µ½‘•°ÁÉ½©•Ñ¥½¸¡…Ì…±É•…‘ä‰••¸½µÁ…Ñ•Ñ¼„ÍÕµµ…Éä€¬Ñ…¥°ì(€€€€€€€€€€€€ŒÑ¡”‘¥ÍÁ±…äÁÉ½©•Ñ¥½¸ÍÑ¥±°½¹Ñ…¥¹Ì•Ù•ÉäÙ¥Í¥‰±”ÑÕÉ¸¸(€€€€€€€€€€€É•ÑÕÉ¸€ (€€€€€€€€€€€€€€€mì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰½µÁ…ÐÍÕµµ…Éä‰õt°(€€€€€€€€€€€€€€€‘¥ÍÁ±…å}¡¥ÍÑ½Éä°(€€€€€€€€€€€€¤((€€€€€€€‘•˜É•…Ñ•}Í•ÍÍ¥½¸¡Í•±˜°}¹•Ý}­•ä°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€É•ÑÕÉ¸9½¹”((€€€€€€€‘•˜…ÁÁ•¹‘}µ•ÍÍ…”¡Í•±˜°€¨©­Ý…ÉÌ¤è(€€€€€€€€€€€Í••¹l‰µÍÌ‰t¹…ÁÁ•¹¡­Ý…ÉÌ¤((€€€€€€€‘•˜…ÁÁ•¹‘}µ•ÍÍ…•Í}‰…Ñ ¡Í•±˜°Í•ÍÍ¥½¹}¥°µ•ÍÍ…•Ì°€¨©­Ý…ÉÌ¤è(€€€€€€€€€€€™½Èµ•ÍÍ…”¥¸µ•ÍÍ…•Ìè(€€€€€€€€€€€€€€€Í••¹l‰µÍÌ‰t¹…ÁÁ•¹¡‘¥Ð¡µ•ÍÍ…”°Í•ÍÍ¥½¹}¥õÍ•ÍÍ¥½¹}¥¤¤(€€€€€€€€€€€É•ÑÕÉ¸±¥ÍÐ¡É…¹” Ä°±•¸¡µ•ÍÍ…•Ì¤€¬€Ä¤¤((€€€€€€€‘•˜Í•Ñ}Í•ÍÍ¥½¹}Ñ¥Ñ±”¡Í•±˜°}­•ä°}Ñ¥Ñ±”¤è(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€€€€€‘•˜Í•Ñ}…ÕÑ½}Ñ¥Ñ±”¡Í•±˜°}­•ä°}Ñ¥Ñ±”°€¨°Í½ÕÉ”ô‰±±´ˆ¤è(€€€€€€€€€€€Í••¹l‰Ñ¥Ñ±•}Í½ÕÉ”‰t€ôÍ½ÕÉ”(€€€€€€€€€€€É•ÑÕÉ¸QÉÕ”((€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¸¡Í•±˜°­•ä¤è(€€€€€€€€€€€É•ÑÕÉ¸ì‰¥ˆè­•ä°€‰ÝˆèÍÑÈ¡ÑµÁ}Á…Ñ ¥ô((€€€€€€€‘•˜ÕÁ‘…Ñ•}Í•ÍÍ¥½¹}Ý¡Í•±˜°€©…ÉÌ°€¨©­Ý…ÉÌ¤è(€€€€€€€€€€€É•ÑÕÉ¸9½¹”((€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸9½¹”((€€€±…ÍÌ…­••¹Ðè(€€€€€€€µ½‘•°€ô€‰Ñ•ÍÐµµ½‘•°ˆ(€€€€€€€Í•ÍÍ¥½¹}¥€ô9½¹”((€€€Á…É•¹Ð€ôì(€€€€€€€€‰Í•ÍÍ¥½¹}­•äˆè€‰Á…É•¹Ðµ­•äˆ°(€€€€€€€€ŒQ¡¥Ì¥ÌÑ¡”µ½‘•°µ™•ÁÉ½©•Ñ¥½¸…™Ñ•È½µÁ…Ñ¥½¸èÑ¡”½±ÑÕÉ¹Ì…É”(€€€€€€€€Œ…‰Í•¹Ð¡•É”•Ù•¸Ñ¡½Õ Ñ¡”‘¥ÍÁ±…äÁÉ½©•Ñ¥½¸…‰½Ù”É•Ñ…¥¹ÌÑ¡•´¸(€€€€€€€€‰¡¥ÍÑ½Éäˆèl(€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰½µÁ…ÐÍÕµµ…Éä‰ô°(€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹ÅÕ•ÍÑ¥½¸‰ô°(€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹…¹ÍÝ•È‰ô°(€€€€€€€t°(€€€€€€€€‰¡¥ÍÑ½Éå}±½¬ˆèÑ¡É•…‘¥¹œ¹1½¬ ¤°(€€€€€€€€‰ÉÕ¹¹¥¹œˆè…±Í”°(€€€€€€€€‰½±Ìˆè€àÀ°(€€€€€€€€‰ÁÉ½™¥±•}¡½µ”ˆèÍÑÈ¡ÁÉ½™¥±•}¡½µ”¤°(€€€€€€€€‰Í½ÕÉ”ˆè€‰ÑÕ¤ˆ°(€€€€€€€€‰…•¹Ðˆè…­••¹Ð ¤°(€€€€€€€€‰É•…Ñ•‘}…Ðˆè€Ä¸À°(€€€€€€€€‰±…ÍÑ}…Ñ¥Ù”ˆè€Ä¸À°(€€€€€€€€‰ÝˆèÍÑÈ¡ÑµÁ}Á…Ñ ¤°(€€€ô(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Á…É•¹Ð‰t€ôÁ…É•¹Ð(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è1…Õ¹¡ ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰¡•Éµ•Í}ÍÑ…Ñ•}É•¥ÍÑÉä¹…ÅÕ¥É”ˆ°AÉ½™¥±•¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±…¥µ}…Ñ¥Ù•}Í•ÍÍ¥½¹}Í±½Ðˆ°±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌè€¡9½¹”°9½¹”¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}µ…­•}…•¹Ðˆ°±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌè…­••¹Ð ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•Ñ}Í•ÍÍ¥½¹}½¹Ñ•áÐˆ°±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌèíô¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±•…É}Í•ÍÍ¥½¹}½¹Ñ•áÐˆ°±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}É•Í½±Ù•}µ½‘•°ˆ°±…µ‰‘„è€‰Ñ•ÍÐµµ½‘•°ˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}Ýˆ°±…µ‰‘„}Í•ÍÍ¥½¸èÍÑÈ¡ÑµÁ}Á…Ñ ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}É•¥ÍÑ•É}Í•ÍÍ¥½¹}Ýˆ°±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}…ÑÑ…¡}Ý½É­•Èˆ°±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌè9½¹”¤((€€€ÑÉäè(€€€€€€€É•ÍÁ½¹Í”€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹‰É…¹ ˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Á…É•¹Ðˆ°€‰½Õ¹Ðˆè€Ñô°(€€€€€€€€€€€ô(€€€€€€€€¤((€€€€€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÁ½¹Í”°É•ÍÁ½¹Í”(€€€€€€€…ÍÍ•ÉÐÍ••¸¹•Ð ‰Ñ¥Ñ±•}Í½ÕÉ”ˆ¤€ôô€‰‘•É¥Ù•ˆ(€€€€€€€…ÍÍ•ÉÐmµ•ÍÍ…•l‰½¹Ñ•¹Ð‰t™½Èµ•ÍÍ…”¥¸Í••¹l‰µÍÌ‰ut€ôôl(€€€€€€€€€€€€‰™¥ÉÍÐÅÕ•ÍÑ¥½¸ˆ°(€€€€€€€€€€€€‰™¥ÉÍÐ…¹ÍÝ•Èˆ°(€€€€€€€€€€€€‰Í•½¹ÅÕ•ÍÑ¥½¸ˆ°(€€€€€€€€€€€€‰Í•½¹…¹ÍÝ•Èˆ°(€€€€€€€t(€€€€€€€…ÍÍ•ÉÐmµ•ÍÍ…•l‰Ñ•áÐ‰t™½Èµ•ÍÍ…”¥¸É•ÍÁ½¹Í•l‰É•ÍÕ±Ð‰ul‰µ•ÍÍ…•Ì‰ut€ôôl(€€€€€€€€€€€€‰™¥ÉÍÐÅÕ•ÍÑ¥½¸ˆ°(€€€€€€€€€€€€‰™¥ÉÍÐ…¹ÍÝ•Èˆ°(€€€€€€€€€€€€‰Í•½¹ÅÕ•ÍÑ¥½¸ˆ°(€€€€€€€€€€€€‰Í•½¹…¹ÍÝ•Èˆ°(€€€€€€€t(€€€™¥¹…±±äè(€€€€€€€™½È­•ä¥¸±¥ÍÐ¡Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¤è(€€€€€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡­•ä°9½¹”¤(((((Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(Œµ½‘•°¹½ÁÑ¥½¹ÌƒŠPÕÉ…Ñ•µ±¥ÍÐÁ…É¥ÑäÝ¥Ñ ¡•Éµ•Ìµ½‘•±€…¹±…ÍÍ¥Œ€½µ½‘•°(Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(()‘•˜Ñ•ÍÑ}µ½‘•±}½ÁÑ¥½¹Í}‘½•Í}¹½Ñ}½Ù•ÉÝÉ¥Ñ•}ÕÉ…Ñ•‘}µ½‘•±Ì¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰Q¡”QU$µ½‘•°¹½ÁÑ¥½¹Ì¡…¹‘±•ÈµÕÍÐÍÕÉ™…”Ñ¡”Í…µ”ÕÉ…Ñ•µ½‘•°(€€€±¥ÍÐ…Ì¡•Éµ•Ìµ½‘•±€…¹Ñ¡”±…ÍÍ¥Œ1$€½µ½‘•°Á¥­•È¸((€€€I•É•ÍÍ¥½¸è•…É±¥•ÈÙ•ÉÍ¥½¹Ì½˜Ñ¡¥Ì¡…¹‘±•ÈÕ¹½¹‘¥Ñ¥½¹…±±äÉ•Á±…•(€€€•… ÁÉ½Ù¥‘•ÈÌÕÉ…Ñ•µ½‘•±Í€™¥•±Ý¥Ñ ÁÉ½Ù¥‘•É}µ½‘•±}¥‘Ì ¥€(€€€€¡±¥Ù”€½µ½‘•±Ì…Ñ…±½œ¤¸€Q¡…ÐÁÕ±±•¥¸¡Õ¹‘É•‘Ì½˜¹½¸µ…•¹Ñ¥Œµ½‘•±Ì(€€€™½ÈÁÉ½Ù¥‘•ÉÌ±¥­”9½ÕÌÝ¡½Í”€½µ½‘•±Ì•¹‘Á½¥¹ÐÉ•ÑÕÉ¹Ì¥µ…”½Ù¥‘•¼(€€€•¹•É…Ñ½ÉÌ°É•É…¹­•ÉÌ°•µ‰•‘‘¥¹Ì°…¹QQLµ½‘•±Ì…±½¹Í¥‘”¡…Ðµ½‘•±Ì¸(€€€€ˆˆˆ(€€€ÕÉ…Ñ•‘}ÁÉ½Ù¥‘•ÉÌ€ôl(€€€€€€€ì(€€€€€€€€€€€€‰Í±Õœˆè€‰¹½ÕÌˆ°(€€€€€€€€€€€€‰¹…µ”ˆè€‰9½ÕÌˆ°(€€€€€€€€€€€€‰µ½‘•±Ìˆèl‰µ½½¹Í¡½Ñ…¤½­¥µ¤µ¬È¸Ôˆ°€‰…¹Ñ¡É½Á¥Œ½±…Õ‘”µ½ÁÕÌ´Ð¸Ü‰t°(€€€€€€€€€€€€‰Ñ½Ñ…±}µ½‘•±Ìˆè€ÌÀ°(€€€€€€€€€€€€‰Í½ÕÉ”ˆè€‰‰Õ¥±Ðµ¥¸ˆ°(€€€€€€€€€€€€‰¥Í}ÕÉÉ•¹Ðˆè…±Í”°(€€€€€€€€€€€€‰¥Í}ÕÍ•É}‘•™¥¹•ˆè…±Í”°(€€€€€€€ô°(€€€t((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°(€€€€€€€€‰}±½…‘}™œˆ°(€€€€€€€±…µ‰‘„èì‰ÁÉ½Ù¥‘•ÉÌˆèíô°€‰ÕÍÑ½µ}ÁÉ½Ù¥‘•ÉÌˆèmuô°(€€€€¤((€€€Ý¥Ñ Á…Ñ  (€€€€€€€€‰¡•Éµ•Í}±¤¹µ½‘•±}ÍÝ¥Ñ ¹±¥ÍÑ}…ÕÑ¡•¹Ñ¥…Ñ•‘}ÁÉ½Ù¥‘•ÉÌˆ°(€€€€€€€É•ÑÕÉ¹}Ù…±Õ”õÕÉ…Ñ•‘}ÁÉ½Ù¥‘•ÉÌ°(€€€€¤è(€€€€€€€€Œ%˜ÁÉ½Ù¥‘•É}µ½‘•±}¥‘Ì•ÑÌ…±±•…Ð…±°°Ñ¡”¡…¹‘±•È¥ÌÍÑ¥±°(€€€€€€€€Œ½Ù•ÉÝÉ¥Ñ¥¹œÕÉ…Ñ•Ý¥Ñ ±¥Ù”ƒŠPÑ¡…ÐÌÑ¡”É•É•ÍÍ¥½¸Ý”É”(€€€€€€€€ŒÕ…É‘¥¹œ……¥¹ÍÐ¸(€€€€€€€Ý¥Ñ Á…Ñ  ‰¡•Éµ•Í}±¤¹µ½‘•±Ì¹ÁÉ½Ù¥‘•É}µ½‘•±}¥‘Ìˆ¤…Ì±¥Ù•}™•Ñ è(€€€€€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹}µ•Ñ¡½‘Íl‰µ½‘•°¹½ÁÑ¥½¹Ì‰t ää°ì‰Í•ÍÍ¥½¹}¥ˆè€ˆ‰ô¤((€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÀ°É•ÍÀ(€€€ÁÉ½Ù¥‘•ÉÌ€ôÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÁÉ½Ù¥‘•ÉÌ‰t(€€€¹½ÕÌ€ô¹•áÐ ¡À™½ÈÀ¥¸ÁÉ½Ù¥‘•ÉÌ¥˜À¹•Ð ‰Í±Õœˆ¤€ôô€‰¹½ÕÌˆ¤°9½¹”¤(€€€…ÍÍ•ÉÐ¹½ÕÌ¥Ì¹½Ð9½¹”(€€€…ÍÍ•ÉÐ¹½ÕÍl‰µ½‘•±Ì‰t€ôôl(€€€€€€€€‰µ½½¹Í¡½Ñ…¤½­¥µ¤µ¬È¸Ôˆ°(€€€€€€€€‰…¹Ñ¡É½Á¥Œ½±…Õ‘”µ½ÁÕÌ´Ð¸Üˆ°(€€€t(€€€…ÍÍ•ÉÐ¹½ÕÍl‰Ñ½Ñ…±}µ½‘•±Ì‰t€ôô€ÌÀ(€€€€Œ!…¹‘±•ÈµÕÍÐ¹½Ð½¹ÍÕ±ÐÑ¡”±¥Ù”…Ñ…±½œƒŠPÕÉ…Ñ•¥ÌÑ¡”ÑÉÕÑ ¸(€€€±¥Ù•}™•Ñ ¹…ÍÍ•ÉÑ}¹½Ñ}…±±• ¤(()‘•˜Ñ•ÍÑ}µ½‘•±}½ÁÑ¥½¹Í}ÁÉ½Á……Ñ•Í}±¥ÍÑ}•á•ÁÑ¥½¸¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰%˜±¥ÍÑ}…ÕÑ¡•¹Ñ¥…Ñ•‘}ÁÉ½Ù¥‘•ÉÌ¥ÑÍ•±˜É…¥Í•Ì°ÍÕÉ™…”…Ì…¸IA(€€€•ÉÉ½ÈÉ…Ñ¡•ÈÑ¡…¸ÍÝ…±±½Ý¥¹œÑ¼„‰±…¹¬Á¥­•È¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°(€€€€€€€€‰}±½…‘}™œˆ°(€€€€€€€±…µ‰‘„èì‰ÁÉ½Ù¥‘•ÉÌˆèíô°€‰ÕÍÑ½µ}ÁÉ½Ù¥‘•ÉÌˆèmuô°(€€€€¤(€€€Ý¥Ñ Á…Ñ  (€€€€€€€€‰¡•Éµ•Í}±¤¹µ½‘•±}ÍÝ¥Ñ ¹±¥ÍÑ}…ÕÑ¡•¹Ñ¥…Ñ•‘}ÁÉ½Ù¥‘•ÉÌˆ°(€€€€€€€Í¥‘•}•™™•ÐõIÕ¹Ñ¥µ•ÉÉ½È ‰…Ñ…±½œ‰±•ÜÕÀˆ¤°(€€€€¤è(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹}µ•Ñ¡½‘Íl‰µ½‘•°¹½ÁÑ¥½¹Ì‰t ÜÜ°ì‰Í•ÍÍ¥½¹}¥ˆè€ˆ‰ô¤(€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¥¸É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÔÀÌÌ(€€€…ÍÍ•ÉÐ€‰…Ñ…±½œ‰±•ÜÕÀˆ¥¸É•ÍÁl‰•ÉÉ½È‰ul‰µ•ÍÍ…”‰t(((()‘•˜Ñ•ÍÑ}µ½‘•±}½ÁÑ¥½¹Í}ÁÉ•Í•ÉÙ•Í}…¹½¹¥…±}ÕÍÑ½µ}É½Ý}…™Ñ•É}…•¹Ñ}¥¹¥Ð¡µ½¹­•åÁ…Ñ ¤è(€€€™É½´¡•Éµ•Í}±¤¹¥¹Ù•¹Ñ½Éä¥µÁ½ÉÐ½¹™¥½¹Ñ•áÐ((€€€±…ÍÌ}•¹Ðè(€€€€€€€ÁÉ½Ù¥‘•È€ô€‰ÕÍÑ½´ˆ(€€€€€€€µ½‘•°€ô€‰ÅÝ•¸Ì¸ØèÌÕˆ´ØÕ¬ˆ(€€€€€€€‰…Í•}ÕÉ°€ô€‰¡ÑÑÀè¼¼ÄÈÜ¸À¸À¸ÄèÄÄÐÌÐ½ØÄˆ((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰ÕÍÑ½´µÍ•ÍÍ¥½¸‰t€ô}Í•ÍÍ¥½¸¡…•¹Ðõ}•¹Ð ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}É•Í½±Ù•}µ½‘•°ˆ°±…µ‰‘„è€ˆˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€‰¡•Éµ•Í}±¤¹¥¹Ù•¹Ñ½Éä¹±½…‘}Á¥­•É}½¹Ñ•áÐˆ°(€€€€€€€±…µ‰‘„è½¹™¥½¹Ñ•áÐ (€€€€€€€€€€€ÕÉÉ•¹Ñ}ÁÉ½Ù¥‘•Èô‰ÕÍÑ½´é±½…°µ½±±…µ„ˆ°(€€€€€€€€€€€ÕÉÉ•¹Ñ}µ½‘•°ô‰ÅÝ•¸Ì¸ØèÌÕˆ´ØÕ¬ˆ°(€€€€€€€€€€€ÕÉÉ•¹Ñ}‰…Í•}ÕÉ°ô‰¡ÑÑÀè¼¼ÄÈÜ¸À¸À¸ÄèÄÄÐÌÐ½ØÄˆ°(€€€€€€€€€€€ÕÍ•É}ÁÉ½Ù¥‘•ÉÌõíô°(€€€€€€€€€€€ÕÍÑ½µ}ÁÉ½Ù¥‘•ÉÌõmt°(€€€€€€€€¤°(€€€€¤(€€€…¹½¹¥…°€ô5½¬¡É•ÑÕÉ¹}Ù…±Õ”ô‰ÕÍÑ½´é±½…°µ½±±…µ„ˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€‰¡•Éµ•Í}±¤¹ÉÕ¹Ñ¥µ•}ÁÉ½Ù¥‘•È¹…¹½¹¥…±}ÕÍÑ½µ}¥‘•¹Ñ¥Ñäˆ°(€€€€€€€…¹½¹¥…°°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€‰¡•Éµ•Í}±¤¹µ½‘•±}ÍÝ¥Ñ ¹±¥ÍÑ}…ÕÑ¡•¹Ñ¥…Ñ•‘}ÁÉ½Ù¥‘•ÉÌˆ°(€€€€€€€±…µ‰‘„€¨©}­Ý…ÉÌèl(€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰Í±Õœˆè€‰ÕÍÑ½´é±½…°µ½±±…µ„ˆ°(€€€€€€€€€€€€€€€€‰¹…µ”ˆè€‰1½…°=±±…µ„ˆ°(€€€€€€€€€€€€€€€€‰¥Í}ÕÉÉ•¹ÐˆèQÉÕ”°(€€€€€€€€€€€€€€€€‰¥Í}ÕÍ•É}‘•™¥¹•ˆèQÉÕ”°(€€€€€€€€€€€€€€€€‰µ½‘•±Ìˆèl‰ÅÝ•¸Ì¸ØèÌÕˆ´ØÕ¬‰t°(€€€€€€€€€€€€€€€€‰Ñ½Ñ…±}µ½‘•±Ìˆè€Ä°(€€€€€€€€€€€ô°(€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰Í±Õœˆè€‰…¹Ñ¡É½Á¥Œˆ°(€€€€€€€€€€€€€€€€‰¹…µ”ˆè€‰¹Ñ¡É½Á¥Œˆ°(€€€€€€€€€€€€€€€€‰¥Í}ÕÉÉ•¹Ðˆè…±Í”°(€€€€€€€€€€€€€€€€‰¥Í}ÕÍ•É}‘•™¥¹•ˆè…±Í”°(€€€€€€€€€€€€€€€€‰µ½‘•±Ìˆèl‰±…Õ‘”µÍ½¹¹•Ð´Ð¸Ø‰t°(€€€€€€€€€€€€€€€€‰Ñ½Ñ…±}µ½‘•±Ìˆè€Ä°(€€€€€€€€€€€ô°(€€€€€€€t°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€‰¡•Éµ•Í}±¤¹…ÕÑ ¹¥Í}ÁÉ½Ù¥‘•É}•áÁ±¥¥Ñ±å}½¹™¥ÕÉ•ˆ°(€€€€€€€±…µ‰‘„}Í±Õœè…±Í”°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰¡•Éµ•Í}±¤¹¥¹Ù•¹Ñ½Éä¹}…ÁÁ±å}ÁÉ¥¥¹œˆ°±…µ‰‘„€©}…ÉÌ°€¨©}­Ý…ÉÌè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰¡•Éµ•Í}±¤¹¥¹Ù•¹Ñ½Éä¹}…ÁÁ±å}…Á…‰¥±¥Ñ¥•Ìˆ°±…µ‰‘„€©}…ÉÌ°€¨©}­Ý…ÉÌè9½¹”¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹}µ•Ñ¡½‘Íl‰µ½‘•°¹½ÁÑ¥½¹Ì‰t (€€€€€€€€ÄÀÈ°(€€€€€€€ì‰Í•ÍÍ¥½¹}¥ˆè€‰ÕÍÑ½´µÍ•ÍÍ¥½¸ˆ°€‰•áÁ±¥¥Ñ}½¹±äˆèQÉÕ•ô°(€€€€¤((€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÀ°É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÁÉ½Ù¥‘•È‰t€ôô€‰ÕÍÑ½´é±½…°µ½±±…µ„ˆ(€€€…ÍÍ•ÉÐmÉ½Ýl‰Í±Õœ‰t™½ÈÉ½Ü¥¸É•ÍÁl‰É•ÍÕ±Ð‰ul‰ÁÉ½Ù¥‘•ÉÌ‰ut€ôôl(€€€€€€€€‰ÕÍÑ½´é±½…°µ½±±…µ„ˆ(€€€t(()‘•˜Ñ•ÍÑ}µ½‘•±}Í…Ù•}­•å}ÕÍ•Í}É•‘•¹Ñ¥…±}±¥™•å±•}…¹‘}Á¥­•É}½¹Ñ•áÐ¡µ½¹­•åÁ…Ñ ¤è(€€€•¹Ù}Ù…È€ô€‰QMQ}AI=Y%I}A%}-dˆ(€€€…•¹Ð€ô½‰©•Ð ¤(€€€Á¥­•É}Ñà€ô½‰©•Ð ¤(€€€ÁÉ½Ù¥‘•È€ôì(€€€€€€€€‰Í±Õœˆè€‰Ñ•ÍÐµÁÉ½Ù¥‘•Èˆ°(€€€€€€€€‰¹…µ”ˆè€‰Q•ÍÐAÉ½Ù¥‘•Èˆ°(€€€€€€€€‰µ½‘•±Ìˆèl‰Ñ•ÍÐµµ½‘•°‰t°(€€€€€€€€‰Ñ½Ñ…±}µ½‘•±Ìˆè€Ä°(€€€ô(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í…Ù”µ­•äµÍ•ÍÍ¥½¸‰t€ô}Í•ÍÍ¥½¸¡…•¹Ðõ…•¹Ð¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€‰¡•Éµ•Í}±¤¹…ÕÑ ¹AI=Y%I}I%MQIdˆ°(€€€€€€€ì(€€€€€€€€€€€€‰Ñ•ÍÐµÁÉ½Ù¥‘•ÈˆèÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€€€€€€€€€¹…µ”ô‰Q•ÍÐAÉ½Ù¥‘•Èˆ°(€€€€€€€€€€€€€€€…ÕÑ¡}ÑåÁ”ô‰…Á¥}­•äˆ°(€€€€€€€€€€€€€€€…Á¥}­•å}•¹Ù}Ù…ÉÌô¡•¹Ù}Ù…È°¤°(€€€€€€€€€€€€¤(€€€€€€€ô°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰¡•Éµ•Í}±¤¹½¹™¥œ¹¥Í}µ…¹…•ˆ°±…µ‰‘„è…±Í”¤(€€€Í…Ù•}É•‘•¹Ñ¥…°€ô5½¬ ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€‰¡•Éµ•Í}±¤¹É•‘•¹Ñ¥…±}±¥™•å±”¹Í…Ù•}ÁÉ½Ù¥‘•É}•¹Ù}É•‘•¹Ñ¥…°ˆ°(€€€€€€€Í…Ù•}É•‘•¹Ñ¥…°°(€€€€¤(€€€Á¥­•É}½¹Ñ•áÐ€ô5½¬¡É•ÑÕÉ¹}Ù…±Õ”õÁ¥­•É}Ñà¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}µ½‘•±}Á¥­•É}½¹Ñ•áÐˆ°Á¥­•É}½¹Ñ•áÐ¤(€€€‰Õ¥±‘}Á…å±½…€ô5½¬¡É•ÑÕÉ¹}Ù…±Õ”õì‰ÁÉ½Ù¥‘•ÉÌˆèmÁÉ½Ù¥‘•Éuô¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€‰¡•Éµ•Í}±¤¹¥¹Ù•¹Ñ½Éä¹‰Õ¥±‘}µ½‘•±Í}Á…å±½…ˆ°(€€€€€€€‰Õ¥±‘}Á…å±½…°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø¡•¹Ù}Ù…È°€‰ÁÉ•Ù¥½ÕÌµÙ…±Õ”ˆ¤(€€€™…­•}­•ä€ô€‰É•Á±…•µ•¹Ð´ˆ€¬€‰Ù…±Õ”ˆ((€€€É•ÍÀ€ôÍ•ÉÙ•È¹}µ•Ñ¡½‘Íl‰µ½‘•°¹Í…Ù•}­•ä‰t (€€€€€€€€ÄÀÌ°(€€€€€€€ì(€€€€€€€€€€€€‰Í±Õœˆè€‰Ñ•ÍÐµÁÉ½Ù¥‘•Èˆ°(€€€€€€€€€€€€‰…Á¥}­•äˆè™…­•}­•ä°(€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰Í…Ù”µ­•äµÍ•ÍÍ¥½¸ˆ°(€€€€€€€ô°(€€€€¤((€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÀ°É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÁÉ½Ù¥‘•È‰t€ôôì¨©ÁÉ½Ù¥‘•È°€‰…ÕÑ¡•¹Ñ¥…Ñ•ˆèQÉÕ•ô(€€€Í…Ù•}É•‘•¹Ñ¥…°¹…ÍÍ•ÉÑ}…±±•‘}½¹•}Ý¥Ñ ¡•¹Ù}Ù…È°™…­•}­•ä¤(()‘•˜Ñ•ÍÑ}µ½‘•±}Í…Ù•}­•å}É•½¹¥±•Í}Ñ¡•}±…Õ¹¡}ÁÉ½™¥±•Í}ÍÑ…±•}Í•ÑÕÁ}É•½É¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰Q¡”…Ñ•Á¥­•ÈÌ½Ý¸¡…ÐÝ…¥ÑÌ½¸Í•ÑÕÀ¹ÍÑ…ÑÕÍ€°Ý¡¥ …¹ÍÝ•ÉÌ™É½´Ñ¡”‰½½ÐÉ•½Éè(€€€„­•äÍ…Ù•™½ÈÑ¡”±…Õ¹ ÁÉ½™¥±”µÕÍÐ™±¥À„…±Í•€É•½É€ ¬Í•ÑÕÀ¹É•…‘å€¤…Ð½¹”ì(€€€„­•äÍ…Ù•™½È…¹½Ñ¡•ÈÁÉ½™¥±”€¡ÁÉ½™¥±•€Á…É…´¤µÕÍÐ±•…Ù”Ñ¡”±…Õ¹ É•½É…±½¹”¸ˆˆˆ(€€€™É½´¡•Éµ•Í}±¤¥µÁ½ÉÐ™É••}Ñ¥•É}‰½½ÑÍÑÉ…À…Ì™ˆ((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰¡•Éµ•Í}±¤¹…ÕÑ ¹AI=Y%I}I%MQIdˆ°ì‰Ñ•ÍÐµÁÉ½Ù¥‘•ÈˆèÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€¹…µ”ô‰Q•ÍÐAÉ½Ù¥‘•Èˆ°…ÕÑ¡}ÑåÁ”ô‰…Á¥}­•äˆ°…Á¥}­•å}•¹Ù}Ù…ÉÌô ‰QMQ}AI=Y%I}A%}-dˆ°¤¥ô¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰¡•Éµ•Í}±¤¹½¹™¥œ¹¥Í}µ…¹…•ˆ°±…µ‰‘„è…±Í”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰¡•Éµ•Í}±¤¹É•‘•¹Ñ¥…±}±¥™•å±”¹Í…Ù•}ÁÉ½Ù¥‘•É}•¹Ù}É•‘•¹Ñ¥…°ˆ°5½¬ ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰¡•Éµ•Í}±¤¹¥¹Ù•¹Ñ½Éä¹‰Õ¥±‘}µ½‘•±Í}Á…å±½…ˆ°5½¬¡É•ÑÕÉ¹}Ù…±Õ”õì‰ÁÉ½Ù¥‘•ÉÌˆèmuô¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰QMQ}AI=Y%I}A%}-dˆ°€‰ÁÉ•Ù¥½ÕÌµÙ…±Õ”ˆ¤€€ŒÍ…Ù•}­•ä•áÁ½ÉÑÌÑ¡”¹•Ü­•ä(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡™ˆ°€‰}¥¹Ù•¹Ñ½Éå}½Ñ¡•É}ÁÉ½Ù¥‘•ÉÌˆ°±…µ‰‘„èQÉÕ”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡™ˆ°€‰}É•Í½±Ù•}¥¹™•É•¹”ˆ°±…µ‰‘„è€‰Ñ•ÍÐµÁÉ½Ù¥‘•Èˆ¤(€€€‰É½…‘…ÍÑÌ€ômt(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡™ˆ°€‰}‰É½…‘…ÍÐˆ°‰É½…‘…ÍÑÌ¹…ÁÁ•¹¤(€€€½Ñ¡•É}¡½µ”€ôÑµÁ}Á…Ñ €¼€‰ÁÉ½™¥±•Ìˆ€¼€‰½Ñ¡•Èˆ€€Œµ½‘•°¹Í…Ù•}­•ä¥ÌÁÉ½™¥±”Í½Á•è€‰½Ñ¡•ÈˆµÕÍÐ•á¥ÍÐ(€€€½Ñ¡•É}¡½µ”¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÁÉ½™¥±•}¡½µ”ˆ°±…µ‰‘„¹…µ”è½Ñ¡•É}¡½µ”¥˜¹…µ”€ôô€‰½Ñ¡•Èˆ•±Í”9½¹”¤(€€€™ˆ¹É•Í•Ñ}™½É}Ñ•ÍÑÌ ¤(€€€ÍÑ…±”€ô™ˆ¹M•ÑÕÁI•½É¡ÁÉ½Ù¥‘•É}½¹™¥ÕÉ•õ…±Í”°¥¹™•É•¹•}ÁÉ½Ù¥‘•Èôˆˆ°™É••}Ñ¥•Èõ…±Í”°(€€€€€€€€€€€€€€€€€€€€€€€€€€¡…Í}¥‘•¹Ñ¥Ñäõ…±Í”°½Ñ¡•É}ÁÉ½Ù¥‘•ÉÌõ…±Í”¤(€€€Ý¥Ñ ™ˆ¹}±½¬è(€€€€€€€™ˆ¹}É•½É°™ˆ¹}ÍÑ…ÉÑ•€ôÍÑ…±”°QÉÕ”(€€€€€€€™ˆ¹}‘½¹”¹Í•Ð ¤(€€€ÑÉäè(€€€€€€€Á…É…µÌ€ôì‰Í±Õœˆè€‰Ñ•ÍÐµÁÉ½Ù¥‘•Èˆ°€‰…Á¥}­•äˆè€‰¬´ˆ€¬€ˆÄ‰ô(€€€€€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸Í•ÉÙ•È¹}µ•Ñ¡½‘Íl‰µ½‘•°¹Í…Ù•}­•ä‰t ÄÀÐ°ì¨©Á…É…µÌ°€‰ÁÉ½™¥±”ˆè€‰½Ñ¡•È‰ô¤(€€€€€€€…ÍÍ•ÉÐ™ˆ¹ÕÉÉ•¹Ñ}É•½É ¤¥ÌÍÑ…±”…¹‰É½…‘…ÍÑÌ€ôômt°€‰…¹½Ñ¡•ÈÁÉ½™¥±”Ì­•ä¥Ì¹½Ð½ÕÉÌˆ(€€€€€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸Í•ÉÙ•È¹}µ•Ñ¡½‘Íl‰µ½‘•°¹Í…Ù•}­•ä‰t ÄÀÔ°Á…É…µÌ¤(€€€€€€€É•½É€ô™ˆ¹ÕÉÉ•¹Ñ}É•½É ¤(€€€€€€€…ÍÍ•ÉÐÉ•½É¹ÁÉ½Ù¥‘•É}½¹™¥ÕÉ•¥ÌQÉÕ”…¹É•½É¹¥¹™•É•¹•}ÁÉ½Ù¥‘•È€ôô€‰Ñ•ÍÐµÁÉ½Ù¥‘•Èˆ(€€€€€€€…ÍÍ•ÉÐ‰É½…‘…ÍÑÌ€ôômÉ•½É‘t(€€€™¥¹…±±äè(€€€€€€€™ˆ¹É•Í•Ñ}™½É}Ñ•ÍÑÌ ¤(((Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(ŒÁÉ½µÁÐ¹ÍÕ‰µ¥ÐƒŠP…ÕÑ¼µÑ¥Ñ±”(Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(((()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}É•±•…Í•Í}½±‘}¡¥ÍÑ½Éå}‰•™½É•}¡•…Á}ÑÉ¥´¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰Q¡”Á½ÍÐµÑÕÉ¸¡•…ÀÑÉ¥´µÕÍÐÉÕ¸…™Ñ•ÈÑ¡”ÑÕÉ¸ÌÁÉ”µÑÕÉ¸¡¥ÍÑ½ÉäÍ¹…ÁÍ¡½ÑÌ…É”‘É½ÁÁ•è(€€€µ…±±½}ÑÉ¥´…¹¹½ÐÉ•ÑÕÉ¸Á…•ÌÍÑ¥±°É•™•É•¹•°Í¼„É•Ñ…¥¹•Í¹…ÁÍ¡½Ð½˜„±…É”Ñ½½°(€€€É•ÍÕ±ÐÁ¥¹ÌÑ¡•´™½ÈÑ¡”±¥™”½˜Ñ¡”ÁÉ½•ÍÌ¸=‰Í•ÉÙ•Ñ¡É½Õ „Ý•…¬É•™•É•¹”Ñ¼Ñ¡”½±(€€€µ•ÍÍ…”°¹½Ð‰äÉ•…‘¥¹œÑ¡”™¥¹¥Í¡•ÈÌ±½…°Ù…É¥…‰±”¹…µ•Ì¸ˆˆˆ(€€€¥µÁ½ÉÐ½¹Ñ•áÑ±¥ˆ(€€€¥µÁ½ÉÐŒ(€€€¥µÁ½ÉÐÝ•…­É•˜((€€€±…ÍÌ}5Íœ¡‘¥Ð¤è(€€€€€€€€ˆˆ‰‘¥Ð¥ÑÍ•±˜¥Ì¹½ÐÝ•…­É•˜µ…‰±”¸ˆˆˆ((€€€½‰Í•ÉÙ•€ôíô(€€€½É‘•È€ômt((€€€±…ÍÌ}•¹Ðè(€€€€€€€‘•˜ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°ÁÉ½µÁÐ°½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éäõ9½¹”°ÍÑÉ•…µ}…±±‰…¬õ9½¹”°€¨©}­Ü¤è(€€€€€€€€€€€É•ÑÕÉ¸ì‰™¥¹…±}É•ÍÁ½¹Í”ˆè€‰É•Á±äˆ°€‰µ•ÍÍ…•Ìˆèmì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä‰õuô((€€€‘•˜}ÑÉ¥´ ¨©}­Ý…ÉÌ¤è(€€€€€€€½É‘•È¹…ÁÁ•¹ ‰ÑÉ¥´ˆ¤(€€€€€€€Œ¹½±±•Ð ¤(€€€€€€€½‰Í•ÉÙ•‘l‰…±¥Ù•}…Ñ}ÑÉ¥´‰t€ô½±‘}É•˜ ¤¥Ì¹½Ð9½¹”((€€€½±€ô}5Íœ¡É½±”ô‰Ñ½½°ˆ°Ñ½½±}…±±}¥ô‰½±ˆ°½¹Ñ•¹Ðô‰àˆ€¨€ÈÁ|ÀÀÀ¤(€€€½±‘}É•˜€ôÝ•…­É•˜¹É•˜¡½±¤(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡…•¹Ðõ}•¹Ð ¤¤(€€€ÁÉ½™¥±•}¡½µ”€ôÑµÁ}Á…Ñ €¼€‰ÁÉ½™¥±•Ìˆ€¼€‰Ý½É­•Èˆ(€€€ÁÉ½™¥±•}¡½µ”¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}!=5ˆ°ÍÑÈ¡ÑµÁ}Á…Ñ ¤¤(€€€Í•ÍÍ¥½¹l‰ÁÉ½™¥±•}¡½µ”‰t€ôÍÑÈ¡ÁÉ½™¥±•}¡½µ”¤(€€€Í•ÍÍ¥½¹l‰¡¥ÍÑ½Éä‰t€ôm½±‘t(€€€‘•°½±(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‘}ÑÉ¥´‰t€ôÍ•ÍÍ¥½¸(€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}%µµ•‘¥…Ñ•Q¡É•…¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}ÕÍ…”ˆ°±…µ‰‘„}„èíô¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰É•¹‘•É}µ•ÍÍ…”ˆ°±…µ‰‘„}Ð°}Œè€ˆˆ¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰Í•Ñ}¡•Éµ•Í}¡½µ•}½Ù•ÉÉ¥‘”ˆ°±…µ‰‘„}¡½µ”è½‰©•Ð ¤¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰É•Í•Ñ}¡•Éµ•Í}¡½µ•}½Ù•ÉÉ¥‘”ˆ°±…µ‰‘„}Ñ½­•¸è½É‘•È¹…ÁÁ•¹ ‰É•Í•Ñ}¡½µ”ˆ¤¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}ÁÉ½™¥±•}ÉÕ¹Ñ¥µ•}Í½Á”ˆ°±…µ‰‘„}Í•ÍÍ¥½¸°€¨©}­Üè½¹Ñ•áÑ±¥ˆ¹¹Õ±±½¹Ñ•áÐ ¤¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰¡•Éµ•Í}±¤¹µ•µ}ÑÉ¥´¹ÑÉ¥µ}µ•µ½Éäˆ°}ÑÉ¥´¤((€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥‘}ÑÉ¥´ˆ°€‰Ñ•áÐˆè€‰¡¤‰õô¤((€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¥Ì¹½Ð9½¹”…¹É•ÍÀ¹•Ð ‰É•ÍÕ±Ðˆ¤(€€€€€€€…ÍÍ•ÉÐ½É‘•È€ôôl‰ÑÉ¥´ˆ°€‰É•Í•Ñ}¡½µ”‰t(€€€€€€€…ÍÍ•ÉÐ½‰Í•ÉÙ•‘l‰…±¥Ù•}…Ñ}ÑÉ¥´‰t¥Ì…±Í”°€‰„ÁÉ”µÑÕÉ¸¡¥ÍÑ½ÉäÍ¹…ÁÍ¡½ÐÍÕÉÙ¥Ù•Ñ¼Ñ¡”¡•…ÀÑÉ¥´ˆ(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥‘}ÑÉ¥´ˆ°9½¹”¤(()±…ÍÌ}%µµ•‘¥…Ñ•Q¡É•…è(€€€€ˆˆ‰IÕ¹ÌÑ¡”Ñ…É•Ð…±±…‰±”Íå¹¡É½¹½ÕÍ±äÍ¼…ÍÍ•ÉÑ¥½¹Ì…¸™½±±½Ü¸ˆˆˆ((€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©}Ñ¡É•…‘}½ÁÑ¥½¹Ì¤è(€€€€€€€Í•±˜¹}Ñ…É•Ð€ôÑ…É•Ð((€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€Í•±˜¹}Ñ…É•Ð ¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}Ý¥É•Í}±¥Ù•}Ñ¥Ñ±•}É•¹…µ•}…±±‰…¬¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰Q¡”…Ñ•Ý…ä¡…¹‘ÌÑ¡”…•¹Ð„¡½½¬Í¼„¹•ÜÑ¥Ñ±”É•Á…¥¹ÑÌÑ¡”Í¥‘•‰…È¸((€€€Q¥Ñ±¥¹œ¥ÑÍ•±˜µ½Ù•¥¹Ñ¼Ñ¡”Í¡…É•ÑÕÉ¸ÁÉ½±½Õ”€¡…•¹Ð½ÑÕÉ¹}½¹Ñ•áÐ¹Áä¤°(€€€Í¼Ñ¡”…Ñ•Ý…äÌ½¹±äÉ•µ…¥¹¥¹œ©½ˆ¥Ì‘•±¥Ù•É¥¹œÑ¡”É•¹…µ”•Ù•¹Ð¸ÍÍ•ÉÑ•(€€€‰ä…±±¥¹œÑ¡”¡½½¬Ñ¡”…Ñ•Ý…ä¥¹ÍÑ…±±•…¹¡•­¥¹œÝ¡…Ð¥Ð•µ¥ÑÌ¸(€€€€ˆˆˆ((€€€±…ÍÌ}•¹Ðè(€€€€€€€µ½‘•°€ô€‰ÁÐ´Ô¸ØµÍ½°ˆ(€€€€€€€ÁÉ½Ù¥‘•È€ô€‰½Á•¹…¤µ½‘•àˆ(€€€€€€€‰…Í•}ÕÉ°€ô€‰¡ÑÑÁÌè¼½¡…ÑÁÐ¹•á…µÁ±”¹Ñ•ÍÐ½‰…­•¹µ…Á¤½½‘•àˆ(€€€€€€€…Á¥}­•ä€ô½‰©•Ð ¤(€€€€€€€…Á¥}µ½‘”€ô€‰½‘•á}É•ÍÁ½¹Í•Ìˆ((€€€€€€€‘•˜ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°ÁÉ½µÁÐ°½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éäõ9½¹”°ÍÑÉ•…µ}…±±‰…¬õ9½¹”°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€€€€€‰™¥¹…±}É•ÍÁ½¹Í”ˆè€‰I½µ”Ý…Ì™½Õ¹‘•¥¸€ÜÔÌ	¸ˆ°(€€€€€€€€€€€€€€€€‰µ•ÍÍ…•Ìˆèl(€€€€€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Q•±°µ”…‰½ÕÐI½µ”‰ô°(€€€€€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰I½µ”Ý…Ì™½Õ¹‘•¥¸€ÜÔÌ	¸‰ô°(€€€€€€€€€€€€€€€t°(€€€€€€€€€€€ô((€€€…•¹Ð€ô}•¹Ð ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸¡…•¹Ðõ…•¹Ð¤(€€€•µ¥ÑÑ•€ômt(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}%µµ•‘¥…Ñ•Q¡É•…¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„­¥¹°Í¥°Á…å±½…õ9½¹”°€¨©­Üè•µ¥ÑÑ•¹…ÁÁ•¹ ¡­¥¹°Á…å±½…¤¤(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰µ…­•}ÍÑÉ•…µ}É•¹‘•É•Èˆ°±…µ‰‘„½±Ìè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰É•¹‘•É}µ•ÍÍ…”ˆ°±…µ‰‘„É…Ü°½±Ìè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è9½¹”¤((€€€Í•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰Q•±°µ”…‰½ÕÐI½µ”‰ô°(€€€€€€€ô(€€€€¤((€€€¡½½¬€ô•Ñ…ÑÑÈ¡…•¹Ð°€‰}½¹}Í•ÍÍ¥½¹}Ñ¥Ñ±”ˆ°9½¹”¤(€€€…ÍÍ•ÉÐ…±±…‰±”¡¡½½¬¤°€‰…Ñ•Ý…ä‘¥¹½Ð¥¹ÍÑ…±°„±¥Ù”Ñ¥Ñ±”µÉ•¹…µ”¡½½¬ˆ(€€€€ŒQ¥Ñ±¥¹œ¥ÌÑÝ¼µÍÑ…”°…¹„±½…°ÍÕÉ™…”Ý…¹ÑÌ‰½Ñ èÑ¡”Í¥‘•‰…ÈÉ•¹…µ•Ì(€€€€Œ½™˜Ñ¡”‘•É¥Ù•Í±¥”¥¹ÍÑ…¹Ñ±ä…¹Í¡…ÉÁ•¹ÌÝ¡•¸Ñ¡”µ½‘•°Ì±…¹‘Ì¸=¹±ä(€€€€ŒÑ¡”±…¹•ÌÑ¡…ÐÍÁ•¹„É…Ñ”µ±¥µ¥Ñ•É•µ½Ñ”É•¹…µ”™¥±Ñ•È‰äÍÑ…”¸(€€€¡½½¬ ‰Ñ•±°µ”…‰½ÕÐÉ½µ”ˆ°€‰‘•É¥Ù•ˆ¤(€€€¡½½¬ ‰½Õ¹‘¥¹œ½˜I½µ”ˆ°€‰±±´ˆ¤(€€€…ÍÍ•ÉÐmÁ…å±½…‘l‰Ñ¥Ñ±”‰t™½È­¥¹°Á…å±½…¥¸•µ¥ÑÑ•¥˜­¥¹€ôô€‰Í•ÍÍ¥½¸¹Ñ¥Ñ±”‰t€ôôl(€€€€€€€€‰Ñ•±°µ”…‰½ÕÐÉ½µ”ˆ°(€€€€€€€€‰½Õ¹‘¥¹œ½˜I½µ”ˆ°(€€€t(€€€…ÍÍ•ÉÐ€ (€€€€€€€€‰Í•ÍÍ¥½¸¹Ñ¥Ñ±”ˆ°(€€€€€€€ì‰Í•ÍÍ¥½¹}¥ˆè€‰Í•ÍÍ¥½¸µ­•äˆ°€‰Ñ¥Ñ±”ˆè€‰½Õ¹‘¥¹œ½˜I½µ”‰ô°(€€€€¤¥¸•µ¥ÑÑ•(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}ÍÕÉ™…•Í}‰…­•¹‘}•ÉÉ½É}…Í}Ù¥Í¥‰±•}Ñ•áÐ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰]¡•¸Ñ¡”‰…­•¹™…¥±ÌÝ¥Ñ ¹¼Ù¥Í¥‰±”É•ÍÁ½¹Í”€¡”¹œ¸¥¹Ù…±¥µ½‘•°Í±Õœ(€€€ƒŠHÁÉ½Ù¥‘•È€Ñáà¤°Ñ¡”QU$µÕÍÐÍÕÉ™…”É•ÍÕ±Ñl•ÉÉ½Èt…ÌÙ¥Í¥‰±”Ñ•áÐ(€€€¥¹ÍÑ•…½˜•µ¥ÑÑ¥¹œ„‰±…¹¬µ•ÍÍ…”¹½µÁ±•Ñ”ÑÕÉ¸¸ˆˆˆ((€€€±…ÍÌ}•¹Ðè(€€€€€€€‘•˜ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°ÁÉ½µÁÐ°½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éäõ9½¹”°ÍÑÉ•…µ}…±±‰…¬õ9½¹”°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€€€€€‰™¥¹…±}É•ÍÁ½¹Í”ˆè9½¹”°(€€€€€€€€€€€€€€€€‰µ•ÍÍ…•Ìˆèmt°(€€€€€€€€€€€€€€€€‰…Á¥}…±±Ìˆè€À°(€€€€€€€€€€€€€€€€‰½µÁ±•Ñ•ˆè…±Í”°(€€€€€€€€€€€€€€€€‰™…¥±•ˆèQÉÕ”°(€€€€€€€€€€€€€€€€‰•ÉÉ½Èˆè€‰!QQ@€ÐÀÀè¥¹Ù…±¥µ½‘•°¥€­¥µ¤µ¬È¸Øœˆ°(€€€€€€€€€€€ô((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸¡…•¹Ðõ}•¹Ð ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}%µµ•‘¥…Ñ•Q¡É•…¤((€€€•µ¥ÑÑ•è±¥ÍÑmÑÕÁ±•mÍÑÈ°ÍÑÈ°‘¥Ñut€ômt(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°(€€€€€€€€‰}•µ¥Ðˆ°(€€€€€€€±…µ‰‘„•Ù•¹Ð°Í¥°Á…å±½…õ9½¹”è•µ¥ÑÑ•¹…ÁÁ•¹ ¡•Ù•¹Ð°Í¥°Á…å±½…½Èíô¤¤°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰µ…­•}ÍÑÉ•…µ}É•¹‘•É•Èˆ°±…µ‰‘„½±Ìè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰É•¹‘•É}µ•ÍÍ…”ˆ°±…µ‰‘„É…Ü°½±Ìè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è9½¹”¤((€€€Í•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰¡•±±¼‰ô°(€€€€€€€ô(€€€€¤((€€€½µÁ±•Ñ•}•Ù•¹ÑÌ€ôm”™½È”¥¸•µ¥ÑÑ•¥˜•lÁt€ôô€‰µ•ÍÍ…”¹½µÁ±•Ñ”‰t(€€€…ÍÍ•ÉÐ½µÁ±•Ñ•}•Ù•¹ÑÌ°€‰•áÁ•Ñ•µ•ÍÍ…”¹½µÁ±•Ñ”Ñ¼‰”•µ¥ÑÑ•ˆ(€€€Á…å±½…€ô½µÁ±•Ñ•}•Ù•¹ÑÍl´ÅulÉt(€€€…ÍÍ•ÉÐÁ…å±½…¹•Ð ‰ÍÑ…ÑÕÌˆ¤€ôô€‰•ÉÉ½Èˆ(€€€Ñ•áÐ€ôÁ…å±½…¹•Ð ‰Ñ•áÐˆ°€ˆˆ¤(€€€…ÍÍ•ÉÐ€‰¥¹Ù…±¥µ½‘•°¥€­¥µ¤µ¬È¸Øœˆ¥¸Ñ•áÐ(€€€…ÍÍ•ÉÐÁ…å±½…¹•Ð ‰•ÉÉ½Èˆ¤€ôô€‰!QQ@€ÐÀÀè¥¹Ù…±¥µ½‘•°¥€­¥µ¤µ¬È¸Øœˆ(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}ÁÉ•Í•ÉÙ•Í}•µÁÑå}É•ÍÁ½¹Í•}Ý¥Ñ¡½ÕÑ}•ÉÉ½È¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰¸•µÁÑä™¥¹…±}É•ÍÁ½¹Í”Ý¥Ñ 9<‰…­•¹•ÉÉ½ÈµÕÍÐÍÑ…ä•µÁÑäƒŠP‘¼¹½Ð(€€€Íå¹Ñ¡•Í¥é”…¸•ÉÉ½ÈÍÑÉ¥¹œ¸AÉ•Í•ÉÙ•ÌÑ¡”•á¥ÍÑ¥¹œ9½¹”½•µÁÑäµÍ•¹Ñ¥¹•°(€€€Í•µ…¹Ñ¥Ì½Ý¹•‰ä‘½Ý¹ÍÑÉ•…´¡…¹‘±•ÉÌ¸ˆˆˆ((€€€±…ÍÌ}•¹Ðè(€€€€€€€‘•˜ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°ÁÉ½µÁÐ°½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éäõ9½¹”°ÍÑÉ•…µ}…±±‰…¬õ9½¹”°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€€€€€‰™¥¹…±}É•ÍÁ½¹Í”ˆè9½¹”°(€€€€€€€€€€€€€€€€‰µ•ÍÍ…•Ìˆèmt°(€€€€€€€€€€€€€€€€‰…Á¥}…±±Ìˆè€Ä°(€€€€€€€€€€€€€€€€‰½µÁ±•Ñ•ˆèQÉÕ”°(€€€€€€€€€€€ô((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸¡…•¹Ðõ}•¹Ð ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}%µµ•‘¥…Ñ•Q¡É•…¤((€€€•µ¥ÑÑ•è±¥ÍÑmÑÕÁ±•mÍÑÈ°ÍÑÈ°‘¥Ñut€ômt(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°(€€€€€€€€‰}•µ¥Ðˆ°(€€€€€€€±…µ‰‘„•Ù•¹Ð°Í¥°Á…å±½…õ9½¹”è•µ¥ÑÑ•¹…ÁÁ•¹ ¡•Ù•¹Ð°Í¥°Á…å±½…½Èíô¤¤°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰µ…­•}ÍÑÉ•…µ}É•¹‘•É•Èˆ°±…µ‰‘„½±Ìè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰É•¹‘•É}µ•ÍÍ…”ˆ°±…µ‰‘„É…Ü°½±Ìè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è9½¹”¤((€€€Í•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰¡•±±¼‰ô°(€€€€€€€ô(€€€€¤((€€€½µÁ±•Ñ•}•Ù•¹ÑÌ€ôm”™½È”¥¸•µ¥ÑÑ•¥˜•lÁt€ôô€‰µ•ÍÍ…”¹½µÁ±•Ñ”‰t(€€€…ÍÍ•ÉÐ½µÁ±•Ñ•}•Ù•¹ÑÌ°€‰•áÁ•Ñ•µ•ÍÍ…”¹½µÁ±•Ñ”Ñ¼‰”•µ¥ÑÑ•ˆ(€€€Á…å±½…€ô½µÁ±•Ñ•}•Ù•¹ÑÍl´ÅulÉt(€€€€ŒMÑ…ÑÕÌÍÑ…åÌ€‰½µÁ±•Ñ”ˆ‰•…ÕÍ”¹¼•ÉÉ½È™±…œÝ…ÌÍ•Ð(€€€…ÍÍ•ÉÐÁ…å±½…¹•Ð ‰ÍÑ…ÑÕÌˆ¤€ôô€‰½µÁ±•Ñ”ˆ(€€€€ŒQ•áÐÍÑ…åÌ•µÁÑäƒŠPÝ”‘¥9=P™…‰É¥…Ñ”…¸€‰ÉÉ½ÈèˆÍÑÉ¥¹œ(€€€Ñ•áÐ€ôÁ…å±½…¹•Ð ‰Ñ•áÐˆ°€ˆˆ¤(€€€…ÍÍ•ÉÐÑ•áÐ¥¸ìˆˆ°9½¹•ô°˜‰•áÁ•Ñ••µÁÑäÑ•áÐ°½ÐíÑ•áÐ…Éôˆ(((ŒƒŠRŠR …Ñ¥Ù”±¥Ù”QU$Í•ÍÍ¥½¹ÌƒŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠR (()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}…Ñ¥Ù•}±¥ÍÑ}É•Á½ÉÑÍ}±¥Ù•}Í•ÍÍ¥½¹Ì¡µ½¹­•åÁ…Ñ ¤è(€€€±…ÍÌ}è(€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¹}Ñ¥Ñ±”¡Í•±˜°­•ä¤è(€€€€€€€€€€€É•ÑÕÉ¸ì‰­•äµ„ˆè€‰I•Í•…É ˆ°€‰­•äµˆˆè€‰%µÁ±•µ•¹Ð‰ô¹•Ð¡­•ä°€ˆˆ¤((€€€ÁÉ•Ù¥½ÕÍ}Í•ÍÍ¥½¹Ì€ô‘¥Ð¡Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è} ¤¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥µ„‰t€ô}Í•ÍÍ¥½¸ (€€€€€€€…•¹ÐõÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡µ½‘•°ô‰µ½‘•°µ„ˆ¤°(€€€€€€€¡¥ÍÑ½Éäõmì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰™¥¹‘½Ì‰õt°(€€€€€€€Í•ÍÍ¥½¹}­•äô‰­•äµ„ˆ°(€€€€€€€É•…Ñ•‘}…ÐôÄÀ¸À°(€€€€€€€±…ÍÑ}…Ñ¥Ù”ôÈÀ¸À°(€€€€¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥µˆ‰t€ô}Í•ÍÍ¥½¸ (€€€€€€€…•¹ÐõÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡µ½‘•°ô‰µ½‘•°µˆˆ¤°(€€€€€€€¡¥ÍÑ½Éäõmì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰ÝÉ¥Ñ¥¹œ½‘”‰õt°(€€€€€€€ÉÕ¹¹¥¹œõQÉÕ”°(€€€€€€€Í•ÍÍ¥½¹}­•äô‰­•äµˆˆ°(€€€€€€€É•…Ñ•‘}…ÐôÄÄ¸À°(€€€€€€€±…ÍÑ}…Ñ¥Ù”ôÌÀ¸À°(€€€€¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹…Ñ¥Ù•}±¥ÍÐˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰ÕÉÉ•¹Ñ}Í•ÍÍ¥½¹}¥ˆè€‰Í¥µˆ‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹ÕÁ‘…Ñ”¡ÁÉ•Ù¥½ÕÍ}Í•ÍÍ¥½¹Ì¤((€€€Í•ÍÍ¥½¹}É½ÝÌ€ôÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹Ì‰t(€€€…ÍÍ•ÉÐmÉ½Ýl‰¥‰t™½ÈÉ½Ü¥¸Í•ÍÍ¥½¹}É½ÝÍt€ôôl‰Í¥µ„ˆ°€‰Í¥µˆ‰t((€€€É½ÝÌ€ôíÉ½Ýl‰¥‰tèÉ½Ü™½ÈÉ½Ü¥¸Í•ÍÍ¥½¹}É½ÝÍô(€€€…ÍÍ•ÉÐÉ½ÝÍl‰Í¥µ„‰t€ôôì(€€€€€€€€‰ÕÉÉ•¹Ðˆè…±Í”°(€€€€€€€€‰¥ˆè€‰Í¥µ„ˆ°(€€€€€€€€‰±…ÍÑ}…Ñ¥Ù”ˆè€ÈÀ¸À°(€€€€€€€€‰µ•ÍÍ…•}½Õ¹Ðˆè€Ä°(€€€€€€€€‰µ½‘•°ˆè€‰µ½‘•°µ„ˆ°(€€€€€€€€‰ÁÉ•Ù¥•Üˆè€‰™¥¹‘½Ìˆ°(€€€€€€€€‰Í•ÍÍ¥½¹}­•äˆè€‰­•äµ„ˆ°(€€€€€€€€‰ÍÑ…ÉÑ•‘}…Ðˆè€ÄÀ¸À°(€€€€€€€€‰ÍÑ…ÑÕÌˆè€‰¥‘±”ˆ°(€€€€€€€€‰Ñ¥Ñ±”ˆè€‰I•Í•…É ˆ°(€€€ô(€€€…ÍÍ•ÉÐÉ½ÝÍl‰Í¥µˆ‰ul‰ÕÉÉ•¹Ð‰t¥ÌQÉÕ”(€€€…ÍÍ•ÉÐÉ½ÝÍl‰Í¥µˆ‰ul‰ÍÑ…ÑÕÌ‰t€ôô€‰Ý½É­¥¹œˆ(€€€…ÍÍ•ÉÐÉ½ÝÍl‰Í¥µˆ‰ul‰Ñ¥Ñ±”‰t€ôô€‰%µÁ±•µ•¹Ðˆ(€€€…ÍÍ•ÉÐÉ½ÝÍl‰Í¥µˆ‰ul‰ÁÉ•Ù¥•Ü‰t€ôô€‰ÝÉ¥Ñ¥¹œ½‘”ˆ(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}…Ñ¥Ù•}±¥ÍÑ}•á±Õ‘•Í}™¥¹…±¥é•‘}Í•ÍÍ¥½¹Ì¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆˆŒÌàäÔÀè„™¥¹…±¥é•µ‰ÕÐµ¹½Ðµå•ÐµÁ½ÁÁ•Í•ÍÍ¥½¸µÕÍÐ¹½Ð¥¹™±…Ñ”Ñ¡”½Õ¹Ð¸((€€€Q¡”]LÉ…”µÉ•…À…¹¥‘±”É•…Á•ÈÍ•Ð}™¥¹…±¥é•‘€¥¹Í¥‘”(€€€}Ñ•…É‘½Ý¹}Í•ÍÍ¥½¹€‰•™½É”Á½ÁÁ¥¹œÑ¡”•¹ÑÉä™É½´}Í•ÍÍ¥½¹Í€¸ÕÉ¥¹œ(€€€Ñ¡…ÐÝ¥¹‘½ÜÍ•ÍÍ¥½¸¹…Ñ¥Ù•}±¥ÍÑ€Ý½Õ±½Ñ¡•ÉÝ¥Í”ÍÑ¥±°É•Á½ÉÐÑ¡”‘•…(€€€Í•ÍÍ¥½¸°Ý¡¥ ¥Ì•á…Ñ±äÑ¡”™½½Ñ•È€‰8Í•ÍÍ¥½¹Ìˆ½Õ¹ÐÑ¡…Ð½¹±ä•Ù•ÈÉ•Ü(€€€Õ¹Ñ¥°„…Ñ•Ý…äÉ•ÍÑ…ÉÐ¸±¥Ù”Í•ÍÍ¥½¸½¸Ñ¡”É•…°ÍÑ‘¥¼ÑÉ…¹ÍÁ½ÉÐ€¡Ñ¡”(€€€ÍÑ…¹‘…±½¹”¡•Éµ•Ì€´µÑÕ¥€…Í”¤µÕÍÐÍÑ¥±°‰”É•Á½ÉÑ•¸(€€€€ˆˆˆ(€€€±…ÍÌ}è(€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¹}Ñ¥Ñ±”¡Í•±˜°­•ä¤è(€€€€€€€€€€€É•ÑÕÉ¸ì‰­•äµ±¥Ù”ˆè€‰1¥Ù”ˆ°€‰­•äµ‘•…ˆè€‰•…‰ô¹•Ð¡­•ä°€ˆˆ¤((€€€ÁÉ•Ù¥½ÕÍ}Í•ÍÍ¥½¹Ì€ô‘¥Ð¡Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è} ¤¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥µ±¥Ù”‰t€ô}Í•ÍÍ¥½¸ (€€€€€€€…•¹ÐõÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡µ½‘•°ô‰µ½‘•°µ±¥Ù”ˆ¤°(€€€€€€€¡¥ÍÑ½Éäõmì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰ÍÑ¥±°¡•É”‰õt°(€€€€€€€Í•ÍÍ¥½¹}­•äô‰­•äµ±¥Ù”ˆ°(€€€€€€€É•…Ñ•‘}…ÐôÄÀ¸À°(€€€€€€€±…ÍÑ}…Ñ¥Ù”ôÈÀ¸À°(€€€€¤(€€€‘•…€ô}Í•ÍÍ¥½¸ (€€€€€€€…•¹ÐõÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡µ½‘•°ô‰µ½‘•°µ‘•…ˆ¤°(€€€€€€€¡¥ÍÑ½Éäõmì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰½¹”‰õt°(€€€€€€€Í•ÍÍ¥½¹}­•äô‰­•äµ‘•…ˆ°(€€€€€€€É•…Ñ•‘}…ÐôÄÄ¸À°(€€€€€€€±…ÍÑ}…Ñ¥Ù”ôÈÄ¸À°(€€€€¤(€€€‘•…‘l‰}™¥¹…±¥é•‰t€ôQÉÕ”(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥µ‘•…‰t€ô‘•…(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹…Ñ¥Ù•}±¥ÍÐˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèíô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹ÕÁ‘…Ñ”¡ÁÉ•Ù¥½ÕÍ}Í•ÍÍ¥½¹Ì¤((€€€Í•ÍÍ¥½¹}É½ÝÌ€ôÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹Ì‰t(€€€…ÍÍ•ÉÐmÉ½Ýl‰¥‰t™½ÈÉ½Ü¥¸Í•ÍÍ¥½¹}É½ÝÍt€ôôl‰Í¥µ±¥Ù”‰t((()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}…Ñ¥Ù…Ñ•}É•ÑÕÉ¹Í}¥¹™±¥¡Ñ}ÍÑÉ•…µ}‰•™½É•}½µÁ±•Ñ¥½¸¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰MÝ¥Ñ¡¥¹œ¥¹Ñ¼„ÍÑ¥±°µÉÕ¹¹¥¹œ±¥Ù”Í•ÍÍ¥½¸µÕÍÐ¡å‘É…Ñ”Á…ÉÑ¥…°½ÕÑÁÕÐ¸((€€€Q¡”½µµ¥ÑÑ•Í•ÍÍ¥½¸¡¥ÍÑ½Éä¥Ì½¹±äÕÁ‘…Ñ•…™Ñ•ÈÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸É•ÑÕÉ¹Ì°(€€€Í¼Í•ÍÍ¥½¸¹…Ñ¥Ù…Ñ”¹••‘Ì…¸•áÁ±¥¥Ð¥¸µ™±¥¡ÐÁ…å±½…Í½ÕÉ•™É½´Ñ¡”(€€€‰…­•¹ÍÑÉ•…´…±±‰…¬¸(€€€€ˆˆˆ(€€€ÍÑ…ÉÑ•€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤(€€€É•±•…Í”€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤(€€€‘½¹”€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤((€€€±…ÍÌ}•¹Ðè(€€€€€€€µ½‘•°€ô€‰µ½‘•°µ±¥Ù”ˆ((€€€€€€€‘•˜ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°ÁÉ½µÁÐ°½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éäõ9½¹”°ÍÑÉ•…µ}…±±‰…¬õ9½¹”°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€…ÍÍ•ÉÐÁÉ½µÁÐ€ôô€‰ÝÉ¥Ñ”„±½¹œ…¹ÍÝ•Èˆ(€€€€€€€€€€€…ÍÍ•ÉÐ½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éä€ôômt(€€€€€€€€€€€ÍÑÉ•…µ}…±±‰…¬ ‰Á…ÉÑ¥…°€ˆ¤(€€€€€€€€€€€ÍÑÉ•…µ}…±±‰…¬ ‰…¹ÍÝ•Èˆ¤(€€€€€€€€€€€ÍÑ…ÉÑ•¹Í•Ð ¤(€€€€€€€€€€€…ÍÍ•ÉÐÉ•±•…Í”¹Ý…¥Ð È¤°€‰Ñ•ÍÐÑ¥µ•½ÕÐÝ…¥Ñ¥¹œÑ¼™¥¹¥Í ™…­”µ½‘•°ÑÕÉ¸ˆ(€€€€€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€€€€€‰™¥¹…±}É•ÍÁ½¹Í”ˆè€‰Á…ÉÑ¥…°…¹ÍÝ•È½µÁ±•Ñ”ˆ°(€€€€€€€€€€€€€€€€‰µ•ÍÍ…•Ìˆèl(€€€€€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰ÝÉ¥Ñ”„±½¹œ…¹ÍÝ•È‰ô°(€€€€€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰Á…ÉÑ¥…°…¹ÍÝ•È½µÁ±•Ñ”‰ô°(€€€€€€€€€€€€€€€t°(€€€€€€€€€€€ô((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥µ±¥Ù”‰t€ô}Í•ÍÍ¥½¸¡…•¹Ðõ}•¹Ð ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰µ…­•}ÍÑÉ•…µ}É•¹‘•É•Èˆ°±…µ‰‘„½±Ìè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰É•¹‘•É}µ•ÍÍ…”ˆ°±…µ‰‘„É…Ü°½±Ìè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„…•¹Ðèì‰µ½‘•°ˆè…•¹Ð¹µ½‘•±ô¤((€€€‘•˜}•µ¥Ð¡•Ù•¹Ð°Í¥°Á…å±½…õ9½¹”¤è(€€€€€€€¥˜•Ù•¹Ð€ôô€‰µ•ÍÍ…”¹½µÁ±•Ñ”ˆè(€€€€€€€€€€€‘½¹”¹Í•Ð ¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°}•µ¥Ð¤((€€€ÑÉäè(€€€€€€€ÍÕ‰µ¥Ð€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€‰ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥µ±¥Ù”ˆ°€‰Ñ•áÐˆè€‰ÝÉ¥Ñ”„±½¹œ…¹ÍÝ•È‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÍÕ‰µ¥Ñl‰É•ÍÕ±Ð‰ul‰ÍÑ…ÑÕÌ‰t€ôô€‰ÍÑÉ•…µ¥¹œˆ(€€€€€€€…ÍÍ•ÉÐÍÑ…ÉÑ•¹Ý…¥Ð È¤°€‰™…­”µ½‘•°‘¥¹½ÐÍÑÉ•…´‰•™½É”…Ñ¥Ù…Ñ¥½¸ˆ((€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€‰…Ñ¥Ù…Ñ”ˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹…Ñ¥Ù…Ñ”ˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥µ±¥Ù”‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤((€€€€€€€¥¹™±¥¡Ð€ôÉ•ÍÁl‰É•ÍÕ±Ð‰t¹•Ð ‰¥¹™±¥¡Ðˆ¤(€€€€€€€…ÍÍ•ÉÐ¥¹™±¥¡Ð€ôôì(€€€€€€€€€€€€‰…ÍÍ¥ÍÑ…¹Ðˆè€‰Á…ÉÑ¥…°…¹ÍÝ•Èˆ°(€€€€€€€€€€€€‰ÍÑÉ•…µ¥¹œˆèQÉÕ”°(€€€€€€€€€€€€‰ÕÍ•Èˆè€‰ÝÉ¥Ñ”„±½¹œ…¹ÍÝ•Èˆ°(€€€€€€€ô(€€€€€€€ÑÕÉ¹}ÍÑ…ÉÑ•‘}…Ð€ôÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÑÕÉ¹}ÍÑ…ÉÑ•‘}…Ð‰t(€€€€€€€…ÍÍ•ÉÐÑÕÉ¹}ÍÑ…ÉÑ•‘}…Ð€ôôÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥µ±¥Ù”‰ul‰¥¹™±¥¡Ñ}ÑÕÉ¸‰ul‰ÍÑ…ÉÑ•‘}…Ð‰t(€€€€€€€…ÍÍ•ÉÐÑÕÉ¹}ÍÑ…ÉÑ•‘}…Ð€ø€À(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰µ•ÍÍ…•Ì‰t€ôômt((€€€€€€€É•±•…Í”¹Í•Ð ¤(€€€€€€€…ÍÍ•ÉÐ‘½¹”¹Ý…¥Ð È¤°€‰™…­”µ½‘•°ÑÕÉ¸‘¥¹½Ð½µÁ±•Ñ”ˆ(€€€€€€€½µÁ±•Ñ•€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€‰…Ñ¥Ù…Ñ”µ‘½¹”ˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹…Ñ¥Ù…Ñ”ˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥µ±¥Ù”‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐ½µÁ±•Ñ•‘l‰É•ÍÕ±Ð‰t¹•Ð ‰¥¹™±¥¡Ðˆ¤¥Ì9½¹”(€€€€€€€…ÍÍ•ÉÐ½µÁ±•Ñ•‘l‰É•ÍÕ±Ð‰ul‰ÑÕÉ¹}ÍÑ…ÉÑ•‘}…Ð‰t¥Ì9½¹”(€€€€€€€…ÍÍ•ÉÐ½µÁ±•Ñ•‘l‰É•ÍÕ±Ð‰ul‰µ•ÍÍ…•Ì‰t€ôôl(€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰Ñ•áÐˆè€‰ÝÉ¥Ñ”„±½¹œ…¹ÍÝ•È‰ô°(€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰Ñ•áÐˆè€‰Á…ÉÑ¥…°…¹ÍÝ•È½µÁ±•Ñ”‰ô°(€€€€€€€t(€€€™¥¹…±±äè(€€€€€€€É•±•…Í”¹Í•Ð ¤(€€€€€€€‘½¹”¹Ý…¥Ð È¤(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥µ±¥Ù”ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}…Ñ¥Ù…Ñ•}É•ÑÕÉ¹Í}ÁÉ½µÁÑ}ÅÕ•Õ•‘}‘ÕÉ¥¹}‰ÕÍå}ÑÕÉ¸¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰™Õ±°±¥•¹ÐÉ•ÍÑ…ÉÐµÕÍÐÉ•½Ù•È…¸…•ÁÑ•¹•áÐµÑÕÉ¸ÁÉ½µÁÐ¸((€€€	ÕÍäÁÉ½µÁÑÌ…É”¥¹Ñ•¹Ñ¥½¹…±±ä¹½Ð‘ÕÉ…‰±”Õ¹Ñ¥°Ñ¡•ä‘É…¥¸¸Q¡•¥È½¹±ä(€€€…ÕÑ¡½É¥Ñ…Ñ¥Ù”½Áä¥ÌÅÕ•Õ•‘}ÁÉ½µÁÑ€°Í¼Ñ¡”±¥Ù”ÁÉ½©•Ñ¥½¸µÕÍÐ•áÁ½Í”(€€€Ñ¡…Ð½ÁäÝ¥Ñ¡½ÕÐ±•…­¥¹œÑ¡”ÑÉ…¹ÍÁ½ÉÐ½‰©•Ð¸(€€€€ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}‰ÕÍå}¥¹ÁÕÑ}µ½‘”ˆ°±…µ‰‘„è€‰ÅÕ•Õ”ˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„…•¹Ðèì‰µ½‘•°ˆè…•¹Ð¹µ½‘•±ô¤(€€€…•¹Ð€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡µ½‘•°ô‰µ½‘•°µ±¥Ù”ˆ¤(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸ (€€€€€€€…•¹Ðõ…•¹Ð°(€€€€€€€ÉÕ¹¹¥¹œõQÉÕ”°(€€€€€€€¥¹™±¥¡Ñ}ÑÕÉ¸õì(€€€€€€€€€€€€‰…ÍÍ¥ÍÑ…¹Ðˆè€‰Á…ÉÑ¥…°…¹ÍÝ•Èˆ°(€€€€€€€€€€€€‰ÍÑÉ•…µ¥¹œˆèQÉÕ”°(€€€€€€€€€€€€‰ÕÍ•Èˆè€‰ÕÉÉ•¹ÐÁÉ½µÁÐˆ°(€€€€€€€ô°(€€€€¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥µ±¥Ù”‰t€ôÍ•ÍÍ¥½¸(€€€ÑÉäè(€€€€€€€ÅÕ•Õ•€ôÍ•ÉÙ•È¹}¡…¹‘±•}‰ÕÍå}ÍÕ‰µ¥Ð (€€€€€€€€€€€€‰ÍÕ‰µ¥Ðˆ°€‰Í¥µ±¥Ù”ˆ°Í•ÍÍ¥½¸°€‰¹•Ý•ÍÐÁÉ½µÁÐˆ°½‰©•Ð ¤(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÅÕ•Õ•‘l‰É•ÍÕ±Ð‰ul‰ÍÑ…ÑÕÌ‰t€ôô€‰ÅÕ•Õ•ˆ((€€€€€€€…Ñ¥Ù…Ñ•€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€‰…Ñ¥Ù…Ñ”ˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹…Ñ¥Ù…Ñ”ˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥µ±¥Ù”‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤((€€€€€€€…ÍÍ•ÉÐ…Ñ¥Ù…Ñ•‘l‰É•ÍÕ±Ð‰ul‰ÅÕ•Õ•‰t€ôôì‰ÕÍ•Èˆè€‰¹•Ý•ÍÐÁÉ½µÁÐ‰ô(€€€€€€€…ÍÍ•ÉÐ€‰ÑÉ…¹ÍÁ½ÉÐˆ¹½Ð¥¸…Ñ¥Ù…Ñ•‘l‰É•ÍÕ±Ð‰ul‰ÅÕ•Õ•‰t(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥µ±¥Ù”ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}…Ñ¥Ù…Ñ•}ÍÝ¥Ñ¡•Í}±¥Ù•}Í•ÍÍ¥½¹}Ý¥Ñ¡½ÕÑ}±½Í¥¹}Í¥‰±¥¹Ì¡µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„…•¹Ðèì‰µ½‘•°ˆè…•¹Ð¹µ½‘•±ô¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥µ„‰t€ô}Í•ÍÍ¥½¸ (€€€€€€€…•¹ÐõÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡µ½‘•°ô‰µ½‘•°µ„ˆ¤°(€€€€€€€¡¥ÍÑ½Éäõmì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰½±‰õt°(€€€€€€€Í•ÍÍ¥½¹}­•äô‰­•äµ„ˆ°(€€€€¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥µˆ‰t€ô}Í•ÍÍ¥½¸ (€€€€€€€…•¹ÐõÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡µ½‘•°ô‰µ½‘•°µˆˆ¤°(€€€€€€€¡¥ÍÑ½Éäõl(€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰¹•ÜÁÉ½µÁÐ‰ô°(€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰¹•Ü…¹ÍÝ•È‰ô°(€€€€€€€t°(€€€€€€€ÉÕ¹¹¥¹œõQÉÕ”°(€€€€€€€Í•ÍÍ¥½¹}­•äô‰­•äµˆˆ°(€€€€¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹…Ñ¥Ù…Ñ”ˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥µˆ‰õô(€€€€€€€€¤((€€€€€€€…ÍÍ•ÉÐ€‰Í¥µ„ˆ¥¸Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì(€€€€€€€…ÍÍ•ÉÐ€‰Í¥µˆˆ¥¸Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰t€ôô€‰Í¥µˆˆ(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}­•ä‰t€ôô€‰­•äµˆˆ(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÉÕ¹¹¥¹œ‰t¥ÌQÉÕ”(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÍÑ…ÑÕÌ‰t€ôô€‰Ý½É­¥¹œˆ(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰¥¹™¼‰t€ôôì‰µ½‘•°ˆè€‰µ½‘•°µˆ‰ô(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰µ•ÍÍ…•Ì‰t€ôôl(€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰Ñ•áÐˆè€‰¹•ÜÁÉ½µÁÐ‰ô°(€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰Ñ•áÐˆè€‰¹•Ü…¹ÍÝ•È‰ô°(€€€€€€€t(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥µ„ˆ°9½¹”¤(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥µˆˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}…Ñ¥Ù…Ñ•}…¹}½µ¥Ñ}‘ÕÁ±¥…Ñ•}‘•Í­Ñ½Á}ÑÉ…¹ÍÉ¥ÁÐ¡µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„…•¹Ðèì‰µ½‘•°ˆè…•¹Ð¹µ½‘•±ô¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥µ±…É”‰t€ô}Í•ÍÍ¥½¸ (€€€€€€€…•¹ÐõÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡µ½‘•°ô‰µ½‘•°µ±…É”ˆ¤°(€€€€€€€¡¥ÍÑ½Éäõl(€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰±…É”ÁÉ½µÁÐ‰ô°(€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰±…É”…¹ÍÝ•È‰ô°(€€€€€€€t°(€€€€€€€Í•ÍÍ¥½¹}­•äô‰­•äµ±…É”ˆ°(€€€€¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹…Ñ¥Ù…Ñ”ˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥µ±…É”ˆ°€‰½µ¥Ñ}µ•ÍÍ…•ÌˆèQÉÕ•ô°(€€€€€€€€€€€ô(€€€€€€€€¤((€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰µ•ÍÍ…•Ì‰t€ôômt(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰µ•ÍÍ…•}½Õ¹Ð‰t€ôô€È(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰µ•ÍÍ…•Í}½µ¥ÑÑ•‰t¥ÌQÉÕ”(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}­•ä‰t€ôô€‰­•äµ±…É”ˆ(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥µ±…É”ˆ°9½¹”¤(((ŒƒŠRŠR Í•ÍÍ¥½¸¹µ½ÍÑ}É••¹ÐƒŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠR (()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}µ½ÍÑ}É••¹Ñ}É•ÑÕÉ¹Í}™¥ÉÍÑ}¹½¹}‘•¹¥•¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰É½ÁÌÑ½½±€É½ÝÌ±¥­”Í•ÍÍ¥½¸¹±¥ÍÐ‘½•Ì°É•ÑÕÉ¹ÌÑ¡”™¥ÉÍÐ¡¥Ð¸ˆˆˆ((€€€±…ÍÌ}è(€€€€€€€‘•˜±¥ÍÑ}Í•ÍÍ¥½¹Í}É¥ ¡Í•±˜°€¨°Í½ÕÉ”õ9½¹”°±¥µ¥ÐôÈÀÀ°½É‘•É}‰å}±…ÍÑ}…Ñ¥Ù”õ…±Í”°½µÁ…Ñ}É½ÝÌõ…±Í”¤è(€€€€€€€€€€€É•ÑÕÉ¸l(€€€€€€€€€€€€€€€ì‰¥ˆè€‰Ñ½½°´Äˆ°€‰Í½ÕÉ”ˆè€‰Ñ½½°ˆ°€‰Ñ¥Ñ±”ˆè€‰¹½¥Í”ˆ°€‰ÍÑ…ÉÑ•‘}…Ðˆè€ÄÀÁô°(€€€€€€€€€€€€€€€ì‰¥ˆè€‰ÑÕ¤´Äˆ°€‰Í½ÕÉ”ˆè€‰ÑÕ¤ˆ°€‰Ñ¥Ñ±”ˆè€‰É•…°ˆ°€‰ÍÑ…ÉÑ•‘}…Ðˆè€äåô°(€€€€€€€€€€€t((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è} ¤¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹µ½ÍÑ}É••¹Ðˆ°€‰Á…É…µÌˆèíõô(€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰t€ôô€‰ÑÕ¤´Äˆ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Ñ¥Ñ±”‰t€ôô€‰É•…°ˆ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í½ÕÉ”‰t€ôô€‰ÑÕ¤ˆ(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}µ½ÍÑ}É••¹Ñ}É•ÑÕÉ¹Í}¹Õ±±}Ý¡•¹}½¹±å}Ñ½½±}É½ÝÌ¡µ½¹­•åÁ…Ñ ¤è(€€€±…ÍÌ}è(€€€€€€€‘•˜±¥ÍÑ}Í•ÍÍ¥½¹Í}É¥ ¡Í•±˜°€¨°Í½ÕÉ”õ9½¹”°±¥µ¥ÐôÈÀÀ°½É‘•É}‰å}±…ÍÑ}…Ñ¥Ù”õ…±Í”°½µÁ…Ñ}É½ÝÌõ…±Í”¤è(€€€€€€€€€€€É•ÑÕÉ¸mì‰¥ˆè€‰Ñ½½°´Äˆ°€‰Í½ÕÉ”ˆè€‰Ñ½½°ˆ°€‰ÍÑ…ÉÑ•‘}…Ðˆè€Åõt((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è} ¤¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹µ½ÍÑ}É••¹Ðˆ°€‰Á…É…µÌˆèíõô(€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰t¥Ì9½¹”(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}µ½ÍÑ}É••¹Ñ}Í­¥ÁÍ}Õ¹­¹½Ý¹}Í½ÕÉ•}É½ÝÌ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆˆŒÔÐÌÈÀè„Ñ½­•¸µ…½Õ¹Ñ¥¹œÕ…ÉÁ±…•¡½±‘•È€¡Í½ÕÉ”ôÕ¹­¹½Ý¸œ¤µÕÍÐ¹•Ù•È‰”(€€€Á¥­•™½È…ÕÑ¼µÉ•ÍÕµ”ƒŠPÑ¡”É½Ü…¸½ÕÑÉ…¹¬Ñ¡”Í•ÍÍ¥½¸Ñ¡”ÕÍ•È…ÑÕ…±±ä½Á•¹•¸ˆˆˆ((€€€±…ÍÌ}è(€€€€€€€‘•˜±¥ÍÑ}Í•ÍÍ¥½¹Í}É¥ ¡Í•±˜°€¨°Í½ÕÉ”õ9½¹”°±¥µ¥ÐôÈÀÀ°½É‘•É}‰å}±…ÍÑ}…Ñ¥Ù”õ…±Í”°½µÁ…Ñ}É½ÝÌõ…±Í”¤è(€€€€€€€€€€€É•ÑÕÉ¸l(€€€€€€€€€€€€€€€ì‰¥ˆè€‰Õ…É´Äˆ°€‰Í½ÕÉ”ˆè€‰Õ¹­¹½Ý¸ˆ°€‰Ñ¥Ñ±”ˆè€ˆˆ°€‰ÍÑ…ÉÑ•‘}…Ðˆè€ÄÀÅô°(€€€€€€€€€€€€€€€ì‰¥ˆè€‰ÑÕ¤´Äˆ°€‰Í½ÕÉ”ˆè€‰ÑÕ¤ˆ°€‰Ñ¥Ñ±”ˆè€‰É•…°ˆ°€‰ÍÑ…ÉÑ•‘}…Ðˆè€ÄÀÁô°(€€€€€€€€€€€t((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è} ¤¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹µ½ÍÑ}É••¹Ðˆ°€‰Á…É…µÌˆèíõô(€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰t€ôô€‰ÑÕ¤´Äˆ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í½ÕÉ”‰t€ôô€‰ÑÕ¤ˆ(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}µ½ÍÑ}É••¹Ñ}É•ÑÕÉ¹Í}¹Õ±±}Ý¡•¹}½¹±å}Õ¹­¹½Ý¹}É½ÝÌ¡µ½¹­•åÁ…Ñ ¤è(€€€±…ÍÌ}è(€€€€€€€‘•˜±¥ÍÑ}Í•ÍÍ¥½¹Í}É¥ ¡Í•±˜°€¨°Í½ÕÉ”õ9½¹”°±¥µ¥ÐôÈÀÀ°½É‘•É}‰å}±…ÍÑ}…Ñ¥Ù”õ…±Í”°½µÁ…Ñ}É½ÝÌõ…±Í”¤è(€€€€€€€€€€€É•ÑÕÉ¸mì‰¥ˆè€‰Õ…É´Äˆ°€‰Í½ÕÉ”ˆè€‰Õ¹­¹½Ý¸ˆ°€‰ÍÑ…ÉÑ•‘}…Ðˆè€Åõt((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è} ¤¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹µ½ÍÑ}É••¹Ðˆ°€‰Á…É…µÌˆèíõô(€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰t¥Ì9½¹”(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}µ½ÍÑ}É••¹Ñ}™½±‘Í}‘‰}•á•ÁÑ¥½¹}¥¹Ñ½}¹Õ±±}É•ÍÕ±Ð¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰A•È½¹ÑÉ…Ð°•ÉÉ½ÉÌ…É”™½±‘•¥¹Ñ¼Ñ¡”¹Õ±°µÉ•ÍÕ±ÐÍ¡…Á”Í¼(€€€…±±•ÉÌ‘½¸Ð¡…Ù”Ñ¼ÍÁ•¥…°µ…Í”)M=8µIA•ÉÉ½È•¹Ù•±½Á•Ì™½È(€€€€¹¼…¹ÍÝ•Èœ€¡½Á¥±½ÐÉ•Ù¥•Ü½¸€ŒÄÜÄÌÀ¤¸ˆˆˆ((€€€±…ÍÌ}	É½­•¹è(€€€€€€€‘•˜±¥ÍÑ}Í•ÍÍ¥½¹Í}É¥ ¡Í•±˜°€¨°Í½ÕÉ”õ9½¹”°±¥µ¥ÐôÈÀÀ°½É‘•É}‰å}±…ÍÑ}…Ñ¥Ù”õ…±Í”°½µÁ…Ñ}É½ÝÌõ…±Í”¤è(€€€€€€€€€€€É…¥Í”IÕ¹Ñ¥µ•ÉÉ½È ‰‘ˆ±½­•ˆ¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è}	É½­•¹ ¤¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹µ½ÍÑ}É••¹Ðˆ°€‰Á…É…µÌˆèíõô(€€€€¤((€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¹½Ð¥¸É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰t¥Ì9½¹”(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}µ½ÍÑ}É••¹Ñ}¡…¹‘±•Í}‘‰}Õ¹…Ù…¥±…‰±”¡µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è9½¹”¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹µ½ÍÑ}É••¹Ðˆ°€‰Á…É…µÌˆèíõô(€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰t¥Ì9½¹”(((ŒƒŠRŠR Ù•É¥™¥…Ñ¥½¸¹ÍÑ…ÑÕÌƒŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠR (()‘•˜Ñ•ÍÑ}Ù•É¥™¥…Ñ¥½¹}ÍÑ…ÑÕÍ}É•ÑÕÉ¹Í}É•½É‘•‘}•Ù¥‘•¹”¡ÑµÁ}Á…Ñ °µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}YI%e}=9}MQ=@ˆ°€ˆÄˆ¤€€Œ±•‘•È¥Ì¥¹•ÉÐÝ¡•¸Ñ¡”Õ…É¥Ì½™˜(€€€ÁÉ½™¥±•}¡½µ”€ôÑµÁ}Á…Ñ €¼€‰ÁÉ½™¥±•Ìˆ€¼€‰Ù•É¥™äˆ(€€€ÁÉ½™¥±•}¡½µ”¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÁÉ½™¥±•}¡½µ”ˆ°±…µ‰‘„ÀèÁÉ½™¥±•}¡½µ”¥˜À€ôô€‰Ù•É¥™äˆ•±Í”9½¹”¤(€€€Ñ½­•¸€ôÍ•Ñ}¡•Éµ•Í}¡½µ•}½Ù•ÉÉ¥‘”¡ÁÉ½™¥±•}¡½µ”¤(€€€ÁÉ½©•Ð€ôÑµÁ}Á…Ñ €¼€‰ÁÉ½©•Ðˆ(€€€ÁÉ½©•Ð¹µ­‘¥È ¤(€€€€¡ÁÉ½©•Ð€¼€ˆ¹¥Ðˆ¤¹µ­‘¥È ¤(€€€€¡ÁÉ½©•Ð€¼€‰Á…­…”¹©Í½¸ˆ¤¹ÝÉ¥Ñ•}Ñ•áÐ (€€€€€€€©Í½¸¹‘ÕµÁÌ¡ì‰ÍÉ¥ÁÑÌˆèì‰Ñ•ÍÐˆè€‰Ù¥Ñ•ÍÐ‰õô¤°(€€€€€€€•¹½‘¥¹œô‰ÕÑ˜´àˆ°(€€€€¤(€€€€¡ÁÉ½©•Ð€¼€‰Á¹Á´µ±½¬¹å…µ°ˆ¤¹ÝÉ¥Ñ•}Ñ•áÐ ˆˆ°•¹½‘¥¹œô‰ÕÑ˜´àˆ¤(€€€ÑÉäè(€€€€€€€™É½´…•¹Ð¹Ù•É¥™¥…Ñ¥½¹}•Ù¥‘•¹”¥µÁ½ÉÐÉ•½É‘}Ñ•Éµ¥¹…±}É•ÍÕ±Ð((€€€€€€€É•½É‘}Ñ•Éµ¥¹…±}É•ÍÕ±Ð (€€€€€€€€€€€½µµ…¹ô‰Á¹Á´ÉÕ¸Ñ•ÍÐˆ°(€€€€€€€€€€€ÝõÁÉ½©•Ð°(€€€€€€€€€€€Í•ÍÍ¥½¹}¥ô‰Í¥ˆ°(€€€€€€€€€€€•á¥Ñ}½‘”ôÀ°(€€€€€€€€€€€½ÕÑÁÕÐô‰É••¸ˆ°(€€€€€€€€¤((€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Ù•É¥™¥…Ñ¥½¸¹ÍÑ…ÑÕÌˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰ÝˆèÍÑÈ¡ÁÉ½©•Ð¤°€‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰ÁÉ½™¥±”ˆè€‰Ù•É¥™ä‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€É•Í•Ñ}¡•Éµ•Í}¡½µ•}½Ù•ÉÉ¥‘”¡Ñ½­•¸¤((€€€Ù•É¥™¥…Ñ¥½¸€ôÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Ù•É¥™¥…Ñ¥½¸‰t(€€€…ÍÍ•ÉÐÙ•É¥™¥…Ñ¥½¹l‰ÍÑ…ÑÕÌ‰t€ôô€‰Á…ÍÍ•ˆ(€€€…ÍÍ•ÉÐÙ•É¥™¥…Ñ¥½¹l‰•Ù¥‘•¹”‰ul‰…¹½¹¥…±}½µµ…¹‰t€ôô€‰Á¹Á´ÉÕ¸Ñ•ÍÐˆ(€€€…ÍÍ•ÉÐÙ•É¥™¥…Ñ¥½¹l‰•Ù¥‘•¹”‰ul‰Í½Á”‰t€ôô€‰™Õ±°ˆ(()‘•˜Ñ•ÍÑ}Ù•É¥™¥…Ñ¥½¹}ÍÑ…ÑÕÍ}½ÕÑÍ¥‘•}Ý½É­ÍÁ…•}¥Í}¹½Ñ}…ÁÁ±¥…‰±”¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}YI%e}=9}MQ=@ˆ°€ˆÄˆ¤€€Œ±•‘•È¥Ì¥¹•ÉÐÝ¡•¸Ñ¡”Õ…É¥Ì½™˜(€€€€ŒÝÝ¥Ñ ¹¼ÁÉ½©•Ð™…ÑÌ€¡½ÕÑÍ¥‘”…¹ä½‘”Ý½É­ÍÁ…”¤µÕÍÐÉ•Á½ÉÐ(€€€€Œ¹½Ñ}…ÁÁ±¥…‰±”¸½É”Ñ¡”€‰¹¼™…ÑÌˆÁÉ•½¹‘¥Ñ¥½¸É…Ñ¡•ÈÑ¡…¸É•±å¥¹œ½¸(€€€€ŒÑµÁ}Á…Ñ Ì…¹•ÍÑ½ÉÌ‰•¥¹œÁÉ¥ÍÑ¥¹”ƒŠP„ÍÑÉ…äµ…É­•È™¥±”¥¸„Í¡…É•(€€€€ŒÑµÀµÉ½½Ð…¹•ÍÑ½È€¡”¹œ¸€½ÑµÀ½Á…­…”¹©Í½¸±•™Ð‰ä…¹½Ñ¡•ÈÑ½½°¤Ý½Õ±(€€€€Œ½Ñ¡•ÉÝ¥Í”µ…­”}µ…É­•É}É½½Ð ¤É•Í½±Ù”ÑµÁ}Á…Ñ …Ì„Ý½É­ÍÁ…”…¹™±¥À(€€€€ŒÑ¡”ÍÑ…ÑÕÌÑ¼€‰Õ¹Ù•É¥™¥•ˆ¸(€€€¥µÁ½ÉÐ…•¹Ð¹½‘¥¹}½¹Ñ•áÐ…Ì½‘¥¹}½¹Ñ•áÐ((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡½‘¥¹}½¹Ñ•áÐ°€‰ÁÉ½©•Ñ}™…ÑÍ}™½Èˆ°±…µ‰‘„}Ýõ9½¹”è9½¹”¤((€€€¡½µ”€ôÑµÁ}Á…Ñ €¼€ˆ¹¡•Éµ•Ìˆ(€€€¡½µ”¹µ­‘¥È ¤(€€€Ñ½­•¸€ôÍ•Ñ}¡•Éµ•Í}¡½µ•}½Ù•ÉÉ¥‘”¡¡½µ”¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Ù•É¥™¥…Ñ¥½¸¹ÍÑ…ÑÕÌˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰ÝˆèÍÑÈ¡ÑµÁ}Á…Ñ ¤°€‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€É•Í•Ñ}¡•Éµ•Í}¡½µ•}½Ù•ÉÉ¥‘”¡Ñ½­•¸¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Ù•É¥™¥…Ñ¥½¸‰ul‰ÍÑ…ÑÕÌ‰t€ôô€‰¹½Ñ}…ÁÁ±¥…‰±”ˆ(((ŒƒŠRŠR ‰É½ÝÍ•È¹µ…¹…”ƒŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠR (()‘•˜}ÍÑÕ‰}ÕÉ±½Á•¸¡µ½¹­•åÁ…Ñ °€¨°½¬è‰½½°¤è(€€€€ˆˆ‰A…Ñ Ñ¡”±½½Á‰…¬µ…Ý…É”½Á•¹•È‰É½ÝÍ•È¹µ…¹…”ÁÉ½‰•ÌÑ¡É½Õ € ŒÄÄÀÔØÔ¤Ñ¼Í¡½ÉÐµ¥ÉÕ¥ÐÁÉ½‰•Ì¸ˆˆˆ((€€€±…ÍÌ}I•ÍÀè(€€€€€€€ÍÑ…ÑÕÌ€ô€ÈÀÀ¥˜½¬•±Í”€ÔÀÌ((€€€€€€€‘•˜}}•¹Ñ•É}|¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸Í•±˜((€€€€€€€‘•˜}}•á¥Ñ}|¡Í•±˜°€©|¤è(€€€€€€€€€€€É•ÑÕÉ¸…±Í”((€€€‘•˜}½Á•¹•È¡}ÕÉ°°Ñ¥µ•½ÕÐôÈ¸À¤è€€Œ¹½Å„èIÀÀÄƒŠPµ…Ñ ÕÉ±±¥ˆÍ¥¹…ÑÕÉ”(€€€€€€€¥˜¹½Ð½¬è(€€€€€€€€€€€É…¥Í”=MÉÉ½È ‰ÁÉ½‰”™…¥±•ˆ¤(€€€€€€€É•ÑÕÉ¸}I•ÍÀ ¤((€€€¥µÁ½ÉÐÕÉ±±¥ˆ¹É•ÅÕ•ÍÐ((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡ÕÉ±±¥ˆ¹É•ÅÕ•ÍÐ¹=Á•¹•É¥É•Ñ½È°€‰½Á•¸ˆ°±…µ‰‘„}Í•±˜°ÕÉ°°€©„°Ñ¥µ•½ÕÐôÈ¸À°€¨©¬è}½Á•¹•È¡ÕÉ°°Ñ¥µ•½ÕÐõÑ¥µ•½ÕÐ¤¤(()‘•˜}ÍÑÕ‰}ÕÉ±½Á•¹}…ÁÑÕÉ”¡µ½¹­•åÁ…Ñ °€¨°½¬è‰½½°¤è(€€€ÕÉ±Ìè±¥ÍÑmÍÑÉt€ômt((€€€±…ÍÌ}I•ÍÀè(€€€€€€€ÍÑ…ÑÕÌ€ô€ÈÀÀ((€€€€€€€‘•˜}}•¹Ñ•É}|¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸Í•±˜((€€€€€€€‘•˜}}•á¥Ñ}|¡Í•±˜°€©|¤è(€€€€€€€€€€€É•ÑÕÉ¸…±Í”((€€€‘•˜}½Á•¹•È¡ÕÉ°°Ñ¥µ•½ÕÐôÈ¸À¤è€€Œ¹½Å„èIÀÀÄƒŠPµ…Ñ ÕÉ±±¥ˆÍ¥¹…ÑÕÉ”(€€€€€€€ÕÉ±Ì¹…ÁÁ•¹¡ÕÉ°¤(€€€€€€€¥˜¹½Ð½¬è(€€€€€€€€€€€É…¥Í”=MÉÉ½È ‰ÁÉ½‰”™…¥±•ˆ¤(€€€€€€€É•ÑÕÉ¸}I•ÍÀ ¤((€€€¥µÁ½ÉÐÕÉ±±¥ˆ¹É•ÅÕ•ÍÐ((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡ÕÉ±±¥ˆ¹É•ÅÕ•ÍÐ¹=Á•¹•É¥É•Ñ½È°€‰½Á•¸ˆ°±…µ‰‘„}Í•±˜°ÕÉ°°€©„°Ñ¥µ•½ÕÐôÈ¸À°€¨©¬è}½Á•¹•È¡ÕÉ°°Ñ¥µ•½ÕÐõÑ¥µ•½ÕÐ¤¤(€€€É•ÑÕÉ¸ÕÉ±Ì(()‘•˜Ñ•ÍÑ}‰É½ÝÍ•É}µ…¹…•}ÍÑ…ÑÕÍ}É•…‘Í}•¹Ù}Ù…È¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰MÑ…ÑÕÌÉ•ÑÕÉ¹ÌÑ¡”•¹ØÙ…ÈÙ•É‰…Ñ¥´€¡¹¼¹•ÑÝ½É¬$½<¤¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰	I=]MI}A}UI0ˆ°€‰¡ÑÑÀè¼¼ÄÈÜ¸À¸À¸ÄèäÈÈÈˆ¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰‰É½ÝÍ•È¹µ…¹…”ˆ°€‰Á…É…µÌˆèì‰…Ñ¥½¸ˆè€‰ÍÑ…ÑÕÌ‰õô(€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰½¹¹•Ñ•‰t¥ÌQÉÕ”(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÕÉ°‰t€ôô€‰¡ÑÑÀè¼¼ÄÈÜ¸À¸À¸ÄèäÈÈÈˆ(()‘•˜Ñ•ÍÑ}‰É½ÝÍ•É}µ…¹…•}ÍÑ…ÑÕÍ}™…±±Í}‰…­}Ñ½}½¹™¥}‘Á}ÕÉ°¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰]¡•¸•¹Ø¥ÌÕ¹Í•Ð°ÍÑ…ÑÕÌÍÕÉ™…•Ì‰É½ÝÍ•È¹‘Á}ÕÉ±€™É½´(€€€½¹™¥œ¹å…µ°Í¼ÕÍ•ÉÌÍ•”Ý¡…ÐÑ¡”¹•áÐÑ½½°…±°Ý¥±°É•…¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹‘•±•¹Ø ‰	I=]MI}A}UI0ˆ°É…¥Í¥¹œõ…±Í”¤((€€€™…­•}™œ€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€É•…‘}É…Ý}½¹™¥œõ±…µ‰‘„èì‰‰É½ÝÍ•Èˆèì‰‘Á}ÕÉ°ˆè€‰¡ÑÑÀè¼½±…¸èäÈÈÈ‰õô(€€€€¤(€€€Ý¥Ñ Á…Ñ ¹‘¥Ð¡ÍåÌ¹µ½‘Õ±•Ì°ì‰¡•Éµ•Í}±¤¹½¹™¥œˆè™…­•}™ô¤è(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰‰É½ÝÍ•È¹µ…¹…”ˆ°€‰Á…É…µÌˆèì‰…Ñ¥½¸ˆè€‰ÍÑ…ÑÕÌ‰õô(€€€€€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰t€ôôì‰½¹¹•Ñ•ˆèQÉÕ”°€‰ÕÉ°ˆè€‰¡ÑÑÀè¼½±…¸èäÈÈÈ‰ô(((()‘•˜Ñ•ÍÑ}‰É½ÝÍ•É}µ…¹…•}½¹¹•Ñ}Í•ÑÍ}•¹Ù}…¹‘}±•…¹Í}ÑÝ¥”¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰€½‰É½ÝÍ•È½¹¹•Ñ€µÕÍÐÉ•… Ñ¡”±¥Ù”ÁÉ½•ÍÌèÍ•Ð•¹Ø°É•…À‰É½ÝÍ•È(€€€Í•ÍÍ¥½¹Ì‰•™½É”9…™Ñ•ÈÁÕ‰±¥Í¡¥¹œÑ¡”¹•ÜUI0¸€Q¡”‘½Õ‰±”µ±•…¹ÕÀ(€€€±½Í•ÌÑ¡”ÍÕÁ•ÉÙ¥Í½ÈÍÝ…ÀÝ¥¹‘½ÜÝ¡•É”}•¹ÍÕÉ•}‘Á}ÍÕÁ•ÉÙ¥Í½É€(€€€½Õ±É”µ…ÑÑ… Ñ¼Ñ¡”€©½±¨@•¹‘Á½¥¹Ð‰•ÑÝ••¸ÍÑ•ÁÌ¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹‘•±•¹Ø ‰	I=]MI}A}UI0ˆ°É…¥Í¥¹œõ…±Í”¤(€€€±•…¹ÕÁ}…±±Ìè±¥ÍÑmÍÑÉt€ômt((€€€‘•˜}±•…¹ÕÁ}…±° ¤è(€€€€€€€±•…¹ÕÁ}…±±Ì¹…ÁÁ•¹¡½Ì¹•¹Ù¥É½¸¹•Ð ‰	I=]MI}A}UI0ˆ°€ˆˆ¤¤((€€€™…­”€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€±•…¹ÕÁ}…±±}‰É½ÝÍ•ÉÌõ}±•…¹ÕÁ}…±°°(€€€€€€€}•Ñ}‘Á}½Ù•ÉÉ¥‘”õ±…µ‰‘„è½Ì¹•¹Ù¥É½¸¹•Ð ‰	I=]MI}A}UI0ˆ°€ˆˆ¤°(€€€€¤(€€€Ý¥Ñ Á…Ñ ¹‘¥Ð¡ÍåÌ¹µ½‘Õ±•Ì°ì‰Ñ½½±Ì¹‰É½ÝÍ•É}Ñ½½±}±¥™•å±”ˆè™…­•ô¤è(€€€€€€€}ÍÑÕ‰}ÕÉ±½Á•¸¡µ½¹­•åÁ…Ñ °½¬õQÉÕ”¤(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰‰É½ÝÍ•È¹µ…¹…”ˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰…Ñ¥½¸ˆè€‰½¹¹•Ðˆ°€‰ÕÉ°ˆè€‰¡ÑÑÀè¼¼ÄÈÜ¸À¸À¸ÄèäÈÈÈ‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰½¹¹•Ñ•‰t¥ÌQÉÕ”(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÕÉ°‰t€ôô€‰¡ÑÑÀè¼¼ÄÈÜ¸À¸À¸ÄèäÈÈÈˆ(€€€…ÍÍ•ÉÐ½Ì¹•¹Ù¥É½¸¹•Ð ‰	I=]MI}A}UI0ˆ¤€ôô€‰¡ÑÑÀè¼¼ÄÈÜ¸À¸À¸ÄèäÈÈÈˆ(€€€€Œ¥ÉÍÐ±•…¹ÕÀÉÕ¹Ì……¥¹ÍÐÑ¡”=1•¹Ø€¡¹½¹”¡•É”¤°Í•½¹……¥¹ÍÐÑ¡”9\¸(€€€…ÍÍ•ÉÐ±•…¹ÕÁ}…±±Ì€ôôlˆˆ°€‰¡ÑÑÀè¼¼ÄÈÜ¸À¸À¸ÄèäÈÈÈ‰t(()‘•˜Ñ•ÍÑ}‰É½ÝÍ•É}µ…¹…•}½¹¹•Ñ}‘•™…Õ±ÑÍ}Ñ½}±½½Á‰…¬¡µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹‘•±•¹Ø ‰	I=]MI}A}UI0ˆ°É…¥Í¥¹œõ…±Í”¤(€€€™…­”€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€±•…¹ÕÁ}…±±}‰É½ÝÍ•ÉÌõ±…µ‰‘„è9½¹”°(€€€€€€€}•Ñ}‘Á}½Ù•ÉÉ¥‘”õ±…µ‰‘„è½Ì¹•¹Ù¥É½¸¹•Ð ‰	I=]MI}A}UI0ˆ°€ˆˆ¤°(€€€€¤(€€€Ý¥Ñ Á…Ñ ¹‘¥Ð¡ÍåÌ¹µ½‘Õ±•Ì°ì‰Ñ½½±Ì¹‰É½ÝÍ•É}Ñ½½±}±¥™•å±”ˆè™…­•ô¤è(€€€€€€€ÕÉ±Ì€ô}ÍÑÕ‰}ÕÉ±½Á•¹}…ÁÑÕÉ”¡µ½¹­•åÁ…Ñ °½¬õQÉÕ”¤(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰‰É½ÝÍ•È¹µ…¹…”ˆ°€‰Á…É…µÌˆèì‰…Ñ¥½¸ˆè€‰½¹¹•Ð‰õô(€€€€€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰½¹¹•Ñ•‰t¥ÌQÉÕ”(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÕÉ°‰t€ôô€‰¡ÑÑÀè¼¼ÄÈÜ¸À¸À¸ÄèäÈÈÈˆ(€€€…ÍÍ•ÉÐÕÉ±ÍlÁt€ôô€‰¡ÑÑÀè¼¼ÄÈÜ¸À¸À¸ÄèäÈÈÈ½©Í½¸½Ù•ÉÍ¥½¸ˆ(()‘•˜Ñ•ÍÑ}‰É½ÝÍ•É}µ…¹…•}½¹¹•Ñ}‘•™…Õ±Ñ}±½…±}É•Á½ÉÑÍ}±…Õ¹¡}¡¥¹Ð¡µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹‘•±•¹Ø ‰	I=]MI}A}UI0ˆ°É…¥Í¥¹œõ…±Í”¤(€€€€Œ9¼Á±…Ñ™½É´¹ÍåÍÑ•µ€™…­”èÑ¡”É•Í½±Ù•ÍåÍÑ•´ÍÑÉ¥¹œ½¹±ä™±½ÝÌ¥¹Ñ¼(€€€€Œ±…Õ¹¡}¡É½µ•}‘•‰Õ€€¼µ…¹Õ…±}¡É½µ•}‘•‰Õ}½µµ…¹‘€€¼(€€€€Œ•Ñ}¡É½µ•}‘•‰Õ}…¹‘¥‘…Ñ•Í€°…±°½˜Ý¡¥ …É”µ½­•‰•±½ÜƒŠPÑ¡”(€€€€Œ¡½ÍÐÌÉ•…°Ù…±Õ”¹•Ù•ÈÉ•…¡•ÌÕ¹µ½­•½‘”¸(€€€•µ¥ÑÑ•è±¥ÍÑmÑÕÁ±•mÍÑÈ°‘¥Ñut€ômt(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°(€€€€€€€€‰}•µ¥Ðˆ°(€€€€€€€±…µ‰‘„•ÙÐ°Í¥°Á…å±½…õ9½¹”è•µ¥ÑÑ•¹…ÁÁ•¹ ¡•ÙÐ°Á…å±½…½Èíô¤¤°(€€€€¤(€€€™…­”€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€±•…¹ÕÁ}…±±}‰É½ÝÍ•ÉÌõ±…µ‰‘„è9½¹”°(€€€€€€€}•Ñ}‘Á}½Ù•ÉÉ¥‘”õ±…µ‰‘„è½Ì¹•¹Ù¥É½¸¹•Ð ‰	I=]MI}A}UI0ˆ°€ˆˆ¤°(€€€€¤(€€€Ý¥Ñ Á…Ñ ¹‘¥Ð¡ÍåÌ¹µ½‘Õ±•Ì°ì‰Ñ½½±Ì¹‰É½ÝÍ•É}Ñ½½±}±¥™•å±”ˆè™…­•ô¤è(€€€€€€€}ÍÑÕ‰}ÕÉ±½Á•¸¡µ½¹­•åÁ…Ñ °½¬õ…±Í”¤(€€€€€€€Ý¥Ñ € (€€€€€€€€€€€Á…Ñ  (€€€€€€€€€€€€€€€€‰¡•Éµ•Í}±¤¹‰É½ÝÍ•É}½¹¹•Ð¹±…Õ¹¡}¡É½µ•}‘•‰Õœˆ°(€€€€€€€€€€€€€€€É•ÑÕÉ¹}Ù…±Õ”õ¡É½µ••‰Õ1…Õ¹  ¤°(€€€€€€€€€€€€¤°(€€€€€€€€€€€Á…Ñ  ‰¡•Éµ•Í}±¤¹‰É½ÝÍ•É}½¹¹•Ð¹±½…±}Á½ÉÑ}¥¹}ÕÍ”ˆ°É•ÑÕÉ¹}Ù…±Õ”õ…±Í”¤°(€€€€€€€€€€€Á…Ñ  ‰¡•Éµ•Í}±¤¹‰É½ÝÍ•É}½¹¹•Ð¹µ…¹Õ…±}¡É½µ•}‘•‰Õ}½µµ…¹ˆ°É•ÑÕÉ¹}Ù…±Õ”õ9½¹”¤°(€€€€€€€€€€€Á…Ñ  (€€€€€€€€€€€€€€€€‰¡•Éµ•Í}±¤¹‰É½ÝÍ•É}½¹¹•Ð¹•Ñ}¡É½µ•}‘•‰Õ}…¹‘¥‘…Ñ•Ìˆ°(€€€€€€€€€€€€€€€É•ÑÕÉ¹}Ù…±Õ”õmt°(€€€€€€€€€€€€¤°(€€€€€€€€¤è(€€€€€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰‰É½ÝÍ•È¹µ…¹…”ˆ°(€€€€€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€€€€€‰…Ñ¥½¸ˆè€‰½¹¹•Ðˆ°(€€€€€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰Í•ÍÌ´Äˆ°(€€€€€€€€€€€€€€€€€€€€€€€€‰ÕÉ°ˆè€‰¡ÑÑÀè¼½±½…±¡½ÍÐèäÈÈÈˆ°(€€€€€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€€€€€ô(€€€€€€€€€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰½¹¹•Ñ•‰t¥Ì…±Í”(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÕÉ°‰t€ôô€‰¡ÑÑÀè¼¼ÄÈÜ¸À¸À¸ÄèäÈÈÈˆ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰µ•ÍÍ…•Ì‰t(€€€…ÍÍ•ÉÐ€‰	I=]MI}A}UI0ˆ¹½Ð¥¸½Ì¹•¹Ù¥É½¸(€€€ÁÉ½É•ÍÌ€ômÁl‰µ•ÍÍ…”‰t™½È•ÙÐ°À¥¸•µ¥ÑÑ•¥˜•ÙÐ€ôô€‰‰É½ÝÍ•È¹ÁÉ½É•ÍÌ‰t(€€€…ÍÍ•ÉÐÁÉ½É•ÍÌ€ôôÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰µ•ÍÍ…•Ì‰t(()‘•˜Ñ•ÍÑ}‰É½ÝÍ•É}µ…¹…•}½¹¹•Ñ}¹½}Í•ÍÍ¥½¹}Í­¥ÁÍ}ÁÉ½É•ÍÍ}•Ù•¹ÑÌ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰]¥Ñ¡½ÕÐ„Í•ÍÍ¥½¹}¥Ñ¡”QU$ÁÉ¥¹ÑÌµ•ÍÍ…•Ì™É½´Ñ¡”É•ÍÁ½¹Í”ì(€€€•µ¥ÑÑ¥¹œ‰É½ÝÍ•È¹ÁÉ½É•ÍÍ€•Ù•¹ÑÌÝ½Õ±‘½Õ‰±”µÉ•¹‘•È¸…Ñ”Ñ¡”(€€€•µ¥ÐÍ¼…±±•ÉÌÝ¥Ñ¡½ÕÐ„Í•ÍÍ¥½¸Í•”Ñ¡”‰Õ¹‘±•±¥ÍÐ½¹±ä¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹‘•±•¹Ø ‰	I=]MI}A}UI0ˆ°É…¥Í¥¹œõ…±Í”¤(€€€•µ¥ÑÑ•è±¥ÍÑmÑÕÁ±•mÍÑÈ°‘¥Ñut€ômt(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°(€€€€€€€€‰}•µ¥Ðˆ°(€€€€€€€±…µ‰‘„•ÙÐ°Í¥°Á…å±½…õ9½¹”è•µ¥ÑÑ•¹…ÁÁ•¹ ¡•ÙÐ°Á…å±½…½Èíô¤¤°(€€€€¤(€€€™…­”€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€±•…¹ÕÁ}…±±}‰É½ÝÍ•ÉÌõ±…µ‰‘„è9½¹”°(€€€€€€€}•Ñ}‘Á}½Ù•ÉÉ¥‘”õ±…µ‰‘„è½Ì¹•¹Ù¥É½¸¹•Ð ‰	I=]MI}A}UI0ˆ°€ˆˆ¤°(€€€€¤(€€€Ý¥Ñ Á…Ñ ¹‘¥Ð¡ÍåÌ¹µ½‘Õ±•Ì°ì‰Ñ½½±Ì¹‰É½ÝÍ•É}Ñ½½±}±¥™•å±”ˆè™…­•ô¤è(€€€€€€€}ÍÑÕ‰}ÕÉ±½Á•¸¡µ½¹­•åÁ…Ñ °½¬õ…±Í”¤(€€€€€€€Ý¥Ñ € (€€€€€€€€€€€Á…Ñ  (€€€€€€€€€€€€€€€€‰¡•Éµ•Í}±¤¹‰É½ÝÍ•É}½¹¹•Ð¹±…Õ¹¡}¡É½µ•}‘•‰Õœˆ°(€€€€€€€€€€€€€€€É•ÑÕÉ¹}Ù…±Õ”õ¡É½µ••‰Õ1…Õ¹  ¤°(€€€€€€€€€€€€¤°(€€€€€€€€€€€Á…Ñ  ‰¡•Éµ•Í}±¤¹‰É½ÝÍ•É}½¹¹•Ð¹µ…¹Õ…±}¡É½µ•}‘•‰Õ}½µµ…¹ˆ°É•ÑÕÉ¹}Ù…±Õ”õ9½¹”¤°(€€€€€€€€€€€Á…Ñ  (€€€€€€€€€€€€€€€€‰¡•Éµ•Í}±¤¹‰É½ÝÍ•É}½¹¹•Ð¹•Ñ}¡É½µ•}‘•‰Õ}…¹‘¥‘…Ñ•Ìˆ°(€€€€€€€€€€€€€€€É•ÑÕÉ¹}Ù…±Õ”õmt°(€€€€€€€€€€€€¤°(€€€€€€€€¤è(€€€€€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰‰É½ÝÍ•È¹µ…¹…”ˆ°(€€€€€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰…Ñ¥½¸ˆè€‰½¹¹•Ðˆ°€‰ÕÉ°ˆè€‰¡ÑÑÀè¼½±½…±¡½ÍÐèäÈÈÈ‰ô°(€€€€€€€€€€€€€€€ô(€€€€€€€€€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰½¹¹•Ñ•‰t¥Ì…±Í”(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰µ•ÍÍ…•Ì‰t€€Œ‰Õ¹‘±•±¥ÍÐÍÑ¥±°Á½ÁÕ±…Ñ•(€€€…ÍÍ•ÉÐm•ÙÐ™½È•ÙÐ°|¥¸•µ¥ÑÑ•¥˜•ÙÐ€ôô€‰‰É½ÝÍ•È¹ÁÉ½É•ÍÌ‰t€ôômt(()‘•˜Ñ•ÍÑ}‰É½ÝÍ•É}µ…¹…•}½¹¹•Ñ}¡…¹‘±•Í}¹Õ±±}ÕÉ°¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰áÁ±¥¥Ðì‰ÕÉ°ˆè¹Õ±±õ€€¡½È•µÁÑäÍÑÉ¥¹œ¤µÕÍÐ™…±°‰…¬Ñ¼Ñ¡”(€€€‘•™…Õ±Ð±½½Á‰…¬UI0¥¹ÍÑ•…½˜É…¥Í¥¹œ„QåÁ•ÉÉ½ÈÑ¡…Ð•ÑÌÍÝ…±±½Ý•(€€€‰äÑ¡”½ÕÑ•È€ÔÀÌÄ…Ñ ¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹‘•±•¹Ø ‰	I=]MI}A}UI0ˆ°É…¥Í¥¹œõ…±Í”¤(€€€™…­”€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€±•…¹ÕÁ}…±±}‰É½ÝÍ•ÉÌõ±…µ‰‘„è9½¹”°(€€€€€€€}•Ñ}‘Á}½Ù•ÉÉ¥‘”õ±…µ‰‘„è½Ì¹•¹Ù¥É½¸¹•Ð ‰	I=]MI}A}UI0ˆ°€ˆˆ¤°(€€€€¤(€€€Ý¥Ñ Á…Ñ ¹‘¥Ð¡ÍåÌ¹µ½‘Õ±•Ì°ì‰Ñ½½±Ì¹‰É½ÝÍ•É}Ñ½½±}±¥™•å±”ˆè™…­•ô¤è(€€€€€€€}ÍÑÕ‰}ÕÉ±½Á•¸¡µ½¹­•åÁ…Ñ °½¬õQÉÕ”¤(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰‰É½ÝÍ•È¹µ…¹…”ˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰…Ñ¥½¸ˆè€‰½¹¹•Ðˆ°€‰ÕÉ°ˆè9½¹•ô°(€€€€€€€€€€€ô(€€€€€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰½¹¹•Ñ•‰t¥ÌQÉÕ”(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÕÉ°‰t€ôô€‰¡ÑÑÀè¼¼ÄÈÜ¸À¸À¸ÄèäÈÈÈˆ(()‘•˜Ñ•ÍÑ}‰É½ÝÍ•É}µ…¹…•}½¹¹•Ñ}É•©•ÑÍ}¹½¹}ÍÑÉ¥¹}ÕÉ°¡µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹‘•±•¹Ø ‰	I=]MI}A}UI0ˆ°É…¥Í¥¹œõ…±Í”¤(€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰‰É½ÝÍ•È¹µ…¹…”ˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì‰…Ñ¥½¸ˆè€‰½¹¹•Ðˆ°€‰ÕÉ°ˆè€äÈÈÉô°(€€€€€€€ô(€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÄÔ(€€€…ÍÍ•ÉÐ€‰	I=]MI}A}UI0ˆ¹½Ð¥¸½Ì¹•¹Ù¥É½¸(()‘•˜Ñ•ÍÑ}‰É½ÝÍ•É}µ…¹…•}½¹¹•Ñ}‘•™…Õ±Ñ}±½…±}É•ÑÉ¥•Í}…™Ñ•É}±…Õ¹ ¡µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹‘•±•¹Ø ‰	I=]MI}A}UI0ˆ°É…¥Í¥¹œõ…±Í”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¥µ”°€‰Í±••Àˆ°±…µ‰‘„}Í•½¹‘Ìè9½¹”¤(€€€™…­”€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€±•…¹ÕÁ}…±±}‰É½ÝÍ•ÉÌõ±…µ‰‘„è9½¹”°(€€€€€€€}•Ñ}‘Á}½Ù•ÉÉ¥‘”õ±…µ‰‘„è½Ì¹•¹Ù¥É½¸¹•Ð ‰	I=]MI}A}UI0ˆ°€ˆˆ¤°(€€€€¤((€€€±…ÍÌ}I•ÍÀè(€€€€€€€ÍÑ…ÑÕÌ€ô€ÈÀÀ((€€€€€€€‘•˜}}•¹Ñ•É}|¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸Í•±˜((€€€€€€€‘•˜}}•á¥Ñ}|¡Í•±˜°€©|¤è(€€€€€€€€€€€É•ÑÕÉ¸…±Í”((€€€€Œ%AØÐ…¹ÍÝ•ÉÌ½¹±ä™É½´Ñ¡”€ÍÉÁÉ½‰”½¹Ý…É‘Ì€¡‰É½ÝÍ•ÈÍÑ¥±°ÍÑ…ÉÑ¥¹œ¤ì(€€€€ŒÑ¡”%AØØ±½½Á‰…¬¹•Ù•È…¹ÍÝ•ÉÌ¸(€€€…ÑÑ•µÁÑÌ€ôì‰¸ˆè€Áô((€€€‘•˜}½Á•¹•È¡ÕÉ°°Ñ¥µ•½ÕÐôÈ¸À¤è€€Œ¹½Å„èIÀÀÄƒŠPµ…Ñ ÕÉ±±¥ˆÍ¥¹…ÑÕÉ”(€€€€€€€¥˜€‰lèèÅtˆ¥¸ÕÉ°è(€€€€€€€€€€€É…¥Í”=MÉÉ½È ‰¹¼%AØØ±¥ÍÑ•¹•Èˆ¤(€€€€€€€…ÑÑ•µÁÑÍl‰¸‰t€¬ô€Ä(€€€€€€€¥˜…ÑÑ•µÁÑÍl‰¸‰t€ð€Ìè(€€€€€€€€€€€É…¥Í”=MÉÉ½È ‰¹½ÐÉ•…‘äˆ¤(€€€€€€€É•ÑÕÉ¸}I•ÍÀ ¤((€€€¥µÁ½ÉÐÕÉ±±¥ˆ¹É•ÅÕ•ÍÐ((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡ÕÉ±±¥ˆ¹É•ÅÕ•ÍÐ¹=Á•¹•É¥É•Ñ½È°€‰½Á•¸ˆ°±…µ‰‘„}Í•±˜°ÕÉ°°€©„°Ñ¥µ•½ÕÐôÈ¸À°€¨©¬è}½Á•¹•È¡ÕÉ°°Ñ¥µ•½ÕÐõÑ¥µ•½ÕÐ¤¤(€€€±…Õ¹¡•€ô¡É½µ••‰Õ1…Õ¹ ¡±…Õ¹¡•õQÉÕ”¤(€€€Ý¥Ñ Á…Ñ ¹‘¥Ð¡ÍåÌ¹µ½‘Õ±•Ì°ì‰Ñ½½±Ì¹‰É½ÝÍ•É}Ñ½½±}±¥™•å±”ˆè™…­•ô¤è(€€€€€€€Ý¥Ñ € (€€€€€€€€€€€Á…Ñ  (€€€€€€€€€€€€€€€€‰¡•Éµ•Í}±¤¹‰É½ÝÍ•É}½¹¹•Ð¹±…Õ¹¡}¡É½µ•}‘•‰Õœˆ°(€€€€€€€€€€€€€€€É•ÑÕÉ¹}Ù…±Õ”õ±…Õ¹¡•°(€€€€€€€€€€€€¤°(€€€€€€€€€€€Á…Ñ  ‰¡•Éµ•Í}±¤¹‰É½ÝÍ•É}½¹¹•Ð¹±½…±}Á½ÉÑ}¥¹}ÕÍ”ˆ°É•ÑÕÉ¹}Ù…±Õ”õ…±Í”¤°(€€€€€€€€¤è(€€€€€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰‰É½ÝÍ•È¹µ…¹…”ˆ°€‰Á…É…µÌˆèì‰…Ñ¥½¸ˆè€‰½¹¹•Ð‰õô(€€€€€€€€€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰½¹¹•Ñ•‰t¥ÌQÉÕ”(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÕÉ°‰t€ôô€‰¡ÑÑÀè¼¼ÄÈÜ¸À¸À¸ÄèäÈÈÈˆ(€€€…ÍÍ•ÉÐ½Ì¹•¹Ù¥É½¹l‰	I=]MI}A}UI0‰t€ôô€‰¡ÑÑÀè¼¼ÄÈÜ¸À¸À¸ÄèäÈÈÈˆ(()‘•˜Ñ•ÍÑ}‰É½ÝÍ•É}µ…¹…•}½¹¹•Ñ}™¥¹‘Í}¥ÁØÙ}½¹±å}‰É½ÝÍ•È¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰I•É•ÍÍ¥½¸è…¸%‘•‰Õ•ÈÍÅÕ…ÑÑ¥¹œ€ÄÈÜ¸À¸À¸ÄèäÈÈÈÁÕÍ¡•ÌÑ¡”‘•‰Õœ(€€€‰É½ÝÍ•È½¹Ñ¼lèèÅtèäÈÈÈ¸½¹¹•ÐµÕÍÐ‘¥Í½Ù•È…¹…‘½ÁÐÑ¡”%AØØ(€€€•¹‘Á½¥¹Ð¥¹ÍÑ•…½˜Ñ¥µ¥¹œ½ÕÐ……¥¹ÍÐÑ¡”ÍÅÕ…ÑÑ•È¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹‘•±•¹Ø ‰	I=]MI}A}UI0ˆ°É…¥Í¥¹œõ…±Í”¤(€€€™…­”€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€±•…¹ÕÁ}…±±}‰É½ÝÍ•ÉÌõ±…µ‰‘„è9½¹”°(€€€€€€€}•Ñ}‘Á}½Ù•ÉÉ¥‘”õ±…µ‰‘„è½Ì¹•¹Ù¥É½¸¹•Ð ‰	I=]MI}A}UI0ˆ°€ˆˆ¤°(€€€€¤((€€€±…ÍÌ}I•ÍÀè(€€€€€€€ÍÑ…ÑÕÌ€ô€ÈÀÀ((€€€€€€€‘•˜}}•¹Ñ•É}|¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸Í•±˜((€€€€€€€‘•˜}}•á¥Ñ}|¡Í•±˜°€©|¤è(€€€€€€€€€€€É•ÑÕÉ¸…±Í”((€€€‘•˜}½Á•¹•È¡ÕÉ°°Ñ¥µ•½ÕÐôÈ¸À¤è€€Œ¹½Å„èIÀÀÄƒŠPµ…Ñ ÕÉ±±¥ˆÍ¥¹…ÑÕÉ”(€€€€€€€¥˜€‰lèèÅtˆ¥¸ÕÉ°è(€€€€€€€€€€€É•ÑÕÉ¸}I•ÍÀ ¤(€€€€€€€É…¥Í”=MÉÉ½È ‰%AØÐ±½½Á‰…¬¡•±‰ä„¹½¸µ@ÍÅÕ…ÑÑ•Èˆ¤((€€€¥µÁ½ÉÐÕÉ±±¥ˆ¹É•ÅÕ•ÍÐ((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡ÕÉ±±¥ˆ¹É•ÅÕ•ÍÐ¹=Á•¹•É¥É•Ñ½È°€‰½Á•¸ˆ°±…µ‰‘„}Í•±˜°ÕÉ°°€©„°Ñ¥µ•½ÕÐôÈ¸À°€¨©¬è}½Á•¹•È¡ÕÉ°°Ñ¥µ•½ÕÐõÑ¥µ•½ÕÐ¤¤(€€€Ý¥Ñ Á…Ñ ¹‘¥Ð¡ÍåÌ¹µ½‘Õ±•Ì°ì‰Ñ½½±Ì¹‰É½ÝÍ•É}Ñ½½±}±¥™•å±”ˆè™…­•ô¤è(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰‰É½ÝÍ•È¹µ…¹…”ˆ°€‰Á…É…µÌˆèì‰…Ñ¥½¸ˆè€‰½¹¹•Ð‰õô(€€€€€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰½¹¹•Ñ•‰t¥ÌQÉÕ”(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÕÉ°‰t€ôô€‰¡ÑÑÀè¼½lèèÅtèäÈÈÈˆ(€€€…ÍÍ•ÉÐ½Ì¹•¹Ù¥É½¹l‰	I=]MI}A}UI0‰t€ôô€‰¡ÑÑÀè¼½lèèÅtèäÈÈÈˆ(()‘•˜Ñ•ÍÑ}‰É½ÝÍ•É}µ…¹…•}½¹¹•Ñ}ÍÅÕ…ÑÑ•‘}Á½ÉÑ}±…Õ¹¡•Í}½¹}…±Ñ•É¹…Ñ”¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰]¡•¸¹•¥Ñ¡•È±½½Á‰…¬ÍÁ•…­Ì@‰ÕÐÑ¡”Á½ÉÐ¥Ì¡•±‰ä…¹½Ñ¡•È(€€€…ÁÁ±¥…Ñ¥½¸°½¹¹•ÐµÕÍÐÁ¥¬…¸…±Ñ•É¹…Ñ”Á½ÉÐ™½ÈÑ¡”±…Õ¹ …¹(€€€Í…äÍ¼ƒŠP¹•Ù•È™¥¡ÐÑ¡”ÍÅÕ…ÑÑ•È™½È€äÈÈÈ¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹‘•±•¹Ø ‰	I=]MI}A}UI0ˆ°É…¥Í¥¹œõ…±Í”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¥µ”°€‰Í±••Àˆ°±…µ‰‘„}Í•½¹‘Ìè9½¹”¤(€€€™…­”€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€±•…¹ÕÁ}…±±}‰É½ÝÍ•ÉÌõ±…µ‰‘„è9½¹”°(€€€€€€€}•Ñ}‘Á}½Ù•ÉÉ¥‘”õ±…µ‰‘„è½Ì¹•¹Ù¥É½¸¹•Ð ‰	I=]MI}A}UI0ˆ°€ˆˆ¤°(€€€€¤((€€€±…ÍÌ}I•ÍÀè(€€€€€€€ÍÑ…ÑÕÌ€ô€ÈÀÀ((€€€€€€€‘•˜}}•¹Ñ•É}|¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸Í•±˜((€€€€€€€‘•˜}}•á¥Ñ}|¡Í•±˜°€©|¤è(€€€€€€€€€€€É•ÑÕÉ¸…±Í”((€€€‘•˜}½Á•¹•È¡ÕÉ°°Ñ¥µ•½ÕÐôÈ¸À¤è€€Œ¹½Å„èIÀÀÄƒŠPµ…Ñ ÕÉ±±¥ˆÍ¥¹…ÑÕÉ”(€€€€€€€¥˜€ˆèäÈÈÌˆ¥¸ÕÉ°…¹€ˆÄÈÜ¸À¸À¸Äˆ¥¸ÕÉ°è(€€€€€€€€€€€É•ÑÕÉ¸}I•ÍÀ ¤€€ŒÉ•±…Õ¹¡•‰É½ÝÍ•È½µ•ÌÕÀ½¸Ñ¡”…±Ñ•É¹…Ñ”Á½ÉÐ(€€€€€€€É…¥Í”=MÉÉ½È ˆäÈÈÈÍÅÕ…ÑÑ•€¼¹½Ñ¡¥¹œ•±Í”±¥ÍÑ•¹¥¹œˆ¤((€€€¥µÁ½ÉÐÕÉ±±¥ˆ¹É•ÅÕ•ÍÐ((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡ÕÉ±±¥ˆ¹É•ÅÕ•ÍÐ¹=Á•¹•É¥É•Ñ½È°€‰½Á•¸ˆ°±…µ‰‘„}Í•±˜°ÕÉ°°€©„°Ñ¥µ•½ÕÐôÈ¸À°€¨©¬è}½Á•¹•È¡ÕÉ°°Ñ¥µ•½ÕÐõÑ¥µ•½ÕÐ¤¤(€€€±…Õ¹¡}Á½ÉÑÌè±¥ÍÑm¥¹Ñt€ômt((€€€‘•˜}±…Õ¹ ¡Á½ÉÐ°}ÍåÍÑ•´¤è(€€€€€€€±…Õ¹¡}Á½ÉÑÌ¹…ÁÁ•¹¡Á½ÉÐ¤(€€€€€€€É•ÑÕÉ¸¡É½µ••‰Õ1…Õ¹ ¡±…Õ¹¡•õQÉÕ”¤((€€€Ý¥Ñ Á…Ñ ¹‘¥Ð¡ÍåÌ¹µ½‘Õ±•Ì°ì‰Ñ½½±Ì¹‰É½ÝÍ•É}Ñ½½±}±¥™•å±”ˆè™…­•ô¤è(€€€€€€€Ý¥Ñ € (€€€€€€€€€€€Á…Ñ  ‰¡•Éµ•Í}±¤¹‰É½ÝÍ•É}½¹¹•Ð¹±…Õ¹¡}¡É½µ•}‘•‰Õœˆ°Í¥‘•}•™™•Ðõ}±…Õ¹ ¤°(€€€€€€€€€€€Á…Ñ  ‰¡•Éµ•Í}±¤¹‰É½ÝÍ•É}½¹¹•Ð¹±½…±}Á½ÉÑ}¥¹}ÕÍ”ˆ°É•ÑÕÉ¹}Ù…±Õ”õQÉÕ”¤°(€€€€€€€€€€€Á…Ñ  ‰¡•Éµ•Í}±¤¹‰É½ÝÍ•É}½¹¹•Ð¹™¥¹‘}™É••}‘•‰Õ}Á½ÉÐˆ°É•ÑÕÉ¹}Ù…±Õ”ôäÈÈÌ¤°(€€€€€€€€¤è(€€€€€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰‰É½ÝÍ•È¹µ…¹…”ˆ°€‰Á…É…µÌˆèì‰…Ñ¥½¸ˆè€‰½¹¹•Ð‰õô(€€€€€€€€€€€€¤((€€€…ÍÍ•ÉÐ±…Õ¹¡}Á½ÉÑÌ€ôôläÈÈÍt(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰½¹¹•Ñ•‰t¥ÌQÉÕ”(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÕÉ°‰t€ôô€‰¡ÑÑÀè¼¼ÄÈÜ¸À¸À¸ÄèäÈÈÌˆ(€€€…ÍÍ•ÉÐ½Ì¹•¹Ù¥É½¹l‰	I=]MI}A}UI0‰t€ôô€‰¡ÑÑÀè¼¼ÄÈÜ¸À¸À¸ÄèäÈÈÌˆ(()‘•˜Ñ•ÍÑ}‰É½ÝÍ•É}µ…¹…•}½¹¹•Ñ}É•©•ÑÍ}Õ¹É•…¡…‰±•}•¹‘Á½¥¹Ð¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰¸Õ¹É•…¡…‰±”•¹‘Á½¥¹ÐµÕÍÐ9=PµÕÑ…Ñ”Ñ¡”•¹Ø½ÈÉ•…ÀÍ•ÍÍ¥½¹Ì¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰	I=]MI}A}UI0ˆ°€‰¡ÑÑÀè¼½•á¥ÍÑ¥¹œèäÈÈÈˆ¤(€€€±•…¹ÕÁ}…±±Ìè±¥ÍÑmÍÑÉt€ômt(€€€™…­”€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€±•…¹ÕÁ}…±±}‰É½ÝÍ•ÉÌõ±…µ‰‘„è±•…¹ÕÁ}…±±Ì¹…ÁÁ•¹ (€€€€€€€€€€€½Ì¹•¹Ù¥É½¸¹•Ð ‰	I=]MI}A}UI0ˆ°€ˆˆ¤(€€€€€€€€¤°(€€€€€€€}•Ñ}‘Á}½Ù•ÉÉ¥‘”õ±…µ‰‘„è½Ì¹•¹Ù¥É½¸¹•Ð ‰	I=]MI}A}UI0ˆ°€ˆˆ¤°(€€€€¤(€€€Ý¥Ñ Á…Ñ ¹‘¥Ð¡ÍåÌ¹µ½‘Õ±•Ì°ì‰Ñ½½±Ì¹‰É½ÝÍ•É}Ñ½½±}±¥™•å±”ˆè™…­•ô¤è(€€€€€€€}ÍÑÕ‰}ÕÉ±½Á•¸¡µ½¹­•åÁ…Ñ °½¬õ…±Í”¤(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰‰É½ÝÍ•È¹µ…¹…”ˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰…Ñ¥½¸ˆè€‰½¹¹•Ðˆ°€‰ÕÉ°ˆè€‰¡ÑÑÀè¼½Õ¹É•…¡…‰±”èäÈÈÈ‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤((€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¥¸É•ÍÀ(€€€€Œ¹ØÁÉ•Í•ÉÙ•ì¹½Ñ¡¥¹œÉ•…Á•¸(€€€…ÍÍ•ÉÐ½Ì¹•¹Ù¥É½¹l‰	I=]MI}A}UI0‰t€ôô€‰¡ÑÑÀè¼½•á¥ÍÑ¥¹œèäÈÈÈˆ(€€€…ÍÍ•ÉÐ±•…¹ÕÁ}…±±Ì€ôômt(()‘•˜Ñ•ÍÑ}‰É½ÝÍ•É}µ…¹…•}½¹¹•Ñ}¹½Éµ…±¥é•Í}‰…É•}¡½ÍÑ}Á½ÉÐ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰A•ÉÍ¥ÍÐ„Á…ÉÍ•Í¡•µ”è¼½¡½ÍÐéÁ½ÉÑ€UI0Í¼}•Ñ}‘Á}½Ù•ÉÉ¥‘•€(€€€…¸¹½Éµ…±¥é”¥ÐìÍÑ½É¥¹œ„‰…É”¡½ÍÐéÁ½ÉÐÝ½Õ±‰É•…¬ÍÕ‰Í•ÅÕ•¹Ð(€€€Ñ½½°…±±Ì€¡½Á¥±½ÐÉ•Ù¥•Ü½¸€ŒÄÜÄÈÀ¤¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹‘•±•¹Ø ‰	I=]MI}A}UI0ˆ°É…¥Í¥¹œõ…±Í”¤(€€€™…­”€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€±•…¹ÕÁ}…±±}‰É½ÝÍ•ÉÌõ±…µ‰‘„è9½¹”°(€€€€€€€}•Ñ}‘Á}½Ù•ÉÉ¥‘”õ±…µ‰‘„è½Ì¹•¹Ù¥É½¸¹•Ð ‰	I=]MI}A}UI0ˆ°€ˆˆ¤°(€€€€¤(€€€Ý¥Ñ Á…Ñ ¹‘¥Ð¡ÍåÌ¹µ½‘Õ±•Ì°ì‰Ñ½½±Ì¹‰É½ÝÍ•É}Ñ½½±}±¥™•å±”ˆè™…­•ô¤è(€€€€€€€}ÍÑÕ‰}ÕÉ±½Á•¸¡µ½¹­•åÁ…Ñ °½¬õQÉÕ”¤(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰‰É½ÝÍ•È¹µ…¹…”ˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰…Ñ¥½¸ˆè€‰½¹¹•Ðˆ°€‰ÕÉ°ˆè€ˆÄÈÜ¸À¸À¸ÄèäÈÈÈ‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰½¹¹•Ñ•‰t¥ÌQÉÕ”(€€€€Œ	…É”¡½ÍÐéÁ½ÉÐ½ÐÁÉ½µ½Ñ•Ñ¼„™Õ±°UI0Ý¥Ñ •áÁ±¥¥ÐÍ¡•µ”¸(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÕÉ°‰t¹ÍÑ…ÉÑÍÝ¥Ñ  ‰¡ÑÑÀè¼¼ˆ¤(€€€…ÍÍ•ÉÐ½Ì¹•¹Ù¥É½¹l‰	I=]MI}A}UI0‰t¹ÍÑ…ÉÑÍÝ¥Ñ  ‰¡ÑÑÀè¼¼ˆ¤(()‘•˜Ñ•ÍÑ}‰É½ÝÍ•É}µ…¹…•}½¹¹•Ñ}ÍÑÉ¥ÁÍ}‘¥Í½Ù•Éå}Á…Ñ ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰UÍ•ÈµÍÕÁÁ±¥•‘¥Í½Ù•ÉäÁ…Ñ¡Ì±¥­”€½©Í½¹€½È€½©Í½¸½Ù•ÉÍ¥½¹€(€€€µÕÍÐ½±±…ÁÍ”Ñ¼‰…É”Í¡•µ”è¼½¡½ÍÐéÁ½ÉÑ€ì½Ñ¡•ÉÝ¥Í”(€€€}É•Í½±Ù•}‘Á}½Ù•ÉÉ¥‘•€Ý¥±°…ÁÁ•¹€½©Í½¸½Ù•ÉÍ¥½¹€……¥¸…¹(€€€ÁÉ½‘Õ”„‘ÕÁ±¥…Ñ”Á…Ñ €¡½Á¥±½ÐÉ•Ù¥•ÜÉ½Õ¹´È½¸€ŒÄÜÄÈÀ¤¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹‘•±•¹Ø ‰	I=]MI}A}UI0ˆ°É…¥Í¥¹œõ…±Í”¤(€€€™…­”€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€±•…¹ÕÁ}…±±}‰É½ÝÍ•ÉÌõ±…µ‰‘„è9½¹”°(€€€€€€€}•Ñ}‘Á}½Ù•ÉÉ¥‘”õ±…µ‰‘„è½Ì¹•¹Ù¥É½¸¹•Ð ‰	I=]MI}A}UI0ˆ°€ˆˆ¤°(€€€€¤(€€€Ý¥Ñ Á…Ñ ¹‘¥Ð¡ÍåÌ¹µ½‘Õ±•Ì°ì‰Ñ½½±Ì¹‰É½ÝÍ•É}Ñ½½±}±¥™•å±”ˆè™…­•ô¤è(€€€€€€€}ÍÑÕ‰}ÕÉ±½Á•¸¡µ½¹­•åÁ…Ñ °½¬õQÉÕ”¤(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰‰É½ÝÍ•È¹µ…¹…”ˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰…Ñ¥½¸ˆè€‰½¹¹•Ðˆ°€‰ÕÉ°ˆè€‰¡ÑÑÀè¼¼ÄÈÜ¸À¸À¸ÄèäÈÈÈ½©Í½¸‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰½¹¹•Ñ•‰t¥ÌQÉÕ”(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÕÉ°‰t€ôô€‰¡ÑÑÀè¼¼ÄÈÜ¸À¸À¸ÄèäÈÈÈˆ(€€€…ÍÍ•ÉÐ½Ì¹•¹Ù¥É½¹l‰	I=]MI}A}UI0‰t€ôô€‰¡ÑÑÀè¼¼ÄÈÜ¸À¸À¸ÄèäÈÈÈˆ(()‘•˜Ñ•ÍÑ}‰É½ÝÍ•É}µ…¹…•}½¹¹•Ñ}ÁÉ•Í•ÉÙ•Í}‘•ÙÑ½½±Í}‰É½ÝÍ•É}•¹‘Á½¥¹Ð¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰½¹É•Ñ”‘•ÙÑ½½±ÌÝ•‰Í½­•Ð•¹‘Á½¥¹ÑÌ€¡”¹œ¸	É½ÝÍ•É‰…Í”¤µÕÍÐ(€€€ÍÕÉÙ¥Ù”Ù•É‰…Ñ¥´ƒŠPÝ”½¹±ä½±±…ÁÍ”‘¥Í½Ù•ÉäµÍÑå±”Á…Ñ¡Ì¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹‘•±•¹Ø ‰	I=]MI}A}UI0ˆ°É…¥Í¥¹œõ…±Í”¤(€€€™…­”€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€±•…¹ÕÁ}…±±}‰É½ÝÍ•ÉÌõ±…µ‰‘„è9½¹”°(€€€€€€€}•Ñ}‘Á}½Ù•ÉÉ¥‘”õ±…µ‰‘„è½Ì¹•¹Ù¥É½¸¹•Ð ‰	I=]MI}A}UI0ˆ°€ˆˆ¤°(€€€€¤(€€€½¹É•Ñ”€ô€‰ÝÌè¼½‰É½ÝÍ•É‰…Í”¹•á…µÁ±”½‘•ÙÑ½½±Ì½‰É½ÝÍ•È½…‰ŒÄÈÌˆ((€€€±…ÍÌ}=­M½­•Ðè(€€€€€€€‘•˜}}•¹Ñ•É}|¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸Í•±˜((€€€€€€€‘•˜}}•á¥Ñ}|¡Í•±˜°€©„¤è(€€€€€€€€€€€É•ÑÕÉ¸…±Í”((€€€Ý¥Ñ Á…Ñ ¹‘¥Ð¡ÍåÌ¹µ½‘Õ±•Ì°ì‰Ñ½½±Ì¹‰É½ÝÍ•É}Ñ½½±}±¥™•å±”ˆè™…­•ô¤è(€€€€€€€€Œ%˜ÕÉ±½Á•¸¥ÌÉ•…¡•™½È„½¹É•Ñ”ÝÌ•¹‘Á½¥¹Ð°Ñ¡”Ñ•ÍÐ(€€€€€€€€ŒÝ½Õ±ÍÑ¥±°Á…ÍÌ‰•…ÕÍ”}ÍÑÕ‰}ÕÉ±½Á•¸É•ÑÕÉ¹•½¬õQÉÕ”‰•™½É”ì(€€€€€€€€ŒÁ…Ñ ¥ÐÑ¼…ÍÍ•ÉÐµ™…¥°Í¼Ý”ÁÉ½Ù”Ñ¡”!QQ@ÁÉ½‰”¥ÌÍ­¥ÁÁ•¸(€€€€€€€Ý¥Ñ Á…Ñ  (€€€€€€€€€€€€‰ÕÉ±±¥ˆ¹É•ÅÕ•ÍÐ¹ÕÉ±½Á•¸ˆ°Í¥‘•}•™™•ÐõÍÍ•ÉÑ¥½¹ÉÉ½È ‰ÕÉ±½Á•¸…±±•ˆ¤(€€€€€€€€¤è(€€€€€€€€€€€Ý¥Ñ Á…Ñ  ‰Í½­•Ð¹É•…Ñ•}½¹¹•Ñ¥½¸ˆ°É•ÑÕÉ¹}Ù…±Õ”õ}=­M½­•Ð ¤¤è(€€€€€€€€€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰‰É½ÝÍ•È¹µ…¹…”ˆ°(€€€€€€€€€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰…Ñ¥½¸ˆè€‰½¹¹•Ðˆ°€‰ÕÉ°ˆè½¹É•Ñ•ô°(€€€€€€€€€€€€€€€€€€€ô(€€€€€€€€€€€€€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰½¹¹•Ñ•‰t¥ÌQÉÕ”(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÕÉ°‰t€ôô½¹É•Ñ”(€€€…ÍÍ•ÉÐ½Ì¹•¹Ù¥É½¹l‰	I=]MI}A}UI0‰t€ôô½¹É•Ñ”(()‘•˜Ñ•ÍÑ}‰É½ÝÍ•É}µ…¹…•}½¹¹•Ñ}±½…±}‘•ÙÑ½½±Í}ÝÍ}ÁÉ•Í•ÉÙ•Í}Á…Ñ ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰I•É•ÍÍ¥½¸èÝÌè¼¼ÄÈÜ¸À¸À¸ÄèäÈÈÈ½‘•ÙÑ½½±Ì½‰É½ÝÍ•È¼ñ¥ù€¥Ì„É•…°(€€€½¹¹•Ñ…‰±”•¹‘Á½¥¹Ðì‘•™…Õ±Ðµ±½…°¹½Éµ…±¥é…Ñ¥½¸µÕÍÐ¹½ÐÍÑÉ¥ÀÑ¡”(€€€€½‘•ÙÑ½½±Ì½‰É½ÝÍ•È¼¸¸¹€Á…Ñ ½È¥Ð‰É•…­ÌÙ…±¥±½…°@½¹¹•ÑÌ¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹‘•±•¹Ø ‰	I=]MI}A}UI0ˆ°É…¥Í¥¹œõ…±Í”¤(€€€™…­”€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€±•…¹ÕÁ}…±±}‰É½ÝÍ•ÉÌõ±…µ‰‘„è9½¹”°(€€€€€€€}•Ñ}‘Á}½Ù•ÉÉ¥‘”õ±…µ‰‘„è½Ì¹•¹Ù¥É½¸¹•Ð ‰	I=]MI}A}UI0ˆ°€ˆˆ¤°(€€€€¤(€€€½¹É•Ñ”€ô€‰ÝÌè¼¼ÄÈÜ¸À¸À¸ÄèäÈÈÈ½‘•ÙÑ½½±Ì½‰É½ÝÍ•È½…‰ŒÄÈÌˆ((€€€±…ÍÌ}=­M½­•Ðè(€€€€€€€‘•˜}}•¹Ñ•É}|¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸Í•±˜((€€€€€€€‘•˜}}•á¥Ñ}|¡Í•±˜°€©„¤è(€€€€€€€€€€€É•ÑÕÉ¸…±Í”((€€€Ý¥Ñ Á…Ñ ¹‘¥Ð¡ÍåÌ¹µ½‘Õ±•Ì°ì‰Ñ½½±Ì¹‰É½ÝÍ•É}Ñ½½±}±¥™•å±”ˆè™…­•ô¤è(€€€€€€€Ý¥Ñ Á…Ñ  ‰Í½­•Ð¹É•…Ñ•}½¹¹•Ñ¥½¸ˆ°É•ÑÕÉ¹}Ù…±Õ”õ}=­M½­•Ð ¤¤è(€€€€€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰‰É½ÝÍ•È¹µ…¹…”ˆ°(€€€€€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰…Ñ¥½¸ˆè€‰½¹¹•Ðˆ°€‰ÕÉ°ˆè½¹É•Ñ•ô°(€€€€€€€€€€€€€€€ô(€€€€€€€€€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰½¹¹•Ñ•‰t¥ÌQÉÕ”(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÕÉ°‰t€ôô½¹É•Ñ”(€€€…ÍÍ•ÉÐ½Ì¹•¹Ù¥É½¹l‰	I=]MI}A}UI0‰t€ôô½¹É•Ñ”(()‘•˜Ñ•ÍÑ}‰É½ÝÍ•É}µ…¹…•}½¹¹•Ñ}É•©•ÑÍ}¥¹Ù…±¥‘}Á½ÉÐ¡µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹‘•±•¹Ø ‰	I=]MI}A}UI0ˆ°É…¥Í¥¹œõ…±Í”¤(€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰‰É½ÝÍ•È¹µ…¹…”ˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì‰…Ñ¥½¸ˆè€‰½¹¹•Ðˆ°€‰ÕÉ°ˆè€‰¡ÑÑÀè¼½±½…±¡½ÍÐé…‰Œ‰ô°(€€€€€€€ô(€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÄÔ(€€€…ÍÍ•ÉÐ€‰	I=]MI}A}UI0ˆ¹½Ð¥¸½Ì¹•¹Ù¥É½¸(()‘•˜Ñ•ÍÑ}‰É½ÝÍ•É}µ…¹…•}½¹¹•Ñ}É•©•ÑÍ}µ¥ÍÍ¥¹}¡½ÍÐ¡µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹‘•±•¹Ø ‰	I=]MI}A}UI0ˆ°É…¥Í¥¹œõ…±Í”¤(€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰‰É½ÝÍ•È¹µ…¹…”ˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì‰…Ñ¥½¸ˆè€‰½¹¹•Ðˆ°€‰ÕÉ°ˆè€‰¡ÑÑÀè¼¼èäÈÈÈ‰ô°(€€€€€€€ô(€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÄÔ(€€€…ÍÍ•ÉÐ€‰	I=]MI}A}UI0ˆ¹½Ð¥¸½Ì¹•¹Ù¥É½¸(()‘•˜Ñ•ÍÑ}‰É½ÝÍ•É}µ…¹…•}½¹¹•Ñ}½¹É•Ñ•}ÝÍ}Í­¥ÁÍ}¡ÑÑÁ}ÁÉ½‰”¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰I•É•ÍÍ¥½¸™½ÈÉ½Õ¹´È½Á¥±½ÐÉ•Ù¥•Üè„¡½ÍÑ•@•¹‘Á½¥¹Ð(€€€€¡¹¼!QQ@‘¥Í½Ù•Éä¤µÕÍÐ½¹¹•ÐÙ¥„Q@µ½¹±äÉ•…¡…‰¥±¥Ñä¡•¬¸(€€€Q¡”!QQ@ÁÉ½‰”ÕÍ•Ñ¼É•©•ÐÑ¡•Í”•Ù•¸Ñ¡½Õ Ñ¡•äÉ”Ù…±¥¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹‘•±•¹Ø ‰	I=]MI}A}UI0ˆ°É…¥Í¥¹œõ…±Í”¤(€€€™…­”€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€±•…¹ÕÁ}…±±}‰É½ÝÍ•ÉÌõ±…µ‰‘„è9½¹”°(€€€€€€€}•Ñ}‘Á}½Ù•ÉÉ¥‘”õ±…µ‰‘„è½Ì¹•¹Ù¥É½¸¹•Ð ‰	I=]MI}A}UI0ˆ°€ˆˆ¤°(€€€€¤(€€€½¹É•Ñ”€ô€‰ÝÍÌè¼½¡É½µ”¹‰É½ÝÍ•É±•ÍÌ¹¥¼½‘•ÙÑ½½±Ì½‰É½ÝÍ•È½Í•ÍÌ´Äˆ((€€€Í••¹}Ñ…É•ÑÌè±¥ÍÑmÑÕÁ±•mÍÑÈ°¥¹Ñut€ômt((€€€±…ÍÌ}=­M½­•Ðè(€€€€€€€‘•˜}}•¹Ñ•É}|¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸Í•±˜((€€€€€€€‘•˜}}•á¥Ñ}|¡Í•±˜°€©„¤è(€€€€€€€€€€€É•ÑÕÉ¸…±Í”((€€€‘•˜}™…­•}É•…Ñ•}½¹¹•Ñ¥½¸¡…‘‘È°Ñ¥µ•½ÕÐõ9½¹”¤è(€€€€€€€Í••¹}Ñ…É•ÑÌ¹…ÁÁ•¹¡…‘‘È¤(€€€€€€€É•ÑÕÉ¸}=­M½­•Ð ¤((€€€Ý¥Ñ Á…Ñ ¹‘¥Ð¡ÍåÌ¹µ½‘Õ±•Ì°ì‰Ñ½½±Ì¹‰É½ÝÍ•É}Ñ½½±}±¥™•å±”ˆè™…­•ô¤è(€€€€€€€€ŒÕÉ±½Á•¸Ý½Õ±€ÐÀÐ½=99IUM½¸„É•…°¡½ÍÑ•@•¹‘Á½¥¹Ðì(€€€€€€€€Œ…ÍÍ•ÉÑ¥¹œ¥ÐÌ¹•Ù•È…±±•ÁÉ½Ù•ÌÑ¡”ÁÉ½‰”Ý…ÌÍ­¥ÁÁ•¸(€€€€€€€Ý¥Ñ Á…Ñ  (€€€€€€€€€€€€‰ÕÉ±±¥ˆ¹É•ÅÕ•ÍÐ¹ÕÉ±½Á•¸ˆ°Í¥‘•}•™™•ÐõÍÍ•ÉÑ¥½¹ÉÉ½È ‰ÕÉ±½Á•¸…±±•ˆ¤(€€€€€€€€¤è(€€€€€€€€€€€Ý¥Ñ Á…Ñ  ‰Í½­•Ð¹É•…Ñ•}½¹¹•Ñ¥½¸ˆ°Í¥‘•}•™™•Ðõ}™…­•}É•…Ñ•}½¹¹•Ñ¥½¸¤è(€€€€€€€€€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰‰É½ÝÍ•È¹µ…¹…”ˆ°(€€€€€€€€€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰…Ñ¥½¸ˆè€‰½¹¹•Ðˆ°€‰ÕÉ°ˆè½¹É•Ñ•ô°(€€€€€€€€€€€€€€€€€€€ô(€€€€€€€€€€€€€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰t€ôôì‰½¹¹•Ñ•ˆèQÉÕ”°€‰ÕÉ°ˆè½¹É•Ñ•ô(€€€€ŒÝÍÌƒŠHÁ½ÉÐ€ÐÐÌ°¡½ÍÐÁÉ•Í•ÉÙ•Ù•É‰…Ñ¥´¸(€€€…ÍÍ•ÉÐÍ••¹}Ñ…É•ÑÌ€ôôl ‰¡É½µ”¹‰É½ÝÍ•É±•ÍÌ¹¥¼ˆ°€ÐÐÌ¥t(()‘•˜Ñ•ÍÑ}‰É½ÝÍ•É}µ…¹…•}½¹¹•Ñ}½¹É•Ñ•}ÝÍ}ÑÁ}Õ¹É•…¡…‰±”¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰%˜Ñ¡”Q@É•…¡…‰¥±¥Ñä¡•¬™…¥±Ì™½È„½¹É•Ñ”ÝÌ•¹‘Á½¥¹Ð°(€€€É•ÑÕÉ¸„±•…È€ÔÀÌÄ•ÉÉ½ÈƒŠP¹¼™…±±‰…¬Ñ¼Ñ¡”!QQ@ÁÉ½‰”€¡Ý¡¥ (€€€…¸¹•Ù•ÈÍÕ••™½ÈÑ¡•Í”UI1Ì…¹åÝ…ä¤¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹‘•±•¹Ø ‰	I=]MI}A}UI0ˆ°É…¥Í¥¹œõ…±Í”¤(€€€™…­”€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€±•…¹ÕÁ}…±±}‰É½ÝÍ•ÉÌõ±…µ‰‘„è9½¹”°(€€€€€€€}•Ñ}‘Á}½Ù•ÉÉ¥‘”õ±…µ‰‘„è½Ì¹•¹Ù¥É½¸¹•Ð ‰	I=]MI}A}UI0ˆ°€ˆˆ¤°(€€€€¤(€€€½¹É•Ñ”€ô€‰ÝÌè¼½½™™±¥¹”¹•á…µÁ±”½‘•ÙÑ½½±Ì½‰É½ÝÍ•È½µ¥ÍÍ¥¹œˆ((€€€Ý¥Ñ Á…Ñ ¹‘¥Ð¡ÍåÌ¹µ½‘Õ±•Ì°ì‰Ñ½½±Ì¹‰É½ÝÍ•É}Ñ½½±}±¥™•å±”ˆè™…­•ô¤è(€€€€€€€Ý¥Ñ Á…Ñ  ‰Í½­•Ð¹É•…Ñ•}½¹¹•Ñ¥½¸ˆ°Í¥‘•}•™™•Ðõ=MÉÉ½È ‰=99IUMˆ¤¤è(€€€€€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰‰É½ÝÍ•È¹µ…¹…”ˆ°(€€€€€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰…Ñ¥½¸ˆè€‰½¹¹•Ðˆ°€‰ÕÉ°ˆè½¹É•Ñ•ô°(€€€€€€€€€€€€€€€ô(€€€€€€€€€€€€¤((€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¥¸É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÔÀÌÄ(()‘•˜Ñ•ÍÑ}‰É½ÝÍ•É}µ…¹…•}‘¥Í½¹¹•Ñ}‘É½ÁÍ}•¹Ù}…¹‘}±•…¹Ì¡µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰	I=]MI}A}UI0ˆ°€‰¡ÑÑÀè¼¼ÄÈÜ¸À¸À¸ÄèäÈÈÈˆ¤(€€€±•…¹ÕÁ}½Õ¹Ð€ôì‰¸ˆè€Áô(€€€™…­”€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€±•…¹ÕÁ}…±±}‰É½ÝÍ•ÉÌõ±…µ‰‘„è±•…¹ÕÁ}½Õ¹Ð¹}}Í•Ñ¥Ñ•µ}| (€€€€€€€€€€€€‰¸ˆ°±•…¹ÕÁ}½Õ¹Ñl‰¸‰t€¬€Ä(€€€€€€€€¤°(€€€€€€€}•Ñ}‘Á}½Ù•ÉÉ¥‘”õ±…µ‰‘„è½Ì¹•¹Ù¥É½¸¹•Ð ‰	I=]MI}A}UI0ˆ°€ˆˆ¤°(€€€€¤(€€€Ý¥Ñ Á…Ñ ¹‘¥Ð¡ÍåÌ¹µ½‘Õ±•Ì°ì‰Ñ½½±Ì¹‰É½ÝÍ•É}Ñ½½±}±¥™•å±”ˆè™…­•ô¤è(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰‰É½ÝÍ•È¹µ…¹…”ˆ°€‰Á…É…µÌˆèì‰…Ñ¥½¸ˆè€‰‘¥Í½¹¹•Ð‰õô(€€€€€€€€¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰t€ôôì‰½¹¹•Ñ•ˆè…±Í•ô(€€€…ÍÍ•ÉÐ€‰	I=]MI}A}UI0ˆ¹½Ð¥¸½Ì¹•¹Ù¥É½¸(€€€€ŒQÝ¼±•…¹ÕÁÌè½¹”‰•™½É”•¹ØÉ•µ½Ù…°°½¹”…™Ñ•È°µ…Ñ¡¥¹œ½¹¹•Ð¸(€€€…ÍÍ•ÉÐ±•…¹ÕÁ}½Õ¹Ñl‰¸‰t€ôô€È(((ŒƒŠRŠR ½¹™¥œ¹•Ð¥¹‘¥…Ñ½È¹½Éµ…±¥é…Ñ¥½¸ƒŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠR (((()‘•˜Ñ•ÍÑ}½¹™¥}•Ñ}¥¹‘¥…Ñ½É}¹½Éµ…±¥é•Í}…Í¥¹}…¹‘}Ý¡¥Ñ•ÍÁ…”¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰!…¹µ•‘¥Ñ•½¹™¥œ¹å…µ°ÍÑ…åÌ½¹Í¥ÍÑ•¹ÐÝ¥Ñ Ý¡…ÐÑ¡”QU$Í¡½ÝÌ¸((€€€É½¹Ñ•¹Ì¹½Éµ…±¥é•%¹‘¥…Ñ½ÉMÑå±•€±½Ý•É…Í•Ì€¬ÑÉ¥µÌ°Í¼½¹™¥œ¹•Ð(€€€µÕÍÐ‘¼Ñ¡”Í…µ”ƒŠP½Ñ¡•ÉÝ¥Í”€½¥¹‘¥…Ñ½É€ÁÉ¥¹ÑÌ€5=)$€œÝ¡¥±”Ñ¡”(€€€U$¥Ì…ÑÕ…±±äÉ•¹‘•É¥¹œÑ¡”­…½µ½©¤‘•™…Õ±Ð¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}±½…‘}™œˆ°±…µ‰‘„èì‰‘¥ÍÁ±…äˆèì‰ÑÕ¥}ÍÑ…ÑÕÍ}¥¹‘¥…Ñ½Èˆè€ˆ5=)$€‰õô(€€€€¤(€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰½¹™¥œ¹•Ðˆ°€‰Á…É…µÌˆèì‰­•äˆè€‰¥¹‘¥…Ñ½È‰õô(€€€€¤(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰t€ôôì‰Ù…±Õ”ˆè€‰•µ½©¤‰ô(()‘•˜Ñ•ÍÑ}½¹™¥}•Ñ}¥¹‘¥…Ñ½É}™…±±Í}‰…­}Ñ½}‘•™…Õ±Ñ}™½É}Õ¹­¹½Ý¸¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰¸Õ¹­¹½Ý¸Ù…±Õ”¥¸½¹™¥œ¹å…µ°™…±±Ì‰…¬Ñ¼Ñ¡”Í…µ”‘•™…Õ±Ð(€€€Ñ¡”™É½¹Ñ•¹ÕÍ•Ì€¡}%9%Q=I}U1Q€¤¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}±½…‘}™œˆ°±…µ‰‘„èì‰‘¥ÍÁ±…äˆèì‰ÑÕ¥}ÍÑ…ÑÕÍ}¥¹‘¥…Ñ½Èˆè€‰É…¥¹‰½Ü‰õô(€€€€¤(€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰½¹™¥œ¹•Ðˆ°€‰Á…É…µÌˆèì‰­•äˆè€‰¥¹‘¥…Ñ½È‰õô(€€€€¤(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰t€ôôì‰Ù…±Õ”ˆè€‰­…½µ½©¤‰ô(((((ŒƒŠRŠR ½¹™¥œ¹Í•Ð¥¹‘¥…Ñ½ÈÙ…±¥‘…Ñ¥½¸ƒŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠR (()‘•˜Ñ•ÍÑ}½¹™¥}Í•Ñ}¥¹‘¥…Ñ½É}…•ÁÑÍ}­¹½Ý¹}Ù…±Õ”¡µ½¹­•åÁ…Ñ ¤è(€€€ÝÉ¥ÑÑ•¸è‘¥Ð€ôíô(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°(€€€€€€€€‰}ÝÉ¥Ñ•}½¹™¥}­•äˆ°(€€€€€€€±…µ‰‘„¬°ØèÝÉ¥ÑÑ•¸¹ÕÁ‘…Ñ”¡í¬èÙô¤°(€€€€¤(€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰½¹™¥œ¹Í•Ðˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì‰­•äˆè€‰¥¹‘¥…Ñ½Èˆ°€‰Ù…±Õ”ˆè€‰5=)$‰ô°(€€€€€€€ô(€€€€¤(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰t€ôôì‰­•äˆè€‰¥¹‘¥…Ñ½Èˆ°€‰Ù…±Õ”ˆè€‰•µ½©¤‰ô(€€€…ÍÍ•ÉÐÝÉ¥ÑÑ•¸€ôôì‰‘¥ÍÁ±…ä¹ÑÕ¥}ÍÑ…ÑÕÍ}¥¹‘¥…Ñ½Èˆè€‰•µ½©¤‰ô(((((((ŒƒŠRŠR É•±½…¹•¹ØƒŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠR (((()‘•˜Ñ•ÍÑ}É•±½…‘}•¹Ù}ÉÁ}ÍÕÉ™…•Í}•ÉÉ½ÉÌ¡µ½¹­•åÁ…Ñ ¤è(€€€‘•˜}‰É½­•¸ ¤è(€€€€€€€É…¥Í”IÕ¹Ñ¥µ•ÉÉ½È ‰•¹ØÁ…Ñ ±½­•ˆ¤((€€€™…­”€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡É•±½…‘}•¹Øõ}‰É½­•¸¤(€€€Ý¥Ñ Á…Ñ ¹‘¥Ð¡ÍåÌ¹µ½‘Õ±•Ì°ì‰¡•Éµ•Í}±¤¹½¹™¥œˆè™…­•ô¤è(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ¡ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰É•±½…¹•¹Øˆ°€‰Á…É…µÌˆèíõô¤((€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¥¸É•ÍÀ(€€€…ÍÍ•ÉÐ€‰•¹ØÁ…Ñ ±½­•ˆ¥¸É•ÍÁl‰•ÉÉ½È‰ul‰µ•ÍÍ…”‰t(((ŒƒŠRŠR µ…á}¥Ñ•É…Ñ¥½¹Ì½¹™¥œÉ•…‘¥¹œƒŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠR (()‘•˜}Í•ÑÕÁ}µ…­•}…•¹Ñ}µ½­Ì¡µ½¹­•åÁ…Ñ °™œ¤è(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}™œˆ°±…µ‰‘„è™œ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}É•Í½±Ù•}ÍÑ…ÉÑÕÁ}ÉÕ¹Ñ¥µ”ˆ°±…µ‰‘„è€ ‰Ñ•ÍÐµµ½‘•°ˆ°9½¹”¤(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€‰¡•Éµ•Í}±¤¹ÉÕ¹Ñ¥µ•}ÁÉ½Ù¥‘•È¹É•Í½±Ù•}ÉÕ¹Ñ¥µ•}ÁÉ½Ù¥‘•Èˆ°(€€€€€€€±…µ‰‘„É•ÅÕ•ÍÑ•õ9½¹”°Ñ…É•Ñ}µ½‘•°õ9½¹”èì(€€€€€€€€€€€€‰ÁÉ½Ù¥‘•Èˆè9½¹”°(€€€€€€€€€€€€‰‰…Í•}ÕÉ°ˆè9½¹”°(€€€€€€€€€€€€‰…Á¥}­•äˆè9½¹”°(€€€€€€€€€€€€‰…Á¥}µ½‘”ˆè9½¹”°(€€€€€€€€€€€€‰½µµ…¹ˆè9½¹”°(€€€€€€€€€€€€‰…ÉÌˆè9½¹”°(€€€€€€€€€€€€‰É•‘•¹Ñ¥…±}Á½½°ˆè9½¹”°(€€€€€€€ô°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}Ñ½½±}ÁÉ½É•ÍÍ}µ½‘”ˆ°±…µ‰‘„è€‰½™˜ˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}É•…Í½¹¥¹}½¹™¥œˆ°±…µ‰‘„µ½‘•°ôˆˆè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}Í•ÉÙ¥•}Ñ¥•Èˆ°±…µ‰‘„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}•¹…‰±•‘}Ñ½½±Í•ÑÌˆ°±…µ‰‘„€©}„°€¨©}­Üè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}…•¹Ñ}‰Ìˆ°±…µ‰‘„Í¥èíô¤(()‘•˜Ñ•ÍÑ}µ…­•}…•¹Ñ}É•…‘Í}¹•ÍÑ•‘}µ…á}ÑÕÉ¹Ì¡µ½¹­•åÁ…Ñ ¤è(€€€}Í•ÑÕÁ}µ…­•}…•¹Ñ}µ½­Ì¡µ½¹­•åÁ…Ñ °ì‰…•¹Ðˆèì‰µ…á}ÑÕÉ¹Ìˆè€ÈÀÁõô¤((€€€Ý¥Ñ Á…Ñ  ‰ÉÕ¹}…•¹Ð¹%•¹Ðˆ¤…Ìµ½­}…•¹Ðè(€€€€€€€Í•ÉÙ•È¹}µ…­•}…•¹Ð ‰Í¥Äˆ°€‰­•äÄˆ¤((€€€…ÍÍ•ÉÐµ½­}…•¹Ð¹…±±}…ÉÌ¹­Ý…ÉÍl‰µ…á}¥Ñ•É…Ñ¥½¹Ì‰t€ôô€ÈÀÀ(((()‘•˜Ñ•ÍÑ}µ…­•}…•¹Ñ}¹•ÍÑ•‘}µ…á}ÑÕÉ¹Í}Ñ…­•Í}ÁÉ¥½É¥Ñä¡µ½¹­•åÁ…Ñ ¤è(€€€}Í•ÑÕÁ}µ…­•}…•¹Ñ}µ½­Ì (€€€€€€€µ½¹­•åÁ…Ñ °ì‰…•¹Ðˆèì‰µ…á}ÑÕÉ¹Ìˆè€ÐÀÁô°€‰µ…á}ÑÕÉ¹Ìˆè€ÄÀÁô(€€€€¤((€€€Ý¥Ñ Á…Ñ  ‰ÉÕ¹}…•¹Ð¹%•¹Ðˆ¤…Ìµ½­}…•¹Ðè(€€€€€€€Í•ÉÙ•È¹}µ…­•}…•¹Ð ‰Í¥Äˆ°€‰­•äÄˆ¤((€€€…ÍÍ•ÉÐµ½­}…•¹Ð¹…±±}…ÉÌ¹­Ý…ÉÍl‰µ…á}¥Ñ•É…Ñ¥½¹Ì‰t€ôô€ÐÀÀ(((()‘•˜Ñ•ÍÑ}µ…­•}…•¹Ñ}ÕÍ•Í}Í•ÍÍ¥½¹}ÉÕ¹Ñ¥µ•}½Ù•ÉÉ¥‘•Ì¡µ½¹­•åÁ…Ñ ¤è(€€€}Í•ÑÕÁ}µ…­•}…•¹Ñ}µ½­Ì¡µ½¹­•åÁ…Ñ °íô¤(€€€É•Í½±Ù•€ôíô((€€€‘•˜™…­•}É•Í½±Ù•}ÉÕ¹Ñ¥µ•}ÁÉ½Ù¥‘•È¡É•ÅÕ•ÍÑ•õ9½¹”°Ñ…É•Ñ}µ½‘•°õ9½¹”¤è(€€€€€€€É•Í½±Ù•‘l‰É•ÅÕ•ÍÑ•‰t€ôÉ•ÅÕ•ÍÑ•(€€€€€€€É•Í½±Ù•‘l‰Ñ…É•Ñ}µ½‘•°‰t€ôÑ…É•Ñ}µ½‘•°(€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€‰ÁÉ½Ù¥‘•ÈˆèÉ•ÅÕ•ÍÑ•°(€€€€€€€€€€€€‰‰…Í•}ÕÉ°ˆè9½¹”°(€€€€€€€€€€€€‰…Á¥}­•äˆè9½¹”°(€€€€€€€€€€€€‰…Á¥}µ½‘”ˆè9½¹”°(€€€€€€€€€€€€‰½µµ…¹ˆè9½¹”°(€€€€€€€€€€€€‰…ÉÌˆè9½¹”°(€€€€€€€€€€€€‰É•‘•¹Ñ¥…±}Á½½°ˆè9½¹”°(€€€€€€€ô((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€‰¡•Éµ•Í}±¤¹ÉÕ¹Ñ¥µ•}ÁÉ½Ù¥‘•È¹É•Í½±Ù•}ÉÕ¹Ñ¥µ•}ÁÉ½Ù¥‘•Èˆ°(€€€€€€€™…­•}É•Í½±Ù•}ÉÕ¹Ñ¥µ•}ÁÉ½Ù¥‘•È°(€€€€¤((€€€Ý¥Ñ Á…Ñ  ‰ÉÕ¹}…•¹Ð¹%•¹Ðˆ¤…Ìµ½­}…•¹Ðè(€€€€€€€Í•ÉÙ•È¹}µ…­•}…•¹Ð (€€€€€€€€€€€€‰Í¥Äˆ°(€€€€€€€€€€€€‰­•äÄˆ°(€€€€€€€€€€€µ½‘•±}½Ù•ÉÉ¥‘”ô‰ÁÐ´Ô¸Ðˆ°(€€€€€€€€€€€ÁÉ½Ù¥‘•É}½Ù•ÉÉ¥‘”ô‰½Á•¹…¤µ½‘•àˆ°(€€€€€€€€€€€É•…Í½¹¥¹}½¹™¥}½Ù•ÉÉ¥‘”õì‰•¹…‰±•ˆèQÉÕ”°€‰•™™½ÉÐˆè€‰¡¥ ‰ô°(€€€€€€€€€€€Í•ÉÙ¥•}Ñ¥•É}½Ù•ÉÉ¥‘”ô‰ÁÉ¥½É¥Ñäˆ°(€€€€€€€€¤((€€€…ÍÍ•ÉÐÉ•Í½±Ù•€ôôì‰É•ÅÕ•ÍÑ•ˆè€‰½Á•¹…¤µ½‘•àˆ°€‰Ñ…É•Ñ}µ½‘•°ˆè€‰ÁÐ´Ô¸Ð‰ô(€€€…ÍÍ•ÉÐµ½­}…•¹Ð¹…±±}…ÉÌ¹­Ý…ÉÍl‰µ½‘•°‰t€ôô€‰ÁÐ´Ô¸Ðˆ(€€€…ÍÍ•ÉÐµ½­}…•¹Ð¹…±±}…ÉÌ¹­Ý…ÉÍl‰ÁÉ½Ù¥‘•È‰t€ôô€‰½Á•¹…¤µ½‘•àˆ(€€€…ÍÍ•ÉÐµ½­}…•¹Ð¹…±±}…ÉÌ¹­Ý…ÉÍl‰É•…Í½¹¥¹}½¹™¥œ‰t€ôôì‰•¹…‰±•ˆèQÉÕ”°€‰•™™½ÉÐˆè€‰¡¥ ‰ô(€€€…ÍÍ•ÉÐµ½­}…•¹Ð¹…±±}…ÉÌ¹­Ý…ÉÍl‰Í•ÉÙ¥•}Ñ¥•È‰t€ôô€‰ÁÉ¥½É¥Ñäˆ(()‘•˜Ñ•ÍÑ}µ…­•}…•¹Ñ}¡…¹‘±•Í}¹Õ±±}…•¹Ñ}½¹™¥œ¡µ½¹­•åÁ…Ñ ¤è(€€€}Í•ÑÕÁ}µ…­•}…•¹Ñ}µ½­Ì¡µ½¹­•åÁ…Ñ °ì‰…•¹Ðˆè9½¹”°€‰µ…á}ÑÕÉ¹Ìˆè€àÁô¤((€€€Ý¥Ñ Á…Ñ  ‰ÉÕ¹}…•¹Ð¹%•¹Ðˆ¤…Ìµ½­}…•¹Ðè(€€€€€€€Í•ÉÙ•È¹}µ…­•}…•¹Ð ‰Í¥Äˆ°€‰­•äÄˆ¤((€€€…ÍÍ•ÉÐµ½­}…•¹Ð¹…±±}…ÉÌ¹­Ý…ÉÍl‰µ…á}¥Ñ•É…Ñ¥½¹Ì‰t€ôô€àÀ(()±…ÍÌ}…­••¹Ñ½É	…­É½Õ¹è(€€€‰…Í•}ÕÉ°€ô9½¹”(€€€…Á¥}­•ä€ô9½¹”(€€€ÁÉ½Ù¥‘•È€ô9½¹”(€€€…Á¥}µ½‘”€ô9½¹”(€€€…Á}½µµ…¹€ô9½¹”(€€€…Á}…ÉÌ€ô9½¹”(€€€µ½‘•°€ô€‰Ñ•ÍÐµµ½‘•°ˆ(€€€•¹…‰±•‘}Ñ½½±Í•ÑÌ€ô9½¹”(€€€•Á¡•µ•É…±}ÍåÍÑ•µ}ÁÉ½µÁÐ€ô9½¹”(€€€ÁÉ½Ù¥‘•ÉÍ}…±±½Ý•€ô9½¹”(€€€ÁÉ½Ù¥‘•ÉÍ}¥¹½É•€ô9½¹”(€€€ÁÉ½Ù¥‘•ÉÍ}½É‘•È€ô9½¹”(€€€ÁÉ½Ù¥‘•É}Í½ÉÐ€ô9½¹”(€€€ÁÉ½Ù¥‘•É}É•ÅÕ¥É•}Á…É…µ•Ñ•ÉÌ€ô…±Í”(€€€ÁÉ½Ù¥‘•É}‘…Ñ…}½±±•Ñ¥½¸€ô9½¹”(€€€É•…Í½¹¥¹}½¹™¥œ€ô9½¹”(€€€Í•ÉÙ¥•}Ñ¥•È€ô9½¹”(€€€É•ÅÕ•ÍÑ}½Ù•ÉÉ¥‘•Ì€ôíô(€€€}™…±±‰…­}µ½‘•°€ô9½¹”(()‘•˜Ñ•ÍÑ}‰…­É½Õ¹‘}…•¹Ñ}­Ý…ÉÍ}É•…‘Í}¹•ÍÑ•‘}µ…á}ÑÕÉ¹Ì¡µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}™œˆ°±…µ‰‘„èì‰…•¹Ðˆèì‰µ…á}ÑÕÉ¹Ìˆè€ÌÀÁõô¤((€€€­Ý…ÉÌ€ôÍ•ÉÙ•È¹}‰…­É½Õ¹‘}…•¹Ñ}­Ý…ÉÌ¡}…­••¹Ñ½É	…­É½Õ¹ ¤°€‰Ñ…Í­|Äˆ¤((€€€…ÍÍ•ÉÐ­Ý…ÉÍl‰µ…á}¥Ñ•É…Ñ¥½¹Ì‰t€ôô€ÌÀÀ(()‘•˜Ñ•ÍÑ}‰…­É½Õ¹‘}…•¹Ñ}­Ý…ÉÍ}™…±±Í}‰…­}Ñ½}É½½Ñ}µ…á}ÑÕÉ¹Ì¡µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}™œˆ°±…µ‰‘„èì‰µ…á}ÑÕÉ¹Ìˆè€ÔÁô¤((€€€­Ý…ÉÌ€ôÍ•ÉÙ•È¹}‰…­É½Õ¹‘}…•¹Ñ}­Ý…ÉÌ¡}…­••¹Ñ½É	…­É½Õ¹ ¤°€‰Ñ…Í­|Äˆ¤((€€€…ÍÍ•ÉÐ­Ý…ÉÍl‰µ…á}¥Ñ•É…Ñ¥½¹Ì‰t€ôô€ÔÀ(((()‘•˜Ñ•ÍÑ}‰…­É½Õ¹‘}…•¹Ñ}­Ý…ÉÍ}¡…¹‘±•Í}¹Õ±±}…•¹Ñ}½¹™¥œ¡µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}™œˆ°±…µ‰‘„èì‰…•¹Ðˆè9½¹”°€‰µ…á}ÑÕÉ¹Ìˆè€ÐÁô¤((€€€­Ý…ÉÌ€ôÍ•ÉÙ•È¹}‰…­É½Õ¹‘}…•¹Ñ}­Ý…ÉÌ¡}…­••¹Ñ½É	…­É½Õ¹ ¤°€‰Ñ…Í­|Äˆ¤((€€€…ÍÍ•ÉÐ­Ý…ÉÍl‰µ…á}¥Ñ•É…Ñ¥½¹Ì‰t€ôô€ÐÀ(((()‘•˜Ñ•ÍÑ}¹½Ñ¥™¥…Ñ¥½¹}Á½±±•É}‘•±¥Ù•ÉÍ}½µÁ±•Ñ¥½¸¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰A½±±•ÈÁ¥­ÌÕÀ½µÁ±•Ñ¥½¸•Ù•¹ÑÌ…¹ÑÉ¥•ÉÌ…•¹ÐÑÕÉ¹Ì¸ˆˆˆ(€€€¥µÁ½ÉÐÅÕ•Õ”…Ì}ÅÕ•Õ•}µ½((€€€™É½´Ñ½½±Ì¹ÁÉ½•ÍÍ}É•¥ÍÑÉä¥µÁ½ÉÐÁÉ½•ÍÍ}É•¥ÍÑÉä((€€€ÑÕÉ¹Ì€ômt(€€€•µ¥ÑÑ•€ômt((€€€±…ÍÌ}•¹Ðè(€€€€€€€‘•˜ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°ÁÉ½µÁÐ°½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éäõ9½¹”°ÍÑÉ•…µ}…±±‰…¬õ9½¹”°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€ÑÕÉ¹Ì¹…ÁÁ•¹¡ÁÉ½µÁÐ¤(€€€€€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€€€€€‰™¥¹…±}É•ÍÁ½¹Í”ˆè€‰½¬ˆ°(€€€€€€€€€€€€€€€€‰µ•ÍÍ…•Ìˆèmì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰½¬‰õt°(€€€€€€€€€€€ô((€€€±…ÍÌ}%µµ•‘¥…Ñ•Q¡É•…è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©}Ñ¡É•…‘}½ÁÑ¥½¹Ì¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð€ôÑ…É•Ð(€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð ¤((€€€Í•ÍÌ€ô}Í•ÍÍ¥½¸¡…•¹Ðõ}•¹Ð ¤¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‘}Á½±°‰t€ôÍ•ÍÌ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}%µµ•‘¥…Ñ•Q¡É•…¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„°€¨©­Üè•µ¥ÑÑ•¹…ÁÁ•¹¡„¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰µ…­•}ÍÑÉ•…µ}É•¹‘•É•Èˆ°±…µ‰‘„½±Ìè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰É•¹‘•É}µ•ÍÍ…”ˆ°±…µ‰‘„É…Ü°½±Ìè9½¹”¤((€€€€Œ%Í½±…Ñ”Ñ¡”½µÁ±•Ñ¥½¸ÅÕ•Õ”™½ÈÑ¡”‘ÕÉ…Ñ¥½¸½˜Ñ¡¥ÌÑ•ÍÐ¸Q¡”Á½±±•È(€€€€ŒÉ•…‘ÌÁÉ½•ÍÍ}É•¥ÍÑÉä¹½µÁ±•Ñ¥½¹}ÅÕ•Õ”‰ä…ÑÑÉ¥‰ÕÑ”…ÐÉÕ¹Ñ¥µ”ìÑ¡”(€€€€Œ•Ù•¹Ð‰•±½Ü…ÉÉ¥•Ì¹¼Í•ÍÍ¥½¹}­•ä°Í¼…¹ä€©½Ñ¡•È¨Á½±±•È€¡„±•…­•(€€€€Œ‘…•µ½¸Ñ¡É•…™É½´…¹½Ñ¡•ÈÑ•ÍÐ°½È„½¹ÕÉÉ•¹Ð½¹”¥¸Ñ¡”Í…µ”á‘¥ÍÐ(€€€€ŒÝ½É­•È¤¥Ì…±±½Ý•Ñ¼‘•ÅÕ•Õ”…¹‘¥ÍÁ…Ñ ¥ÐÑ¼¥ÑÌ½Ý¸Í•ÍÍ¥½¸ƒŠPÝ¡½Í”(€€€€Œ…•¹Ðµ…ä‰”„™¥áÑÕÉ”‘½Õ‰±”Ý¥Ñ¡½ÕÐÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¸™É•Í EÕ•Õ”(€€€€Œ¡•É”™Õ±±ä¥Í½±…Ñ•ÌÑ¡¥ÌÑ•ÍÐìµ½¹­•åÁ…Ñ É•ÍÑ½É•ÌÑ¡”½É¥¥¹…°½¸(€€€€ŒÑ•…É‘½Ý¸¸€¡M…µ”Á…ÑÑ•É¸…ÌÑ•ÍÑ}¹½Ñ¥™¥…Ñ¥½¹}Á½±±•É}É•ÅÕ•Õ•Í}Ý¡•¹}‰ÕÍä¸¤(€€€¥Í½±…Ñ•‘}ÅÕ•Õ”è}ÅÕ•Õ•}µ½¹EÕ•Õ”€ô}ÅÕ•Õ•}µ½¹EÕ•Õ” ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡ÁÉ½•ÍÍ}É•¥ÍÑÉä°€‰½µÁ±•Ñ¥½¹}ÅÕ•Õ”ˆ°¥Í½±…Ñ•‘}ÅÕ•Õ”¤(€€€ÁÉ½•ÍÍ}É•¥ÍÑÉä¹}½µÁ±•Ñ¥½¹}½¹ÍÕµ•¹‘¥Í…É ‰ÁÉ½}Á½±±•É}Ñ•ÍÐˆ¤((€€€ÍÑ½À€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤((€€€€ŒAÕÐ•Ù•¹Ð½¸ÅÕ•Õ”°Ñ¡•¸¥µµ•‘¥…Ñ•±äÍ¥¹…°ÍÑ½ÀÍ¼Ñ¡”Á½±±•È(€€€€ŒÉÕ¹Ì•á…Ñ±ä½¹”¥Ñ•É…Ñ¥½¸¸(€€€¥Í½±…Ñ•‘}ÅÕ•Õ”¹ÁÕÐ¡ì(€€€€€€€€‰ÑåÁ”ˆè€‰½µÁ±•Ñ¥½¸ˆ°(€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰ÁÉ½}Á½±±•É}Ñ•ÍÐˆ°(€€€€€€€€‰½µµ…¹ˆè€‰•¡¼¡•±±¼ˆ°(€€€€€€€€‰•á¥Ñ}½‘”ˆè€À°(€€€€€€€€‰½ÕÑÁÕÐˆè€‰¡•±±¼ˆ°(€€€ô¤(€€€ÍÑ½À¹Í•Ð ¤((€€€ÑÉäè(€€€€€€€Í•ÉÙ•È¹}¹½Ñ¥™¥…Ñ¥½¹}Á½±±•É}±½½À¡ÍÑ½À°€‰Í¥‘}Á½±°ˆ°Í•ÍÌ¤((€€€€€€€€ŒM¡½Õ±¡…Ù”•µ¥ÑÑ•„ÍÑ…ÑÕÌ¹ÕÁ‘…Ñ”Ý¥Ñ ­¥¹õÁÉ½•ÍÌ(€€€€€€€ÍÑ…ÑÕÍ}…±±Ì€ôm„™½È„¥¸•µ¥ÑÑ•¥˜…lÁt€ôô€‰ÍÑ…ÑÕÌ¹ÕÁ‘…Ñ”‰t(€€€€€€€…ÍÍ•ÉÐ±•¸¡ÍÑ…ÑÕÍ}…±±Ì¤€øô€Ä(€€€€€€€…ÍÍ•ÉÐÍÑ…ÑÕÍ}…±±ÍlÁulÉul‰­¥¹‰t€ôô€‰ÁÉ½•ÍÌˆ((€€€€€€€€ŒM¡½Õ±¡…Ù”ÑÉ¥•É•…¸…•¹ÐÑÕÉ¸(€€€€€€€…ÍÍ•ÉÐ±•¸¡ÑÕÉ¹Ì¤€ôô€Ä(€€€€€€€…ÍÍ•ÉÐ€‰ÁÉ½}Á½±±•É}Ñ•ÍÐˆ¥¸ÑÕÉ¹ÍlÁt(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥‘}Á½±°ˆ°9½¹”¤(€€€€€€€Ý¡¥±”¹½ÐÁÉ½•ÍÍ}É•¥ÍÑÉä¹½µÁ±•Ñ¥½¹}ÅÕ•Õ”¹•µÁÑä ¤è(€€€€€€€€€€€ÁÉ½•ÍÍ}É•¥ÍÑÉä¹½µÁ±•Ñ¥½¹}ÅÕ•Õ”¹•Ñ}¹½Ý…¥Ð ¤(()‘•˜Ñ•ÍÑ}¹½Ñ¥™¥…Ñ¥½¹}Á½±±•É}Í­¥ÁÍ}½¹ÍÕµ•¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰±É•…‘äµ½¹ÍÕµ•½µÁ±•Ñ¥½¹Ì…É”¹½Ð‘¥ÍÁ…Ñ¡•‰äÑ¡”Á½±±•È¸ˆˆˆ(€€€¥µÁ½ÉÐÅÕ•Õ”…Ì}ÅÕ•Õ•}µ½((€€€™É½´Ñ½½±Ì¹ÁÉ½•ÍÍ}É•¥ÍÑÉä¥µÁ½ÉÐÁÉ½•ÍÍ}É•¥ÍÑÉä((€€€ÑÕÉ¹Ì€ômt((€€€±…ÍÌ}•¹Ðè(€€€€€€€‘•˜ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°ÁÉ½µÁÐ°½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éäõ9½¹”°ÍÑÉ•…µ}…±±‰…¬õ9½¹”°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€ÑÕÉ¹Ì¹…ÁÁ•¹¡ÁÉ½µÁÐ¤(€€€€€€€€€€€É•ÑÕÉ¸ì‰™¥¹…±}É•ÍÁ½¹Í”ˆè€‰½¬ˆ°€‰µ•ÍÍ…•Ìˆèmuô((€€€±…ÍÌ}%µµ•‘¥…Ñ•Q¡É•…è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©}Ñ¡É•…‘}½ÁÑ¥½¹Ì¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð€ôÑ…É•Ð(€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð ¤((€€€Í•ÍÌ€ô}Í•ÍÍ¥½¸¡…•¹Ðõ}•¹Ð ¤¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‘}Í­¥À‰t€ôÍ•ÍÌ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}%µµ•‘¥…Ñ•Q¡É•…¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„°€¨©­Üè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰µ…­•}ÍÑÉ•…µ}É•¹‘•É•Èˆ°±…µ‰‘„½±Ìè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰É•¹‘•É}µ•ÍÍ…”ˆ°±…µ‰‘„É…Ü°½±Ìè9½¹”¤((€€€€Œ%Í½±…Ñ”Ñ¡”½µÁ±•Ñ¥½¸ÅÕ•Õ”Í¼„½¹ÕÉÉ•¹Ð½±•…­•Á½±±•È¥¸Ñ¡”Í…µ”(€€€€Œá‘¥ÍÐÝ½É­•È…¸Ð‘•ÅÕ•Õ”Ñ¡¥ÌÍ•ÍÍ¥½¹}­•äµ±•ÍÌ•Ù•¹Ð‰•™½É”½ÕÈÁ½±±•È(€€€€Œ‘½•Ì¸µ½¹­•åÁ…Ñ É•ÍÑ½É•ÌÑ¡”Í¡…É•Í¥¹±•Ñ½¸½¸Ñ•…É‘½Ý¸¸€¡M…µ”(€€€€ŒÁ…ÑÑ•É¸…ÌÑ•ÍÑ}¹½Ñ¥™¥…Ñ¥½¹}Á½±±•É}É•ÅÕ•Õ•Í}Ý¡•¹}‰ÕÍä¸¤(€€€¥Í½±…Ñ•‘}ÅÕ•Õ”è}ÅÕ•Õ•}µ½¹EÕ•Õ”€ô}ÅÕ•Õ•}µ½¹EÕ•Õ” ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡ÁÉ½•ÍÍ}É•¥ÍÑÉä°€‰½µÁ±•Ñ¥½¹}ÅÕ•Õ”ˆ°¥Í½±…Ñ•‘}ÅÕ•Õ”¤((€€€ÁÉ½•ÍÍ}É•¥ÍÑÉä¹}½µÁ±•Ñ¥½¹}½¹ÍÕµ•¹…‘ ‰ÁÉ½}…±É•…‘å}‘½¹”ˆ¤(€€€¥Í½±…Ñ•‘}ÅÕ•Õ”¹ÁÕÐ¡ì(€€€€€€€€‰ÑåÁ”ˆè€‰½µÁ±•Ñ¥½¸ˆ°(€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰ÁÉ½}…±É•…‘å}‘½¹”ˆ°(€€€€€€€€‰½µµ…¹ˆè€‰•¡¼àˆ°(€€€€€€€€‰•á¥Ñ}½‘”ˆè€À°(€€€€€€€€‰½ÕÑÁÕÐˆè€‰àˆ°(€€€ô¤((€€€ÍÑ½À€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤(€€€ÍÑ½À¹Í•Ð ¤((€€€ÑÉäè(€€€€€€€Í•ÉÙ•È¹}¹½Ñ¥™¥…Ñ¥½¹}Á½±±•É}±½½À¡ÍÑ½À°€‰Í¥‘}Í­¥Àˆ°Í•ÍÌ¤(€€€€€€€…ÍÍ•ÉÐ±•¸¡ÑÕÉ¹Ì¤€ôô€À(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥‘}Í­¥Àˆ°9½¹”¤(€€€€€€€ÁÉ½•ÍÍ}É•¥ÍÑÉä¹}½µÁ±•Ñ¥½¹}½¹ÍÕµ•¹‘¥Í…É ‰ÁÉ½}…±É•…‘å}‘½¹”ˆ¤(€€€€€€€Ý¡¥±”¹½ÐÁÉ½•ÍÍ}É•¥ÍÑÉä¹½µÁ±•Ñ¥½¹}ÅÕ•Õ”¹•µÁÑä ¤è(€€€€€€€€€€€ÁÉ½•ÍÍ}É•¥ÍÑÉä¹½µÁ±•Ñ¥½¹}ÅÕ•Õ”¹•Ñ}¹½Ý…¥Ð ¤(()‘•˜Ñ•ÍÑ}¹½Ñ¥™¥…Ñ¥½¹}Á½±±•É}É•ÅÕ•Õ•Í}Ý¡•¹}‰ÕÍä¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰]¡•¸Ñ¡”…•¹Ð¥Ì‰ÕÍä°Ñ¡”Á½±±•ÈÉ•ÅÕ•Õ•ÌÑ¡”•Ù•¹Ð¸ˆˆˆ(€€€¥µÁ½ÉÐÅÕ•Õ”…Ì}ÅÕ•Õ•}µ½((€€€™É½´Ñ½½±Ì¹ÁÉ½•ÍÍ}É•¥ÍÑÉä¥µÁ½ÉÐÁÉ½•ÍÍ}É•¥ÍÑÉä((€€€•µ¥ÑÑ•€ômt((€€€Í•ÍÌ€ô}Í•ÍÍ¥½¸¡ÉÕ¹¹¥¹œõQÉÕ”¤€€Œ…•¹Ð¥Ì‰ÕÍä(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‘}‰ÕÍä‰t€ôÍ•ÍÌ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„°€¨©­Üè•µ¥ÑÑ•¹…ÁÁ•¹¡„¤¤((€€€€Œ%Í½±…Ñ”Ñ¡”½µÁ±•Ñ¥½¸ÅÕ•Õ”™½ÈÑ¡”‘ÕÉ…Ñ¥½¸½˜Ñ¡¥ÌÑ•ÍÐ¸Q¡”Á½±±•È(€€€€ŒÉ•…‘ÌÁÉ½•ÍÍ}É•¥ÍÑÉä¹½µÁ±•Ñ¥½¹}ÅÕ•Õ”‰ä…ÑÑÉ¥‰ÕÑ”…ÐÉÕ¹Ñ¥µ”°Í¼„(€€€€Œ™É•Í EÕ•Õ”¡•É”µ•…¹Ì¹¼½¹ÕÉÉ•¹Ñ±äµÉÕ¹¹¥¹œÑ•ÍÐ¥¸Ñ¡”Í…µ”á‘¥ÍÐ(€€€€ŒÝ½É­•È…¸ÁÕÐ½•Ð½¸Ñ¡”Í¡…É•Í¥¹±•Ñ½¸µ¥µÉÕ¸…¹‘É…¥¸Ñ¡”•Ù•¹Ð(€€€€ŒÝ”•áÁ•ÐÑ¼‰”É•ÅÕ•Õ•¸µ½¹­•åÁ…Ñ É•ÍÑ½É•ÌÑ¡”½É¥¥¹…°½¸Ñ•…É‘½Ý¸¸(€€€¥Í½±…Ñ•‘}ÅÕ•Õ”è}ÅÕ•Õ•}µ½¹EÕ•Õ”€ô}ÅÕ•Õ•}µ½¹EÕ•Õ” ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡ÁÉ½•ÍÍ}É•¥ÍÑÉä°€‰½µÁ±•Ñ¥½¹}ÅÕ•Õ”ˆ°¥Í½±…Ñ•‘}ÅÕ•Õ”¤(€€€ÁÉ½•ÍÍ}É•¥ÍÑÉä¹}½µÁ±•Ñ¥½¹}½¹ÍÕµ•¹‘¥Í…É ‰ÁÉ½}‰ÕÍå}Ñ•ÍÐˆ¤((€€€•ÙÐ€ôì(€€€€€€€€‰ÑåÁ”ˆè€‰½µÁ±•Ñ¥½¸ˆ°(€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰ÁÉ½}‰ÕÍå}Ñ•ÍÐˆ°(€€€€€€€€‰½µµ…¹ˆè€‰µ…­”‰Õ¥±ˆ°(€€€€€€€€‰•á¥Ñ}½‘”ˆè€À°(€€€€€€€€‰½ÕÑÁÕÐˆè€‰½¬ˆ°(€€€ô(€€€¥Í½±…Ñ•‘}ÅÕ•Õ”¹ÁÕÐ¡•ÙÐ¤((€€€ÍÑ½À€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤(€€€ÍÑ½À¹Í•Ð ¤((€€€ÑÉäè(€€€€€€€Í•ÉÙ•È¹}¹½Ñ¥™¥…Ñ¥½¹}Á½±±•É}±½½À¡ÍÑ½À°€‰Í¥‘}‰ÕÍäˆ°Í•ÍÌ¤((€€€€€€€€ŒMÑ…ÑÕÌÕÁ‘…Ñ”Ý…Ì•µ¥ÑÑ•€¡ÕÍ•ÈÍ••Ì¥Ð¤(€€€€€€€ÍÑ…ÑÕÍ}…±±Ì€ôm„™½È„¥¸•µ¥ÑÑ•¥˜…lÁt€ôô€‰ÍÑ…ÑÕÌ¹ÕÁ‘…Ñ”‰t(€€€€€€€…ÍÍ•ÉÐ±•¸¡ÍÑ…ÑÕÍ}…±±Ì¤€ôô€Ä((€€€€€€€€ŒÙ•¹ÐÝ…ÌÉ•ÅÕ•Õ•€¡…•¹ÐÝ…Ì‰ÕÍä°¹¼ÑÕÉ¸ÑÉ¥•É•¤(€€€€€€€…ÍÍ•ÉÐ¹½Ð¥Í½±…Ñ•‘}ÅÕ•Õ”¹•µÁÑä ¤(€€€€€€€É•ÅÕ•Õ•€ô¥Í½±…Ñ•‘}ÅÕ•Õ”¹•Ñ}¹½Ý…¥Ð ¤(€€€€€€€…ÍÍ•ÉÐÉ•ÅÕ•Õ•‘l‰Í•ÍÍ¥½¹}¥‰t€ôô€‰ÁÉ½}‰ÕÍå}Ñ•ÍÐˆ(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥‘}‰ÕÍäˆ°9½¹”¤(€€€€€€€Ý¡¥±”¹½ÐÁÉ½•ÍÍ}É•¥ÍÑÉä¹½µÁ±•Ñ¥½¹}ÅÕ•Õ”¹•µÁÑä ¤è(€€€€€€€€€€€ÁÉ½•ÍÍ}É•¥ÍÑÉä¹½µÁ±•Ñ¥½¹}ÅÕ•Õ”¹•Ñ}¹½Ý…¥Ð ¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}Í…Ù•}ÝÉ¥Ñ•Í}Õ¹‘•É}¡•Éµ•Í}¡½µ•}Ý¥Ñ¡}ÍåÍÑ•µ}ÁÉ½µÁÐ¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰QU$€½Í…Ù”€¡Í•ÍÍ¥½¸¹Í…Ù”IA¤µÕÍÐÍ¹…ÁÍ¡½ÐÕ¹‘•ÈÑ¡”!•Éµ•ÌÁÉ½™¥±”(€€€¡½µ”ƒŠP¹½ÐÑ¡”ÁÉ½©•Ð½Ý½É­ÍÁ…”]ƒŠP…¹¥¹±Õ‘”Ñ¡”ÍåÍÑ•´ÁÉ½µÁÐ°(€€€µ¥ÉÉ½É¥¹œÑ¡”±…ÍÍ¥Œ1$€½Í…Ù”…¹Ñ¡”‘…Í¡‰½…ÉÍ…Ù”•áÁ½ÉÐ¸((€€€I•É•ÍÍ¥½¸èÑ¡”…Ñ•Ý…ä¡…¹‘±•ÈÝÉ½Ñ”¡•Éµ•Í}½¹Ù•ÉÍ…Ñ¥½¹|¨¹©Í½¹€Ñ¼(€€€½Ì¹Á…Ñ ¹…‰ÍÁ…Ñ  ¸¸¸¥€€¡Ñ¡”Ý½É­ÍÁ…”]¤…¹½¹±ä•áÁ½ÉÑ•µ½‘•±€(€€€…¹µ•ÍÍ…•Í€°Í¼ÍåÍÑ•µ}ÁÉ½µÁÑ€Ý…Ìµ¥ÍÍ¥¹œ¸(€€€€ˆˆˆ(€€€¡½µ”€ôÑµÁ}Á…Ñ €¼€ˆ¹¡•Éµ•Ìˆ(€€€¡½µ”¹µ­‘¥È ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}!=5ˆ°ÍÑÈ¡¡½µ”¤¤((€€€€ŒIÕ¸™É½´„‘¥™™•É•¹Ð]Ñ¼ÁÉ½Ù”Ñ¡”Í¹…ÁÍ¡½Ð‘½•Ì9=P±•…¬Ñ¡•É”¸(€€€Ý½É¬€ôÑµÁ}Á…Ñ €¼€‰Ý½É­ÍÁ…”ˆ(€€€Ý½É¬¹µ­‘¥È ¤(€€€µ½¹­•åÁ…Ñ ¹¡‘¥È¡Ý½É¬¤((€€€Í¥€ô€‰Í…Ù”µÍ¥ˆ(€€€…•¹Ð€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€µ½‘•°ô‰¡•Éµ•ÌµÑ•ÍÐˆ°(€€€€€€€Í•ÍÍ¥½¹}¥ôˆÈÀÈØÀÄÀÅ|ÄÈÀÀÀÁ}…‰ŒÄÈÌˆ°(€€€€€€€Í•ÍÍ¥½¹}ÍÑ…ÉÐõ‘…Ñ•Ñ¥µ” ÈÀÈØ°€Ä°€Ä°€ÄÈ°€À°€À¤°(€€€€€€€}…¡•‘}ÍåÍÑ•µ}ÁÉ½µÁÐô‰e½Ô…É”!•Éµ•Ì¸ˆ°(€€€€¤(€€€¡¥ÍÑ½Éä€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰¡¤‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰¡•±±¼‰ô°(€€€t(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÍ¥‘t€ôì(€€€€€€€€‰…•¹Ðˆè…•¹Ð°(€€€€€€€€‰Í•ÍÍ¥½¹}­•äˆè€‰Í…Ù”µ­•äˆ°(€€€€€€€€‰¡¥ÍÑ½Éäˆè¡¥ÍÑ½Éä°(€€€€€€€€‰¡¥ÍÑ½Éå}±½¬ˆèÑ¡É•…‘¥¹œ¹1½¬ ¤°(€€€€€€€€‰É•…Ñ•‘}…Ðˆè€ÄÜÌÔÜÌÈàÀÀ¸À°(€€€ô(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹}µ•Ñ¡½‘Íl‰Í•ÍÍ¥½¸¹Í…Ù”‰t ˆÄˆ°ì‰Í•ÍÍ¥½¹}¥ˆèÍ¥‘ô¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡Í¥°9½¹”¤((€€€…ÍÍ•ÉÐ€‰É•ÍÕ±Ðˆ¥¸É•ÍÀ°É•ÍÀ(€€€Í…Ù•‘}™¥±”€ôA…Ñ ¡É•ÍÁl‰É•ÍÕ±Ð‰ul‰™¥±”‰t¤((€€€€Œ5ÕÍÐ9=P±•…¬¥¹Ñ¼Ñ¡”Ý½É­ÍÁ…”½ÁÉ½©•Ð]¸(€€€…ÍÍ•ÉÐ¹½Ð±¥ÍÐ¡Ý½É¬¹±½ˆ ‰¡•Éµ•Í}½¹Ù•ÉÍ…Ñ¥½¹|¨¹©Í½¸ˆ¤¤((€€€Í…Ù•‘}‘¥È€ô¡½µ”€¼€‰Í•ÍÍ¥½¹Ìˆ€¼€‰Í…Ù•ˆ(€€€…ÍÍ•ÉÐÍ…Ù•‘}™¥±”¹Á…É•¹Ð€ôôÍ…Ù•‘}‘¥È(€€€…ÍÍ•ÉÐÍ…Ù•‘}™¥±”¹•á¥ÍÑÌ ¤((€€€Á…å±½…€ô©Í½¸¹±½…‘Ì¡Í…Ù•‘}™¥±”¹É•…‘}Ñ•áÐ ¤¤(€€€…ÍÍ•ÉÐÁ…å±½…‘l‰µ½‘•°‰t€ôô€‰¡•Éµ•ÌµÑ•ÍÐˆ(€€€…ÍÍ•ÉÐÁ…å±½…‘l‰Í•ÍÍ¥½¹}¥‰t€ôô€ˆÈÀÈØÀÄÀÅ|ÄÈÀÀÀÁ}…‰ŒÄÈÌˆ(€€€…ÍÍ•ÉÐÁ…å±½…‘l‰Í•ÍÍ¥½¹}ÍÑ…ÉÐ‰t€ôô€ˆÈÀÈØ´ÀÄ´ÀÅPÄÈèÀÀèÀÀˆ(€€€…ÍÍ•ÉÐÁ…å±½…‘l‰ÍåÍÑ•µ}ÁÉ½µÁÐ‰t€ôô€‰e½Ô…É”!•Éµ•Ì¸ˆ(€€€…ÍÍ•ÉÐÁ…å±½…‘l‰µ•ÍÍ…•Ì‰t€ôô¡¥ÍÑ½Éä(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}Í…Ù•}ÁÉ½á¥•Í}Ñ½}½µÁÕÑ•}¡½ÍÑ}¡¥ÍÑ½Éä¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰%Í½±…Ñ•ÑÕÉ¹Ì½Ý¸¡¥ÍÑ½Éä¥¸Ñ¡”¡½ÍÐì€½Í…Ù”µÕÍÐ¹½Ð•áÁ½ÉÐÑ¡”ÍÑ…±”Á…É•¹Ðµ¥ÉÉ½È¸ˆˆˆ(€€€Í¥€ô€‰Í…Ù”µ¡½ÍÐµÍ¥ˆ(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÍ¥‘t€ô}Í•ÍÍ¥½¸¡…•¹Ðõ9½¹”°}½µÁÕÑ•}¡½ÍÑ}…Ñ¥Ù”õQÉÕ”¤(€€€…±±Ì€ômt((€€€‘•˜Í•¹‘}½¹ÑÉ½°¡½¹ÑÉ½±}Í¥°€¨©­Ý…ÉÌ¤è(€€€€€€€…±±Ì¹…ÁÁ•¹ ¡½¹ÑÉ½±}Í¥°­Ý…ÉÌ¤¤(€€€€€€€É•ÑÕÉ¸ì‰ÑåÁ”ˆè€‰½¹ÑÉ½°¹…¬ˆ°€‰É•ÍÕ±Ðˆèì‰™¥±”ˆè€ˆ½ÑµÀ½¡½ÍÐµÍ…Ù”¹©Í½¸‰õô((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}ÕÍ•Í}½µÁÕÑ•}¡½ÍÐˆ°±…µ‰‘„}Í•ÍÍ¥½¸èQÉÕ”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•¹‘}½µÁÕÑ•}¡½ÍÑ}½¹ÑÉ½°ˆ°Í•¹‘}½¹ÑÉ½°¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹}µ•Ñ¡½‘Íl‰Í•ÍÍ¥½¸¹Í…Ù”‰t ˆÄˆ°ì‰Í•ÍÍ¥½¹}¥ˆèÍ¥‘ô¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡Í¥°9½¹”¤((€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰t€ôôì‰™¥±”ˆè€ˆ½ÑµÀ½¡½ÍÐµÍ…Ù”¹©Í½¸‰ô(€€€…ÍÍ•ÉÐ…±±Ì€ôôl¡Í¥°ì‰É½ÕÑ•}¹…µ”ˆè€‰Í•ÍÍ¥½¸¹Í…Ù”ˆ°€‰Ý…¥ÐˆèQÉÕ•ô¥t(()‘•˜Ñ•ÍÑ}¹½Ñ¥™¥…Ñ¥½¹}•Ù•¹Ñ}‘•‘ÕÁ}­•å}ÁÉ•Í•ÉÙ•Í}‘¥ÍÑ¥¹Ñ}Ý…Ñ¡}µ…Ñ¡•Ì ¤è(€€€€ˆˆ‰]…Ñ µµ…Ñ ¥‘•¹Ñ¥Ñä¥¹±Õ‘•Ìµ…Ñ ½¹Ñ•¹Ð°¹½Ð©ÕÍÐÍ•ÍÍ¥½¸½ÑåÁ”¸ˆˆˆ(€€€‰…Í”€ôì(€€€€€€€€‰ÑåÁ”ˆè€‰Ý…Ñ¡}µ…Ñ ˆ°(€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰ÁÉ½}Ý…Ñ ˆ°(€€€€€€€€‰½µµ…¹ˆè€‰Ñ…¥°€µ˜…ÁÀ¹±½œˆ°(€€€€€€€€‰Á…ÑÑ•É¸ˆè€‰Idˆ°(€€€€€€€€‰½ÕÑÁÕÐˆè€‰Id½¸Á½ÉÐ€àÀÀÀˆ°(€€€€€€€€‰ÍÕÁÁÉ•ÍÍ•ˆè€À°(€€€ô((€€€¥‘•¹Ñ¥…°€ô‘¥Ð¡‰…Í”¤(€€€‘¥ÍÑ¥¹Ñ}½ÕÑÁÕÐ€ôì¨©‰…Í”°€‰½ÕÑÁÕÐˆè€‰Id½¸Á½ÉÐ€äÀÀÀ‰ô(€€€‘¥ÍÑ¥¹Ñ}Á…ÑÑ•É¸€ôì¨©‰…Í”°€‰Á…ÑÑ•É¸ˆè€‰5%IQ%=9}=9‰ô((€€€‰…Í•}­•ä€ôÍ•ÉÙ•È¹}¹½Ñ¥™¥…Ñ¥½¹}•Ù•¹Ñ}‘•‘ÕÁ}­•ä¡‰…Í”¤(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}¹½Ñ¥™¥…Ñ¥½¹}•Ù•¹Ñ}‘•‘ÕÁ}­•ä¡¥‘•¹Ñ¥…°¤€ôô‰…Í•}­•ä(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}¹½Ñ¥™¥…Ñ¥½¹}•Ù•¹Ñ}‘•‘ÕÁ}­•ä¡‘¥ÍÑ¥¹Ñ}½ÕÑÁÕÐ¤€„ô‰…Í•}­•ä(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}¹½Ñ¥™¥…Ñ¥½¹}•Ù•¹Ñ}‘•‘ÕÁ}­•ä¡‘¥ÍÑ¥¹Ñ}Á…ÑÑ•É¸¤€„ô‰…Í•}­•ä(()‘•˜Ñ•ÍÑ}¹½Ñ¥™¥…Ñ¥½¹}Á½±±•É}•µ¥ÑÍ}‘¥ÍÑ¥¹Ñ}Ý…Ñ¡}µ…Ñ¡•Í}½¹”¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰¥ÍÑ¥¹ÐÝ…Ñ µ…Ñ¡•Ì™É½´½¹”ÁÉ½•ÍÌ•µ¥Ðì•á…ÐÉ•Á±…ä¥Ì‘•‘ÕÁ•¸ˆˆˆ(€€€¥µÁ½ÉÐÅÕ•Õ”…Ì}ÅÕ•Õ•}µ½((€€€™É½´Ñ½½±Ì¹ÁÉ½•ÍÍ}É•¥ÍÑÉä¥µÁ½ÉÐÁÉ½•ÍÍ}É•¥ÍÑÉä((€€€ÑÕÉ¹Ì€ômt(€€€•µ¥ÑÑ•€ômt((€€€‘•˜}™…­•}ÉÕ¹}ÁÉ½µÁÑ}ÍÕ‰µ¥Ð¡É¥°Í¥°Í•ÍÍ¥½¸°Ñ•áÐ¤è(€€€€€€€ÑÕÉ¹Ì¹…ÁÁ•¹¡Ñ•áÐ¤(€€€€€€€Ý¥Ñ Í•ÍÍ¥½¹l‰¡¥ÍÑ½Éå}±½¬‰tè(€€€€€€€€€€€Í•ÍÍ¥½¹l‰ÉÕ¹¹¥¹œ‰t€ô…±Í”((€€€Í•ÍÌ€ô}Í•ÍÍ¥½¸ ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‘}Ý…Ñ¡}‘•‘ÕÀ‰t€ôÍ•ÍÌ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„°€¨©­Üè•µ¥ÑÑ•¹…ÁÁ•¹¡„¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÉÕ¹}ÁÉ½µÁÑ}ÍÕ‰µ¥Ðˆ°}™…­•}ÉÕ¹}ÁÉ½µÁÑ}ÍÕ‰µ¥Ð¤((€€€¥Í½±…Ñ•‘}ÅÕ•Õ”è}ÅÕ•Õ•}µ½¹EÕ•Õ”€ô}ÅÕ•Õ•}µ½¹EÕ•Õ” ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡ÁÉ½•ÍÍ}É•¥ÍÑÉä°€‰½µÁ±•Ñ¥½¹}ÅÕ•Õ”ˆ°¥Í½±…Ñ•‘}ÅÕ•Õ”¤((€€€‰…Í”€ôì(€€€€€€€€‰ÑåÁ”ˆè€‰Ý…Ñ¡}µ…Ñ ˆ°(€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰ÁÉ½}Ý…Ñ¡}‘•‘ÕÀˆ°(€€€€€€€€‰½µµ…¹ˆè€‰Ñ…¥°€µ˜…ÁÀ¹±½œˆ°(€€€€€€€€‰Á…ÑÑ•É¸ˆè€‰Idˆ°(€€€€€€€€‰½ÕÑÁÕÐˆè€‰Id½¸Á½ÉÐ€àÀÀÀˆ°(€€€€€€€€‰ÍÕÁÁÉ•ÍÍ•ˆè€À°(€€€ô(€€€¥Í½±…Ñ•‘}ÅÕ•Õ”¹ÁÕÐ¡‰…Í”¤(€€€¥Í½±…Ñ•‘}ÅÕ•Õ”¹ÁÕÐ¡ì¨©‰…Í”°€‰½ÕÑÁÕÐˆè€‰Id½¸Á½ÉÐ€äÀÀÀ‰ô¤(€€€¥Í½±…Ñ•‘}ÅÕ•Õ”¹ÁÕÐ¡‘¥Ð¡‰…Í”¤¤((€€€ÍÑ½À€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤(€€€ÍÑ½À¹Í•Ð ¤((€€€ÑÉäè(€€€€€€€Í•ÉÙ•È¹}¹½Ñ¥™¥…Ñ¥½¹}Á½±±•É}±½½À¡ÍÑ½À°€‰Í¥‘}Ý…Ñ¡}‘•‘ÕÀˆ°Í•ÍÌ¤(€€€€€€€ÍÑ…ÑÕÍ}…±±Ì€ôm„™½È„¥¸•µ¥ÑÑ•¥˜…lÁt€ôô€‰ÍÑ…ÑÕÌ¹ÕÁ‘…Ñ”‰t(€€€€€€€…ÍÍ•ÉÐ±•¸¡ÍÑ…ÑÕÍ}…±±Ì¤€ôô€È(€€€€€€€ÍÑ…ÑÕÍ}Ñ•áÐ€ô€‰q¸ˆ¹©½¥¸¡…±±lÉul‰Ñ•áÐ‰t™½È…±°¥¸ÍÑ…ÑÕÍ}…±±Ì¤(€€€€€€€…ÍÍ•ÉÐ€‰Id½¸Á½ÉÐ€àÀÀÀˆ¥¸ÍÑ…ÑÕÍ}Ñ•áÐ(€€€€€€€…ÍÍ•ÉÐ€‰Id½¸Á½ÉÐ€äÀÀÀˆ¥¸ÍÑ…ÑÕÍ}Ñ•áÐ(€€€€€€€…ÍÍ•ÉÐ±•¸¡ÑÕÉ¹Ì¤€ôô€Ì(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥‘}Ý…Ñ¡}‘•‘ÕÀˆ°9½¹”¤(€€€€€€€Ý¡¥±”¹½ÐÁÉ½•ÍÍ}É•¥ÍÑÉä¹½µÁ±•Ñ¥½¹}ÅÕ•Õ”¹•µÁÑä ¤è(€€€€€€€€€€€ÁÉ½•ÍÍ}É•¥ÍÑÉä¹½µÁ±•Ñ¥½¹}ÅÕ•Õ”¹•Ñ}¹½Ý…¥Ð ¤(()‘•˜Ñ•ÍÑ}¹½Ñ¥™¥…Ñ¥½¹}•Ù•¹Ñ}‘•‘ÕÁ}­•å}­••ÁÍ}½µÁ±•Ñ¥½¹Í}½¹•}Í¡½Ð ¤è(€€€€ˆˆ‰½µÁ±•Ñ¥½¸¥‘•¹Ñ¥ÑäÉ•µ…¥¹ÌÁÉ½•ÍÌµÍ•ÍÍ¥½¸Í½Á•Ñ¼…Ù½¥™±½½‘Ì¸ˆˆˆ(€€€™¥ÉÍÐ€ôì(€€€€€€€€‰ÑåÁ”ˆè€‰½µÁ±•Ñ¥½¸ˆ°(€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰ÁÉ½}‘½¹”ˆ°(€€€€€€€€‰½µµ…¹ˆè€‰µ…­”‰Õ¥±ˆ°(€€€€€€€€‰•á¥Ñ}½‘”ˆè€À°(€€€€€€€€‰½ÕÑÁÕÐˆè€‰™¥ÉÍÐ½ÕÑÁÕÐˆ°(€€€ô(€€€É•Á±…ä€ôì(€€€€€€€€‰ÑåÁ”ˆè€‰½µÁ±•Ñ¥½¸ˆ°(€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰ÁÉ½}‘½¹”ˆ°(€€€€€€€€‰½µµ…¹ˆè€‰µ…­”‰Õ¥±€´µ……¥¸ˆ°(€€€€€€€€‰•á¥Ñ}½‘”ˆè€Ä°(€€€€€€€€‰½ÕÑÁÕÐˆè€‰‘¥™™•É•¹Ð½ÕÑÁÕÐÍ¡½Õ±¹½Ð¡…¹”½µÁ±•Ñ¥½¸­•äˆ°(€€€ô((€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}¹½Ñ¥™¥…Ñ¥½¹}•Ù•¹Ñ}‘•‘ÕÁ}­•ä¡™¥ÉÍÐ¤€ôôÍ•ÉÙ•È¹}¹½Ñ¥™¥…Ñ¥½¹}•Ù•¹Ñ}‘•‘ÕÁ}­•ä (€€€€€€€É•Á±…ä(€€€€¤(((Œ€´´´¥µ…”¹…ÑÑ…¡}‰åÑ•Ì€¼Á‘˜¹…ÑÑ… €¡É•µ½Ñ”µ±¥•¹Ð‰åÑ”ÕÁ±½…¤€´´´´´´´´´´´´´((ŒMµ…±±•ÍÐÙ…±¥€ÅàÄA9°‰…Í”ØÐµ•¹½‘•¸)}A9|Å`Å}ØÐ€ô€ (€€€€‰¥Y	=IÜÁ-½9MU¡U	EÅ!ÝÁ±EYHÐÉµ9¬ˆ(€€€€ˆ­4åEÝ¡]©Hå…Ý	)ITÕÉ­)œôôˆ(¤(()‘•˜}…ÑÑ…¡}‰åÑ•Í}±¤¡µ½¹­•åÁ…Ñ ¤è(€€€™…­•}±¤€ôÑåÁ•Ì¹5½‘Õ±•QåÁ” ‰±¤ˆ¤(€€€™…­•}±¤¹}%5}aQ9M%=9L€ôìˆ¹Á¹œˆ°€ˆ¹©Áœˆ°€ˆ¹©Á•œˆ°€ˆ¹¥˜ˆ°€ˆ¹Ý•‰Àˆ°€ˆ¹‰µÀ‰ô(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ¥Ñ•´¡ÍåÌ¹µ½‘Õ±•Ì°€‰±¤ˆ°™…­•}±¤¤(()‘•˜Ñ•ÍÑ}¥µ…•}…ÑÑ…¡}‰åÑ•Í}ÝÉ¥Ñ•Í}Ñ½}…Ñ•Ý…å}‘¥È¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰I•µ½Ñ”±¥•¹ÐÕÁ±½…‘Ì‰…Í”ØÐ‰åÑ•Ìì…Ñ•Ý…äÝÉ¥Ñ•ÌÑ¡•´Ñ¼¥ÑÌ½Ý¸‘¥Í¬¸ˆˆˆ(€€€}…ÑÑ…¡}‰åÑ•Í}±¤¡µ½¹­•åÁ…Ñ ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}¡•Éµ•Í}¡½µ”ˆ°ÑµÁ}Á…Ñ ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰…‰à‰t€ô}Í•ÍÍ¥½¸ ¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰¥µ…”¹…ÑÑ…¡}‰åÑ•Ìˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰…‰àˆ°(€€€€€€€€€€€€€€€€‰½¹Ñ•¹Ñ}‰…Í”ØÐˆè}A9|Å`Å}ØÐ°(€€€€€€€€€€€€€€€€‰™¥±•¹…µ”ˆè€‰Í¡½Ð¹Á¹œˆ°(€€€€€€€€€€€ô°(€€€€€€€ô(€€€€¤((€€€É•Ì€ôÉ•ÍÁl‰É•ÍÕ±Ð‰t(€€€…ÍÍ•ÉÐÉ•Íl‰…ÑÑ…¡•‰t¥ÌQÉÕ”(€€€ÝÉ¥ÑÑ•¸€ôA…Ñ ¡É•Íl‰Á…Ñ ‰t¤(€€€…ÍÍ•ÉÐÝÉ¥ÑÑ•¸¹¥Í}™¥±” ¤(€€€…ÍÍ•ÉÐÝÉ¥ÑÑ•¸¹Á…É•¹Ð€ôôÑµÁ}Á…Ñ €¼€‰¥µ…•Ìˆ(€€€…ÍÍ•ÉÐÝÉ¥ÑÑ•¸¹É•…‘}‰åÑ•Ì ¤¹ÍÑ…ÉÑÍÝ¥Ñ ¡ˆ‰qààåA9ˆ¤(€€€…ÍÍ•ÉÐ±•¸¡Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰…‰à‰ul‰…ÑÑ…¡•‘}¥µ…•Ì‰t¤€ôô€Ä(€€€…ÍÍ•ÉÐÉ•Íl‰‰åÑ•Ì‰t€ø€À(()‘•˜Ñ•ÍÑ}¥µ…•}…ÑÑ…¡}‰åÑ•Í}…•ÁÑÍ}‘…Ñ…}ÕÉ±}ÁÉ•™¥à¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€}…ÑÑ…¡}‰åÑ•Í}±¤¡µ½¹­•åÁ…Ñ ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}¡•Éµ•Í}¡½µ”ˆ°ÑµÁ}Á…Ñ ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰…‰àÈ‰t€ô}Í•ÍÍ¥½¸ ¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰¥µ…”¹…ÑÑ…¡}‰åÑ•Ìˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰…‰àÈˆ°(€€€€€€€€€€€€€€€€‰½¹Ñ•¹Ñ}‰…Í”ØÐˆè˜‰‘…Ñ„é¥µ…”½Á¹œí‰…Í”ØÐ±í}A9|Å`Å}ØÑôˆ°(€€€€€€€€€€€ô°(€€€€€€€ô(€€€€¤(€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰…ÑÑ…¡•‰t¥ÌQÉÕ”(()‘•˜Ñ•ÍÑ}¥µ…•}…ÑÑ…¡}‰åÑ•Í}‘…Ñ…}…±¥…Í}…¹‘}µ…¥}Í¹¥™˜¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰=±‘•È‘•Í­Ñ½À‰Õ¥±‘ÌÍ•¹‘…Ñ…€€¡¹½Ð½¹Ñ•¹Ñ}‰…Í”ØÐ¤ì•áÐÍ¹¥™™•™É½´‰åÑ•Ì¸ˆˆˆ(€€€}…ÑÑ…¡}‰åÑ•Í}±¤¡µ½¹­•åÁ…Ñ ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}¡•Éµ•Í}¡½µ”ˆ°ÑµÁ}Á…Ñ ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰…‰àÌ‰t€ô}Í•ÍÍ¥½¸ ¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰¥µ…”¹…ÑÑ…¡}‰åÑ•Ìˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰…‰àÌˆ°€‰‘…Ñ„ˆè}A9|Å`Å}ØÑô°(€€€€€€€ô(€€€€¤(€€€É•Ì€ôÉ•ÍÁl‰É•ÍÕ±Ð‰t(€€€…ÍÍ•ÉÐÉ•Íl‰…ÑÑ…¡•‰t¥ÌQÉÕ”(€€€…ÍÍ•ÉÐA…Ñ ¡É•Íl‰Á…Ñ ‰t¤¹ÍÕ™™¥à€ôô€ˆ¹Á¹œˆ€€ŒÍ¹¥™™•™É½´µ…¥Œ‰åÑ•Ì(()‘•˜Ñ•ÍÑ}¥µ…•}…ÑÑ…¡}‰åÑ•Í}É•©•ÑÍ}¥¹Ù…±¥‘}‰…Í”ØÐ¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€}…ÑÑ…¡}‰åÑ•Í}±¤¡µ½¹­•åÁ…Ñ ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}¡•Éµ•Í}¡½µ”ˆ°ÑµÁ}Á…Ñ ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰…‰àÐ‰t€ô}Í•ÍÍ¥½¸ ¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰¥µ…”¹…ÑÑ…¡}‰åÑ•Ìˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰…‰àÐˆ°€‰½¹Ñ•¹Ñ}‰…Í”ØÐˆè€ˆ„„…¹½Ð‰…Í”ØÐ„„„‰ô°(€€€€€€€ô(€€€€¤(€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¥¸É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÄÜ(()‘•˜Ñ•ÍÑ}¥µ…•}…ÑÑ…¡}‰åÑ•Í}É•©•ÑÍ}½Ù•ÉÍ¥é”¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€¥µÁ½ÉÐ‰…Í”ØÐ…Ì}ˆØÐ((€€€}…ÑÑ…¡}‰åÑ•Í}±¤¡µ½¹­•åÁ…Ñ ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}¡•Éµ•Í}¡½µ”ˆ°ÑµÁ}Á…Ñ ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}QQ!}	eQM}5a}	eQLˆ°€ÄÀ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰…‰àÔ‰t€ô}Í•ÍÍ¥½¸ ¤((€€€‰¥œ€ô}ˆØÐ¹ˆØÑ•¹½‘”¡ˆ‰qààåA9qÉq¹qàÅ…q¸ˆ€¬ˆˆÀˆ€¨€ÄÀÀ¤¹‘•½‘” ‰…Í¥¤ˆ¤(€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰¥µ…”¹…ÑÑ…¡}‰åÑ•Ìˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰…‰àÔˆ°€‰½¹Ñ•¹Ñ}‰…Í”ØÐˆè‰¥ô°(€€€€€€€ô(€€€€¤(€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¥¸É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÄà(()‘•˜Ñ•ÍÑ}¥µ…•}…ÑÑ…¡}‰åÑ•Í}É•©•ÑÍ}Õ¹ÍÕÁÁ½ÉÑ•‘}•áÑ•¹Í¥½¸¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€}…ÑÑ…¡}‰åÑ•Í}±¤¡µ½¹­•åÁ…Ñ ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}¡•Éµ•Í}¡½µ”ˆ°ÑµÁ}Á…Ñ ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰…‰àØ‰t€ô}Í•ÍÍ¥½¸ ¤((€€€€Œ™¥±•¹…µ”¡¥¹Ð™½É•Ì„¹½¸µ¥µ…”•áÑ•¹Í¥½¸ìµ…¥ŒÍ¹¥™˜¥Ì‰åÁ…ÍÍ•‰ä¡¥¹Ð(€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰¥µ…”¹…ÑÑ…¡}‰åÑ•Ìˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰…‰àØˆ°(€€€€€€€€€€€€€€€€‰½¹Ñ•¹Ñ}‰…Í”ØÐˆè}A9|Å`Å}ØÐ°(€€€€€€€€€€€€€€€€‰™¥±•¹…µ”ˆè€‰•Ù¥°¹•á”ˆ°(€€€€€€€€€€€ô°(€€€€€€€ô(€€€€¤(€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¥¸É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÄØ(()‘•˜Ñ•ÍÑ}Á‘™}…ÑÑ…¡}É•ÅÕ¥É•Í}Á½ÁÁ±•È¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰]¥Ñ¡½ÕÐÁ‘™Ñ½ÁÁ´½¸AQ °Á‘˜¹…ÑÑ… É•ÑÕÉ¹Ì„±•…È€ÔÀÈà¸ˆˆˆ(€€€}…ÑÑ…¡}‰åÑ•Í}±¤¡µ½¹­•åÁ…Ñ ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}¡•Éµ•Í}¡½µ”ˆ°ÑµÁ}Á…Ñ ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰Í¡ÕÑ¥°¹Ý¡¥ ˆ°±…µ‰‘„}¹…µ”è9½¹”¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Á‘˜Ä‰t€ô}Í•ÍÍ¥½¸ ¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Á‘˜¹…ÑÑ… ˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Á‘˜Äˆ°€‰½¹Ñ•¹Ñ}‰…Í”ØÐˆè€‰)Y	I¤Áá1©E,‰ô°(€€€€€€€ô(€€€€¤(€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¥¸É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÔÀÈà(()‘•˜Ñ•ÍÑ}Á‘™}…ÑÑ…¡}É•©•ÑÍ}¹½¹}Á‘™}‰åÑ•Ì¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€¥µÁ½ÉÐ‰…Í”ØÐ…Ì}ˆØÐ((€€€}…ÑÑ…¡}‰åÑ•Í}±¤¡µ½¹­•åÁ…Ñ ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}¡•Éµ•Í}¡½µ”ˆ°ÑµÁ}Á…Ñ ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰Í¡ÕÑ¥°¹Ý¡¥ ˆ°±…µ‰‘„}¹…µ”è€ˆ½ÕÍÈ½‰¥¸½Á‘™Ñ½ÁÁ´ˆ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Á‘˜È‰t€ô}Í•ÍÍ¥½¸ ¤((€€€¹½Ñ}Á‘˜€ô}ˆØÐ¹ˆØÑ•¹½‘”¡ˆ‰Ñ¡¥Ì¥Ì¹½Ð„Á‘˜ˆ¤¹‘•½‘” ‰…Í¥¤ˆ¤(€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì(€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Á‘˜¹…ÑÑ… ˆ°(€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Á‘˜Èˆ°€‰½¹Ñ•¹Ñ}‰…Í”ØÐˆè¹½Ñ}Á‘™ô°(€€€€€€€ô(€€€€¤(€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¥¸É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÄÜ(()‘•˜Ñ•ÍÑ}Á‘™}…ÑÑ…¡}É•ÅÕ¥É•Í}Á…Ñ¡}½É}‰åÑ•Ì¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€}…ÑÑ…¡}‰åÑ•Í}±¤¡µ½¹­•åÁ…Ñ ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}¡•Éµ•Í}¡½µ”ˆ°ÑµÁ}Á…Ñ ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰Í¡ÕÑ¥°¹Ý¡¥ ˆ°±…µ‰‘„}¹…µ”è€ˆ½ÕÍÈ½‰¥¸½Á‘™Ñ½ÁÁ´ˆ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Á‘˜Ì‰t€ô}Í•ÍÍ¥½¸ ¤((€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Á‘˜¹…ÑÑ… ˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Á‘˜Ì‰õô(€€€€¤(€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¥¸É•ÍÀ(€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÄÔ(()‘•˜Ñ•ÍÑ}‘•½‘•}…ÑÑ…¡}‰…Í”ØÑ}¡•±Á•È ¤è(€€€¥µÁ½ÉÐ‰…Í”ØÐ…Ì}ˆØÐ((€€€É…Ü€ô}ˆØÐ¹ˆØÑ•¹½‘”¡ˆ‰¡•±±¼ˆ¤¹‘•½‘” ‰…Í¥¤ˆ¤(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}‘•½‘•}…ÑÑ…¡}‰…Í”ØÐ¡É…Ü°µ¥µ•}ÁÉ•™¥àô‰¥µ…”¼ˆ¤€ôôˆ‰¡•±±¼ˆ(€€€…ÍÍ•ÉÐ€ (€€€€€€€Í•ÉÙ•È¹}‘•½‘•}…ÑÑ…¡}‰…Í”ØÐ¡˜‰‘…Ñ„é¥µ…”½Á¹œí‰…Í”ØÐ±íÉ…Ýôˆ°µ¥µ•}ÁÉ•™¥àô‰¥µ…”¼ˆ¤(€€€€€€€€ôôˆ‰¡•±±¼ˆ(€€€€¤(€€€€ŒÝ¡¥Ñ•ÍÁ…”¥¹Í¥‘”Á…å±½…¥ÌÑ½±•É…Ñ•(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}‘•½‘•}…ÑÑ…¡}‰…Í”ØÐ¡É…ÝlèÑt€¬€‰q¸ˆ€¬É…ÝlÐét°µ¥µ•}ÁÉ•™¥àô‰¥µ…”¼ˆ¤€ôôˆ‰¡•±±¼ˆ(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}‘•½‘•}…ÑÑ…¡}‰…Í”ØÐ ‰ ˆ°µ¥µ•}ÁÉ•™¥àô‰¥µ…”¼ˆ¤¥Ì9½¹”(()‘•˜Ñ•ÍÑ}Í¹¥™™}¥µ…•}•áÑ}µ…¥}…¹‘}™¥±•¹…µ” ¤è(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í¹¥™™}¥µ…•}•áÐ¡ˆ‰qààåA9qÉq¹qàÅ…q¸ˆ¤€ôô€ˆ¹Á¹œˆ(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í¹¥™™}¥µ…•}•áÐ¡ˆ‰qá™™qááqá™™qá”Àˆ¤€ôô€ˆ¹©Áœˆ(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í¹¥™™}¥µ…•}•áÐ¡ˆ‰%àå„¸¸¸¸ˆ¤€ôô€ˆ¹¥˜ˆ(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í¹¥™™}¥µ…•}•áÐ¡ˆ‰I%ÄÈÌÑ]	Aáááàˆ¤€ôô€ˆ¹Ý•‰Àˆ(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í¹¥™™}¥µ…•}•áÐ¡ˆ‰	4¸¸¸¸¸¸ˆ¤€ôô€ˆ¹‰µÀˆ(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í¹¥™™}¥µ…•}•áÐ¡ˆ‰Õ¹­¹½Ý¸ˆ¤€ôô€ˆ¹Á¹œˆ€€Œ™…±±‰…¬(€€€€Œ™¥±•¹…µ”¡¥¹ÐÝ¥¹Ì½Ù•Èµ…¥Œ‰åÑ•Ì(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í¹¥™™}¥µ…•}•áÐ¡ˆ‰qààåA9ˆ°€‰Á¡½Ñ¼¹©Á•œˆ¤€ôô€ˆ¹©Á•œˆ(()‘•˜Ñ•ÍÑ}Í±…Í¡}Ý½É­•É}±½Í•}É•…ÁÍ}é½µ‰¥•}…¹‘}±½Í•Í}™‘Ì ¤è(€€€€ˆˆ‰¡Õ¹œÝ½É­•È¥ÌM%-%11•°Ñ¡”é½µ‰¥”É•…Á•°…±°Á¥Á•Ì±½Í•ƒŠP½¹”¸ˆˆˆ(€€€…±±Ì€ôí¬è€À™½È¬¥¸€ ‰Ñ•Éµ¥¹…Ñ”ˆ°€‰­¥±°ˆ°€‰Ý…¥Ðˆ°€‰ÍÑ‘¥¸ˆ°€‰ÍÑ‘½ÕÐˆ°€‰ÍÑ‘•ÉÈˆ¥ô((€€€±…ÍÌ…­•MÑÉ•…´è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°¹…µ”¤è(€€€€€€€€€€€Í•±˜¹¹…µ”€ô¹…µ”((€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€…±±ÍmÍ•±˜¹¹…µ•t€¬ô€Ä((€€€±…ÍÌ…­•AÉ½Œè(€€€€€€€ÍÑ‘¥¸°ÍÑ‘½ÕÐ°ÍÑ‘•ÉÈ€ô€¡…­•MÑÉ•…´¡¸¤™½È¸¥¸€ ‰ÍÑ‘¥¸ˆ°€‰ÍÑ‘½ÕÐˆ°€‰ÍÑ‘•ÉÈˆ¤¤((€€€€€€€‘•˜Á½±°¡Í•±˜¤è(€€€€€€€€€€€É•ÑÕÉ¸9½¹”€€Œ…±Ý…åÌ…±¥Ù”€´ø™½É•ÌÑ•Éµ¥¹…Ñ”Ñ¡•¸­¥±°((€€€€€€€‘•˜Ñ•Éµ¥¹…Ñ”¡Í•±˜¤è(€€€€€€€€€€€…±±Íl‰Ñ•Éµ¥¹…Ñ”‰t€¬ô€Ä((€€€€€€€‘•˜­¥±°¡Í•±˜¤è(€€€€€€€€€€€…±±Íl‰­¥±°‰t€¬ô€Ä((€€€€€€€‘•˜Ý…¥Ð¡Í•±˜°Ñ¥µ•½ÕÐõ9½¹”¤è(€€€€€€€€€€€…±±Íl‰Ý…¥Ð‰t€¬ô€Ä(€€€€€€€€€€€É…¥Í”ÍÕ‰ÁÉ½•ÍÌ¹Q¥µ•½ÕÑáÁ¥É•¡µô‰àˆ°Ñ¥µ•½ÕÐõÑ¥µ•½ÕÐ¤((€€€Ý½É­•È€ô½‰©•Ð¹}}¹•Ý}|¡Í•ÉÙ•È¹}M±…Í¡]½É­•È¤(€€€Ý½É­•È¹ÁÉ½Œ€ô…­•AÉ½Œ ¤((€€€Ý½É­•È¹±½Í” ¤(€€€Ý½É­•È¹±½Í” ¤€€Œ¥‘•µÁ½Ñ•¹Ð((€€€…ÍÍ•ÉÐ…±±Íl‰Ñ•Éµ¥¹…Ñ”‰t€ôô€Ä(€€€…ÍÍ•ÉÐ…±±Íl‰­¥±°‰t€ôô€Ä(€€€…ÍÍ•ÉÐ…±±Íl‰Ý…¥Ð‰t€øô€È€€ŒÉ•…Á•…™Ñ•È‰½Ñ Ñ•Éµ¥¹…Ñ”…¹­¥±°(€€€…ÍÍ•ÉÐ…±±Íl‰ÍÑ‘¥¸‰t€ôô…±±Íl‰ÍÑ‘½ÕÐ‰t€ôô…±±Íl‰ÍÑ‘•ÉÈ‰t€ôô€Ä(()‘•˜Ñ•ÍÑ}±½Í•}Í•ÍÍ¥½¹}‰å}¥‘}¥Í}¥‘•µÁ½Ñ•¹Ñ}…¹‘}™Õ±°¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰=¹”…±°Ñ•…ÉÌÑ¡”Í•ÍÍ¥½¸‘½Ý¸™Õ±±äì„Í•½¹¥Ì„¹¼µ½À¸ˆˆˆ(€€€…±±Ì€ôì‰Ý½É­•Èˆè€À°€‰…•¹Ðˆè€À°€‰Õ¹É•œˆè€À°€‰™¥¹…±¥é”ˆè€Áô((€€€±…ÍÌ\è(€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€…±±Íl‰Ý½É­•È‰t€¬ô€Ä((€€€±…ÍÌè(€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€…±±Íl‰…•¹Ð‰t€¬ô€Ä((€€€‘•˜}™…­•}™¥¹…±¥é”¡Ì°•¹‘}É•…Í½¸ô‰ÑÕ¥}±½Í”ˆ¤è(€€€€€€€€ŒI•…°}™¥¹…±¥é•}Í•ÍÍ¥½¸¥ÌÑ¡”Í¥¹±”¡½­•Á½¥¹ÐÑ¡…Ð±½Í•ÌÑ¡”(€€€€€€€€ŒÍ±…Í µÝ½É­•Èìµ¥ÉÉ½ÈÑ¡…Ð¡•É”Í¼Ñ¡”Ñ•ÍÐ•á•É¥Í•ÌÑ¡”…ÑÕ…°(€€€€€€€€ŒÑ•…É‘½Ý¸½¹ÑÉ…Ð€¡Ý½É­•È±½Í”±¥Ù•Ì¥¸™¥¹…±¥é”°¹½ÐÑ¡”…±±•È¤¸(€€€€€€€…±±Íl‰™¥¹…±¥é”‰t€¬ô€Ä(€€€€€€€Ü€ôÌ¹•Ð ‰Í±…Í¡}Ý½É­•Èˆ¤(€€€€€€€¥˜Üè(€€€€€€€€€€€Ü¹±½Í” ¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}™¥¹…±¥é•}Í•ÍÍ¥½¸ˆ°}™…­•}™¥¹…±¥é”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€‰Ñ½½±Ì¹…ÁÁÉ½Ù…°¹Õ¹É•¥ÍÑ•É}…Ñ•Ý…å}¹½Ñ¥™äˆ°(€€€€€€€±…µ‰‘„­•äè…±±Ì¹}}Í•Ñ¥Ñ•µ}| ‰Õ¹É•œˆ°…±±Íl‰Õ¹É•œ‰t€¬€Ä¤°É…¥Í¥¹œõ…±Í”°(€€€€¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥´Ä‰t€ôì‰Í•ÍÍ¥½¹}­•äˆè€‰¬Äˆ°€‰…•¹Ðˆè ¤°€‰Í±…Í¡}Ý½É­•Èˆè\ ¥ô((€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}±½Í•}Í•ÍÍ¥½¹}‰å}¥ ‰Í¥´Äˆ°•¹‘}É•…Í½¸ô‰ÝÍ}‘¥Í½¹¹•Ðˆ¤¥ÌQÉÕ”(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}±½Í•}Í•ÍÍ¥½¹}‰å}¥ ‰Í¥´Äˆ°•¹‘}É•…Í½¸ô‰ÝÍ}‘¥Í½¹¹•Ðˆ¤¥Ì…±Í”(€€€…ÍÍ•ÉÐ…±±Ì€ôôì‰Ý½É­•Èˆè€Ä°€‰…•¹Ðˆè€Ä°€‰Õ¹É•œˆè€Ä°€‰™¥¹…±¥é”ˆè€Åô(€€€…ÍÍ•ÉÐ€‰Í¥´Äˆ¹½Ð¥¸Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì(()‘•˜Ñ•ÍÑ}…ÑÑ…¡}Ý½É­•É}±½Í•Í}½ÉÁ¡…¹}Ý¡•¹}Í•ÍÍ¥½¹}…±É•…‘å}Ñ½É¹}‘½Ý¸ ¤è(€€€€ˆˆ‰Ý½É­•È‰Õ¥±Ð…™Ñ•È¥ÑÌÍ•ÍÍ¥½¸Ý…ÌÉ•…Á•µÕÍÐ‰”±½Í•°¹½Ð½ÉÁ¡…¹•¸ˆˆˆ(€€€±½Í•€ômt((€€€±…ÍÌ\è(€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€±½Í•¹…ÁÁ•¹¡QÉÕ”¤((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰½¹”ˆ°9½¹”¤(€€€‘•Ñ…¡•€ôì‰Í•ÍÍ¥½¹}­•äˆè€‰¬‰ô€€Œ¹½Ð¥¸}Í•ÍÍ¥½¹Ì€´ø…±É•…‘äÑ½É¸‘½Ý¸(€€€Í•ÉÙ•È¹}…ÑÑ…¡}Ý½É­•È ‰½¹”ˆ°‘•Ñ…¡•°\ ¤¤((€€€…ÍÍ•ÉÐ±½Í•€ôômQÉÕ•t(€€€…ÍÍ•ÉÐ€‰Í±…Í¡}Ý½É­•Èˆ¹½Ð¥¸‘•Ñ…¡•(€€€…ÍÍ•ÉÐ€‰½¹”ˆ¹½Ð¥¸Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì(()‘•˜Ñ•ÍÑ}…ÑÑ…¡}Ý½É­•É}ÍÑ½É•Í}Ý½É­•É}½¹}±¥Ù•}Í•ÍÍ¥½¸ ¤è(€€€±…ÍÌ\è(€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€É…¥Í”ÍÍ•ÉÑ¥½¹ÉÉ½È ‰µÕÍÐ¹½Ð±½Í”„Ý½É­•È™½È„±¥Ù”Í•ÍÍ¥½¸ˆ¤((€€€±¥Ù”€ôì‰Í•ÍÍ¥½¹}­•äˆè€‰¬‰ô(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰±¥Ù”‰t€ô±¥Ù”(€€€Ý½É­•È€ô\ ¤(€€€ÑÉäè(€€€€€€€Í•ÉÙ•È¹}…ÑÑ…¡}Ý½É­•È ‰±¥Ù”ˆ°±¥Ù”°Ý½É­•È¤(€€€€€€€…ÍÍ•ÉÐ±¥Ù•l‰Í±…Í¡}Ý½É­•È‰t¥ÌÝ½É­•È(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰±¥Ù”ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}É•ÍÑ…ÉÑ}Í±…Í¡}Ý½É­•É}±½Í•Í}½ÉÁ¡…¹}Ý¡•¹}Í•ÍÍ¥½¹}É•…Á•¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰A½ÍÐµÑÕÉ¸É•ÍÑ…ÉÐ½˜„Í•ÍÍ¥½¸É•…Á•µ¥µ™±¥¡Ð€¡”¹œ¸±½Í•}½¹}‘¥Í½¹¹•Ð(€€€™¥É•Ý¡¥±”ÉÕ¹¹¥¹€™±¥ÁÁ•™…±Í”¤µÕÍÐ±½Í”‰½Ñ Ñ¡”ÍÑ…±”Ý½É­•È…¹(€€€Ñ¡”™É•Í É•Á±…•µ•¹Ð°¹½Ð½ÉÁ¡…¸•¥Ñ¡•È¸ˆˆˆ(€€€±½Í•€ômt((€€€±…ÍÌ}…­•]½É­•Èè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°€©„°€¨©¬¤è(€€€€€€€€€€€Á…ÍÌ((€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€±½Í•¹…ÁÁ•¹¡QÉÕ”¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}M±…Í¡]½É­•Èˆ°}…­•]½É­•È¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰É•…Á•ˆ°9½¹”¤(€€€€Œ¹½Ð¥¸}Í•ÍÍ¥½¹Ì€´øÑ½É¸‘½Ý¸½¹ÕÉÉ•¹Ñ±äì…ÉÉ¥•Ì„±¥Ù”Ý½É­•ÈÍ¼Ñ¡”(€€€€ŒÉ•ÍÑ…ÉÐÁ…Ñ …ÑÕ…±±äÉÕ¹Ì€¡„Ý½É­•É±•ÍÌÍ•ÍÍ¥½¸¥Ì„É•ÍÑ…ÉÐ¹¼µ½À¹½Ü¤(€€€É•…Á•€ôì‰Í•ÍÍ¥½¹}­•äˆè€‰¬ˆ°€‰Í±…Í¡}Ý½É­•Èˆè}…­•]½É­•È ¥ô(€€€Í•ÉÙ•È¹}É•ÍÑ…ÉÑ}Í±…Í¡}Ý½É­•È ‰É•…Á•ˆ°É•…Á•¤((€€€€ŒÍÑ…±”Ý½É­•È±½Í•‰äÑ¡”É•ÍÑ…ÉÐ°™É•Í Ý½É­•È±½Í•‰ä}…ÑÑ…¡}Ý½É­•È(€€€€Œ€¡Í¥¹¼±½¹•Èµ…ÁÌÑ¼Ñ¡¥ÌÍ•ÍÍ¥½¸¤(€€€…ÍÍ•ÉÐ±½Í•€ôômQÉÕ”°QÉÕ•t(€€€…ÍÍ•ÉÐ€‰É•…Á•ˆ¹½Ð¥¸Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì(()‘•˜Ñ•ÍÑ}É•ÍÑ…ÉÑ}Í±…Í¡}Ý½É­•É}ÍÑ½É•Í}½¹}±¥Ù•}Í•ÍÍ¥½¸¡µ½¹­•åÁ…Ñ ¤è(€€€±…ÍÌ}…­•]½É­•Èè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°€©„°€¨©¬¤è(€€€€€€€€€€€Á…ÍÌ((€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€Á…ÍÌ((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}M±…Í¡]½É­•Èˆ°}…­•]½É­•È¤(€€€½±‘}Ý½É­•È€ô}…­•]½É­•È ¤(€€€±¥Ù”€ôì‰Í•ÍÍ¥½¹}­•äˆè€‰¬ˆ°€‰Í±…Í¡}Ý½É­•Èˆè½±‘}Ý½É­•Éô(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰±¥Ù”µÉ•ÍÑ…ÉÐ‰t€ô±¥Ù”(€€€ÑÉäè(€€€€€€€Í•ÉÙ•È¹}É•ÍÑ…ÉÑ}Í±…Í¡}Ý½É­•È ‰±¥Ù”µÉ•ÍÑ…ÉÐˆ°±¥Ù”¤(€€€€€€€…ÍÍ•ÉÐ¥Í¥¹ÍÑ…¹”¡±¥Ù•l‰Í±…Í¡}Ý½É­•È‰t°}…­•]½É­•È¤(€€€€€€€…ÍÍ•ÉÐ±¥Ù•l‰Í±…Í¡}Ý½É­•È‰t¥Ì¹½Ð½±‘}Ý½É­•È(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰±¥Ù”µÉ•ÍÑ…ÉÐˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}É•ÍÑ…ÉÑ}Í±…Í¡}Ý½É­•É}¹½½Á}Ý¥Ñ¡½ÕÑ}Ý½É­•È¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰Í•ÍÍ¥½¸Ñ¡…Ð¹•Ù•ÈÍÁ…Ý¹•„Ý½É­•È€¡Í±…Í ¹•á•Œ¹½ÐÕÍ•å•Ð¤µÕÍÐ(€€€ÍÑ…äÝ½É­•É±•ÍÌ…É½ÍÌ„É•ÍÑ…ÉÐƒŠPÍÁ…Ý¹¥¹œ¡•É”Ý½Õ±™½É¬Ñ¡”Á•ÈµÝ½É­•È(€€€ÍÑ‘¥¼5@™±••Ð™½ÈÍ•ÍÍ¥½¹ÌÑ¡…Ð¹•Ù•ÈÉÕ¸Ý½É­•ÈµÉ½ÕÑ•½µµ…¹‘Ì¸ˆˆˆ(€€€ÍÁ…Ý¹•€ômt((€€€±…ÍÌ}…­•]½É­•Èè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°€©„°€¨©¬¤è(€€€€€€€€€€€ÍÁ…Ý¹•¹…ÁÁ•¹¡QÉÕ”¤((€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€Á…ÍÌ((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}M±…Í¡]½É­•Èˆ°}…­•]½É­•È¤(€€€±¥Ù”€ôì‰Í•ÍÍ¥½¹}­•äˆè€‰¬ˆ°€‰Í±…Í¡}Ý½É­•Èˆè9½¹•ô(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰±…éäµ¹½½À‰t€ô±¥Ù”(€€€ÑÉäè(€€€€€€€Í•ÉÙ•È¹}É•ÍÑ…ÉÑ}Í±…Í¡}Ý½É­•È ‰±…éäµ¹½½Àˆ°±¥Ù”¤(€€€€€€€…ÍÍ•ÉÐÍÁ…Ý¹•€ôômt(€€€€€€€…ÍÍ•ÉÐ±¥Ù•l‰Í±…Í¡}Ý½É­•È‰t¥Ì9½¹”(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰±…éäµ¹½½Àˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}Í±…Í¡}•á•}½¹ÕÉÉ•¹Ñ}™¥ÉÍÑ}ÕÍ•}ÍÁ…Ý¹Í}Í¥¹±•}Ý½É­•È¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰]¥Ñ •…•ÈÁÉ”µÝ…É´É•µ½Ù•°Í±…Í ¹•á•Œ¥ÌÑ¡”½¹±äÍÁ…Ý¸Á…Ñ ƒŠPÑÝ¼(€€€½¹ÕÉÉ•¹ÐÝ½É­•ÈµÉ½ÕÑ•½µµ…¹‘Ì½¸„™É•Í Í•ÍÍ¥½¸µÕÍÐ¹½Ð•… ™½É¬„(€€€™Õ±°5@µ™±••ÐÝ½É­•È¸Q¡”Á•ÈµÍ•ÍÍ¥½¸ÍÁ…Ý¸±½¬Í•É¥…±¥é•Ì™¥ÉÍÐÕÍ”¸ˆˆˆ(€€€¥µÁ½ÉÐÑ¥µ”…Ì}Ñ¥µ”((€€€ÍÁ…Ý¹•€ômt(€€€‰…ÉÉ¥•È€ôÑ¡É•…‘¥¹œ¹	…ÉÉ¥•È È°Ñ¥µ•½ÕÐôÔ¤((€€€±…ÍÌ}M±½Ý]½É­•Èè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°€©„°€¨©¬¤è(€€€€€€€€€€€ÍÁ…Ý¹•¹…ÁÁ•¹¡Í•±˜¤(€€€€€€€€€€€}Ñ¥µ”¹Í±••À À¸ÀÔ¤€€ŒÝ¥‘•¸Ñ¡”9½¹”µ½‰Í•ÉÙ…Ñ¥½¸Ý¥¹‘½Ü((€€€€€€€‘•˜ÉÕ¸¡Í•±˜°µ¤è(€€€€€€€€€€€É•ÑÕÉ¸˜‰É…¸íµ‘ôˆ((€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€Á…ÍÌ((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}M±…Í¡]½É­•Èˆ°}M±½Ý]½É­•È¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}µ¥ÉÉ½É}Í±…Í¡}Í¥‘•}•™™•ÑÌˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸¡Í±…Í¡}Ý½É­•Èõ9½¹”¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰É…”µÍÁ…Ý¸‰t€ôÍ•ÍÍ¥½¸((€€€É•ÍÕ±ÑÌ€ômt((€€€‘•˜}•á•Œ¡¸¤è(€€€€€€€‰…ÉÉ¥•È¹Ý…¥Ð ¤(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆèÍÑÈ¡¸¤°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰Í±…Í ¹•á•Œˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰½µµ…¹ˆè€ˆ½½¹Ñ•áÐˆ°€‰Í•ÍÍ¥½¹}¥ˆè€‰É…”µÍÁ…Ý¸‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€É•ÍÕ±ÑÌ¹…ÁÁ•¹¡É•ÍÀ¤((€€€ÑÉäè(€€€€€€€Ñ¡É•…‘Ì€ômÑ¡É•…‘¥¹œ¹Q¡É•…¡Ñ…É•Ðõ}•á•Œ°…ÉÌô¡¤°¤¤™½È¤¥¸É…¹” È¥t(€€€€€€€™½ÈÐ¥¸Ñ¡É•…‘Ìè(€€€€€€€€€€€Ð¹ÍÑ…ÉÐ ¤(€€€€€€€™½ÈÐ¥¸Ñ¡É•…‘Ìè(€€€€€€€€€€€Ð¹©½¥¸¡Ñ¥µ•½ÕÐôÄÀ¤(€€€€€€€…ÍÍ•ÉÐ±•¸¡ÍÁ…Ý¹•¤€ôô€Ä°€ (€€€€€€€€€€€˜‰½¹ÕÉÉ•¹ÐÍ±…Í ¹•á•ŒÍÁ…Ý¹•í±•¸¡ÍÁ…Ý¹•¥ôÝ½É­•ÉÌƒŠP™¥ÉÍÐµÕÍ”€ˆ(€€€€€€€€€€€˜‰ÍÁ…Ý¸µÕÍÐ‰”Í•É¥…±¥é•Á•ÈÍ•ÍÍ¥½¸ˆ(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰Í±…Í¡}Ý½É­•È‰t¥ÌÍÁ…Ý¹•‘lÁt(€€€€€€€…ÍÍ•ÉÐ…±° ‰É•ÍÕ±Ðˆ¥¸È™½ÈÈ¥¸É•ÍÕ±ÑÌ¤°É•ÍÕ±ÑÌ(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰É…”µÍÁ…Ý¸ˆ°9½¹”¤(((()‘•˜Ñ•ÍÑ}±½Í•}Í•ÍÍ¥½¹Í}™½É}ÑÉ…¹ÍÁ½ÉÑ}±½Í•Í}™±…•‘}É•Á½¥¹ÑÍ}É•ÍÐ¡µ½¹­•åÁ…Ñ ¤è(€€€Í••¸€ômt(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°(€€€€€€€€‰}Ñ•…É‘½Ý¹}Á½ÁÁ•‘}Í•ÍÍ¥½¸ˆ°(€€€€€€€±…µ‰‘„Í•ÍÍ¥½¸°€¨°•¹‘}É•…Í½¸èÍ••¸¹…ÁÁ•¹ ¡Í•ÍÍ¥½¹l‰}Í¥‰t°•¹‘}É•…Í½¸¤¤½ÈQÉÕ”°(€€€€¤(€€€€Œ•Ñ…¡•Í•ÍÍ¥½¸€‰ˆˆÝ½Õ±Í¡•‘Õ±”„É•…°É…”µÉ•…ÀÑ¡É•…‘¥¹œ¹Q¥µ•ÈÑ¡…Ð(€€€€Œ½ÕÑ±¥Ù•ÌÑ¡”Ñ•ÍÐìÉ…”ôÀÍ¡½ÉÐµ¥ÉÕ¥ÑÌ¥ÐÍ¼¹¼Ñ¡É•…±¥¹•ÉÌ¸(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}]M}=IA!9}IA}I}Lˆ°€À¤(€€€ÑÉ…¹ÍÁ½ÉÐ€ô½‰©•Ð ¤€€ŒÑ¡”‘¥Í½¹¹•Ñ¥¹œÑÉ…¹ÍÁ½ÉÐ(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰„‰t€ôì‰ÑÉ…¹ÍÁ½ÉÐˆèÑÉ…¹ÍÁ½ÉÐ°€‰±½Í•}½¹}‘¥Í½¹¹•ÐˆèQÉÕ•ô(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰ˆ‰t€ôì‰ÑÉ…¹ÍÁ½ÉÐˆèÑÉ…¹ÍÁ½ÉÐ°€‰±½Í•}½¹}‘¥Í½¹¹•Ðˆè…±Í•ô(€€€ÑÉäè(€€€€€€€Í•ÉÙ•È¹}±½Í•}Í•ÍÍ¥½¹Í}™½É}ÑÉ…¹ÍÁ½ÉÐ¡ÑÉ…¹ÍÁ½ÉÐ°•¹‘}É•…Í½¸ô‰ÝÍ}‘¥Í½¹¹•Ðˆ¤(€€€€€€€…ÍÍ•ÉÐÍ••¸€ôôl ‰„ˆ°€‰ÝÍ}‘¥Í½¹¹•Ðˆ¥t€€Œ½¹±äÑ¡”™±…•½¹”±½Í•(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰ˆ‰ul‰ÑÉ…¹ÍÁ½ÉÐ‰t¥ÌÍ•ÉÙ•È¹}‘•Ñ…¡•‘}ÝÍ}ÑÉ…¹ÍÁ½ÉÐ€€ŒÉ”µÁ½¥¹Ñ•(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(()ÁåÑ•ÍÐ¹µ…É¬¹Á…É…µ•ÑÉ¥é” ‰±½Í•}½¹}‘¥Í½¹¹•Ðˆ°mQÉÕ”°…±Í•t¤)‘•˜Ñ•ÍÑ}±½Í•}Í•ÍÍ¥½¹Í}™½É}ÑÉ…¹ÍÁ½ÉÑ}Í­¥ÁÍ}Í•ÍÍ¥½¹}É•‰½Õ¹‘}‰•™½É•}±…¥´ (€€€µ½¹­•åÁ…Ñ °±½Í•}½¹}‘¥Í½¹¹•Ð(¤è(€€€€ˆˆ‰É•ÍÕµ”‰•ÑÝ••¸Í¹…ÁÍ¡½Ð…¹±…¥´­••ÁÌ•¥Ñ¡•ÈÍ•ÍÍ¥½¸ÑåÁ”…±¥Ù”¸ˆˆˆ(€€€É•…ÁÌ€ômt(€€€Ñ•…É‘½Ý¹Ì€ômt(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}Í¡•‘Õ±•}ÝÍ}½ÉÁ¡…¹}É•…Àˆ°±…µ‰‘„Í¥èÉ•…ÁÌ¹…ÁÁ•¹¡Í¥¤(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°(€€€€€€€€‰}Ñ•…É‘½Ý¹}Á½ÁÁ•‘}Í•ÍÍ¥½¸ˆ°(€€€€€€€±…µ‰‘„Í•ÍÍ¥½¸°€¨°•¹‘}É•…Í½¸èÑ•…É‘½Ý¹Ì¹…ÁÁ•¹ ¡Í•ÍÍ¥½¸°•¹‘}É•…Í½¸¤¤½ÈQÉÕ”°(€€€€¤(€€€½±‘}ÑÉ…¹ÍÁ½ÉÐ€ô½‰©•Ð ¤€€ŒÑ¡”‘¥Í½¹¹•Ñ¥¹œÑÉ…¹ÍÁ½ÉÐ(€€€¹•Ý}ÑÉ…¹ÍÁ½ÉÐ€ô½‰©•Ð ¤€€Œ±¥Ù”É•‰¥¹Ñ…É•Ð€¡¹¼}±½Í•…ÑÑÈƒŠH…±¥Ù”¤(€€€Í•ÍÍ¥½¸€ôì‰ÑÉ…¹ÍÁ½ÉÐˆè½±‘}ÑÉ…¹ÍÁ½ÉÐ°€‰±½Í•}½¹}‘¥Í½¹¹•Ðˆè±½Í•}½¹}‘¥Í½¹¹•Ñô(€€€½É¥¥¹…±}Í•ÍÍ¥½¹Í}±½¬€ôÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Í}±½¬(€€€É•‰½Õ¹€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤((€€€±…ÍÌ}M¹…ÁÍ¡½Ñ%¹Ñ•É±½¬è(€€€€€€€€ˆˆ‰I•‰¥¹¥¸„Í•½¹Ñ¡É•…¥µµ•‘¥…Ñ•±ä…™Ñ•ÈÑ¡”½Ý¹•ÉÍ¡¥ÀÍ¹…ÁÍ¡½Ð¸ˆˆˆ((€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹}Í¹…ÁÍ¡½Ñ}É•±•…Í•€ô…±Í”((€€€€€€€‘•˜}}•¹Ñ•É}|¡Í•±˜¤è(€€€€€€€€€€€½É¥¥¹…±}Í•ÍÍ¥½¹Í}±½¬¹…ÅÕ¥É” ¤(€€€€€€€€€€€É•ÑÕÉ¸Í•±˜((€€€€€€€‘•˜}}•á¥Ñ}|¡Í•±˜°•á}ÑåÁ”°•áŒ°ÑÉ…•‰…¬¤è(€€€€€€€€€€€½É¥¥¹…±}Í•ÍÍ¥½¹Í}±½¬¹É•±•…Í” ¤(€€€€€€€€€€€¥˜¹½ÐÍ•±˜¹}Í¹…ÁÍ¡½Ñ}É•±•…Í•è(€€€€€€€€€€€€€€€Í•±˜¹}Í¹…ÁÍ¡½Ñ}É•±•…Í•€ôQÉÕ”((€€€€€€€€€€€€€€€‘•˜}É•ÍÕµ•}É•‰¥¹ ¤è(€€€€€€€€€€€€€€€€€€€Ý¥Ñ Í•ÉÙ•È¹}Í•ÍÍ¥½¹}É•ÍÕµ•}±½¬è(€€€€€€€€€€€€€€€€€€€€€€€Í•ÍÍ¥½¹l‰ÑÉ…¹ÍÁ½ÉÐ‰t€ô¹•Ý}ÑÉ…¹ÍÁ½ÉÐ(€€€€€€€€€€€€€€€€€€€É•‰½Õ¹¹Í•Ð ¤((€€€€€€€€€€€€€€€Ñ¡É•…€ôÑ¡É•…‘¥¹œ¹Q¡É•…¡Ñ…É•Ðõ}É•ÍÕµ•}É•‰¥¹¤(€€€€€€€€€€€€€€€Ñ¡É•…¹ÍÑ…ÉÐ ¤(€€€€€€€€€€€€€€€…ÍÍ•ÉÐÉ•‰½Õ¹¹Ý…¥Ð¡Ñ¥µ•½ÕÐôÄ¤(€€€€€€€€€€€€€€€Ñ¡É•…¹©½¥¸¡Ñ¥µ•½ÕÐôÄ¤(€€€€€€€€€€€É•ÑÕÉ¸…±Í”((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹Í}±½¬ˆ°}M¹…ÁÍ¡½Ñ%¹Ñ•É±½¬ ¤¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰É•‰½Õ¹‰t€ôÍ•ÍÍ¥½¸(€€€ÑÉäè(€€€€€€€É•…Á•°‘•Ñ…¡•€ôÍ•ÉÙ•È¹}±½Í•}Í•ÍÍ¥½¹Í}™½É}ÑÉ…¹ÍÁ½ÉÐ¡½±‘}ÑÉ…¹ÍÁ½ÉÐ¤(€€€€€€€…ÍÍ•ÉÐÉ•…Á•€ôô€À(€€€€€€€…ÍÍ•ÉÐ‘•Ñ…¡•€ôô€À(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰É•‰½Õ¹‰t¥ÌÍ•ÍÍ¥½¸(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰ÑÉ…¹ÍÁ½ÉÐ‰t¥Ì¹•Ý}ÑÉ…¹ÍÁ½ÉÐ(€€€€€€€…ÍÍ•ÉÐÑ•…É‘½Ý¹Ì€ôômt(€€€€€€€…ÍÍ•ÉÐÉ•…ÁÌ€ôômt(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}É•…Ñ•}É•½É‘Í}±½Í•}½¹}‘¥Í½¹¹•Ñ}™±…œ¡µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„Í¥°Í•ÍÍ¥½¸è9½¹”¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(€€€ÑÉäè(€€€€€€€½¸€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹É•…Ñ”ˆ°€‰Á…É…µÌˆèì‰±½Í•}½¹}‘¥Í½¹¹•ÐˆèQÉÕ•õô(€€€€€€€€¥l‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰t(€€€€€€€½™˜€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÈˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹É•…Ñ”ˆ°€‰Á…É…µÌˆèíõô(€€€€€€€€¥l‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰t(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Ím½¹ul‰±½Í•}½¹}‘¥Í½¹¹•Ð‰t(€€€€€€€…ÍÍ•ÉÐ¹½ÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Ím½™™ul‰±½Í•}½¹}‘¥Í½¹¹•Ð‰t(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}É•…Ñ•}É•½É‘Í}Í½ÕÉ”¡µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„Í¥°Í•ÍÍ¥½¸è9½¹”¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(€€€ÑÉäè(€€€€€€€Í¥€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹É•…Ñ”ˆ°€‰Á…É…µÌˆèì‰Í½ÕÉ”ˆè€‰Ñ½½°‰õô(€€€€€€€€¥l‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰t(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÍ¥‘ul‰Í½ÕÉ”‰t€ôô€‰Ñ½½°ˆ(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(()‘•˜Ñ•ÍÑ}Í¡ÕÑ‘½Ý¹}Í•ÍÍ¥½¹Í}±½Í•Í}•Ù•Éå}Í•ÍÍ¥½¹}Ù¥…}¡•±Á•È¡µ½¹­•åÁ…Ñ ¤è(€€€Í••¸€ômt(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}±½Í•}Í•ÍÍ¥½¹}‰å}¥ˆ°(€€€€€€€±…µ‰‘„Í¥°€¨°•¹‘}É•…Í½¸èÍ••¸¹…ÁÁ•¹ ¡Í¥°•¹‘}É•…Í½¸¤¤°(€€€€¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰„‰t€ôíô(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰ˆ‰t€ôíô(€€€ÑÉäè(€€€€€€€Í•ÉÙ•È¹}Í¡ÕÑ‘½Ý¹}Í•ÍÍ¥½¹Ì ¤(€€€€€€€…ÍÍ•ÉÐÍ½ÉÑ•¡Í¥™½ÈÍ¥°|¥¸Í••¸¤€ôôl‰„ˆ°€‰ˆ‰t(€€€€€€€…ÍÍ•ÉÐíÉ•…Í½¸™½È|°É•…Í½¸¥¸Í••¹ô€ôôì‰ÑÕ¥}Í¡ÕÑ‘½Ý¸‰ô(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(()‘•˜}¥‘±•}•Ù¥Ñ…‰±•}Í•ÍÍ¥½¸¡¹½Ü¤è(€€€€ˆˆ‰Í•ÍÍ¥½¸Ñ¡…ÐÍ…Ñ¥Í™¥•Ì•Ù•Éä•Ù¥Ñ¥½¸ÁÉ•½¹‘¥Ñ¥½¸¸ˆˆˆ(€€€É•…‘ä€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤(€€€É•…‘ä¹Í•Ð ¤(€€€½±€ô¹½Ü€´€ÄÀ€¨€ÌØÀÀ€€ŒÝ•±°Á…ÍÐÑ¡”€Ù QQ0(€€€É•ÑÕÉ¸ì(€€€€€€€€‰ÉÕ¹¹¥¹œˆè…±Í”°(€€€€€€€€‰…•¹Ñ}É•…‘äˆèÉ•…‘ä°(€€€€€€€€‰ÑÉ…¹ÍÁ½ÉÐˆèÍ•ÉÙ•È¹}‘•Ñ…¡•‘}ÝÍ}ÑÉ…¹ÍÁ½ÉÐ°€€Œ‘•…½‘•Ñ…¡•(€€€€€€€€‰±…ÍÑ}…Ñ¥Ù”ˆè½±°(€€€€€€€€‰É•…Ñ•‘}…Ðˆè½±°(€€€ô(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}¥Í}•Ù¥Ñ…‰±•}Ý¡•¹}¥‘±•}‘•…‘}…¹‘}ÅÕ¥•Í•¹Ð¡µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}Á•¹‘¥¹}­¥¹ˆ°±…µ‰‘„Í¥è€ˆˆ¤(€€€¹½Ü€ôÑ¥µ”¹Ñ¥µ” ¤(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹}¥Í}•Ù¥Ñ…‰±” ‰Ìˆ°}¥‘±•}•Ù¥Ñ…‰±•}Í•ÍÍ¥½¸¡¹½Ü¤°¹½Ü¤¥ÌQÉÕ”(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}¹½Ñ}•Ù¥Ñ…‰±•}Ù¥½±…Ñ¥¹}•…¡}•á•µÁÑ¥½¸¡µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}Á•¹‘¥¹}­¥¹ˆ°±…µ‰‘„Í¥è€ˆˆ¤(€€€¹½Ü€ôÑ¥µ”¹Ñ¥µ” ¤(€€€±¥Ù•}ÑÉ…¹ÍÁ½ÉÐ€ôÑåÁ” ‰Pˆ°€ ¤°ì‰}±½Í•ˆè…±Í•ô¤ ¤((€€€ÉÕ¹¹¥¹œ€ô}¥‘±•}•Ù¥Ñ…‰±•}Í•ÍÍ¥½¸¡¹½Ü¤ðì‰ÉÕ¹¹¥¹œˆèQÉÕ•ô(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹}¥Í}•Ù¥Ñ…‰±” ‰Ìˆ°ÉÕ¹¹¥¹œ°¹½Ü¤¥Ì…±Í”((€€€ÍÑ…ÉÑ¥¹œ€ô}¥‘±•}•Ù¥Ñ…‰±•}Í•ÍÍ¥½¸¡¹½Ü¤(€€€ÍÑ…ÉÑ¥¹l‰…•¹Ñ}É•…‘ä‰t€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤€€Œ¹½ÐÍ•Ð€´øÍÑ¥±°ÍÑ…ÉÑ¥¹œ(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹}¥Í}•Ù¥Ñ…‰±” ‰Ìˆ°ÍÑ…ÉÑ¥¹œ°¹½Ü¤¥Ì…±Í”((€€€½¹}Í½­•Ð€ô}¥‘±•}•Ù¥Ñ…‰±•}Í•ÍÍ¥½¸¡¹½Ü¤ðì‰ÑÉ…¹ÍÁ½ÉÐˆè±¥Ù•}ÑÉ…¹ÍÁ½ÉÑô(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹}¥Í}•Ù¥Ñ…‰±” ‰Ìˆ°½¹}Í½­•Ð°¹½Ü¤¥Ì…±Í”((€€€É••¹Ð€ô}¥‘±•}•Ù¥Ñ…‰±•}Í•ÍÍ¥½¸¡¹½Ü¤ðì‰±…ÍÑ}…Ñ¥Ù”ˆè¹½Ýô(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹}¥Í}•Ù¥Ñ…‰±” ‰Ìˆ°É••¹Ð°¹½Ü¤¥Ì…±Í”((€€€å½Õ¹œ€ô}¥‘±•}•Ù¥Ñ…‰±•}Í•ÍÍ¥½¸¡¹½Ü¤ðì‰É•…Ñ•‘}…Ðˆè¹½Ýô(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹}¥Í}•Ù¥Ñ…‰±” ‰Ìˆ°å½Õ¹œ°¹½Ü¤¥Ì…±Í”((€€€€ŒA•¹‘¥¹œ¥¹ÁÕÐÉ•ÅÕ•ÍÐ°•Ù•¸Ý¡•¸•Ù•ÉåÑ¡¥¹œ•±Í”±½½­Ì¥‘±”¸(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}Á•¹‘¥¹}­¥¹ˆ°±…µ‰‘„Í¥è€‰¥¹ÁÕÐˆ¤(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹}¥Í}•Ù¥Ñ…‰±” ‰Ìˆ°}¥‘±•}•Ù¥Ñ…‰±•}Í•ÍÍ¥½¸¡¹½Ü¤°¹½Ü¤¥Ì…±Í”(()‘•˜Ñ•ÍÑ}É•…Á}¥‘±•}Í•ÍÍ¥½¹Í}±½Í•Í}½¹±å}•Ù¥Ñ…‰±”¡µ½¹­•åÁ…Ñ ¤è(€€€±½Í•€ômt(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}Á•¹‘¥¹}­¥¹ˆ°±…µ‰‘„Í¥è€ˆˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}±½Í•}Í•ÍÍ¥½¹}‰å}¥ˆ°(€€€€€€€±…µ‰‘„Í¥°€¨°•¹‘}É•…Í½¸°ÁÉ•‘¥…Ñ”õ9½¹”è±½Í•¹…ÁÁ•¹ ¡Í¥°•¹‘}É•…Í½¸¤¤°(€€€€¤(€€€¹½Ü€ôÑ¥µ”¹Ñ¥µ” ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰ÍÑ…±”‰t€ô}¥‘±•}•Ù¥Ñ…‰±•}Í•ÍÍ¥½¸¡¹½Ü¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰™É•Í ‰t€ô}¥‘±•}•Ù¥Ñ…‰±•}Í•ÍÍ¥½¸¡¹½Ü¤ðì‰±…ÍÑ}…Ñ¥Ù”ˆè¹½Ýô(€€€ÑÉäè(€€€€€€€Í•ÉÙ•È¹}É•…Á}¥‘±•}Í•ÍÍ¥½¹Ì ¤(€€€€€€€…ÍÍ•ÉÐ±½Í•€ôôl ‰ÍÑ…±”ˆ°€‰¥‘±•}Ñ¥µ•½ÕÐˆ¥t(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(()‘•˜}Á•É¥½‘¥}ÑÉ¥µ}…±±Ì¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰MÑÕˆÑ¡”É•…Á•ÈÌÍ¥‘”•™™•ÑÌ…¹…ÁÑÕÉ”ÑÉ¥µ}µ•µ½Éä…±±Ì€¡‘•±…å•¥µÁ½ÉÐƒŠHÁ…Ñ Ñ¡”µ½‘Õ±”…ÑÑÈ¤¸ˆˆˆ(€€€¥µÁ½ÉÐ¡•Éµ•Í}±¤¹µ•µ}ÑÉ¥´…Ìµ•µ}ÑÉ¥´((€€€…±±Ì€ômt(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}Á•¹‘¥¹}­¥¹ˆ°±…µ‰‘„Í¥è€ˆˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½Í•}Í•ÍÍ¥½¹}‰å}¥ˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•¹™½É•}Í•ÍÍ¥½¹}…Àˆ°±…µ‰‘„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}É•±…¥µ}½ÉÁ¡…¹•‘}±•…Í•Ìˆ°±…µ‰‘„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡µ•µ}ÑÉ¥´°€‰ÑÉ¥µ}µ•µ½Éäˆ°±…µ‰‘„€¨©­Üè…±±Ì¹…ÁÁ•¹¡­Ü¹•Ð ‰É•…Í½¸ˆ°€ˆˆ¤¤½ÈQÉÕ”¤(€€€É•ÑÕÉ¸…±±Ì(()‘•˜Ñ•ÍÑ}Á•É¥½‘¥}ÑÉ¥µ}‘•™•ÉÉ•‘}Ý¡¥±•}…}Í•ÍÍ¥½¹}¥Í}‰ÕÍå}½É}…ÑÑ…¡•¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰Q¡”•¸´È½±±•Ð€¬µ…±±½}ÑÉ¥´ÍÑ…±±ÌÑ¡”±½½À™½È¥ÑÌÝ¡½±”‘ÕÉ…Ñ¥½¸€ ŒÔàÔÜØ¤è¥ÐµÕÍÐ¹½ÐÉÕ¸(€€€Ý¡¥±”…¹äÍ•ÍÍ¥½¸¥Ìµ¥µÑÕÉ¸½ÈÍÑ¥±°¡½±‘Ì„±¥Ù”±¥•¹Ð°Ý¡…Ñ•Ù•ÈÑ¡”½Ñ¡•ÈÍ•ÍÍ¥½¹Ì±½½¬±¥­”¸ˆˆˆ(€€€…±±Ì€ô}Á•É¥½‘¥}ÑÉ¥µ}…±±Ì¡µ½¹­•åÁ…Ñ ¤(€€€¹½Ü€ôÑ¥µ”¹Ñ¥µ” ¤(€€€±¥Ù”€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡}±½Í•õ…±Í”¤(€€€™½È‰ÕÍä¥¸€¡ì‰ÉÕ¹¹¥¹œˆèQÉÕ•ô°ì‰ÑÉ…¹ÍÁ½ÉÐˆè±¥Ù•ô¤è(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰¥‘±”‰t€ô}¥‘±•}•Ù¥Ñ…‰±•}Í•ÍÍ¥½¸¡¹½Ü¤(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰‰ÕÍä‰t€ô}¥‘±•}•Ù¥Ñ…‰±•}Í•ÍÍ¥½¸¡¹½Ü¤ð‰ÕÍä(€€€€€€€ÑÉäè(€€€€€€€€€€€Í•ÉÙ•È¹}É•…Á}¥‘±•}Í•ÍÍ¥½¹Ì ¤(€€€€€€€€€€€…ÍÍ•ÉÐ…±±Ì€ôômt°‰ÕÍä(€€€€€€€™¥¹…±±äè(€€€€€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(()‘•˜Ñ•ÍÑ}Á•É¥½‘¥}ÑÉ¥µ}ÉÕ¹Í}½¹•}•Ù•Éå}Í•ÍÍ¥½¹}¥Í}ÅÕ¥•Í•¹Ð¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰Ù•ÉäÅÕ¥•Í•¹ÐÍ…¸ÑÉ¥µÌ°•Ù•¸Ý¥Ñ ¹¼Ù¥Ñ¥µÌè¹¼Í•ÍÍ¥½¹Ì…Ð…±°°½È½¹±äÉ••¹Ð€¡¹½Ðå•Ð(€€€QQ0µ•Ù¥Ñ…‰±”¤Í•ÍÍ¥½¹ÌÑ¡…Ð…É”‘•Ñ…¡•…¹¥‘±”¸ˆˆˆ(€€€…±±Ì€ô}Á•É¥½‘¥}ÑÉ¥µ}…±±Ì¡µ½¹­•åÁ…Ñ ¤(€€€¹½Ü€ôÑ¥µ”¹Ñ¥µ” ¤(€€€™½ÈÍ•ÍÍ¥½¹Ì¥¸€¡íô°ì‰Á…É­•ˆè}¥‘±•}•Ù¥Ñ…‰±•}Í•ÍÍ¥½¸¡¹½Ü¤ðì‰±…ÍÑ}…Ñ¥Ù”ˆè¹½Ü°€‰É•…Ñ•‘}…Ðˆè¹½Ýõô¤è(€€€€€€€…±±Ì¹±•…È ¤(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹ÕÁ‘…Ñ”¡Í•ÍÍ¥½¹Ì¤(€€€€€€€ÑÉäè(€€€€€€€€€€€Í•ÉÙ•È¹}É•…Á}¥‘±•}Í•ÍÍ¥½¹Ì ¤(€€€€€€€€€€€…ÍÍ•ÉÐ±•¸¡…±±Ì¤€ôô€Ä°Í•ÍÍ¥½¹Ì(€€€€€€€™¥¹…±±äè(€€€€€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(()‘•˜Ñ•ÍÑ}ÑÕÉ¹}½µÁ±•Ñ¥½¹}ÑÉ¥µ}Í­¥ÁÍ}Ý¡¥±•}…¹½Ñ¡•É}Í•ÍÍ¥½¹}¥Í}ÉÕ¹¹¥¹œ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰Q¡”™¥¹¥Í¡¥¹œÍ•ÍÍ¥½¸¥ÌÍÑ¥±°µ…É­•ÉÕ¹¹¥¹œÝ¡•¸}™¥¹¥Í¡}ÑÕÉ¸ÉÕ¹Ì°Í¼½¹±ä=Q!HÍ•ÍÍ¥½¹Ì…Ñ”¥ÑÌ(€€€ÑÉ¥´è„Í½±”Í•ÍÍ¥½¸ÑÉ¥µÌ…Ð•Ù•ÉäÑÕÉ¸•¹ì„Í•½¹¥¸µ™±¥¡ÐÑÕÉ¸‘•™•ÉÌ¥Ð€ ŒÔàÔÜØ¤¸ˆˆˆ(€€€…±±Ì€ô}Á•É¥½‘¥}ÑÉ¥µ}…±±Ì¡µ½¹­•åÁ…Ñ ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±•…É}Í•ÍÍ¥½¹}½¹Ñ•áÐˆ°±…µ‰‘„Ñ½­•¹Ìè9½¹”¤(€€€¹½Ü€ôÑ¥µ”¹Ñ¥µ” ¤(€€€½Ý¸€ô}¥‘±•}•Ù¥Ñ…‰±•}Í•ÍÍ¥½¸¡¹½Ü¤ðì‰ÉÕ¹¹¥¹œˆèQÉÕ”°€‰ÑÉ…¹ÍÁ½ÉÐˆèÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡}±½Í•õ…±Í”¥ô(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰½Ý¸‰t€ô½Ý¸(€€€ÑÉäè(€€€€€€€Í•ÉÙ•È¹}™¥¹¥Í¡}ÑÕÉ¸ ‰½Ý¸ˆ°½Ý¸°Í•ÉÙ•È¹}QÕÉ¹IÕ¸¡…•¹Ðõ9½¹”°½¹•}ÑÕÉ¹}É•ÍÑ½É”õ9½¹”°Ñ•Éµ¥¹…±}…±±‰…¬õ9½¹”°É••¥ÁÑ}½µµ¥ÑÑ•õQÉÕ”¤¤(€€€€€€€…ÍÍ•ÉÐ±•¸¡…±±Ì¤€ôô€Ä((€€€€€€€…±±Ì¹±•…È ¤(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰½Ñ¡•È‰t€ô}¥‘±•}•Ù¥Ñ…‰±•}Í•ÍÍ¥½¸¡¹½Ü¤ðì‰ÉÕ¹¹¥¹œˆèQÉÕ•ô(€€€€€€€Í•ÉÙ•È¹}™¥¹¥Í¡}ÑÕÉ¸ ‰½Ý¸ˆ°½Ý¸°Í•ÉÙ•È¹}QÕÉ¹IÕ¸¡…•¹Ðõ9½¹”°½¹•}ÑÕÉ¹}É•ÍÑ½É”õ9½¹”°Ñ•Éµ¥¹…±}…±±‰…¬õ9½¹”°É••¥ÁÑ}½µµ¥ÑÑ•õQÉÕ”¤¤(€€€€€€€…ÍÍ•ÉÐ…±±Ì€ôômt(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(()‘•˜Ñ•ÍÑ}É•…Á}¥‘±•}Í•ÍÍ¥½¹Í}±½Í}ÑÉ¥µ}™…¥±ÕÉ”¡µ½¹­•åÁ…Ñ °…Á±½œ¤è(€€€¥µÁ½ÉÐ¡•Éµ•Í}±¤¹µ•µ}ÑÉ¥´…Ìµ•µ}ÑÉ¥´((€€€}Á•É¥½‘¥}ÑÉ¥µ}…±±Ì¡µ½¹­•åÁ…Ñ ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡µ•µ}ÑÉ¥´°€‰ÑÉ¥µ}µ•µ½Éäˆ°±…µ‰‘„€¨©}­Üè€¡|™½È|¥¸€ ¤¤¹Ñ¡É½Ü¡IÕ¹Ñ¥µ•ÉÉ½È ‰‰½½´ˆ¤¤¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(€€€ÑÉäè(€€€€€€€Ý¥Ñ …Á±½œ¹…Ñ}±•Ù•° ‰	Uˆ°±½•Èô‰ÑÕ¥}…Ñ•Ý…ä¹Í•ÉÙ•Èˆ¤è(€€€€€€€€€€€Í•ÉÙ•È¹}É•…Á}¥‘±•}Í•ÍÍ¥½¹Ì ¤(€€€€€€€…ÍÍ•ÉÐ€‰‰½½´ˆ¥¸…Á±½œ¹Ñ•áÐ(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(()‘•˜Ñ•ÍÑ}ÑÑ±}É•…Á•É}ÍÁ…É•Í}Í•ÍÍ¥½¹}Ý¥Ñ¡}…Ñ¥Ù•}‘•±•…Ñ¥½¸¡µ½¹­•åÁ…Ñ ¤è(€€€™É½´Ñ½½±Ì¥µÁ½ÉÐ…Íå¹}‘•±•…Ñ¥½¸((€€€±½Í•€ômt(€€€‘•±•…Ñ¥½¹}¥€ô€‰‘•±•}ÑÑ±}É•…Á•É}Ñ•ÍÐˆ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}Á•¹‘¥¹}­¥¹ˆ°±…µ‰‘„Í¥è€ˆˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•¹™½É•}Í•ÍÍ¥½¹}…Àˆ°±…µ‰‘„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}É•±…¥µ}½ÉÁ¡…¹•‘}±•…Í•Ìˆ°±…µ‰‘„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°(€€€€€€€€‰}±½Í•}Í•ÍÍ¥½¹}‰å}¥ˆ°(€€€€€€€±…µ‰‘„Í¥°€¨°•¹‘}É•…Í½¸°ÁÉ•‘¥…Ñ”õ9½¹”è±½Í•¹…ÁÁ•¹ ¡Í¥°•¹‘}É•…Í½¸¤¤°(€€€€¤(€€€¹½Ü€ôÑ¥µ”¹Ñ¥µ” ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰‘•±•…Ñ¥¹œµÑÑ°‰t€ô}¥‘±•}•Ù¥Ñ…‰±•}Í•ÍÍ¥½¸¡¹½Ü¤(€€€Ý¥Ñ …Íå¹}‘•±•…Ñ¥½¸¹}É•½É‘Í}±½¬è(€€€€€€€…Íå¹}‘•±•…Ñ¥½¸¹}É•½É‘Ím‘•±•…Ñ¥½¹}¥‘t€ôì(€€€€€€€€€€€€‰ÍÑ…ÑÕÌˆè€‰ÉÕ¹¹¥¹œˆ°(€€€€€€€€€€€€‰½É¥¥¹}Õ¥}Í•ÍÍ¥½¹}¥ˆè€‰‘•±•…Ñ¥¹œµÑÑ°ˆ°(€€€€€€€ô((€€€ÑÉäè(€€€€€€€Í•ÉÙ•È¹}É•…Á}¥‘±•}Í•ÍÍ¥½¹Ì ¤(€€€€€€€…ÍÍ•ÉÐ±½Í•€ôômt(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(€€€€€€€Ý¥Ñ …Íå¹}‘•±•…Ñ¥½¸¹}É•½É‘Í}±½¬è(€€€€€€€€€€€…Íå¹}‘•±•…Ñ¥½¸¹}É•½É‘Ì¹Á½À¡‘•±•…Ñ¥½¹}¥°9½¹”¤(()‘•˜Ñ•ÍÑ}±ÉÕ}É•…Á•É}ÍÁ…É•Í}…Ñ¥Ù•}‘•±•…Ñ¥½¹}…¹‘}•Ù¥ÑÍ}¥‘±•}Á••È¡µ½¹­•åÁ…Ñ ¤è(€€€™É½´Ñ½½±Ì¥µÁ½ÉÐ…Íå¹}‘•±•…Ñ¥½¸((€€€±½Í•€ômt(€€€‘•±•…Ñ¥½¹}¥€ô€‰‘•±•}±ÉÕ}É•…Á•É}Ñ•ÍÐˆ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}Á•¹‘¥¹}­¥¹ˆ°±…µ‰‘„Í¥è€ˆˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}µ…á}±¥Ù•}Í•ÍÍ¥½¹Ìˆ°±…µ‰‘„è€Ä¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°(€€€€€€€€‰}±½Í•}Í•ÍÍ¥½¹}‰å}¥ˆ°(€€€€€€€±…µ‰‘„Í¥°€¨°•¹‘}É•…Í½¸°ÁÉ•‘¥…Ñ”õ9½¹”è±½Í•¹…ÁÁ•¹ ¡Í¥°•¹‘}É•…Í½¸¤¤°(€€€€¤(€€€¹½Ü€ôÑ¥µ”¹Ñ¥µ” ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰‘•±•…Ñ¥¹œµ±ÉÔ‰t€ô}¥‘±•}•Ù¥Ñ…‰±•}Í•ÍÍ¥½¸¡¹½Ü¤ðì(€€€€€€€€‰±…ÍÑ}…Ñ¥Ù”ˆè¹½Ü€´€ÈÀ€¨€ÌØÀÀ(€€€ô(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰¥‘±”µÁ••È‰t€ô}¥‘±•}•Ù¥Ñ…‰±•}Í•ÍÍ¥½¸¡¹½Ü¤(€€€Ý¥Ñ …Íå¹}‘•±•…Ñ¥½¸¹}É•½É‘Í}±½¬è(€€€€€€€…Íå¹}‘•±•…Ñ¥½¸¹}É•½É‘Ím‘•±•…Ñ¥½¹}¥‘t€ôì(€€€€€€€€€€€€‰ÍÑ…ÑÕÌˆè€‰ÉÕ¹¹¥¹œˆ°(€€€€€€€€€€€€‰½É¥¥¹}Õ¥}Í•ÍÍ¥½¹}¥ˆè€‰‘•±•…Ñ¥¹œµ±ÉÔˆ°(€€€€€€€ô((€€€ÑÉäè(€€€€€€€Í•ÉÙ•È¹}•¹™½É•}Í•ÍÍ¥½¹}…À ¤(€€€€€€€…ÍÍ•ÉÐ±½Í•€ôôl ‰¥‘±”µÁ••Èˆ°€‰±ÉÕ}•Ù¥Ðˆ¥t(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(€€€€€€€Ý¥Ñ …Íå¹}‘•±•…Ñ¥½¸¹}É•½É‘Í}±½¬è(€€€€€€€€€€€…Íå¹}‘•±•…Ñ¥½¸¹}É•½É‘Ì¹Á½À¡‘•±•…Ñ¥½¹}¥°9½¹”¤(()‘•˜Ñ•ÍÑ}ÑÑ±}É•…Á•É}É•Ù…±¥‘…Ñ•Í}Í•ÍÍ¥½¹}‰•™½É•}Ñ•…É‘½Ý¸¡µ½¹­•åÁ…Ñ ¤è(€€€±½Í•€ômt(€€€…±±Ì€ôì‰½Õ¹Ðˆè€Áô(€€€±¥Ù•}ÑÉ…¹ÍÁ½ÉÐ€ôÑåÁ” ‰Pˆ°€ ¤°ì‰}±½Í•ˆè…±Í•ô¤ ¤(€€€½É¥¥¹…±}¥Í}•Ù¥Ñ…‰±”€ôÍ•ÉÙ•È¹}Í•ÍÍ¥½¹}¥Í}•Ù¥Ñ…‰±”(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}Á•¹‘¥¹}­¥¹ˆ°±…µ‰‘„Í¥è€ˆˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•¹™½É•}Í•ÍÍ¥½¹}…Àˆ°±…µ‰‘„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}É•±…¥µ}½ÉÁ¡…¹•‘}±•…Í•Ìˆ°±…µ‰‘„è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°(€€€€€€€€‰}Ñ•…É‘½Ý¹}Í•ÍÍ¥½¸ˆ°(€€€€€€€±…µ‰‘„Í•ÍÍ¥½¸°€¨°•¹‘}É•…Í½¸è±½Í•¹…ÁÁ•¹ ¡Í•ÍÍ¥½¸°•¹‘}É•…Í½¸¤¤°(€€€€¤((€€€‘•˜}É•…ÑÑ…¡}…™Ñ•É}Í…¸¡Í¥°Í•ÍÍ¥½¸°¹½Ü¤è(€€€€€€€…±±Íl‰½Õ¹Ð‰t€¬ô€Ä(€€€€€€€•Ù¥Ñ…‰±”€ô½É¥¥¹…±}¥Í}•Ù¥Ñ…‰±”¡Í¥°Í•ÍÍ¥½¸°¹½Ü¤(€€€€€€€¥˜…±±Íl‰½Õ¹Ð‰t€ôô€Äè(€€€€€€€€€€€Í•ÍÍ¥½¹l‰ÑÉ…¹ÍÁ½ÉÐ‰t€ô±¥Ù•}ÑÉ…¹ÍÁ½ÉÐ(€€€€€€€É•ÑÕÉ¸•Ù¥Ñ…‰±”((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥Í}•Ù¥Ñ…‰±”ˆ°}É•…ÑÑ…¡}…™Ñ•É}Í…¸¤(€€€¹½Ü€ôÑ¥µ”¹Ñ¥µ” ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰ÑÑ°µÉ…”‰t€ô}¥‘±•}•Ù¥Ñ…‰±•}Í•ÍÍ¥½¸¡¹½Ü¤((€€€ÑÉäè(€€€€€€€Í•ÉÙ•È¹}É•…Á}¥‘±•}Í•ÍÍ¥½¹Ì ¤(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰ÑÑ°µÉ…”‰ul‰ÑÉ…¹ÍÁ½ÉÐ‰t¥Ì±¥Ù•}ÑÉ…¹ÍÁ½ÉÐ(€€€€€€€…ÍÍ•ÉÐ…±±Íl‰½Õ¹Ð‰t€ôô€È(€€€€€€€…ÍÍ•ÉÐ±½Í•€ôômt(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(()‘•˜Ñ•ÍÑ}±ÉÕ}É•…Á•É}É•Ù…±¥‘…Ñ•Í}…¹‘}ÑÉ¥•Í}¹•áÑ}…¹‘¥‘…Ñ”¡µ½¹­•åÁ…Ñ ¤è(€€€±½Í•€ômt(€€€±¥Ù•}ÑÉ…¹ÍÁ½ÉÐ€ôÑåÁ” ‰Pˆ°€ ¤°ì‰}±½Í•ˆè…±Í•ô¤ ¤(€€€½É¥¥¹…±}¥Í}•Ù¥Ñ…‰±”€ôÍ•ÉÙ•È¹}Í•ÍÍ¥½¹}¥Í}±ÉÕ}•Ù¥Ñ…‰±”(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}Á•¹‘¥¹}­¥¹ˆ°±…µ‰‘„Í¥è€ˆˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}µ…á}±¥Ù•}Í•ÍÍ¥½¹Ìˆ°±…µ‰‘„è€Ä¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°(€€€€€€€€‰}Ñ•…É‘½Ý¹}Í•ÍÍ¥½¸ˆ°(€€€€€€€±…µ‰‘„Í•ÍÍ¥½¸°€¨°•¹‘}É•…Í½¸è±½Í•¹…ÁÁ•¹ ¡Í•ÍÍ¥½¸°•¹‘}É•…Í½¸¤¤°(€€€€¤((€€€‘•˜}É•…ÑÑ…¡}½±‘•ÍÑ}…™Ñ•É}Í…¸¡Í¥°Í•ÍÍ¥½¸¤è(€€€€€€€•Ù¥Ñ…‰±”€ô½É¥¥¹…±}¥Í}•Ù¥Ñ…‰±”¡Í¥°Í•ÍÍ¥½¸¤(€€€€€€€¥˜Í¥€ôô€‰±ÉÔµÉ…”ˆ…¹Í•ÍÍ¥½¸¹•Ð ‰ÑÉ…¹ÍÁ½ÉÐˆ¤¥ÌÍ•ÉÙ•È¹}‘•Ñ…¡•‘}ÝÍ}ÑÉ…¹ÍÁ½ÉÐè(€€€€€€€€€€€Í•ÍÍ¥½¹l‰ÑÉ…¹ÍÁ½ÉÐ‰t€ô±¥Ù•}ÑÉ…¹ÍÁ½ÉÐ(€€€€€€€É•ÑÕÉ¸•Ù¥Ñ…‰±”((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥Í}±ÉÕ}•Ù¥Ñ…‰±”ˆ°}É•…ÑÑ…¡}½±‘•ÍÑ}…™Ñ•É}Í…¸¤(€€€¹½Ü€ôÑ¥µ”¹Ñ¥µ” ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰±ÉÔµÉ…”‰t€ô}¥‘±•}•Ù¥Ñ…‰±•}Í•ÍÍ¥½¸¡¹½Ü¤ðì(€€€€€€€€‰±…ÍÑ}…Ñ¥Ù”ˆè¹½Ü€´€ÈÀ€¨€ÌØÀÀ(€€€ô(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰¥‘±”µÁ••È‰t€ô}¥‘±•}•Ù¥Ñ…‰±•}Í•ÍÍ¥½¸¡¹½Ü¤((€€€ÑÉäè(€€€€€€€Í•ÉÙ•È¹}•¹™½É•}Í•ÍÍ¥½¹}…À ¤(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰±ÉÔµÉ…”‰ul‰ÑÉ…¹ÍÁ½ÉÐ‰t¥Ì±¥Ù•}ÑÉ…¹ÍÁ½ÉÐ(€€€€€€€…ÍÍ•ÉÐ€‰¥‘±”µÁ••Èˆ¹½Ð¥¸Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì(€€€€€€€…ÍÍ•ÉÐmÉ•…Í½¸™½È}Í•ÍÍ¥½¸°É•…Í½¸¥¸±½Í•‘t€ôôl‰±ÉÕ}•Ù¥Ð‰t(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}É•…Ñ•}É•½É‘Í}Õ¥}µ½‘•±}…Í}Í•ÍÍ¥½¹}½Ù•ÉÉ¥‘”¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰Q¡”‘•Í­Ñ½À½µÁ½Í•È½Ý¹Ì¥ÑÌµ½‘•°…ÌÁ±…¥¸U$ÍÑ…Ñ”…¹Í¡¥ÁÌ¥Ð½¸(€€€Í•ÍÍ¥½¸¹É•…Ñ”¸Q¡”…Ñ•Ý…äµÕÍÐÉ•½É¥Ð…Ì„AHµMMM%=8½Ù•ÉÉ¥‘”€¡‰Õ¥±Ð(€€€¥¹Ñ¼Ñ¡”…•¹Ð¤°¹•Ù•È„±½‰…°½¹™¥œÝÉ¥Ñ”ƒŠPÁ¥­¥¹œ„µ½‘•°™½È„¹•Ü¡…Ð(€€€µÕÍÐ¹½ÐµÕÑ…Ñ”Ñ¡”ÁÉ½™¥±”‘•™…Õ±Ð¸(€€€€ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•¹…‰±•}…Ñ•Ý…å}ÁÉ½µÁÑÌˆ°±…µ‰‘„è9½¹”¤(€€€€Œ½¸ÐÉÕ¸Ñ¡”É•…°‘•™•ÉÉ•‰Õ¥±¥¸Ñ¡¥ÌÍÑ½É…”µ™½ÕÍ•Ñ•ÍÐ¸(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹}µ•Ñ¡½‘Íl‰Í•ÍÍ¥½¸¹É•…Ñ”‰t (€€€€€€€€€€€€‰ÈÄˆ°(€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰½±Ìˆè€àÀ°(€€€€€€€€€€€€€€€€‰µ½‘•°ˆè€‰±…Õ‘”µÍ½¹¹•Ð´Ð¸Øˆ°(€€€€€€€€€€€€€€€€‰ÁÉ½Ù¥‘•Èˆè€‰…¹Ñ¡É½Á¥Œˆ°(€€€€€€€€€€€€€€€€‰É•…Í½¹¥¹}•™™½ÉÐˆè€‰¡¥ ˆ°(€€€€€€€€€€€€€€€€‰™…ÍÐˆèQÉÕ”°(€€€€€€€€€€€ô°(€€€€€€€€¤(€€€€€€€Í¥€ôÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰t(€€€€€€€Í•ÍÌ€ôÍ•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÍ¥‘t(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍl‰µ½‘•±}½Ù•ÉÉ¥‘”‰t€ôôì‰µ½‘•°ˆè€‰±…Õ‘”µÍ½¹¹•Ð´Ð¸Øˆ°€‰ÁÉ½Ù¥‘•Èˆè€‰…¹Ñ¡É½Á¥Œ‰ô(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍl‰É•…Ñ•}É•…Í½¹¥¹}½Ù•ÉÉ¥‘”‰t¥Ì¹½Ð9½¹”(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍl‰É•…Ñ•}Í•ÉÙ¥•}Ñ¥•É}½Ù•ÉÉ¥‘”‰t€ôô€‰ÁÉ¥½É¥Ñäˆ(€€€€€€€€ŒQ¡”¥µµ•‘¥…Ñ”É•ÍÁ½¹Í”É•™±•ÑÌÑ¡”½Ù•ÉÉ¥‘”€¡¹½ÐÑ¡”±½‰…°‘•™…Õ±Ð¤Í¼(€€€€€€€€ŒÑ¡”±¥•¹Ð¹•Ù•È±½‰‰•ÉÌ¥ÑÌÍÑ¥­äÁ¥¬‰•™½É”Ñ¡”‰Õ¥±±…¹‘Ì¸(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰¥¹™¼‰ul‰µ½‘•°‰t€ôô€‰±…Õ‘”µÍ½¹¹•Ð´Ð¸Øˆ(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰¥¹™¼‰ul‰ÁÉ½Ù¥‘•È‰t€ôô€‰…¹Ñ¡É½Á¥Œˆ((€€€€€€€€ŒáÁ±¥¥Ð™…±Í”¥Ì¹½ÐÑ¡”Í…µ”…Ì½µ¥ÍÍ¥½¸è¥ÐµÕÍÐÍÕÁÁÉ•ÍÌ„…ÍÐ(€€€€€€€€ŒÁÉ½™¥±”‘•™…Õ±Ð™½ÈÑ¡¥ÌÍ•ÍÍ¥½¸Ì™¥ÉÍÐÉ•ÅÕ•ÍÐ¸(€€€€€€€¹½Éµ…°€ôÍ•ÉÙ•È¹}µ•Ñ¡½‘Íl‰Í•ÍÍ¥½¸¹É•…Ñ”‰t (€€€€€€€€€€€€‰ÈÈˆ°ì‰½±Ìˆè€àÀ°€‰™…ÍÐˆè…±Í•ô(€€€€€€€€¤(€€€€€€€¹½Éµ…±}Í•ÍÌ€ôÍ•ÉÙ•È¹}Í•ÍÍ¥½¹Ím¹½Éµ…±l‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰ut(€€€€€€€…ÍÍ•ÉÐ¹½Éµ…±}Í•ÍÍl‰É•…Ñ•}Í•ÉÙ¥•}Ñ¥•É}½Ù•ÉÉ¥‘”‰t€ôô€ˆˆ((€€€€€€€€Œ9¼­¹½‰ÌƒŠH¹¼½Ù•ÉÉ¥‘•ÌìÑ¡”Í•ÍÍ¥½¸‰Õ¥±‘Ì™É½´Ñ¡”ÁÉ½™¥±”‘•™…Õ±Ð¸(€€€€€€€Á±…¥¸€ôÍ•ÉÙ•È¹}µ•Ñ¡½‘Íl‰Í•ÍÍ¥½¸¹É•…Ñ”‰t ‰ÈÌˆ°ì‰½±Ìˆè€àÁô¤(€€€€€€€Á±…¥¹}Í•ÍÌ€ôÍ•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÁ±…¥¹l‰É•ÍÕ±Ð‰ul‰Í•ÍÍ¥½¹}¥‰ut(€€€€€€€…ÍÍ•ÉÐÁ±…¥¹}Í•ÍÍl‰µ½‘•±}½Ù•ÉÉ¥‘”‰t¥Ì9½¹”(€€€€€€€…ÍÍ•ÉÐÁ±…¥¹}Í•ÍÍl‰É•…Ñ•}É•…Í½¹¥¹}½Ù•ÉÉ¥‘”‰t¥Ì9½¹”(€€€€€€€…ÍÍ•ÉÐÁ±…¥¹}Í•ÍÍl‰É•…Ñ•}Í•ÉÙ¥•}Ñ¥•É}½Ù•ÉÉ¥‘”‰t¥Ì9½¹”(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(()ÁåÑ•ÍÐ¹µ…É¬¹Á…É…µ•ÑÉ¥é” ‰Í•ÉÙ¥•}Ñ¥•É}½Ù•ÉÉ¥‘”ˆ°l‰ÁÉ¥½É¥Ñäˆ°€ˆ‰t¤)‘•˜Ñ•ÍÑ}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±‘}Á…ÍÍ•Í}Í•ÍÍ¥½¹}µ½‘•±}½Ù•ÉÉ¥‘” (€€€µ½¹­•åÁ…Ñ °Í•ÉÙ¥•}Ñ¥•É}½Ù•ÉÉ¥‘”(¤è(€€€€ˆˆ‰µ½‘•°ÍÑ…•½¸Ñ¡”Í•ÍÍ¥½¸€¡”¹œ¸‰äÍ•ÍÍ¥½¸¹É•…Ñ”™É½´Ñ¡”‘•Í­Ñ½À(€€€½µÁ½Í•È¤µÕÍÐÉ•… }µ…­•}…•¹ÐÍ¼Ñ¡”™¥ÉÍÐ‰Õ¥±ÉÕ¹Ì½¸¥Ð‘¥É•Ñ±äƒŠP(€€€¹¼±½‰…°½¹™¥œ°¹¼‰Õ¥±µÑ¡•¸µÍÝ¥Ñ ¸(€€€€ˆˆˆ(€€€…ÁÑÕÉ•€ôíô((€€€±…ÍÌ…­•]½É­•Èè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°€©}„°€¨©}¬¤è(€€€€€€€€€€€Á…ÍÌ((€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€Á…ÍÌ((€€€‘•˜™…­•}µ…­•}…•¹Ð¡Í¥°­•ä°Í•ÍÍ¥½¹}¥õ9½¹”°Í•ÍÍ¥½¹}‘ˆõ9½¹”°€¨©­Ý…ÉÌ¤è(€€€€€€€…ÁÑÕÉ•¹ÕÁ‘…Ñ”¡­Ý…ÉÌ¤(€€€€€€€É•ÑÕÉ¸ÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡µ½‘•°ô‰±…Õ‘”µÍ½¹¹•Ð´Ð¸Øˆ¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}Í•Ñ}Í•ÍÍ¥½¹}½¹Ñ•áÐˆ°±…µ‰‘„Ñ…É•Ð°Ýõ9½¹”èmt(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±•…É}Í•ÍÍ¥½¹}½¹Ñ•áÐˆ°±…µ‰‘„Ñ½­•¹Ìè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}µ…­•}…•¹Ðˆ°™…­•}µ…­•}…•¹Ð¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}M±…Í¡]½É­•Èˆ°…­•]½É­•È¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}…ÑÑ…¡}Ý½É­•Èˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Ý¥É•}…±±‰…­Ìˆ°±…µ‰‘„}Í¥è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„€©„°€¨©¬èíô¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}¹½Ñ¥™¥…Ñ¥½¹}Á½±±•Èˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}¹½Ñ¥™å}Í•ÍÍ¥½¹}‰½Õ¹‘…Éäˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÁÉ½‰•}½¹™¥}¡•…±Ñ ˆ°±…µ‰‘„€©}„è9½¹”¤((€€€Í¥€ô€‰‰Õ¥±µÍ¥ˆ(€€€½Ù•ÉÉ¥‘”€ôì‰µ½‘•°ˆè€‰±…Õ‘”µÍ½¹¹•Ð´Ð¸Øˆ°€‰ÁÉ½Ù¥‘•Èˆè€‰…¹Ñ¡É½Á¥Œ‰ô(€€€É•…Í½¹¥¹œ€ôì‰•¹…‰±•ˆèQÉÕ”°€‰•™™½ÉÐˆè€‰¡¥ ‰ô(€€€Í•ÍÍ¥½¸€ôì(€€€€€€€€‰…•¹Ðˆè9½¹”°(€€€€€€€€‰…•¹Ñ}É•…‘äˆèÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤°(€€€€€€€€‰Í•ÍÍ¥½¹}­•äˆè€‰¬Äˆ°(€€€€€€€€‰ÁÉ½™¥±•}¡½µ”ˆè9½¹”°(€€€€€€€€‰µ½‘•±}½Ù•ÉÉ¥‘”ˆè½Ù•ÉÉ¥‘”°(€€€€€€€€‰É•…Ñ•}É•…Í½¹¥¹}½Ù•ÉÉ¥‘”ˆèÉ•…Í½¹¥¹œ°(€€€€€€€€‰É•…Ñ•}Í•ÉÙ¥•}Ñ¥•É}½Ù•ÉÉ¥‘”ˆèÍ•ÉÙ¥•}Ñ¥•É}½Ù•ÉÉ¥‘”°(€€€ô(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÍ¥‘t€ôÍ•ÍÍ¥½¸(€€€ÑÉäè(€€€€€€€Í•ÉÙ•È¹}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±¡Í¥°Í•ÍÍ¥½¸¤(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰…•¹Ñ}É•…‘ä‰t¹Ý…¥Ð¡Ñ¥µ•½ÕÐôÌ¤°€‰…•¹Ð‰Õ¥±‘¥¹½Ð™¥¹¥Í ˆ(€€€€€€€…ÍÍ•ÉÐ…ÁÑÕÉ•¹•Ð ‰µ½‘•±}½Ù•ÉÉ¥‘”ˆ¤€ôô½Ù•ÉÉ¥‘”(€€€€€€€…ÍÍ•ÉÐ…ÁÑÕÉ•¹•Ð ‰É•…Í½¹¥¹}½¹™¥}½Ù•ÉÉ¥‘”ˆ¤€ôôÉ•…Í½¹¥¹œ(€€€€€€€…ÍÍ•ÉÐ…ÁÑÕÉ•¹•Ð ‰Í•ÉÙ¥•}Ñ¥•É}½Ù•ÉÉ¥‘”ˆ¤€ôôÍ•ÉÙ¥•}Ñ¥•É}½Ù•ÉÉ¥‘”(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰…•¹Ð‰t¹µ½‘•°€ôô€‰±…Õ‘”µÍ½¹¹•Ð´Ð¸Øˆ(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹±•…È ¤(((ŒƒŠRŠR ‰¥±±¥¹œ½ÍÕ‰ÍÉ¥ÁÑ¥½¸ÍÑ…Ñ”€¬•ÉÉ½ÈÍ•É¥…±¥é…Ñ¥½¸ƒŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠR (()‘•˜Ñ•ÍÑ}É•Í•Ñ}Í•ÍÍ¥½¹}…•¹Ñ}±•…ÉÍ}Í•ÍÍ¥½¹}½Ù•ÉÉ¥‘•Ì¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆˆ½¹•Ü¥Ì„™Õ±°½¹Ù•ÉÍ…Ñ¥½¸‰½Õ¹‘…ÉäèÍ•ÍÍ¥½¸µÍ½Á•€½µ½‘•°°€½É•…Í½¹¥¹œ°(€€€…¹€½™…ÍÐ½Ù•ÉÉ¥‘•Ì‘¼9=P…ÉÉä¥¹Ñ¼Ñ¡”™É•Í …•¹ÐƒŠP¥ÐÉ”µ‘•É¥Ù•Ì(€€€•Ù•ÉåÑ¡¥¹œ™É½´½¹™¥œ¹å…µ°€ ŒÐàÀÔÔ°€ŒÈÌÄÌÄ¤¸ˆˆˆ(€€€…ÁÑÕÉ•€ôíô(€€€¹•Ý}…•¹Ð€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡µ½‘•°ô‰½Á•¹…¤½ÁÐ´Ô¸Ðˆ°Í•ÉÙ¥•}Ñ¥•Èôˆˆ¤(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸ (€€€€€€€…•¹ÐõÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€€€€€µ½‘•°ô‰½Á•¹…¤½ÁÐ´Ô¸Ðˆ°(€€€€€€€€€€€É•…Í½¹¥¹}½¹™¥œõì‰•¹…‰±•ˆèQÉÕ”°€‰•™™½ÉÐˆè€‰¡¥ ‰ô°(€€€€€€€€€€€Í•ÉÙ¥•}Ñ¥•Èôˆˆ°(€€€€€€€€¤°(€€€€€€€µ½‘•±}½Ù•ÉÉ¥‘”õì‰µ½‘•°ˆè€‰½Á•¹…¤½ÁÐ´Ô¸Ð‰ô°(€€€€€€€É•…Ñ•}É•…Í½¹¥¹}½Ù•ÉÉ¥‘”õì‰•¹…‰±•ˆèQÉÕ”°€‰•™™½ÉÐˆè€‰¡¥ ‰ô°(€€€€€€€É•…Ñ•}Í•ÉÙ¥•}Ñ¥•É}½Ù•ÉÉ¥‘”ôˆˆ°(€€€€¤((€€€‘•˜µ…­•}…•¹Ð ©}…ÉÌ°€¨©­Ý…ÉÌ¤è(€€€€€€€…ÁÑÕÉ•¹ÕÁ‘…Ñ”¡­Ý…ÉÌ¤(€€€€€€€É•ÑÕÉ¸¹•Ý}…•¹Ð((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•Ñ}Í•ÍÍ¥½¹}½¹Ñ•áÐˆ°±…µ‰‘„}­•äèmt¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±•…É}Í•ÍÍ¥½¹}½¹Ñ•áÐˆ°±…µ‰‘„}Ñ½­•¹Ìè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}µ…­•}…•¹Ðˆ°µ…­•}…•¹Ð¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}½¹™¥}µ½‘•±}Ñ…É•Ðˆ°±…µ‰‘„è€ ˆˆ°€ˆˆ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}Í¡½Ý}É•…Í½¹¥¹œˆ°±…µ‰‘„èQÉÕ”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}Ñ½½±}ÁÉ½É•ÍÍ}µ½‘”ˆ°±…µ‰‘„è€‰…±°ˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„€©}…ÉÌèíô¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©}…ÉÌè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}É•ÍÑ…ÉÑ}Í±…Í¡}Ý½É­•Èˆ°±…µ‰‘„€©}…ÉÌè9½¹”¤((€€€Í•ÉÙ•È¹}É•Í•Ñ}Í•ÍÍ¥½¹}…•¹Ð ‰Í¥ˆ°Í•ÍÍ¥½¸¤((€€€€Œ9¼Í•ÍÍ¥½¸½Ù•ÉÉ¥‘•Ì™½ÉÝ…É‘•ƒŠP™É•Í …•¹Ð‰Õ¥±‘Ì™É½´½¹™¥œ¸(€€€…ÍÍ•ÉÐ€‰µ½‘•±}½Ù•ÉÉ¥‘”ˆ¹½Ð¥¸…ÁÑÕÉ•(€€€…ÍÍ•ÉÐ€‰É•…Í½¹¥¹}½¹™¥}½Ù•ÉÉ¥‘”ˆ¹½Ð¥¸…ÁÑÕÉ•(€€€…ÍÍ•ÉÐ€‰Í•ÉÙ¥•}Ñ¥•É}½Ù•ÉÉ¥‘”ˆ¹½Ð¥¸…ÁÑÕÉ•(€€€€Œ¹Ñ¡”Í•ÍÍ¥½¸Á¥¹Ì…É”½¹”Í¼„±…Ñ•ÈÉ•‰Õ¥±…¸ÐÉ•ÍÕÉÉ•ÐÑ¡•´¸(€€€…ÍÍ•ÉÐ€‰µ½‘•±}½Ù•ÉÉ¥‘”ˆ¹½Ð¥¸Í•ÍÍ¥½¸(€€€…ÍÍ•ÉÐ€‰É•…Ñ•}É•…Í½¹¥¹}½Ù•ÉÉ¥‘”ˆ¹½Ð¥¸Í•ÍÍ¥½¸(€€€…ÍÍ•ÉÐ€‰É•…Ñ•}Í•ÉÙ¥•}Ñ¥•É}½Ù•ÉÉ¥‘”ˆ¹½Ð¥¸Í•ÍÍ¥½¸(€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰…•¹Ð‰t¥Ì¹•Ý}…•¹Ð(()ÁåÑ•ÍÐ¹µ…É¬¹Á…É…µ•ÑÉ¥é” (€€€€‰…É±•áÁ•Ñ•ˆ°(€€€l(€€€€€€€€ ‰…¹½¹¥…°ˆ°ì‰­¥¹ˆè€‰…¹½¹¥…°‰ô¤°(€€€€€€€€ (€€€€€€€€€€€€‰‘¥ÍÑ¥¹Ðˆ°(€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰­¥¹ˆè€‰‘¥ÍÑ¥¹Ðˆ°(€€€€€€€€€€€€€€€€‰Á…åµ•¹Ñ}µ•Ñ¡½‘}¥ˆè€‰Áµ}…ÕÑ¼ˆ°(€€€€€€€€€€€€€€€€‰‰É…¹ˆè9½¹”°(€€€€€€€€€€€€€€€€‰±…ÍÐÐˆè9½¹”°(€€€€€€€€€€€ô°(€€€€€€€€¤°(€€€€€€€€ ‰¹½¹”ˆ°ì‰­¥¹ˆè€‰¹½¹”‰ô¤°(€€€t°(¤)‘•˜Ñ•ÍÑ}‰¥±±¥¹}ÍÑ…Ñ•}Í•É¥…±¥é•Í}…ÕÑ½}É•±½…‘}…É‘}Õ¹¥½¸¡µ½¹­•åÁ…Ñ °…É°•áÁ•Ñ•¤è(€€€™É½´…•¹Ð¹‰¥±±¥¹}Ù¥•Ü¥µÁ½ÉÐÕÑ½I•±½…°ÕÑ½I•±½…‘…É°	¥±±¥¹MÑ…Ñ”((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÕÍ…•}Á…å±½…ˆ°±…µ‰‘„ÍÑ…Ñ”èì‰…Ù…¥±…‰±”ˆè…±Í•ô¤(€€€…ÕÑ½}É•±½…‘}…É€ôÕÑ½I•±½…‘…É (€€€€€€€­¥¹õ…É°(€€€€€€€Á…åµ•¹Ñ}µ•Ñ¡½‘}¥ô‰Áµ}…ÕÑ¼ˆ¥˜…É€ôô€‰‘¥ÍÑ¥¹Ðˆ•±Í”9½¹”°(€€€€¤(€€€ÍÑ…Ñ”€ô	¥±±¥¹MÑ…Ñ” (€€€€€€€±½•‘}¥¸õQÉÕ”°(€€€€€€€…ÕÑ½}É•±½…õÕÑ½I•±½…¡•¹…‰±•õQÉÕ”°…Éõ…ÕÑ½}É•±½…‘}…É¤°(€€€€¤((€€€É•ÍÕ±Ð€ôÍ•ÉÙ•È¹}Í•É¥…±¥é•}‰¥±±¥¹}ÍÑ…Ñ”¡ÍÑ…Ñ”¤((€€€…ÍÍ•ÉÐÉ•ÍÕ±Ñl‰…ÕÑ½}É•±½…‰ul‰…É‰t€ôô•áÁ•Ñ•(()‘•˜Ñ•ÍÑ}‰¥±±¥¹}ÍÑ…Ñ•}Í•É¥…±¥é•Í}Í•ÉÙ•É}Á±…¹}…Á…‰¥±¥Ñä¡µ½¹­•åÁ…Ñ ¤è(€€€™É½´…•¹Ð¹‰¥±±¥¹}Ù¥•Ü¥µÁ½ÉÐ	¥±±¥¹MÑ…Ñ”((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÕÍ…•}Á…å±½…ˆ°±…µ‰‘„ÍÑ…Ñ”èì‰…Ù…¥±…‰±”ˆè…±Í•ô¤(€€€ÍÑ…Ñ”€ô	¥±±¥¹MÑ…Ñ” (€€€€€€€±½•‘}¥¸õQÉÕ”°(€€€€€€€É½±”ô‰55	Hˆ°(€€€€€€€…¹}¡…¹•}Á±…¹}É…ÜõQÉÕ”°(€€€€¤((€€€É•ÍÕ±Ð€ôÍ•ÉÙ•È¹}Í•É¥…±¥é•}‰¥±±¥¹}ÍÑ…Ñ”¡ÍÑ…Ñ”¤((€€€…ÍÍ•ÉÐÉ•ÍÕ±Ñl‰¥Í}…‘µ¥¸‰t¥Ì…±Í”(€€€…ÍÍ•ÉÐÉ•ÍÕ±Ñl‰…¹}¡…¹•}Á±…¸‰t¥ÌQÉÕ”(()±…ÍÌ}	¥±±¥¹!•…‘•ÉÌè(€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ù…±Õ•Ì¤è(€€€€€€€Í•±˜¹}Ù…±Õ•Ì€ôÙ…±Õ•Ì((€€€‘•˜•Ð¡Í•±˜°­•ä¤è(€€€€€€€É•ÑÕÉ¸Í•±˜¹}Ù…±Õ•Ì¹•Ð¡­•ä¤(()ÁåÑ•ÍÐ¹µ…É¬¹Á…É…µ•ÑÉ¥é” (€€€€‰ÍÑ…ÑÕÌ±•ÉÉ½È±É•ÑÉå}…™Ñ•Èˆ°(€€€l(€€€€€€€€ ÔÀÌ°€‰ÍÑÉ¥Á•}Õ¹…Ù…¥±…‰±”ˆ°€ÜÔ¤°(€€€€€€€€ ÐÈä°€‰ÕÁÉ…‘•}…Á}•á••‘•ˆ°9½¹”¤°(€€€€€€€€ ÐÈä°€‰É…Ñ•}±¥µ¥Ñ•ˆ°9½¹”¤°(€€€t°(¤)‘•˜Ñ•ÍÑ}‰¥±±¥¹}•ÉÉ½É}Í•É¥…±¥é…Ñ¥½¹}ÁÉ•Í•ÉÙ•Í}Í•ÉÙ•É}½‘” (€€€ÍÑ…ÑÕÌ°•ÉÉ½È°É•ÑÉå}…™Ñ•È(¤è(€€€¥µÁ½ÉÐ¡•Éµ•Í}±¤¹¹½ÕÍ}‰¥±±¥¹œ…Ì¹ˆ((€€€¡•…‘•ÉÌ€ô}	¥±±¥¹!•…‘•ÉÌ¡ì‰I•ÑÉäµ™Ñ•ÈˆèÍÑÈ¡É•ÑÉå}…™Ñ•È¥ô¤¥˜É•ÑÉå}…™Ñ•È•±Í”9½¹”(€€€Ý¥Ñ ÁåÑ•ÍÐ¹É…¥Í•Ì¡¹ˆ¹	¥±±¥¹QÉ…¹Í¥•¹Ð¤…Ì•¤è(€€€€€€€¹ˆ¹}É…¥Í•}™½É}•ÉÉ½È¡ÍÑ…ÑÕÌ°ì‰•ÉÉ½Èˆè•ÉÉ½Éô°¡•…‘•ÉÌ¤((€€€É•ÍÕ±Ð€ôÍ•ÉÙ•È¹}Í•É¥…±¥é•}‰¥±±¥¹}•ÉÉ½È¡•¤¹Ù…±Õ”¤((€€€…ÍÍ•ÉÐÉ•ÍÕ±Ñl‰•ÉÉ½È‰t€ôô•ÉÉ½È(€€€…ÍÍ•ÉÐ•¤¹Ù…±Õ”¹•ÉÉ½È€ôô•ÉÉ½È(€€€…ÍÍ•ÉÐÉ•ÍÕ±Ñl‰É•ÑÉå}…™Ñ•È‰t€ôôÉ•ÑÉå}…™Ñ•È(()‘•˜Ñ•ÍÑ}‰¥±±¥¹}É…Ñ•}±¥µ¥Ñ}Ý¥Ñ¡½ÕÑ}•ÉÉ½É}‘•™…Õ±ÑÍ}Ý¥É•}½‘” ¤è(€€€¥µÁ½ÉÐ¡•Éµ•Í}±¤¹¹½ÕÍ}‰¥±±¥¹œ…Ì¹ˆ((€€€•áŒ€ô¹ˆ¹	¥±±¥¹I…Ñ•1¥µ¥Ñ• ‰Í±½Ü‘½Ý¸ˆ°ÍÑ…ÑÕÌôÐÈä°É•ÑÉå}…™Ñ•ÈôÄÀ¤((€€€É•ÍÕ±Ð€ôÍ•ÉÙ•È¹}Í•É¥…±¥é•}‰¥±±¥¹}•ÉÉ½È¡•áŒ¤((€€€…ÍÍ•ÉÐÉ•ÍÕ±Ñl‰•ÉÉ½È‰t€ôô€‰É…Ñ•}±¥µ¥Ñ•ˆ(((ŒƒŠRŠR ÍÕ‰ÍÉ¥ÁÑ¥½¸¡…¹”IAÌ€¡XÌ¤èÁÉ•Ù¥•Ü€¬Á•¹‘¥¹œµ¡…¹”€¬ÕÁÉ…‘”ƒŠRŠR (()‘•˜}ÍÕ‰}ÉÁŒ¡µ•Ñ¡½°Á…É…µÌ¤è(€€€€ŒQ¡•Í”IAÌ…É”¥¸}1=9}!91IL€¡Á½½°µÉ½ÕÑ•ƒŠH‘¥ÍÁ…Ñ É•ÑÕÉ¹Ì9½¹”…¹Ñ¡”(€€€€ŒÝ½É­•ÈÝÉ¥Ñ•ÌÙ¥„Ñ¡”ÑÉ…¹ÍÁ½ÉÐ¤°Í¼‘É¥Ù”Ñ¡”¥¹±¥¹”¡…¹‘±•È‘¥É•Ñ±ä¸(€€€É•ÑÕÉ¸Í•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ¡ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆèµ•Ñ¡½°€‰Á…É…µÌˆèÁ…É…µÍô¥l‰É•ÍÕ±Ð‰t(()‘•˜Ñ•ÍÑ}ÍÕ‰ÍÉ¥ÁÑ¥½¹}ÁÉ•Ù¥•Ý}Í•É¥…±¥é•Í}ÅÕ½Ñ”¡µ½¹­•åÁ…Ñ ¤è(€€€¥µÁ½ÉÐ¡•Éµ•Í}±¤¹¹½ÕÍ}‰¥±±¥¹œ…Ì¹ˆ((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€¹ˆ°(€€€€€€€€‰Á½ÍÑ}ÍÕ‰ÍÉ¥ÁÑ¥½¹}ÁÉ•Ù¥•Üˆ°(€€€€€€€±…µ‰‘„ÍÕ‰ÍÉ¥ÁÑ¥½¹}ÑåÁ•}¥èì(€€€€€€€€€€€€‰•™™•Ðˆè€‰¡…É•}¹½Üˆ°(€€€€€€€€€€€€‰É•…Í½¸ˆè9½¹”°(€€€€€€€€€€€€‰ÕÉÉ•¹ÑQ¥•É%ˆè€‰Á±ÕÌˆ°(€€€€€€€€€€€€‰ÕÉÉ•¹ÑQ¥•É9…µ”ˆè€‰A±ÕÌˆ°(€€€€€€€€€€€€‰Ñ…É•ÑQ¥•É%ˆè€‰Õ±ÑÉ„ˆ°(€€€€€€€€€€€€‰Ñ…É•ÑQ¥•É9…µ”ˆè€‰U±ÑÉ„ˆ°(€€€€€€€€€€€€‰µ½¹Ñ¡±åÉ•‘¥ÑÍ•±Ñ„ˆè€ˆØÀÀÀˆ°(€€€€€€€€€€€€‰…µ½Õ¹ÑÕ•9½Ý•¹ÑÌˆè€ÄÈÌÐ°(€€€€€€€€€€€€‰•™™•Ñ¥Ù•Ðˆè9½¹”°(€€€€€€€ô°(€€€€¤(€€€É•Ì€ô}ÍÕ‰}ÉÁŒ ‰ÍÕ‰ÍÉ¥ÁÑ¥½¸¹ÁÉ•Ù¥•Üˆ°ì‰ÍÕ‰ÍÉ¥ÁÑ¥½¹}ÑåÁ•}¥ˆè€‰Õ±ÑÉ„‰ô¤(€€€…ÍÍ•ÉÐÉ•Íl‰½¬‰t¥ÌQÉÕ”(€€€…ÍÍ•ÉÐÉ•Íl‰•™™•Ð‰t€ôô€‰¡…É•}¹½Üˆ(€€€…ÍÍ•ÉÐÉ•Íl‰…µ½Õ¹Ñ}‘Õ•}¹½Ý}•¹ÑÌ‰t€ôô€ÄÈÌÐ(€€€…ÍÍ•ÉÐÉ•Íl‰Ñ…É•Ñ}Ñ¥•É}¹…µ”‰t€ôô€‰U±ÑÉ„ˆ(€€€…ÍÍ•ÉÐÉ•Íl‰µ½¹Ñ¡±å}É•‘¥ÑÍ}‘•±Ñ„‰t€ôô€ˆØÀÀÀˆ(()‘•˜Ñ•ÍÑ}ÍÕ‰ÍÉ¥ÁÑ¥½¹}ÁÉ•Ù¥•Ý}É•ÅÕ¥É•Í}Ñ¥•È ¤è(€€€É•Ì€ô}ÍÕ‰}ÉÁŒ ‰ÍÕ‰ÍÉ¥ÁÑ¥½¸¹ÁÉ•Ù¥•Üˆ°íô¤(€€€…ÍÍ•ÉÐÉ•Íl‰½¬‰t¥Ì…±Í”(€€€…ÍÍ•ÉÐÉ•Íl‰•ÉÉ½È‰t€ôô€‰¥¹Ù…±¥‘}É•ÅÕ•ÍÐˆ(()‘•˜Ñ•ÍÑ}ÍÕ‰ÍÉ¥ÁÑ¥½¹}ÁÉ•Ù¥•Ý}Í½Á•}•ÉÉ½É}µ…ÁÍ}Ñ½}ÍÑ•Á}ÕÀ¡µ½¹­•åÁ…Ñ ¤è(€€€¥µÁ½ÉÐ¡•Éµ•Í}±¤¹¹½ÕÍ}‰¥±±¥¹œ…Ì¹ˆ((€€€‘•˜}É…¥Í”¡ÍÕ‰ÍÉ¥ÁÑ¥½¹}ÑåÁ•}¥¤è(€€€€€€€É…¥Í”¹ˆ¹	¥±±¥¹M½Á•I•ÅÕ¥É• ‰‰¥±±¥¹œéµ…¹…”É•ÅÕ¥É•ˆ¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡¹ˆ°€‰Á½ÍÑ}ÍÕ‰ÍÉ¥ÁÑ¥½¹}ÁÉ•Ù¥•Üˆ°}É…¥Í”¤(€€€É•Ì€ô}ÍÕ‰}ÉÁŒ ‰ÍÕ‰ÍÉ¥ÁÑ¥½¸¹ÁÉ•Ù¥•Üˆ°ì‰ÍÕ‰ÍÉ¥ÁÑ¥½¹}ÑåÁ•}¥ˆè€‰Õ±ÑÉ„‰ô¤(€€€…ÍÍ•ÉÐÉ•Íl‰½¬‰t¥Ì…±Í”(€€€…ÍÍ•ÉÐÉ•Íl‰•ÉÉ½È‰t€ôô€‰¥¹ÍÕ™™¥¥•¹Ñ}Í½Á”ˆ(()‘•˜Ñ•ÍÑ}ÍÕ‰ÍÉ¥ÁÑ¥½¹}¡…¹•}…¹•±±…Ñ¥½¸¡µ½¹­•åÁ…Ñ ¤è(€€€¥µÁ½ÉÐ¡•Éµ•Í}±¤¹¹½ÕÍ}‰¥±±¥¹œ…Ì¹ˆ((€€€Í••¸€ôíô((€€€‘•˜}ÁÕÐ ¨°ÍÕ‰ÍÉ¥ÁÑ¥½¹}ÑåÁ•}¥õ9½¹”°…¹•°õ…±Í”¤è(€€€€€€€Í••¹l‰Ñ¥•È‰t€ôÍÕ‰ÍÉ¥ÁÑ¥½¹}ÑåÁ•}¥(€€€€€€€Í••¹l‰…¹•°‰t€ô…¹•°(€€€€€€€É•ÑÕÉ¸ì‰É…¥°ˆè€‰ÍÑÉ¥Á”ˆ°€‰…¹•±ÑA•É¥½‘¹ˆèQÉÕ”°€‰µ•ÍÍ…”ˆè€‰M¡•‘Õ±•Ñ¼…¹•°¸‰ô((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡¹ˆ°€‰ÁÕÑ}ÍÕ‰ÍÉ¥ÁÑ¥½¹}Á•¹‘¥¹}¡…¹”ˆ°}ÁÕÐ¤(€€€É•Ì€ô}ÍÕ‰}ÉÁŒ ‰ÍÕ‰ÍÉ¥ÁÑ¥½¸¹¡…¹”ˆ°ì‰…¹•°ˆèQÉÕ•ô¤(€€€…ÍÍ•ÉÐÉ•Íl‰½¬‰t¥ÌQÉÕ”(€€€…ÍÍ•ÉÐÍ••¸€ôôì‰Ñ¥•Èˆè9½¹”°€‰…¹•°ˆèQÉÕ•ô(€€€…ÍÍ•ÉÐÉ•Íl‰µ•ÍÍ…”‰t€ôô€‰M¡•‘Õ±•Ñ¼…¹•°¸ˆ(()‘•˜Ñ•ÍÑ}ÍÕ‰ÍÉ¥ÁÑ¥½¹}¡…¹•}Ñ¥•É}‘½Ý¹É…‘”¡µ½¹­•åÁ…Ñ ¤è(€€€¥µÁ½ÉÐ¡•Éµ•Í}±¤¹¹½ÕÍ}‰¥±±¥¹œ…Ì¹ˆ((€€€Í••¸€ôíô((€€€‘•˜}ÁÕÐ ¨°ÍÕ‰ÍÉ¥ÁÑ¥½¹}ÑåÁ•}¥õ9½¹”°…¹•°õ…±Í”¤è(€€€€€€€Í••¹l‰Ñ¥•È‰t€ôÍÕ‰ÍÉ¥ÁÑ¥½¹}ÑåÁ•}¥(€€€€€€€Í••¹l‰…¹•°‰t€ô…¹•°(€€€€€€€É•ÑÕÉ¸ì‰É…¥°ˆè€‰ÍÑÉ¥Á”ˆ°€‰¡…¹•QåÁ”ˆè€‰‘½Ý¹É…‘”ˆ°€‰Ñ…É•ÑQ¥•É9…µ”ˆè€‰A±ÕÌˆ°€‰µ•ÍÍ…”ˆè€‰M¡•‘Õ±•¸‰ô((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡¹ˆ°€‰ÁÕÑ}ÍÕ‰ÍÉ¥ÁÑ¥½¹}Á•¹‘¥¹}¡…¹”ˆ°}ÁÕÐ¤(€€€É•Ì€ô}ÍÕ‰}ÉÁŒ ‰ÍÕ‰ÍÉ¥ÁÑ¥½¸¹¡…¹”ˆ°ì‰ÍÕ‰ÍÉ¥ÁÑ¥½¹}ÑåÁ•}¥ˆè€‰Á±ÕÌ‰ô¤(€€€…ÍÍ•ÉÐÉ•Íl‰½¬‰t¥ÌQÉÕ”(€€€…ÍÍ•ÉÐÍ••¸€ôôì‰Ñ¥•Èˆè€‰Á±ÕÌˆ°€‰…¹•°ˆè…±Í•ô(()‘•˜Ñ•ÍÑ}ÍÕ‰ÍÉ¥ÁÑ¥½¹}¡…¹•}É•ÅÕ¥É•Í}Ñ¥•É}½É}…¹•° ¤è(€€€É•Ì€ô}ÍÕ‰}ÉÁŒ ‰ÍÕ‰ÍÉ¥ÁÑ¥½¸¹¡…¹”ˆ°íô¤(€€€…ÍÍ•ÉÐÉ•Íl‰½¬‰t¥Ì…±Í”(€€€…ÍÍ•ÉÐÉ•Íl‰•ÉÉ½È‰t€ôô€‰¥¹Ù…±¥‘}É•ÅÕ•ÍÐˆ(((()‘•˜Ñ•ÍÑ}ÍÕ‰ÍÉ¥ÁÑ¥½¹}ÕÁÉ…‘•}•¡½•Í}ÍÑ…ÑÕÍ}…¹‘}¥‘•µÁ½Ñ•¹ä¡µ½¹­•åÁ…Ñ ¤è(€€€¥µÁ½ÉÐ¡•Éµ•Í}±¤¹¹½ÕÍ}‰¥±±¥¹œ…Ì¹ˆ((€€€Í••¸€ôíô((€€€‘•˜}ÕÁÉ…‘” ¨°ÍÕ‰ÍÉ¥ÁÑ¥½¹}ÑåÁ•}¥°¥‘•µÁ½Ñ•¹å}­•ä¤è(€€€€€€€Í••¹l‰­•ä‰t€ô¥‘•µÁ½Ñ•¹å}­•ä(€€€€€€€É•ÑÕÉ¸ì‰ÍÑ…ÑÕÌˆè€‰ÕÁÉ…‘•ˆ°€‰Ñ…É•ÑQ¥•É%ˆè€‰Õ±ÑÉ„ˆ°€‰Ñ…É•ÑQ¥•É9…µ”ˆè€‰U±ÑÉ„‰ô((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡¹ˆ°€‰Á½ÍÑ}ÍÕ‰ÍÉ¥ÁÑ¥½¹}ÕÁÉ…‘”ˆ°}ÕÁÉ…‘”¤(€€€É•Ì€ô}ÍÕ‰}ÉÁŒ ‰ÍÕ‰ÍÉ¥ÁÑ¥½¸¹ÕÁÉ…‘”ˆ°ì‰ÍÕ‰ÍÉ¥ÁÑ¥½¹}ÑåÁ•}¥ˆè€‰Õ±ÑÉ„ˆ°€‰¥‘•µÁ½Ñ•¹å}­•äˆè€‰¬´Ä‰ô¤(€€€…ÍÍ•ÉÐÉ•Íl‰½¬‰t¥ÌQÉÕ”(€€€…ÍÍ•ÉÐÉ•Íl‰ÍÑ…ÑÕÌ‰t€ôô€‰ÕÁÉ…‘•ˆ(€€€…ÍÍ•ÉÐÉ•Íl‰Ñ…É•Ñ}Ñ¥•É}¹…µ”‰t€ôô€‰U±ÑÉ„ˆ(€€€…ÍÍ•ÉÐÉ•Íl‰¥‘•µÁ½Ñ•¹å}­•ä‰t€ôô€‰¬´Äˆ(€€€…ÍÍ•ÉÐÍ••¹l‰­•ä‰t€ôô€‰¬´Äˆ(()‘•˜Ñ•ÍÑ}ÍÕ‰ÍÉ¥ÁÑ¥½¹}ÕÁÉ…‘•}É•ÅÕ¥É•Í}…Ñ¥½¹}ÍÕÉ™…•Í}É•½Ù•Éä¡µ½¹­•åÁ…Ñ ¤è(€€€¥µÁ½ÉÐ¡•Éµ•Í}±¤¹¹½ÕÍ}‰¥±±¥¹œ…Ì¹ˆ((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€¹ˆ°(€€€€€€€€‰Á½ÍÑ}ÍÕ‰ÍÉ¥ÁÑ¥½¹}ÕÁÉ…‘”ˆ°(€€€€€€€±…µ‰‘„€¨°ÍÕ‰ÍÉ¥ÁÑ¥½¹}ÑåÁ•}¥°¥‘•µÁ½Ñ•¹å}­•äèì(€€€€€€€€€€€€‰ÍÑ…ÑÕÌˆè€‰É•ÅÕ¥É•Í}…Ñ¥½¸ˆ°(€€€€€€€€€€€€‰É•…Í½¸ˆè€‰…ÕÑ¡•¹Ñ¥…Ñ¥½¹}É•ÅÕ¥É•ˆ°(€€€€€€€€€€€€‰É•½Ù•ÉåUÉ°ˆè€‰¡ÑÑÁÌè¼½Á½ÉÑ…°¹•á…µÁ±”½ÍÕ‰ÍÉ¥ÁÑ¥½¸ý½É}¥õ¼ˆ°(€€€€€€€ô°(€€€€¤(€€€É•Ì€ô}ÍÕ‰}ÉÁŒ ‰ÍÕ‰ÍÉ¥ÁÑ¥½¸¹ÕÁÉ…‘”ˆ°ì‰ÍÕ‰ÍÉ¥ÁÑ¥½¹}ÑåÁ•}¥ˆè€‰Õ±ÑÉ„‰ô¤(€€€€ŒQ¡”IAÍÕ••‘ÌìÑ¡”!I¹••‘Ì€ÍLƒŠHÍÑ…ÑÕÌ€¬É•½Ù•Éå}ÕÉ°™½ÈÑ¡”Á½ÉÑ…°¸(€€€…ÍÍ•ÉÐÉ•Íl‰½¬‰t¥ÌQÉÕ”(€€€…ÍÍ•ÉÐÉ•Íl‰ÍÑ…ÑÕÌ‰t€ôô€‰É•ÅÕ¥É•Í}…Ñ¥½¸ˆ(€€€…ÍÍ•ÉÐÉ•Íl‰É•½Ù•Éå}ÕÉ°‰t¹ÍÑ…ÉÑÍÝ¥Ñ  ‰¡ÑÑÁÌè¼½Á½ÉÑ…°¹•á…µÁ±”ˆ¤(€€€…ÍÍ•ÉÐÉ•Íl‰¥‘•µÁ½Ñ•¹å}­•ä‰t€€Œµ¥¹Ñ•Ý¡•¸Ñ¡”…±±•È½µ¥ÑÌ½¹”(ŒƒŠRŠR }•Ñ}ÕÍ…”…Ñ¥Ù•}ÍÕ‰…•¹ÑÌ€¡QU$ÍÑ…ÑÕÌµ‰…ÈƒŠnL¥¹‘¥…Ñ½È¤ƒŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠR (Œ5¥ÉÉ½ÉÌÑ¡”±…ÍÍ¥Œ1$ÍÑ…ÑÕÌ‰…Èè}•Ñ}ÕÍ…”•µ‰•‘Ì„±¥Ù”½Õ¹Ð½˜(Œ‰…­É½Õ¹½…Íå¹ŒÍÕ‰…•¹ÑÌ™É½´Ñ½½±Ì¹…Íå¹}‘•±•…Ñ¥½¸¹…Ñ¥Ù•}½Õ¹Ð ¤Í¼Ñ¡”(Œ%¹¬ÍÑ…ÑÕÌ‰…È…¸É•¹‘•ÈƒŠnL8¸M½ÕÉ”½˜ÑÉÕÑ ¥ÌÑ¡”Í…µ”É•¥ÍÑÉäÑ¡”1$(ŒÉ•…‘ÌìÑ¡”™¥•±É¥‘•ÌÑ¡”•á¥ÍÑ¥¹œÁ•ÈµÕÁ‘…Ñ”ÕÍ…•€Á…å±½…¸(()±…ÍÌ}	…É••¹Ðè(€€€€ˆˆ‰•¹ÐÍÑÕˆÝ¥Ñ ¹¼½µÁÉ•ÍÍ½ÈƒŠP•á•É¥Í•ÌÑ¡”…Ñ¥Ù•}ÍÕ‰…•¹ÑÌÁ…Ñ (€€€¥¹‘•Á•¹‘•¹Ð½˜Ñ¡”¥˜½µÀé€½¹Ñ•áÐµÁ•É•¹Ð‰±½¬¸ˆˆˆ((€€€µ½‘•°€ô€‰àˆ(()‘•˜Ñ•ÍÑ}•Ñ}ÕÍ…•}Á•É™}É•…‘½ÕÑÍ}ÁÉ•Í•¹Ð ¤è(€€€€ˆˆ‰…¡•}¡¥Ñ}ÁÐ€¼…Ù}±…Ñ•¹å}Ì€¼…Ù}ÑÁÌµ¥ÉÉ½ÈÑ¡”±…ÍÍ¥Œ1$‰…È¸ˆˆˆ(€€€™É½´½±±•Ñ¥½¹Ì¥µÁ½ÉÐ‘•ÅÕ”((€€€±…ÍÌ}A•É™•¹Ðè(€€€€€€€µ½‘•°€ô€‰àˆ(€€€€€€€Í•ÍÍ¥½¹}ÁÉ½µÁÑ}Ñ½­•¹Ì€ô€ÈÝ|àÜÌ(€€€€€€€Í•ÍÍ¥½¹}…¡•}É•…‘}Ñ½­•¹Ì€ô€ÈÑ|ÌØä(€€€€€€€}…Á¥}±…Ñ•¹å}¡¥ÍÑ½Éä€ô‘•ÅÕ”¡lÈ¸Ä°€Ð¸Ít°µ…á±•¸ôÄÀ¤(€€€€€€€}…Á¥}½ÕÑÁÕÑ}¡¥ÍÑ½Éä€ô‘•ÅÕ”¡lÄÌÀ°€ÄäÁt°µ…á±•¸ôÄÀ¤((€€€ÕÍ…”€ôÍ•ÉÙ•È¹}•Ñ}ÕÍ…”¡}A•É™•¹Ð ¤¤(€€€…ÍÍ•ÉÐÕÍ…•l‰…¡•}¡¥Ñ}ÁÐ‰t€ôô€àÜ(€€€…ÍÍ•ÉÐÕÍ…•l‰…Ù}±…Ñ•¹å}Ì‰t€ôô€Ì¸È(€€€…ÍÍ•ÉÐÕÍ…•l‰…Ù}ÑÁÌ‰t€ôô€ÔÀ¸À€€ŒÑÉÕ”Ñ¡É½Õ¡ÁÕÐÍÕ´¡½ÕÐ¤½ÍÕ´¡±…Ð¤°¹½Ðµ•…¸½˜É…Ñ¥½Ì(()‘•˜Ñ•ÍÑ}•Ñ}ÕÍ…•}Á•É™}É•…‘½ÕÑÍ}½µ¥ÑÑ•‘}Ý¥Ñ¡½ÕÑ}‘…Ñ„ ¤è(€€€€ˆˆ‰i•É¼…¡”É•…‘Ì€¼•µÁÑä¡¥ÍÑ½Éä½µ¥ÐÑ¡”­•åÌƒŠP¹•Ù•È™…‰É¥…Ñ”€ÁÌ¸ˆˆˆ((€€€±…ÍÌ}½±‘•¹Ðè(€€€€€€€µ½‘•°€ô€‰àˆ(€€€€€€€Í•ÍÍ¥½¹}ÁÉ½µÁÑ}Ñ½­•¹Ì€ô€ÄÀÀ(€€€€€€€Í•ÍÍ¥½¹}…¡•}É•…‘}Ñ½­•¹Ì€ô€À((€€€ÕÍ…”€ôÍ•ÉÙ•È¹}•Ñ}ÕÍ…”¡}½±‘•¹Ð ¤¤(€€€…ÍÍ•ÉÐ€‰…¡•}¡¥Ñ}ÁÐˆ¹½Ð¥¸ÕÍ…”(€€€…ÍÍ•ÉÐ€‰…Ù}±…Ñ•¹å}Ìˆ¹½Ð¥¸ÕÍ…”(€€€…ÍÍ•ÉÐ€‰…Ù}ÑÁÌˆ¹½Ð¥¸ÕÍ…”(()‘•˜Ñ•ÍÑ}•Ñ}ÕÍ…•}Á•É™}É•…‘½ÕÑÍ}Õ…É‘}¹•…Ñ¥Ù•}±…Ñ•¹ä ¤è(€€€€ˆˆ‰=‘ÁÉ½Ù¥‘•ÈÑ¥µ¥¹Ì€¡¹•…Ñ¥Ù”‘ÕÉ…Ñ¥½¹ÌÍ••¸¥¸±½Ì¤…É”‘É½ÁÁ•¸ˆˆˆ(€€€™É½´½±±•Ñ¥½¹Ì¥µÁ½ÉÐ‘•ÅÕ”((€€€±…ÍÌ}]•¥É‘•¹Ðè(€€€€€€€µ½‘•°€ô€‰àˆ(€€€€€€€}…Á¥}±…Ñ•¹å}¡¥ÍÑ½Éä€ô‘•ÅÕ”¡l´À¸át°µ…á±•¸ôÄÀ¤(€€€€€€€}…Á¥}½ÕÑÁÕÑ}¡¥ÍÑ½Éä€ô‘•ÅÕ”¡lÄÀÁt°µ…á±•¸ôÄÀ¤((€€€ÕÍ…”€ôÍ•ÉÙ•È¹}•Ñ}ÕÍ…”¡}]•¥É‘•¹Ð ¤¤(€€€…ÍÍ•ÉÐ€‰…Ù}±…Ñ•¹å}Ìˆ¹½Ð¥¸ÕÍ…”(€€€…ÍÍ•ÉÐ€‰…Ù}ÑÁÌˆ¹½Ð¥¸ÕÍ…”(()‘•˜Ñ•ÍÑ}•Ñ}ÕÍ…•}¥¹±Õ‘•Í}…Ñ¥Ù•}ÍÕ‰…•¹ÑÌ¡µ½¹­•åÁ…Ñ ¤è(€€€¥µÁ½ÉÐÑ½½±Ì¹…Íå¹}‘•±•…Ñ¥½¸…Ì…‘}µ½(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡…‘}µ½°€‰…Ñ¥Ù•}½Õ¹Ðˆ°±…µ‰‘„è€Ð¤(€€€ÕÍ…”€ôÍ•ÉÙ•È¹}•Ñ}ÕÍ…”¡}	…É••¹Ð ¤¤(€€€…ÍÍ•ÉÐÕÍ…•l‰…Ñ¥Ù•}ÍÕ‰…•¹ÑÌ‰t€ôô€Ð(((()‘•˜Ñ•ÍÑ}•Ñ}ÕÍ…•}Í…™•}Ý¡•¹}…Ñ¥Ù•}½Õ¹Ñ}É…¥Í•Ì¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰É…¥Í¥¹œ…Ñ¥Ù•}½Õ¹Ð ¤µÕÍÐ¹½Ð‰É•…¬Ñ¡”ÕÍ…”Á…å±½…¸ˆˆˆ(€€€¥µÁ½ÉÐÑ½½±Ì¹…Íå¹}‘•±•…Ñ¥½¸…Ì…‘}µ½((€€€‘•˜}‰½½´ ¤è(€€€€€€€É…¥Í”IÕ¹Ñ¥µ•ÉÉ½È ‰‰½½´ˆ¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡…‘}µ½°€‰…Ñ¥Ù•}½Õ¹Ðˆ°}‰½½´¤(€€€ÕÍ…”€ôÍ•ÉÙ•È¹}•Ñ}ÕÍ…”¡}	…É••¹Ð ¤¤(€€€€Œ¥•±½µ¥ÑÑ•°‰ÕÐÑ¡”É•ÍÐ½˜Ñ¡”Á…å±½…¥Ì¥¹Ñ…Ð¸(€€€…ÍÍ•ÉÐ€‰…Ñ¥Ù•}ÍÕ‰…•¹ÑÌˆ¹½Ð¥¸ÕÍ…”(€€€…ÍÍ•ÉÐÕÍ…•l‰µ½‘•°‰t€ôô€‰àˆ(((Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(Œ}É•Í½±Ù•}ÉÕ¹Ñ¥µ•}Ý¥Ñ¡}™…±±‰…¬ƒŠP¥¹¥ÐµÑ¥µ”ÁÉ½Ù¥‘•È™…±±‰…¬(Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´()±…ÍÌQ•ÍÑI•Í½±Ù•IÕ¹Ñ¥µ•]¥Ñ¡…±±‰…¬è(€€€€ˆˆ‰Q•ÍÑÌ™½È}É•Í½±Ù•}ÉÕ¹Ñ¥µ•}Ý¥Ñ¡}™…±±‰…¬ ¤è¥¹¥ÐµÑ¥µ”ÁÉ½Ù¥‘•È(€€€™…±±‰…¬Ý¡•¸Ñ¡”ÁÉ¥µ…ÉäÁÉ½Ù¥‘•ÈÉ…¥Í•ÌÕÑ¡ÉÉ½È¸ˆˆˆ((€€€‘•˜Ñ•ÍÑ}ÁÉ¥µ…Éå}ÍÕ•ÍÍ}É•ÑÕÉ¹Í}ÉÕ¹Ñ¥µ”¡Í•±˜°µ½¹­•åÁ…Ñ ¤è(€€€€€€€€ˆˆ‰]¡•¸ÁÉ¥µ…ÉäÉ•Í½±Ù”ÍÕ••‘Ì°É•ÑÕÉ¸¥ÑÌÉ•ÍÕ±Ð‘¥É•Ñ±ä¸ˆˆˆ(€€€€€€€•áÁ•Ñ•€ôì‰ÁÉ½Ù¥‘•Èˆè€‰½Á•¹…¤ˆ°€‰…Á¥}­•äˆè€‰Ñ½¬‰ô(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€€‰¡•Éµ•Í}±¤¹ÉÕ¹Ñ¥µ•}ÁÉ½Ù¥‘•È¹É•Í½±Ù•}ÉÕ¹Ñ¥µ•}ÁÉ½Ù¥‘•Èˆ°(€€€€€€€€€€€±…µ‰‘„€¨©­Üè•áÁ•Ñ•°(€€€€€€€€¤(€€€€€€€É•Í½±ÕÑ¥½¸€ôÍ•ÉÙ•È¹}É•Í½±Ù•}ÉÕ¹Ñ¥µ•}Ý¥Ñ¡}™…±±‰…¬ (€€€€€€€€€€€ì‰É•ÅÕ•ÍÑ•ˆè€‰½Á•¹…¤‰ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•Í½±ÕÑ¥½¸¹ÉÕ¹Ñ¥µ”€ôô•áÁ•Ñ•(€€€€€€€…ÍÍ•ÉÐÉ•Í½±ÕÑ¥½¸¹Í•±•Ñ•‘}µ½‘•°¥Ì9½¹”(€€€€€€€…ÍÍ•ÉÐÉ•Í½±ÕÑ¥½¸¹ÕÍ•‘}™…±±‰…¬¥Ì…±Í”((€€€‘•˜Ñ•ÍÑ}…ÕÑ¡}•ÉÉ½É}ÑÉ¥•Í}™…±±‰…­}¡…¥¸¡Í•±˜°µ½¹­•åÁ…Ñ ¤è(€€€€€€€€ˆˆ‰=¸ÕÑ¡ÉÉ½È™É½´ÁÉ¥µ…Éä°Ý…±¬™…±±‰…­}ÁÉ½Ù¥‘•ÉÌ¡…¥¸¸ˆˆˆ(€€€€€€€™É½´¡•Éµ•Í}±¤¹…ÕÑ ¥µÁ½ÉÐÕÑ¡ÉÉ½È((€€€€€€€™…±±‰…­}ÉÕ¹Ñ¥µ”€ôì‰ÁÉ½Ù¥‘•Èˆè€‰‘••ÁÍ••¬ˆ°€‰…Á¥}­•äˆè€‰™ˆµÑ½¬‰ô((€€€€€€€‘•˜™…­•}É•Í½±Ù” ¨©­Ý…ÉÌ¤è(€€€€€€€€€€€¥˜­Ý…ÉÌ¹•Ð ‰É•ÅÕ•ÍÑ•ˆ¤€ôô€‰½Á•¹…¤µ½‘•àˆè(€€€€€€€€€€€€€€€É…¥Í”ÕÑ¡ÉÉ½È ‰9¼½‘•àÉ•‘•¹Ñ¥…±ÌÍÑ½É•ˆ¤(€€€€€€€€€€€É•ÑÕÉ¸™…±±‰…­}ÉÕ¹Ñ¥µ”((€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€€‰¡•Éµ•Í}±¤¹ÉÕ¹Ñ¥µ•}ÁÉ½Ù¥‘•È¹É•Í½±Ù•}ÉÕ¹Ñ¥µ•}ÁÉ½Ù¥‘•Èˆ°(€€€€€€€€€€€™…­•}É•Í½±Ù”°(€€€€€€€€¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€Í•ÉÙ•È°(€€€€€€€€€€€€‰}±½…‘}™…±±‰…­}µ½‘•°ˆ°(€€€€€€€€€€€±…µ‰‘„èmì‰ÁÉ½Ù¥‘•Èˆè€‰‘••ÁÍ••¬ˆ°€‰µ½‘•°ˆè€‰‘••ÁÍ••¬µØÐµÁÉ¼‰õt°(€€€€€€€€¤(€€€€€€€É•Í½±ÕÑ¥½¸€ôÍ•ÉÙ•È¹}É•Í½±Ù•}ÉÕ¹Ñ¥µ•}Ý¥Ñ¡}™…±±‰…¬ (€€€€€€€€€€€ì‰É•ÅÕ•ÍÑ•ˆè€‰½Á•¹…¤µ½‘•à‰ô°(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•Í½±ÕÑ¥½¸¹ÉÕ¹Ñ¥µ”€ôô™…±±‰…­}ÉÕ¹Ñ¥µ”(€€€€€€€…ÍÍ•ÉÐÉ•Í½±ÕÑ¥½¸¹Í•±•Ñ•‘}µ½‘•°€ôô€‰‘••ÁÍ••¬µØÐµÁÉ¼ˆ(€€€€€€€…ÍÍ•ÉÐÉ•Í½±ÕÑ¥½¸¹ÕÍ•‘}™…±±‰…¬¥ÌQÉÕ”(((€€€‘•˜Ñ•ÍÑ}…ÕÑ¡}•ÉÉ½É}Í­¥ÁÍ}ÁÉ½Ù¥‘•É}½¹±å}™…±±‰…¬¡Í•±˜°µ½¹­•åÁ…Ñ ¤è(€€€€€€€€ˆˆ‰ÕÑ ™…±±‰…¬É•ÅÕ¥É•Ì½¹”½µÁ±•Ñ”ÁÉ½Ù¥‘•È½µ½‘•°Á…¥È¸ˆˆˆ(€€€€€€€™É½´¡•Éµ•Í}±¤¹…ÕÑ ¥µÁ½ÉÐÕÑ¡ÉÉ½È((€€€€€€€É•ÅÕ•ÍÑ•€ômt(€€€€€€€™…±±‰…­}ÉÕ¹Ñ¥µ”€ôì‰ÁÉ½Ù¥‘•Èˆè€‰½Á•¹É½ÕÑ•Èˆ°€‰…Á¥}­•äˆè€‰™ˆµÑ½¬‰ô((€€€€€€€‘•˜™…­•}É•Í½±Ù” ¨©­Ý…ÉÌ¤è(€€€€€€€€€€€É•ÅÕ•ÍÑ•¹…ÁÁ•¹¡­Ý…ÉÌ¹•Ð ‰É•ÅÕ•ÍÑ•ˆ¤¤(€€€€€€€€€€€¥˜­Ý…ÉÌ¹•Ð ‰É•ÅÕ•ÍÑ•ˆ¤€ôô€‰½Á•¹…¤µ½‘•àˆè(€€€€€€€€€€€€€€€É…¥Í”ÕÑ¡ÉÉ½È ‰9¼½‘•àÉ•‘•¹Ñ¥…±ÌÍÑ½É•ˆ¤(€€€€€€€€€€€É•ÑÕÉ¸™…±±‰…­}ÉÕ¹Ñ¥µ”((€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€€‰¡•Éµ•Í}±¤¹ÉÕ¹Ñ¥µ•}ÁÉ½Ù¥‘•È¹É•Í½±Ù•}ÉÕ¹Ñ¥µ•}ÁÉ½Ù¥‘•Èˆ°(€€€€€€€€€€€™…­•}É•Í½±Ù”°(€€€€€€€€¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€Í•ÉÙ•È°(€€€€€€€€€€€€‰}±½…‘}™…±±‰…­}µ½‘•°ˆ°(€€€€€€€€€€€±…µ‰‘„èl(€€€€€€€€€€€€€€€ì‰ÁÉ½Ù¥‘•Èˆè€‰…¹Ñ¡É½Á¥Œ‰ô°(€€€€€€€€€€€€€€€ì‰ÁÉ½Ù¥‘•Èˆè€‰½Á•¹É½ÕÑ•Èˆ°€‰µ½‘•°ˆè€‰èµ…¤½±´´Ô¸È‰ô°(€€€€€€€€€€€t°(€€€€€€€€¤((€€€€€€€É•Í½±ÕÑ¥½¸€ôÍ•ÉÙ•È¹}É•Í½±Ù•}ÉÕ¹Ñ¥µ•}Ý¥Ñ¡}™…±±‰…¬ (€€€€€€€€€€€ì‰É•ÅÕ•ÍÑ•ˆè€‰½Á•¹…¤µ½‘•à‰ô(€€€€€€€€¤((€€€€€€€…ÍÍ•ÉÐÉ•ÅÕ•ÍÑ•€ôôl‰½Á•¹…¤µ½‘•àˆ°€‰½Á•¹É½ÕÑ•È‰t(€€€€€€€…ÍÍ•ÉÐÉ•Í½±ÕÑ¥½¸¹ÉÕ¹Ñ¥µ”€ôô™…±±‰…­}ÉÕ¹Ñ¥µ”(€€€€€€€…ÍÍ•ÉÐÉ•Í½±ÕÑ¥½¸¹Í•±•Ñ•‘}µ½‘•°€ôô€‰èµ…¤½±´´Ô¸Èˆ(€€€€€€€…ÍÍ•ÉÐÉ•Í½±ÕÑ¥½¸¹ÕÍ•‘}™…±±‰…¬¥ÌQÉÕ”((€€€‘•˜Ñ•ÍÑ}™…±±‰…­}•¹ÑÉå}­•å}•¹Ù}É•Í½±Ù•Í}…Á¥}­•ä¡Í•±˜°µ½¹­•åÁ…Ñ ¤è(€€€€€€€€ˆˆ‰™…±±‰…¬•¹ÑÉä¹…µ¥¹œ¥ÑÌ­•äÙ¥„­•å}•¹ØÁ…ÍÍ•ÌÑ¡”É•Í½±Ù•(€€€€€€€•¹ØÙ…±Õ”…Ì•áÁ±¥¥Ñ}…Á¥}­•ä€ ŒÐÌàØÄ°YÉÑá=µ•„¤¸ˆˆˆ(€€€€€€€™É½´¡•Éµ•Í}±¤¹…ÕÑ ¥µÁ½ÉÐÕÑ¡ÉÉ½È((€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰	}QMQ}-dˆ°€‰•¹ØµÉ•Í½±Ù•µ­•äˆ¤(€€€€€€€…ÁÑÕÉ•€ôíô(€€€€€€€™…±±‰…­}ÉÕ¹Ñ¥µ”€ôì‰ÁÉ½Ù¥‘•Èˆè€‰½Á•¹É½ÕÑ•Èˆ°€‰…Á¥}­•äˆè€‰à‰ô((€€€€€€€‘•˜™…­•}É•Í½±Ù” ¨©­Ý…ÉÌ¤è(€€€€€€€€€€€¥˜­Ý…ÉÌ¹•Ð ‰É•ÅÕ•ÍÑ•ˆ¤€ôô€‰½Á•¹…¤µ½‘•àˆè(€€€€€€€€€€€€€€€É…¥Í”ÕÑ¡ÉÉ½È ‰9¼½‘•àÉ•‘•¹Ñ¥…±ÌÍÑ½É•ˆ¤(€€€€€€€€€€€…ÁÑÕÉ•¹ÕÁ‘…Ñ”¡­Ý…ÉÌ¤(€€€€€€€€€€€É•ÑÕÉ¸™…±±‰…­}ÉÕ¹Ñ¥µ”((€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€€‰¡•Éµ•Í}±¤¹ÉÕ¹Ñ¥µ•}ÁÉ½Ù¥‘•È¹É•Í½±Ù•}ÉÕ¹Ñ¥µ•}ÁÉ½Ù¥‘•Èˆ°(€€€€€€€€€€€™…­•}É•Í½±Ù”°(€€€€€€€€¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€Í•ÉÙ•È°(€€€€€€€€€€€€‰}±½…‘}™…±±‰…­}µ½‘•°ˆ°(€€€€€€€€€€€±…µ‰‘„èl(€€€€€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€€€€€‰ÁÉ½Ù¥‘•Èˆè€‰½Á•¹É½ÕÑ•Èˆ°(€€€€€€€€€€€€€€€€€€€€‰µ½‘•°ˆè€‰èµ…¤½±´´Ô¸Èˆ°(€€€€€€€€€€€€€€€€€€€€‰­•å}•¹Øˆè€‰	}QMQ}-dˆ°(€€€€€€€€€€€€€€€ô(€€€€€€€€€€€t°(€€€€€€€€¤(€€€€€€€É•Í½±ÕÑ¥½¸€ôÍ•ÉÙ•È¹}É•Í½±Ù•}ÉÕ¹Ñ¥µ•}Ý¥Ñ¡}™…±±‰…¬ (€€€€€€€€€€€ì‰É•ÅÕ•ÍÑ•ˆè€‰½Á•¹…¤µ½‘•à‰ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•Í½±ÕÑ¥½¸¹ÕÍ•‘}™…±±‰…¬¥ÌQÉÕ”(€€€€€€€…ÍÍ•ÉÐ…ÁÑÕÉ•¹•Ð ‰•áÁ±¥¥Ñ}…Á¥}­•äˆ¤€ôô€‰•¹ØµÉ•Í½±Ù•µ­•äˆ((€€€‘•˜Ñ•ÍÑ}…ÕÑ¡}•ÉÉ½É}…±±}™…±±‰…­Í}™…¥±}É…¥Í•Ì¡Í•±˜°µ½¹­•åÁ…Ñ ¤è(€€€€€€€€ˆˆ‰]¡•¸…±°™…±±‰…­Ì…±Í¼™…¥°°É”µÉ…¥Í”Ñ¡”½É¥¥¹…°ÕÑ¡ÉÉ½È¸ˆˆˆ(€€€€€€€™É½´¡•Éµ•Í}±¤¹…ÕÑ ¥µÁ½ÉÐÕÑ¡ÉÉ½È((€€€€€€€‘•˜™…­•}É•Í½±Ù” ¨©­Ý…ÉÌ¤è(€€€€€€€€€€€É…¥Í”ÕÑ¡ÉÉ½È ‰9¼É•‘•¹Ñ¥…±Ì™½È€ˆ€¬ÍÑÈ¡­Ý…ÉÌ¹•Ð ‰É•ÅÕ•ÍÑ•ˆ¤¤¤((€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€€‰¡•Éµ•Í}±¤¹ÉÕ¹Ñ¥µ•}ÁÉ½Ù¥‘•È¹É•Í½±Ù•}ÉÕ¹Ñ¥µ•}ÁÉ½Ù¥‘•Èˆ°(€€€€€€€€€€€™…­•}É•Í½±Ù”°(€€€€€€€€¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€Í•ÉÙ•È°(€€€€€€€€€€€€‰}±½…‘}™…±±‰…­}µ½‘•°ˆ°(€€€€€€€€€€€±…µ‰‘„èmì‰ÁÉ½Ù¥‘•Èˆè€‰‘••ÁÍ••¬ˆ°€‰µ½‘•°ˆè€‰‘••ÁÍ••¬µØÐµÁÉ¼‰õt°(€€€€€€€€¤(€€€€€€€¥µÁ½ÉÐÁåÑ•ÍÐ((€€€€€€€Ý¥Ñ ÁåÑ•ÍÐ¹É…¥Í•Ì¡ÕÑ¡ÉÉ½È°µ…Ñ ô‰9¼É•‘•¹Ñ¥…±Ì™½È½Á•¹…¤µ½‘•àˆ¤è(€€€€€€€€€€€Í•ÉÙ•È¹}É•Í½±Ù•}ÉÕ¹Ñ¥µ•}Ý¥Ñ¡}™…±±‰…¬ (€€€€€€€€€€€€€€€ì‰É•ÅÕ•ÍÑ•ˆè€‰½Á•¹…¤µ½‘•à‰ô°(€€€€€€€€€€€€¤((€€€‘•˜Ñ•ÍÑ}…ÕÑ¡}•ÉÉ½É}Í­¥ÁÍ}¹½¹}‘¥Ñ}•¹ÑÉ¥•Ì¡Í•±˜°µ½¹­•åÁ…Ñ ¤è(€€€€€€€€ˆˆ‰…±±‰…¬¡…¥¸•¹ÑÉ¥•ÌÑ¡…Ð…É”¹½Ð‘¥ÑÌ…É”Í­¥ÁÁ•¸ˆˆˆ(€€€€€€€™É½´¡•Éµ•Í}±¤¹…ÕÑ ¥µÁ½ÉÐÕÑ¡ÉÉ½È((€€€€€€€™…±±‰…­}ÉÕ¹Ñ¥µ”€ôì‰ÁÉ½Ù¥‘•Èˆè€‰…¹Ñ¡É½Á¥Œˆ°€‰…Á¥}­•äˆè€‰…¹ÐµÑ½¬‰ô((€€€€€€€‘•˜™…­•}É•Í½±Ù” ¨©­Ý…ÉÌ¤è(€€€€€€€€€€€¥˜­Ý…ÉÌ¹•Ð ‰É•ÅÕ•ÍÑ•ˆ¤€ôô€‰½Á•¹…¤µ½‘•àˆè(€€€€€€€€€€€€€€€É…¥Í”ÕÑ¡ÉÉ½È ‰9¼½‘•àÉ•‘•¹Ñ¥…±ÌÍÑ½É•ˆ¤(€€€€€€€€€€€É•ÑÕÉ¸™…±±‰…­}ÉÕ¹Ñ¥µ”((€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€€‰¡•Éµ•Í}±¤¹ÉÕ¹Ñ¥µ•}ÁÉ½Ù¥‘•È¹É•Í½±Ù•}ÉÕ¹Ñ¥µ•}ÁÉ½Ù¥‘•Èˆ°(€€€€€€€€€€€™…­•}É•Í½±Ù”°(€€€€€€€€¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€Í•ÉÙ•È°(€€€€€€€€€€€€‰}±½…‘}™…±±‰…­}µ½‘•°ˆ°(€€€€€€€€€€€±…µ‰‘„èl(€€€€€€€€€€€€€€€€‰¥¹Ù…±¥µÍÑÉ¥¹œµ•¹ÑÉäˆ°(€€€€€€€€€€€€€€€ì‰ÁÉ½Ù¥‘•Èˆè€‰…¹Ñ¡É½Á¥Œˆ°€‰µ½‘•°ˆè€‰±…Õ‘”µÍ½¹¹•Ð´Ð´Ø‰ô°(€€€€€€€€€€€t°(€€€€€€€€¤(€€€€€€€É•Í½±ÕÑ¥½¸€ôÍ•ÉÙ•È¹}É•Í½±Ù•}ÉÕ¹Ñ¥µ•}Ý¥Ñ¡}™…±±‰…¬ (€€€€€€€€€€€ì‰É•ÅÕ•ÍÑ•ˆè€‰½Á•¹…¤µ½‘•à‰ô°(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•Í½±ÕÑ¥½¸¹ÉÕ¹Ñ¥µ”€ôô™…±±‰…­}ÉÕ¹Ñ¥µ”(€€€€€€€…ÍÍ•ÉÐÉ•Í½±ÕÑ¥½¸¹Í•±•Ñ•‘}µ½‘•°€ôô€‰±…Õ‘”µÍ½¹¹•Ð´Ð´Øˆ(€€€€€€€…ÍÍ•ÉÐÉ•Í½±ÕÑ¥½¸¹ÕÍ•‘}™…±±‰…¬¥ÌQÉÕ”((€€€‘•˜Ñ•ÍÑ}µ…­•}…•¹Ñ}ÕÍ•Í}™…±±‰…­}½¹}…ÕÑ¡}•ÉÉ½È¡Í•±˜°µ½¹­•åÁ…Ñ ¤è(€€€€€€€€ˆˆ‰%¹Ñ•É…Ñ¥½¸è}µ…­•}…•¹Ð™…±±Ì‰…¬Ñ¼½¹™¥ÕÉ•™…±±‰…¬(€€€€€€€ÁÉ½Ù¥‘•ÈÝ¡•¸Ñ¡”ÁÉ¥µ…ÉäÁÉ½Ù¥‘•ÈÉ…¥Í•ÌÕÑ¡ÉÉ½È¸ˆˆˆ(€€€€€€€¥µÁ½ÉÐÑåÁ•Ì((€€€€€€€™É½´¡•Éµ•Í}±¤¹…ÕÑ ¥µÁ½ÉÐÕÑ¡ÉÉ½È((€€€€€€€…ÁÑÕÉ•€ôíô(€€€€€€€™…±±‰…­}ÉÕ¹Ñ¥µ”€ôì(€€€€€€€€€€€€‰ÁÉ½Ù¥‘•Èˆè€‰‘••ÁÍ••¬ˆ°(€€€€€€€€€€€€‰…Á¥}­•äˆè€‰™ˆµÑ½¬ˆ°(€€€€€€€€€€€€‰‰…Í•}ÕÉ°ˆè€‰¡ÑÑÁÌè¼½™…±±‰…¬¹¥¹Ù…±¥½ØÄˆ°(€€€€€€€ô((€€€€€€€‘•˜™…­•}É•Í½±Ù” ¨©­Ý…ÉÌ¤è(€€€€€€€€€€€¥˜­Ý…ÉÌ¹•Ð ‰É•ÅÕ•ÍÑ•ˆ¤¥¸€¡9½¹”°€‰½Á•¹…¤µ½‘•àˆ¤è(€€€€€€€€€€€€€€€É…¥Í”ÕÑ¡ÉÉ½È ‰9¼½‘•àÉ•‘•¹Ñ¥…±ÌÍÑ½É•ˆ¤(€€€€€€€€€€€É•ÑÕÉ¸™…±±‰…­}ÉÕ¹Ñ¥µ”((€€€€€€€‘•˜™…­•}…•¹Ð ¨©­Ý…ÉÌ¤è(€€€€€€€€€€€…ÁÑÕÉ•¹ÕÁ‘…Ñ”¡­Ý…ÉÌ¤(€€€€€€€€€€€É•ÑÕÉ¸ÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡µ½‘•°õ­Ý…ÉÌ¹•Ð ‰µ½‘•°ˆ¤¤((€€€€€€€µ½¹­•åÁ…Ñ ¹‘•±•¹Ø ‰!I5M}5=0ˆ°É…¥Í¥¹œõ…±Í”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹‘•±•¹Ø ‰!I5M}%9I9}5=0ˆ°É…¥Í¥¹œõ…±Í”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹‘•±•¹Ø ‰!I5M}QU%}AI=Y%Hˆ°É…¥Í¥¹œõ…±Í”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€Í•ÉÙ•È°(€€€€€€€€€€€€‰}±½…‘}™œˆ°(€€€€€€€€€€€±…µ‰‘„èì(€€€€€€€€€€€€€€€€‰µ½‘•°ˆèì‰‘•™…Õ±Ðˆè€‰ÁÐ´Ô¸Ôˆ°€‰ÁÉ½Ù¥‘•Èˆè€‰½Á•¹…¤µ½‘•à‰ô°(€€€€€€€€€€€€€€€€‰™…±±‰…­}ÁÉ½Ù¥‘•ÉÌˆèl(€€€€€€€€€€€€€€€€€€€ì‰ÁÉ½Ù¥‘•Èˆè€‰‘••ÁÍ••¬ˆ°€‰µ½‘•°ˆè€‰‘••ÁÍ••¬µØÐµÁÉ¼‰ô°(€€€€€€€€€€€€€€€t°(€€€€€€€€€€€ô°(€€€€€€€€¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€€‰¡•Éµ•Í}±¤¹ÉÕ¹Ñ¥µ•}ÁÉ½Ù¥‘•È¹É•Í½±Ù•}ÉÕ¹Ñ¥µ•}ÁÉ½Ù¥‘•Èˆ°(€€€€€€€€€€€™…­•}É•Í½±Ù”°(€€€€€€€€¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰ÉÕ¹}…•¹Ð¹%•¹Ðˆ°™…­•}…•¹Ð¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}•¹…‰±•‘}Ñ½½±Í•ÑÌˆ°±…µ‰‘„€©}„°€¨©}­Üèl‰™¥±”‰t¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è9½¹”¤((€€€€€€€…•¹Ð€ôÍ•ÉÙ•È¹}µ…­•}…•¹Ð (€€€€€€€€€€€€‰Í¥ˆ°(€€€€€€€€€€€€‰Í•ÍÍ¥½¸µ­•äˆ°(€€€€€€€€€€€µ½‘•±}½Ù•ÉÉ¥‘”õì(€€€€€€€€€€€€€€€€‰µ½‘•°ˆè€‰ÁÐ´Ô¸Ôˆ°(€€€€€€€€€€€€€€€€‰ÁÉ½Ù¥‘•Èˆè€‰½Á•¹…¤µ½‘•àˆ°(€€€€€€€€€€€€€€€€‰‰…Í•}ÕÉ°ˆè€‰¡ÑÑÁÌè¼½¡…ÑÁÐ¹½´½‰…­•¹µ…Á¤½½‘•àˆ°(€€€€€€€€€€€€€€€€‰…Á¥}­•äˆè€‰ÍÑ…±”µ½‘•àµÑ½­•¸ˆ°(€€€€€€€€€€€ô°(€€€€€€€€¤((€€€€€€€…ÍÍ•ÉÐ…•¹Ð¹µ½‘•°€ôô€‰‘••ÁÍ••¬µØÐµÁÉ¼ˆ(€€€€€€€…ÍÍ•ÉÐ…ÁÑÕÉ•‘l‰ÁÉ½Ù¥‘•È‰t€ôô€‰‘••ÁÍ••¬ˆ(€€€€€€€…ÍÍ•ÉÐ…ÁÑÕÉ•‘l‰‰…Í•}ÕÉ°‰t€ôô€‰¡ÑÑÁÌè¼½™…±±‰…¬¹¥¹Ù…±¥½ØÄˆ(€€€€€€€…ÍÍ•ÉÐ…ÁÑÕÉ•‘l‰…Á¥}­•ä‰t€ôô€‰™ˆµÑ½¬ˆ(()‘•˜Ñ•ÍÑ}•Ñ}ÕÍ…•}‘½•Í}¹½Ñ}ÍÕ‰ÍÑ¥ÑÕÑ•}ÕµÕ±…Ñ¥Ù•}Ñ½Ñ…±}™½É}½¹Ñ•áÑ}ÕÍ• ¤è(€€€€ˆˆ‰¸•áÑ•É¹…°½¹Ñ•áÐ•¹¥¹”Ñ¡…Ð‘½•Ì¹½ÐÉ•Á½ÉÐ±…ÍÑ}ÁÉ½µÁÑ}Ñ½­•¹ÌµÕÍÐ(€€€¹½Ð¡…Ù”Ñ¡”ÕµÕ±…Ñ¥Ù”±¥™•Ñ¥µ”Í•ÍÍ¥½¹}Ñ½Ñ…±}Ñ½­•¹ÌÍ¡½Ý¸…Ì¥ÑÌÕÉÉ•¹Ð(€€€½¹Ñ•áÐ½ÕÁ…¹äƒŠPÑ¡…ÐÍÕ‰ÍÑ¥ÑÕÑ¥½¸ÁÉ½‘Õ•¥µÁ½ÍÍ¥‰±”€Ä¸å´¼ÄÈÁ¬€ ÄÀÀ”¤(€€€ÍÑ…ÑÕÌµ‰…ÈÉ•…‘¥¹Ì€ ŒÔÀÐÈÄ¤¸]¥Ñ ¹¼É•…°ÕÉÉ•¹Ð½ÕÁ…¹ä­¹½Ý¸°(€€€½¹Ñ•áÑ}ÕÍ•½Á•É•¹ÐÍÑ…äÕ¹Í•ÐÉ…Ñ¡•ÈÑ¡…¸ÝÉ½¹œ¸ˆˆˆ(€€€…•¹Ð€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€µ½‘•°ô‰Ñ•ÍÐµµ½‘•°ˆ°(€€€€€€€Í•ÍÍ¥½¹}Ñ½Ñ…±}Ñ½­•¹ÌôÅ|äÀÁ|ÀÀÀ°(€€€€€€€½¹Ñ•áÑ}½µÁÉ•ÍÍ½ÈõÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€€€€€±…ÍÑ}ÁÉ½µÁÑ}Ñ½­•¹ÌôÀ°(€€€€€€€€€€€½¹Ñ•áÑ}±•¹Ñ ôÄÈÁ|ÀÀÀ°(€€€€€€€€€€€½µÁÉ•ÍÍ¥½¹}½Õ¹ÐôÀ°(€€€€€€€€¤°(€€€€¤(€€€ÕÍ…”€ôÍ•ÉÙ•È¹}•Ñ}ÕÍ…”¡…•¹Ð¤(€€€…ÍÍ•ÉÐÕÍ…”¹•Ð ‰½¹Ñ•áÑ}ÕÍ•ˆ¤€„ô€Å|äÀÁ|ÀÀÀ(€€€…ÍÍ•ÉÐ€‰½¹Ñ•áÑ}ÕÍ•ˆ¹½Ð¥¸ÕÍ…”(€€€…ÍÍ•ÉÐ€‰½¹Ñ•áÑ}Á•É•¹Ðˆ¹½Ð¥¸ÕÍ…”(()‘•˜Ñ•ÍÑ}•Ñ}ÕÍ…•}É•Á½ÉÑÍ}É•…±}ÕÉÉ•¹Ñ}½ÕÁ…¹ä ¤è(€€€€ˆˆ‰]¡•¸Ñ¡”½µÁÉ•ÍÍ½ÈÉ•Á½ÉÑÌ„É•…°ÕÉÉ•¹ÐÁÉ½µÁÐÍ¥é”°½¹Ñ•áÑ}ÕÍ•¥Ì(€€€Ñ¡…ÐÙ…±Õ”€¡¹½ÐÑ¡”ÕµÕ±…Ñ¥Ù”Ñ½Ñ…°¤…¹Ñ¡”Á•É•¹Ð¥ÌÍ…¹”¸ˆˆˆ(€€€…•¹Ð€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€µ½‘•°ô‰Ñ•ÍÐµµ½‘•°ˆ°(€€€€€€€Í•ÍÍ¥½¹}Ñ½Ñ…±}Ñ½­•¹ÌôÅ|äÀÁ|ÀÀÀ°(€€€€€€€½¹Ñ•áÑ}½µÁÉ•ÍÍ½ÈõÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€€€€€±…ÍÑ}ÁÉ½µÁÑ}Ñ½­•¹ÌôØÁ|ÀÀÀ°(€€€€€€€€€€€½¹Ñ•áÑ}±•¹Ñ ôÄÈÁ|ÀÀÀ°(€€€€€€€€€€€½µÁÉ•ÍÍ¥½¹}½Õ¹ÐôÈ°(€€€€€€€€¤°(€€€€¤(€€€ÕÍ…”€ôÍ•ÉÙ•È¹}•Ñ}ÕÍ…”¡…•¹Ð¤(€€€…ÍÍ•ÉÐÕÍ…•l‰½¹Ñ•áÑ}ÕÍ•‰t€ôô€ØÁ|ÀÀÀ(€€€…ÍÍ•ÉÐÕÍ…•l‰½¹Ñ•áÑ}µ…à‰t€ôô€ÄÈÁ|ÀÀÀ(€€€…ÍÍ•ÉÐÕÍ…•l‰½¹Ñ•áÑ}Á•É•¹Ð‰t€ôô€ÔÀ(()‘•˜Ñ•ÍÑ}•Ñ}ÕÍ…•}±…µÁÍ}Á½ÍÑ}½µÁÉ•ÍÍ¥½¹}Í•¹Ñ¥¹•° ¤è(€€€€ˆˆ‰I¥¡Ð…™Ñ•È„½µÁÉ•ÍÍ¥½¸°±…ÍÑ}ÁÉ½µÁÑ}Ñ½­•¹Ì¥ÌÑ¡”€´ÄÍ•¹Ñ¥¹•°(€€€€¡½¹Ù•ÉÍ…Ñ¥½¹}½µÁÉ•ÍÍ¥½¸Í•ÑÌ¥ÐÕ¹Ñ¥°Ñ¡”¹•áÐÉ•…°ÕÍ…”É•Á½ÉÐ¤¸%Ð¥Ì(€€€ÑÉÕÑ¡ä°Í¼½È€Á€‘½•Í¸Ð¹•ÕÑÉ…±¥é”¥ÐƒŠPÑ¡”Õ…ÉµÕÍÐ±…µÀ€ðÀÑ¼€ÀÍ¼(€€€Ñ¡”ÑÉ…¹Í¥Ñ¥½¹…°ÑÕÉ¸•µ¥ÑÌ¹¼…Õ”¥¹ÍÑ•…½˜±•…­¥¹œ½¹Ñ•áÑ}ÕÍ•ô´Ä¸ˆˆˆ(€€€…•¹Ð€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€µ½‘•°ô‰Ñ•ÍÐµµ½‘•°ˆ°(€€€€€€€Í•ÍÍ¥½¹}Ñ½Ñ…±}Ñ½­•¹ÌôÑ|ÀÀÁ|ÀÀÀ°(€€€€€€€½¹Ñ•áÑ}½µÁÉ•ÍÍ½ÈõÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€€€€€±…ÍÑ}ÁÉ½µÁÑ}Ñ½­•¹Ìô´Ä°(€€€€€€€€€€€½¹Ñ•áÑ}±•¹Ñ ôÅ|ÀÐá|ÔÜØ°(€€€€€€€€€€€½µÁÉ•ÍÍ¥½¹}½Õ¹ÐôØ°(€€€€€€€€¤°(€€€€¤(€€€ÕÍ…”€ôÍ•ÉÙ•È¹}•Ñ}ÕÍ…”¡…•¹Ð¤(€€€…ÍÍ•ÉÐ€‰½¹Ñ•áÑ}ÕÍ•ˆ¹½Ð¥¸ÕÍ…”(€€€…ÍÍ•ÉÐ€‰½¹Ñ•áÑ}Á•É•¹Ðˆ¹½Ð¥¸ÕÍ…”(((Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(ŒMÑÉ•…µ¥¹œQQLƒŠPÁ•ÈµÑÕÉ¸Á¥Á•±¥¹”€¬‰…É”µ¥¸(Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´()‘•˜}™…­•}ÑÑÍ}µ½‘Õ±•Ì¡µ½¹­•åÁ…Ñ °€¨°É•ÅÕ¥É•µ•¹ÑÌõQÉÕ”°Á±…å‰…­}ÍÑ½ÁÌõ9½¹”°±¥ÍÑ•¸õ9½¹”°ÑÉ…¹ÍÉ¥‰”õ9½¹”¤è(€€€€ˆˆ‰%¹ÍÑ…±°±¥¡ÑÝ•¥¡ÐÑ½½±Ì¹ÑÑÍ}Ñ½½°€¼Ñ½½±Ì¹Ù½¥•}µ½‘”™…­•Ì¸ˆˆˆ(€€€ÍÑ…ÉÑ•€ôíô((€€€‘•˜™…­•}ÍÑÉ•…´¡Ñ•áÑ}ÅÕ•Õ”°ÍÑ½À°‘½¹”°€¨©}­Ü¤è(€€€€€€€ÍÑ…ÉÑ•‘l‰ÅÕ•Õ”‰t€ôÑ•áÑ}ÅÕ•Õ”(€€€€€€€ÍÑ½À¹Ý…¥Ð Ô¤(€€€€€€€‘½¹”¹Í•Ð ¤((€€€‘•˜‘•™…Õ±Ñ}±¥ÍÑ•¸¡Í¡½Õ±‘}ÍÑ½À°…ÁÑÕÉ”õ…±Í”°½¹}ÑÉ¥•Èõ9½¹”°€¨©}­Ü¤è(€€€€€€€É•ÑÕÉ¸9½¹”¥˜…ÁÑÕÉ”•±Í”…±Í”((€€€‘•˜‘•™…Õ±Ñ}™‘}±¥ÍÑ•¸¡Í¡½Õ±‘}ÍÑ½À°¥Í}Á±…å¥¹œõ9½¹”°½¹}ÑÉ¥•Èõ9½¹”°€¨©}­Ü¤è(€€€€€€€É•ÑÕÉ¸9½¹”((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ¥Ñ•´ (€€€€€€€ÍåÌ¹µ½‘Õ±•Ì°(€€€€€€€€‰Ñ½½±Ì¹ÑÑÍ}Ñ½½°ˆ°(€€€€€€€ÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€€€€€¡•­}ÑÑÍ}É•ÅÕ¥É•µ•¹ÑÌõ±…µ‰‘„èÉ•ÅÕ¥É•µ•¹ÑÌ°(€€€€€€€€€€€}•Ñ}ÁÉ½Ù¥‘•Èõ±…µ‰‘„™œè€‰•‘”ˆ°(€€€€€€€€€€€}±½…‘}ÑÑÍ}½¹™¥œõ±…µ‰‘„èíô°(€€€€€€€€€€€•Ñ}•¹Ù}Ù…±Õ”õ±…µ‰‘„­•ä°‘•™…Õ±Ðôˆˆè‘•™…Õ±Ð°(€€€€€€€€¤°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ¥Ñ•´ (€€€€€€€ÍåÌ¹µ½‘Õ±•Ì°(€€€€€€€€‰Ñ½½±Ì¹ÑÑÍ}Ñ½½±}ÍÁ•…­•Èˆ°(€€€€€€€ÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡ÍÑÉ•…µ}ÑÑÍ}Ñ½}ÍÁ•…­•Èõ™…­•}ÍÑÉ•…´¤°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ¥Ñ•´ (€€€€€€€ÍåÌ¹µ½‘Õ±•Ì°(€€€€€€€€‰Ñ½½±Ì¹Ù½¥•}µ½‘”ˆ°(€€€€€€€ÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€€€€€ÍÑ½Á}Á±…å‰…¬õ±…µ‰‘„è€¡Á±…å‰…­}ÍÑ½ÁÌ¹…ÁÁ•¹¡QÉÕ”¤¥˜Á±…å‰…­}ÍÑ½ÁÌ¥Ì¹½Ð9½¹”•±Í”9½¹”¤°(€€€€€€€€€€€±¥ÍÑ•¹}™½É}ÍÁ•• õ±¥ÍÑ•¸½È‘•™…Õ±Ñ}±¥ÍÑ•¸°(€€€€€€€€€€€™Õ±±}‘ÕÁ±•á}±¥ÍÑ•¸õ±¥ÍÑ•¸½È‘•™…Õ±Ñ}™‘}±¥ÍÑ•¸°(€€€€€€€€€€€¥Í}…Õ‘¥½}½ÕÑÁÕÑ}…Ñ¥Ù”õ±…µ‰‘„è…±Í”°(€€€€€€€€€€€ÑÉ…¹ÍÉ¥‰•}É•½É‘¥¹œõÑÉ…¹ÍÉ¥‰”½È€¡±…µ‰‘„Á…Ñ °µ½‘•°õ9½¹”èì‰ÍÕ•ÍÌˆèQÉÕ”°€‰ÑÉ…¹ÍÉ¥ÁÐˆè€ˆ‰ô¤°(€€€€€€€€¤°(€€€€¤(€€€€ŒÉ•Í ±¥ÍÑ•¹•ÈÍ±½ÐÁ•ÈÑ•ÍÐƒŠPÑ¡”…É´¥Ì¥‘•µÁ½Ñ•¹ÐÁ•ÈÁÉ½•ÍÌ¸(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}™‘}±¥ÍÑ•¹•É}…Ñ¥Ù”ˆ°…±Í”¤(€€€É•ÑÕÉ¸ÍÑ…ÉÑ•(()‘•˜Ñ•ÍÑ}ÑÑÍ}ÍÑÉ•…µ}‰•¥¹}É•ÅÕ¥É•Í}Ù½¥•}ÑÑÌ¡µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}Y=%}QQLˆ°€ˆÀˆ¤(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}‰•¥¸ ¤¥Ì9½¹”(()‘•˜Ñ•ÍÑ}ÑÑÍ}ÍÑÉ•…µ}‰•¥¹}É•ÅÕ¥É•Í}Ý½É­¥¹}ÁÉ½Ù¥‘•È¡µ½¹­•åÁ…Ñ ¤è(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}Y=%}QQLˆ°€ˆÄˆ¤(€€€}™…­•}ÑÑÍ}µ½‘Õ±•Ì¡µ½¹­•åÁ…Ñ °É•ÅÕ¥É•µ•¹ÑÌõ…±Í”¤(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}‰•¥¸ ¤¥Ì9½¹”(()‘•˜Ñ•ÍÑ}ÑÑÍ}ÍÑÉ•…µ}‰•¥¹}…¹‘}ÍÑ½Á}±¥™•å±”¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰‰•¥¸ ¤ÍÁ…Ý¹ÌÑ¡”½¹ÍÕµ•ÈìÍÑ½À ¤ÕÑÌ¥Ð…¹±•…ÉÌÑ¡”Í±½Ð¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}Y=%}QQLˆ°€ˆÄˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}Y=%ˆ°€ˆÀˆ¤€€Œ¹¼‰…É”µ¥¸µ½¹¥Ñ½È€¡¹¼µ¥Œ¤(€€€Á±…å‰…­}ÍÑ½ÁÌè±¥ÍÐ€ômt(€€€ÍÑ…ÉÑ•€ô}™…­•}ÑÑÍ}µ½‘Õ±•Ì¡µ½¹­•åÁ…Ñ °Á±…å‰…­}ÍÑ½ÁÌõÁ±…å‰…­}ÍÑ½ÁÌ¤((€€€Ñ•áÑ}ÅÕ•Õ”€ôÍ•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}‰•¥¸ ¤(€€€…ÍÍ•ÉÐÑ•áÑ}ÅÕ•Õ”¥Ì¹½Ð9½¹”(€€€…ÍÍ•ÉÐÍÑ…ÉÑ•‘l‰ÅÕ•Õ”‰t¥ÌÑ•áÑ}ÅÕ•Õ”((€€€Ý¥Ñ Í•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}±½¬è(€€€€€€€ÍÑ…Ñ”€ôÍ•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}ÍÑ…Ñ”(€€€…ÍÍ•ÉÐÍÑ…Ñ”¥Ì¹½Ð9½¹”…¹¹½ÐÍÑ…Ñ•l‰ÍÑ½À‰t¹¥Í}Í•Ð ¤((€€€Í•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}ÍÑ½À ¤(€€€…ÍÍ•ÉÐÍÑ…Ñ•l‰ÍÑ½À‰t¹¥Í}Í•Ð ¤(€€€…ÍÍ•ÉÐÁ±…å‰…­}ÍÑ½ÁÌ€ôômQÉÕ•t(€€€Ý¥Ñ Í•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}±½¬è(€€€€€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}ÍÑ…Ñ”¥Ì9½¹”(()‘•˜Ñ•ÍÑ}ÑÑÍ}ÍÑÉ•…µ}‰•¥¹}‰…É•Í}¥¹}½¹}ÁÉ•Ù¥½ÕÍ}Á¥Á•±¥¹”¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰¹•ÜÑÕÉ¸ÌÁ¥Á•±¥¹”ÍÑ½ÁÌÑ¡”ÁÉ•Ù¥½ÕÌÑÕÉ¸ÌÍÁ•• €¡½¹”ÍÁ•…­•È¤¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}Y=%}QQLˆ°€ˆÄˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}Y=%ˆ°€ˆÀˆ¤(€€€}™…­•}ÑÑÍ}µ½‘Õ±•Ì¡µ½¹­•åÁ…Ñ ¤((€€€Í•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}‰•¥¸ ¤(€€€Ý¥Ñ Í•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}±½¬è(€€€€€€€™¥ÉÍÐ€ôÍ•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}ÍÑ…Ñ”(€€€Í•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}‰•¥¸ ¤(€€€…ÍÍ•ÉÐ™¥ÉÍÐ¥Ì¹½Ð9½¹”…¹™¥ÉÍÑl‰ÍÑ½À‰t¹¥Í}Í•Ð ¤(€€€Í•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}ÍÑ½À ¤(()‘•˜Ñ•ÍÑ}ÑÑÍ}ÍÑÉ•…µ}ÍÑ½Á}±…Ñ¡•Í}¥¹Ñ•ÉÉÕÁÑ¥½¹}™½É}¹•áÑ}ÑÕÉ¸¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰ÕÑÑ¥¹œ±¥Ù”ÍÁ•• €¡¥¹Ñ•ÉÉÕÁÐ€¼ÑåÁ¥¹œ‰…É”¤µ…É­ÌÑ¡”±…Ñ Ñ¡”¹•áÐ(€€€ÑÕÉ¸Ìµ½‘•°¹½Ñ”½¹ÍÕµ•Ìì„µ½‘”¡…¹”€¡ÕÍ•É}‰…É”õ…±Í”¤‘½•Ì¹½Ð¸ˆˆˆ(€€€¥µÁ½ÉÐÑ½½±Ì¹ÑÑÍ}ÍÑÉ•…µ¥¹œ…ÌÑÌ((€€€ÑÌ¹}¥¹Ñ•ÉÉÕÁÑ•‘}…Ð€ô9½¹”(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}Y=%}QQLˆ°€ˆÄˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}Y=%ˆ°€ˆÀˆ¤(€€€}™…­•}ÑÑÍ}µ½‘Õ±•Ì¡µ½¹­•åÁ…Ñ ¤((€€€Í•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}‰•¥¸ ¤(€€€Í•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}ÍÑ½À ¤€€Œ‘•™…Õ±ÐèÕÍ•È‰…É”(€€€…ÍÍ•ÉÐÑÌ¹Ñ…­•}ÍÁ••¡}¥¹Ñ•ÉÉÕÁÑ• ¤¥ÌQÉÕ”((€€€Í•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}‰•¥¸ ¤(€€€Í•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}ÍÑ½À¡ÕÍ•É}‰…É”õ…±Í”¤€€Œ€½Ù½¥”½™˜(€€€…ÍÍ•ÉÐÑÌ¹Ñ…­•}ÍÁ••¡}¥¹Ñ•ÉÉÕÁÑ• ¤¥Ì…±Í”(()‘•˜Ñ•ÍÑ}ÑÑÍ}ÍÑÉ•…µ}ÍÑ½Á}…™Ñ•É}¹…ÑÕÉ…±}™¥¹¥Í¡}‘½•Í}¹½Ñ}±…Ñ ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰MÁ•• Ñ¡…Ð…±É•…‘ä™¥¹¥Í¡•€¡‘½¹”Í•Ð¤¥Í¸Ð…¸¥¹Ñ•ÉÉÕÁÑ¥½¸¸ˆˆˆ(€€€¥µÁ½ÉÐÑ½½±Ì¹ÑÑÍ}ÍÑÉ•…µ¥¹œ…ÌÑÌ((€€€ÑÌ¹}¥¹Ñ•ÉÉÕÁÑ•‘}…Ð€ô9½¹”(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}Y=%}QQLˆ°€ˆÄˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}Y=%ˆ°€ˆÀˆ¤(€€€}™…­•}ÑÑÍ}µ½‘Õ±•Ì¡µ½¹­•åÁ…Ñ ¤((€€€Í•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}‰•¥¸ ¤(€€€Ý¥Ñ Í•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}±½¬è(€€€€€€€Í•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}ÍÑ…Ñ•l‰‘½¹”‰t¹Í•Ð ¤(€€€Í•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}ÍÑ½À ¤(€€€…ÍÍ•ÉÐÑÌ¹Ñ…­•}ÍÁ••¡}¥¹Ñ•ÉÉÕÁÑ• ¤¥Ì…±Í”(()‘•˜Ñ•ÍÑ}ÑÑÍ}ÍÑÉ•…µ}Ù…‘}‰…É•}¥¹}ÕÑÍ}Á¥Á•±¥¹•}…¹‘}ÍÕ‰µ¥ÑÍ}…ÁÑÕÉ”¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰UÍ•ÈÍÁ•• ‘ÕÉ¥¹œÁ±…å‰…¬ÕÑÌQQL…ÐÑ¡”µ½µ•¹Ð½˜‘•Ñ•Ñ¥½¸(€€€€¡Ù½¥”¹¥¹Ñ•ÉÉÕÁÑ•¤°Ñ¡•¸Ñ¡”…ÁÑÕÉ•¥¹Ñ•ÉÉÕÁÑ¥½¸¥ÌÑÉ…¹ÍÉ¥‰•…¹(€€€•µ¥ÑÑ•…ÌÙ½¥”¹ÑÉ…¹ÍÉ¥ÁÐÍ¼Ñ¡”QU$ÍÕ‰µ¥ÑÌ¥ÐƒŠP½µÁ±•Ñ”™É½´¥ÑÌ(€€€™¥ÉÍÐÍå±±…‰±”°¹¼É”µÉ•½ÉÉ½Õ¹ÑÉ¥À¸Q¡”ÕÐ…±Í¼±…Ñ¡•ÌÑ¡”(€€€ÍÁ•• µ¥¹Ñ•ÉÉÕÁÑ•¹½Ñ”™½ÈÑ¡”¹•áÐÑÕÉ¸¸ˆˆˆ(€€€¥µÁ½ÉÐÑ½½±Ì¹ÑÑÍ}ÍÑÉ•…µ¥¹œ…ÌÑÌ((€€€ÑÌ¹}¥¹Ñ•ÉÉÕÁÑ•‘}…Ð€ô9½¹”(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}Y=%}QQLˆ°€ˆÄˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}Y=%ˆ°€ˆÄˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}™œˆ°±…µ‰‘„èì‰Ù½¥”ˆèì‰‰…É•}¥¸ˆèQÉÕ•õô¤(€€€•Ù•¹ÑÌè±¥ÍÐ€ômt(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}Ù½¥•}•µ¥Ðˆ°±…µ‰‘„•Ù•¹Ð°Á…å±½…õ9½¹”è•Ù•¹ÑÌ¹…ÁÁ•¹ ¡•Ù•¹Ð°Á…å±½…¤¤(€€€€¤((€€€Ý…Ø€ôÑµÁ}Á…Ñ €¼€‰‰…É”¹Ý…Øˆ(€€€Ý…Ø¹ÝÉ¥Ñ•}‰åÑ•Ì¡ˆ‰I%ˆ¤((€€€‘•˜™…­•}±¥ÍÑ•¸¡Í¡½Õ±‘}ÍÑ½À°¥Í}Á±…å¥¹œõ9½¹”°½¹}ÑÉ¥•Èõ9½¹”°€¨©}­Ü¤è(€€€€€€€½¹}ÑÉ¥•È ‰Á±…å‰…¬ˆ¤€€ŒÁ±…å‰…¬ÕÐ¡…ÁÁ•¹Ì…Ð‘•Ñ•Ñ¥½¸(€€€€€€€É•ÑÕÉ¸ÍÑÈ¡Ý…Ø¤((€€€}™…­•}ÑÑÍ}µ½‘Õ±•Ì (€€€€€€€µ½¹­•åÁ…Ñ °(€€€€€€€±¥ÍÑ•¸õ™…­•}±¥ÍÑ•¸°(€€€€€€€ÑÉ…¹ÍÉ¥‰”õ±…µ‰‘„Á…Ñ °µ½‘•°õ9½¹”èì‰ÍÕ•ÍÌˆèQÉÕ”°€‰ÑÉ…¹ÍÉ¥ÁÐˆè€‰ÍÑ½À°…ÑÕ…±±çŠP‰ô°(€€€€¤((€€€Í•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}‰•¥¸ ¤(€€€‘•…‘±¥¹”€ôÑ¥µ”¹µ½¹½Ñ½¹¥Œ ¤€¬€Ô¸À(€€€Ý¡¥±”Ñ¥µ”¹µ½¹½Ñ½¹¥Œ ¤€ð‘•…‘±¥¹”…¹Ý…Ø¹•á¥ÍÑÌ ¤è(€€€€€€€Ñ¥µ”¹Í±••À À¸ÀÄ¤€€ŒÕ¹±¥¹¬€¡™¥¹…±±ä¤ÉÕ¹Ì…™Ñ•ÈÑ¡”ÑÉ…¹ÍÉ¥ÁÐ•µ¥Ð(€€€…ÍÍ•ÉÐ€ ‰Ù½¥”¹¥¹Ñ•ÉÉÕÁÑ•ˆ°9½¹”¤¥¸•Ù•¹ÑÌ(€€€…ÍÍ•ÉÐ€ ‰Ù½¥”¹ÑÉ…¹ÍÉ¥ÁÐˆ°ì‰Ñ•áÐˆè€‰ÍÑ½À°…ÑÕ…±±çŠP‰ô¤¥¸•Ù•¹ÑÌ(€€€…ÍÍ•ÉÐ¹½ÐÝ…Ø¹•á¥ÍÑÌ ¤€€Œ…ÁÑÕÉ”Ñ•µÀ™¥±”±•…¹•ÕÀ(€€€…ÍÍ•ÉÐÑÌ¹Ñ…­•}ÍÁ••¡}¥¹Ñ•ÉÉÕÁÑ• ¤¥ÌQÉÕ”€€ŒYÕÐ±…Ñ¡•ÌÑ¡”µ½‘•°¹½Ñ”(€€€Í•ÉÙ•È¹}ÑÑÍ}ÍÑÉ•…µ}ÍÑ½À ¤(()‘•˜Ñ•ÍÑ}™Õ±±}‘ÕÁ±•á}•¹•É…Ñ¥½¹}Á¡…Í•}¥¹Ñ•ÉÉÕÁÑÍ}ÉÕ¹¹¥¹}ÑÕÉ¸¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰MÁ•• UI%9114•¹•É…Ñ¥½¸€¡¹¼QQL…Õ‘¥¼å•Ð¤µÕÍÐ¥¹Ñ•ÉÉÕÁÐÑ¡”(€€€¥¸µ™±¥¡Ð…•¹ÐÑÕÉ¸Ù¥„Ñ¡”Í…µ”Í•…´Í•ÍÍ¥½¸¹¥¹Ñ•ÉÉÕÁÐÕÍ•Ì°…¹Ñ¡”(€€€…ÁÑÕÉ•¥¹Ñ•É©•Ñ¥½¸¥Ì•µ¥ÑÑ•…ÌÙ½¥”¹ÑÉ…¹ÍÉ¥ÁÐ¸Q¡¥Ì¥ÌÑ¡”(€€€¡…±˜µ‘ÕÁ±•à…ÀèÁÉ•Ù¥½ÕÍ±ä¹¼±¥ÍÑ•¹•È•á¥ÍÑ•Õ¹Ñ¥°Á±…å‰…¬ÍÑ…ÉÑ•¸ˆˆˆ(€€€¥µÁ½ÉÐÑ½½±Ì¹ÑÑÍ}ÍÑÉ•…µ¥¹œ…ÌÑÌ((€€€ÑÌ¹}¥¹Ñ•ÉÉÕÁÑ•‘}…Ð€ô9½¹”(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}Y=%ˆ°€ˆÄˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}Y=%}QQLˆ°€ˆÀˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}™œˆ°±…µ‰‘„èì‰Ù½¥”ˆèì‰‰…É•}¥¸ˆèQÉÕ•õô¤(€€€•Ù•¹ÑÌè±¥ÍÐ€ômt(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}Ù½¥•}•µ¥Ðˆ°±…µ‰‘„•Ù•¹Ð°Á…å±½…õ9½¹”è•Ù•¹ÑÌ¹…ÁÁ•¹ ¡•Ù•¹Ð°Á…å±½…¤¤(€€€€¤((€€€Ý…Ø€ôÑµÁ}Á…Ñ €¼€‰¥¹Ñ•É©•Ð¹Ý…Øˆ(€€€Ý…Ø¹ÝÉ¥Ñ•}‰åÑ•Ì¡ˆ‰I%ˆ¤((€€€¥¹Ñ•ÉÉÕÁÑ•€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤(€€€™…­•}…•¹Ð€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡¥¹Ñ•ÉÉÕÁÐõ±…µ‰‘„è¥¹Ñ•ÉÉÕÁÑ•¹Í•Ð ¤¤(€€€™…­•}Í•ÍÍ¥½¸€ôì‰ÉÕ¹¹¥¹œˆèQÉÕ”°€‰…•¹Ðˆè™…­•}…•¹Ñô(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹Ìˆ°ì‰Í¥µ™ˆè™…­•}Í•ÍÍ¥½¹ô¤((€€€‘•˜™…­•}±¥ÍÑ•¸¡Í¡½Õ±‘}ÍÑ½À°¥Í}Á±…å¥¹œõ9½¹”°½¹}ÑÉ¥•Èõ9½¹”°€¨©}­Ü¤è(€€€€€€€…ÍÍ•ÉÐ¥Í}Á±…å¥¹œ¥Ì¹½Ð9½¹”…¹¥Í}Á±…å¥¹œ ¤¥Ì…±Í”€€Œ•¹•É…Ñ¥½¸Á¡…Í”(€€€€€€€½¹}ÑÉ¥•È ‰•¹•É…Ñ¥½¸ˆ¤(€€€€€€€É•ÑÕÉ¸ÍÑÈ¡Ý…Ø¤((€€€}™…­•}ÑÑÍ}µ½‘Õ±•Ì (€€€€€€€µ½¹­•åÁ…Ñ °(€€€€€€€±¥ÍÑ•¸õ™…­•}±¥ÍÑ•¸°(€€€€€€€ÑÉ…¹ÍÉ¥‰”õ±…µ‰‘„Á…Ñ °µ½‘•°õ9½¹”èì‰ÍÕ•ÍÌˆèQÉÕ”°€‰ÑÉ…¹ÍÉ¥ÁÐˆè€‰Ý…¥Ð°ÑÉä…¹½Ñ¡•ÈÝ…ä‰ô°(€€€€¤((€€€Í•ÉÙ•È¹}…Éµ}™Õ±±}‘ÕÁ±•á}±¥ÍÑ•¹•È ¤((€€€‘•…‘±¥¹”€ôÑ¥µ”¹µ½¹½Ñ½¹¥Œ ¤€¬€Ô¸À(€€€Ý¡¥±”Ñ¥µ”¹µ½¹½Ñ½¹¥Œ ¤€ð‘•…‘±¥¹”…¹Ý…Ø¹•á¥ÍÑÌ ¤è(€€€€€€€Ñ¥µ”¹Í±••À À¸ÀÄ¤(€€€…ÍÍ•ÉÐ¥¹Ñ•ÉÉÕÁÑ•¹¥Í}Í•Ð ¤€€ŒÑ¡”ÉÕ¹¹¥¹œÑÕÉ¸Ý…Ì¥¹Ñ•ÉÉÕÁÑ•(€€€…ÍÍ•ÉÐ€ ‰Ù½¥”¹¥¹Ñ•ÉÉÕÁÑ•ˆ°9½¹”¤¥¸•Ù•¹ÑÌ(€€€…ÍÍ•ÉÐ€ ‰Ù½¥”¹ÑÉ…¹ÍÉ¥ÁÐˆ°ì‰Ñ•áÐˆè€‰Ý…¥Ð°ÑÉä…¹½Ñ¡•ÈÝ…ä‰ô¤¥¸•Ù•¹ÑÌ(€€€…ÍÍ•ÉÐ¹½ÐÝ…Ø¹•á¥ÍÑÌ ¤(()‘•˜Ñ•ÍÑ}™Õ±±}‘ÕÁ±•á}ÍÑ½Á}Á¡É…Í•}µ¥‘}•¹•É…Ñ¥½¹}•¹‘Í}Ù½¥•}¡…Ð¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰	…É”€ÍÑ½Àœ‘ÕÉ¥¹œ•¹•É…Ñ¥½¸€ô¥¹Ñ•ÉÉÕÁÐÑ¡”ÑÕÉ¸9•¹Ñ¡”Ù½¥”(€€€¡…Ð€ ÍÑ½À•Ù•ÉåÑ¡¥¹œœ¤°•µ¥ÑÑ•…ÌÑ¡”•áÁ±¥¥ÐÍÑ½Á}Á¡É…Í”Í¥¹…°¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}Y=%ˆ°€ˆÄˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}™œˆ°±…µ‰‘„èì‰Ù½¥”ˆèì‰‰…É•}¥¸ˆèQÉÕ•õô¤(€€€•Ù•¹ÑÌè±¥ÍÐ€ômt(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}Ù½¥•}•µ¥Ðˆ°±…µ‰‘„•Ù•¹Ð°Á…å±½…õ9½¹”è•Ù•¹ÑÌ¹…ÁÁ•¹ ¡•Ù•¹Ð°Á…å±½…¤¤(€€€€¤((€€€Ý…Ø€ôÑµÁ}Á…Ñ €¼€‰ÍÑ½À¹Ý…Øˆ(€€€Ý…Ø¹ÝÉ¥Ñ•}‰åÑ•Ì¡ˆ‰I%ˆ¤((€€€¥¹Ñ•ÉÉÕÁÑ•€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤(€€€™…­•}…•¹Ð€ôÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡¥¹Ñ•ÉÉÕÁÐõ±…µ‰‘„è¥¹Ñ•ÉÉÕÁÑ•¹Í•Ð ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹Ìˆ°ì‰Í¥µ™ˆèì‰ÉÕ¹¹¥¹œˆèQÉÕ”°€‰…•¹Ðˆè™…­•}…•¹Ñõô(€€€€¤((€€€‘•˜™…­•}±¥ÍÑ•¸¡Í¡½Õ±‘}ÍÑ½À°¥Í}Á±…å¥¹œõ9½¹”°½¹}ÑÉ¥•Èõ9½¹”°€¨©}­Ü¤è(€€€€€€€½¹}ÑÉ¥•È ‰•¹•É…Ñ¥½¸ˆ¤(€€€€€€€É•ÑÕÉ¸ÍÑÈ¡Ý…Ø¤((€€€}™…­•}ÑÑÍ}µ½‘Õ±•Ì (€€€€€€€µ½¹­•åÁ…Ñ °(€€€€€€€±¥ÍÑ•¸õ™…­•}±¥ÍÑ•¸°(€€€€€€€ÑÉ…¹ÍÉ¥‰”õ±…µ‰‘„Á…Ñ °µ½‘•°õ9½¹”èì‰ÍÕ•ÍÌˆèQÉÕ”°€‰ÑÉ…¹ÍÉ¥ÁÐˆè€‰ÍÑ½À‰ô°(€€€€¤(€€€€Œ¥Í}Ù½¥•}ÍÑ½Á}Á¡É…Í”±¥Ù•Ì¥¸Ñ¡”™…­•Ñ½½±Ì¹Ù½¥•}µ½‘”¹…µ•ÍÁ…”¸(€€€ÍåÌ¹µ½‘Õ±•Íl‰Ñ½½±Ì¹Ù½¥•}µ½‘”‰t¹¥Í}Ù½¥•}ÍÑ½Á}Á¡É…Í”€ô€ (€€€€€€€±…µ‰‘„Ñ•áÐèÑ•áÐ¹ÍÑÉ¥À ¤¹±½Ý•È ¤€ôô€‰ÍÑ½Àˆ(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ¥Ñ•´ (€€€€€€€ÍåÌ¹µ½‘Õ±•Ì°(€€€€€€€€‰¡•Éµ•Í}±¤¹Ù½¥”ˆ°(€€€€€€€ÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡ÍÑ½Á}½¹Ñ¥¹Õ½ÕÌõ±…µ‰‘„€¨©}­Üè9½¹”°ÍÁ•…­}Ñ•áÐõ±…µ‰‘„€©„°€¨©¬è9½¹”¤°(€€€€¤((€€€Í•ÉÙ•È¹}…Éµ}™Õ±±}‘ÕÁ±•á}±¥ÍÑ•¹•È ¤((€€€‘•…‘±¥¹”€ôÑ¥µ”¹µ½¹½Ñ½¹¥Œ ¤€¬€Ô¸À(€€€Ý¡¥±”Ñ¥µ”¹µ½¹½Ñ½¹¥Œ ¤€ð‘•…‘±¥¹”…¹Ý…Ø¹•á¥ÍÑÌ ¤è(€€€€€€€Ñ¥µ”¹Í±••À À¸ÀÄ¤(€€€…ÍÍ•ÉÐ¥¹Ñ•ÉÉÕÁÑ•¹¥Í}Í•Ð ¤(€€€…ÍÍ•ÉÐ€ ‰Ù½¥”¹ÑÉ…¹ÍÉ¥ÁÐˆ°ì‰ÍÑ½Á}Á¡É…Í”ˆèQÉÕ”°€‰Ñ•áÐˆè€‰ÍÑ½À‰ô¤¥¸•Ù•¹ÑÌ(€€€…ÍÍ•ÉÐ½Ì¹•¹Ù¥É½¸¹•Ð ‰!I5M}Y=%ˆ¤€ôô€ˆÀˆ€€ŒÙ½¥”¡…Ð•¹‘•(()‘•˜Ñ•ÍÑ}ÍÁ•…­}Ñ•áÑ}Ý¥Ñ¡}‰…É•}…ÉµÍ}µ½¹¥Ñ½É}…¹‘}ÕÑÍ}Á±…å‰…¬¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰Q¡”™…±±‰…¬Ý¡½±”µÉ•Á±äÍÁ•…¬Á…Ñ €¡ÍÑÉ•…µ¥¹œÁ¥Á•±¥¹”½Õ±‘¸Ð(€€€ÍÑ…ÉÐ¤…¹Ñ¡”Ù½¥”¹ÑÑÌIAµÕÍÐ‰”‰…É”µ…‰±”Ñ½¼èÍÁ•…­¥¹œ½Ù•ÈÑ¡”(€€€É•Á±äÕÑÌÁ±…å‰…¬…¹Ñ¡”…ÁÑÕÉ•¥¹Ñ•ÉÉÕÁÑ¥½¸¥Ì•µ¥ÑÑ•…Ì(€€€Ù½¥”¹ÑÉ…¹ÍÉ¥ÁÐƒŠPÁÉ•Ù¥½ÕÍ±äÑ¡•Í”Á…Ñ¡Ì…±±•ÍÁ•…­}Ñ•áÐ‰…É”…¹(€€€Ý•É”Õ¹¥¹Ñ•ÉÉÕÁÑ¥‰±”‰äÙ½¥”¸ˆˆˆ(€€€¥µÁ½ÉÐÑ½½±Ì¹ÑÑÍ}ÍÑÉ•…µ¥¹œ…ÌÑÌ((€€€ÑÌ¹}¥¹Ñ•ÉÉÕÁÑ•‘}…Ð€ô9½¹”(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}Y=%ˆ°€ˆÄˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}Y=%}QQLˆ°€ˆÄˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°(€€€€€€€€‰}±½…‘}™œˆ°(€€€€€€€±…µ‰‘„èì‰Ù½¥”ˆèì‰‰…É•}¥¸ˆèQÉÕ”°€‰‰…É•}¥¹}É…•}Í•½¹‘Ìˆè€Áõô°(€€€€¤(€€€•Ù•¹ÑÌè±¥ÍÐ€ômt(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}Ù½¥•}•µ¥Ðˆ°±…µ‰‘„•Ù•¹Ð°Á…å±½…õ9½¹”è•Ù•¹ÑÌ¹…ÁÁ•¹ ¡•Ù•¹Ð°Á…å±½…¤¤(€€€€¤((€€€Ý…Ø€ôÑµÁ}Á…Ñ €¼€‰‰…É”¹Ý…Øˆ(€€€Ý…Ø¹ÝÉ¥Ñ•}‰åÑ•Ì¡ˆ‰I%ˆ¤((€€€ÍÁ•…­}…±±Ì€ôíô(€€€ÍÁ•…­}ÍÑ…ÉÑ•€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤(€€€É•±•…Í•}ÍÁ•…¬€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤((€€€‘•˜™…­•}ÍÁ•…­}Ñ•áÐ¡Ñ•áÐ°ÍÑ½Á}•Ù•¹Ðõ9½¹”¤è(€€€€€€€ÍÁ•…­}…±±Íl‰Ñ•áÐ‰t€ôÑ•áÐ(€€€€€€€ÍÁ•…­}…±±Íl‰ÍÑ½Á}•Ù•¹Ð‰t€ôÍÑ½Á}•Ù•¹Ð(€€€€€€€ÍÁ•…­}ÍÑ…ÉÑ•¹Í•Ð ¤(€€€€€€€É•±•…Í•}ÍÁ•…¬¹Ý…¥Ð Ô¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ¥Ñ•´ (€€€€€€€ÍåÌ¹µ½‘Õ±•Ì°(€€€€€€€€‰¡•Éµ•Í}±¤¹Ù½¥”ˆ°(€€€€€€€ÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…”¡ÍÁ•…­}Ñ•áÐõ™…­•}ÍÁ•…­}Ñ•áÐ¤°(€€€€¤((€€€‘•˜™…­•}±¥ÍÑ•¸¡Í¡½Õ±‘}ÍÑ½À°¥Í}Á±…å¥¹œõ9½¹”°½¹}ÑÉ¥•Èõ9½¹”°€¨©}­Ü¤è(€€€€€€€ÍÁ•…­}ÍÑ…ÉÑ•¹Ý…¥Ð Ô¤(€€€€€€€½¹}ÑÉ¥•È ‰Á±…å‰…¬ˆ¤€€ŒÕÍ•ÈÑ…±­Ì½Ù•ÈÑ¡”É•Á±äƒŠHÕÐ¹½Ü(€€€€€€€É•ÑÕÉ¸ÍÑÈ¡Ý…Ø¤((€€€}™…­•}ÑÑÍ}µ½‘Õ±•Ì (€€€€€€€µ½¹­•åÁ…Ñ °(€€€€€€€±¥ÍÑ•¸õ™…­•}±¥ÍÑ•¸°(€€€€€€€ÑÉ…¹ÍÉ¥‰”õ±…µ‰‘„Á…Ñ °µ½‘•°õ9½¹”èì‰ÍÕ•ÍÌˆèQÉÕ”°€‰ÑÉ…¹ÍÉ¥ÁÐˆè€‰¡…¹œ½¸‰ô°(€€€€¤((€€€Í•ÉÙ•È¹}ÍÁ•…­}Ñ•áÑ}Ý¥Ñ¡}‰…É” ‰„±½¹œÍÁ½­•¸É•Á±äˆ¤((€€€‘•…‘±¥¹”€ôÑ¥µ”¹µ½¹½Ñ½¹¥Œ ¤€¬€Ô¸À(€€€Ý¡¥±”Ñ¥µ”¹µ½¹½Ñ½¹¥Œ ¤€ð‘•…‘±¥¹”…¹€ ‰Ù½¥”¹ÑÉ…¹ÍÉ¥ÁÐˆ°ì‰Ñ•áÐˆè€‰¡…¹œ½¸‰ô¤¹½Ð¥¸•Ù•¹ÑÌè(€€€€€€€Ñ¥µ”¹Í±••À À¸ÀÄ¤(€€€É•±•…Í•}ÍÁ•…¬¹Í•Ð ¤((€€€…ÍÍ•ÉÐÍÁ•…­}…±±Íl‰Ñ•áÐ‰t€ôô€‰„±½¹œÍÁ½­•¸É•Á±äˆ(€€€€ŒQ¡”Á¥Á•±¥¹”ÍÑ½À•Ù•¹Ð¥ÌÍ¡…É•Ý¥Ñ ÍÁ•…­}Ñ•áÐÍ¼„ÍÑÉ•…µ¥¹œ(€€€€Œ‘¥ÍÁ…Ñ ¥¹Í¥‘”¥Ð¥ÌÕÐÑ½¼¸(€€€…ÍÍ•ÉÐÍÁ•…­}…±±Íl‰ÍÑ½Á}•Ù•¹Ð‰t¥Ì¹½Ð9½¹”(€€€…ÍÍ•ÉÐÍÁ•…­}…±±Íl‰ÍÑ½Á}•Ù•¹Ð‰t¹¥Í}Í•Ð ¤(€€€…ÍÍ•ÉÐ€ ‰Ù½¥”¹¥¹Ñ•ÉÉÕÁÑ•ˆ°9½¹”¤¥¸•Ù•¹ÑÌ(€€€…ÍÍ•ÉÐ€ ‰Ù½¥”¹ÑÉ…¹ÍÉ¥ÁÐˆ°ì‰Ñ•áÐˆè€‰¡…¹œ½¸‰ô¤¥¸•Ù•¹ÑÌ(€€€…ÍÍ•ÉÐÑÌ¹Ñ…­•}ÍÁ••¡}¥¹Ñ•ÉÉÕÁÑ• ¤¥ÌQÉÕ”(()‘•˜Ñ•ÍÑ}ÍÁ•…­}Ñ•áÑ}Ý¥Ñ¡}‰…É•}¹½}µ½¹¥Ñ½É}Ý¡•¹}Ù½¥•}µ½‘•}½™˜¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰ÕÑ¼µÍÁ•…¬Ý¥Ñ Ù½¥”µ½‘”½™˜€¡¹¼µ¥Œ±½½À¤µÕÍÐ¹½Ð½Á•¸Ñ¡”µ¥Œ¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}Y=%ˆ°€ˆÀˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰!I5M}Y=%}QQLˆ°€ˆÄˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±½…‘}™œˆ°±…µ‰‘„èì‰Ù½¥”ˆèì‰‰…É•}¥¸ˆèQÉÕ•õô¤((€€€±¥ÍÑ•¹•€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤((€€€‘•˜™…­•}±¥ÍÑ•¸¡Í¡½Õ±‘}ÍÑ½À°…ÁÑÕÉ”õ…±Í”°½¹}ÑÉ¥•Èõ9½¹”°€¨©}­Ü¤è(€€€€€€€±¥ÍÑ•¹•¹Í•Ð ¤(€€€€€€€É•ÑÕÉ¸9½¹”((€€€‘½¹•}ÍÁ•…­¥¹œ€ôÑ¡É•…‘¥¹œ¹Ù•¹Ð ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ¥Ñ•´ (€€€€€€€ÍåÌ¹µ½‘Õ±•Ì°(€€€€€€€€‰¡•Éµ•Í}±¤¹Ù½¥”ˆ°(€€€€€€€ÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€€€€€ÍÁ•…­}Ñ•áÐõ±…µ‰‘„Ñ•áÐ°ÍÑ½Á}•Ù•¹Ðõ9½¹”è‘½¹•}ÍÁ•…­¥¹œ¹Í•Ð ¤(€€€€€€€€¤°(€€€€¤(€€€}™…­•}ÑÑÍ}µ½‘Õ±•Ì¡µ½¹­•åÁ…Ñ °±¥ÍÑ•¸õ™…­•}±¥ÍÑ•¸¤((€€€Í•ÉÙ•È¹}ÍÁ•…­}Ñ•áÑ}Ý¥Ñ¡}‰…É” ‰ÅÕ¥•ÐÉ•Á±äˆ¤(€€€…ÍÍ•ÉÐ‘½¹•}ÍÁ•…­¥¹œ¹Ý…¥Ð Ô¤(€€€Ñ¥µ”¹Í±••À À¸Ä¤(€€€…ÍÍ•ÉÐ¹½Ð±¥ÍÑ•¹•¹¥Í}Í•Ð ¤(()‘•˜}…ÁÑÕÉ•}Í•ÉÙ•É}É•ÅÕ•ÍÐ¡µ½¹­•åÁ…Ñ °É•ÍÕ±Ð¤è(€€€€ˆˆ‰MÑÕˆÑ¡”Í•ÉÙ•ÈµÉ•ÅÕ•ÍÐÍ•¹…¹…ÁÑÕÉ”€¡µ•Ñ¡½°Í¥°Á…É…µÌ°Ñ¥µ•½ÕÐ¤¸ˆˆˆ(€€€™É½´ÑÕ¥}…Ñ•Ý…ä¥µÁ½ÉÐÍ•ÉÙ•É}É•ÅÕ•ÍÑÌ(€€€…ÁÑÕÉ•€ôíô((€€€‘•˜™…­•}Í•¹¡µ•Ñ¡½°Í¥°Á…É…µÌ°€¨°Ñ¥µ•½ÕÐ°Å¥‘Ìõ9½¹”¤è(€€€€€€€…ÁÑÕÉ•¹ÕÁ‘…Ñ”¡µ•Ñ¡½õµ•Ñ¡½°Í¥õÍ¥°Á…É…µÌõÁ…É…µÌ°Ñ¥µ•½ÕÐõÑ¥µ•½ÕÐ°Å¥‘ÌõÅ¥‘Ì¤(€€€€€€€É•ÑÕÉ¸É•ÍÕ±Ð((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•É}É•ÅÕ•ÍÑÌ°€‰Í•¹ˆ°™…­•}Í•¹¤(€€€É•ÑÕÉ¸…ÁÑÕÉ•(()‘•˜Ñ•ÍÑ}±…É¥™å}…±±‰…­}ÕÍ•Í}½¹™¥ÕÉ•‘}Ñ¥µ•½ÕÐ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰Q¡”QU$½‘•Í­Ñ½À±…É¥™ä‰É¥‘”Í•¹‘Ì„±…É¥™å€Í•ÉÙ•ÈÉ•ÅÕ•ÍÐÝ¥Ñ Ñ¡”…¹½¹¥…°±…É¥™äÑ¥µ•½ÕÐ(€€€€¡Ù¥„}±…É¥™å}Ñ¥µ•½ÕÑ}Í•½¹‘Ì¤°…¹É•ÑÕÉ¹ÌÑ¡”É•ÍÁ½¹Í”Ì…¹ÍÝ•É€¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}±…É¥™å}Ñ¥µ•½ÕÑ}Í•½¹‘Ìˆ°±…µ‰‘„è€ÐÈ¤(€€€…ÁÑÕÉ•€ô}…ÁÑÕÉ•}Í•ÉÙ•É}É•ÅÕ•ÍÐ¡µ½¹­•åÁ…Ñ °ì‰…¹ÍÝ•Èˆè€‰…¹ÍÝ•È‰ô¤((€€€É•ÍÕ±Ð€ôÍ•ÉÙ•È¹}…•¹Ñ}‰Ì ‰Í¥´Äˆ¥l‰±…É¥™å}…±±‰…¬‰t ‰A¥¬½¹”ˆ°l‰„ˆ°€‰ˆ‰t¤((€€€…ÍÍ•ÉÐÉ•ÍÕ±Ð€ôô€‰…¹ÍÝ•Èˆ(€€€…ÍÍ•ÉÐ…ÁÑÕÉ•‘l‰µ•Ñ¡½‰t€ôô€‰±…É¥™äˆ…¹…ÁÑÕÉ•‘l‰Í¥‰t€ôô€‰Í¥´Äˆ(€€€…ÍÍ•ÉÐ…ÁÑÕÉ•‘l‰Ñ¥µ•½ÕÐ‰t€ôô€ÐÈ(€€€…ÍÍ•ÉÐ…ÁÑÕÉ•‘l‰Á…É…µÌ‰t€ôôì‰ÅÕ•ÍÑ¥½¸ˆè€‰A¥¬½¹”ˆ°€‰¡½¥•Ìˆèl‰„ˆ°€‰ˆ‰uô(()‘•˜Ñ•ÍÑ}±…É¥™å}…±±‰…­}µÕ±Ñ¥}Í•±•Ñ}¡¥¹Ð¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰µÕ±Ñ¥}Í•±•ÐõQÉÕ”…‘‘ÌÑ¡”¡¥¹ÐÑ¼Ñ¡”Á…É…µÌìÑ¡”Í¥¹±”µÍ•±•ÐÍ¡…Á”ÍÑ…åÌ‰åÑ”µ¥‘•¹Ñ¥…°Ñ¼Ñ¡”(€€€ÁÉ”µµÕ±Ñ¤µÍ•±•ÐÁÉ½Ñ½½°€¡½±‘•ÈÉ•¹‘•É•ÉÌµÕÍÐ¹•Ù•ÈÍ•”Ñ¡”•áÑÉ„™¥•±¤¸ˆˆˆ(€€€…ÁÑÕÉ•€ô}…ÁÑÕÉ•}Í•ÉÙ•É}É•ÅÕ•ÍÐ¡µ½¹­•åÁ…Ñ °ì‰…¹ÍÝ•Èˆè€‰…¹ÍÝ•È‰ô¤(€€€ˆ€ôÍ•ÉÙ•È¹}…•¹Ñ}‰Ì ‰Í¥´Äˆ¥l‰±…É¥™å}…±±‰…¬‰t((€€€ˆ ‰A¥¬µ…¹äˆ°l‰„ˆ°€‰ˆ‰t°µÕ±Ñ¥}Í•±•ÐõQÉÕ”¤(€€€…ÍÍ•ÉÐ…ÁÑÕÉ•‘l‰Á…É…µÌ‰t€ôôì‰ÅÕ•ÍÑ¥½¸ˆè€‰A¥¬µ…¹äˆ°€‰¡½¥•Ìˆèl‰„ˆ°€‰ˆ‰t°€‰µÕ±Ñ¥}Í•±•ÐˆèQÉÕ•ô((€€€ˆ ‰A¥¬½¹”ˆ°l‰„ˆ°€‰ˆ‰t°µÕ±Ñ¥}Í•±•Ðõ…±Í”¤(€€€…ÍÍ•ÉÐ…ÁÑÕÉ•‘l‰Á…É…µÌ‰t€ôôì‰ÅÕ•ÍÑ¥½¸ˆè€‰A¥¬½¹”ˆ°€‰¡½¥•Ìˆèl‰„ˆ°€‰ˆ‰uô(()ÁåÑ•ÍÐ¹µ…É¬¹Á…É…µ•ÑÉ¥é” (€€€€ ‰½¹™¥ÕÉ•ˆ°€‰•áÁ•Ñ•ˆ¤°(€€€l À°9½¹”¤°€ ´Ä°9½¹”¤°€ ÐÈ°€ÐÈ¥t°(¤)‘•˜Ñ•ÍÑ}±…É¥™å}Ñ¥µ•½ÕÑ}Í•½¹‘Í}µ…ÁÍ}¹½¹}Á½Í¥Ñ¥Ù•}Ñ½}Õ¹±¥µ¥Ñ•¡µ½¹­•åÁ…Ñ °½¹™¥ÕÉ•°•áÁ•Ñ•¤è(€€€€ˆˆ‰€ðô€Á€±…É¥™äÑ¥µ•½ÕÐµ•…¹ÌÕ¹±¥µ¥Ñ•…¹É•…¡•ÌÑ¡”Í•ÉÙ•ÈÉ•ÅÕ•ÍÐ…Ì9½¹”(€€€€¡Ý…¥Ð¡9½¹”¤Ý…¥ÑÌ™½É•Ù•È¤É…Ñ¡•ÈÑ¡…¸…¸¥µµ•‘¥…Ñ”Ý…¥Ð À¤Í­¥À¸ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ ‰Ñ½½±Ì¹±…É¥™å}…Ñ•Ý…ä¹•Ñ}±…É¥™å}Ñ¥µ•½ÕÐˆ°±…µ‰‘„è½¹™¥ÕÉ•¤((€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}±…É¥™å}Ñ¥µ•½ÕÑ}Í•½¹‘Ì ¤€ôô•áÁ•Ñ•(()‘•˜Ñ•ÍÑ}‰Õ¥±‘}Á•ÉÍ¥ÍÑ}µ•ÍÍ…•}Ý¥Ñ¡}¥µ…•}É•™Í}Ý¥Ñ¡½ÕÑ}¥µ…•Í}É•ÑÕÉ¹Í}Ñ•áÐ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆˆŒÜÀÜÈÀèÝ¡•¸¹¼¥µ…•Ì…É”…ÑÑ…¡•Ñ¡”Á•ÉÍ¥ÍÑ•µ•ÍÍ…”¥ÌÑ¡”É…Ü(€€€ÁÉ½µÁÐƒŠP¹¼¥µ…”‘¥É•Ñ¥Ù”ÁÉ•™¥à¥Ì¥¹ÑÉ½‘Õ•¸ˆˆˆ(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}‰Õ¥±‘}Á•ÉÍ¥ÍÑ}µ•ÍÍ…•}Ý¥Ñ¡}¥µ…•}É•™Ì ‰Ý¡…Ð¥ÌÑ¡¥Ìüˆ°mt¤€ôô€‰Ý¡…Ð¥ÌÑ¡¥Ìüˆ(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}‰Õ¥±‘}Á•ÉÍ¥ÍÑ}µ•ÍÍ…•}Ý¥Ñ¡}¥µ…•}É•™Ì ˆˆ°mt¤€ôô€ˆˆ(()‘•˜Ñ•ÍÑ}‰Õ¥±‘}Á•ÉÍ¥ÍÑ}µ•ÍÍ…•}Ý¥Ñ¡}¥µ…•}É•™Í}…ÁÁ•¹‘Í}•á¥ÍÑ¥¹}Á…Ñ¡Ì¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰ÑÑ…¡•¥µ…•ÌÑ¡…ÐÍÑ¥±°•á¥ÍÐ½¸‘¥Í¬…É”Á•ÉÍ¥ÍÑ•…ÌÑÉ…¥±¥¹œ(€€€¥µ…”èñÁ…Ñ ù€‘¥É•Ñ¥Ù”±¥¹•ÌÍ¼Ñ¡”‘•Í­Ñ½ÀÉ•¹‘•ÉÌÑ¡•´…™Ñ•È„(€€€É•ÍÑ…ÉÐ€¡¥¹ÍÑ•…½˜Ñ¡”Ù¥Í¥½¸µ½¹±ä•¹É¥¡µ•¹ÐÑ¡…ÐÍ¥±•¹Ñ±ä‰É•…­Ì¤¸ˆˆˆ(€€€¥µœ€ôÑµÁ}Á…Ñ €¼€‰…Ð¹Á¹œˆ(€€€¥µœ¹ÝÉ¥Ñ•}‰åÑ•Ì¡ˆ‰qààåA9ˆ¤((€€€É•ÍÕ±Ð€ôÍ•ÉÙ•È¹}‰Õ¥±‘}Á•ÉÍ¥ÍÑ}µ•ÍÍ…•}Ý¥Ñ¡}¥µ…•}É•™Ì ‰Ý¡…Ð¥Ì¥¸Ñ¡¥ÌÁ¡½Ñ¼üˆ°mÍÑÈ¡¥µœ¥t¤((€€€…ÍÍ•ÉÐÉ•ÍÕ±Ð€ôô˜‰Ý¡…Ð¥Ì¥¸Ñ¡¥ÌÁ¡½Ñ¼ýq¹¥µ…”éí¥µôˆ(((()‘•˜Ñ•ÍÑ}‰Õ¥±‘}Á•ÉÍ¥ÍÑ}µ•ÍÍ…•}Ý¥Ñ¡}¥µ…•}É•™Í}Í­¥ÁÍ}µ¥ÍÍ¥¹}Á…Ñ¡Ì¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰=¹±äÁ…Ñ¡ÌÑ¡…ÐÍÑ¥±°•á¥ÍÐ…É”Á•ÉÍ¥ÍÑ•ì„µ¥ÍÍ¥¹œ™¥±”µÕÍÐ¹½Ð(€€€¥¹©•Ð„‘…¹±¥¹œ¥µ…”É•˜¥¹Ñ¼Ñ¡”ÑÉ…¹ÍÉ¥ÁÐ¸ˆˆˆ(€€€•á¥ÍÑ¥¹œ€ôÑµÁ}Á…Ñ €¼€‰„¹Á¹œˆ(€€€•á¥ÍÑ¥¹œ¹ÝÉ¥Ñ•}‰åÑ•Ì¡ˆ‰Á¹œˆ¤(€€€µ¥ÍÍ¥¹œ€ôÍÑÈ¡ÑµÁ}Á…Ñ €¼€‰½¹”¹Á¹œˆ¤((€€€É•ÍÕ±Ð€ôÍ•ÉÙ•È¹}‰Õ¥±‘}Á•ÉÍ¥ÍÑ}µ•ÍÍ…•}Ý¥Ñ¡}¥µ…•}É•™Ì ‰½µÁ…É”Ñ¡•´ˆ°mÍÑÈ¡•á¥ÍÑ¥¹œ¤°µ¥ÍÍ¥¹t¤((€€€…ÍÍ•ÉÐÉ•ÍÕ±Ð€ôô˜‰½µÁ…É”Ñ¡•µq¹¥µ…”éí•á¥ÍÑ¥¹ôˆ(()‘•˜Ñ•ÍÑ}‰Õ¥±‘}Á•ÉÍ¥ÍÑ}µ•ÍÍ…•}Ý¥Ñ¡}¥µ…•}É•™Í}Ý¥Ñ¡½ÕÑ}Ñ•áÑ}¥Í}É•™Í}½¹±ä¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰ÍÑ…¹µ…±½¹”…ÑÑ…¡µ•¹Ð€¡¹¼…ÁÑ¥½¸¤Á•ÉÍ¥ÍÑÌ…Ì©ÕÍÐÑ¡”‘¥É•Ñ¥Ù”(€€€±¥¹”°Í¼„‰…É”¥µ…”ÍÕÉÙ¥Ù•Ì¥¸¡¥ÍÑ½Éä…¹¥Ì¹½Ð‘É½ÁÁ•…Ì•µÁÑä¸ˆˆˆ(€€€¥µœ€ôÑµÁ}Á…Ñ €¼€‰½¹±ä¹Á¹œˆ(€€€¥µœ¹ÝÉ¥Ñ•}‰åÑ•Ì¡ˆ‰Á¹œˆ¤((€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}‰Õ¥±‘}Á•ÉÍ¥ÍÑ}µ•ÍÍ…•}Ý¥Ñ¡}¥µ…•}É•™Ì ˆˆ°mÍÑÈ¡¥µœ¥t¤€ôô˜‰¥µ…”éí¥µôˆ(()‘•˜Ñ•ÍÑ}‰Õ¥±‘}Á•ÉÍ¥ÍÑ}µ•ÍÍ…•}ÅÕ½Ñ•Í}Á…Ñ¡Í}½¹Ñ…¥¹¥¹}ÍÁ…•Ì¡ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰Q¡”Õ¹ÅÕ½Ñ•…±Ñ•É¹…Ñ¥Ù”¥¸Ñ¡”‘¥É•Ñ¥Ù”Á…ÑÑ•É¸¥ÌqqL­€°Í¼„Á…Ñ (€€€Ý¥Ñ „ÍÁ…”Á…ÉÍ•Ì…Ì„ÑÉÕ¹…Ñ•É•˜Ý¥Ñ Ñ¡”Ñ…¥°±•™Ð…Ì±½½Í”Ñ•áÐ¸(€€€•Í­Ñ½À½µÁ½Í•È¥µ…•Ì±¥Ù”¥¸Ñ¡”…ÁÀÌÕÍ•É…Ñ„‘¥È°Ý¡¥ ½¸µ…=L¥Ì(€€€ø½1¥‰É…Éä½ÁÁ±¥…Ñ¥½¸MÕÁÁ½ÉÐ¼¸¸¹€ƒŠP„ÍÁ…”•Ù•ÉäÑ¥µ”¸ˆˆˆ(€€€¥µ}‘¥È€ôÑµÁ}Á…Ñ €¼€‰ÁÁ±¥…Ñ¥½¸MÕÁÁ½ÉÐˆ€¼€‰!•Éµ•Ìˆ€¼€‰½µÁ½Í•Èµ¥µ…•Ìˆ(€€€¥µ}‘¥È¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”¤(€€€¥µœ€ô¥µ}‘¥È€¼€‰…Ð¹Á¹œˆ(€€€¥µœ¹ÝÉ¥Ñ•}‰åÑ•Ì¡ˆ‰Á¹œˆ¤((€€€É•ÍÕ±Ð€ôÍ•ÉÙ•È¹}‰Õ¥±‘}Á•ÉÍ¥ÍÑ}µ•ÍÍ…•}Ý¥Ñ¡}¥µ…•}É•™Ì ‰Ý¡…Ð¥ÌÑ¡¥Ìüˆ°mÍÑÈ¡¥µœ¥t¤((€€€…ÍÍ•ÉÐÉ•ÍÕ±Ð€ôô˜‰Ý¡…Ð¥ÌÑ¡¥Ìýq¹¥µ…”éí¥µõ€ˆ(()‘•˜Ñ•ÍÑ}Á•ÉÍ¥ÍÑ}ÕÍ•É}µ•ÍÍ…•}µ¥ÉÉ½ÉÍ}Ñ¡•}Í¡…Á•}Í•¹Ñ}Ñ½}Ñ¡•}µ½‘•°¡ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰¹…Ñ¥Ù”µÙ¥Í¥½¸ÑÕÉ¸Í•¹‘Ì½¹Ñ•¹Ñ€…Ì„Á…ÉÑÌ±¥ÍÐ°…¹Ñ¡”Í•ÍÍ¥½¸(€€€ÍÑ½É”¥¹½É•Ì„Á±…¥¸µÍÑÉ¥¹œ½Ù•ÉÉ¥‘”™½È„±¥ÍÐÁ…å±½…¸Q¡”½Ù•ÉÉ¥‘”µÕÍÐ(€€€µ¥ÉÉ½ÈÑ¡”±¥ÍÐÍ¡…Á”€¡É•˜Ñ•áÐ€¬Ñ¡”½É¥¥¹…°¥µ…”Á…ÉÑÌ¤½È¥Ð¥Ì(€€€Í¥±•¹Ñ±ä‘É½ÁÁ•…¹Ñ¡”…ÑÑ…¡µ•¹Ð¹•Ù•ÈÉ•…¡•Ì¡¥ÍÑ½Éä¸ˆˆˆ(€€€¥µœ€ôÑµÁ}Á…Ñ €¼€‰…Ð¹Á¹œˆ(€€€¥µœ¹ÝÉ¥Ñ•}‰åÑ•Ì¡ˆ‰Á¹œˆ¤(€€€¥µ…•}Á…ÉÐ€ôì‰ÑåÁ”ˆè€‰¥µ…•}ÕÉ°ˆ°€‰¥µ…•}ÕÉ°ˆèì‰ÕÉ°ˆè€‰‘…Ñ„é¥µ…”½Á¹œí‰…Í”ØÐ±‰õô(€€€¹…Ñ¥Ù•}Á…ÉÑÌ€ômì‰ÑåÁ”ˆè€‰Ñ•áÐˆ°€‰Ñ•áÐˆè€‰…Á¤µ½¹±äÑ•áÐ‰ô°¥µ…•}Á…ÉÑt((€€€½Ù•ÉÉ¥‘”€ôÍ•ÉÙ•È¹}‰Õ¥±‘}Á•ÉÍ¥ÍÑ}ÕÍ•É}µ•ÍÍ…” ‰Ý¡…Ð¥ÌÑ¡¥Ìüˆ°mÍÑÈ¡¥µœ¥t°¹…Ñ¥Ù•}Á…ÉÑÌ¤((€€€…ÍÍ•ÉÐ½Ù•ÉÉ¥‘”€ôômì‰ÑåÁ”ˆè€‰Ñ•áÐˆ°€‰Ñ•áÐˆè˜‰Ý¡…Ð¥ÌÑ¡¥Ìýq¹¥µ…”éí¥µô‰ô°¥µ…•}Á…ÉÑt(()‘•˜Ñ•ÍÑ}Á•ÉÍ¥ÍÑ}ÕÍ•É}µ•ÍÍ…•}ÍÑ…åÍ}…}ÍÑÉ¥¹}™½É}Ñ•áÑ}µ½‘”¡ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰Q•áÐµµ½‘”€¡Ù¥Í¥½¸µÁÉ•ÁÉ½•ÍÍ•¤ÑÕÉ¹ÌÍ•¹„ÍÑÉ¥¹œ°Í¼Ñ¡”½Ù•ÉÉ¥‘”(€€€ÍÑ…åÌ„ÍÑÉ¥¹œƒŠPÑ¡”Í¡…Á”Ñ¡”Í•ÍÍ¥½¸ÍÑ½É”É•ÝÉ¥Ñ•Ì‘¥É•Ñ±ä¸ˆˆˆ(€€€¥µœ€ôÑµÁ}Á…Ñ €¼€‰…Ð¹Á¹œˆ(€€€¥µœ¹ÝÉ¥Ñ•}‰åÑ•Ì¡ˆ‰Á¹œˆ¤((€€€½Ù•ÉÉ¥‘”€ôÍ•ÉÙ•È¹}‰Õ¥±‘}Á•ÉÍ¥ÍÑ}ÕÍ•É}µ•ÍÍ…” ‰Ý¡…Ð¥ÌÑ¡¥Ìüˆ°mÍÑÈ¡¥µœ¥t°€‰•¹É¥¡•…Á¤µ½¹±äÑ•áÐˆ¤((€€€…ÍÍ•ÉÐ½Ù•ÉÉ¥‘”€ôô˜‰Ý¡…Ð¥ÌÑ¡¥Ìýq¹¥µ…”éí¥µôˆ(()‘•˜Ñ•ÍÑ}¹…Ñ¥Ù•}Ù¥Í¥½¹}ÑÕÉ¹}Á•ÉÍ¥ÍÑÍ}…}É•¹‘•É…‰±•}¥µ…•}É•˜¡ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰¹Ñ¼•¹Ñ¡É½Õ Ñ¡”É•…°Í•ÍÍ¥½¸µÍÑ½É”™±ÕÍ èÝ¡¥¡•Ù•È¥µ…”¥¹ÁÕÐ(€€€µ½‘”Ñ¡”ÑÕÉ¸ÕÍ•°Ñ¡”‘ÕÉ…‰±”É½Ü…ÉÉ¥•Ì…¸¥µ…”é€É•˜Ñ¡”‘•Í­Ñ½À(€€€…¸É•¹‘•È…™Ñ•È„É•ÍÑ…ÉÐ¸ˆˆˆ(€€€™É½´Õ¹¥ÑÑ•ÍÐ¹µ½¬¥µÁ½ÉÐ5…¥5½¬((€€€™É½´…•¹Ð¹¥µ…•}É½ÕÑ¥¹œ¥µÁ½ÉÐ‰Õ¥±‘}¹…Ñ¥Ù•}½¹Ñ•¹Ñ}Á…ÉÑÌ(€€€™É½´ÉÕ¹}…•¹Ð¥µÁ½ÉÐ%•¹Ð((€€€¥µ}‘¥È€ôÑµÁ}Á…Ñ €¼€‰ÁÁ±¥…Ñ¥½¸MÕÁÁ½ÉÐˆ€¼€‰½µÁ½Í•Èµ¥µ…•Ìˆ(€€€¥µ}‘¥È¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”¤(€€€¥µœ€ô¥µ}‘¥È€¼€‰…Ð¹Á¹œˆ(€€€¥µœ¹ÝÉ¥Ñ•}‰åÑ•Ì (€€€€€€€‰åÑ•Ì¹™É½µ¡•à (€€€€€€€€€€€€ˆàäÔÀÑ”ÐÜÁÁ„Å„Á„ÀÀÀÀÀÀÁÐäÐàÐÐÔÈÀÀÀÀÀÀÀÄÀÀÀÀÀÀÀÄÀàÀØÀÀÀÀÀˆ(€€€€€€€€€€€€ˆÀÅ˜ÄÕŒÐàäÀÀÀÀÀÀÁ„ÐäÐÐÐÄÔÐÜàåŒØÌØÀÀÀÀÀÀÈÀÀÀÄÀÁ™™™˜ÀÌÀÀÀÀÀØÀÀˆ(€€€€€€€€€€€€ˆÀÔÔÝ‰™…‰ÐÀÀÀÀÀÀÀÀÐäÐÔÑ”ÐÑ…”ÐÈØÀàÈˆ(€€€€€€€€¤(€€€€¤(€€€¹…Ñ¥Ù•}Á…ÉÑÌ°Í­¥ÁÁ•€ô‰Õ¥±‘}¹…Ñ¥Ù•}½¹Ñ•¹Ñ}Á…ÉÑÌ ‰Ý¡…Ð¥Ì¥¸Ñ¡¥ÌÁ¡½Ñ¼üˆ°mÍÑÈ¡¥µœ¥t¤(€€€…ÍÍ•ÉÐ¹½ÐÍ­¥ÁÁ•((€€€…•¹Ð€ô%•¹Ð¹}}¹•Ý}|¡%•¹Ð¤(€€€…•¹Ð¹}Í•ÍÍ¥½¹}‘ˆ€ô5…¥5½¬ ¤(€€€…•¹Ð¹}Í•ÍÍ¥½¹}‘‰}É•…Ñ•€ôQÉÕ”(€€€…•¹Ð¹Í•ÍÍ¥½¹}¥€ô€‰Ì´Äˆ(€€€…•¹Ð¹}±…ÍÑ}™±ÕÍ¡•‘}‘‰}¥‘à€ô€À(€€€…•¹Ð¹}Á•ÉÍ¥ÍÑ}‘¥Í…‰±•€ô…±Í”(€€€…•¹Ð¹}™±ÕÍ¡•‘}‘‰}µ•ÍÍ…•}¥‘Ì€ôÍ•Ð ¤(€€€…•¹Ð¹}™±ÕÍ¡•‘}‘‰}µ•ÍÍ…•}Í•ÍÍ¥½¹}¥€ô9½¹”(€€€…•¹Ð¹}Á•¹‘¥¹}±¥}ÕÍ•É}µ•ÍÍ…”€ô9½¹”(€€€…•¹Ð¹}Á•ÉÍ¥ÍÑ}ÕÍ•É}µ•ÍÍ…•}Ñ¥µ•ÍÑ…µÀ€ô9½¹”(€€€…•¹Ð¹}Á•ÉÍ¥ÍÑ}ÕÍ•É}µ•ÍÍ…•}¥‘à€ô€À(€€€…•¹Ð¹}Á•ÉÍ¥ÍÑ}ÕÍ•É}µ•ÍÍ…•}½Ù•ÉÉ¥‘”€ôÍ•ÉÙ•È¹}‰Õ¥±‘}Á•ÉÍ¥ÍÑ}ÕÍ•É}µ•ÍÍ…” (€€€€€€€€‰Ý¡…Ð¥Ì¥¸Ñ¡¥ÌÁ¡½Ñ¼üˆ°mÍÑÈ¡¥µœ¥t°¹…Ñ¥Ù•}Á…ÉÑÌ(€€€€¤((€€€…•¹Ð¹}™±ÕÍ¡}µ•ÍÍ…•Í}Ñ½}Í•ÍÍ¥½¹}‘ˆ¡mì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè¹…Ñ¥Ù•}Á…ÉÑÍõt°mt¤((€€€ÝÉ¥ÑÑ•¸€ô…•¹Ð¹}Í•ÍÍ¥½¹}‘ˆ¹…ÁÁ•¹‘}µ•ÍÍ…•Í}‰…Ñ ¹…±±}…ÉÌ¹­Ý…ÉÍl‰µ•ÍÍ…•Ì‰ulÁul‰½¹Ñ•¹Ð‰t(€€€…ÍÍ•ÉÐ˜‰¥µ…”éí¥µõ€ˆ¥¸ÝÉ¥ÑÑ•¸(€€€…ÍÍ•ÉÐ€‰Ý¡…Ð¥Ì¥¸Ñ¡¥ÌÁ¡½Ñ¼üˆ¥¸ÝÉ¥ÑÑ•¸(€€€€ŒQ¡”µ½‘•°­••ÁÌÑ¡”Á¥á•±Ì™½ÈÑ¡”É•ÍÐ½˜Ñ¡”Í•ÍÍ¥½¸¸(€€€…ÍÍ•ÉÐ…¹ä¡Á…ÉÐ¹•Ð ‰ÑåÁ”ˆ¤€ôô€‰¥µ…•}ÕÉ°ˆ™½ÈÁ…ÉÐ¥¸…•¹Ð¹}Á•ÉÍ¥ÍÑ}ÕÍ•É}µ•ÍÍ…•}½Ù•ÉÉ¥‘”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}Á…ÍÍ•Í}Á•ÉÍ¥ÍÑ}ÕÍ•É}µ•ÍÍ…•}Ñ½}…•¹Ð¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆˆŒÜÀÜÈÀè}ÉÕ¹}ÁÉ½µÁÑ}ÍÕ‰µ¥ÐµÕÍÐ™½ÉÝ…ÉÑ¡”€¡¥µ…”µÉ•˜µ…Ý…É”¤Á•ÉÍ¥ÍÑ•(€€€ÕÍ•Èµ•ÍÍ…”Ñ¼ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸Ù¥„Á•ÉÍ¥ÍÑ}ÕÍ•É}µ•ÍÍ…”°Í¼Ñ¡”…Ñ•Ý…ä(€€€ÍÑ½É•ÌÑ¡”U$µÉ•½¹¥é…‰±”™½É´¥¹ÍÑ•…½˜Ñ¡”Ù¥Í¥½¸•¹É¥¡µ•¹Ð¸ˆˆˆ(€€€…ÁÑÕÉ•€ôíô((€€€±…ÍÌ}•¹Ðè(€€€€€€€‘•˜ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°ÁÉ½µÁÐ°½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éäõ9½¹”°ÍÑÉ•…µ}…±±‰…¬õ9½¹”°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€…ÁÑÕÉ•‘l‰Á•ÉÍ¥ÍÑ}ÕÍ•É}µ•ÍÍ…”‰t€ô}­Ý…ÉÌ¹•Ð ‰Á•ÉÍ¥ÍÑ}ÕÍ•É}µ•ÍÍ…”ˆ¤(€€€€€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€€€€€‰™¥¹…±}É•ÍÁ½¹Í”ˆè€‰É•Á±äˆ°(€€€€€€€€€€€€€€€€‰µ•ÍÍ…•Ìˆèmì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä‰õt°(€€€€€€€€€€€ô((€€€±…ÍÌ}%µµ•‘¥…Ñ•Q¡É•…è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©}Ñ¡É•…‘}½ÁÑ¥½¹Ì¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð€ôÑ…É•Ð((€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð ¤((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸¡…•¹Ðõ}•¹Ð ¤¤(€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}%µµ•‘¥…Ñ•Q¡É•…¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}ÕÍ…”ˆ°±…µ‰‘„}„èíô¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰É•¹‘•É}µ•ÍÍ…”ˆ°±…µ‰‘„}Ð°}Œè€ˆˆ¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„è9½¹”¤((€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥ˆ°€‰Ñ•áÐˆè€‰¡¤‰ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰É•ÍÕ±Ðˆ¤((€€€€€€€€Œ]¥Ñ¡½ÕÐ…ÑÑ…¡µ•¹ÑÌÑ¡”Á•ÉÍ¥ÍÐ™½É´•ÅÕ…±ÌÑ¡”É…ÜÁÉ½µÁÐ¸(€€€€€€€…ÍÍ•ÉÐ…ÁÑÕÉ•¹•Ð ‰Á•ÉÍ¥ÍÑ}ÕÍ•É}µ•ÍÍ…”ˆ¤€ôô€‰¡¤ˆ(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(((()‘•˜Ñ•ÍÑ}™…±±‰…­}Í•ÍÍ¥½¹}¥¹™½}É•Á½ÉÑÍ}Í•ÍÍ¥½¹}Ý‘}¹½Ñ}±…Õ¹¡}‘¥È¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰±…é¥±äµÉ•ÍÕµ•Í•ÍÍ¥½¸µÕÍÐÉ•Á½ÉÐ%QLÝ½É­ÍÁ…”°¹½ÐÑ¡”…Ñ•Ý…äÌ¸((€€€}™…±±‰…­}Í•ÍÍ¥½¹}¥¹™½€ÕÍ•}‘•™…Õ±Ñ}Í•ÍÍ¥½¹}Ý ¥€ƒŠPÑ¡”‘¥É•Ñ½ÉäÑ¡”(€€€…Ñ•Ý…äÁÉ½•ÍÌ¡…ÁÁ•¹•Ñ¼ÍÑ…ÉÐ¥¸ƒŠPÍ¼Ñ¡”‘•Í­Ñ½À¥±•ÌÁ…¹”Á…¥¹Ñ•Ñ¡”(€€€ÝÉ½¹œÁÉ½©•Ð™½È…¹äÍ•ÍÍ¥½¸É•ÍÕµ•Ý¥Ñ¡½ÕÐ„‰Õ¥±Ð…•¹Ð€ ŒÜÄÈÔÐ¤¸(€€€€ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}‘•™…Õ±Ñ}Í•ÍÍ¥½¹}Ýˆ°±…µ‰‘„è€ˆ½…Ñ•Ý…ä½±…Õ¹ ½‘¥Èˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹¥Ñ}ÁÉ½‰”°€‰‰É…¹ ˆ°±…µ‰‘„Ýè€‰‰ˆ½™•…ÑÕÉ”ˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÁÉ½©•Ñ}¥¹™½}™½É}Ýˆ°±…µ‰‘„Ýè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}É•Í½±Ù•}µ½‘•°ˆ°±…µ‰‘„è€‰Ñ•ÍÐµµ½‘•°ˆ¤((€€€¥¹™¼€ôÍ•ÉÙ•È¹}™…±±‰…­}Í•ÍÍ¥½¹}¥¹™¼¡ì‰Ýˆè€ˆ½ÁÉ½©•ÑÌ½Í•ÍÍ¥½¸µ½Ý¸µÉ•Á¼‰ô¤((€€€…ÍÍ•ÉÐ¥¹™½l‰Ý‰t€ôô€ˆ½ÁÉ½©•ÑÌ½Í•ÍÍ¥½¸µ½Ý¸µÉ•Á¼ˆ(€€€…ÍÍ•ÉÐ¥¹™½l‰‰É…¹ ‰t€ôô€‰‰ˆ½™•…ÑÕÉ”ˆ(()‘•˜Ñ•ÍÑ}™…±±‰…­}Í•ÍÍ¥½¹}¥¹™½}…±Ý…åÍ}•µ¥ÑÍ}‰É…¹ ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰‰É…¹¡€¥Ì…±Ý…åÌÁÉ•Í•¹ÐÍ¼„±¥•¹Ð…¸1H„ÍÑ…±”±…‰•°¸((€€€=µ¥ÑÑ¥¹œÑ¡”­•ä±•™ÐÑ¡”‘•Í­Ñ½ÀÍ¡½Ý¥¹œÑ¡”ÁÉ•Ù¥½ÕÌ½¹Ù•ÉÍ…Ñ¥½¸Ì‰É…¹ (€€€…™Ñ•ÈÍÝ¥Ñ¡¥¹œ¥¹Ñ¼„¹½¸µ¥ÐÍ•ÍÍ¥½¸¸(€€€€ˆˆˆ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}‘•™…Õ±Ñ}Í•ÍÍ¥½¹}Ýˆ°±…µ‰‘„è€ˆ½…Ñ•Ý…ä½±…Õ¹ ½‘¥Èˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹¥Ñ}ÁÉ½‰”°€‰‰É…¹ ˆ°±…µ‰‘„Ýè€ˆˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÁÉ½©•Ñ}¥¹™½}™½É}Ýˆ°±…µ‰‘„Ýè9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}É•Í½±Ù•}µ½‘•°ˆ°±…µ‰‘„è€‰Ñ•ÍÐµµ½‘•°ˆ¤((€€€¥¹™¼€ôÍ•ÉÙ•È¹}™…±±‰…­}Í•ÍÍ¥½¹}¥¹™¼¡ì‰Ýˆè€ˆ½Á±…¥¸½™½±‘•È‰ô¤((€€€…ÍÍ•ÉÐ€‰‰É…¹ ˆ¥¸¥¹™¼(€€€…ÍÍ•ÉÐ¥¹™½l‰‰É…¹ ‰t€ôô€ˆˆ(()	I9!}IM=9%9€ô€‰Ñ¡”Á…É•¹ÐÌ¡…¥¸½˜Ñ¡½Õ¡Ðˆ)	I9!}IM=9%9}=9Q9P€ô€‰Ñ¡”Á…É•¹ÐÌÉ•…Í½¹¥¹œ½¹Ñ•¹Ðˆ)	I9!}IM=9%9}Q%1L€ôl(€€€ì‰ÑåÁ”ˆè€‰É•…Í½¹¥¹œ¹Ñ•áÐˆ°€‰Ñ•áÐˆè€‰­••ÀÑ¡”Á…É•¹ÐÌÁ±…¸ˆ°€‰™½Éµ…Ðˆè€‰Õ¹­¹½Ý¸‰ô)t)	I9!}=a}IM=9%9}%Q5L€ôl(€€€ì‰¥ˆè€‰ÉÍ|Äˆ°€‰ÑåÁ”ˆè€‰É•…Í½¹¥¹œˆ°€‰•¹ÉåÁÑ•‘}½¹Ñ•¹Ðˆè€‰½Á…ÅÕ”µ‰±½ˆ‰ô)t)	I9!}=a}5MM}%Q5L€ôl(€€€ì(€€€€€€€€‰¥ˆè€‰µÍ|Äˆ°(€€€€€€€€‰ÑåÁ”ˆè€‰µ•ÍÍ…”ˆ°(€€€€€€€€‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°(€€€€€€€€‰½¹Ñ•¹Ðˆèmì‰ÑåÁ”ˆè€‰½ÕÑÁÕÑ}Ñ•áÐˆ°€‰Ñ•áÐˆè€‰‘½¹”‰õt°(€€€ô)t(()‘•˜}‰É…¹¡}¡¥ÍÑ½Éä ¤è(€€€É•ÑÕÉ¸l(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰¡•±±¼‰ô°(€€€€€€€ì(€€€€€€€€€€€€‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°(€€€€€€€€€€€€‰½¹Ñ•¹Ðˆè€‰‘½¹”ˆ°(€€€€€€€€€€€€‰É•…Í½¹¥¹œˆè	I9!}IM=9%9°(€€€€€€€€€€€€‰É•…Í½¹¥¹}½¹Ñ•¹Ðˆè	I9!}IM=9%9}=9Q9P°(€€€€€€€€€€€€‰É•…Í½¹¥¹}‘•Ñ…¥±Ìˆè	I9!}IM=9%9}Q%1L°(€€€€€€€€€€€€‰½‘•á}É•…Í½¹¥¹}¥Ñ•µÌˆè	I9!}=a}IM=9%9}%Q5L°(€€€€€€€€€€€€‰½‘•á}µ•ÍÍ…•}¥Ñ•µÌˆè	I9!}=a}5MM}%Q5L°(€€€€€€€ô°(€€€€€€€€ŒQ¥µ•±¥¹”µ…É­•ÈèÉ¥‘•Ì…ÌÉ½±”õÕÍ•È‰ÕÐµÕÍÐ­••À¥ÑÌÑ…œÑ¡É½Õ (€€€€€€€€ŒÑ¡”‰É…¹ ½Áä°½È¥ÐÉ”µ•¹Ñ•ÉÌÑ¡”ÑÉÕ¹…Ñ”½É‘¥¹…°…‘‘É•ÍÌÍÁ…”(€€€€€€€€Œ…Ì„Á¡…¹Ñ½´ÕÍ•ÈÑÕÉ¸…™Ñ•È„É•ÍÑ…ÉÐ€ ŒàÈÜÔØ¤¸(€€€€€€€ì(€€€€€€€€€€€€‰É½±”ˆè€‰ÕÍ•Èˆ°(€€€€€€€€€€€€‰½¹Ñ•¹Ðˆè€‰mMåÍÑ•´èÁ•ÉÍ½¹…±¥Ñä¡…¹•‘tˆ°(€€€€€€€€€€€€‰‘¥ÍÁ±…å}­¥¹ˆè€‰Á•ÉÍ½¹…±¥Ñå}ÍÝ¥Ñ ˆ°(€€€€€€€ô°(€€€t(()‘•˜}‰É…¹¡•‘}µ…É­•È¡‘ˆ°Í•ÍÍ¥½¹}­•ä¤è(€€€É•ÑÕÉ¸¹•áÐ (€€€€€€€€ (€€€€€€€€€€€´(€€€€€€€€€€€™½È´¥¸‘ˆ¹•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸¡Í•ÍÍ¥½¹}­•ä¤(€€€€€€€€€€€¥˜´¹•Ð ‰‘¥ÍÁ±…å}­¥¹ˆ¤€ôô€‰Á•ÉÍ½¹…±¥Ñå}ÍÝ¥Ñ ˆ(€€€€€€€€¤°(€€€€€€€9½¹”°(€€€€¤(()‘•˜}‰É…¹¡•‘}…ÍÍ¥ÍÑ…¹Ð¡‘ˆ°Í•ÍÍ¥½¹}­•ä¤è(€€€É•ÑÕÉ¸¹•áÐ (€€€€€€€´(€€€€€€€™½È´¥¸‘ˆ¹•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸¡Í•ÍÍ¥½¹}­•ä¤(€€€€€€€¥˜µl‰É½±”‰t€ôô€‰…ÍÍ¥ÍÑ…¹Ðˆ(€€€€¤(()‘•˜Ñ•ÍÑ}Á•ÉÍ¥ÍÑ}‰É…¹¡}Í••‘}­••ÁÍ}É•…Í½¹¥¹}™¥•±‘Ì¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰Q¡”Í••ÝÉ¥Ñ”µÕÍÐ…ÉÉäÑ¡”Á…É•¹ÐÌÉ•…Í½¹¥¹œ™¥•±‘Ì¸((€€€‰É…¹ ¥Ì„‘É…™ÐÕ¹Ñ¥°¥ÑÌ™¥ÉÍÐÍÕ‰µ¥Ð°Í¼Ñ¡¥Ì¥ÌÑ¡”½¹±äÝÉ¥Ñ”Ñ¡…Ð(€€€•Ù•ÈÁ•ÉÍ¥ÍÑÌÑ¡”½Á¥•ÑÉ…¹ÍÉ¥ÁÐ¸A•ÉÍ¥ÍÑ¥¹œÉ½±”½½¹Ñ•¹Ð…±½¹”±•™ÐÑ¡”(€€€‰É…¹ É•ÍÕµ¥¹œÝ¥Ñ¡½ÕÐÑ¡”Á…É•¹ÐÌÉ•…Í½¹¥¹œ°ÁÉ•Í•ÉÙ•Ñ¡¥¹­¥¹œ‰±½­Ì½È(€€€½‘•à•¹ÉåÁÑ•µÉ•…Í½¹¥¹œ½µ•ÍÍ…”µ¥Ñ•´½¹Ñ¥¹Õ…Ñ¥½¸ÍÑ…Ñ”¸(€€€€ˆˆˆ(€€€™É½´¡•Éµ•Í}ÍÑ…Ñ”¥µÁ½ÉÐM•ÍÍ¥½¹((€€€‘ˆ€ôM•ÍÍ¥½¹¡‘‰}Á…Ñ õÑµÁ}Á…Ñ €¼€‰ÍÑ…Ñ”¹‘ˆˆ¤(€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸ (€€€€€€€Í•ÍÍ¥½¹}­•äô‰‰É…¹ µ­•äˆ°(€€€€€€€Á…É•¹Ñ}Í•ÍÍ¥½¹}¥ô‰Á…É•¹Ðµ­•äˆ°(€€€€€€€¡¥ÍÑ½Éäõ}‰É…¹¡}¡¥ÍÑ½Éä ¤°(€€€€€€€Í••‘•õQÉÕ”°€€ŒÍÑ…µÁ•‰äÍ•ÍÍ¥½¸¹É•…Ñ”èÑ¡¥Ì¡¥ÍÑ½Éä•á¥ÍÑÌ½¹±ä¥¸µ•µ½Éä(€€€€¤(€€€ÑÉäè(€€€€€€€‘ˆ¹É•…Ñ•}Í•ÍÍ¥½¸ ‰‰É…¹ µ­•äˆ°Í½ÕÉ”ô‰ÑÕ¤ˆ¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è‘ˆ¤((€€€€€€€Í•ÉÙ•È¹}Á•ÉÍ¥ÍÑ}‰É…¹¡}Í••¡Í•ÍÍ¥½¸¤((€€€€€€€…ÍÍ¥ÍÑ…¹Ð€ô}‰É…¹¡•‘}…ÍÍ¥ÍÑ…¹Ð¡‘ˆ°€‰‰É…¹ µ­•äˆ¤(€€€€€€€…ÍÍ•ÉÐ…ÍÍ¥ÍÑ…¹Ñl‰É•…Í½¹¥¹œ‰t€ôô	I9!}IM=9%9(€€€€€€€…ÍÍ•ÉÐ…ÍÍ¥ÍÑ…¹Ñl‰É•…Í½¹¥¹}½¹Ñ•¹Ð‰t€ôô	I9!}IM=9%9}=9Q9P(€€€€€€€…ÍÍ•ÉÐ…ÍÍ¥ÍÑ…¹Ñl‰É•…Í½¹¥¹}‘•Ñ…¥±Ì‰t€ôô	I9!}IM=9%9}Q%1L(€€€€€€€…ÍÍ•ÉÐ…ÍÍ¥ÍÑ…¹Ñl‰½‘•á}É•…Í½¹¥¹}¥Ñ•µÌ‰t€ôô	I9!}=a}IM=9%9}%Q5L(€€€€€€€…ÍÍ•ÉÐ…ÍÍ¥ÍÑ…¹Ñl‰½‘•á}µ•ÍÍ…•}¥Ñ•µÌ‰t€ôô	I9!}=a}5MM}%Q5L(€€€€€€€µ…É­•È€ô}‰É…¹¡•‘}µ…É­•È¡‘ˆ°€‰‰É…¹ µ­•äˆ¤(€€€€€€€…ÍÍ•ÉÐµ…É­•È¥Ì¹½Ð9½¹”°€ (€€€€€€€€€€€€‰Ñ¡”‰É…¹ Í••‘É½ÁÁ•‘¥ÍÁ±…å}­¥¹èÑ¡”µ…É­•ÈÉ”µ•¹Ñ•É•Ñ¡”€ˆ(€€€€€€€€€€€€‰ÑÉÕ¹…Ñ”½É‘¥¹…°…‘‘É•ÍÌÍÁ…”…Ì„Á¡…¹Ñ½´ÕÍ•ÈÑÕÉ¸€ ŒàÈÜÔØ¤ˆ(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍ¥½¹l‰}‰É…¹¡}Í••‘}Á•ÉÍ¥ÍÑ•‰t¥ÌQÉÕ”(€€€™¥¹…±±äè(€€€€€€€‘ˆ¹±½Í” ¤(()‘•˜Ñ•ÍÑ}Í•ÍÍ¥½¹}‰É…¹¡}­••ÁÍ}É•…Í½¹¥¹}™¥•±‘Ì¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰Í•ÍÍ¥½¸¹‰É…¹ ½Á¥•ÌÑ¡”±¥Ù”ÑÉ…¹ÍÉ¥ÁÐÝ¥Ñ ¥ÑÌÉ•…Í½¹¥¹œ™¥•±‘Ì¸((€€€M…µ”‘É½À…ÌÑ¡”Í••Á…Ñ èÑ¡”½Áä±½½ÀÝÉ½Ñ”½¹±äÉ½±”½½¹Ñ•¹Ð°Í¼Ñ¡”(€€€¹•ÜÍ•ÍÍ¥½¸É½ÜÉ•Á±…å•Ý¥Ñ¡½ÕÐÑ¡”É•…Í½¹¥¹œ½¹Ñ•áÐÑ¡”Á…É•¹Ð¡…¸(€€€€ˆˆˆ(€€€™É½´¡•Éµ•Í}ÍÑ…Ñ”¥µÁ½ÉÐM•ÍÍ¥½¹((€€€‘ˆ€ôM•ÍÍ¥½¹¡‘‰}Á…Ñ õÑµÁ}Á…Ñ €¼€‰ÍÑ…Ñ”¹‘ˆˆ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Í¥‰t€ô}Í•ÍÍ¥½¸¡¡¥ÍÑ½Éäõ}‰É…¹¡}¡¥ÍÑ½Éä ¤¤(€€€ÑÉäè(€€€€€€€‘ˆ¹É•…Ñ•}Í•ÍÍ¥½¸ ‰Í•ÍÍ¥½¸µ­•äˆ°Í½ÕÉ”ô‰ÑÕ¤ˆ¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è‘ˆ¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}¹•Ý}Í•ÍÍ¥½¹}­•äˆ°±…µ‰‘„è€‰‰É…¹ µ­•äˆ¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„Í¥°Í•ÍÍ¥½¸è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Ý…¥Ñ}…•¹Ðˆ°±…µ‰‘„Í•ÍÍ¥½¸°É¥è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€Í•ÉÙ•È°€‰}±…¥µ}…Ñ¥Ù•}Í•ÍÍ¥½¹}Í±½Ðˆ°±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌè€¡9½¹”°9½¹”¤(€€€€€€€€¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}É•Í½±Ù•}µ½‘•°ˆ°±…µ‰‘„è€‰Ñ•ÍÐµµ½‘•°ˆ¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}Ýˆ°±…µ‰‘„Í•ÍÍ¥½¸èÍÑÈ¡ÑµÁ}Á…Ñ ¤¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€€€€€Í•ÉÙ•È°€‰}µ…­•}…•¹Ðˆ°±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌèÑåÁ•Ì¹M¥µÁ±•9…µ•ÍÁ…” ¤(€€€€€€€€¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}¥¹¥Ñ}Í•ÍÍ¥½¸ˆ°±…µ‰‘„€©…ÉÌ°€¨©­Ý…ÉÌè9½¹”¤((€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì‰¥ˆè€ˆÄˆ°€‰µ•Ñ¡½ˆè€‰Í•ÍÍ¥½¸¹‰É…¹ ˆ°€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆè€‰Í¥‰õô(€€€€€€€€¤((€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÉ•ÍÀ¹•Ð •ÉÉ½Èœ¥ôˆ(€€€€€€€…ÍÍ•ÉÐ‘ˆ¹•Ñ}Í•ÍÍ¥½¹}Ñ¥Ñ±•}Í½ÕÉ” ‰‰É…¹ µ­•äˆ¤€ôôM•ÍÍ¥½¹¹Q%Q1}M=UI}I%Y(€€€€€€€…ÍÍ¥ÍÑ…¹Ð€ô}‰É…¹¡•‘}…ÍÍ¥ÍÑ…¹Ð¡‘ˆ°€‰‰É…¹ µ­•äˆ¤(€€€€€€€…ÍÍ•ÉÐ…ÍÍ¥ÍÑ…¹Ñl‰É•…Í½¹¥¹œ‰t€ôô	I9!}IM=9%9(€€€€€€€…ÍÍ•ÉÐ…ÍÍ¥ÍÑ…¹Ñl‰É•…Í½¹¥¹}½¹Ñ•¹Ð‰t€ôô	I9!}IM=9%9}=9Q9P(€€€€€€€…ÍÍ•ÉÐ…ÍÍ¥ÍÑ…¹Ñl‰É•…Í½¹¥¹}‘•Ñ…¥±Ì‰t€ôô	I9!}IM=9%9}Q%1L(€€€€€€€…ÍÍ•ÉÐ…ÍÍ¥ÍÑ…¹Ñl‰½‘•á}É•…Í½¹¥¹}¥Ñ•µÌ‰t€ôô	I9!}=a}IM=9%9}%Q5L(€€€€€€€…ÍÍ•ÉÐ…ÍÍ¥ÍÑ…¹Ñl‰½‘•á}µ•ÍÍ…•}¥Ñ•µÌ‰t€ôô	I9!}=a}5MM}%Q5L(€€€€€€€µ…É­•È€ô}‰É…¹¡•‘}µ…É­•È¡‘ˆ°€‰‰É…¹ µ­•äˆ¤(€€€€€€€…ÍÍ•ÉÐµ…É­•È¥Ì¹½Ð9½¹”°€ (€€€€€€€€€€€€‰Í•ÍÍ¥½¸¹‰É…¹ ‘É½ÁÁ•‘¥ÍÁ±…å}­¥¹èÑ¡”µ…É­•ÈÉ”µ•¹Ñ•É•Ñ¡”€ˆ(€€€€€€€€€€€€‰ÑÉÕ¹…Ñ”½É‘¥¹…°…‘‘É•ÍÌÍÁ…”…Ì„Á¡…¹Ñ½´ÕÍ•ÈÑÕÉ¸€ ŒàÈÜÔØ¤ˆ(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Í¥ˆ°9½¹”¤(€€€€€€€‘ˆ¹±½Í” ¤(((ŒƒŠRŠR }Í…Ù•}™œ½µµ•¹ÐµÁÉ•Í•ÉÙ…Ñ¥½¸É•É•ÍÍ¥½¸Ñ•ÍÑÌƒŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠR (Œ(ŒU¹Ñ¥°ùµ¥´ÈÀÈØ}Í…Ù•}™œÕÍ•å…µ°¹Í…™•}‘ÕµÀ½¸„‘••Àµ±½…‘•½¹™¥œ‘¥Ð¸(ŒÙ•Éä€½Á•ÉÍ½¹…±¥Ñä€¡½È€½É•…Í½¹¥¹œ°€½‘•Ñ…¥±Í}µ½‘”°€½ÁÉ½µÁÐ°€¸¸¸¤ÝÉ¥Ñ”(ŒÍ¥±•¹Ñ±äÉ•ÝÉ½Ñ”ø¼¹¡•Éµ•Ì½½¹™¥œ¹å…µ°Ñ½ÀµÑ¼µ‰½ÑÑ½´ƒŠPÑ½Àµ±•Ù•°­•åÌ(ŒÉ•½É‘•É•…±Á¡…‰•Ñ¥…±±ä°½µµ•¹ÑÌÍÑÉ¥ÁÁ•°­…½µ½©¤½¡¥¹•Í”¥¸ÍÑ½É•(ŒÁ•ÉÍ½¹…±¥ÑäÁÉ½µÁÑÌµ…¹±•Ñ¼qÕaaa`•Í…Á•Ì¸Q¡•Í”Ñ•ÍÑÌÁ¥¸Ñ¡”(ŒÕÍ•ÈµÙ¥Í¥‰±”‰•¡…Ù¥½È½˜Ñ¡”½µµ•¹ÐµÁÉ•Í•ÉÙ¥¹œÉ•Á±…•µ•¹ÐÍ¼Ý”‘½¸Ð(ŒÉ•É•ÍÌ¸(()‘•˜Ñ•ÍÑ}Í…Ù•}™}ÁÉ•Í•ÉÙ•Í}ÕÍ•É}½µµ•¹ÑÌ¡ÑµÁ}Á…Ñ °µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰Q¡”QU$…Ñ•Ý…äµÕÍÐ¹½ÐÍÑÉ¥ÀÕÍ•Èµ•‘¥Ñ•½µµ•¹ÑÌ½¸Í•ÑÑ¥¹œÝÉ¥Ñ•Ì¸ˆˆˆ(€€€™}Á…Ñ €ôÑµÁ}Á…Ñ €¼€‰½¹™¥œ¹å…µ°ˆ(€€€™}Á…Ñ ¹ÝÉ¥Ñ•}Ñ•áÐ (€€€€€€€€ˆŒÑ½À½˜™¥±”¹½Ñ•q¸ˆ(€€€€€€€€‰µ½‘•°éq¸ˆ(€€€€€€€€ˆ€€ŒÁÉ½Ù¥‘•ÈÉ…Ñ¥½¹…±•q¸ˆ(€€€€€€€€ˆ€‘•™…Õ±Ðè±…Õ‘”µ½ÁÕÌ´Ð´Ýq¸ˆ(€€€€€€€€‰‘¥ÍÁ±…äéq¸ˆ(€€€€€€€€ˆ€Í­¥¸è‘•™…Õ±Ð€€ŒÑÉ…¥±¥¹œÍ­¥¸¹½Ñ•q¸ˆ°(€€€€€€€•¹½‘¥¹œô‰ÕÑ˜´àˆ°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}¡•Éµ•Í}¡½µ”ˆ°ÑµÁ}Á…Ñ ¤((€€€Í•ÉÙ•È¹}Í…Ù•}™œ (€€€€€€€ì(€€€€€€€€€€€€‰µ½‘•°ˆèì‰‘•™…Õ±Ðˆè€‰±…Õ‘”µ½ÁÕÌ´Ð´Ü‰ô°(€€€€€€€€€€€€‰‘¥ÍÁ±…äˆèì‰Í­¥¸ˆè€‰µ½¹¼‰ô°(€€€€€€€ô(€€€€¤((€€€Ñ•áÐ€ô™}Á…Ñ ¹É•…‘}Ñ•áÐ¡•¹½‘¥¹œô‰ÕÑ˜´àˆ¤(€€€…ÍÍ•ÉÐ€ˆŒÑ½À½˜™¥±”¹½Ñ”ˆ¥¸Ñ•áÐ(€€€…ÍÍ•ÉÐ€ˆŒÁÉ½Ù¥‘•ÈÉ…Ñ¥½¹…±”ˆ¥¸Ñ•áÐ(€€€…ÍÍ•ÉÐ€ˆŒÑÉ…¥±¥¹œÍ­¥¸¹½Ñ”ˆ¥¸Ñ•áÐ((€€€¥µÁ½ÉÐ¡•Éµ•Í}å…µ°…Ì}å…µ°((€€€Á…ÉÍ•€ô}å…µ°¹Í…™•}±½…¡Ñ•áÐ¤(€€€…ÍÍ•ÉÐÁ…ÉÍ•‘l‰‘¥ÍÁ±…ä‰ul‰Í­¥¸‰t€ôô€‰µ½¹¼ˆ(()‘•˜Ñ•ÍÑ}Í…Ù•}™}ÁÉ•Í•ÉÙ•Í}Ñ½Á}±•Ù•±}­•å}½É‘•È¡ÑµÁ}Á…Ñ °µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰Q½Àµ±•Ù•°­•åÌµÕÍÐ­••ÀÑ¡”™¥±”Ì¡…¹µ•‘¥Ñ•½É‘•É¥¹œ¥¹ÍÑ•…½˜(€€€‰•¥¹œÉ•ÝÉ¥ÑÑ•¸…±Á¡…‰•Ñ¥…±±ä‰äÑ¡”Õ¹‘•É±å¥¹œe50‘ÕµÁ•È¸ˆˆˆ(€€€™}Á…Ñ €ôÑµÁ}Á…Ñ €¼€‰½¹™¥œ¹å…µ°ˆ(€€€™}Á…Ñ ¹ÝÉ¥Ñ•}Ñ•áÐ (€€€€€€€€‰µ½‘•°éq¸ˆ(€€€€€€€€ˆ€‘•™…Õ±Ðè±…Õ‘”µ½ÁÕÌ´Ð´Ýq¸ˆ(€€€€€€€€‰Ñ½½±Í•ÑÌéq¸ˆ(€€€€€€€€ˆ€€´¡•Éµ•Ìµ±¥q¸ˆ(€€€€€€€€‰…•¹Ðéq¸ˆ(€€€€€€€€ˆ€µ…á}ÑÕÉ¹Ìè€äÁq¸ˆ(€€€€€€€€‰‘¥ÍÁ±…äéq¸ˆ(€€€€€€€€ˆ€Í­¥¸è‘•™…Õ±Ñq¸ˆ°(€€€€€€€•¹½‘¥¹œô‰ÕÑ˜´àˆ°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}¡•Éµ•Í}¡½µ”ˆ°ÑµÁ}Á…Ñ ¤((€€€€Œ…±±•ÈÌ‘¥Ð¥Ñ•É…Ñ¥½¸½É‘•È¥Ì¥¹Ñ•¹Ñ¥½¹…±±ä…±Á¡…‰•Ñ¥…°Ñ¼½¹™¥É´(€€€€ŒÑ¡”¡•±Á•È½¹ÍÕ±ÑÌ‘¥Í¬½É‘•È°¹½Ð…±±•È½É‘•È¸(€€€Í•ÉÙ•È¹}Í…Ù•}™œ (€€€€€€€ì(€€€€€€€€€€€€‰…•¹Ðˆèì‰µ…á}ÑÕÉ¹Ìˆè€äÁô°(€€€€€€€€€€€€‰‘¥ÍÁ±…äˆèì‰Í­¥¸ˆè€‰µ½¹¼‰ô°(€€€€€€€€€€€€‰µ½‘•°ˆèì‰‘•™…Õ±Ðˆè€‰±…Õ‘”µ½ÁÕÌ´Ð´Ü‰ô°(€€€€€€€€€€€€‰Ñ½½±Í•ÑÌˆèl‰¡•Éµ•Ìµ±¤‰t°(€€€€€€€ô(€€€€¤((€€€Ñ•áÐ€ô™}Á…Ñ ¹É•…‘}Ñ•áÐ¡•¹½‘¥¹œô‰ÕÑ˜´àˆ¤(€€€Ñ½Á}­•åÌ€ôl(€€€€€€€±¥¹”¹ÍÁ±¥Ð ˆèˆ°€Ä¥lÁt(€€€€€€€™½È±¥¹”¥¸Ñ•áÐ¹ÍÁ±¥Ñ±¥¹•Ì ¤(€€€€€€€¥˜±¥¹”…¹¹½Ð±¥¹”¹ÍÑ…ÉÑÍÝ¥Ñ  ˆ€ˆ¤…¹¹½Ð±¥¹”¹ÍÑ…ÉÑÍÝ¥Ñ  ˆ´ˆ¤(€€€€€€€…¹¹½Ð±¥¹”¹ÍÑ…ÉÑÍÝ¥Ñ  ˆŒˆ¤(€€€t(€€€…ÍÍ•ÉÐÑ½Á}­•åÌ€ôôl‰µ½‘•°ˆ°€‰Ñ½½±Í•ÑÌˆ°€‰…•¹Ðˆ°€‰‘¥ÍÁ±…ä‰t(()‘•˜Ñ•ÍÑ}Í…Ù•}™}­••ÁÍ}Õ¹¥½‘•}Á•ÉÍ½¹…±¥Ñ¥•Í}É•…‘…‰±”¡ÑµÁ}Á…Ñ °µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰Q¡”…Ñ¥É°½­…Ý…¥¤Á•ÉÍ½¹…±¥ÑäÁÉ½µÁÑÌµÕÍÐÍÑ…äÉ•…‘…‰±”½¸‘¥Í¬(€€€¥¹ÍÑ•…½˜‰•¥¹œqqÕaaa`µ•Í…Á•½¸•Ù•ÉäÕ¹É•±…Ñ•Í•ÑÑ¥¹œÝÉ¥Ñ”¸ˆˆˆ(€€€™}Á…Ñ €ôÑµÁ}Á…Ñ €¼€‰½¹™¥œ¹å…µ°ˆ(€€€™}Á…Ñ ¹ÝÉ¥Ñ•}Ñ•áÐ (€€€€€€€€‰…•¹Ðéq¸ˆ(€€€€€€€€ˆ€Á•ÉÍ½¹…±¥Ñ¥•Ìéq¸ˆ(€€€€€€€€ˆ€€€…Ñ¥É°èp‰¹å„€ õ{¾ö—>'¾ö•xô¤ƒ’öƒ––õp‰q¸ˆ(€€€€€€€€‰‘¥ÍÁ±…äéq¸ˆ(€€€€€€€€ˆ€Í­¥¸è‘•™…Õ±Ñq¸ˆ°(€€€€€€€•¹½‘¥¹œô‰ÕÑ˜´àˆ°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}¡•Éµ•Í}¡½µ”ˆ°ÑµÁ}Á…Ñ ¤((€€€€ŒM¥µÕ±…Ñ”…¸Õ¹É•±…Ñ•€½Í­¥¸ÝÉ¥Ñ”ƒŠPµÕÍÐ¹½Ð½ÉÉÕÁÐÑ¡”…Ñ¥É°(€€€€ŒÁ•ÉÍ½¹…±¥ÑäÍÑÉ¥¹œÍ¥ÑÑ¥¹œ¥¸…•¹Ð¹Á•ÉÍ½¹…±¥Ñ¥•Ì¸(€€€Í•ÉÙ•È¹}Í…Ù•}™œ (€€€€€€€ì(€€€€€€€€€€€€‰…•¹Ðˆèì‰Á•ÉÍ½¹…±¥Ñ¥•Ìˆèì‰…Ñ¥É°ˆè€‰¹å„€ õ{¾ö—>'¾ö•xô¤ƒ’öƒ––ô‰õô°(€€€€€€€€€€€€‰‘¥ÍÁ±…äˆèì‰Í­¥¸ˆè€‰µ½¹¼‰ô°(€€€€€€€ô(€€€€¤((€€€Ñ•áÐ€ô™}Á…Ñ ¹É•…‘}Ñ•áÐ¡•¹½‘¥¹œô‰ÕÑ˜´àˆ¤(€€€…ÍÍ•ÉÐ€‹’öƒ––ôˆ¥¸Ñ•áÐ(€€€…ÍÍ•ÉÐ€ˆ õ{¾ö—>'¾ö•xô¤ˆ¥¸Ñ•áÐ(€€€…ÍÍ•ÉÐ€‰qqÔÑ˜ØÀˆ¹½Ð¥¸Ñ•áÐ(()‘•˜Ñ•ÍÑ}Á•ÉÍ½¹…±¥Ñå}µ…É­•É}‘½•Í}¹½Ñ}Í¡¥™Ñ}ÑÉÕ¹…Ñ•}½É‘¥¹…°¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰Á•ÉÍ½¹…±¥ÑäÁ¥Ù½ÐµÕÍÐ¹½Ð½ÕÁä„Í±½Ð¥¸Ñ¡”½É‘¥¹…°…‘‘É•ÍÌÍÁ…”¸((€€€}…ÁÁ±å}Á•ÉÍ½¹…±¥Ñå}Ñ½}Í•ÍÍ¥½¹€¥¹©•ÑÌ¥ÑÌÁ¥Ù½Ð…ÌÉ½±”õÕÍ•É€Í¼(€€€ÍÑÉ¥Ð=Á•¹$µ½µÁ…Ñ¥‰±”ÁÉ½Ù¥‘•ÉÌ…•ÁÐ¥Ðµ¥µ½¹Ù•ÉÍ…Ñ¥½¸¸U¹Ñ…•°Ñ¡”(€€€½É‘¥¹…°™¥±Ñ•È€¡É½±”€ôô€‰ÕÍ•Èˆ…¹¹½Ð‘¥ÍÁ±…å}­¥¹‘€¤½Õ¹Ñ•¥Ð…Ì„(€€€É•…°ÕÍ•ÈÑÕÉ¸Ý¡¥±”¹¼±¥•¹ÐÉ•¹‘•ÉÌ¥Ð…Ì½¹”°Í¼•Ù•ÉäÉ•Ý¥¹¥ÍÍÕ•(€€€…™Ñ•È„Á•ÉÍ½¹…±¥Ñä¡…¹”É•Í½±Ù•½¹”ÑÕÉ¸Ñ½¼•…É±ä…¹(€€€É•Á±…•}µ•ÍÍ…•Ì ¥€¡…Éµ‘•±•Ñ•Ñ¡”•áÑÉ„ÍÁ…¸€ ŒàÈÜÔØ°Ñ¡¥É½ÕÉÉ•¹”(€€€…™Ñ•È€ŒÜÀÔÄØ€¼€ŒàÀÜØÌ¤¸((€€€Q¡”Í¥‰±¥¹œÑ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}ÑÉÕ¹…Ñ•}½É‘¥¹…±}Í­¥ÁÍ}‘¥ÍÁ±…å}­¥¹‘}É½ÝÍ€(€€€Á¥¹ÌÑ¡”™¥±Ñ•È¥ÑÍ•±˜ìÑ¡¥Ì½¹”Á¥¹ÌÑ¡”ÁÉ½‘Õ•È°Ý¡¥ ¥ÌÝ¡•É”Ñ¡”(€€€¥¹Ù…É¥…¹ÐÝ…Ì…ÑÕ…±±ä‰É½­•¸¸(€€€€ˆˆˆ((€€€±…ÍÌ}•¹Ðè(€€€€€€€‘•˜ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°ÁÉ½µÁÐ°½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éäõ9½¹”°ÍÑÉ•…µ}…±±‰…¬õ9½¹”°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€€€€€‰™¥¹…±}É•ÍÁ½¹Í”ˆè€‰É•Á±äˆ°(€€€€€€€€€€€€€€€€‰µ•ÍÍ…•Ìˆèl(€€€€€€€€€€€€€€€€€€€€¨¡½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éä½Èmt¤°(€€€€€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹ÐˆèÁÉ½µÁÑô°(€€€€€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä‰ô°(€€€€€€€€€€€€€€€t°(€€€€€€€€€€€ô((€€€±…ÍÌ}%µµ•‘¥…Ñ•Q¡É•…è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©}Ñ¡É•…‘}½ÁÑ¥½¹Ì¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð€ôÑ…É•Ð((€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð ¤((€€€±…ÍÌ}MÑÕ‰ˆè(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹É•Á±…•€ômt((€€€€€€€‘•˜•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°€©}…ÉÌ°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€É•ÑÕÉ¸mt((€€€€€€€‘•˜É•Á±…•}µ•ÍÍ…•Ì (€€€€€€€€€€€Í•±˜°(€€€€€€€€€€€Í•ÍÍ¥½¹}¥°(€€€€€€€€€€€µ•ÍÍ…•Ì°(€€€€€€€€€€€…Ñ¥Ù•}½¹±äõ…±Í”°(€€€€€€€€€€€…É¡¥Ù•}‘É½ÁÁ•õ…±Í”°(€€€€€€€€€€€É•©•Ñ}…Ñ¥Ù•}ÑÕÉ¹}±•…Í”õ…±Í”°(€€€€€€€€¤è(€€€€€€€€€€€Í•±˜¹É•Á±…•¹…ÁÁ•¹ ¡Í•ÍÍ¥½¹}¥°±¥ÍÐ¡µ•ÍÍ…•Ì¤¤¤((€€€Í•ÍÍ¥½¸€ô}Í•ÍÍ¥½¸ (€€€€€€€…•¹Ðõ}•¹Ð ¤°(€€€€€€€¡¥ÍÑ½Éäõl(€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐ‰ô°(€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐÉ•Á±ä‰ô°(€€€€€€€t°(€€€€¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Á•ÉÍ½¹…±¥Ñäµ½É‘¥¹…°µÍ¥‰t€ôÍ•ÍÍ¥½¸(€€€ÍÑÕ‰}‘ˆ€ô}MÑÕ‰ˆ ¤((€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„€©„°€¨©¬èíô¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„è9½¹”¤((€€€€€€€€ŒI•…°ÁÉ½‘ÕÑ¥½¸¥¹©•Ñ¥½¸Á½¥¹ÐƒŠP¹½Ð„¡…¹µÝÉ¥ÑÑ•¸µ…É­•È‘¥Ð¸(€€€€€€€Í•ÉÙ•È¹}…ÁÁ±å}Á•ÉÍ½¹…±¥Ñå}Ñ½}Í•ÍÍ¥½¸ (€€€€€€€€€€€€‰Á•ÉÍ½¹…±¥Ñäµ½É‘¥¹…°µÍ¥ˆ°Í•ÍÍ¥½¸°€‰Ñ…±¬±¥­”„Á¥É…Ñ”ˆ°€‰Á¥É…Ñ”ˆ(€€€€€€€€¤((€€€€€€€µ…É­•È€ôÍ•ÍÍ¥½¹l‰¡¥ÍÑ½Éä‰ul´Åt(€€€€€€€…ÍÍ•ÉÐµ…É­•Él‰É½±”‰t€ôô€‰ÕÍ•Èˆ°€‰ÁÉ½Ù¥‘•È½µÁ…Ñ¥‰¥±¥ÑäèÁ¥Ù½ÐÉ¥‘•Ì…Ì„ÕÍ•ÈÑÕÉ¸ˆ((€€€€€€€€ŒQÝ¼µ½É”É•…°ÑÕÉ¹Ì±…¹…™Ñ•ÈÑ¡”Á•ÉÍ½¹…±¥Ñä¡…¹”¸(€€€€€€€Í•ÍÍ¥½¹l‰¡¥ÍÑ½Éä‰t¹•áÑ•¹ (€€€€€€€€€€€l(€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹‰ô°(€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹É•Á±ä‰ô°(€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Ñ¡¥É‰ô°(€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰Ñ¡¥ÉÉ•Á±ä‰ô°(€€€€€€€€€€€t(€€€€€€€€¤(€€€€€€€¡¥ÍÑ½Éå}‰•™½É”€ô±¥ÍÐ¡Í•ÍÍ¥½¹l‰¡¥ÍÑ½Éä‰t¤(€€€€€€€Ñ¡¥É‘}¥¹‘•à€ô¡¥ÍÑ½Éå}‰•™½É”¹¥¹‘•à¡ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Ñ¡¥É‰ô¤((€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}%µµ•‘¥…Ñ•Q¡É•…¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}ÕÍ…”ˆ°±…µ‰‘„}„èíô¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰É•¹‘•É}µ•ÍÍ…”ˆ°±…µ‰‘„}Ð°}Œè€ˆˆ¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„èÍÑÕ‰}‘ˆ¤((€€€€€€€€ŒQ¡”±¥•¹Ð½Õ¹ÑÌÑ¡É•”ÕÍ•È‰Õ‰‰±•Ì€¡™¥ÉÍÐôÀ°Í•½¹ôÄ°Ñ¡¥ÉôÈ¤ƒŠP(€€€€€€€€Œ¥Ð¹•Ù•ÈÍ••ÌÑ¡”Á¥Ù½Ð¸I•Ý¥¹‘¥¹œÑ¼€‰Ñ¡¥ÉˆµÕÍÐÕÐ•á…Ñ±äÑ¡•É”¸(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰Á•ÉÍ½¹…±¥Ñäµ½É‘¥¹…°µÍ¥ˆ°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰Ñ¡¥É°É•Ý½É‘•ˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}ÕÍ•É}½É‘¥¹…°ˆè€È°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÉ•ÍÀ¹•Ð •ÉÉ½Èœ¥ôˆ((€€€€€€€•áÁ•Ñ•€ô¡¥ÍÑ½Éå}‰•™½É•léÑ¡¥É‘}¥¹‘•át(€€€€€€€…ÍÍ•ÉÐÍÑÕ‰}‘ˆ¹É•Á±…•€ôôl ‰Í•ÍÍ¥½¸µ­•äˆ°•áÁ•Ñ•¥t°€ (€€€€€€€€€€€€‰Ñ¡”Á¥Ù½ÐÍ¡¥™Ñ•Ñ¡”½É‘¥¹…°èÑ¡”ÕÐ±…¹‘•…Ð€ˆ(€€€€€€€€€€€˜‰í±•¸¡ÍÑÕ‰}‘ˆ¹É•Á±…•‘lÁulÅt¤¥˜ÍÑÕ‰}‘ˆ¹É•Á±…••±Í”9½¹•ô¥¹ÍÑ•…½˜íÑ¡¥É‘}¥¹‘•áôˆ(€€€€€€€€¤(€€€€€€€€ŒQ¡”ÑÕÉ¸‰•™½É”Ñ¡”Ñ…É•ÐµÕÍÐÍÕÉÙ¥Ù”ƒŠPÑ¡…Ð¥ÌÑ¡”ÍÁ…¸Ñ¡”Ñ¡É•”(€€€€€€€€ŒÉ•Á½ÉÑ•¥¹¥‘•¹ÑÌ±½ÍÐ¸(€€€€€€€…ÍÍ•ÉÐì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹‰ô¥¸•áÁ•Ñ•(€€€€€€€€Œ¹Ñ¡”µ•¡…¹¥Í´Ñ¡…Ð­••ÁÌ¥Ð½ÕÐ½˜Ñ¡”…‘‘É•ÍÌÍÁ…”°Í¼„™ÕÑÕÉ”(€€€€€€€€ŒÁÉ½‘Õ•È…¹¹½ÐÉ•É•ÍÌÑ¡¥Ì‰ä‘É½ÁÁ¥¹œÑ¡”Ñ…œ¸(€€€€€€€…ÍÍ•ÉÐµ…É­•È¹•Ð ‰‘¥ÍÁ±…å}­¥¹ˆ¤°€ (€€€€€€€€€€€€‰…¸Õ¹Ñ…•É½±”õÕÍ•ÈÁ¥Ù½ÐÍ¥±•¹Ñ±ä½¹ÍÕµ•Ì„ÑÉÕ¹…Ñ”½É‘¥¹…°Í±½Ðˆ(€€€€€€€€¤(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Á•ÉÍ½¹…±¥Ñäµ½É‘¥¹…°µÍ¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}ÑÉÕ¹…Ñ¥½¹}…É¡¥Ù•Í}¥¹ÍÑ•…‘}½™}‘•±•Ñ¥¹œ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰Q¡”É•Ý¥¹ÝÉ¥Ñ”µÕÍÐ‰”É•½Ù•É…‰±”°¹½Ð„¡…É1Q¸((€€€€ŒÜÀÔÄØ€¼€ŒàÀÜØÌ€¼€ŒàÈÜÔØ…±°•¹‘•…ÐÑ¡¥Ì½¹”…±°°…¹…±°Ñ¡É•”Ý•É”(€€€Õ¹É•½Ù•É…‰±”™½ÈÑ¡”Í…µ”É•…Í½¸èÉ•Á±…•}µ•ÍÍ…•Í€1QÌÑ¡”É½ÝÌ…¹(€€€Ñ¡”QL•¹ÑÉä½•ÌÝ¥Ñ Ñ¡•´¸Õ…É‘¥¹œÑ¡”€©…¥´¨½˜„É•Ý¥¹ÍÑ¥±°±•…Ù•Ì(€€€•Ù•Éä½Ñ¡•ÈÝ…ä½˜…¥µ¥¹œ¥ÐÝÉ½¹œÑ•Éµ¥¹…°°Í¼Ñ¡”ÝÉ¥Ñ”¥ÑÍ•±˜¡…ÌÑ¼(€€€ÍÑ½À‘•ÍÑÉ½å¥¹œ¸Q¡”ÍÑ½É…”µ±…å•È½¹ÑÉ…ÐÑ¡¥ÌÉ•±¥•Ì½¸¥ÌÁ¥¹¹•¥¸(€€€Ñ•ÍÑÌ½¡•Éµ•Í}ÍÑ…Ñ”½Ñ•ÍÑ}É•Á±…•}µ•ÍÍ…•Í}…É¡¥Ù•}Í¥‰±¥¹Ì¹Áä¸(€€€€ˆˆˆ((€€€…ÁÑÕÉ•€ôíô((€€€±…ÍÌ}•¹Ðè(€€€€€€€‘•˜ÉÕ¹}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°ÁÉ½µÁÐ°½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éäõ9½¹”°ÍÑÉ•…µ}…±±‰…¬õ9½¹”°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€€€€€‰™¥¹…±}É•ÍÁ½¹Í”ˆè€‰É•Á±äˆ°(€€€€€€€€€€€€€€€€‰µ•ÍÍ…•Ìˆèl(€€€€€€€€€€€€€€€€€€€€¨¡½¹Ù•ÉÍ…Ñ¥½¹}¡¥ÍÑ½Éä½Èmt¤°(€€€€€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹ÐˆèÁÉ½µÁÑô°(€€€€€€€€€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä‰ô°(€€€€€€€€€€€€€€€t°(€€€€€€€€€€€ô((€€€±…ÍÌ}%µµ•‘¥…Ñ•Q¡É•…è(€€€€€€€‘•˜}}¥¹¥Ñ}|¡Í•±˜°Ñ…É•Ðõ9½¹”°‘…•µ½¸õ9½¹”°€¨©}Ñ¡É•…‘}½ÁÑ¥½¹Ì¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð€ôÑ…É•Ð((€€€€€€€‘•˜ÍÑ…ÉÐ¡Í•±˜¤è(€€€€€€€€€€€Í•±˜¹}Ñ…É•Ð ¤((€€€±…ÍÌ}MÑÕ‰ˆè(€€€€€€€‘•˜•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°€©}…ÉÌ°€¨©}­Ý…ÉÌ¤è(€€€€€€€€€€€É•ÑÕÉ¸mt((€€€€€€€‘•˜É•Á±…•}µ•ÍÍ…•Ì (€€€€€€€€€€€Í•±˜°(€€€€€€€€€€€Í•ÍÍ¥½¹}¥°(€€€€€€€€€€€µ•ÍÍ…•Ì°(€€€€€€€€€€€…Ñ¥Ù•}½¹±äõ…±Í”°(€€€€€€€€€€€…É¡¥Ù•}‘É½ÁÁ•õ…±Í”°(€€€€€€€€€€€É•©•Ñ}…Ñ¥Ù•}ÑÕÉ¹}±•…Í”õ…±Í”°(€€€€€€€€¤è(€€€€€€€€€€€…ÁÑÕÉ•‘l‰…Ñ¥Ù•}½¹±ä‰t€ô…Ñ¥Ù•}½¹±ä(€€€€€€€€€€€…ÁÑÕÉ•‘l‰…É¡¥Ù•}‘É½ÁÁ•‰t€ô…É¡¥Ù•}‘É½ÁÁ•(€€€€€€€€€€€…ÁÑÕÉ•‘l‰É•©•Ñ}…Ñ¥Ù•}ÑÕÉ¹}±•…Í”‰t€ôÉ•©•Ñ}…Ñ¥Ù•}ÑÕÉ¹}±•…Í”((€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰…É¡¥Ù”µÑÉÕ¹ŒµÍ¥‰t€ô}Í•ÍÍ¥½¸ (€€€€€€€…•¹Ðõ}•¹Ð ¤°(€€€€€€€¡¥ÍÑ½Éäõl(€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐ‰ô°(€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐÉ•Á±ä‰ô°(€€€€€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹‰ô°(€€€€€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹É•Á±ä‰ô°(€€€€€€€t°(€€€€¤((€€€ÑÉäè(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È¹Ñ¡É•…‘¥¹œ°€‰Q¡É•…ˆ°}%µµ•‘¥…Ñ•Q¡É•…¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}Í•ÍÍ¥½¹}¥¹™¼ˆ°±…µ‰‘„€©„°€¨©¬èíô¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•µ¥Ðˆ°±…µ‰‘„€©„è9½¹”¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}ÕÍ…”ˆ°±…µ‰‘„}„èíô¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰É•¹‘•É}µ•ÍÍ…”ˆ°±…µ‰‘„}Ð°}Œè€ˆˆ¤(€€€€€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è}MÑÕ‰ˆ ¤¤((€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰…É¡¥Ù”µÑÉÕ¹ŒµÍ¥ˆ°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰Í•½¹°É•Ý½É‘•ˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}ÕÍ•É}½É‘¥¹…°ˆè€Ä°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤((€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰É•ÍÕ±Ðˆ¤°˜‰½Ð•ÉÉ½ÈèíÉ•ÍÀ¹•Ð •ÉÉ½Èœ¥ôˆ(€€€€€€€…ÍÍ•ÉÐ…ÁÑÕÉ•¹•Ð ‰…É¡¥Ù•}‘É½ÁÁ•ˆ¤¥ÌQÉÕ”°€ (€€€€€€€€€€€€‰„É•Ý¥¹µÕÍÐÍ½™Ðµ…É¡¥Ù”Ñ¡”ÑÕÉ¹Ì¥Ð‘É½ÁÌ°¹½Ð1QÑ¡•´ˆ(€€€€€€€€¤(€€€€€€€€Œ€ŒàÀÈÄØèÍÑ¥±°µÕÍÐ¹½ÐÑ½Õ É½ÝÌ…É¡¥Ù•‰ä…¸•…É±¥•È½µÁ…Ñ¥½¸¸(€€€€€€€…ÍÍ•ÉÐ…ÁÑÕÉ•¹•Ð ‰…Ñ¥Ù•}½¹±äˆ¤¥ÌQÉÕ”(€€€€€€€…ÍÍ•ÉÐ…ÁÑÕÉ•¹•Ð ‰É•©•Ñ}…Ñ¥Ù•}ÑÕÉ¹}±•…Í”ˆ¤¥ÌQÉÕ”(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰…É¡¥Ù”µÑÉÕ¹ŒµÍ¥ˆ°9½¹”¤(((()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}Õ¹µ…Ñ¡•‘}É½Ý}¥‘}É•™ÕÍ•Í}•Ù•¹}Ý¥Ñ¡}½É‘¥¹…°¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆˆŒàÈäÔäèU¹­¹½Ý¸É½Ý}¥µÕÍÐÉ•™ÕÍ”€ ÐÀÄà¤°¹½Ð™…±°‰…¬Ñ¼„±¥•¹Ð½É‘¥¹…°¸ˆˆˆ(€€€É•Á±…•€ômt((€€€±…ÍÌ}…­•è(€€€€€€€‘•˜É•Á±…•}µ•ÍÍ…•Ì (€€€€€€€€€€€Í•±˜°(€€€€€€€€€€€­•ä°(€€€€€€€€€€€µ•ÍÍ…•Ì°(€€€€€€€€€€€…Ñ¥Ù•}½¹±äõ…±Í”°(€€€€€€€€€€€…É¡¥Ù•}‘É½ÁÁ•õ…±Í”°(€€€€€€€€€€€É•©•Ñ}…Ñ¥Ù•}ÑÕÉ¹}±•…Í”õ…±Í”°(€€€€€€€€¤è(€€€€€€€€€€€É•Á±…•¹…ÁÁ•¹ ¡­•ä°±¥ÍÐ¡µ•ÍÍ…•Ì¤¤¤((€€€¡¥ÍÑ½Éä€ôl(€€€€€€€ì‰}É½Ý}¥ˆè€ÄÀÄ°€‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐ‰ô°(€€€€€€€ì‰}É½Ý}¥ˆè€ÄÀÈ°€‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä€Ä‰ô°(€€€€€€€ì‰}É½Ý}¥ˆè€ÄÀÌ°€‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹‰ô°(€€€€€€€ì‰}É½Ý}¥ˆè€ÄÀÐ°€‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä€È‰ô°(€€€t(€€€Í•ÍÌ€ô}Í•ÍÍ¥½¸¡¡¥ÍÑ½Éäõ±¥ÍÐ¡¡¥ÍÑ½Éä¤¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰™…±±‰…¬µÉ½Üµ¥µÍ¥‰t€ôÍ•ÍÌ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è}…­• ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„€©„°€¨©¬èÁåÑ•ÍÐ¹™…¥° ‰µÕÍÐ¹½ÐÍÑ…ÉÐ„ÑÕÉ¸ˆ¤(€€€€¤((€€€ÑÉäè(€€€€€€€€ŒMÑ…±”É½Ý}¥€äää¹½Ð¥¸¡¥ÍÑ½ÉäƒŠP•Ù•¸Ý¥Ñ „Ù…±¥½É‘¥¹…°°É•™ÕÍ”¸(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰™…±±‰…¬µÉ½Üµ¥µÍ¥ˆ°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰¹•ÜÑÕÉ¸ˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}É½Ý}¥ˆè€äää°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}ÕÍ•É}½É‘¥¹…°ˆè€Ä°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰•ÉÉ½Èˆ¤¥Ì¹½Ð9½¹”(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÄà(€€€€€€€…ÍÍ•ÉÐÉ•Á±…•€ôômt(€€€€€€€…ÍÍ•ÉÐ±•¸¡Í•ÍÍl‰¡¥ÍÑ½Éä‰t¤€ôô€Ð(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰™…±±‰…¬µÉ½Üµ¥µÍ¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}Õ¹µ…Ñ¡•‘}µ•ÍÍ…•}¥‘}É•™ÕÍ•Í}•Ù•¹}Ý¥Ñ¡}½É‘¥¹…°¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆˆŒàÈäÔäèU¹­¹½Ý¸µ•ÍÍ…•}¥µÕÍÐÉ•™ÕÍ”ì¹¼Í¥±•¹Ð½É‘¥¹…°‘•É…‘…Ñ¥½¸¸ˆˆˆ(€€€É•Á±…•€ômt((€€€±…ÍÌ}…­•è(€€€€€€€‘•˜É•Á±…•}µ•ÍÍ…•Ì (€€€€€€€€€€€Í•±˜°(€€€€€€€€€€€­•ä°(€€€€€€€€€€€µ•ÍÍ…•Ì°(€€€€€€€€€€€…Ñ¥Ù•}½¹±äõ…±Í”°(€€€€€€€€€€€…É¡¥Ù•}‘É½ÁÁ•õ…±Í”°(€€€€€€€€€€€É•©•Ñ}…Ñ¥Ù•}ÑÕÉ¹}±•…Í”õ…±Í”°(€€€€€€€€¤è(€€€€€€€€€€€É•Á±…•¹…ÁÁ•¹ ¡­•ä°±¥ÍÐ¡µ•ÍÍ…•Ì¤¤¤((€€€€ŒAÉ½‘ÕÑ¥½¸µÍ¡…Á•¡¥ÍÑ½Éäè¹¼É•¹‘•É•È€‰¥ˆ­•åÌ½¸ÕÍ•È‘¥ÑÌ¸(€€€¡¥ÍÑ½Éä€ôl(€€€€€€€ì‰}É½Ý}¥ˆè€ÈÀÄ°€‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐ‰ô°(€€€€€€€ì‰}É½Ý}¥ˆè€ÈÀÈ°€‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä€Ä‰ô°(€€€€€€€ì‰}É½Ý}¥ˆè€ÈÀÌ°€‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹‰ô°(€€€€€€€ì‰}É½Ý}¥ˆè€ÈÀÐ°€‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä€È‰ô°(€€€t(€€€Í•ÍÌ€ô}Í•ÍÍ¥½¸¡¡¥ÍÑ½Éäõ±¥ÍÐ¡¡¥ÍÑ½Éä¤¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰Íå¹Ñ¡•Ñ¥ŒµµÍœµ¥µÍ¥‰t€ôÍ•ÍÌ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è}…­• ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„€©„°€¨©¬èÁåÑ•ÍÐ¹™…¥° ‰µÕÍÐ¹½ÐÍÑ…ÉÐ„ÑÕÉ¸ˆ¤(€€€€¤((€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰Íå¹Ñ¡•Ñ¥ŒµµÍœµ¥µÍ¥ˆ°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰™É•Í ÍÑ…ÉÐˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}µ•ÍÍ…•}¥ˆè€‰ÕÍ•È´ÄÜÈÌÐÔØÜàä´Àˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}ÕÍ•É}½É‘¥¹…°ˆè€À°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}•µÁÑå}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰•ÉÉ½Èˆ¤¥Ì¹½Ð9½¹”(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÄà(€€€€€€€…ÍÍ•ÉÐÉ•Á±…•€ôômt(€€€€€€€…ÍÍ•ÉÐ±•¸¡Í•ÍÍl‰¡¥ÍÑ½Éä‰t¤€ôô€Ð(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰Íå¹Ñ¡•Ñ¥ŒµµÍœµ¥µÍ¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}É½Ý}¥‘}É•Í½±Ù•Í}Ù¥…}‘‰}Ý¡•¹}µ•µ½Éå}±…­Í}ÍÑ…µÁÌ¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆˆŒàÈäÔäè™Ñ•ÈÑÕÉ¸É•ÝÉ¥Ñ”ÍÑÉ¥ÁÌ}É½Ý}¥°É•Í½±Ù”……¥¹ÍÐ‘ÕÉ…‰±”¡¥ÍÑ½Éä¸ˆˆˆ(€€€É•Á±…•€ômt(€€€€Œ1¥Ù”µ•µ½Éä…™Ñ•ÈÑÕÉ¸½µÁ±•Ñ¥½¸èÁÉ½Ù¥‘•Èµ™½Éµ…Ð°¹¼}É½Ý}¥ÍÑ…µÁÌ¸(€€€±¥Ù•}¡¥ÍÑ½Éä€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐ‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä€Ä‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä€È‰ô°(€€€t(€€€‘ÕÉ…‰±•}¡¥ÍÑ½Éä€ôl(€€€€€€€ì‰}É½Ý}¥ˆè€ÔÀÄ°€‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐ‰ô°(€€€€€€€ì‰}É½Ý}¥ˆè€ÔÀÈ°€‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä€Ä‰ô°(€€€€€€€ì‰}É½Ý}¥ˆè€ÔÀÌ°€‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹‰ô°(€€€€€€€ì‰}É½Ý}¥ˆè€ÔÀÐ°€‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä€È‰ô°(€€€t((€€€±…ÍÌ}…­•è(€€€€€€€‘•˜É•Á±…•}µ•ÍÍ…•Ì (€€€€€€€€€€€Í•±˜°(€€€€€€€€€€€­•ä°(€€€€€€€€€€€µ•ÍÍ…•Ì°(€€€€€€€€€€€…Ñ¥Ù•}½¹±äõ…±Í”°(€€€€€€€€€€€…É¡¥Ù•}‘É½ÁÁ•õ…±Í”°(€€€€€€€€€€€É•©•Ñ}…Ñ¥Ù•}ÑÕÉ¹}±•…Í”õ…±Í”°(€€€€€€€€¤è(€€€€€€€€€€€É•Á±…•¹…ÁÁ•¹ ¡­•ä°±¥ÍÐ¡µ•ÍÍ…•Ì¤¤¤((€€€€€€€‘•˜•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°­•ä°É•Á…¥É}…±Ñ•É¹…Ñ¥½¸õ…±Í”°¥¹±Õ‘•}É½Ý}¥‘Ìõ…±Í”¤è(€€€€€€€€€€€…ÍÍ•ÉÐ¥¹±Õ‘•}É½Ý}¥‘Ì¥ÌQÉÕ”(€€€€€€€€€€€É•ÑÕÉ¸±¥ÍÐ¡‘ÕÉ…‰±•}¡¥ÍÑ½Éä¤((€€€Í•ÍÌ€ô}Í•ÍÍ¥½¸¡¡¥ÍÑ½Éäõ±¥ÍÐ¡±¥Ù•}¡¥ÍÑ½Éä¤°Í•ÍÍ¥½¹}­•äô‰‘ˆµÉ½Üµ­•äˆ¤(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰‘ˆµÉ½ÜµÉ•Í½±Ù”µÍ¥‰t€ôÍ•ÍÌ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è}…­• ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤((€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆè€‰‘ˆµÉ½ÜµÉ•Í½±Ù”µÍ¥ˆ°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰¹•ÜÑÕÉ¸ˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}É½Ý}¥ˆè€ÔÀÌ°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰•ÉÉ½Èˆ¤¥Ì9½¹”°É•ÍÀ(€€€€€€€…ÍÍ•ÉÐ±•¸¡Í•ÍÍl‰¡¥ÍÑ½Éä‰t¤€ôô€È(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍl‰¡¥ÍÑ½Éä‰ul´Åul‰½¹Ñ•¹Ð‰t€ôô€‰É•Á±ä€Äˆ(€€€€€€€…ÍÍ•ÉÐ±•¸¡É•Á±…•¤€ôô€Ä(€€€€€€€€Œ!•…±¥¹œè±¥Ù”±¥ÍÐÍ¡½Õ±¹½Ü…ÉÉäÍÑ…µÁÌ™½ÈÍÕ‰Í•ÅÕ•¹ÐÉ•Ý¥¹‘Ì¸(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍl‰¡¥ÍÑ½Éä‰ulÁt¹•Ð ‰}É½Ý}¥ˆ¤€ôô€ÔÀÄ(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À ‰‘ˆµÉ½ÜµÉ•Í½±Ù”µÍ¥ˆ°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}É½Ý}¥‘}É•…±}Í•ÍÍ¥½¹‘‰}É•Í½±Ù•}Ý¥Ñ¡½ÕÑ}µ•µ½Éå}ÍÑ…µÁÌ (€€€µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ (¤è(€€€€ˆˆˆŒàÈäÔäÁÉ½‘ÕÑ¥½¸Á…Ñ èÉ•…°M•ÍÍ¥½¹¥¹Í•ÉÐƒŠH±¥Ù”¡¥ÍÑ½ÉäÝ¥Ñ¡½ÕÐ(€€€}É½Ý}¥€¡ÑÕÉ¸É•ÝÉ¥Ñ”¤ƒŠHÑÉÕ¹…Ñ•}‰•™½É•}É½Ý}¥ÕÑÌ‘ÕÉ…‰±”€¬µ•µ½Éä¸((€€€9¼¡…¹µÍ••‘•¥‘Ì…¹¹¼5…¥5½¬ÍÑ…Ñ”µ…¹…•ÈƒŠPÑ¡”½¹ÑÉ…ÐÑ¡…Ð(€€€€ŒàÈÜØØÉ•Ù¥•ÜÍ…¥Õ¹¥Ð™¥áÑÕÉ•ÌµÕÍÐ•á•É¥Í”¸(€€€€ˆˆˆ(€€€™É½´¡•Éµ•Í}ÍÑ…Ñ”¥µÁ½ÉÐM•ÍÍ¥½¹((€€€‘ˆ€ôM•ÍÍ¥½¹¡‘‰}Á…Ñ õÑµÁ}Á…Ñ €¼€‰É½Ý¥µÑÉÕ¹Œ¹‘ˆˆ¤(€€€Í•ÍÍ¥½¹}­•ä€ô€‰É•…°µ‘ˆµÉ½ÜµÑÉÕ¹Œˆ(€€€‘ˆ¹É•…Ñ•}Í•ÍÍ¥½¸¡Í•ÍÍ¥½¹}­•ä°€‰±¤ˆ¤(€€€µÍÌ€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐ‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä€Ä‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä€È‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Ñ¡¥É‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä€Ì‰ô°(€€€t(€€€Ý¥Ñ ‘ˆ¹}±½¬è(€€€€€€€‘ˆ¹}¥¹Í•ÉÑ}µ•ÍÍ…•}É½ÝÌ¡‘ˆ¹}½¹¸°Í•ÍÍ¥½¹}­•ä°µÍÌ¤(€€€€€€€‘ˆ¹}½¹¸¹½µµ¥Ð ¤(€€€É½Ý}¥‘Ì€ômµl‰}É½Ý}¥‰t™½È´¥¸µÍÍt(€€€…ÍÍ•ÉÐ…±°¡¥Í¥¹ÍÑ…¹”¡È°¥¹Ð¤…¹È€ø€À™½ÈÈ¥¸É½Ý}¥‘Ì¤((€€€€Œ™Ñ•ÈÑÕÉ¸½µÁ±•Ñ¥½¸…Ñ•Ý…äÉ•ÝÉ¥Ñ•Ì¡¥ÍÑ½Éä…ÌÁÉ½Ù¥‘•Èµ™½Éµ…Ð‘¥ÑÌ(€€€€ŒÝ¥Ñ¡½ÕÐ}É½Ý}¥ƒŠPÁÉ½‘ÕÑ¥½¸µÍ¡…Á•±¥Ù”µ•µ½Éä¸(€€€±¥Ù•}¡¥ÍÑ½Éä€ômì‰É½±”ˆèµl‰É½±”‰t°€‰½¹Ñ•¹Ðˆèµl‰½¹Ñ•¹Ð‰uô™½È´¥¸µÍÍt(€€€…ÍÍ•ÉÐ…±° ‰}É½Ý}¥ˆ¹½Ð¥¸´™½È´¥¸±¥Ù•}¡¥ÍÑ½Éä¤((€€€Í•ÍÌ€ô}Í•ÍÍ¥½¸¡¡¥ÍÑ½Éäõ±¥ÍÐ¡±¥Ù•}¡¥ÍÑ½Éä¤°Í•ÍÍ¥½¹}­•äõÍ•ÍÍ¥½¹}­•ä¤(€€€Í¥€ô€‰É•…°µ‘ˆµÉ½ÜµÑÉÕ¹ŒµÍ¥ˆ(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÍ¥‘t€ôÍ•ÍÌ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è‘ˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}¥¹™±¥¡Ñ}ÑÕÉ¸ˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤((€€€ÑÉäè(€€€€€€€€ŒÕÐ‰•™½É”Í•½¹ÕÍ•ÈÑÕÉ¸€¡É½Ý}¥‘ÍlÉt¤ƒŠP±•…Ù”™¥ÉÍÐ•á¡…¹”½¹±ä¸(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆèÍ¥°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰É•Ý½Õ¹Í•½¹ˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}É½Ý}¥ˆèÉ½Ý}¥‘ÍlÉt°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}ÕÍ•É}½É‘¥¹…°ˆè€Ä°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰•ÉÉ½Èˆ¤¥Ì9½¹”°É•ÍÀ(€€€€€€€…ÍÍ•ÉÐ±•¸¡Í•ÍÍl‰¡¥ÍÑ½Éä‰t¤€ôô€È(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍl‰¡¥ÍÑ½Éä‰ulÁul‰½¹Ñ•¹Ð‰t€ôô€‰™¥ÉÍÐˆ(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍl‰¡¥ÍÑ½Éä‰ulÅul‰½¹Ñ•¹Ð‰t€ôô€‰É•Á±ä€Äˆ(€€€€€€€€ŒÕÉ…‰±”…Ñ¥Ù”ÑÉ…¹ÍÉ¥ÁÐµ…Ñ¡•ÌÑ¡”ÕÐÁ±ÕÌÑ¡”ÁÉ½µÁÐ©ÕÍÐÍ•¹Ð°Ý¡¥ ¥Ì‘ÕÉ…‰±”…Ð(€€€€€€€€ŒÍÕ‰µ¥Ð€ ŒÄÄÄàØà¤ƒŠP‰•™½É”Ñ¡”ÑÕÉ¸ÉÕ¹Ì€¡…É¡¥Ù•}‘É½ÁÁ•­••ÁÌ¥¹…Ñ¥Ù”É½ÝÌì(€€€€€€€€Œ•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸É•ÑÕÉ¹Ì…Ñ¥Ù”½¹±ä¤¸(€€€€€€€…Ñ¥Ù”€ô‘ˆ¹•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸¡Í•ÍÍ¥½¹}­•ä¤(€€€€€€€…ÍÍ•ÉÐmµl‰½¹Ñ•¹Ð‰t™½È´¥¸…Ñ¥Ù•t€ôôl‰™¥ÉÍÐˆ°€‰É•Á±ä€Äˆ°€‰É•Ý½Õ¹Í•½¹‰t(€€€€€€€€Œ!•…°ÍÑ…µÁÌ™½ÈÍÕ‰Í•ÅÕ•¹ÐÉ•Ý¥¹‘ÌÝ¡•¸µ•µ½Éä±¥¹•ÕÀÝ¥Ñ ¸(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍl‰¡¥ÍÑ½Éä‰ulÁt¹•Ð ‰}É½Ý}¥ˆ¤¥Ì¹½Ð9½¹”(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡Í¥°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}É½Ý}¥‘}É•…±}Í•ÍÍ¥½¹‘‰}Õ¹­¹½Ý¹}É•™ÕÍ•Í}‘•ÍÁ¥Ñ•}½É‘¥¹…° (€€€µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ (¤è(€€€€ˆˆˆŒàÈäÔä™…¥°µ±½Í•èÕ¹­¹½Ý¸É½Ý}¥€¬Ù…±¥½É‘¥¹…°µÕÍÐ¹½ÐÑÉÕ¹…Ñ”(€€€É•…°M•ÍÍ¥½¹€¡Ñ¡”µ…ÍÌµ‘•±•Ñ”±…ÍÌÝ¡•¸‘ÕÉ…‰±”¥…¹¹½ÐÉ•Í½±Ù”¤¸(€€€€ˆˆˆ(€€€™É½´¡•Éµ•Í}ÍÑ…Ñ”¥µÁ½ÉÐM•ÍÍ¥½¹((€€€‘ˆ€ôM•ÍÍ¥½¹¡‘‰}Á…Ñ õÑµÁ}Á…Ñ €¼€‰É½Ý¥µÉ•™ÕÍ”¹‘ˆˆ¤(€€€Í•ÍÍ¥½¹}­•ä€ô€‰É•…°µ‘ˆµÉ½ÜµÉ•™ÕÍ”ˆ(€€€‘ˆ¹É•…Ñ•}Í•ÍÍ¥½¸¡Í•ÍÍ¥½¹}­•ä°€‰±¤ˆ¤(€€€µÍÌ€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐ‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä€Ä‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä€È‰ô°(€€€t(€€€Ý¥Ñ ‘ˆ¹}±½¬è(€€€€€€€‘ˆ¹}¥¹Í•ÉÑ}µ•ÍÍ…•}É½ÝÌ¡‘ˆ¹}½¹¸°Í•ÍÍ¥½¹}­•ä°µÍÌ¤(€€€€€€€‘ˆ¹}½¹¸¹½µµ¥Ð ¤((€€€±¥Ù•}¡¥ÍÑ½Éä€ômì‰É½±”ˆèµl‰É½±”‰t°€‰½¹Ñ•¹Ðˆèµl‰½¹Ñ•¹Ð‰uô™½È´¥¸µÍÍt(€€€Í•ÍÌ€ô}Í•ÍÍ¥½¸¡¡¥ÍÑ½Éäõ±¥ÍÐ¡±¥Ù•}¡¥ÍÑ½Éä¤°Í•ÍÍ¥½¹}­•äõÍ•ÍÍ¥½¹}­•ä¤(€€€Í¥€ô€‰É•…°µ‘ˆµÉ½ÜµÉ•™ÕÍ”µÍ¥ˆ(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÍ¥‘t€ôÍ•ÍÌ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è‘ˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„€©„°€¨©¬èÁåÑ•ÍÐ¹™…¥° ‰µÕÍÐ¹½ÐÍÑ…ÉÐ„ÑÕÉ¸ˆ¤(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}¥¹™±¥¡Ñ}ÑÕÉ¸ˆ°±…µ‰‘„€©„°€¨©¬èÁåÑ•ÍÐ¹™…¥° ‰µÕÍÐ¹½ÐÍÑ…ÉÐ„ÑÕÉ¸ˆ¤(€€€€¤((€€€¹}‰•™½É”€ô±•¸¡‘ˆ¹•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸¡Í•ÍÍ¥½¹}­•ä¤¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆèÍ¥°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰ÍÑ…±”‘ÕÉ…‰±”¥ˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}É½Ý}¥ˆè€ääå|ääå|äää°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}ÕÍ•É}½É‘¥¹…°ˆè€À°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}•µÁÑå}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰•ÉÉ½Èˆ¤¥Ì¹½Ð9½¹”(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÄà(€€€€€€€…ÍÍ•ÉÐ±•¸¡Í•ÍÍl‰¡¥ÍÑ½Éä‰t¤€ôô€Ð(€€€€€€€…ÍÍ•ÉÐ±•¸¡‘ˆ¹•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸¡Í•ÍÍ¥½¹}­•ä¤¤€ôô¹}‰•™½É”(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡Í¥°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}É½Ý}¥‘}µ¥Í…±¥¹•‘}µ•µ½Éå}É•™ÕÍ•Í}½¹Ñ•¹Ñ}ÍÝ…À (€€€µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ (¤è(€€€€ˆˆˆŒàÈäÔä¡•…°µÁ…Ñ Õ…Éè•ÅÕ…°µ±•¹Ñ ‰ÕÐ½¹Ñ•¹Ðµµ¥Í…±¥¹•±¥Ù”µ•µ½Éä(€€€µÕÍÐÉ•™ÕÍ”€ ÐÀÄà¤°¹½Ðé¥ÀµÍÑ…µÀ‘ÕÉ…‰±”¥‘ÌÁ½Í¥Ñ¥½¹…±±ä…¹ÕÐÑ¡”(€€€ÝÉ½¹œÑÕÉ¸¸AÉ½‰”€Ñ„™É½´Ñ¡”AH€ŒàÌÈÀÈÉ•Ù¥•Ü¸(€€€€ˆˆˆ(€€€™É½´¡•Éµ•Í}ÍÑ…Ñ”¥µÁ½ÉÐM•ÍÍ¥½¹((€€€‘ˆ€ôM•ÍÍ¥½¹¡‘‰}Á…Ñ õÑµÁ}Á…Ñ €¼€‰É½Ý¥µµ¥Í…±¥¸µ½¹Ñ•¹Ð¹‘ˆˆ¤(€€€Í•ÍÍ¥½¹}­•ä€ô€‰É•…°µ‘ˆµÉ½Üµµ¥Í…±¥¸µ½¹Ñ•¹Ðˆ(€€€‘ˆ¹É•…Ñ•}Í•ÍÍ¥½¸¡Í•ÍÍ¥½¹}­•ä°€‰±¤ˆ¤(€€€µÍÌ€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É„‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰Éˆ‰ô°(€€€t(€€€Ý¥Ñ ‘ˆ¹}±½¬è(€€€€€€€‘ˆ¹}¥¹Í•ÉÑ}µ•ÍÍ…•}É½ÝÌ¡‘ˆ¹}½¹¸°Í•ÍÍ¥½¹}­•ä°µÍÌ¤(€€€€€€€‘ˆ¹}½¹¸¹½µµ¥Ð ¤(€€€É¥‘}ˆ€ôµÍÍlÉul‰}É½Ý}¥‰t(€€€½É¥¥¹…±}É½Ý}¥‘Ì€ômµ•ÍÍ…•l‰}É½Ý}¥‰t™½Èµ•ÍÍ…”¥¸µÍÍt((€€€€ŒM…µ”±•¹Ñ €¬Í…µ”É½±”Á…ÑÑ•É¸°‰ÕÐ½¹Ñ•¹ÐÁ½Í¥Ñ¥½¹ÌÍÝ…ÁÁ•è„(€€€€ŒÁ½Í¥Ñ¥½¹…°ÍÑ…µÀÝ½Õ±µ…É¬±¥Ù”€‰ˆÝ¥Ñ ‘ÕÉ…‰±”ÌÉ½Ü¥…¹Ñ¡”(€€€€ŒÕÐÝ½Õ±­••ÀÑ¡”Ù•ÉäÑÕÉ¸Ñ¡”ÕÍ•ÈÉ•Ý½Õ¹Á…ÍÐ¸(€€€±¥Ù•}¡¥ÍÑ½Éä€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰Éˆ‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É„‰ô°(€€€t(€€€Í•ÍÌ€ô}Í•ÍÍ¥½¸¡¡¥ÍÑ½Éäõ±¥ÍÐ¡±¥Ù•}¡¥ÍÑ½Éä¤°Í•ÍÍ¥½¹}­•äõÍ•ÍÍ¥½¹}­•ä¤(€€€Í¥€ô€‰µ¥Í…±¥¸µ½¹Ñ•¹ÐµÍ¥ˆ(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÍ¥‘t€ôÍ•ÍÌ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è‘ˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„€©„°€¨©¬èÁåÑ•ÍÐ¹™…¥° ‰µÕÍÐ¹½ÐÍÑ…ÉÐ„ÑÕÉ¸ˆ¤(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}¥¹™±¥¡Ñ}ÑÕÉ¸ˆ°±…µ‰‘„€©„°€¨©¬èÁåÑ•ÍÐ¹™…¥° ‰µÕÍÐ¹½ÐÍÑ…ÉÐ„ÑÕÉ¸ˆ¤(€€€€¤((€€€¹}‰•™½É”€ô±•¸¡‘ˆ¹•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸¡Í•ÍÍ¥½¹}­•ä¤¤(€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆèÍ¥°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰É•Ý¥¹ˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}É½Ý}¥ˆèÉ¥‘}ˆ°(€€€€€€€€€€€€€€€€€€€€‰É•‰¥¹‘}ÍÕÉÙ¥Ù½É}É½Ý}¥‘Ìˆèl©½É¥¥¹…±}É½Ý}¥‘Ì°€ääå|ääåt°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰•ÉÉ½Èˆ¤¥Ì¹½Ð9½¹”(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÄà(€€€€€€€€Œ…¥°µ±½Í•è¹½Ñ¡¥¹œÍÑ…µÁ•½¹Ñ¼Ñ¡”µ¥Í…±¥¹•‘¥ÑÌ°¹½Ñ¡¥¹œÕÐ¸(€€€€€€€…ÍÍ•ÉÐ…±° ‰}É½Ý}¥ˆ¹½Ð¥¸´™½È´¥¸Í•ÍÍl‰¡¥ÍÑ½Éä‰t¤(€€€€€€€…ÍÍ•ÉÐ±•¸¡Í•ÍÍl‰¡¥ÍÑ½Éä‰t¤€ôô€Ð(€€€€€€€…ÍÍ•ÉÐ±•¸¡‘ˆ¹•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸¡Í•ÍÍ¥½¹}­•ä¤¤€ôô¹}‰•™½É”(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡Í¥°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}É½Ý}¥‘}µ¥Í…±¥¹•‘}µ•µ½Éå}É½±•}Í¡¥™Ñ}Ñ…É•ÑÍ}É•…±}ÑÕÉ¸ (€€€µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ (¤è(€€€€ˆˆˆŒàÈäÔä¡•…°µÁ…Ñ Õ…Éè•ÅÕ…°µ±•¹Ñ ‰ÕÐÉ½±”µµ¥Í…±¥¹•±¥Ù”µ•µ½Éä(€€€µÕÍÐ¹½Ð‰”Á½Í¥Ñ¥½¹…±±äÍÑ…µÁ•¸Q¡”½¹Ñ•¹ÐµÙ•É¥™¥•™…±±‰…¬ÍÑ¥±°(€€€É•Í½±Ù•ÌÑ¡”I0Ñ…É•ÐÑÕÉ¸¥¸±¥Ù”½É‘•ÈƒŠPÑ¡”ÕÐ‘É½ÁÌ•á…Ñ±äÑ¡”(€€€…‘‘É•ÍÍ•ÕÍ•ÈÑÕÉ¸°¹•Ù•È„Á½Í¥Ñ¥½¹…±±äµ¥Ìµ…¥µ•½¹”¸AÉ½‰”€Ñˆ™É½´(€€€Ñ¡”AH€ŒàÌÈÀÈÉ•Ù¥•Ü¸(€€€€ˆˆˆ(€€€™É½´¡•Éµ•Í}ÍÑ…Ñ”¥µÁ½ÉÐM•ÍÍ¥½¹((€€€‘ˆ€ôM•ÍÍ¥½¹¡‘‰}Á…Ñ õÑµÁ}Á…Ñ €¼€‰É½Ý¥µµ¥Í…±¥¸µÉ½±”¹‘ˆˆ¤(€€€Í•ÍÍ¥½¹}­•ä€ô€‰É•…°µ‘ˆµÉ½Üµµ¥Í…±¥¸µÉ½±”ˆ(€€€‘ˆ¹É•…Ñ•}Í•ÍÍ¥½¸¡Í•ÍÍ¥½¹}­•ä°€‰±¤ˆ¤(€€€µÍÌ€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É„‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰Éˆ‰ô°(€€€t(€€€Ý¥Ñ ‘ˆ¹}±½¬è(€€€€€€€‘ˆ¹}¥¹Í•ÉÑ}µ•ÍÍ…•}É½ÝÌ¡‘ˆ¹}½¹¸°Í•ÍÍ¥½¹}­•ä°µÍÌ¤(€€€€€€€‘ˆ¹}½¹¸¹½µµ¥Ð ¤(€€€É¥‘}ˆ€ôµÍÍlÉul‰}É½Ý}¥‰t(€€€½É¥¥¹…±}É½Ý}¥‘Ì€ômµ•ÍÍ…•l‰}É½Ý}¥‰t™½Èµ•ÍÍ…”¥¸µÍÍt((€€€±¥Ù•}¡¥ÍÑ½Éä€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É„‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰Éˆ‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰‰ô°(€€€t(€€€Í•ÍÌ€ô}Í•ÍÍ¥½¸¡¡¥ÍÑ½Éäõ±¥ÍÐ¡±¥Ù•}¡¥ÍÑ½Éä¤°Í•ÍÍ¥½¹}­•äõÍ•ÍÍ¥½¹}­•ä¤(€€€Í¥€ô€‰µ¥Í…±¥¸µÉ½±”µÍ¥ˆ(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÍ¥‘t€ôÍ•ÍÌ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è‘ˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}¥¹™±¥¡Ñ}ÑÕÉ¸ˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤((€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆèÍ¥°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰É•Ý¥¹ˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}É½Ý}¥ˆèÉ¥‘}ˆ°(€€€€€€€€€€€€€€€€€€€€‰É•‰¥¹‘}ÍÕÉÙ¥Ù½É}É½Ý}¥‘Ìˆèl©½É¥¥¹…±}É½Ý}¥‘Ì°€ääå|ääåt°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰•ÉÉ½Èˆ¤¥Ì9½¹”°É•ÍÀ(€€€€€€€€ŒQ¡”…‘‘É•ÍÍ•ÑÕÉ¸€ ‰ˆ¤¥Ì‘É½ÁÁ••á…Ñ±äì•…É±¥•È±¥Ù”ÑÕÉ¹Ì(€€€€€€€€ŒÍÕÉÙ¥Ù”Õ¹Ñ½Õ¡•¸	•™½É”Ñ¡”Õ…É°„Á½Í¥Ñ¥½¹…°é¥ÀµÍÑ…µÀÁÕÐ„(€€€€€€€€ŒÕÍ•ÈµÉ½Ü¥½¸…¸…ÍÍ¥ÍÑ…¹Ð‘¥Ð…¹Ñ¡”ÁÉ”µÕ…É™…±±‰…¬ÕÐ…Ð(€€€€€€€€Œ„µ¥Ìµ…¥µ•¥¹‘•à¸€¡É•Í }É½Ý}¥ÍÑ…µÁÌ½¸ÍÕÉÙ¥Ù½ÉÌ…É”•áÁ•Ñ•ƒŠP(€€€€€€€€ŒÉ•Á±…•}µ•ÍÍ…•ÌÉ”µ¥¹Í•ÉÑÌ…¹É”µÍÑ…µÁÌÑ¡”ÍÕÉÙ¥Ù¥¹œ‘¥ÑÌ¸¤(€€€€€€€ÍÕÉÙ¥Ù½ÉÌ€ôl¡µl‰É½±”‰t°µl‰½¹Ñ•¹Ð‰t¤™½È´¥¸Í•ÍÍl‰¡¥ÍÑ½Éä‰ut(€€€€€€€…ÍÍ•ÉÐÍÕÉÙ¥Ù½ÉÌ€ôôl(€€€€€€€€€€€€ ‰ÕÍ•Èˆ°€‰ˆ¤°(€€€€€€€€€€€€ ‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰É„ˆ¤°(€€€€€€€€€€€€ ‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰Éˆˆ¤°(€€€€€€€t(€€€€€€€€ŒQ¡”±¥Ù”±¥ÍÐÝ…ÌÑ½¼µ¥Í…±¥¹•Ñ¼‰¥¹½±ÍÕÉÙ¥Ù½ÉÌÑ¼Ñ¡•¥È¹•Ü(€€€€€€€€ŒÁ¡åÍ¥…°É½ÝÌÍ…™•±ä¸±°É•ÅÕ•ÍÑ•%Ì­¹½Ý¸Ñ¼Ñ¡”ÁÉ”µÝÉ¥Ñ”…Ñ¥Ù”(€€€€€€€€ŒÑÉ…¹ÍÉ¥ÁÐ…É”Ñ¡•É•™½É”±•…É•ì…¸Õ¹É•±…Ñ•…É¡¥Ù•½…¹•ÍÑ½È%(€€€€€€€€ŒÉ•µ…¥¹Ì…‰Í•¹Ð™É½´Ñ¡”‰½Õ¹‘•µ…À…¹­••ÁÌ¥ÑÌ¥‘•¹Ñ¥Ñä¸(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰É•ÍÕ±Ð‰ul‰ÍÕÉÙ¥Ù½É}É½Ý}¥‘}µ…À‰t€ôôì(€€€€€€€€€€€ÍÑÈ¡É½Ý}¥¤è9½¹”™½ÈÉ½Ý}¥¥¸½É¥¥¹…±}É½Ý}¥‘Ì(€€€€€€€ô(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡Í¥°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}É½Ý}¥‘}‘‰}™…±±‰…­}½É‘¥¹…±}µ…ÁÁ¥¹}Ù•É¥™¥•Í}½¹Ñ•¹Ð (€€€µ½¹­•åÁ…Ñ °(¤è(€€€€ˆˆˆŒàÈäÔä‘ˆµ™…±±‰…¬Õ…ÉèÝ¡•¸±¥Ù”½‘ÕÉ…‰±”±•¹Ñ¡Ì‘¥™™•È°µ…ÁÁ¥¹œÑ¡”(€€€‘ÕÉ…‰±”ÕÍ•Èµ½É‘¥¹…°½¹Ñ¼±¥Ù”¥¹‘¥•ÌµÕÍÐÙ•É¥™äÑ¡”µ…ÁÁ•ÑÕÉ¸Í¡½ÝÌ(€€€Ñ¡”‘ÕÉ…‰±”Ñ…É•ÐÌ½¹Ñ•¹ÐƒŠP„É•Á…¥É•ÕÍ•ÈíÕÍ•Èµ•É”Í¡¥™ÑÌ½É‘¥¹…±Ì(€€€…¹Ý½Õ±½Ñ¡•ÉÝ¥Í”ÕÐ…¸•áÑÉ„ÑÕÉ¸Í¥±•¹Ñ±ä¸(€€€€ˆˆˆ(€€€É•Á±…•€ômt((€€€€ŒÕÉ…‰±”ÑÉ…¹ÍÉ¥ÁÐ€¡É•Á…¥É•¤èÑ¡”µ•É”½±±…ÁÍ•ÑÝ¼ÕÍ•ÈÑÕÉ¹Ì°Í¼(€€€€Œ‘ÕÉ…‰±”ÕÍ•Èµ½É‘¥¹…°€Ä€ ‰Í•½¹ˆ¤µ…ÁÌ½¹Ñ¼„%I9P±¥Ù”ÕÍ•ÈÑÕÉ¸¸(€€€‘ÕÉ…‰±•}¡¥ÍÑ½Éä€ôl(€€€€€€€ì‰}É½Ý}¥ˆè€ØÀÄ°€‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÑq¹™½±±½ÜµÕÀ‰ô°(€€€€€€€ì‰}É½Ý}¥ˆè€ØÀÌ°€‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä€Ä‰ô°(€€€€€€€ì‰}É½Ý}¥ˆè€ØÀÐ°€‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹‰ô°(€€€€€€€ì‰}É½Ý}¥ˆè€ØÀÔ°€‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä€È‰ô°(€€€t(€€€€Œ1¥Ù”µ•µ½Éä€¡Õ¹É•Á…¥É•°±½¹•È¤èÕÍ•È½É‘¥¹…°€Ä¡•É”¥Ì€‰™½±±½ÜµÕÀˆ°(€€€€Œ9=P€‰Í•½¹ˆ¸(€€€±¥Ù•}¡¥ÍÑ½Éä€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐ‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰™½±±½ÜµÕÀ‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä€Ä‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä€È‰ô°(€€€t((€€€±…ÍÌ}…­•è(€€€€€€€‘•˜É•Á±…•}µ•ÍÍ…•Ì (€€€€€€€€€€€Í•±˜°(€€€€€€€€€€€­•ä°(€€€€€€€€€€€µ•ÍÍ…•Ì°(€€€€€€€€€€€…Ñ¥Ù•}½¹±äõ…±Í”°(€€€€€€€€€€€…É¡¥Ù•}‘É½ÁÁ•õ…±Í”°(€€€€€€€€€€€É•©•Ñ}…Ñ¥Ù•}ÑÕÉ¹}±•…Í”õ…±Í”°(€€€€€€€€¤è(€€€€€€€€€€€É•Á±…•¹…ÁÁ•¹ ¡­•ä°±¥ÍÐ¡µ•ÍÍ…•Ì¤¤¤((€€€€€€€‘•˜•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°­•ä°É•Á…¥É}…±Ñ•É¹…Ñ¥½¸õ…±Í”°¥¹±Õ‘•}É½Ý}¥‘Ìõ…±Í”¤è(€€€€€€€€€€€É•ÑÕÉ¸m‘¥Ð¡´¤™½È´¥¸‘ÕÉ…‰±•}¡¥ÍÑ½Éåt((€€€Í•ÍÌ€ô}Í•ÍÍ¥½¸¡¡¥ÍÑ½Éäõ±¥ÍÐ¡±¥Ù•}¡¥ÍÑ½Éä¤°Í•ÍÍ¥½¹}­•äô‰‘ˆµ™…±±‰…¬µÙ•É¥™äµ­•äˆ¤(€€€Í¥€ô€‰‘ˆµ™…±±‰…¬µÙ•É¥™äµÍ¥ˆ(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÍ¥‘t€ôÍ•ÍÌ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è}…­• ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„€©„°€¨©¬èÁåÑ•ÍÐ¹™…¥° ‰µÕÍÐ¹½ÐÍÑ…ÉÐ„ÑÕÉ¸ˆ¤(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}¥¹™±¥¡Ñ}ÑÕÉ¸ˆ°±…µ‰‘„€©„°€¨©¬èÁåÑ•ÍÐ¹™…¥° ‰µÕÍÐ¹½ÐÍÑ…ÉÐ„ÑÕÉ¸ˆ¤(€€€€¤((€€€ÑÉäè(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆèÍ¥°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰É•Ý¥¹Ñ¼Í•½¹ˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}É½Ý}¥ˆè€ØÀÐ°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€€Œ5…ÁÁ•±¥Ù”ÑÕÉ¸€ ‰™½±±½ÜµÕÀˆ¤‘½•Ì¹½Ðµ…Ñ ‘ÕÉ…‰±”Ñ…É•Ð(€€€€€€€€Œ€ ‰Í•½¹ˆ¤ƒŠPÉ•™ÕÍ”¥¹ÍÑ•…½˜ÕÑÑ¥¹œÑ¡”ÝÉ½¹œÑÕÉ¸¸(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀ¹•Ð ‰•ÉÉ½Èˆ¤¥Ì¹½Ð9½¹”(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁl‰•ÉÉ½È‰ul‰½‘”‰t€ôô€ÐÀÄà(€€€€€€€…ÍÍ•ÉÐÉ•Á±…•€ôômt(€€€€€€€…ÍÍ•ÉÐ±•¸¡Í•ÍÍl‰¡¥ÍÑ½Éä‰t¤€ôô€Ô(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡Í¥°9½¹”¤(()ÁåÑ•ÍÐ¹µ…É¬¹Á…É…µ•ÑÉ¥é” ‰ÑÕÉ¹}¥Í½±…Ñ¥½¸ˆ°m…±Í”°QÉÕ•t¤)‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}½¹Í•ÕÑ¥Ù•}É•Ý¥¹‘Í}Ý¥Ñ¡}É•ÑÕÉ¹•‘}ÍÕÉÙ¥Ù½É}É½Ý}¥‘Ì (€€€µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ °ÑÕÉ¹}¥Í½±…Ñ¥½¸(¤è(€€€€ˆˆˆŒàÌÈÀÈÉ•Ù¥•Ü€¡½¹Í•ÕÑ¥Ù”µÉ•Ý¥¹ÍÑ…±•¹•ÍÌ¤èÉ•Á±…•}µ•ÍÍ…•ÌÉ”µ¥¹Í•ÉÑÌ(€€€Ñ¡”ÍÕÉÙ¥Ù¥¹œÁÉ•™¥à…Ì9\É½ÝÌ°Í¼Ñ¡”ÁÉ”µÉ•Ý¥¹±¥•¹ÐÉ½Ü¥‘Ì‘¥”½¸(€€€Ñ¡”™¥ÉÍÐÉ•Ý¥¹¸Q¡”ÍÕ‰µ¥ÐÉ•ÍÁ½¹Í”µÕÍÐÉ•ÑÕÉ¸Ñ¡”™É•Í ÍÕÉÙ¥Ù½È¥‘Ì°(€€€…¹„Í•½¹É•Ý¥¹ÕÍ¥¹œÑ¡•´µÕÍÐÍÕ••Ý¡•É”Ñ¡”ÍÑ…±”¥™…¥°µ±½Í•Ì¸(€€€€ˆˆˆ(€€€™É½´¡•Éµ•Í}ÍÑ…Ñ”¥µÁ½ÉÐM•ÍÍ¥½¹((€€€‘ˆ€ôM•ÍÍ¥½¹¡‘‰}Á…Ñ õÑµÁ}Á…Ñ €¼€‰É½Ý¥µ½¹Í•Œ¹‘ˆˆ¤(€€€Í•ÍÍ¥½¹}­•ä€ô€‰É•…°µ‘ˆµ½¹Í•ŒµÉ•Ý¥¹ˆ(€€€‘ˆ¹É•…Ñ•}Í•ÍÍ¥½¸¡Í•ÍÍ¥½¹}­•ä°€‰±¤ˆ¤(€€€µÍÌ€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐ‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä€Ä‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä€È‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Ñ¡¥É‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰É•Á±ä€Ì‰ô°(€€€t(€€€Ý¥Ñ ‘ˆ¹}±½¬è(€€€€€€€‘ˆ¹}¥¹Í•ÉÑ}µ•ÍÍ…•}É½ÝÌ¡‘ˆ¹}½¹¸°Í•ÍÍ¥½¹}­•ä°µÍÌ¤(€€€€€€€‘ˆ¹}½¹¸¹½µµ¥Ð ¤(€€€½É¥¥¹…±}É½Ý}¥‘Ì€ômµl‰}É½Ý}¥‰t™½È´¥¸µÍÍt((€€€Í•ÍÌ€ô}Í•ÍÍ¥½¸¡¡¥ÍÑ½Éäõm‘¥Ð¡´¤™½È´¥¸µÍÍt°Í•ÍÍ¥½¹}­•äõÍ•ÍÍ¥½¹}­•ä¤(€€€Í¥€ô€‰É•…°µ‘ˆµ½¹Í•ŒµÉ•Ý¥¹µÍ¥ˆ(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÍ¥‘t€ôÍ•ÍÌ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è‘ˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°(€€€€€€€€‰}±½…‘}™œˆ°(€€€€€€€±…µ‰‘„èì‰‘…Í¡‰½…Éˆèì‰ÑÕÉ¹}¥Í½±…Ñ¥½¸ˆèÑÕÉ¹}¥Í½±…Ñ¥½¹õô°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°(€€€€€€€€‰}ÍÕ‰µ¥Ñ}ÁÉ½µÁÑ}Ñ½}½µÁÕÑ•}¡½ÍÐˆ°(€€€€€€€±…µ‰‘„€©}…ÉÌ°€¨©}­Ý…ÉÌèÍ•ÉÙ•È¹}½¬ ‰¡½ÍÐˆ°ì‰ÍÑ…ÑÕÌˆè€‰ÍÑÉ•…µ¥¹œ‰ô¤°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}¥¹™±¥¡Ñ}ÑÕÉ¸ˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤((€€€ÑÉäè(€€€€€€€€ŒI•Ý¥¹€ÄèÕÐ‰•™½É”€‰Ñ¡¥Éˆ€¡±…ÍÐÕÍ•ÈÑÕÉ¸¤¸MÕÉÙ¥Ù½ÉÌèÑÕÉ¹Ì(€€€€€€€€Œ€‰™¥ÉÍÐˆ€¬€‰Í•½¹ˆ€ ¬…ÍÍ¥ÍÑ…¹ÐÉ•Á±¥•Ì¤­••ÀÑ¡•¥ÈÉ½ÝÌ€ ŒàÈäÔØ¤¸(€€€€€€€É•ÍÀÄ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆèÍ¥°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰É•Ý½Õ¹Ñ¡¥Éˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}É½Ý}¥ˆè½É¥¥¹…±}É½Ý}¥‘ÍlÑt°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}ÕÍ•É}½É‘¥¹…°ˆè€È°(€€€€€€€€€€€€€€€€€€€€‰É•‰¥¹‘}ÍÕÉÙ¥Ù½É}É½Ý}¥‘Ìˆèl©½É¥¥¹…±}É½Ý}¥‘Ì°€ääå|ääåt°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀÄ¹•Ð ‰•ÉÉ½Èˆ¤¥Ì9½¹”°É•ÍÀÄ(€€€€€€€…ÍÍ•ÉÐ€‰ÍÕÉÙ¥Ù½É}ÕÍ•É}É½Ý}¥‘Ìˆ¹½Ð¥¸É•ÍÀÅl‰É•ÍÕ±Ð‰t(€€€€€€€É½Ý}¥‘}µ…À€ôÉ•ÍÀÅl‰É•ÍÕ±Ð‰t¹•Ð ‰ÍÕÉÙ¥Ù½É}É½Ý}¥‘}µ…Àˆ¤(€€€€€€€…ÍÍ•ÉÐ¥Í¥¹ÍÑ…¹”¡É½Ý}¥‘}µ…À°‘¥Ð¤(€€€€€€€ÍÕÉÙ¥Ù½ÉÌ€ôl(€€€€€€€€€€€É½Ý}¥‘}µ…ÁmÍÑÈ¡½É¥¥¹…±}É½Ý}¥‘ÍlÁt¥t°(€€€€€€€€€€€É½Ý}¥‘}µ…ÁmÍÑÈ¡½É¥¥¹…±}É½Ý}¥‘ÍlÉt¥t°(€€€€€€€t(€€€€€€€€Œ=¹±äÑ¡”‘É½ÁÁ•ÍÕ™™¥à¥Ì…É¡¥Ù•èÑ¡”­•ÁÐÉ½ÝÌ­••ÀÑ¡•¥È¥‘Ì…¹(€€€€€€€€ŒÑ¡”µ…ÀÍ…åÌÍ¼ìÑ¡”ÑÝ¼ÕÐÑÕÉ¹Ìµ…ÀÑ¼9½¹”¸(€€€€€€€…ÍÍ•ÉÐÍÕÉÙ¥Ù½ÉÌ€ôôm½É¥¥¹…±}É½Ý}¥‘ÍlÁt°½É¥¥¹…±}É½Ý}¥‘ÍlÉut(€€€€€€€…ÍÍ•ÉÐÉ½Ý}¥‘}µ…À€ôôì(€€€€€€€€€€€€¨©íÍÑÈ¡É¥¤èÉ¥™½ÈÉ¥¥¸½É¥¥¹…±}É½Ý}¥‘ÍlèÑuô°(€€€€€€€€€€€ÍÑÈ¡½É¥¥¹…±}É½Ý}¥‘ÍlÑt¤è9½¹”°(€€€€€€€€€€€ÍÑÈ¡½É¥¥¹…±}É½Ý}¥‘ÍlÕt¤è9½¹”°(€€€€€€€ô(€€€€€€€…ÍÍ•ÉÐ€ˆääääääˆ¹½Ð¥¸É½Ý}¥‘}µ…À(€€€€€€€Í•ÍÍl‰ÉÕ¹¹¥¹œ‰t€ô…±Í”((€€€€€€€€ŒI•Ý¥¹€ÈèÑ¡”¥Ñ¡”±¥•¹Ð…¡•	=IÉ•Ý¥¹€Ä¥ÌÍÑ¥±°Ñ¡”±¥Ù”É½Ü(€€€€€€€€Œ€¡¹¼€ÐÀÄàÉ•™ÕÍ…°°¹¼É•‰¥¹‘…¹”¤ƒŠPÑ¡”ÕÍ•Èµ™…¥¹œÁ½¥¹Ð½˜€ŒàÈäÔØ¸(€€€€€€€É•ÍÀÈ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÈˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆèÍ¥°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰É•Ý½Õ¹Í•½¹€¡Í…µ”¥¤ˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}É½Ý}¥ˆè½É¥¥¹…±}É½Ý}¥‘ÍlÉt°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}ÕÍ•É}½É‘¥¹…°ˆè€Ä°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÀÈ¹•Ð ‰•ÉÉ½Èˆ¤¥Ì9½¹”°É•ÍÀÈ(€€€€€€€…ÍÍ•ÉÐ±•¸¡Í•ÍÍl‰¡¥ÍÑ½Éä‰t¤€ôô€È(€€€€€€€…ÍÍ•ÉÐÍ•ÍÍl‰¡¥ÍÑ½Éä‰ulÁul‰½¹Ñ•¹Ð‰t€ôô€‰™¥ÉÍÐˆ(€€€€€€€…Ñ¥Ù”€ô‘ˆ¹•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸¡Í•ÍÍ¥½¹}­•ä¤(€€€€€€€€ŒQ¡”ÕÐ°Á±ÕÌÑ¡”ÁÉ½µÁÐ©ÕÍÐÍ•¹Ð€¡‘ÕÉ…‰±”…ÐÍÕ‰µ¥Ð°€ŒÄÄÄàØà¤¸(€€€€€€€…ÍÍ•ÉÐmµl‰½¹Ñ•¹Ð‰t™½È´¥¸…Ñ¥Ù•t€ôôl‰™¥ÉÍÐˆ°€‰É•Á±ä€Äˆ°€‰É•Ý½Õ¹Í•½¹€¡Í…µ”¥¤‰t(€€€€€€€€Œ¹Ñ¡”Í•½¹É•ÍÁ½¹Í”É•‰¥¹‘Ì……¥¸è½¹”ÍÕÉÙ¥Ù¥¹œÕÍ•ÈÑÕÉ¸¸(€€€€€€€ÍÕÉÙ¥Ù½ÉÌÈ€ôÉ•ÍÀÉl‰É•ÍÕ±Ð‰t¹•Ð ‰ÍÕÉÙ¥Ù½É}ÕÍ•É}É½Ý}¥‘Ìˆ¤(€€€€€€€…ÍÍ•ÉÐ¥Í¥¹ÍÑ…¹”¡ÍÕÉÙ¥Ù½ÉÌÈ°±¥ÍÐ¤…¹±•¸¡ÍÕÉÙ¥Ù½ÉÌÈ¤€ôô€Ä(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡Í¥°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}É•‰¥¹‘}µ…Á}±•…ÉÍ}…Ñ¥Ù•}É½Ý}¡¥‘‘•¹}‰å}Í•ÅÕ•¹•}É•Á…¥È (€€€µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ (¤è(€€€€ˆˆ‰Q¡”‰½Õ¹‘•µ…À±…ÍÍ¥™¥•ÌÁ¡åÍ¥…°…Ñ¥Ù”%Ì‰•™½É”ÕÍ•ÈíÕÍ•ÈÉ•Á…¥È¸ˆˆˆ(€€€™É½´¡•Éµ•Í}ÍÑ…Ñ”¥µÁ½ÉÐM•ÍÍ¥½¹((€€€‘ˆ€ôM•ÍÍ¥½¹¡‘‰}Á…Ñ õÑµÁ}Á…Ñ €¼€‰É½Ý¥µÉ•Á…¥É•µÝ•‘”¹‘ˆˆ¤(€€€Í•ÍÍ¥½¹}­•ä€ô€‰É•…°µ‘ˆµÉ½Ý¥µÉ•Á…¥É•µÝ•‘”ˆ(€€€‘ˆ¹É•…Ñ•}Í•ÍÍ¥½¸¡Í•ÍÍ¥½¹}­•ä°€‰±¤ˆ¤(€€€Á¡åÍ¥…°€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐ™É…µ•¹Ð‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹™É…µ•¹Ð‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰½µ‰¥¹•É•Á±ä‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Ñ…É•Ð‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰Ñ…É•ÐÉ•Á±ä‰ô°(€€€t(€€€Ý¥Ñ ‘ˆ¹}±½¬è(€€€€€€€‘ˆ¹}¥¹Í•ÉÑ}µ•ÍÍ…•}É½ÝÌ¡‘ˆ¹}½¹¸°Í•ÍÍ¥½¹}­•ä°Á¡åÍ¥…°¤(€€€€€€€‘ˆ¹}½¹¸¹½µµ¥Ð ¤(€€€Á¡åÍ¥…±}¥‘Ì€ômµ•ÍÍ…•l‰}É½Ý}¥‰t™½Èµ•ÍÍ…”¥¸Á¡åÍ¥…±t(€€€É•Á…¥É•€ô‘ˆ¹•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸ (€€€€€€€Í•ÍÍ¥½¹}­•ä°É•Á…¥É}…±Ñ•É¹…Ñ¥½¸õQÉÕ”°¥¹±Õ‘•}É½Ý}¥‘ÌõQÉÕ”(€€€€¤(€€€€ŒAÉ½Ù¥‘•ÈÉ•Á…¥Èµ•É•ÌÑ¡”Ý•‘”…¹¹••ÍÍ…É¥±ä‘É½ÁÌÑ¡”Í•½¹(€€€€ŒÁ¡åÍ¥…°ÕÍ•ÈÌÉ½Ü¥‘•¹Ñ¥Ñä™É½´Ñ¡”É•Á±…äÙ¥•Ü¸(€€€…ÍÍ•ÉÐÁ¡åÍ¥…±}¥‘ÍlÅt¹½Ð¥¸ì(€€€€€€€Í•ÉÙ•È¹}µ•ÍÍ…•}É½Ý}¥¡µ•ÍÍ…”¤™½Èµ•ÍÍ…”¥¸É•Á…¥É•(€€€ô((€€€Í•ÍÌ€ô}Í•ÍÍ¥½¸ (€€€€€€€¡¥ÍÑ½Éäõm‘¥Ð¡µ•ÍÍ…”¤™½Èµ•ÍÍ…”¥¸É•Á…¥É•‘t°Í•ÍÍ¥½¹}­•äõÍ•ÍÍ¥½¹}­•ä(€€€€¤(€€€Í¥€ô€‰É½Ý¥µÉ•Á…¥É•µÝ•‘”µÍ¥ˆ(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÍ¥‘t€ôÍ•ÍÌ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è‘ˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}¥¹™±¥¡Ñ}ÑÕÉ¸ˆ°±…µ‰‘„€©„°€¨©¬è9½¹”¤((€€€ÑÉäè(€€€€€€€É•ÍÁ½¹Í”€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆèÍ¥°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰É•ÑÉäÑ…É•Ðˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}É½Ý}¥ˆèÁ¡åÍ¥…±}¥‘ÍlÍt°(€€€€€€€€€€€€€€€€€€€€‰É•‰¥¹‘}ÍÕÉÙ¥Ù½É}É½Ý}¥‘Ìˆèl©Á¡åÍ¥…±}¥‘Ì°€ääå|ääåt°(€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ}ÑÉÕ¹…Ñ”ˆèQÉÕ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÁ½¹Í”¹•Ð ‰•ÉÉ½Èˆ¤¥Ì9½¹”°É•ÍÁ½¹Í”(€€€€€€€É½Ý}¥‘}µ…À€ôÉ•ÍÁ½¹Í•l‰É•ÍÕ±Ð‰ul‰ÍÕÉÙ¥Ù½É}É½Ý}¥‘}µ…À‰t(€€€€€€€…ÍÍ•ÉÐÉ½Ý}¥‘}µ…ÁmÍÑÈ¡Á¡åÍ¥…±}¥‘ÍlÅt¥t¥Ì9½¹”(€€€€€€€…ÍÍ•ÉÐ€ˆääääääˆ¹½Ð¥¸É½Ý}¥‘}µ…À(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡Í¥°9½¹”¤(()‘•˜Ñ•ÍÑ}ÁÉ½µÁÑ}ÍÕ‰µ¥Ñ}Õ¹½¹™¥Éµ•‘}ÑÉÕ¹…Ñ¥½¹}É•™ÕÍ•Í}‰•™½É•}Ñ…É•Ñ}É•Í½±ÕÑ¥½¸ (€€€µ½¹­•åÁ…Ñ °(¤è(€€€€ˆˆ‰½¹Í•¹Ð€ ÐÀÈä¤¥Ì¡•­•	=IÑ…É•ÐÉ•Í½±ÕÑ¥½¸è…¸Õ¹½¹™¥Éµ•(€€€ÍÕ‰µ¥Ð…ÉÉå¥¹œÑÉÕ¹…Ñ¥½¸Á…É…µÌµÕÍÐ¹½ÐÁ…äÑ¡”‘ÕÉ…‰±”µÑÉ…¹ÍÉ¥ÁÐ(€€€É•…½È¡•…°µÍÑ…µÀ±¥Ù”¡¥ÍÑ½Éä€¡Í¥µÁ±¥™äÉ•Ù¥•Ü½¸€ŒàÌÜàÔ¤°…¹…¸(€€€½ÕÐµ½˜µÉ…¹”Õ¹½¹™¥Éµ•½É‘¥¹…°É•™ÕÍ•Ì€ÐÀÈä°¹½Ð€ÐÀÄàƒŠPÑ¡”‰…Í•±¥¹”(€€€ÁÉ••‘•¹”‰•™½É”Ñ¡”É½Üµ¥™•…ÑÕÉ”¸(€€€€ˆˆˆ(€€€‘‰}É•…‘Ì€ômt((€€€±…ÍÌ}MÁåè(€€€€€€€‘•˜•Ñ}µ•ÍÍ…•Í}…Í}½¹Ù•ÉÍ…Ñ¥½¸¡Í•±˜°€©„°€¨©¬¤è(€€€€€€€€€€€‘‰}É•…‘Ì¹…ÁÁ•¹ Ä¤(€€€€€€€€€€€É•ÑÕÉ¸mt((€€€€€€€‘•˜É•Á±…•}µ•ÍÍ…•Ì¡Í•±˜°€©„°€¨©¬¤è(€€€€€€€€€€€ÁåÑ•ÍÐ¹™…¥° ‰µÕÍÐ¹½ÐÝÉ¥Ñ”ˆ¤((€€€¡¥ÍÐ€ôl(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰™¥ÉÍÐ‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰ÈÄ‰ô°(€€€€€€€ì‰É½±”ˆè€‰ÕÍ•Èˆ°€‰½¹Ñ•¹Ðˆè€‰Í•½¹‰ô°(€€€€€€€ì‰É½±”ˆè€‰…ÍÍ¥ÍÑ…¹Ðˆ°€‰½¹Ñ•¹Ðˆè€‰ÈÈ‰ô°(€€€t(€€€Í•ÍÌ€ô}Í•ÍÍ¥½¸¡¡¥ÍÑ½Éäõ±¥ÍÐ¡¡¥ÍÐ¤°Í•ÍÍ¥½¹}­•äô‰½¹Í•¹ÐµÁÉ••‘•¹”µ­•äˆ¤(€€€Í¥€ô€‰½¹Í•¹ÐµÁÉ••‘•¹”µÍ¥ˆ(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹ÍmÍ¥‘t€ôÍ•ÍÌ(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}•Ñ}‘ˆˆ°±…µ‰‘„è}MÁå ¤¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}…•¹Ñ}‰Õ¥±ˆ°±…µ‰‘„€©„°€¨©¬èÁåÑ•ÍÐ¹™…¥° ‰µÕÍÐ¹½ÐÍÑ…ÉÐ„ÑÕÉ¸ˆ¤(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È°€‰}ÍÑ…ÉÑ}¥¹™±¥¡Ñ}ÑÕÉ¸ˆ°±…µ‰‘„€©„°€¨©¬èÁåÑ•ÍÐ¹™…¥° ‰µÕÍÐ¹½ÐÍÑ…ÉÐ„ÑÕÉ¸ˆ¤(€€€€¤((€€€ÑÉäè(€€€€€€€€ŒU¹½¹™¥Éµ•É½Ý}¥è€ÐÀÈä°¹¼É•…°¹¼¡•…°ÍÑ…µÁÌ½¸¡¥ÍÑ½Éä¸(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÄˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆèÍ¥°€‰Ñ•áÐˆè€‰àˆ°€‰ÑÉÕ¹…Ñ•}‰•™½É•}É½Ý}¥ˆè€Íô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐ€¡É•ÍÀ¹•Ð ‰•ÉÉ½Èˆ¤½Èíô¤¹•Ð ‰½‘”ˆ¤€ôô€ÐÀÈä°É•ÍÀ(€€€€€€€…ÍÍ•ÉÐ‘‰}É•…‘Ì€ôômt(€€€€€€€…ÍÍ•ÉÐ…±° ‰}É½Ý}¥ˆ¹½Ð¥¸´™½È´¥¸Í•ÍÍl‰¡¥ÍÑ½Éä‰t¤((€€€€€€€€Œ5…±™½Éµ•Á…É…´ÍÑ¥±°‰•…ÑÌ½¹Í•¹Ðè‰½½°É½Ý}¥¥Ì€ÐÀÀÐ¸(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÈˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì‰Í•ÍÍ¥½¹}¥ˆèÍ¥°€‰Ñ•áÐˆè€‰àˆ°€‰ÑÉÕ¹…Ñ•}‰•™½É•}É½Ý}¥ˆèQÉÕ•ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐ€¡É•ÍÀ¹•Ð ‰•ÉÉ½Èˆ¤½Èíô¤¹•Ð ‰½‘”ˆ¤€ôô€ÐÀÀÐ°É•ÍÀ((€€€€€€€€ŒU¹½¹™¥Éµ•½ÕÐµ½˜µÉ…¹”½É‘¥¹…°è€ÐÀÈä€¡½¹Í•¹Ð¤°¹½Ð€ÐÀÄà€¡É…¹”¤¸(€€€€€€€É•ÍÀ€ôÍ•ÉÙ•È¹¡…¹‘±•}É•ÅÕ•ÍÐ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¥ˆè€ˆÌˆ°(€€€€€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰ÁÉ½µÁÐ¹ÍÕ‰µ¥Ðˆ°(€€€€€€€€€€€€€€€€‰Á…É…µÌˆèì(€€€€€€€€€€€€€€€€€€€€‰Í•ÍÍ¥½¹}¥ˆèÍ¥°(€€€€€€€€€€€€€€€€€€€€‰Ñ•áÐˆè€‰àˆ°(€€€€€€€€€€€€€€€€€€€€‰ÑÉÕ¹…Ñ•}‰•™½É•}ÕÍ•É}½É‘¥¹…°ˆè€ää°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉÐ€¡É•ÍÀ¹•Ð ‰•ÉÉ½Èˆ¤½Èíô¤¹•Ð ‰½‘”ˆ¤€ôô€ÐÀÈä°É•ÍÀ(€€€€€€€…ÍÍ•ÉÐ±•¸¡Í•ÍÍl‰¡¥ÍÑ½Éä‰t¤€ôô€Ð(€€€™¥¹…±±äè(€€€€€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Ì¹Á½À¡Í¥°9½¹”¤(()‘•˜Ñ•ÍÑ}Á•ÉÍ¥ÍÑ}±¥Ù•}Í•ÍÍ¥½¹}ÍåÍÑ•µ}ÁÉ½µÁÑ}ÕÍ•Í}ÁÉ½™¥±•}¡½µ”¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰%ÍÍÕ”€ŒÔÀÈÌÌè}Á•ÉÍ¥ÍÑ}±¥Ù•}Í•ÍÍ¥½¹}ÍåÍÑ•µ}ÁÉ½µÁÐµÕÍÐÉ”µ‰¥¹(€€€!I5M}!=5Ñ¼Ñ¡”Í•ÍÍ¥½¸ÌÁÉ½™¥±”‰•™½É”É•‰Õ¥±‘¥¹œÑ¡”ÍåÍÑ•´(€€€ÁÉ½µÁÐ¸€]¥Ñ¡½ÕÐÑ¡¥Ì°„€½µ½‘•°ÍÝ¥Ñ É•‰Õ¥±‘ÌÑ¡”ÁÉ½µÁÐÝ¥Ñ Ñ¡”(€€€É½½ÐÁÉ½™¥±”ÌM=U0¹µ…¹Í­¥±±Ì¥¹ÍÑ•…½˜Ñ¡”Í•ÍÍ¥½¸ÌÁÉ½™¥±”¸(€€€€ˆˆˆ(€€€ÁÉ½™¥±•}¡½µ”€ôÑµÁ}Á…Ñ €¼€‰ÁÉ½™¥±”µÝ½É¬ˆ(€€€ÁÉ½™¥±•}¡½µ”¹µ­‘¥È ¤(€€€€¡ÁÉ½™¥±•}¡½µ”€¼€‰M=U0¹µˆ¤¹ÝÉ¥Ñ•}Ñ•áÐ (€€€€€€€€ˆŒ]½É¬Á•ÉÍ½¹…q¹e½Ô…É”„Ý½É¬…•¹Ð¸ˆ°•¹½‘¥¹œô‰ÕÑ˜´àˆ(€€€€¤((€€€‰Õ¥±Ñ}¡½µ•Ì€ômt((€€€±…ÍÌ…­••¹Ðè(€€€€€€€µ½‘•°€ô€‰Ñ•ÍÐµµ½‘•°ˆ(€€€€€€€ÁÉ½Ù¥‘•È€ô€‰Ñ•ÍÐˆ(€€€€€€€}…¡•‘}ÍåÍÑ•µ}ÁÉ½µÁÐ€ô9½¹”(€€€€€€€}Í•ÍÍ¥½¹}‘ˆ€ô9½¹”((€€€€€€€‘•˜}‰Õ¥±‘}ÍåÍÑ•µ}ÁÉ½µÁÐ¡Í•±˜°ÍåÍÑ•µ}µ•ÍÍ…”õ9½¹”¤è(€€€€€€€€€€€™É½´¡•Éµ•Í}½¹ÍÑ…¹ÑÌ¥µÁ½ÉÐ•Ñ}¡•Éµ•Í}¡½µ”(€€€€€€€€€€€¡½µ”€ô•Ñ}¡•Éµ•Í}¡½µ” ¤(€€€€€€€€€€€‰Õ¥±Ñ}¡½µ•Ì¹…ÁÁ•¹¡ÍÑÈ¡¡½µ”¤¤(€€€€€€€€€€€Í½Õ°€ô€ (€€€€€€€€€€€€€€€€¡¡½µ”€¼€‰M=U0¹µˆ¤¹É•…‘}Ñ•áÐ¡•¹½‘¥¹œô‰ÕÑ˜´àˆ¤(€€€€€€€€€€€€€€€¥˜€¡¡½µ”€¼€‰M=U0¹µˆ¤¹•á¥ÍÑÌ ¤(€€€€€€€€€€€€€€€•±Í”€ˆˆ(€€€€€€€€€€€€¤(€€€€€€€€€€€É•ÑÕÉ¸˜‰MåÍÑ•´ÁÉ½µÁÐ™É½´í¡½µ•õq¹íÍ½Õ±ôˆ((€€€±…ÍÌ…­•è(€€€€€€€‘•˜ÕÁ‘…Ñ•}ÍåÍÑ•µ}ÁÉ½µÁÐ¡Í•±˜°Í•ÍÍ¥½¹}¥°ÁÉ½µÁÐ¤è(€€€€€€€€€€€Á…ÍÌ((€€€…•¹Ð€ô…­••¹Ð ¤(€€€…•¹Ð¹}Í•ÍÍ¥½¹}‘ˆ€ô…­• ¤(€€€Í•ÍÍ¥½¸€ôì(€€€€€€€€‰…•¹Ðˆè…•¹Ð°(€€€€€€€€‰Í•ÍÍ¥½¹}­•äˆè€‰Ñ•ÍÐµ­•äˆ°(€€€€€€€€‰ÁÉ½™¥±•}¡½µ”ˆèÍÑÈ¡ÁÉ½™¥±•}¡½µ”¤°(€€€ô((€€€Í•ÉÙ•È¹}Á•ÉÍ¥ÍÑ}±¥Ù•}Í•ÍÍ¥½¹}ÍåÍÑ•µ}ÁÉ½µÁÐ¡Í•ÍÍ¥½¸¤((€€€€ŒQ¡”ÍåÍÑ•´ÁÉ½µÁÐµÕÍÐ¡…Ù”‰••¸‰Õ¥±ÐÝ¡¥±”Ñ¡”½Ù•ÉÉ¥‘”Á½¥¹Ñ•(€€€€ŒÑ¼Ñ¡”ÁÉ½™¥±”¡½µ”°¹½ÐÑ¡”É½½Ðø¼¹¡•Éµ•Ì¸(€€€…ÍÍ•ÉÐ±•¸¡‰Õ¥±Ñ}¡½µ•Ì¤€ôô€Ä°˜‰•áÁ•Ñ•€Ä‰Õ¥±°½Ðí‰Õ¥±Ñ}¡½µ•Íôˆ(€€€…ÍÍ•ÉÐÍÑÈ¡ÁÉ½™¥±•}¡½µ”¤¥¸‰Õ¥±Ñ}¡½µ•ÍlÁt°€ (€€€€€€€˜‰ÍåÍÑ•´ÁÉ½µÁÐ‰Õ¥±ÐÝ¥Ñ ÝÉ½¹œ¡½µ”èí‰Õ¥±Ñ}¡½µ•ÍlÁuôˆ(€€€€¤(€€€…ÍÍ•ÉÐ€‰]½É¬Á•ÉÍ½¹„ˆ¥¸…•¹Ð¹}…¡•‘}ÍåÍÑ•µ}ÁÉ½µÁÐ((€€€€ŒQ¡”½Ù•ÉÉ¥‘”µÕÍÐ¡…Ù”‰••¸É•Í•Ð…™Ñ•ÈÑ¡”…±°¸(€€€™É½´¡•Éµ•Í}½¹ÍÑ…¹ÑÌ¥µÁ½ÉÐ•Ñ}¡•Éµ•Í}¡½µ•}½Ù•ÉÉ¥‘”(€€€…ÍÍ•ÉÐ•Ñ}¡•Éµ•Í}¡½µ•}½Ù•ÉÉ¥‘” ¤¥Ì9½¹”(()‘•˜Ñ•ÍÑ}Á•ÉÍ¥ÍÑ}±¥Ù•}Í•ÍÍ¥½¹}ÍåÍÑ•µ}ÁÉ½µÁÑ}¹½}ÁÉ½™¥±•}¥Í}Õ¹¡…¹•¡µ½¹­•åÁ…Ñ ¤è(€€€€ˆˆ‰M•ÍÍ¥½¹ÌÝ¥Ñ¡½ÕÐ„ÁÉ½™¥±•}¡½µ”µÕÍÐ¹½ÐÍ•Ð½±•…È…¹ä½Ù•ÉÉ¥‘”ƒŠP(€€€Ñ¡”™Õ¹Ñ¥½¸Í¡½Õ±‰•¡…Ù”¥‘•¹Ñ¥…±±äÑ¼‰•™½É”Ñ¡”™¥à¸ˆˆˆ(€€€±…ÍÌ…­••¹Ðè(€€€€€€€µ½‘•°€ô€‰Ñ•ÍÐˆ(€€€€€€€}…¡•‘}ÍåÍÑ•µ}ÁÉ½µÁÐ€ô9½¹”(€€€€€€€}Í•ÍÍ¥½¹}‘ˆ€ô9½¹”((€€€€€€€‘•˜}‰Õ¥±‘}ÍåÍÑ•µ}ÁÉ½µÁÐ¡Í•±˜°ÍåÍÑ•µ}µ•ÍÍ…”õ9½¹”¤è(€€€€€€€€€€€É•ÑÕÉ¸€‰Á±…¥¸ÁÉ½µÁÐˆ((€€€±…ÍÌ…­•è(€€€€€€€‘•˜ÕÁ‘…Ñ•}ÍåÍÑ•µ}ÁÉ½µÁÐ¡Í•±˜°Í•ÍÍ¥½¹}¥°ÁÉ½µÁÐ¤è(€€€€€€€€€€€Á…ÍÌ((€€€…•¹Ð€ô…­••¹Ð ¤(€€€…•¹Ð¹}Í•ÍÍ¥½¹}‘ˆ€ô…­• ¤(€€€Í•ÍÍ¥½¸€ôì(€€€€€€€€‰…•¹Ðˆè…•¹Ð°(€€€€€€€€‰Í•ÍÍ¥½¹}­•äˆè€‰Ñ•ÍÐµ­•äˆ°(€€€€€€€€‰ÁÉ½™¥±•}¡½µ”ˆè9½¹”°(€€€ô((€€€€ŒM¡½Õ±¹½ÐÉ…¥Í”°Í¡½Õ±ÍÑ¥±°‰Õ¥±…¹…¡”¸(€€€Í•ÉÙ•È¹}Á•ÉÍ¥ÍÑ}±¥Ù•}Í•ÍÍ¥½¹}ÍåÍÑ•µ}ÁÉ½µÁÐ¡Í•ÍÍ¥½¸¤(€€€…ÍÍ•ÉÐ…•¹Ð¹}…¡•‘}ÍåÍÑ•µ}ÁÉ½µÁÐ€ôô€‰Á±…¥¸ÁÉ½µÁÐˆ(()‘•˜Ñ•ÍÑ}Á•ÉÍ¥ÍÑ}±¥Ù•}Í•ÍÍ¥½¹}ÍåÍÑ•µ}ÁÉ½µÁÑ}É•ÍÑ½É•Í}ÁÉ•}•á¥ÍÑ¥¹}½Ù•ÉÉ¥‘”¡ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰É•Í•Ñ}¡•Éµ•Í}¡½µ•}½Ù•ÉÉ¥‘” ¤É•ÍÑ½É•ÌÑ¡”ÁÉ•Ù¥½ÕÌ½¹Ñ•áÑY…ÈÍÑ…Ñ”°(€€€¹½Ð©ÕÍÐÑ¡”Õ¹Í•Ð…Í”èÝ¡•¸„…±±•È…±É•…‘ä¡½±‘Ì…¸½Ù•ÉÉ¥‘”°Ñ¡”(€€€Á•ÉÍ¥ÍÐ…±°µÕÍÐÍ½Á”Ñ¼Ñ¡”Í•ÍÍ¥½¸ÌÁÉ½™¥±”…¹Ñ¡•¸¡…¹Ñ¡”(€€€…±±•ÈÌ½Ù•ÉÉ¥‘”‰…¬°É…Ñ¡•ÈÑ¡…¸±•…É¥¹œ¥ÐÑ¼9½¹”¸ˆˆˆ(€€€™É½´¡•Éµ•Í}½¹ÍÑ…¹ÑÌ¥µÁ½ÉÐ•Ñ}¡•Éµ•Í}¡½µ•}½Ù•ÉÉ¥‘”((€€€½ÕÑ•É}¡½µ”€ôÑµÁ}Á…Ñ €¼€‰ÁÉ½™¥±”µ½ÕÑ•Èˆ(€€€½ÕÑ•É}¡½µ”¹µ­‘¥È ¤(€€€¥¹¹•É}¡½µ”€ôÑµÁ}Á…Ñ €¼€‰ÁÉ½™¥±”µ¥¹¹•Èˆ(€€€¥¹¹•É}¡½µ”¹µ­‘¥È ¤(€€€€¡¥¹¹•É}¡½µ”€¼€‰M=U0¹µˆ¤¹ÝÉ¥Ñ•}Ñ•áÐ (€€€€€€€€ˆŒ%¹¹•ÈÁ•ÉÍ½¹…q¹e½Ô…É”Ñ¡”¥¹¹•È…•¹Ð¸ˆ°•¹½‘¥¹œô‰ÕÑ˜´àˆ(€€€€¤((€€€‰Õ¥±Ñ}¡½µ•Ì€ômt((€€€±…ÍÌ…­••¹Ðè(€€€€€€€µ½‘•°€ô€‰Ñ•ÍÐµµ½‘•°ˆ(€€€€€€€ÁÉ½Ù¥‘•È€ô€‰Ñ•ÍÐˆ(€€€€€€€}…¡•‘}ÍåÍÑ•µ}ÁÉ½µÁÐ€ô9½¹”(€€€€€€€}Í•ÍÍ¥½¹}‘ˆ€ô9½¹”((€€€€€€€‘•˜}‰Õ¥±‘}ÍåÍÑ•µ}ÁÉ½µÁÐ¡Í•±˜°ÍåÍÑ•µ}µ•ÍÍ…”õ9½¹”¤è(€€€€€€€€€€€™É½´¡•Éµ•Í}½¹ÍÑ…¹ÑÌ¥µÁ½ÉÐ•Ñ}¡•Éµ•Í}¡½µ”(€€€€€€€€€€€‰Õ¥±Ñ}¡½µ•Ì¹…ÁÁ•¹¡ÍÑÈ¡•Ñ}¡•Éµ•Í}¡½µ” ¤¤¤(€€€€€€€€€€€É•ÑÕÉ¸€‰¥¹¹•ÈÁÉ½µÁÐˆ((€€€±…ÍÌ…­•è(€€€€€€€‘•˜ÕÁ‘…Ñ•}ÍåÍÑ•µ}ÁÉ½µÁÐ¡Í•±˜°Í•ÍÍ¥½¹}¥°ÁÉ½µÁÐ¤è(€€€€€€€€€€€Á…ÍÌ((€€€…•¹Ð€ô…­••¹Ð ¤(€€€…•¹Ð¹}Í•ÍÍ¥½¹}‘ˆ€ô…­• ¤(€€€Í•ÍÍ¥½¸€ôì(€€€€€€€€‰…•¹Ðˆè…•¹Ð°(€€€€€€€€‰Í•ÍÍ¥½¹}­•äˆè€‰Ñ•ÍÐµ­•äˆ°(€€€€€€€€‰ÁÉ½™¥±•}¡½µ”ˆèÍÑÈ¡¥¹¹•É}¡½µ”¤°(€€€ô((€€€½ÕÑ•É}Ñ½­•¸€ôÍ•Ñ}¡•Éµ•Í}¡½µ•}½Ù•ÉÉ¥‘”¡½ÕÑ•É}¡½µ”¤(€€€ÑÉäè(€€€€€€€Í•ÉÙ•È¹}Á•ÉÍ¥ÍÑ}±¥Ù•}Í•ÍÍ¥½¹}ÍåÍÑ•µ}ÁÉ½µÁÐ¡Í•ÍÍ¥½¸¤((€€€€€€€€ŒQ¡”ÁÉ½µÁÐÝ…Ì‰Õ¥±ÐÕ¹‘•ÈÑ¡”Í•ÍÍ¥½¸ÌÁÉ½™¥±”°¹½ÐÑ¡”½ÕÑ•È½¹”¸(€€€€€€€…ÍÍ•ÉÐ‰Õ¥±Ñ}¡½µ•Ì€ôômÍÑÈ¡¥¹¹•É}¡½µ”¥t(€€€€€€€€ŒQ¡”…±±•ÈÌÁÉ”µ•á¥ÍÑ¥¹œ½Ù•ÉÉ¥‘”ÍÕÉÙ¥Ù•°¥¹ÍÑ•…½˜‰•¥¹œÉ•Í•Ð(€€€€€€€€ŒÑ¼9½¹”¸(€€€€€€€…ÍÍ•ÉÐ•Ñ}¡•Éµ•Í}¡½µ•}½Ù•ÉÉ¥‘” ¤€ôôÍÑÈ¡½ÕÑ•É}¡½µ”¤(€€€™¥¹…±±äè(€€€€€€€É•Í•Ñ}¡•Éµ•Í}¡½µ•}½Ù•ÉÉ¥‘”¡½ÕÑ•É}Ñ½­•¸¤(€€€…ÍÍ•ÉÐ•Ñ}¡•Éµ•Í}¡½µ•}½Ù•ÉÉ¥‘” ¤¥Ì9½¹”(()‘•˜Ñ•ÍÑ}Á•ÉÍ¥ÍÑ}±¥Ù•}Í•ÍÍ¥½¹}ÍåÍÑ•µ}ÁÉ½µÁÑ}‰¥¹‘Í}Í•ÍÍ¥½¹}Ý¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰Q¡”ÁÉ½µÁÐÉ•‰Õ¥±…™Ñ•È„±¥Ù”µ½‘•°ÍÝ¥Ñ µÕÍÐÉ•½ÉÑ¡”MMM%=8Ì(€€€Ý½É­¥¹œ‘¥É•Ñ½Éä°¹½ÐÑ¡”ÁÉ½•ÍÌQI5%91}]¸((€€€Q¡”™Õ¹Ñ¥½¸ÉÕ¹Ì½¸Ñ¡”IA‘¥ÍÁ…Ñ¡•ÈÑ¡É•…€¡µ½‘•°¹ÍÝ¥Ñ °½¹™¥œ¹Í•Ð(€€€µ½‘•°¤¸=¸Ñ¡…ÐÑ¡É•…Ñ¡”}MMM%=9}]½¹Ñ•áÑÙ…È¥Ì¹½ÐÍ•Ð°Í¼(€€€É•Í½±Ù•}…•¹Ñ}Ý ¤™…±±Ì‰…¬Ñ¼QI5%91}]°Ý¡¥ Ñ¡”‘•Í­Ñ½ÀÁ¥¹Ì(€€€Ñ¼Ñ¡”¡½µ”‘¥É•Ñ½Éä¸Q¡”ÝÉ½¹œÝ±¥¹”Ñ¡•¸Á•ÉÍ¥ÍÑÌ¥¹Ñ¼Ñ¡”ÍÑ½É•(€€€ÁÉ½µÁÐ¸1…Ñ•ÈÑÕÉ¹ÌÉ•ÍÑ½É”Ñ¡”ÍÑ½É•‰åÑ•ÌÝ¥Ñ¡½ÕÐ¡…¹”€¡Ñ¡”(€€€ÁÉ½±½Õ”É•‰Õ¥±‘Ì½¹±äÝ¡•¸}…¡•‘}ÍåÍÑ•µ}ÁÉ½µÁÐ¥Ì9½¹”¤°Í¼Ñ¡”(€€€Á½¥Í½¹•±¥¹”¹•Ù•ÈÍ•±˜µ¡•…±Ì¸(€€€€ˆˆˆ(€€€Í•ÍÍ¥½¹}Ý€ôÑµÁ}Á…Ñ €¼€‰ÁÉ½©•Ðˆ(€€€Í•ÍÍ¥½¹}Ý¹µ­‘¥È ¤(€€€ÁÉ½•ÍÍ}Ý€ôÑµÁ}Á…Ñ €¼€‰¡½µ”µ™…±±‰…¬ˆ(€€€ÁÉ½•ÍÍ}Ý¹µ­‘¥È ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ•¹Ø ‰QI5%91}]ˆ°ÍÑÈ¡ÁÉ½•ÍÍ}Ý¤¤((€€€Á•ÉÍ¥ÍÑ•€ôíô((€€€±…ÍÌ…­••¹Ðè(€€€€€€€µ½‘•°€ô€‰Ñ•ÍÐµµ½‘•°ˆ(€€€€€€€ÁÉ½Ù¥‘•È€ô€‰Ñ•ÍÐˆ(€€€€€€€Í•ÍÍ¥½¹}¥€ô€‰ÝµÑ•ÍÐµÍ•ÍÍ¥½¸ˆ(€€€€€€€}…¡•‘}ÍåÍÑ•µ}ÁÉ½µÁÐ€ô9½¹”(€€€€€€€}Í•ÍÍ¥½¹}‘ˆ€ô9½¹”((€€€€€€€‘•˜}‰Õ¥±‘}ÍåÍÑ•µ}ÁÉ½µÁÐ¡Í•±˜°ÍåÍÑ•µ}µ•ÍÍ…”õ9½¹”¤è(€€€€€€€€€€€€ŒQ¡”É•…°‰Õ¥±‘•È•µ‰•‘ÌÉ•Í½±Ù•}…•¹Ñ}Ý ¤Ù¥„(€€€€€€€€€€€€ŒÁÉ½µÁÑ}‰Õ¥±‘•È¹‰Õ¥±‘}•¹Ù¥É½¹µ•¹Ñ}¡¥¹ÑÌ ¤¸(€€€€€€€€€€€™É½´…•¹Ð¹ÉÕ¹Ñ¥µ•}Ý¥µÁ½ÉÐÉ•Í½±Ù•}…•¹Ñ}Ý((€€€€€€€€€€€É•ÑÕÉ¸˜‰ÕÉÉ•¹ÐÝ½É­¥¹œ‘¥É•Ñ½ÉäèíÉ•Í½±Ù•}…•¹Ñ}Ý ¥ôˆ((€€€±…ÍÌ…­•è(€€€€€€€‘•˜ÕÁ‘…Ñ•}ÍåÍÑ•µ}ÁÉ½µÁÐ¡Í•±˜°Í•ÍÍ¥½¹}¥°ÁÉ½µÁÐ¤è(€€€€€€€€€€€Á•ÉÍ¥ÍÑ•‘l‰ÁÉ½µÁÐ‰t€ôÁÉ½µÁÐ((€€€…•¹Ð€ô…­••¹Ð ¤(€€€…•¹Ð¹}Í•ÍÍ¥½¹}‘ˆ€ô…­• ¤(€€€Í•ÍÍ¥½¸€ôì(€€€€€€€€‰…•¹Ðˆè…•¹Ð°(€€€€€€€€‰Í•ÍÍ¥½¹}­•äˆè€‰ÝµÑ•ÍÐµÍ•ÍÍ¥½¸ˆ°(€€€€€€€€‰ÝˆèÍÑÈ¡Í•ÍÍ¥½¹}Ý¤°(€€€€€€€€‰•áÁ±¥¥Ñ}ÝˆèQÉÕ”°(€€€€€€€€‰ÁÉ½™¥±•}¡½µ”ˆè9½¹”°(€€€ô((€€€€Œ‰…É”Ñ¡É•…¡…Ì¹¼}MMM%=9}]½¹Ñ•áÑÙ…ÈƒŠPÑ¡”IA‘¥ÍÁ…Ñ¡•ÈÍ¡…Á”¸(€€€É•ÍÕ±Ð€ôíô((€€€‘•˜‘¥ÍÁ…Ñ¡•É}Ñ¡É•… ¤è(€€€€€€€Í•ÉÙ•È¹}Á•ÉÍ¥ÍÑ}±¥Ù•}Í•ÍÍ¥½¹}ÍåÍÑ•µ}ÁÉ½µÁÐ¡Í•ÍÍ¥½¸¤(€€€€€€€É•ÍÕ±Ñl‰…¡•‰t€ô…•¹Ð¹}…¡•‘}ÍåÍÑ•µ}ÁÉ½µÁÐ((€€€Ð€ôÑ¡É•…‘¥¹œ¹Q¡É•…¡Ñ…É•Ðõ‘¥ÍÁ…Ñ¡•É}Ñ¡É•…¤(€€€Ð¹ÍÑ…ÉÐ ¤(€€€Ð¹©½¥¸ ¤((€€€•áÁ•Ñ•€ô˜‰ÕÉÉ•¹ÐÝ½É­¥¹œ‘¥É•Ñ½ÉäèíÍ•ÍÍ¥½¹}Ý‘ôˆ(€€€…ÍÍ•ÉÐÉ•ÍÕ±Ñl‰…¡•‰t€ôô•áÁ•Ñ•°É•ÍÕ±Ñl‰…¡•‰t(€€€…ÍÍ•ÉÐÁ•ÉÍ¥ÍÑ•‘l‰ÁÉ½µÁÐ‰t€ôô•áÁ•Ñ•°Á•ÉÍ¥ÍÑ•‘l‰ÁÉ½µÁÐ‰t(()‘•˜Ñ•ÍÑ}Ý½É­ÍÁ…•}µ½Ù•}É•¡½µ•Í}ÉÕ¹¹¥¹}Í•ÍÍ¥½¸¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆ‰¸•áÁ±¥¥Ð5½Ù”µÑ¼µÁÉ½©•ÐµÕÍÐÝ¥¸™½È„IU99%9Í•ÍÍ¥½¸èÑ¡”ÍÑ½É•(€€€É½Ü…¹Ñ¡”±¥Ù”ÉÕ¹Ñ¥µ”Í•ÍÍ¥½¸É”µ…¹¡½ÈÑ½•Ñ¡•È°¹•Ù•È„U$µÙÌµ‘ˆ(€€€‘¥Í…É••µ•¹Ð€ ŒàØØÈØ¤¸ˆˆˆ(€€€Ñ…É•Ð€ô€‰ÍÑ½É•µÉÕ¹¹¥¹œµÍ•ÍÍ¥½¸ˆ(€€€¹•Ý}Ý€ôÑµÁ}Á…Ñ €¼€‰‘•ÍÐµÁÉ½©•Ðˆ(€€€¹•Ý}Ý¹µ­‘¥È ¤(€€€…ÁÑÕÉ•€ôíô((€€€±…ÍÌ…­•è(€€€€€€€‘•˜•Ñ}Í•ÍÍ¥½¸¡Í•±˜°Í•ÍÍ¥½¹}¥¤è(€€€€€€€€€€€É•ÑÕÉ¸ì‰¥ˆèÍ•ÍÍ¥½¹}¥‘ô((€€€€€€€‘•˜ÕÁ‘…Ñ•}Í•ÍÍ¥½¹}Ý¡Í•±˜°Í•ÍÍ¥½¹}¥°Ý°‰É…¹ õ9½¹”°É½½Ðõ9½¹”°É•Á±…•}¥Ñ}µ•Ñ„õQÉÕ”¤è(€€€€€€€€€€€…ÁÑÕÉ•‘l‰É½Ý}ÕÁ‘…Ñ”‰t€ô€¡Í•ÍÍ¥½¹}¥°Ý¤((€€€€€€€‘•˜±½Í”¡Í•±˜¤è(€€€€€€€€€€€Á…ÍÌ((€€€¥µÁ½ÉÐ½¹Ñ•áÑ±¥ˆ((€€€½¹Ñ•áÑ±¥ˆ¹½¹Ñ•áÑµ…¹…•È(€€€‘•˜}™…­•}‘ˆ¡}Á…É…µÌ°€¨°ÝÉ¥Ñ•Èõ…±Í”¤è(€€€€€€€å¥•±…­• ¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}ÁÉ½™¥±•}‘ˆˆ°}™…­•}‘ˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È¹¥Ñ}ÁÉ½‰”°(€€€€€€€€‰‰É…¹ ˆ°(€€€€€€€±…µ‰‘„Ýè€‰µ…¥¸ˆ°(€€€€¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ (€€€€€€€Í•ÉÙ•È¹¥Ñ}ÁÉ½‰”°(€€€€€€€€‰½µµ½¹}É•Á½}É½½Ðˆ°(€€€€€€€±…µ‰‘„ÝèÍÑÈ¡¹•Ý}Ý¤°(€€€€¤((€€€±¥Ù”€ôì‰Í•ÍÍ¥½¹}­•äˆèÑ…É•Ð°€‰ÉÕ¹¹¥¹œˆèQÉÕ”°€‰ÝˆèÍÑÈ¡ÑµÁ}Á…Ñ €¼€‰½±µÁÉ½©•Ðˆ¥ô(€€€Í•ÉÙ•È¹}Í•ÍÍ¥½¹Íl‰±¥Ù”µÍ¥‰t€ô±¥Ù”(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}É•¥ÍÑ•É}Í•ÍÍ¥½¹}Ýˆ°±…µ‰‘„}Í•ÍÍ¥½¸è9½¹”¤((€€€É•Ì€ôÍ•ÉÙ•È¹}µ•Ñ¡½‘Íl‰Í•ÍÍ¥½¸¹Ý½É­ÍÁ…”¹µ½Ù”‰t (€€€€€€€€‰É¥ˆ°(€€€€€€€ì‰Í•ÍÍ¥½¹}­•äˆèÑ…É•Ð°€‰ÝˆèÍÑÈ¡¹•Ý}Ý¥ô°(€€€€¤((€€€…ÍÍ•ÉÐ€‰•ÉÉ½Èˆ¹½Ð¥¸É•Ì°É•Ì(€€€…ÍÍ•ÉÐ…ÁÑÕÉ•‘l‰É½Ý}ÕÁ‘…Ñ”‰t€ôô€¡Ñ…É•Ð°ÍÑÈ¡¹•Ý}Ý¤¤(€€€…ÍÍ•ÉÐ±¥Ù•l‰Ý‰t€ôôÍÑÈ¡¹•Ý}Ý¤(€€€…ÍÍ•ÉÐ±¥Ù”¹•Ð ‰•áÁ±¥¥Ñ}Ýˆ¤¥ÌQÉÕ”(()‘•˜Ñ•ÍÑ}±½…‘}™}É…Ý}Í••Í}É•Á±…•µ•¹Ñ}Ý¥Ñ¡}Á¥¹¹•‘}µÑ¥µ•}…¹‘}Í¥é”¡µ½¹­•åÁ…Ñ °ÑµÁ}Á…Ñ ¤è(€€€€ˆˆˆŒÄÄÄÄÀÔèÑ¡”É…Üµ½¹™¥œ…¡”µÕÍÐ¹½ÐÍ•ÉÙ”€¡…¹±…Ñ•ÈÝÉ¥Ñ”‰…¬¤„ÍÑ…±”‘½Õµ•¹Ð…™Ñ•È„(€€€Í…µ”µÍ¥é”É•Á±…•µ•¹ÐÑ¡…Ð­••ÁÌÑ¡”½±µÑ¥µ”¸ˆˆˆ(€€€¥µÁ½ÉÐÍ¡ÕÑ¥°((€€€™œ€ôÑµÁ}Á…Ñ €¼€‰½¹™¥œ¹å…µ°ˆ(€€€™œ¹ÝÉ¥Ñ•}Ñ•áÐ ‰µ½‘•°éq¸€‘•™…Õ±Ðè‰‰‰ˆµÉ½ÕÑ•q¸ˆ°•¹½‘¥¹œô‰ÕÑ˜´àˆ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}…Ñ¥Ù•}½¹™¥}Á…Ñ ˆ°±…µ‰‘„è™œ¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}™}…¡”ˆ°9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}™}Í¥œˆ°9½¹”¤(€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡Í•ÉÙ•È°€‰}™}Á…Ñ ˆ°9½¹”¤(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}±½…‘}™}É…Ü ¥l‰µ½‘•°‰ul‰‘•™…Õ±Ð‰t€ôô€‰‰‰‰ˆµÉ½ÕÑ”ˆ(€€€ÍÐ€ô™œ¹ÍÑ…Ð ¤(€€€½Ñ¡•È€ôÑµÁ}Á…Ñ €¼€‰½Ñ¡•È¹å…µ°ˆ(€€€½Ñ¡•È¹ÝÉ¥Ñ•}Ñ•áÐ ‰µ½‘•°éq¸€‘•™…Õ±Ðè………„µÉ½ÕÑ•q¸ˆ°•¹½‘¥¹œô‰ÕÑ˜´àˆ¤(€€€Í¡ÕÑ¥°¹½ÁäÈ¡½Ñ¡•È°™œ¤(€€€½Ì¹ÕÑ¥µ”¡™œ°¹Ìô¡ÍÐ¹ÍÑ}…Ñ¥µ•}¹Ì°ÍÐ¹ÍÑ}µÑ¥µ•}¹Ì¤¤(€€€…ÍÍ•ÉÐÍ•ÉÙ•È¹}±½…‘}™}É…Ü ¥l‰µ½‘•°‰ul‰‘•™…Õ±Ð‰t€ôô€‰………„µÉ½ÕÑ”ˆ