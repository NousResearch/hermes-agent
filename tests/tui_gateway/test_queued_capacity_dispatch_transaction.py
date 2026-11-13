"""Queue admission is transactional until execution, including settlement races."""

import copy
import threading
import types

import pytest

from hermes_cli import active_sessions
from hermes_state import SessionDB
from tui_gateway import server
from tests.tui_gateway.test_queued_capacity_recovery import make_poller_tick


@pytest.fixture
def poller_tick(monkeypatch):
    return make_poller_tick(monkeypatch)


@pytest.fixture(params=["inline", "compute_host"])
def dispatch_env(monkeypatch, tmp_path, request):
    execution = request.param
    cfg = {"max_concurrent_sessions": 1}
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr(server, "_canonical_ledger_refusal", lambda session: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_transport_is_dead", lambda transport: False)
    monkeypatch.setattr(server, "_attach_session_transport", lambda *args: None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda session, cfg=None: execution == "compute_host")
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {"turn_isolation": True})
    monkeypatch.setattr(server, "_compute_host_session_info", lambda session: {})
    workers, frames = [], []
    monkeypatch.setattr(server, "_start_session_work", lambda target, **kwargs: workers.append(target) or object())
    supervisor = types.SimpleNamespace(submit_turn=lambda frame, **kwargs: frames.append(frame))
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda *args: supervisor)
    response = server.handle_request({"id": "create", "method": "session.create", "params": {"source": "desktop"}})
    assert "result" in response, response
    sid = response["result"]["session_id"]
    session = server._sessions[sid]
    session["transport"] = "socket-test"
    session["agent"] = types.SimpleNamespace(clear_interrupt=lambda: None)
    session["history"] = [{"role": "user", "content": "original context"}]
    assert server._ensure_active_session_slot(sid, session) is None
    server._ensure_session_db_row(session)
    env = types.SimpleNamespace(execution=execution, db=db, sid=sid, session=session, workers=workers,
                                frames=frames, supervisor=supervisor)
    env.started = lambda: len(workers if execution == "inline" else frames)
    env.accept = lambda text: _accept(env, text)
    try:
        yield env
    finally:
        server._release_active_session_slot(session)
        server._sessions.pop(sid, None)
        db.close()


def _accept(env, text):
    env.session["attached_images"] = [text + ".png"]
    response = server.handle_request({"id": text, "method": "prompt.submit",
                                      "params": {"session_id": env.sid, "text": text, "queued": True}})
    assert response.get("result", {}).get("status") == "queued", response
    return response


def _accepted_pair(env):
    env.session["running"] = True
    env.accept("Q1")
    env.accept("Q2")
    env.session["running"] = False
    return env.session["queued_prompt"], env.session["queued_prompts"][0]


def _run_thread(target):
    errors = []

    def run():
        try:
            target()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, errors


def _join(thread, errors):
    thread.join(10)
    assert not thread.is_alive(), "dispatch barrier did not complete"
    assert errors == []


def test_second_activation_refusal_restores_identical_fifo_then_poller_recovers(
    monkeypatch, dispatch_env, poller_tick
):
    env, session = dispatch_env, dispatch_env.session
    head, tail = _accepted_pair(env)
    accepted = copy.deepcopy([head, tail])
    durable = env.db.get_messages_as_conversation(session["session_key"], include_inactive=True, include_row_ids=True)
    activate = server._activate_active_session_slot
    at_second, refuse = threading.Event(), threading.Event()
    calls = []

    def second_activation(s):
        calls.append(s)
        if len(calls) == 2:
            at_second.set()
            assert refuse.wait(10)
            return active_sessions.ActiveSessionRefusal(
                "registry temporarily unavailable", active_sessions.SESSION_COORDINATION_UNAVAILABLE)
        return activate(s)

    monkeypatch.setattr(server, "_activate_active_session_slot", second_activation)
    thread, errors = _run_thread(lambda: server._drain_queued_prompt("settled", env.sid, session))
    assert at_second.wait(10)
    assert env.started() == 0
    refuse.set()
    _join(thread, errors)
    assert session.get("queued_prompt") is head
    assert session.get("queued_prompts") == [tail]
    assert [head, tail] == accepted
    assert session.get("_queued_prompt_capacity_wait") == {"rid": "settled", "generation": 0}
    assert not session["running"] and env.started() == 0
    assert env.db.get_messages_as_conversation(
        session["session_key"], include_inactive=True, include_row_ids=True) == durable
    lease = session["active_session_lease"]
    assert next(entry for entry in active_sessions.active_session_registry_snapshot()
                if entry["lease_id"] == lease.lease_id)["capacity_active"] is False
    # Admission of another real running lease proves preflight did not leak capacity.
    blocker, refusal = active_sessions.try_acquire_active_session(
        session_id="other-turn", surface="gateway:telegram", config={"max_concurrent_sessions": 1})
    assert refusal is None and blocker is not None
    blocker.release()
    stop = poller_tick(env.sid, session)
    assert env.started() == 1 and session["queued_prompt"] is tail
    poller_tick(env.sid, session, stop=stop)
    assert env.started() == 1
    server._notif_release_turn(session)
    assert server._drain_queued_prompt("settled", env.sid, session)
    assert env.started() == 2 and not session.get("queued_prompt")
    if env.execution == "compute_host":
        assert [frame["text"] for frame in env.frames] == ["Q1", "Q2"]


@pytest.mark.parametrize("invalidation", ["stop", "generation"])
def test_second_activation_refusal_does_not_restore_invalidated_claim(monkeypatch, dispatch_env, invalidation):
    env, session = dispatch_env, dispatch_env.session
    _accepted_pair(env)
    activate = server._activate_active_session_slot
    calls = []
    replacement = {"text": "replacement"}

    def second_activation(s):
        calls.append(s)
        if len(calls) == 2:
            # Admission owns history_lock: Stop/generation mutation is atomic here.
            server._ac_set_queue(s, [] if invalidation == "stop" else [replacement])
            s["_queued_prompt_generation"] = 1
            if invalidation == "stop":
                s["_turn_cancel_requested"] = True
            return active_sessions.ActiveSessionRefusal("unavailable", active_sessions.SESSION_COORDINATION_UNAVAILABLE)
        return activate(s)

    monkeypatch.setattr(server, "_activate_active_session_slot", second_activation)
    assert server._drain_queued_prompt("settled", env.sid, session)
    assert session.get("queued_prompt") is (None if invalidation == "stop" else replacement)
    assert not session.get("_queued_prompt_capacity_wait")
    assert not session["running"] and env.started() == 0
    blocker, refusal = active_sessions.try_acquire_active_session(
        session_id="other-turn", surface="gateway:telegram", config={"max_concurrent_sessions": 1})
    assert refusal is None and blocker is not None
    blocker.release()


def test_settlement_window_keeps_accepted_fifo_ahead_of_fresh_rpc(monkeypatch, dispatch_env):
    env, session = dispatch_env, dispatch_env.session
    head, tail = _accepted_pair(env)
    session["running"] = True
    settled, resume = threading.Event(), threading.Event()
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    monkeypatch.setattr(server, "_interrupt_busy_session", lambda *args: pytest.fail("no live turn to interrupt"))

    def barrier(*args):
        assert session["running"] is False
        assert not session.get("_queued_prompt_capacity_wait")
        settled.set()
        assert resume.wait(10)

    if env.execution == "compute_host":
        monkeypatch.setattr(server, "_apply_compute_host_metadata_mirror", barrier)
        settle = lambda: server._on_compute_host_turn_done("original", env.sid, session, {"type": "turn.done"})
    else:
        # Exercise the actual inline worker's finally and follow-up drain without
        # invoking a model, voice, tools, or writing a crash marker.
        monkeypatch.setattr(server, "_record_turn_marker", lambda *args, **kwargs: "")
        def model_failure(*args):
            raise RuntimeError("provider ended the original turn")
        monkeypatch.setattr(server, "_prepare_turn_input", model_failure)
        monkeypatch.setattr(server, "_recover_turn_exception", lambda *args: None)
        monkeypatch.setattr(server, "_goal_followup_after_turn", lambda *args: None)
        monkeypatch.setattr(server, "_release_turn_scopes", lambda *args: None)
        monkeypatch.setattr(server, "_emit_settled_session_info", barrier)
        monkeypatch.setattr(server, "_post_turn_housekeeping", lambda *args: None)
        monkeypatch.setattr(server, "_retire_turn_marker", lambda *args: None)
        assert server._run_prompt_submit("original", env.sid, session, "original") is True
        settle = env.workers.pop()
    thread, errors = _run_thread(settle)
    assert settled.wait(10)
    try:
        session["attached_images"] = []
        response = server.handle_request({"id": "Q3", "method": "prompt.submit",
            "params": {"session_id": env.sid, "text": "Q3"}})
        assert response.get("result", {}).get("status") == "queued", response
        assert env.started() == 0
        assert session["queued_prompt"] is head
        assert session["queued_prompts"][0] is tail
        assert [entry["text"] for entry in session["queued_prompts"]] == ["Q2", "Q3"]
    finally:
        resume.set()
        _join(thread, errors)
    assert env.started() == 1
    assert session["queued_prompt"] is tail
    if env.execution == "compute_host":
        assert env.frames[0]["text"] == "Q1"


def test_idle_rpc_capacity_refusal_uses_canonical_formatter(monkeypatch, dispatch_env):
    env, session = dispatch_env, dispatch_env.session
    blocker, refusal = active_sessions.try_acquire_active_session(
        session_id="other-turn", surface="gateway:telegram", config={"max_concurrent_sessions": 1})
    assert refusal is None and blocker is not None
    try:
        response = server.handle_request({"id": "refused", "method": "prompt.submit",
                                          "params": {"session_id": env.sid, "text": "must not run"}})
        assert response.get("error", {}).get("code") == 4090, response
        assert response["error"]["data"]["reason"] == active_sessions.MAX_CONCURRENT_SESSIONS
        assert not session["running"] and env.started() == 0
        assert not session.get("queued_prompt") and not session.get("inflight_turn")
        assert env.db.get_messages_as_conversation(session["session_key"]) == []
    finally:
        blocker.release()


def test_post_admission_failure_does_not_replay_accepted_head(monkeypatch, dispatch_env, poller_tick):
    env, session = dispatch_env, dispatch_env.session
    head, _tail = _accepted_pair(env)
    if env.execution == "compute_host":
        def fail_after_send(frame, **kwargs):
            env.frames.append(frame)
            raise RuntimeError("pipe result uncertain after send")
        env.supervisor.submit_turn = fail_after_send
    else:
        def model_failure(*args):
            raise RuntimeError("provider failed after the worker began")

        monkeypatch.setattr(server, "_record_turn_marker", lambda *args, **kwargs: "")
        monkeypatch.setattr(server, "_prepare_turn_input", model_failure)
        monkeypatch.setattr(server, "_recover_turn_exception", lambda *args: None)
        monkeypatch.setattr(server, "_goal_followup_after_turn", lambda *args: None)
        monkeypatch.setattr(server, "_release_turn_scopes", lambda *args: None)
        monkeypatch.setattr(server, "_emit_settled_session_info", lambda *args: None)
        monkeypatch.setattr(server, "_post_turn_housekeeping", lambda *args: None)
        monkeypatch.setattr(server, "_retire_turn_marker", lambda *args: None)
        monkeypatch.setattr(server, "_run_post_turn_followups", lambda *args: None)

        def fail_after_start(target, **kwargs):
            env.workers.append(target)
            target()  # the real worker enters, fails, and settles before start's result is lost
            raise RuntimeError("worker start result uncertain")
        monkeypatch.setattr(server, "_start_session_work", fail_after_start)
    assert server._drain_queued_prompt("settled", env.sid, session)
    assert session.get("queued_prompt") is not head
    assert not session.get("_queued_prompt_capacity_wait")
    before = env.started()
    poller_tick(env.sid, session)
    assert env.started() == before
