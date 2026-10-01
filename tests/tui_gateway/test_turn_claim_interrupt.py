"""An automatic turn's claim on the session is ended by a Stop that lands before its turn starts.

Automatic turns (a notification wake, crash auto-continue, the queued-prompt drain, a goal continuation, a bot
mailbox delivery) claim the session (``running=True``) and only later start the worker thread that
``session.interrupt`` watches through ``_run_thread``. A Stop in that window, followed by a user prompt, must leave
exactly one turn running: the user's. It keeps ``running``, its in-flight turn and the published handle.

A turn source that fails to start its turn releases its claim, and that release must not clear a claim that a user
prompt took in the meantime."""

import contextlib
import threading
import time
import types

import pytest

from tui_gateway import server

KEY = "turn-claim-key"
SID = "turn-claim-sid"


class _Lease:
    lease_id = "lease-1"
    released = False
    track_liveness = False
    enabled = True

    def release(self):
        self.released = True


def _idle_session() -> dict:
    ready = threading.Event()
    ready.set()
    return {
        "agent": types.SimpleNamespace(session_id="agent-sid", clear_interrupt=lambda: None, interrupt=lambda *_a: None),
        "agent_ready": ready,
        "active_session_lease": _Lease(),
        "session_key": KEY,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "transport": None,
        "inflight_turn": None,
    }


class _Turns:
    """The turn body: records which prompts reached it and keeps each turn open until ``finish`` is set."""

    def __init__(self):
        self.ran, self.threads, self.started = [], {}, {}
        self.finish = threading.Event()
        self.workers = []

    def label(self, text) -> str:
        return "user" if text == "user" else "automatic"

    def prepare(self, _sid, _session, _st, text, _images):
        label = self.label(text)
        self.ran.append(label)
        self.threads[label] = threading.current_thread()
        self.started.setdefault(label, threading.Event()).set()
        assert self.finish.wait(10)
        return None

    def wait_started(self, label) -> bool:
        return self.started.setdefault(label, threading.Event()).wait(10)

    def join_all(self):
        index = 0
        while index < len(self.workers):
            self.workers[index].join(10)
            index += 1


@pytest.fixture
def turn_env(monkeypatch):
    turns = _Turns()
    real_start = server._start_session_work

    def start_session_work(target, **kwargs):
        thread = real_start(target, **kwargs)
        if thread is not None:
            turns.workers.append(thread)
        return thread

    monkeypatch.setattr(server, "_start_session_work", start_session_work)
    monkeypatch.setattr(server, "_prepare_turn_input", turns.prepare)
    monkeypatch.setattr(server, "_record_turn_marker", lambda *_a, **_k: "")
    monkeypatch.setattr(server, "_retire_turn_marker", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_finish_turn", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_emit_settled_session_info", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_routing_provenance_db", lambda _session: contextlib.nullcontext(None))
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_write_submit_user_row", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *_a, **_k: False)
    monkeypatch.setattr(server, "_wait_agent", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
    return turns


def _user_submit() -> dict:
    resp = server.handle_request({"id": "user", "method": "prompt.submit", "params": {"session_id": SID, "text": "user"}})
    return resp.get("result") or resp["error"]


def _kanban_wake(monkeypatch, session):
    monkeypatch.setattr(server, "_collect_kanban_notifications", lambda _session: [])
    session["_kanban_pending"] = ["kanban"]
    return threading.Thread(target=server._notif_poll_kanban_scoped, args=(SID, session))


def _auto_continue(monkeypatch, session):
    monkeypatch.setattr(server, "read_turn_marker", lambda *_a: {
        "prompt": "interrupted", "started_at": time.time(), "attempts": 0, "writer_pid": 0})
    monkeypatch.setattr(server, "marker_writer_state", lambda _marker: "dead")
    return threading.Thread(target=server._maybe_schedule_auto_continue, args=(SID, session, KEY))


def _plugin_inject_drain(_monkeypatch, _session):
    return threading.Thread(target=server.inject_tui_session_message, kwargs={"session_key": KEY, "content": "injected"})


def _goal_continuation(_monkeypatch, session):
    # The previous turn's worker runs the continuation after its turn, and is still the published handle.
    thread = threading.Thread(target=server._run_post_turn_followups, args=("goal", SID, session, {}, "goal"))
    session["_run_thread"] = thread
    return thread


def _bot_delivery(monkeypatch, session):
    from tools import bot_live_delivery

    monkeypatch.setattr(bot_live_delivery, "has_mailbox", lambda _home: True)
    monkeypatch.setattr(bot_live_delivery, "find_canonical_live_owner", lambda _home: {
        "lease_id": session["active_session_lease"].lease_id, "live_session_id": SID, "session_id": KEY})
    monkeypatch.setattr(bot_live_delivery, "claim_pending_delivery", lambda *_a: {"id": "d1", "message": "delivery"})
    monkeypatch.setattr(bot_live_delivery, "complete_delivery", lambda *_a, **_k: None)
    return threading.Thread(target=server._poll_bot_live_delivery_once, args=(SID, session))


@pytest.mark.parametrize("start_automatic_turn", [
    pytest.param(_kanban_wake, id="notification-wake"),
    pytest.param(_auto_continue, id="auto-continue"),
    pytest.param(_plugin_inject_drain, id="queued-prompt-drain"),
    pytest.param(_goal_continuation, id="goal-continuation"),
    pytest.param(_bot_delivery, id="bot-delivery"),
])
def test_a_stop_before_an_automatic_turn_starts_leaves_only_the_next_prompt_running(
    monkeypatch, turn_env, start_automatic_turn
):
    session = _idle_session()
    claimed, release_claimant = threading.Event(), threading.Event()
    claimant_threads = []

    def ensure_row(_session):
        # The automatic turn is the first to enter _run_prompt_submit. Block it there, after its claim.
        if not claimed.is_set():
            claimant_threads.append(threading.current_thread())
            claimed.set()
            assert release_claimant.wait(10)
        return True

    monkeypatch.setattr(server, "_ensure_session_db_row", ensure_row)
    server._sessions[SID] = session
    try:
        start_automatic_turn(monkeypatch, session).start()
        assert claimed.wait(10)
        stopped = server.handle_request({"id": "stop", "method": "session.interrupt", "params": {"session_id": SID}})
        assert stopped["result"] == {"status": "interrupted"}
        user = _user_submit()
        if user == {"status": "streaming"}:
            assert turn_env.wait_started("user")
        release_claimant.set()
        claimant_threads[0].join(10)

        with session["history_lock"]:
            published = session.get("_run_thread")
            outcome = {
                "user": user,
                "running": session["running"],
                "inflight_user": (session.get("inflight_turn") or {}).get("user"),
                "published": next((label for label, t in turn_env.threads.items() if t is published), None),
            }
    finally:
        release_claimant.set()
        turn_env.finish.set()
        turn_env.join_all()
        server._sessions.pop(SID, None)

    assert {**outcome, "ran": turn_env.ran} == {
        "user": {"status": "streaming"},
        "running": True,
        "inflight_user": "user",
        "published": "user",
        "ran": ["user"],
    }


def _heartbeat_tick(monkeypatch, session):
    from hermes_cli import heartbeat

    class _DueHeartbeat:
        state = types.SimpleNamespace(is_due=lambda: True, fire_count=1)

        def __init__(self, session_id):
            del session_id

        def is_active(self):
            return True

        def due_prompt(self):
            return "heartbeat"

        def abandon_fire(self):
            return None

    monkeypatch.setattr(heartbeat, "HeartbeatManager", _DueHeartbeat)
    monkeypatch.setattr(server, "_notif_gateway_owns_heartbeat", lambda *_a: False)
    return lambda: server._maybe_fire_tui_heartbeat_tick(SID, session)


def _refused_bot_delivery(monkeypatch, session):
    _bot_delivery(monkeypatch, session)
    return lambda: server._poll_bot_live_delivery_once(SID, session)


@pytest.mark.parametrize("automatic_turn", [
    pytest.param(_heartbeat_tick, id="heartbeat"),
    pytest.param(_refused_bot_delivery, id="bot-delivery"),
])
def test_releasing_a_refused_automatic_turn_leaves_a_later_prompt_running(monkeypatch, turn_env, automatic_turn):
    session = _idle_session()
    refusals, user = ["another backend owns this session"], {}

    def ensure_slot(_sid, _session):
        return refusals.pop() if refusals else None

    def emit(event, _sid, *_a, **_k):
        # The refusal has released the automatic turn's claim. A user prompt claims the session before the turn
        # source's own release runs.
        if event == "error" and not user:
            user.update(_user_submit())
            assert turn_env.wait_started("user")

    monkeypatch.setattr(server, "_ensure_active_session_slot", ensure_slot)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: True)
    monkeypatch.setattr(server, "_emit", emit)
    server._sessions[SID] = session
    run_automatic_turn = automatic_turn(monkeypatch, session)
    try:
        run_automatic_turn()
        with session["history_lock"]:
            published = session.get("_run_thread")
            outcome = {
                "user": user,
                "running": session["running"],
                "inflight_user": (session.get("inflight_turn") or {}).get("user"),
                "published": next((label for label, t in turn_env.threads.items() if t is published), None),
            }
    finally:
        turn_env.finish.set()
        turn_env.join_all()
        server._sessions.pop(SID, None)

    assert {**outcome, "ran": turn_env.ran} == {
        "user": {"status": "streaming"},
        "running": True,
        "inflight_user": "user",
        "published": "user",
        "ran": ["user"],
    }


@pytest.mark.parametrize("source", ["automatic", "submit"])
@pytest.mark.parametrize("replace_claim", [False, True], ids=["cancelled", "replaced"])
def test_stop_during_lease_admission_prevents_the_turn(monkeypatch, turn_env, source, replace_claim):
    session = _idle_session()
    entered, release = threading.Event(), threading.Event()
    errors = []

    def ensure_slot(_sid, _session):
        if threading.current_thread() is threading.main_thread():
            return None
        entered.set()
        assert release.wait(5)
        return None

    def emit(event, _sid, payload=None, *_args, **_kwargs):
        if event == "error":
            errors.append(payload)

    monkeypatch.setattr(server, "_ensure_active_session_slot", ensure_slot)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: True)
    monkeypatch.setattr(server, "_emit", emit)
    server._sessions[SID] = session
    dispatch = None
    successor = object()
    try:
        if source == "automatic":
            with session["history_lock"]:
                claim = server._claim_session_turn(session)
            dispatch = threading.Thread(target=lambda: server._run_prompt_submit(
                "automatic", SID, session, "automatic", turn_claim=claim))
            dispatch.start()
        else:
            assert _user_submit() == {"status": "streaming"}
            dispatch = session["_run_thread"]
        assert entered.wait(5)
        stopped = server.handle_request({"id": "stop", "method": "session.interrupt",
            "params": {"session_id": SID}})
        assert stopped["result"] == {"status": "interrupted"}
        if replace_claim:
            with session["history_lock"]:
                session["running"] = False
                server._claim_session_turn(session)
                session["_turn_cancel_requested"] = False
                session["inflight_turn"] = {"user": "successor"}
                session["_submit_user_row"] = {"content": "successor"}
                session["_run_thread"] = successor
        release.set()
        dispatch.join(5)
        observed = {
            "ran": turn_env.ran,
            "running": session["running"],
            "errors": errors,
            "inflight": session.get("inflight_turn"),
            "staged": session.get("_submit_user_row"),
            "successor_published": session.get("_run_thread") is successor,
        }
    finally:
        release.set()
        turn_env.finish.set()
        if dispatch is not None:
            dispatch.join(5)
        turn_env.join_all()
        server._sessions.pop(SID, None)

    expected = {
        "ran": [], "running": False,
        "errors": [{"message": "Turn cancelled before the agent started"}],
        "inflight": None, "staged": None, "successor_published": False,
    }
    if replace_claim:
        expected.update(
            running=True, errors=[], inflight={"user": "successor"},
            staged={"content": "successor"}, successor_published=True)
    assert observed == expected


def test_stale_auto_continue_does_not_stage_marker_inputs(monkeypatch, turn_env):
    session = _idle_session()
    entered, release = threading.Event(), threading.Event()
    marker_inputs = []

    def ensure_slot(_sid, _session):
        if threading.current_thread() is threading.main_thread() or entered.is_set():
            return None
        entered.set()
        assert release.wait(5)
        return None

    def record_marker(_session, text, **_kwargs):
        marker_inputs.append((text, _session.pop("_auto_continue_prompt", None)))
        _session.pop("_auto_continue_attempt", None)
        return ""

    monkeypatch.setattr(server, "_ensure_active_session_slot", ensure_slot)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: True)
    monkeypatch.setattr(server, "_record_turn_marker", record_marker)
    server._sessions[SID] = session
    scheduler = _auto_continue(monkeypatch, session)
    try:
        scheduler.start()
        assert entered.wait(5)
        server.handle_request({"id": "stop", "method": "session.interrupt", "params": {"session_id": SID}})
        assert _user_submit() == {"status": "streaming"}
        assert turn_env.wait_started("user")
        release.set()
        scheduler.join(5)
        for worker in turn_env.workers:
            if worker.name.startswith("auto-continue"):
                worker.join(5)
        marker_state = {key: session[key] for key in ("_auto_continue_prompt", "_auto_continue_attempt") if key in session}
    finally:
        release.set()
        turn_env.finish.set()
        scheduler.join(5)
        turn_env.join_all()
        server._sessions.pop(SID, None)
    assert {"markers": marker_inputs, "staged": marker_state} == {"markers": [("user", None)], "staged": {}}


@pytest.mark.parametrize("submitted", [False, True], ids=["automatic", "submit"])
@pytest.mark.parametrize("stop_after_publication", [False, True], ids=["before-start", "after-publication"])
def test_a_stop_during_worker_publication_emits_a_terminal_frame(
    monkeypatch, turn_env, submitted, stop_after_publication
):
    session = _idle_session()
    events, workers = [], []
    worker_gate = threading.Event()
    real_decide = server._decide_turn_thread
    stopped = False

    def stop():
        nonlocal stopped
        stopped = True
        response = server.handle_request(
            {"id": "stop", "method": "session.interrupt", "params": {"session_id": SID}})
        assert response["result"] == {"status": "interrupted"}

    class Worker(threading.Thread):
        def start(self):
            if not stop_after_publication:
                stop()
            super().start()

    def spawn(target, *, name, **_kwargs):
        def delayed_target():
            if stop_after_publication:
                assert worker_gate.wait(10)
            target()

        worker = Worker(target=delayed_target, name=name, daemon=True)
        workers.append(worker)
        return worker

    def decide(target_session, worker, claim, decision):
        admitted = real_decide(target_session, worker, claim, decision)
        if stop_after_publication and threading.current_thread() is not worker and not stopped:
            stop()
            worker_gate.set()
        return admitted

    def prepare(_sid, _session, _st, text, _images):
        turn_env.ran.append(turn_env.label(text))
        return None

    monkeypatch.setattr(server, "_prepare_turn_input", prepare)
    monkeypatch.setattr("agent.memory_provider.spawn_context_thread", spawn)
    monkeypatch.setattr(server, "_decide_turn_thread", decide)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: True)
    monkeypatch.setattr(server, "_emit", lambda event, _sid, *payload: events.append((event, payload)))
    server._sessions[SID] = session
    try:
        if submitted:
            response = _user_submit()
            assert response == {"status": "streaming"}
            session["_run_thread"].join(10)
        else:
            with session["history_lock"]:
                claim = server._claim_session_turn(session)
            server._run_prompt_submit("automatic", SID, session, "automatic", turn_claim=claim)
        for worker in workers:
            worker.join(10)
        outcome = {
            "running": session["running"], "inflight": session.get("inflight_turn"),
            "ran": turn_env.ran, "terminal": [payload for event, payload in events if event == "error"],
        }
    finally:
        worker_gate.set()
        turn_env.finish.set()
        for worker in workers:
            worker.join(10)
        turn_env.join_all()
        server._sessions.pop(SID, None)

    assert outcome == {
        "running": False, "inflight": None, "ran": [],
        "terminal": [({"message": "Turn cancelled before the agent started"},)],
    }
