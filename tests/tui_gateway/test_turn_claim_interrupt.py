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
    monkeypatch.setattr(server, "_release_turn_scopes", lambda *_a, **_k: None)
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

def test_a_replaced_worker_keeps_the_successor_input_mark(monkeypatch, turn_env):
    session = _idle_session()
    entered, release = threading.Event(), threading.Event()

    def prepare(_sid, _session, _st, _text, _images):
        entered.set()
        assert release.wait(10)
        return None

    monkeypatch.setattr(server, "_prepare_turn_input", prepare)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: True)
    server._sessions[SID] = session
    worker: threading.Thread | None = None
    try:
        submitted = _user_submit()
        assert entered.wait(10)
        worker = session["_run_thread"]
        assert isinstance(worker, threading.Thread)
        with session["history_lock"]:
            session["running"] = False
            successor_claim = server._claim_session_turn(session)
            session["_turn_user_input"] = True
            session["inflight_turn"] = {"user": "successor"}
        release.set()
        worker.join(10)
        with session["history_lock"]:
            outcome = {
                "submit": submitted,
                "worker_finished": not worker.is_alive(),
                "running": session["running"],
                "input_mark": session.get("_turn_user_input"),
                "claim": session.get("_turn_claim"),
                "inflight": session.get("inflight_turn"),
            }
    finally:
        release.set()
        if worker is not None:
            worker.join(10)
        server._sessions.pop(SID, None)

    assert outcome == {
        "submit": {"status": "streaming"},
        "worker_finished": True,
        "running": True,
        "input_mark": True,
        "claim": successor_claim,
        "inflight": {"user": "successor"},
    }
