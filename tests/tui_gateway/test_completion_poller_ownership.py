"""A live poller and post-turn fallback must not split one ready backlog (#104671).

Real TUI routing, ProcessRegistry and queue operations; the model-submit boundary
and unrelated periodic work are replaced. No inference or renderer is exercised.
"""

import queue
import threading

import pytest


class _PausedFirstGet(queue.Queue):
    """Preempt the poller after its first dequeue, before its ready snapshot."""

    def __init__(self, poller_name):
        super().__init__()
        self.poller_name = poller_name
        self.popped = threading.Event()
        self.resume = threading.Event()
        self.timed_out = threading.Event()

    def get(self, block=True, timeout=None):
        event = super().get(block=block, timeout=timeout)
        if threading.current_thread().name == self.poller_name and not self.popped.is_set():
            self.popped.set()
            if not self.resume.wait(10):
                self.timed_out.set()
                raise TimeoutError("test did not release the paused poller")
        return event


@pytest.fixture
def delivery(monkeypatch):
    from tools import process_registry as pr
    from tui_gateway import server

    registry = pr.ProcessRegistry()
    # Discard any restored delegation envelopes; this fixture supplies its own backlog.
    registry.completion_queue = queue.Queue()
    monkeypatch.setattr(pr, "process_registry", registry)
    sid = "completion-owner-ui"
    session = {"session_key": "completion-owner", "history_lock": threading.RLock(), "running": False}
    calls, errors = [], []
    dispatched = threading.Event()
    monkeypatch.setattr(server, "_sessions", {sid: session})
    monkeypatch.setattr(server, "_notification_pollers", [])
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_args: False)
    for name in ("_wire_desktop_sinks", "_emit", "_poll_bot_live_delivery_guarded",
                 "_maybe_fire_tui_loop_tick", "_maybe_fire_tui_heartbeat_tick", "_notif_poll_kanban"):
        monkeypatch.setattr(server, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_notif_log_failure", lambda what, exc: errors.append((what, str(exc))))
    monkeypatch.setattr(server, "_hook_failure", lambda what, exc: errors.append((what, str(exc))))

    def submit(_rid, delivered_sid, current, text, **kwargs):
        calls.append((delivered_sid, str(text), kwargs))
        with current["history_lock"]:
            current["running"] = False
        dispatched.set()
        return True

    monkeypatch.setattr(server, "_run_prompt_submit", submit)
    yield server, registry, sid, session, calls, errors, dispatched
    # Even an assertion failure must not leave a poller stealing the next test's events.
    for stop, thread in server._notification_pollers:
        stop.set()
    if isinstance(registry.completion_queue, _PausedFirstGet):
        registry.completion_queue.resume.set()
    for _stop, thread in server._notification_pollers:
        thread.join(10)
        assert not thread.is_alive(), thread.name


def _completion(index, owner="completion-owner"):
    return {"type": "completion", "session_id": f"proc_backlog_{index:02d}",
            "session_key": owner, "task_id": owner, "owner_task_id": owner,
            "command": f"echo fixture-{index}", "exit_code": 7 if index == 11 else 0,
            "output": f"RESULT[{index:02d}]", "started_at": 1.0}


@pytest.mark.parametrize("stopping", [False, True], ids=["running-poller", "shutdown-drain"])
def test_post_turn_does_not_split_the_live_pollers_ready_backlog(delivery, stopping):
    server, registry, sid, session, calls, errors, dispatched = delivery
    events = [_completion(index) for index in range(12)]
    pending = registry.completion_queue = _PausedFirstGet(f"tui-notif-poller-{sid}")
    for event in events:
        pending.put(event)
    stop = session["_notif_stop"] = server._start_notification_poller(sid, session)
    try:
        assert pending.popped.wait(10), errors
        if stopping:
            stop.set()  # Still alive: its final drain has not finished yet.
        server._run_post_turn_followups("finished-human-turn", sid, session, {}, None)
        # On the old path, post-turn steals the other eleven and submits them separately.
        assert calls == [], calls
        assert pending.qsize() == len(events) - 1
        pending.resume.set()
        assert dispatched.wait(10), errors
    finally:
        pending.resume.set()
        stop.set()
        for token, thread in server._notification_pollers:
            if token is stop:
                thread.join(10)
                assert not thread.is_alive(), thread.name
    assert not pending.timed_out.is_set()
    assert not errors, errors
    assert len(calls) == 1, calls
    assert calls[0][0] == sid
    for event in events:
        assert calls[0][1].count(event["output"]) == 1
    from tools.process_registry_notifications import format_process_notification
    # Preserve the failure, not just successful output markers.
    assert format_process_notification(events[-1]) in calls[0][1]
    assert pending.empty()


@pytest.mark.parametrize("poller_state", ["absent", "dead", "foreign"])
def test_post_turn_keeps_the_fallback_without_its_own_live_poller(delivery, poller_state):
    server, registry, sid, session, calls, errors, _dispatched = delivery
    release = threading.Event()
    thread = None
    if poller_state != "absent":
        token = session["_notif_stop"] = threading.Event()
        thread = threading.Thread(target=release.wait, args=(10,), name="fixture-poller")
        thread.start()
        if poller_state == "dead":
            release.set()
            thread.join(10)
            assert not thread.is_alive()
        registered_token = token if poller_state == "dead" else threading.Event()
        server._notification_pollers.append((registered_token, thread))
    owned, foreign = _completion(0), _completion(1, "other-owner")
    registry.completion_queue.put(owned)
    registry.completion_queue.put(foreign)
    try:
        server._run_post_turn_followups("finished-human-turn", sid, session, {}, None)
        assert not errors, errors
        assert len(calls) == 1, calls
        assert calls[0][0] == sid
        assert owned["output"] in calls[0][1]
        assert foreign["output"] not in calls[0][1]
        assert registry.completion_queue.get_nowait() == foreign
        assert registry.completion_queue.empty()
    finally:
        release.set()
        if thread is not None:
            thread.join(10)
            assert not thread.is_alive()
