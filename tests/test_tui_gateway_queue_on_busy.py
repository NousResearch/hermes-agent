"""A prompt that lands mid-turn is interrupted + queued, never dropped.

Before this, ``prompt.submit`` on a running session returned ``session busy``,
forcing clients into a deadline-bounded busy-retry. When turn teardown outlived
the deadline — e.g. a slow, non-interruptible tool (``web_search``) still
running when the user hit stop — the resubmitted message was silently dropped
("it just doesn't listen"). The gateway now applies the ``busy_input_mode``
policy: interrupt the live turn (default) and queue the message to run as the
next turn, drained in ``run``'s tail.
"""

import threading
import types

import tools.async_delegation as ad
from tui_gateway import server


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "transport": None,
        "attached_images": [],
        **extra,
    }


# ── _enqueue_prompt ────────────────────────────────────────────────────────

def test_enqueue_pins_text_and_transport():
    session = _session()
    server._enqueue_prompt(session, "hello", "ws-1")
    assert session["queued_prompt"] == {"text": "hello", "transport": "ws-1"}


def test_enqueue_merges_second_arrival_losslessly():
    session = _session()
    server._enqueue_prompt(session, "first", "ws-1")
    server._enqueue_prompt(session, "second", "ws-2")
    assert session["queued_prompt"]["text"] == "first\n\nsecond"
    # Latest transport wins so the drain streams to the most recent client.
    assert session["queued_prompt"]["transport"] == "ws-2"


# ── _handle_busy_submit (policy) ───────────────────────────────────────────

def test_busy_interrupt_mode_interrupts_and_queues(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    calls = {"interrupt": 0}
    agent = types.SimpleNamespace(interrupt=lambda *a, **k: calls.__setitem__("interrupt", calls["interrupt"] + 1))
    session = _session(agent=agent)

    resp = server._handle_busy_submit("r1", "sid", session, "redirect", "ws-1")

    assert resp["result"]["status"] == "queued"
    assert calls["interrupt"] == 1
    assert session["queued_prompt"]["text"] == "redirect"


def test_busy_queue_mode_queues_without_interrupting(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    calls = {"interrupt": 0}
    agent = types.SimpleNamespace(interrupt=lambda *a, **k: calls.__setitem__("interrupt", calls["interrupt"] + 1))
    session = _session(agent=agent)

    resp = server._handle_busy_submit("r1", "sid", session, "later", "ws-1")

    assert resp["result"]["status"] == "queued"
    assert calls["interrupt"] == 0
    assert session["queued_prompt"]["text"] == "later"


def test_busy_interrupt_mode_ignores_completed_background_delegation(monkeypatch):
    """A terminal delegation must not suppress normal busy-turn interruption."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    calls = {"interrupt": 0}
    agent = types.SimpleNamespace(
        interrupt=lambda *a, **k: calls.__setitem__("interrupt", calls["interrupt"] + 1)
    )
    session = _session(agent=agent)

    with ad._records_lock:
        ad._records["deleg_completed"] = {
            "delegation_id": "deleg_completed",
            "status": "completed",
            "session_key": "session-key",
            "origin_ui_session_id": "sid",
        }

    try:
        resp = server._handle_busy_submit("r1", "sid", session, "continue", "ws-1")
    finally:
        with ad._records_lock:
            ad._records.clear()

    assert resp["result"]["status"] == "queued"
    assert calls["interrupt"] == 1
    assert session["queued_prompt"]["text"] == "continue"


def test_busy_interrupt_mode_ignores_foreign_background_delegation(monkeypatch):
    """Another tab's background work must not suppress this tab's interrupt."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    calls = {"interrupt": 0}
    agent = types.SimpleNamespace(
        interrupt=lambda *a, **k: calls.__setitem__("interrupt", calls["interrupt"] + 1)
    )
    session = _session(agent=agent)

    with ad._records_lock:
        ad._records["deleg_foreign"] = {
            "delegation_id": "deleg_foreign",
            "status": "running",
            "session_key": "foreign-key",
            "origin_ui_session_id": "foreign-sid",
        }

    try:
        resp = server._handle_busy_submit("r1", "sid", session, "interrupt me", "ws-1")
    finally:
        with ad._records_lock:
            ad._records.clear()

    assert resp["result"]["status"] == "queued"
    assert calls["interrupt"] == 1
    assert session["queued_prompt"]["text"] == "interrupt me"


def test_busy_steer_mode_injects_when_accepted(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    agent = types.SimpleNamespace(steer=lambda text: True, interrupt=lambda *a, **k: None)
    session = _session(agent=agent)

    resp = server._handle_busy_submit("r1", "sid", session, "nudge", "ws-1")

    assert resp["result"]["status"] == "steered"
    assert session.get("queued_prompt") is None


def test_busy_steer_mode_falls_back_to_queue_when_rejected(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    agent = types.SimpleNamespace(steer=lambda text: False, interrupt=lambda *a, **k: None)
    session = _session(agent=agent)

    resp = server._handle_busy_submit("r1", "sid", session, "nudge", "ws-1")

    assert resp["result"]["status"] == "queued"
    assert session["queued_prompt"]["text"] == "nudge"


# ── _drain_queued_prompt ───────────────────────────────────────────────────

def test_drain_fires_queued_prompt_and_claims_running(monkeypatch):
    fired = {}
    monkeypatch.setattr(
        server, "_run_prompt_submit",
        lambda rid, sid, session, text: fired.update(rid=rid, sid=sid, text=text),
    )
    session = _session(queued_prompt={"text": "go", "transport": "ws-9"})

    assert server._drain_queued_prompt("r1", "sid", session) is True
    assert fired == {"rid": "r1", "sid": "sid", "text": "go"}
    assert session["running"] is True
    assert session["queued_prompt"] is None
    assert session["transport"] == "ws-9"


def test_drain_noop_when_nothing_queued(monkeypatch):
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not fire")))
    session = _session()
    assert server._drain_queued_prompt("r1", "sid", session) is False
    assert session["running"] is False


def test_drain_noop_when_session_already_running(monkeypatch):
    """A fresh turn that claimed the session beats a stale queued entry —
    the drain leaves it for that turn's own tail."""
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not fire")))
    session = _session(running=True, queued_prompt={"text": "go", "transport": None})
    assert server._drain_queued_prompt("r1", "sid", session) is False
    assert session["queued_prompt"]["text"] == "go"


def test_drain_releases_running_on_dispatch_failure(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("dispatch failed")
    monkeypatch.setattr(server, "_run_prompt_submit", _boom)
    session = _session(queued_prompt={"text": "go", "transport": None})

    assert server._drain_queued_prompt("r1", "sid", session) is True
    # Failure must not leave the session wedged as running.
    assert session["running"] is False


def test_busy_interrupt_mode_preserves_real_background_batch_completion(
    monkeypatch, tmp_path
):
    """Foreground interruption must not cancel its detached async batch."""
    import json
    import queue
    import time

    import tools.delegate_tool as dt
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.process_registry import process_registry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")

    isolated_queue = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    ad._reset_for_tests()

    calls = {"interrupt": 0}

    class _Parent:
        def __init__(self):
            self._delegate_depth = 0
            self.session_id = "session-key"
            self._interrupt_requested = False
            self._active_children = []
            self._active_children_lock = None

        def interrupt(self, *_args, **_kwargs):
            calls["interrupt"] += 1
            self._interrupt_requested = True

    parent = _Parent()
    session = _session(agent=parent, running=True)

    release_children = threading.Event()
    all_children_started = threading.Event()
    started_lock = threading.Lock()
    started = {"count": 0}
    child_ids = iter(("child-1", "child-2", "child-3"))

    def _build_child(**_kwargs):
        return types.SimpleNamespace(
            _delegate_role="leaf",
            _subagent_id=next(child_ids),
        )

    def _blocking_child(task_index, goal, child=None, parent_agent=None, **_kwargs):
        with started_lock:
            started["count"] += 1
            if started["count"] == 3:
                all_children_started.set()

        release_children.wait(timeout=10)
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": f"done: {goal}",
            "api_calls": 1,
            "duration_seconds": 0.1,
            "model": "test-model",
            "exit_reason": "completed",
        }

    credentials = {
        "model": "test-model",
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "command": None,
        "args": None,
    }

    monkeypatch.setattr(dt, "_build_child_agent", _build_child)
    monkeypatch.setattr(dt, "_run_single_child", _blocking_child)
    monkeypatch.setattr(
        dt,
        "_resolve_delegation_credentials",
        lambda *_args, **_kwargs: credentials,
    )

    context_tokens = set_session_vars(
        source="tui",
        session_key="session-key",
        ui_session_id="sid",
    )

    response = None
    event = None
    try:
        dispatched = json.loads(
            dt.delegate_task(
                tasks=[
                    {"goal": "first"},
                    {"goal": "second"},
                    {"goal": "third"},
                ],
                background=True,
                parent_agent=parent,
            )
        )
        assert dispatched["status"] == "dispatched"
        assert all_children_started.wait(timeout=5)

        response = server._handle_busy_submit(
            "r1",
            "sid",
            session,
            "follow-up",
            "ws-1",
        )

        # The old detached-batch loop polls this parent flag every 0.5 seconds.
        time.sleep(0.7)
        release_children.set()
        event = isolated_queue.get(timeout=5)
    finally:
        release_children.set()
        clear_session_vars(context_tokens)
        ad._reset_for_tests()

    assert response["result"]["status"] == "queued"
    assert session["queued_prompt"]["text"] == "follow-up"
    assert calls["interrupt"] == 1

    assert event["type"] == "async_delegation"
    assert event["origin_ui_session_id"] == "sid"
    assert event["session_key"] == "session-key"
    assert [result["status"] for result in event["results"]] == [
        "completed",
        "completed",
        "completed",
    ]
    assert sorted(result["summary"] for result in event["results"]) == [
        "done: first",
        "done: second",
        "done: third",
    ]

    # Exercise the same positive-proof ownership gate used by the TUI's
    # post-turn delivery path, not just event production.
    isolated_queue.put(event)
    drained = process_registry.drain_notifications(
        session_key=session.get("session_key", ""),
        owns_event=lambda candidate: server._session_owns_notification_event(
            "sid", session, candidate
        ),
    )
    assert len(drained) == 1
    delivered_event, synthetic_prompt = drained[0]
    assert delivered_event is event
    assert synthetic_prompt
    assert isolated_queue.empty()
