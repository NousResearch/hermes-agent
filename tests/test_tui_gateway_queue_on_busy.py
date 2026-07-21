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
import time
import types

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


def test_identified_queue_submission_rejects_a_second_distinct_pending_entry(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    session = _session(running=True)
    server._enqueue_prompt(
        session,
        "first",
        "ws-1",
        submission_id="queued-entry-a",
    )

    response = server._handle_busy_submit(
        "r2",
        "sid",
        session,
        "second",
        "ws-2",
        submission_id="queued-entry-b",
    )

    assert response["error"]["code"] == 4009
    assert "session busy" in response["error"]["message"]
    assert session["queued_prompt"]["text"] == "first"
    assert session["queued_prompt"]["submission_id"] == "queued-entry-a"


def test_submission_receipt_is_shared_across_live_windows_for_one_stored_session():
    first_window = _session(profile_home="/profiles/eni")
    second_window = _session(profile_home="/profiles/eni")

    try:
        assert server._claim_prompt_submission(first_window, "queued-entry-a") is True
        assert server._claim_prompt_submission(second_window, "queued-entry-a") is False
    finally:
        server._release_prompt_submission(first_window, "queued-entry-a")

    assert server._claim_prompt_submission(second_window, "queued-entry-a") is True
    server._release_prompt_submission(second_window, "queued-entry-a")


# ── _handle_busy_submit (policy) ───────────────────────────────────────────

def test_busy_interrupt_mode_interrupts_and_queues(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    calls = {"interrupt": 0}
    agent = types.SimpleNamespace(interrupt=lambda *a, **k: calls.__setitem__("interrupt", calls["interrupt"] + 1))
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, "redirect", "ws-1")

    assert resp["result"]["status"] == "queued"
    deadline = time.monotonic() + 1
    while calls["interrupt"] != 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert calls["interrupt"] == 1
    assert session["queued_prompt"]["text"] == "redirect"


def test_busy_queue_mode_queues_without_interrupting(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    calls = {"interrupt": 0}
    agent = types.SimpleNamespace(interrupt=lambda *a, **k: calls.__setitem__("interrupt", calls["interrupt"] + 1))
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, "later", "ws-1")

    assert resp["result"]["status"] == "queued"
    assert calls["interrupt"] == 0
    assert session["queued_prompt"]["text"] == "later"


def test_busy_steer_mode_injects_when_accepted(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    agent = types.SimpleNamespace(steer=lambda text: True, interrupt=lambda *a, **k: None)
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, "nudge", "ws-1")

    assert resp["result"]["status"] == "steered"
    assert session.get("queued_prompt") is None


def test_busy_steer_mode_falls_back_to_queue_when_rejected(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    agent = types.SimpleNamespace(steer=lambda text: False, interrupt=lambda *a, **k: None)
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, "nudge", "ws-1")

    assert resp["result"]["status"] == "queued"
    assert session["queued_prompt"]["text"] == "nudge"


def test_busy_steer_mode_skips_steer_for_queue_origin_submissions(monkeypatch):
    """A queue-origin busy submission must use the durable FIFO queue path.

    Steering an already-accepted queued prompt into the live turn silently
    drops it: the in-memory steer ACK never carries the submission_id into a
    terminal completion, so the Desktop queue drainer cannot reconcile the
    matching local custody and the row lingers (BLOCKER 1, identified
    submissions). The steer branch is for direct live nudges only.
    """
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    steer_calls = []
    agent = types.SimpleNamespace(
        steer=lambda text: steer_calls.append(text) or True,
        interrupt=lambda *a, **k: None,
    )
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit(
        "r1",
        "sid",
        session,
        "queued-and-accepted",
        "ws-1",
        submission_id="queue-entry-77",
        from_queue=True,
    )

    assert steer_calls == [], "steer() must not be called for queue-origin submits"
    assert resp["result"]["status"] == "queued"
    assert session["queued_prompt"]["text"] == "queued-and-accepted"
    assert session["queued_prompt"]["submission_id"] == "queue-entry-77"


def test_busy_interrupt_mode_keeps_queue_origin_in_durable_fifo(monkeypatch):
    """Interrupt mode still accepts the queue-origin submission into the FIFO
    queue (the only durable path), and routes the submission_id through."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    agent = types.SimpleNamespace(interrupt=lambda *a, **k: None)
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit(
        "r1",
        "sid",
        session,
        "next-up",
        "ws-1",
        submission_id="queue-entry-88",
        from_queue=True,
    )

    assert resp["result"]["status"] == "queued"
    assert session["queued_prompt"]["text"] == "next-up"
    assert session["queued_prompt"]["submission_id"] == "queue-entry-88"


def test_busy_interrupt_does_not_hold_history_lock_or_delay_queue(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    interrupt_started = threading.Event()
    release_interrupt = threading.Event()

    def blocking_interrupt():
        interrupt_started.set()
        release_interrupt.wait(timeout=2)

    session = _session(
        agent=types.SimpleNamespace(interrupt=blocking_interrupt),
        running=True,
    )

    started = time.monotonic()
    resp = server._handle_busy_submit("r1", "sid", session, "keep this", "ws-1")

    assert resp["result"]["status"] == "queued"
    assert time.monotonic() - started < 0.25
    assert session["queued_prompt"]["text"] == "keep this"
    assert interrupt_started.wait(timeout=1)
    assert session["history_lock"].acquire(timeout=0.25)
    session["history_lock"].release()
    release_interrupt.set()


def test_busy_helper_retries_when_turn_finished(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    session = _session(running=False)

    assert server._handle_busy_submit("r1", "sid", session, "run now", "ws-1") is None
    assert session.get("queued_prompt") is None


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


def test_compute_worker_drain_preserves_queued_submission_identity(monkeypatch):
    fired = {}
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: True)

    def submit(rid, sid, _session, text, submission_id=""):
        fired.update(
            rid=rid,
            sid=sid,
            text=text,
            submission_id=submission_id,
        )
        return {"result": {"status": "streaming"}}

    monkeypatch.setattr(server, "_submit_prompt_to_compute_host", submit)
    session = _session(
        queued_prompt={
            "text": "go next",
            "transport": None,
            "submission_id": "queued-entry-123",
        }
    )

    assert server._drain_queued_prompt("r1", "sid", session) is True
    assert fired == {
        "rid": "r1",
        "sid": "sid",
        "text": "go next",
        "submission_id": "queued-entry-123",
    }


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
    retries = []
    monkeypatch.setattr(
        server,
        "_schedule_queued_prompt_retry",
        lambda rid, sid, session: retries.append((rid, sid, session["queued_prompt"]["text"])),
    )
    session = _session(
        queued_prompt={
            "text": "go",
            "transport": None,
            "submission_id": "queued-entry-retry",
        }
    )

    assert server._drain_queued_prompt("r1", "sid", session) is True
    # Failure must not leave the session wedged as running.
    assert session["running"] is False
    assert session["queued_prompt"]["text"] == "go"
    assert session["queued_prompt"]["submission_id"] == "queued-entry-retry"
    assert retries == [("r1", "sid", "go")]


def test_live_snapshots_include_stable_submission_identity():
    session = _session()
    server._start_inflight_turn(session, "running", "entry-running")
    server._enqueue_prompt(
        session,
        "next",
        "ws",
        submission_id="entry-next",
    )

    assert server._inflight_snapshot(session)["submission_id"] == "entry-running"
    assert server._queued_prompt_snapshot(session)["submission_id"] == "entry-next"
