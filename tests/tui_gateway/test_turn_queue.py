"""Tests for the agent-side TurnQueue and the session.queue.* gateway RPCs.

The queue lives on the agent (agent.turn_queue) so it drains even when no
client window is open — see agent/turn_queue.py. The gateway RPCs are thin
manipulators over it; _drain_queued_prompt is the single drain point.
"""

import io
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_queue import TurnQueue

_original_stdout = sys.stdout


@pytest.fixture(autouse=True)
def _restore_stdout():
    yield
    sys.stdout = _original_stdout


# ── TurnQueue unit behaviour ─────────────────────────────────────────


def test_enqueue_drain_fifo():
    q = TurnQueue()
    a = q.enqueue("first")
    b = q.enqueue("second")

    assert len(q) == 2
    assert bool(q)
    assert q.drain().id == a.id
    assert q.drain().id == b.id
    assert q.drain() is None
    assert not q


def test_remove_promote_update_clear():
    q = TurnQueue()
    a = q.enqueue("a")
    b = q.enqueue("b")
    c = q.enqueue("c")

    assert q.promote(c.id) is True
    assert [e.id for e in q.peek()] == [c.id, a.id, b.id]
    # Already at head → False (callers treat membership as "will go next").
    assert q.promote(c.id) is False
    assert q.promote("missing") is False

    assert q.update_text(a.id, "a2") is True
    assert q.update_text("missing", "x") is False

    assert q.remove(b.id) is True
    assert q.remove(b.id) is False

    assert q.clear() == 2
    assert q.drain() is None


def test_to_dict_shape_excludes_transport():
    q = TurnQueue()
    q.enqueue("hello", transport=object(), source="busy_submit")
    (d,) = q.to_list()

    assert d["text"] == "hello"
    assert d["source"] == "busy_submit"
    assert "transport" not in d
    assert isinstance(d["queued_at"], float)


def test_thread_safety_under_concurrent_enqueue_drain():
    q = TurnQueue()
    drained = []

    def producer():
        for i in range(200):
            q.enqueue(f"msg-{i}")

    def consumer():
        deadline = time.time() + 5
        while len(drained) < 200 and time.time() < deadline:
            entry = q.drain()
            if entry is not None:
                drained.append(entry.text)

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert len(drained) == 200
    assert len(set(drained)) == 200


# ── Gateway RPC integration ──────────────────────────────────────────


@pytest.fixture()
def server():
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test")),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        import importlib
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()


def _make_session(server, sid, running=True):
    agent = MagicMock(spec=["turn_queue", "steer", "interrupt"])
    agent.turn_queue = TurnQueue()
    session = {
        "session_key": sid,
        "agent": agent,
        "running": running,
        "history_lock": threading.Lock(),
        "last_active": 0,
    }
    server._sessions[sid] = session
    return session


def test_queue_add_lists_and_emits(server):
    sid = "s1"
    session = _make_session(server, sid, running=True)
    events = []
    with patch.object(server, "_emit", side_effect=lambda ev, s, p=None: events.append((ev, p))):
        resp = server.handle_request({
            "id": "r1",
            "method": "session.queue.add",
            "params": {"session_id": sid, "text": "queued msg"},
        })

    assert "error" not in resp
    assert resp["result"]["status"] == "queued"
    entry_id = resp["result"]["entry_id"]
    assert entry_id

    # Mirrors into the agent queue; busy session → NOT drained.
    assert [e.text for e in session["agent"].turn_queue.peek()] == ["queued msg"]
    assert ("queue.updated", {"entries": session["agent"].turn_queue.to_list()}) in events

    listed = server.handle_request({
        "id": "r2",
        "method": "session.queue.list",
        "params": {"session_id": sid},
    })
    assert [e["id"] for e in listed["result"]["entries"]] == [entry_id]


def test_queue_add_empty_text_rejected(server):
    _make_session(server, "s1")
    resp = server.handle_request({
        "id": "r1",
        "method": "session.queue.add",
        "params": {"session_id": "s1", "text": "   "},
    })
    assert resp["error"]["code"] == 4002


def test_queue_add_idle_session_drains_immediately(server):
    sid = "s1"
    session = _make_session(server, sid, running=False)
    drained = threading.Event()
    submitted = {}

    def fake_submit(rid, s, sess, text):
        submitted["text"] = text
        drained.set()

    with patch.object(server, "_run_prompt_submit", side_effect=fake_submit), \
         patch.object(server, "_emit"):
        resp = server.handle_request({
            "id": "r1",
            "method": "session.queue.add",
            "params": {"session_id": sid, "text": "run now"},
        })
        assert resp["result"]["status"] == "queued"
        assert drained.wait(timeout=5), "idle enqueue should drain immediately"

    assert submitted["text"] == "run now"
    assert session["agent"].turn_queue.drain() is None


def test_queue_remove_promote_update_clear_rpcs(server):
    sid = "s1"
    session = _make_session(server, sid, running=True)
    q = session["agent"].turn_queue
    a = q.enqueue("a")
    b = q.enqueue("b")

    with patch.object(server, "_emit"):
        promoted = server.handle_request({
            "id": "r1", "method": "session.queue.promote",
            "params": {"session_id": sid, "entry_id": b.id},
        })
        assert promoted["result"]["promoted"] is True
        assert [e.id for e in q.peek()] == [b.id, a.id]

        updated = server.handle_request({
            "id": "r2", "method": "session.queue.update",
            "params": {"session_id": sid, "entry_id": a.id, "text": "a-edited"},
        })
        assert updated["result"]["updated"] is True

        removed = server.handle_request({
            "id": "r3", "method": "session.queue.remove",
            "params": {"session_id": sid, "entry_id": b.id},
        })
        assert removed["result"]["removed"] is True

        cleared = server.handle_request({
            "id": "r4", "method": "session.queue.clear",
            "params": {"session_id": sid},
        })
        assert cleared["result"]["cleared"] == 1
        assert q.drain() is None


def test_queue_promote_interrupt_forwards_to_agent(server):
    sid = "s1"
    session = _make_session(server, sid, running=True)
    q = session["agent"].turn_queue
    q.enqueue("a")
    b = q.enqueue("b")

    with patch.object(server, "_emit"):
        resp = server.handle_request({
            "id": "r1", "method": "session.queue.promote",
            "params": {"session_id": sid, "entry_id": b.id, "interrupt": True},
        })

    assert resp["result"]["promoted"] is True
    session["agent"].interrupt.assert_called_once()
    # The queue survives the interrupt — that's the whole point of send-now.
    assert len(q) == 2


def test_queue_rpcs_unknown_session(server):
    for method in (
        "session.queue.list",
        "session.queue.clear",
    ):
        resp = server.handle_request({
            "id": "r", "method": method, "params": {"session_id": "nope"},
        })
        assert resp["error"]["code"] == 4001


def test_drain_queued_prompt_pops_agent_queue_and_emits_drained(server):
    sid = "s1"
    session = _make_session(server, sid, running=False)
    session["agent"].turn_queue.enqueue("next turn", source="queue")
    events = []
    with patch.object(server, "_run_prompt_submit"), \
         patch.object(server, "_emit", side_effect=lambda ev, s, p=None: events.append((ev, p))):
        assert server._drain_queued_prompt("rid", sid, session) is True

    kinds = [ev for ev, _ in events]
    assert "queue.updated" in kinds
    drained = dict(events)["queue.drained"]
    assert drained == {"text": "next turn", "source": "queue"}


def test_drain_queued_prompt_skips_when_running(server):
    sid = "s1"
    session = _make_session(server, sid, running=True)
    session["agent"].turn_queue.enqueue("waiting")

    with patch.object(server, "_run_prompt_submit") as submit:
        assert server._drain_queued_prompt("rid", sid, session) is False
        submit.assert_not_called()

    assert len(session["agent"].turn_queue) == 1


def test_busy_submit_enqueues_on_agent_queue(server):
    """_handle_busy_submit routes mid-turn prompts into agent.turn_queue with
    source=busy_submit so drain events tell clients the text was already echoed."""
    sid = "s1"
    session = _make_session(server, sid, running=True)

    with patch.object(server, "_emit"), \
         patch.object(server, "_load_busy_input_mode", return_value="queue"):
        resp = server._handle_busy_submit("rid", sid, session, "mid-turn msg", None)

    assert resp["result"]["status"] == "queued"
    (entry,) = session["agent"].turn_queue.peek()
    assert entry.text == "mid-turn msg"
    assert entry.source == "busy_submit"
