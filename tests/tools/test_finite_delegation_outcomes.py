"""Finite dispatch returns sibling outcomes, not an orphaned async handle.

Only child construction/conversation is synthetic: dispatch, parallel executor,
child timeout/status conversion, interrupt propagation and cleanup are real.
"""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from agent.interrupt_control import InterruptControlMixin
from gateway import session_context as sc
from tools import async_delegation, delegate_tool as dt
from tools.delegate_tool_child_run import _attach_child
from tools.process_registry import process_registry


class _Parent(InterruptControlMixin, SimpleNamespace):
    pass


class _Child:
    def __init__(self, outcome="completed"):
        self.outcome = outcome
        self.tool_progress_callback = None
        self._credential_pool = None
        self._delegate_saved_tool_names = []
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self._subagent_id = None
        self.model = "test-model"
        self.session_prompt_tokens = self.session_completion_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.started = threading.Event()
        self.interrupted = threading.Event()
        self.unwinding = threading.Event()
        self.allow_finish = threading.Event()
        self.finished = threading.Event()
        self.closed = threading.Event()
        self.close_while_running = False
        self.worker = None

    def run_conversation(self, **_kwargs):
        self.worker = threading.current_thread()
        self.started.set()
        try:
            if self.outcome == "slow":
                assert self.interrupted.wait(5), "child never received stop"
                self.unwinding.set()
                assert self.allow_finish.wait(5), "child teardown was not released"
                return {"final_response": "partial", "completed": False,
                        "interrupted": True, "api_calls": 1, "messages": []}
            if self.outcome == "error":
                raise RuntimeError("synthetic child exception")
            if self.outcome == "failed":
                return {"final_response": "provider rejection", "completed": False,
                        "failed": True, "error": "synthetic provider rejection",
                        "api_calls": 1, "messages": []}
            return {"final_response": "sibling evidence", "completed": True,
                    "api_calls": 1, "messages": []}
        finally:
            self.finished.set()

    def hard_interrupt(self, _reason=None, **_kwargs):
        self.interrupted.set()

    def get_activity_summary(self):
        return {"api_call_count": 1}

    def close(self):
        self.close_while_running |= not self.finished.is_set()
        self.closed.set()


@pytest.fixture
def harness(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    tokens = [(v, v.set(sc._UNSET)) for v in sc._VAR_MAP.values()]
    tokens += [(v, v.set(sc._UNSET)) for v in
               (sc._SESSION_ASYNC_DELIVERY, sc._SESSION_HISTORY_DELIVERY)]
    async_delegation._reset_for_tests()
    parent = _Parent(session_id="finite-outcomes", _delegate_depth=0,
                     _current_task_id=None, _interrupt_requested=False,
                     _execution_thread_id=None, quiet_mode=True,
                     _active_children=[], _active_children_lock=threading.Lock())
    children = []

    def build_child(task_index, parent_agent, **_kwargs):
        child = children[task_index]
        _attach_child(parent_agent, child)
        return child

    monkeypatch.setattr(dt, "_build_child_agent", build_child)
    monkeypatch.setattr(dt, "_load_config", lambda: {})
    monkeypatch.setattr(dt, "_get_max_concurrent_children", lambda: 2)
    monkeypatch.setattr(dt, "_get_worktree_isolation", lambda: False)
    monkeypatch.setattr(dt, "_get_child_timeout", lambda: 4)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *_a, **_k: {
        "model": "test-model", "provider": None, "base_url": None,
        "api_key": None, "api_mode": None, "command": None, "args": None,
    })

    def dispatch(*outcomes):
        children.extend(_Child(outcome) for outcome in outcomes)
        return json.loads(dt.delegate_task(
            tasks=[{"goal": f"Exercise child outcome {i}"} for i in range(len(children))],
            background=True, parent_agent=parent,
        ))

    yield parent, children, dispatch
    # Release on assertion failure too. Never leave conversation workers behind.
    for child in children:
        child.interrupted.set()
        child.allow_finish.set()
    async_delegation._reset_for_tests()
    for child in children:
        if child.started.is_set():
            assert child.finished.wait(5)
            assert child.closed.wait(5)
            child.worker.join(5)
            assert not child.worker.is_alive()
            assert not child.close_while_running
    assert not parent._active_children
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    for var, token in reversed(tokens):
        var.reset(token)


def _joined(result, statuses):
    assert result.get("status") != "dispatched", result
    assert [entry["task_index"] for entry in result["results"]] == [0, 1]
    assert [entry["status"] for entry in result["results"]] == statuses
    assert result["results"][0]["summary"] == "sibling evidence"
    assert process_registry.completion_queue.empty()


@pytest.mark.parametrize("outcome", ["error", "failed"])
def test_finite_batch_returns_success_and_child_failure(harness, outcome):
    _parent, children, dispatch = harness
    result = dispatch("completed", outcome)
    _joined(result, ["completed", outcome])
    assert "synthetic" in result["results"][1]["error"]
    assert result["results"][1]["exit_reason"] == "error"
    assert all(child.closed.is_set() for child in children)


def test_finite_batch_returns_timeout_while_child_unwinds(harness, monkeypatch):
    _parent, children, dispatch = harness
    monkeypatch.setattr(dt, "_get_child_timeout", lambda: 0.5)
    result = dispatch("completed", "slow")
    _joined(result, ["completed", "timeout"])
    assert "timed out" in result["results"][1]["error"]
    assert result["results"][1]["exit_reason"] == "timeout"
    slow = children[1]
    assert slow.unwinding.wait(1)
    assert not slow.finished.is_set()
    assert not slow.closed.is_set(), "timeout must not close an unwinding child"
    slow.allow_finish.set()
    assert slow.closed.wait(2)


def test_finite_batch_returns_parent_interruption(harness):
    parent, children, dispatch = harness
    children.extend([_Child(), _Child("slow")])
    slow = children[1]
    stop_errors = []

    def stop_parent():
        try:
            assert slow.started.wait(3)
            assert children[0].closed.wait(3)
            parent.hard_interrupt("test parent stop")
        except Exception as exc:
            stop_errors.append(exc)
        finally:
            slow.allow_finish.set()

    stopper = threading.Thread(target=stop_parent)
    stopper.start()
    try:
        result = dispatch()
        _joined(result, ["completed", "interrupted"])
        assert parent._interrupt_requested is True
        assert slow.interrupted.is_set()
    finally:
        slow.allow_finish.set()
        stopper.join(5)
        assert not stopper.is_alive()
        assert not stop_errors


def test_finite_marker_overrides_api_history_continuation(harness):
    _parent, _children, dispatch = harness
    sc.set_session_vars(platform="api_server", chat_id="history-session",
                        session_key="history-session", session_id="history-session",
                        session_history_delivery="1", async_delivery=False)
    assert sc.session_history_delivery_supported()
    result = dispatch("completed", "error")
    _joined(result, ["completed", "error"])


@pytest.mark.parametrize("marker", [None, "0", "false"])
@pytest.mark.parametrize("api_history", [False, True])
def test_nonfinite_marker_preserves_background_dispatch(harness, monkeypatch, marker, api_history):
    _parent, children, dispatch = harness
    if marker is None:
        monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION")
    else:
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", marker)
    if api_history:
        sc.set_session_vars(platform="api_server", chat_id="history-session",
                            session_key="history-session", session_id="history-session",
                            session_history_delivery="1", async_delivery=False)
    result = dispatch("completed", "completed")
    assert result["status"] == "dispatched", result
    assert result["mode"] == "background"
    assert "results" not in result
    event = process_registry.completion_queue.get(timeout=5)
    assert event["type"] == "async_delegation"
    if api_history:
        assert event["origin_session_id"] == "history-session"
    assert all(child.closed.wait(2) for child in children)
