"""#94248 (native half): delegation timeout must drain transports FD-safely.

A timed-out child's daemon worker is typically parked inside an in-flight
OpenSSL read. The timeout thread must (1) never hard-close the child while the
worker future is running (deferred close, #90889), and (2) drain the child's
transports with socket ``shutdown()`` only — never ``client.close()`` — so the
blocked read settles with EOF/EPIPE and the worker can unwind (bounded drain).
Cross-thread FD release under a live SSL BIO is the #29507/#67142/#70773
native-corruption family.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from tools import delegate_tool


class _SslBlockedChild:
    """Worker blocks (modelling an in-flight SSL read) until drained."""

    def __init__(self) -> None:
        self.tool_progress_callback = None
        self._credential_pool = None
        self._delegate_saved_tool_names = []
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self._subagent_id = None
        self.model = "test-model"
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.read_settled = threading.Event()   # drain "EOF" signal
        self.unwound = threading.Event()
        self.closed = threading.Event()
        self.close_while_blocked = False
        self.drain_calls: list[str] = []
        self.drain_threads: list[str] = []

    def run_conversation(self, **_kwargs):
        # Models the worker blocked in ssl.read: only the FD-safe drain
        # (socket shutdown -> EOF) settles it; interrupts alone do not.
        assert self.read_settled.wait(timeout=10), "drain never settled the read"
        time.sleep(0.05)  # post-read unwind work (turn-finally flush)
        self.unwound.set()
        return {
            "final_response": "",
            "completed": False,
            "interrupted": True,
            "api_calls": 1,
            "messages": [],
        }

    def hard_interrupt(self, *_a, **_k):
        # Cooperative interrupt cannot unblock a thread inside OpenSSL read.
        pass

    def get_activity_summary(self):
        return {"api_call_count": 1}

    def _drain_transports_after_abandonment(self, *, reason: str) -> int:
        self.drain_calls.append(reason)
        self.drain_threads.append(threading.current_thread().name)
        self.read_settled.set()
        return 1

    def close(self):
        if not self.unwound.is_set():
            self.close_while_blocked = True
        self.closed.set()


def _run(child, monkeypatch, timeout=0.4):
    parent = SimpleNamespace(
        session_id="parent-94248-drain",
        _current_task_id=None,
        _active_children=[child],
        _active_children_lock=threading.Lock(),
    )
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: timeout)
    if hasattr(delegate_tool, "_get_worktree_isolation"):
        monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)
    return delegate_tool._run_single_child(
        task_index=0,
        goal="exercise timeout transport drain",
        child=child,
        parent_agent=parent,
    )


def test_timeout_drains_transports_so_blocked_worker_can_unwind(monkeypatch):
    child = _SslBlockedChild()

    result = _run(child, monkeypatch)

    assert result["status"] == "timeout"
    # The drain ran from the timeout path (immediate sweep) and settled the
    # blocked read; without it the worker would still be parked in ssl.read.
    assert any(r.startswith("delegate_timeout") for r in child.drain_calls), (
        "timeout path never drained the abandoned child's transports"
    )
    assert child.unwound.wait(timeout=5), (
        "worker never unwound — the drain did not settle its blocked read"
    )
    assert child.closed.wait(timeout=5)
    assert not child.close_while_blocked, (
        "child.close() ran while the worker was still inside its blocked read"
    )


def test_timeout_drain_failure_does_not_break_timeout_result(monkeypatch):
    child = _SslBlockedChild()

    def _raising_drain(*, reason: str) -> int:
        child.drain_calls.append(reason)
        raise RuntimeError("transport sweep exploded")

    child._drain_transports_after_abandonment = _raising_drain

    result = _run(child, monkeypatch)

    assert result["status"] == "timeout"
    assert child.drain_calls, "drain hook was never attempted"
    # Unblock the worker manually so the deferred close can run.
    child.read_settled.set()
    assert child.unwound.wait(timeout=5)
    assert child.closed.wait(timeout=5)


def test_timeout_without_drain_hook_still_defers_close(monkeypatch):
    """Children lacking the hook (test doubles, third-party agents) keep the
    plain deferred-close behavior."""
    child = _SslBlockedChild()
    # Shadow the hook with a non-callable: the timeout path must skip it.
    child.__dict__["_drain_transports_after_abandonment"] = None

    result = _run(child, monkeypatch)

    assert result["status"] == "timeout"
    assert not child.closed.is_set(), (
        "close must stay deferred while the worker future is running"
    )
    child.read_settled.set()
    assert child.unwound.wait(timeout=5)
    assert child.closed.wait(timeout=5)
    assert not child.close_while_blocked


class _FakeSocket:
    def __init__(self):
        self.shutdown_calls = 0
        self.closed = False

    def settimeout(self, _v):
        pass

    def shutdown(self, _how):
        self.shutdown_calls += 1

    def close(self):
        self.closed = True


def test_agent_drain_shuts_sockets_down_without_fd_release(monkeypatch):
    """AIAgent._drain_transports_after_abandonment must shutdown(), not close()."""
    import threading as _threading
    from unittest.mock import patch

    with patch("run_agent.AIAgent.__init__", return_value=None):
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)

    sock = _FakeSocket()
    close_calls = {"n": 0}

    class _FakeClient:
        def close(self):
            close_calls["n"] += 1

    agent.client = _FakeClient()
    agent._client_lock = _threading.RLock()
    agent._codex_session = None
    agent._active_request_abort = None

    import agent.agent_runtime_helpers as arh

    monkeypatch.setattr(arh, "_iter_pool_sockets", lambda _c: iter([sock]))

    drained = agent._drain_transports_after_abandonment(reason="delegate_timeout_test")

    assert drained == 1
    assert sock.shutdown_calls == 1
    assert not sock.closed, "drain must never release socket FDs"
    assert close_calls["n"] == 0, "drain must never call client.close()"
