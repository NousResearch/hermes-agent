"""A user message sent while a foreground terminal command runs must not wait for the command.

``AIAgent.redirect()`` during tool execution degrades to ``steer()``, whose delivery rides
the tool result — so a long foreground command (a 5-minute ``sleep`` poller, a build) parked
the user's message until it exited. Now redirect() also asks the tool workers to YIELD: the
local terminal backend hands the still-running process to the process registry as a
notify-on-complete background session and returns immediately, without killing it.
"""
import json
import os
import threading
import time

import pytest

from agent.interrupt_control import InterruptControlMixin
from tools import interrupt as interrupt_mod
from tools.process_registry import process_registry
from tools.terminal_tool import terminal_tool

pytestmark = pytest.mark.linux_only


class _Agent(InterruptControlMixin):
    _executing_tools = True
    _interrupt_requested = False
    _pending_steer = None
    _pending_redirect = None
    api_mode = "chat_completions"

    def __init__(self):
        self._pending_steer_lock = threading.Lock()
        self._pending_redirect_lock = threading.Lock()
        self._tool_worker_threads = set()
        self._tool_worker_threads_lock = threading.Lock()
        self._execution_thread_id = None


def test_redirect_mid_command_yields_it_to_background_without_killing_it(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = _Agent()
    res = {}

    def worker():
        with agent._tool_worker_threads_lock:
            agent._tool_worker_threads.add(threading.current_thread().ident)
        t0 = time.monotonic()
        res["result"] = json.loads(terminal_tool("echo started; sleep 60; echo done", task_id="yield-test", timeout=90))
        res["elapsed"] = time.monotonic() - t0

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(1.5)
    assert agent.redirect("also, check the session ids") is True
    assert agent._pending_steer == "also, check the session ids"  # still delivered as a steer

    t.join(timeout=15)
    assert not t.is_alive(), "terminal tool still blocked on the command after redirect()"
    r = res["result"]
    try:
        assert r["status"] == "yielded_to_background"
        assert r["exit_code"] is None and "started" in r["output"]
        assert r["notify_on_complete"] is True
        # The process is alive and tracked: poll/wait/kill and the completion notification work.
        assert os.path.exists(f"/proc/{r['pid']}")
        assert process_registry.poll(r["session_id"])["status"] == "running"
        assert not interrupt_mod.is_thread_yield_requested(t.ident)
    finally:
        killed = process_registry.kill_process(r["session_id"])
    assert killed["status"] == "killed"
    evt = process_registry.completion_queue.get(timeout=5)
    assert evt["session_id"] == r["session_id"]


def test_yield_request_without_steer_leaves_foreground_wait_alone():
    """No yield handler (internal env.execute consumers) -> the request is ignored and the
    command runs to completion; an unrelated stale yield bit must not leak into it either."""
    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment()
    try:
        interrupt_mod.request_yield(threading.current_thread().ident)
        result = env.execute("echo alpha; sleep 0.3; echo omega", timeout=10)
        assert result["returncode"] == 0
        assert "omega" in result["output"]
    finally:
        interrupt_mod.consume_yield(threading.current_thread().ident)
        env.cleanup()
