"""Explicit detach (``display.background_key`` / ``/detach``): the user sends the running
foreground terminal command to the background with no message. Reuses the yield path of
``redirect()`` but must not inject a steer, and must tell the model why it was moved."""
import json
import threading
import time

import psutil
import pytest

from agent.interrupt_control import InterruptControlMixin
from tools import interrupt as interrupt_mod
from tools.process_registry import process_registry
from tools.terminal_tool import terminal_tool

pytestmark = pytest.mark.platforms("linux")


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


def _run_in_worker(agent, command, timeout):
    res = {}

    def worker():
        with agent._tool_worker_threads_lock:
            agent._tool_worker_threads.add(threading.current_thread().ident)
        res["result"] = json.loads(terminal_tool(command, task_id="detach-test", timeout=timeout))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t, res


def test_detach_foreground_without_tools_is_a_noop():
    agent = _Agent()
    agent._executing_tools = False
    assert agent.detach_foreground() == 0


def test_detach_skips_workers_not_in_a_yieldable_wait():
    """A tool worker that is not blocked in a local foreground terminal wait (a non-terminal tool,
    or a remote backend where no yield_handler is armed) must not be signalled: nothing would
    consume the bit and the UI must be able to say 'nothing to detach'."""
    agent = _Agent()
    tid = 424242
    agent._tool_worker_threads.add(tid)
    assert agent.detach_foreground() == 0
    assert not interrupt_mod.is_thread_yield_requested(tid)
    with interrupt_mod.yield_armed(tid):
        assert agent.detach_foreground() == 1
    assert interrupt_mod.consume_yield(tid)
    assert interrupt_mod.pop_yield_reason(tid) == "user_detach"
    assert not interrupt_mod.is_yield_armed(tid)


def test_remote_backend_is_never_armed(monkeypatch):
    """Only the local backend builds a yield handler; others run the wait unarmed."""
    from tools.terminal_tool_background import yield_to_background_handler
    for env_type in ("docker", "ssh", "modal", "singularity", "daytona"):
        assert yield_to_background_handler(
            command="sleep 1", env_type=env_type, cwd="/", effective_task_id="t", task_id="t",
            session_key="") is None, env_type


def test_request_yield_reason_roundtrip_and_clear():
    tid = 987654321
    interrupt_mod.request_yield(tid, reason="user_detach")
    assert interrupt_mod.is_thread_yield_requested(tid)
    interrupt_mod.set_interrupt(False, tid)  # clearing drops yield bit AND reason
    assert not interrupt_mod.is_thread_yield_requested(tid)
    assert interrupt_mod.pop_yield_reason(tid) is None
    interrupt_mod.request_yield(tid, reason="user_detach")
    interrupt_mod.request_yield(tid)  # a plain redirect yield overrides a stale detach reason
    assert interrupt_mod.pop_yield_reason(tid) is None
    interrupt_mod.consume_yield(tid)


def _wait_for(path, deadline_s):
    """Block until *path* exists: the command signals it is really running. A fixed sleep races
    shell cold-start (Git Bash on a Windows runner can take seconds), and detaching before the
    shell printed anything makes the output assertion flaky."""
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


@pytest.mark.live_system_guard_bypass
def test_detach_moves_command_to_background_and_outlives_foreground_timeout(tmp_path, monkeypatch):
    """Detach mid-command: no steer, detach note, process alive AFTER the foreground timeout
    would have fired (the promoted process must not inherit it)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = _Agent()
    ready = tmp_path / "ready"
    fg_timeout = 10
    t0 = time.monotonic()
    t, res = _run_in_worker(
        agent, f"echo started; touch '{ready.as_posix()}'; sleep 14; echo done", timeout=fg_timeout)
    assert _wait_for(ready, fg_timeout - 1), "command never started within the foreground timeout"
    assert agent.detach_foreground() == 1
    assert agent._pending_steer is None  # detach carries no user message

    t.join(timeout=10)
    assert not t.is_alive(), "terminal tool still blocked after detach_foreground()"
    r = res["result"]
    try:
        assert r["status"] == "yielded_to_background"
        assert r.get("detached_by_user") is True
        assert "on purpose" in r["note"] and "Do NOT poll" in r["note"]
        assert "started" in r["output"]
        assert not interrupt_mod.is_thread_yield_requested(t.ident)
        assert interrupt_mod.pop_yield_reason(t.ident) is None  # consumed by the tool
        # Past the foreground timeout (measured from launch): the promoted process must survive it.
        time.sleep(max(0.0, fg_timeout + 0.5 - (time.monotonic() - t0)))
        assert psutil.pid_exists(r["pid"]), "promoted process was killed by the foreground timeout"
        assert process_registry.poll(r["session_id"])["status"] == "running"
        waited = process_registry.wait(r["session_id"], timeout=30)
        assert "done" in json.dumps(waited)
    finally:
        if process_registry.poll(r["session_id"])["status"] == "running":
            process_registry.kill_process(r["session_id"])
        # The completion queue is process-global: drain our own event so it can't leak into
        # later tests that read the next completion.
        _drain_completion(r["session_id"])


def _drain_completion(session_id, timeout=5.0):
    import queue as _queue
    deadline = time.monotonic() + timeout
    kept = []
    while time.monotonic() < deadline:
        try:
            evt = process_registry.completion_queue.get(timeout=0.2)
        except _queue.Empty:
            continue
        if evt.get("session_id") == session_id:
            break
        kept.append(evt)
    for evt in kept:
        process_registry.completion_queue.put(evt)
