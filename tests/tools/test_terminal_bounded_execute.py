"""Foreground terminal execute must return when the inner wait loop wedges.

#94285: a hung ``_wait_for_process`` (Windows pipe/poll, blocked loop thread)
silently disabled every asyncio timer in the process. ``execute()`` now
bounds spawn+wait with ``run_bounded_sync`` so the wall-clock deadline
survives a wedged wait, and ``on_timeout`` kills the process tree.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from tools.environments.local import LocalEnvironment
import tools.environments.base as base_mod


def test_execute_returns_when_wait_loop_never_returns(monkeypatch):
    """A wedged inner wait cannot hold execute() past timeout + grace."""
    monkeypatch.setattr(base_mod, "_EXECUTE_WAIT_BOUND_GRACE_S", 0.05)

    env = LocalEnvironment()
    fake_proc = SimpleNamespace(pid=424242)
    monkeypatch.setattr(env, "_run_bash", lambda *a, **k: fake_proc)

    def _hang(*_a, **_k):
        time.sleep(30)
        return {"output": "late", "returncode": 0}

    monkeypatch.setattr(env, "_wait_for_process", _hang)
    killed: list = []
    monkeypatch.setattr(env, "_kill_process", lambda proc: killed.append(("kill", proc)))
    monkeypatch.setattr(
        "agent.deadline.kill_process_tree",
        lambda pid, **_k: killed.append(("tree", pid)),
    )
    monkeypatch.setattr(env, "_update_cwd", lambda _result: None)

    start = time.monotonic()
    result = env.execute("sleep 30", timeout=1)
    elapsed = time.monotonic() - start

    assert elapsed < 4.0, f"execute hung {elapsed:.1f}s past the 1s bound"
    assert result["returncode"] == 124
    assert "timed out" in result["output"].lower()
    assert ("kill", fake_proc) in killed
    assert ("tree", 424242) in killed


def test_execute_parent_interrupt_still_kills_wait_on_deadline_worker(monkeypatch):
    """/stop targets the tool-worker tid; the deadline worker must honor it."""
    from tools.interrupt import set_interrupt, is_interrupted

    env = LocalEnvironment()
    fake_proc = SimpleNamespace(pid=None, poll=lambda: None, stdout=None)
    monkeypatch.setattr(env, "_run_bash", lambda *a, **k: fake_proc)

    seen = {"parent": False}

    def _wait(_proc, timeout=120, *, bounded_capture=False, watch_interrupt_tid=None):
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            from tools.interrupt import is_thread_interrupted

            if is_interrupted() or is_thread_interrupted(watch_interrupt_tid):
                seen["parent"] = True
                return {"output": "[Command interrupted]", "returncode": 130}
            time.sleep(0.02)
        return {"output": "missed interrupt", "returncode": 0}

    monkeypatch.setattr(env, "_wait_for_process", _wait)
    monkeypatch.setattr(env, "_kill_process", lambda _proc: None)
    monkeypatch.setattr(env, "_update_cwd", lambda _result: None)

    parent_tid = __import__("threading").get_ident()

    def _interrupt_soon():
        time.sleep(0.05)
        set_interrupt(True, thread_id=parent_tid)

    import threading

    threading.Thread(target=_interrupt_soon, daemon=True).start()
    result = env.execute("sleep 30", timeout=5)
    set_interrupt(False, thread_id=parent_tid)

    assert seen["parent"] is True
    assert result["returncode"] == 130


def test_execute_worker_sees_caller_activity_callback(monkeypatch):
    """Heartbeats must fire on the deadline worker, not only the tool thread."""
    env = LocalEnvironment()
    fake_proc = SimpleNamespace(pid=None)
    monkeypatch.setattr(env, "_run_bash", lambda *a, **k: fake_proc)
    seen = {"cb": "unset"}

    def _wait(*_a, **_k):
        seen["cb"] = base_mod.get_activity_callback()
        return {"output": "ok", "returncode": 0}

    monkeypatch.setattr(env, "_wait_for_process", _wait)
    monkeypatch.setattr(env, "_update_cwd", lambda _r: None)

    def _cb(_msg):
        pass

    base_mod.set_activity_callback(_cb)
    try:
        result = env.execute("echo ok", timeout=5)
    finally:
        base_mod.set_activity_callback(None)

    assert result["returncode"] == 0
    assert seen["cb"] is _cb
