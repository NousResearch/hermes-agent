"""Regression tests: a user-approved command runs from a clean interrupt slate.

Bug (manual approvals, the default): a user approves a scanner-flagged command,
then hits Stop / sends a message.  `agent.interrupt()` sets the per-thread
interrupt bit on the execution thread *during* the blocking approval-wait; the
deny that follows is a no-op once the approval was granted, so the bit persists.
Nothing cleared it between approval-grant and `env.execute`, so
`_wait_for_process` SIGINT-killed the just-approved command on its first poll and
returned exit 130 + "[Command interrupted]" while still carrying the
"...approved by the user." note (the 3-part signature).

Fix: clear the current thread's interrupt bit once before the approved command
spawns its child (terminal foreground; execute_code local + remote), and enrich
the note on a genuine post-start interrupt instead of implying success.

Invariant preserved: a genuine interrupt arriving AFTER execution starts (or
during a retry backoff) must still SIGINT the command (exit 130); non-approved
commands keep current interrupt behavior.
"""
import json
import threading
import time

import pytest

from tools import terminal_tool as tt
from tools.interrupt import (
    set_interrupt,
    is_interrupted,
    clear_current_thread_interrupt,
    _interrupted_threads,
    _lock,
)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "logs").mkdir(exist_ok=True)
    # Clean interrupt slate before and after every test so a stale tid left in
    # the module-global set can't leak across tests in the same worker.
    with _lock:
        _interrupted_threads.clear()
    yield
    with _lock:
        _interrupted_threads.clear()


def _wait_for_sentinel(sentinel, timeout=10.0):
    """Block until the running command created its sentinel (proving the
    clean-slate clear already ran and the command is in its poll loop)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sentinel.exists():
            return True
        time.sleep(0.02)
    return sentinel.exists()


# ---------------------------------------------------------------------------
# terminal_tool
# ---------------------------------------------------------------------------

def test_approved_command_clears_stale_interrupt_bit():
    """force=True marks the run user-approved -> the stale bit is cleared and
    the command completes (exit 0), not killed with 130."""
    set_interrupt(True)  # simulate a bit that landed during the approval-wait
    assert is_interrupted()

    result = json.loads(tt.terminal_tool(command="sleep 0.5; echo DONE", force=True))

    assert result["exit_code"] == 0, result
    assert "DONE" in result["output"]
    assert "[Command interrupted]" not in result["output"]


def test_non_approved_command_still_interrupts_on_stale_bit(monkeypatch):
    """A command that is auto-approved but NOT user-approved keeps the current
    interrupt behavior: a pre-existing bit still kills it (DO-NOT-BREAK)."""
    monkeypatch.setattr(tt, "_check_all_guards", lambda *a, **k: {"approved": True})
    set_interrupt(True)

    result = json.loads(tt.terminal_tool(command="sleep 0.5; echo DONE"))

    assert result["exit_code"] == 130, result
    assert "[Command interrupted]" in result["output"]


def test_approved_command_genuine_interrupt_after_start_still_kills(tmp_path):
    """The clean-slate clear must NOT make approved commands un-interruptible:
    an interrupt that arrives after execution starts still SIGINTs (130)."""
    sentinel = tmp_path / "cmd_started_c"
    holder = {}

    def worker():
        holder["result"] = tt.terminal_tool(
            command=f"touch {sentinel}; sleep 5; echo DONE", force=True
        )

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    # Barrier: the command is genuinely running (so the clear already ran) before
    # we fire the interrupt -- no fixed-sleep timing guess.
    assert _wait_for_sentinel(sentinel), "command did not start"
    set_interrupt(True, thread_id=t.ident)  # genuine interrupt, AFTER start
    t.join(timeout=15)
    assert not t.is_alive(), "worker did not exit after a genuine interrupt"

    result = json.loads(holder["result"])
    assert result["exit_code"] == 130, result
    assert "[Command interrupted]" in result["output"]
    set_interrupt(False, thread_id=t.ident)


def test_approved_note_enriched_not_misleading_on_interrupt(monkeypatch, tmp_path):
    """On a genuine post-start interrupt of an approved command, the note must
    read '...approved by the user, then interrupted.' — the bare
    '...approved by the user.' must never co-occur with exit 130."""
    monkeypatch.setattr(
        tt,
        "_check_all_guards",
        lambda *a, **k: {"approved": True, "user_approved": True, "description": "rm -rf x"},
    )
    sentinel = tmp_path / "cmd_started_d"
    holder = {}

    def worker():
        holder["result"] = tt.terminal_tool(command=f"touch {sentinel}; sleep 5; echo DONE")

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    assert _wait_for_sentinel(sentinel), "command did not start"
    set_interrupt(True, thread_id=t.ident)
    t.join(timeout=15)
    assert not t.is_alive()

    result = json.loads(holder["result"])
    assert result["exit_code"] == 130, result
    note = result.get("approval", "")
    assert note.endswith("then interrupted."), note
    assert "approved by the user, then interrupted." in note
    assert "approved by the user." not in note  # success-implying string is gone
    set_interrupt(False, thread_id=t.ident)


def test_natural_exit_130_not_mislabeled_as_interrupt(monkeypatch):
    """A command that legitimately exits 130 on its own (no interrupt) must NOT
    get its approval note rewritten to '...then interrupted.'."""
    monkeypatch.setattr(
        tt,
        "_check_all_guards",
        lambda *a, **k: {"approved": True, "user_approved": True, "description": "x"},
    )
    # Clean slate: no interrupt at all.
    result = json.loads(tt.terminal_tool(command="bash -c 'exit 130'"))

    assert result["exit_code"] == 130, result
    note = result.get("approval", "")
    assert note == "Command required approval (x) and was approved by the user.", note
    assert "then interrupted" not in note
    assert "[Command interrupted]" not in result["output"]


def test_retry_backoff_does_not_clear_genuine_interrupt(monkeypatch):
    """A genuine interrupt that lands during the retry backoff must survive
    (the clear runs ONCE before the loop, never re-clearing on retries)."""
    from tools.environments.local import LocalEnvironment

    calls = {"n": 0, "interrupted_at_retry": None}

    def fake_execute(self, command, **kw):
        if "sleep 1" not in command:  # ignore any incidental execute calls
            return {"output": "", "returncode": 0}
        calls["n"] += 1
        if calls["n"] == 1:
            set_interrupt(True)  # Stop lands during the first attempt / backoff
            raise RuntimeError("transient backend error")
        # Second attempt: the bit set during the backoff must NOT be re-cleared.
        calls["interrupted_at_retry"] = is_interrupted()
        return {"output": "partial\n[Command interrupted]", "returncode": 130}

    monkeypatch.setattr(LocalEnvironment, "execute", fake_execute)
    monkeypatch.setattr("tools.terminal_tool.time.sleep", lambda *a, **k: None)
    set_interrupt(False)

    result = json.loads(tt.terminal_tool(command="sleep 1", force=True, task_id="retry-test"))

    assert calls["n"] == 2, calls
    assert calls["interrupted_at_retry"] is True, "retry must NOT re-clear a genuine interrupt"
    assert result["exit_code"] == 130, result


# ---------------------------------------------------------------------------
# execute_code (same root cause, its own approval-wait + spawn/poll loop)
# ---------------------------------------------------------------------------

def test_execute_code_approved_clears_stale_interrupt_bit(monkeypatch):
    """An approved execute_code script (local path) runs from a clean slate."""
    from tools.code_execution_tool import execute_code

    monkeypatch.setattr(
        "tools.approval.check_execute_code_guard",
        lambda *a, **k: {"approved": True, "user_approved": True},
    )
    set_interrupt(True)
    assert is_interrupted()

    result = json.loads(execute_code(
        code='import time; time.sleep(0.5); print("CODE_DONE")',
        task_id="test-clean-slate",
    ))

    assert result["status"] == "success", result
    assert "CODE_DONE" in result["output"]
    assert "execution interrupted" not in result["output"]


def test_execute_code_non_approved_still_interrupts_on_stale_bit(monkeypatch):
    """Non-user-approved execute_code keeps current interrupt behavior."""
    from tools.code_execution_tool import execute_code

    monkeypatch.setattr(
        "tools.approval.check_execute_code_guard",
        lambda *a, **k: {"approved": True},  # approved, but NOT user_approved
    )
    set_interrupt(True)

    result = json.loads(execute_code(
        code='import time; time.sleep(0.5); print("CODE_DONE")',
        task_id="test-clean-slate-2",
    ))

    # Killed on the first poll before the script can print.
    assert "CODE_DONE" not in result["output"], result


def test_execute_code_remote_clears_stale_bit(monkeypatch):
    """The clear sits above the local/remote split, so an approved remote (ssh)
    script also dispatches from a clean slate."""
    from tools import code_execution_tool as cet

    monkeypatch.setattr(
        "tools.approval.check_execute_code_guard",
        lambda *a, **k: {"approved": True, "user_approved": True},
    )
    monkeypatch.setattr("tools.terminal_tool._get_env_config", lambda *a, **k: {"env_type": "ssh"})

    captured = {}

    def fake_remote(code, task_id, enabled_tools):
        captured["interrupted"] = is_interrupted()
        return json.dumps({"status": "success", "output": ""})

    monkeypatch.setattr(cet, "_execute_remote", fake_remote)
    set_interrupt(True)  # stale bit present before dispatch

    cet.execute_code(code="print(1)", task_id="remote-clean-slate")

    assert captured["interrupted"] is False, "clear must run before the remote dispatch"
