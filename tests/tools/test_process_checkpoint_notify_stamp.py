"""A background job's notify intent must reach the crash checkpoint before the gateway can die.

The spawn-time checkpoint predates every async stamp (notify flag, watcher interval, routing),
and nothing rewrote it until the next state change: an unclean gateway death in that window
adopted the still-running job with no watcher and no notify intent, and its completion was
never detected — the agent's finished run stayed silently invisible (#157309).
"""

import json
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace

from tools.process_registry import ProcessRegistry, _checkpoint_path
from tools.terminal_tool_background import spawn_background_process

_GATEWAY_SESSION = "agent:main:telegram:dm:123"


def _install_runner(monkeypatch, runner):
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)


def _sleep_proc() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
    )


def _wait_exit(proc, timeout=5.0) -> None:
    deadline = time.time() + timeout
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.02)


def test_async_notify_stamp_is_written_to_the_checkpoint(tmp_path, monkeypatch):
    """After a gateway background launch with notify=true, the on-disk checkpoint entry must
    already carry what a crashed gateway needs to re-arm: the notify flag, the completion
    watcher interval, and the chat routing — not only the in-memory session."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "123")
    _install_runner(monkeypatch, SimpleNamespace(arm_process_watcher=lambda watcher: True))

    result = json.loads(spawn_background_process(
        command=f"{shlex.quote(sys.executable)} -c \"import time; time.sleep(30)\"",
        env=None, env_type="local",
        effective_task_id="bg-stamp", task_id="bg-stamp", session_key=_GATEWAY_SESSION,
        workdir=None, cwd=str(tmp_path), effective_pty=False,
        notify_on_complete=True, watch_patterns=None, approval_note=None,
        pty_disabled_reason=None,
    ))
    assert "session_id" in result, result
    session_id = result["session_id"]
    try:
        entries = json.loads(_checkpoint_path().read_text(encoding="utf-8-sig"))
        entry = next(e for e in entries if e.get("session_id") == session_id)
        assert entry["notify_on_complete"] is True
        assert entry["watcher_interval"] > 0, "completion watcher must be armed in the persisted entry"
        assert entry["watcher_platform"] == "telegram" and entry["watcher_chat_id"] == "123"
    finally:
        from tools.process_registry import process_registry
        process_registry.kill_process(session_id)


def test_recovery_requeues_the_completion_watcher_from_a_stamped_entry(tmp_path, monkeypatch):
    """The other half of the contract: given a checkpoint entry stamped the way the fix writes
    it, startup recovery adopts the live process AND re-enqueues its notify watcher, so the
    new gateway detects the completion the crashed one would have missed."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    proc = _sleep_proc()
    try:
        start = ProcessRegistry._safe_host_start_time(proc.pid)
        assert start is not None
        checkpoint = tmp_path / "processes.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_crashed",
            "command": "sleep-for-orphan",
            "pid": proc.pid,
            "pid_scope": "host",
            "host_start_time": start,
            "notify_on_complete": True,
            "watcher_interval": 5,
            "watcher_platform": "telegram",
            "watcher_chat_id": "123",
            "parent_session_id": "local-parent-session",
            "session_key": _GATEWAY_SESSION,
        }]), encoding="utf-8")

        registry = ProcessRegistry()
        assert registry.recover_from_checkpoint() == 1

        assert "proc_crashed" in registry._running
        armed = [w for w in registry.pending_watchers if w["session_id"] == "proc_crashed"]
        assert armed, "a notify-only job must come back with its completion watcher re-armed"
        assert armed[0]["notify_on_complete"] is True
        assert armed[0]["parent_session_id"] == "local-parent-session"
        assert armed[0]["chat_id"] == "123"
    finally:
        proc.terminate()
        _wait_exit(proc)
