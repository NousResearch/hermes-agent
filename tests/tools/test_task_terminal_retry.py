"""A revoked task must not retry a foreground backend execution."""
import json
from types import SimpleNamespace

from tools import terminal_tool as terminal
from tools.approval_task import bind_task, from_composer, revoke_session_task


def test_foreground_retry_stops_when_task_revoked_during_backoff(monkeypatch):
    lease = from_composer("s", {"kind": "desktop_composer", "raw_text": "run command"})
    session = {"_approval_task_lease": lease}
    attempts, sleeps = [], []

    def execute(*args, **kwargs):
        attempts.append(args)
        raise RuntimeError("transient backend failure")

    def backoff(seconds):
        sleeps.append(seconds)
        revoke_session_task(session)  # Queue-new-input need not signal terminal interrupt.

    monkeypatch.setattr(terminal.time, "sleep", backoff)
    monkeypatch.setattr(terminal, "_resolve_command_cwd", lambda **kw: ".")
    monkeypatch.setattr(terminal, "_yield_kwargs", lambda *a, **kw: {})
    plan = SimpleNamespace(env_type="local", effective_task_id="s", effective_timeout=30, cwd=".")
    with bind_task(lease):
        result = json.loads(terminal._run_foreground(
            "echo harmless", SimpleNamespace(execute=execute), plan,
            task_id="s", session_id=None, session_key="s", workdir=None,
            approval_note=None, clear_interrupt=False))
    assert len(attempts) == 1
    assert sleeps == [2]
    assert result["status"] == "blocked"
    assert "Task ended" in result["error"]
