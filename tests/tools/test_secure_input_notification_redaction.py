"""Secure-input values must not reach the model through background-process
notifications.

Regression: ``env_secret_refs`` values were redacted on the foreground and poll/log/wait
routes but not in the completion / watch_match notifications that the CLI
drain, the TUI poller and the gateway inject as the next agent turn.
"""

import json
import queue
import sys
import time

import pytest
from tests.waiting import wait_until

from agent import secure_input_broker as broker
from tools.process_registry import (
    ProcessRegistry,
    ProcessSession,
    format_process_notification,
)

SECRET = "qv9Lp2WmVer3P18compl"
MARK = "[REDACTED SECURE INPUT]"


def setup_function():
    broker.clear_all()


def teardown_function():
    broker.clear_all()


def _register(value=SECRET):
    return broker.register_secret(value, purpose="unit_test", allowed_consumers=["terminal"])


def _session(sid, output, **kw):
    return ProcessSession(id=sid, command="./deploy.sh", task_id="t1", started_at=time.time(),
                          output_buffer=output, **kw)


def test_completion_event_and_drained_notification_omit_secret(monkeypatch):
    _register()
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    s = _session("proc_secret_done", f"auth value={SECRET}\n{SECRET}\n", notify_on_complete=True)
    s.exited, s.exit_code = True, 0
    registry._running[s.id] = s

    registry._move_to_finished(s)
    evt = registry.completion_queue.queue[0]
    assert SECRET not in evt["output"]
    assert MARK in evt["output"]

    drained = registry.drain_notifications()
    assert len(drained) == 1
    assert SECRET not in drained[0][1]
    assert MARK in drained[0][1]


def test_completion_tail_never_leaks_a_secret_cut_at_the_boundary(monkeypatch):
    secret = "BoundarySecretValue-0123456789abcdef"
    _register(secret)
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    # The 2000-char tail starts in the middle of the secret.
    output = "x" * 100 + secret + "y" * (2000 - len(secret) // 2)
    s = _session("proc_secret_edge", output, notify_on_complete=True)
    s.exited, s.exit_code = True, 0
    registry._running[s.id] = s

    registry._move_to_finished(s)
    tail = registry.completion_queue.queue[0]["output"]
    assert len(tail) <= 2000
    assert secret[len(secret) // 2:] not in tail


def test_watch_match_event_omits_secret():
    _register()
    registry = ProcessRegistry()
    s = _session("proc_secret_watch", "", watch_patterns=["MARKER"])
    registry._running[s.id] = s

    registry._check_watch_patterns(s, f"noise\nMARKER {SECRET}\n")
    evt = registry.completion_queue.get_nowait()
    assert evt["type"] == "watch_match"
    assert SECRET not in evt["output"]
    assert MARK in evt["output"]


@pytest.mark.parametrize("evt_type", ["completion", "watch_match"])
def test_formatter_redacts_raw_producer_output(evt_type):
    """The gateway watcher builds its completion event from the raw output
    buffer; the shared formatter is the last point before model input."""
    _register()
    text = format_process_notification({
        "type": evt_type, "session_id": "proc_raw", "command": "./deploy.sh",
        "exit_code": 0, "pattern": "auth", "output": f"auth value={SECRET}",
    })
    assert SECRET not in text
    assert MARK in text


class _RecordingQueue(queue.Queue):
    """Records every event put on the shared queue: a TUI notification poller
    leaked by another test may consume it before the drain below runs."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def put(self, item, block=True, timeout=None):
        self.seen.append(item)
        super().put(item, block, timeout)


def _isolated_queue(monkeypatch):
    from tools.process_registry import process_registry

    q = _RecordingQueue()
    monkeypatch.setattr(process_registry, "completion_queue", q)
    return process_registry, q


def _model_input(registry, q, evt_type, timeout=20.0):
    """Wait for an event of *evt_type*; return the text the CLI drain (or the
    TUI poller, same formatter) would inject as the next agent turn."""
    assert wait_until(lambda: any(e.get("type") == evt_type for e in q.seen), timeout=timeout)
    events = [e for e in q.seen if e.get("type") == evt_type]
    drained = [t for _evt, t in registry.drain_notifications()]
    formatted = [format_process_notification(dict(e)) for e in events]
    return events, "\n".join(drained + formatted)


def _py(code):
    return f"{sys.executable} -c '{code}'"


def test_terminal_background_notify_on_complete_does_not_leak_env_secret(tmp_path, monkeypatch):
    """End-to-end: terminal(background, notify_on_complete, env_secret_refs)."""
    from tools.terminal_tool import terminal_tool

    monkeypatch.setenv("TERMINAL_ENV", "local")
    registry, q = _isolated_queue(monkeypatch)
    ref = _register()["secret_ref"]
    started = json.loads(terminal_tool(
        command=_py('import os,time; time.sleep(1); v=os.environ["VTOK"]; print("auth value=" + v); print(v)'),
        workdir=str(tmp_path), background=True, notify_on_complete=True,
        env_secret_refs={"VTOK": ref}, task_id="sec-notify-e2e", timeout=30))
    assert started.get("notify_on_complete") is True, started

    events, text = _model_input(registry, q, "completion")
    assert all(SECRET not in e["output"] for e in events)
    assert "auth value=" in text
    assert SECRET not in text
    assert MARK in text


def test_terminal_background_watch_patterns_does_not_leak_env_secret(tmp_path, monkeypatch):
    from tools.terminal_tool import terminal_tool

    monkeypatch.setenv("TERMINAL_ENV", "local")
    registry, q = _isolated_queue(monkeypatch)
    ref = _register()["secret_ref"]
    started = json.loads(terminal_tool(
        command=_py('import os,sys,time; time.sleep(1); print("MARKER " + os.environ["WTOK"]); '
                    'sys.stdout.flush(); time.sleep(3)'),
        workdir=str(tmp_path), background=True, watch_patterns=["MARKER"],
        env_secret_refs={"WTOK": ref}, task_id="sec-watch-e2e", timeout=30))
    session_id = started.get("session_id")
    assert started.get("watch_patterns") == ["MARKER"], started
    try:
        events, text = _model_input(registry, q, "watch_match")
        assert all(SECRET not in e["output"] for e in events)
        assert "MARKER" in text
        assert SECRET not in text
        assert MARK in text
    finally:
        if session_id:
            registry.kill_process(session_id)
