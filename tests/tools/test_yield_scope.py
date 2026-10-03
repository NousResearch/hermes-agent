"""A yield armed BEFORE a command started must not move that command to the background.

``redirect()`` asks every live tool worker to yield. The tool-worker tid is pooled and
reused, so a bit no consumer took — a tool with no yield handler, or a hand-off that lost
the race with the tool call it was armed for — outlived its tool call and landed on a
LATER, unrelated command: that command was handed to the background at its first poll and
the result told the model a user message had arrived, which is false, so the command
looked lost and got re-issued (a fresh process per re-issue). A yield only ever applies to
a command that was already running when the message arrived.
"""
import json
import threading
import time

from tools import interrupt as interrupt_mod
from tools.process_registry import process_registry
from tools.terminal_tool import terminal_tool


def test_yield_armed_before_the_command_started_does_not_background_it(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tid = threading.current_thread().ident
    interrupt_mod.request_yield(tid)  # nothing is running on this thread to consume it
    try:
        result = json.loads(
            terminal_tool("echo alpha; sleep 0.3; echo omega", task_id="stale-yield", timeout=20))
    finally:
        interrupt_mod.consume_yield(tid)

    # Bites when the guard is removed: the stale bit backgrounds the command at its first
    # poll, so the result carries status=yielded_to_background and a null exit code instead
    # of the completed run.
    assert result.get("status") != "yielded_to_background", result
    assert result["exit_code"] == 0, result
    assert "omega" in result["output"], result


def test_yield_armed_while_the_command_runs_still_backgrounds_it(tmp_path, monkeypatch):
    """The guard must not swallow a genuine yield: a message arriving mid-command still
    hands the running command over instead of parking it behind the message."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    res = {}

    def worker():
        res["result"] = json.loads(
            terminal_tool("echo started; sleep 30; echo done", task_id="live-yield", timeout=60))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        time.sleep(1.5)  # the command is provably running: it prints, then sleeps 30s
        interrupt_mod.request_yield(t.ident)
        t.join(timeout=20)
        assert not t.is_alive(), "foreground command was parked instead of yielded"
        r = res["result"]
        assert r["status"] == "yielded_to_background", r
        assert r["exit_code"] is None and "started" in r["output"], r
        assert process_registry.poll(r["session_id"])["status"] == "running"
    finally:
        for s in process_registry.list_sessions():
            if getattr(s, "command", "").startswith("echo started"):
                process_registry.kill_process(s.id)
