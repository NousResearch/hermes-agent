"""TUI gateway SIGINT recovery and stdin-EOF exit (#53362)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

from tui_gateway import entry, server

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_sigint_interrupts_only_running_sessions_off_the_handler(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    interrupted, side_effects = [], []

    def fake_interrupt(sid, session):
        entered.set()
        release.wait(5)
        interrupted.append(sid)

    monkeypatch.setattr(server, "_sessions", {"a": {"running": True}, "b": {"running": False}})
    monkeypatch.setattr(server, "_interrupt_session_turn", fake_interrupt)
    # The interrupt path writes under the non-reentrant _stdout_lock, so the handler itself must not.
    monkeypatch.setattr(server, "write_json", lambda obj: side_effects.append(obj) or True)
    monkeypatch.setattr(server, "_clear_pending", lambda sid=None: side_effects.append(sid))
    monkeypatch.setattr(entry.os, "_exit", lambda code: side_effects.append(f"os._exit({code})"))
    monkeypatch.setattr(entry, "_last_sigint_at", None, raising=False)

    entry._handle_sigint(2, None)

    assert entered.wait(5), "running session was never interrupted"
    assert interrupted == []  # the handler returned while the interrupt was still blocked
    release.set()
    for t in threading.enumerate():
        if t.name == "tui-sigint-interrupt":
            t.join(5)
    assert interrupted == ["a"]
    assert side_effects == []  # idle session b's pending prompts were never withdrawn


def test_second_sigint_in_window_takes_the_sigterm_exit_path(monkeypatch):
    exits = []
    clock = iter([100.0, 110.0, 111.0])
    monkeypatch.setattr(entry.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(entry, "_last_sigint_at", None, raising=False)
    monkeypatch.setattr(entry, "_interrupt_running_sessions", lambda: None, raising=False)
    monkeypatch.setattr(server, "_sessions", {})

    def bare_exit(code):
        raise AssertionError("bare os._exit skips the session flush and _hard_exit")

    monkeypatch.setattr(entry.os, "_exit", bare_exit)
    monkeypatch.setattr(entry, "_log_signal", lambda signum, frame: exits.append(signum))

    entry._handle_sigint(2, None)
    entry._handle_sigint(2, None)  # 10 s later: a fresh first press
    assert exits == []
    entry._handle_sigint(2, None)  # 1 s later: escalate
    assert exits == [2]


def test_eof_child_exits_despite_a_live_non_daemon_thread(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent("""
        import runpy, threading, time
        threading.Thread(target=lambda: time.sleep(3600), name="stuck-worker", daemon=False).start()
        runpy.run_module("tui_gateway.entry", run_name="__main__", alter_sys=True)
    """))
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT), "HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S": "0.5",
           "HOME": str(tmp_path), "HERMES_HOME": str(tmp_path / ".hermes")}
    env.pop("HERMES_TUI_DASHBOARD", None)
    proc = subprocess.Popen(
        [sys.executable, str(script)], cwd=str(_REPO_ROOT), env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        _, stderr = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        _, stderr = proc.communicate()
        raise AssertionError(f"EOF'd gateway child never exited:\n{stderr.decode(errors='replace')[-2000:]}")
    assert proc.returncode == 0, stderr.decode(errors="replace")[-2000:]
