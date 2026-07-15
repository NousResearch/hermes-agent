#!/usr/bin/env python3
"""Parent-death watchdog supervisor for stdio MCP subprocesses.

Problem this fixes (#TBD): a stdio MCP server (e.g. ``npx -y mcp-remote
<url>``) is spawned as a direct child of the Hermes process. Hermes's own
teardown path (``MCPServerTask.shutdown()`` / ``_kill_orphaned_mcp_children``
at final exit) reaps it cleanly on a *graceful* exit. But if the spawning
Hermes process dies hard — ``kill -9``, an OS-level crash, a force-quit of
the TUI/desktop app — that teardown code never runs, and the child (plus any
of its own descendants, e.g. mcp-remote's spawned ``node`` process) is
orphaned. macOS has no direct equivalent of Linux's
``prctl(PR_SET_PDEATHSIG)`` to make the kernel auto-kill a child when its
parent dies, so nothing reaps these until the next Hermes startup's opt-in
``_kill_orphaned_mcp_children()`` sweep — which only runs if something calls
it. Repeated ungraceful session restarts can pile up N orphaned processes,
all racing to hold the same upstream SSE session, producing errors like
"Invalid request parameters" / "Received request before initialization was
complete" on the *legitimate* new connection.

Fix: don't spawn the MCP server command directly. Spawn this supervisor
instead, which:
  1. execs the real command as its own child (own process group via
     ``start_new_session``, so it doesn't inherit the supervisor's
     controlling terminal weirdly and so we can killpg it cleanly);
  2. transparently passes stdin/stdout/stderr through — the MCP stdio
     protocol talks directly over those pipes, so the supervisor must be a
     no-op relay, not a bytes-in-the-middle proxy;
  3. runs a background thread that polls the ORIGINAL parent PID using the
     exact same orphan-detection algorithm already proven in
     ``tui_gateway/slash_worker.py`` (``_is_orphaned``): compare current
     ``getppid()`` against the recorded original, and guard PID reuse via
     ``psutil`` process creation time;
  4. the instant the original parent is gone, terminates the real child's
     process group (SIGTERM, grace period, then SIGKILL) and exits.

This is intentionally a thin, dependency-light script (``psutil`` only,
already a hard dependency via ``tui_gateway/slash_worker.py``) so it starts
fast and can't itself become a resource leak.

Usage (see ``tools/mcp_tool.py::_run_stdio``)::

    python3 -m tools.mcp_stdio_watchdog \\
        --ppid <original_parent_pid> --create-time <original_parent_create_time> \\
        -- <real_command> <arg1> <arg2> ...
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a hard dependency elsewhere
    psutil = None

_POLL_INTERVAL_S = 2.0
_TERM_GRACE_S = 3.0
# psutil derives create_time() from wall-clock boot time. On WSL, host clock
# resynchronisation can shift that value by a few seconds for the *same* live
# PID. Treat only a material mismatch as PID reuse; a new process reusing the
# same PID within this window is vanishingly unlikely.
_CREATE_TIME_TOLERANCE_S = 5.0


def _read_proc_start_ticks(pid: int) -> int | None:
    """Return Linux's stable process start token from ``/proc/<pid>/stat``.

    Field 22 is measured in clock ticks since boot and does not move when WSL
    resynchronises its wall clock. ``comm`` may contain spaces or parentheses,
    so split only after its final closing parenthesis.
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stat_file:
            stat_line = stat_file.read()
        fields_after_comm = stat_line[stat_line.rfind(")") + 2 :].split()
        return int(fields_after_comm[19])
    except (OSError, ValueError, IndexError):
        return None


def _is_orphaned(
    original_ppid: int,
    parent_create_time: float,
    getppid=os.getppid,
    parent_start_ticks: int | None = None,
) -> bool:
    """Return whether the original parent process is gone or was replaced.

    Linux procfs start ticks are preferred because they are monotonic and
    stable on WSL. Other POSIX platforms fall back to psutil create_time with
    a small tolerance for wall-clock adjustments.
    """
    if getppid() != original_ppid:
        return True
    if parent_start_ticks is not None:
        current_start_ticks = _read_proc_start_ticks(original_ppid)
        return current_start_ticks is None or current_start_ticks != parent_start_ticks
    if psutil is None:
        # No reliable staleness check available; fall back to the ppid
        # comparison alone (still catches the common case).
        return False
    try:
        if not psutil.pid_exists(original_ppid):
            return True
        actual_create_time = psutil.Process(original_ppid).create_time()
        return abs(actual_create_time - parent_create_time) > _CREATE_TIME_TOLERANCE_S
    except psutil.Error:
        return True


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Best-effort SIGTERM-then-SIGKILL of the child's process group.

    This module only ever runs on POSIX (the wrap site in tools/mcp_tool.py
    gates on ``os.name == "posix"``), but guard the POSIX-only primitives
    anyway so an accidental Windows import/execute degrades to a plain
    child kill instead of AttributeError.
    """
    killpg = getattr(os, "killpg", None)
    if killpg is None:  # windows-footgun: ok — non-POSIX fallback
        try:
            proc.terminate()
            proc.wait(timeout=_TERM_GRACE_S)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return
    sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
    for sig in (signal.SIGTERM, sigkill):
        try:
            killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            return
        try:
            proc.wait(timeout=_TERM_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            continue


def _watchdog_loop(proc: subprocess.Popen, original_ppid: int, parent_create_time: float) -> None:
    parent_start_ticks = _read_proc_start_ticks(original_ppid)
    while proc.poll() is None:
        if _is_orphaned(
            original_ppid,
            parent_create_time,
            parent_start_ticks=parent_start_ticks,
        ):
            _terminate_process_group(proc)
            return
        time.sleep(_POLL_INTERVAL_S)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parent-death watchdog for a stdio MCP subprocess.",
    )
    parser.add_argument("--ppid", type=int, required=True)
    parser.add_argument("--create-time", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    real_argv = list(args.command)
    if real_argv and real_argv[0] == "--":
        real_argv = real_argv[1:]
    if not real_argv:
        print("mcp_stdio_watchdog: no command given after '--'", file=sys.stderr)
        return 2

    # New process group so we can killpg() the whole tree the real command
    # may spawn (e.g. mcp-remote's own child `node` process), without
    # touching our own group or the (already-gone) original parent's.
    proc = subprocess.Popen(
        real_argv,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
        start_new_session=True,
    )

    # Because the real server lives in its OWN process group (above), the
    # parent's graceful-shutdown killpg of *our* group no longer reaches it.
    # Forward SIGTERM/SIGINT to the child's group so graceful teardown
    # (`_kill_orphaned_mcp_children`, shutdown sweeps) still kills a wedged
    # server that ignores stdin EOF — otherwise the watchdog wrap would
    # invert the bug it fixes.
    def _forward_shutdown(signum, frame):  # noqa: ARG001
        _terminate_process_group(proc)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _forward_shutdown)
    signal.signal(signal.SIGINT, _forward_shutdown)

    watchdog = threading.Thread(
        target=_watchdog_loop,
        args=(proc, args.ppid, args.create_time),
        daemon=True,
    )
    watchdog.start()

    try:
        return proc.wait()
    except KeyboardInterrupt:
        _terminate_process_group(proc)
        return 130


if __name__ == "__main__":
    sys.exit(main())
