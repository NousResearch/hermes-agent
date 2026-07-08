#!/usr/bin/env python3
"""Parent-death watchdog supervisor for stdio MCP subprocesses.

Problem this fixes (#61059): a stdio MCP server (e.g. ``npx -y
@modelcontextprotocol/server-filesystem``) is spawned as a direct child
of the Hermes process. Hermes's own teardown path
(`MCPServerTask.shutdown()` / `_kill_orphaned_mcp_children` at final
exit) reaps it cleanly on a *graceful* exit. But if the spawning Hermes
process dies hard — ``kill -9``, an OS-level crash, a force-quit of the
TUI/desktop app — that teardown code never runs, and the child (plus any
of its own descendants, e.g. mcp-remote's spawned ``node`` process) is
orphaned.

On POSIX, macOS has no direct equivalent of Linux's ``prctl(PR_SET_PDEATHSIG)``
to make the kernel auto-kill a child when its parent dies, so nothing
reaps these until the next Hermes startup's opt-in ``_kill_orphaned_mcp_children()``
sweep — which only runs if something calls it. Repeated ungraceful
session restarts can pile up N orphaned processes, all racing to hold the
same upstream SSE session, producing errors like "Invalid request parameters"
/ "Received request before initialization was complete" on the *legitimate*
new connection.

On Windows, the situation is even worse: when a parent dies, children are
reparented to a system process (often ``svchost.exe`` or ``System``) but are
not automatically terminated. Because Windows lacks ``killpg()`` and the
watchdog supervisor spawned with ``start_new_session=True`` cannot signal
the child's process group, orphaned MCP subprocesses accumulate unchecked.
Over 106 restarts, 40+ ``node.exe`` processes can leak, consuming ~1.6GB RAM
and causing 228+ event loop stalls (max 102s) — the core issue in #61059.

Fix: don't spawn the MCP server command directly. Spawn this supervisor
instead, which:
  1. execs the real command as its own child (own process group on POSIX
     via ``start_new_session``; direct child on Windows, where process
     groups are not used for signal delivery);
  2. transparently passes stdin/stdout/stderr through — the MCP stdio
     protocol talks directly over those pipes, so the supervisor must be a
     no-op relay, not a bytes-in-the-middle proxy;
  3. runs a background thread that polls the ORIGINAL parent PID using the
     exact same orphan-detection algorithm already proven in
     ``tui_gateway/slash_worker.py`` (``_is_orphaned``): compare current
     ``getppid()`` against the recorded original, and guard PID reuse via
     ``psutil`` process creation time;
  4. the instant the original parent is gone, terminates the real child and
     ALL of its descendants (POSIX: SIGTERM to process group; Windows: walk
     the process tree via psutil and terminate each), then exits.

On POSIX, we terminate the entire process group (``killpg``) to catch
grandchildren spawned by the MCP server itself (e.g. ``node`` from ``npx``).
On Windows, where ``killpg`` does not exist, we use ``psutil.Process.children(recursive=True)``
to find all descendants and terminate them individually.

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


def _is_orphaned(original_ppid: int, parent_create_time: float, getppid=os.getppid) -> bool:
    """Mirrors ``tui_gateway.slash_worker._is_orphaned`` exactly.

    True once the process that spawned us is gone. Never trusts a bare
    ``getppid() == 1`` check (Linux reparents orphans to a subreaper, not
    always PID 1), and guards against PID reuse via the recorded creation
    time of the original parent.
    """
    if getppid() != original_ppid:
        return True
    if psutil is None:
        # No reliable staleness check available; fall back to the ppid
        # comparison alone (still catches the common case).
        return False
    try:
        if not psutil.pid_exists(original_ppid):
            return True
        return psutil.Process(original_ppid).create_time() != parent_create_time
    except psutil.Error:
        return True


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Best-effort SIGTERM-then-SIGKILL of the child and all descendants.

    On POSIX: uses ``killpg`` to signal the child's process group, catching
    grandchildren spawned by the MCP server itself (e.g. ``node`` from ``npx``).

    On Windows: walks the process tree via psutil to find all descendants and
    terminates them individually, since ``killpg`` is not available.
    """
    killpg = getattr(os, "killpg", None)
    if killpg is not None:
        # POSIX path: terminate the process group
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
        return

    # Windows path: walk the process tree and terminate individually
    if psutil is None:
        # No psutil available — fall back to direct child termination only
        try:
            proc.terminate()
            proc.wait(timeout=_TERM_GRACE_S)
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except OSError:
                pass
        return

    try:
        child = psutil.Process(proc.pid)
    except psutil.Error:
        return

    # Collect all descendants (recursive) including the child itself
    descendants = [child] + child.children(recursive=True)

    # First pass: SIGTERM
    for descendant in descendants:
        try:
            descendant.terminate()
        except psutil.Error:
            pass

    # Wait for graceful termination
    try:
        child.wait(timeout=_TERM_GRACE_S)
    except (psutil.Error, subprocess.TimeoutExpired):
        # Second pass: SIGKILL survivors
        for descendant in descendants:
            try:
                descendant.kill()
            except psutil.Error:
                pass


def _watchdog_loop(proc: subprocess.Popen, original_ppid: int, parent_create_time: float) -> None:
    while proc.poll() is None:
        if _is_orphaned(original_ppid, parent_create_time):
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

    # On POSIX, create a new process group so we can killpg() the whole tree
    # the real command may spawn (e.g. mcp-remote's own child `node` process),
    # without touching our own group or the (already-gone) original parent's.
    # On Windows, process groups are not used for signal delivery, so we spawn
    # as a direct child and use psutil to walk the tree for termination.
    popen_kwargs = {
        "stdin": sys.stdin,
        "stdout": sys.stdout,
        "stderr": sys.stderr,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(real_argv, **popen_kwargs)

    # Forward SIGTERM/SIGINT to the child on POSIX so graceful teardown
    # (`_kill_orphaned_mcp_children`, shutdown sweeps) still kills a wedged
    # server that ignores stdin EOF — otherwise the watchdog wrap would
    # invert the bug it fixes. Windows does not use POSIX signals.
    if os.name == "posix":
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