"""Persistent slash-command worker — one HermesCLI per TUI session.

Protocol: reads JSON lines from stdin {id, command}, writes {id, ok, output|error} to stdout.

Self-protection (defence-in-depth against orphaned workers):
  1. **Parent watchdog** (best-effort) — if ``psutil`` is available, a daemon
     thread monitors the parent's PID + create_time fingerprint every 10 s and
     exits if the parent disappears. This handles crashes, SIGKILL, and PID
     reuse (critical on Windows).
  2. **Parent-PID poll** (fallback) — the main loop checks ``os.getppid()``
     on each stdin timeout; works without ``psutil`` but cannot detect PID reuse.
  3. **Idle timeout** — exits after ``_IDLE_TIMEOUT_S`` (30 min) without a
     command, bounding worst-case resource usage even if both parent-orphan
     guards fail.
"""

import argparse
import contextlib
import io
import json
import os
import select
import sys
import threading
import time

import cli as cli_mod
from cli import HermesCLI
from rich.console import Console

# ── Optional psutil for parent fingerprinting ────────────────────────
try:
    import psutil
except ImportError:
    psutil = None

# Max seconds of inactivity before the worker self-exits.
_IDLE_TIMEOUT_S = 1800  # 30 minutes

# How often the stdin poll returns to re-check conditions when idle.
_POLL_INTERVAL_S = 60


def _start_parent_watchdog() -> None:
    """Start a daemon thread that exits this process if the parent disappears.

    Uses ``psutil.Process(ppid).create_time()`` as a fingerprint to reliably
    detect parent replacement (PID reuse).  Falls back to no-op when psutil
    is not installed — the main loop's ``getppid()`` check provides a less
    robust but dependency-free alternative.
    """
    if not psutil:
        return

    ppid = os.getppid()
    try:
        parent = psutil.Process(ppid)
        parent_started = parent.create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        os._exit(0)  # parent already gone

    def _watch() -> None:
        time.sleep(5)  # let the main process stabilise
        while True:
            try:
                if not psutil.pid_exists(ppid):
                    os._exit(0)
                # Fingerprint check — catches PID reuse on Windows / fast-restart
                if psutil.Process(ppid).create_time() != parent_started:
                    os._exit(0)
                # POSIX orphan check: adopted by init
                if os.name != "nt" and os.getppid() == 1:
                    os._exit(0)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                os._exit(0)
            except Exception:
                pass  # transient psutil error — retry
            time.sleep(10)

    t = threading.Thread(target=_watch, daemon=True, name="ParentWatchdog")
    t.start()


def _run(cli: HermesCLI, command: str) -> str:
    cmd = (command or "").strip()
    if not cmd:
        return ""
    if not cmd.startswith("/"):
        cmd = f"/{cmd}"

    buf = io.StringIO()

    # Rich Console captures its file handle at construction time, so
    # contextlib.redirect_stdout won't affect it. Swap the console's
    # underlying file to our buffer so self.console.print() is captured.
    cli.console = Console(file=buf, force_terminal=True, width=120)

    old = getattr(cli_mod, "_cprint", None)
    if old is not None:
        cli_mod._cprint = lambda text: print(text)

    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            cli.process_command(cmd)
    finally:
        if old is not None:
            cli_mod._cprint = old

    return buf.getvalue().rstrip()


def main():
    _start_parent_watchdog()

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--session-key", required=True)
    p.add_argument("--model", default="")
    args = p.parse_args()

    os.environ["HERMES_SESSION_KEY"] = args.session_key
    os.environ["HERMES_INTERACTIVE"] = "1"

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        cli = HermesCLI(model=args.model or None, compact=True, resume=args.session_key, verbose=False)

    parent_pid = os.getppid()
    last_command_time = time.monotonic()

    while True:
        # Guard 1: parent-PID check (works without psutil, covers basic
        # orphan scenarios but not PID reuse).
        if os.getppid() != parent_pid:
            break

        # Guard 2: idle timeout — bounds resource use when the upstream
        # cleanup path (close_on_disconnect + server-side teardown) misses
        # a code path.
        idle = time.monotonic() - last_command_time
        if idle >= _IDLE_TIMEOUT_S:
            break

        # Poll stdin with a timeout so we can periodically re-check the
        # conditions above. Without this, a blocking ``for raw in sys.stdin``
        # would hang forever if the pipe is left open.
        poll_timeout = min(_POLL_INTERVAL_S, _IDLE_TIMEOUT_S - idle)
        try:
            r, _, _ = select.select([sys.stdin], [], [], poll_timeout)
        except (ValueError, OSError):
            break  # stdin closed or invalid

        if not r:
            continue  # Timeout — loop back to check parent/idle

        raw = sys.stdin.readline()
        if not raw:
            break  # EOF — stdin closed

        line = raw.strip()
        if not line:
            continue

        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            out = _run(cli, req.get("command", ""))
            sys.stdout.write(json.dumps({"id": rid, "ok": True, "output": out}) + "\n")
            sys.stdout.flush()
            last_command_time = time.monotonic()
        except Exception as e:
            sys.stdout.write(json.dumps({"id": rid, "ok": False, "error": str(e)}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
