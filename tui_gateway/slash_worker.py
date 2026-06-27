"""Persistent slash-command worker — one HermesCLI per TUI session.

Protocol: reads JSON lines from stdin {id, command}, writes {id, ok, output|error} to stdout.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import threading
import time

import psutil

import cli as cli_mod
from cli import HermesCLI
from rich.console import Console

# Env-overridable so the integration test can drive sub-second timing.
def _env_float(name: str, default: float) -> float:
    """Parse a float env knob, falling back to ``default`` on absent/malformed
    values. A bare ``float(os.environ.get(...))`` would raise ValueError at
    import time on a typo (e.g. ``HERMES_SLASH_WATCHDOG_POLL_S=2s``) and kill
    the worker before it can serve a single command."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


_WATCHDOG_POLL_S = max(0.05, _env_float("HERMES_SLASH_WATCHDOG_POLL_S", 2.0))
_ORPHAN_GRACE_S = max(0.0, _env_float("HERMES_SLASH_WATCHDOG_GRACE_S", 5.0))
_in_flight = threading.Event()  # set while a command is executing


def _is_orphaned(original_ppid, parent_create_time, getppid=os.getppid) -> bool:
    """True once our spawning gateway is gone. Compare to the ORIGINAL ppid
    (never ==1: Linux reparents to a subreaper) and guard PID reuse via
    create_time."""
    if getppid() != original_ppid:
        return True
    try:
        if not psutil.pid_exists(original_ppid):
            return True
        parent = psutil.Process(original_ppid)
        if parent.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
            return True
        return parent.create_time() != parent_create_time
    except psutil.Error:
        return True


def _start_parent_death_watchdog(original_ppid, parent_create_time) -> None:
    def _loop():
        while not _is_orphaned(original_ppid, parent_create_time):
            time.sleep(_WATCHDOG_POLL_S)
        deadline = time.monotonic() + _ORPHAN_GRACE_S
        while _in_flight.is_set() and time.monotonic() < deadline:
            time.sleep(0.05)  # let an in-flight command finish/flush
        os._exit(0)

    threading.Thread(target=_loop, daemon=True).start()


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
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--session-key", required=True)
    p.add_argument("--model", default="")
    args = p.parse_args()

    os.environ["HERMES_SESSION_KEY"] = args.session_key
    os.environ["HERMES_INTERACTIVE"] = "1"

    # Start before the (hundreds-of-ms) HermesCLI build — that window is itself
    # an orphan risk if the gateway dies mid-spawn.
    orig_ppid = os.getppid()
    try:
        parent_create_time = psutil.Process(orig_ppid).create_time()
    except psutil.Error:
        parent_create_time = 0.0
    _start_parent_death_watchdog(orig_ppid, parent_create_time)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        cli = HermesCLI(model=args.model or None, compact=True, resume=args.session_key, verbose=False)

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue

        _in_flight.set()
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            out = _run(cli, req.get("command", ""))
            sys.stdout.write(json.dumps({"id": rid, "ok": True, "output": out}) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"id": rid, "ok": False, "error": str(e)}) + "\n")
            sys.stdout.flush()
        finally:
            _in_flight.clear()


if __name__ == "__main__":
    main()
