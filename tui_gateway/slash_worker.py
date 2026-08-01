"""Persistent slash-command worker — one HermesCLI per TUI session.

Protocol: reads JSON lines from stdin {id, command}, writes {id, ok, output|error} to stdout.
"""

# Stop a ``utils/`` (or ``proxy/``, ``ui/``) package in the launch directory
# from shadowing Hermes's own top-level modules.  This worker is spawned as
# ``-m tui_gateway.slash_worker`` and inherits the user's CWD, so the ``import
# cli`` below would otherwise resolve ``utils`` to a colliding local package
# and crash the child in a retry loop (issue #51286).  ``hermes_bootstrap``
# lives at the repo root, so importing it is safe before the guard runs (its
# name won't collide with a user package), and it owns the canonical
# path-hardening logic shared with the other entry points — #51693 added the
# guard to ``entry.py``/``acp_adapter/entry.py`` but missed this child.
import hermes_bootstrap

hermes_bootstrap.harden_import_path()

import argparse
import contextlib
import io
import json
import os
import queue
import sys
import threading
import time

import cli as cli_mod
from cli import HermesCLI
from tui_gateway._stdin_recovery import handle_spurious_eof
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

# Hard ceiling on HermesCLI(...) construction, which opens SessionDB (state.db)
# and -- when resuming -- replays that session's history, all before this
# worker can process a single command. Root-caused from a live incident: a
# worker sat with zero progress for 17+ minutes at exactly this call, on a
# profile whose state.db had grown to 150+MB. py-spy showed the main thread
# blocked in native code with no deeper Python frames (consistent with a slow
# SQLite operation, not a Python-level infinite loop), and this codebase has
# a documented precedent of a large state.db causing multi-minute stalls
# elsewhere (`hermes update`). SessionDB's own connection has only a 1s
# *lock-wait* timeout -- nothing bounds how long schema init/integrity work
# takes once it has the lock. Unlike the terminal-tool watchdogs, there is no
# "gracefully continue" option here: if HermesCLI can't be built, this worker
# cannot serve any command at all, so on timeout we fail loudly and exit
# rather than sit invisibly frozen for the rest of the process's life (which
# is what happened before this fix -- the UI just showed stale state forever
# with no error, because nothing was left to report one).
_CLI_INIT_WATCHDOG_S = max(10.0, _env_float("HERMES_SLASH_CLI_INIT_WATCHDOG_S", 120.0))


def _build_cli_with_watchdog(model, session_key):
    """Construct HermesCLI on a daemon thread with a hard wall-clock ceiling.

    Plain daemon thread (not concurrent.futures.ThreadPoolExecutor): the
    latter's non-daemon workers plus its atexit join-all-threads hook would
    make this worker process itself refuse to exit promptly if construction
    ever actually wedged -- defeating the point of failing loudly here.
    """
    result_queue = queue.Queue(maxsize=1)

    def _run():
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                built = HermesCLI(model=model, compact=True, resume=session_key, verbose=False)
            result_queue.put(("ok", built))
        except BaseException as e:  # noqa: BLE001 - must forward any failure to the waiter
            result_queue.put(("error", e))

    threading.Thread(
        target=_run, name="cli-init-watchdog", daemon=True,
    ).start()
    try:
        kind, payload = result_queue.get(timeout=_CLI_INIT_WATCHDOG_S)
    except queue.Empty:
        print(
            f"[slash-worker] FATAL: HermesCLI(resume={session_key!r}) did not "
            f"return within {_CLI_INIT_WATCHDOG_S:.0f}s -- state.db is likely "
            "very large or under heavy contention. Exiting so the gateway/UI "
            "sees a clear failure instead of an invisible freeze.",
            file=sys.stderr, flush=True,
        )
        os._exit(1)

    if kind == "error":
        raise payload
    return payload


def _is_orphaned(original_ppid, getppid=os.getppid) -> bool:
    """Return whether this worker no longer has its original POSIX parent."""
    return getppid() != original_ppid


def _prepare_slash_worker_runtime() -> None:
    """Start bounded MCP discovery before HermesCLI snapshots tools.

    Each slash_worker child is its own process — the parent ``hermes serve``
    discovery thread does not populate this registry (issue #61891).
    """
    import logging

    from hermes_cli.mcp_startup import (
        start_background_mcp_discovery,
        wait_for_mcp_discovery,
    )

    logger = logging.getLogger(__name__)
    start_background_mcp_discovery(
        logger=logger,
        thread_name="slash-worker-mcp-discovery",
    )
    wait_for_mcp_discovery()


def _start_parent_death_watchdog(original_ppid) -> None:
    def _loop():
        while not _is_orphaned(original_ppid):
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

    # Desktop chat bubbles render plain text, not ANSI. A worker-routed command
    # that emits Rich color (e.g. /journey building its own Console, which picks
    # up truecolor from the gateway's inherited COLORTERM) would otherwise leak
    # raw escapes; strip them at the single choke point. (The TUI opens /journey
    # as an overlay, so it never travels this path.)
    from tools.ansi_strip import strip_ansi

    return strip_ansi(buf.getvalue().rstrip())


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
    _start_parent_death_watchdog(orig_ppid)
    _prepare_slash_worker_runtime()

    cli = _build_cli_with_watchdog(args.model or None, args.session_key)

    # Spurious stdin-EOF recovery (same O_NONBLOCK shared file-description
    # issue as the gateway entry point — any child inheriting fd 0 can flip
    # the flag and launder EAGAIN into an apparent EOF).
    _sw_recovery_times: list[float] = []

    def _sw_log(reason: str) -> None:
        print(f"[slash-worker] {reason}", file=sys.stderr, flush=True)

    while True:
        raw = sys.stdin.readline()
        if not raw:
            if not handle_spurious_eof(_sw_recovery_times, _sw_log):
                break
            continue

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
