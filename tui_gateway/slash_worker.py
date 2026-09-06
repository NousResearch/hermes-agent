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
import logging
import os
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
logger = logging.getLogger(__name__)


def _is_orphaned(original_ppid, getppid=os.getppid) -> bool:
    """Return whether this worker no longer has its original parent."""
    if sys.platform == "win32" and getppid is os.getppid:
        # On Windows os.getppid() does not reliably return the spawning
        # parent's PID — the PEB value can differ from the actual creator
        # PID. Actively probe whether the parent PID is still alive. The
        # ``getppid is os.getppid`` guard keeps the injectable seam intact
        # for unit tests that fake the parent PID.
        import ctypes
        from ctypes import wintypes

        _kernel32 = ctypes.windll.kernel32
        _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        _STILL_ACTIVE = 259
        # Declare signatures: without restype/argtypes, ctypes defaults to
        # c_int and truncates the 64-bit HANDLE returned by OpenProcess.
        _kernel32.OpenProcess.restype = wintypes.HANDLE
        _kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        _kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        _kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        _kernel32.CloseHandle.restype = wintypes.BOOL
        _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        try:
            handle = _kernel32.OpenProcess(
                _PROCESS_QUERY_LIMITED_INFORMATION, False, original_ppid
            )
            if not handle:
                return True
            try:
                code = wintypes.DWORD()
                _kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
                return code.value != _STILL_ACTIVE
            finally:
                _kernel32.CloseHandle(handle)
        except Exception as exc:  # keep worker alive; log for observability
            logger.debug("Windows orphan check failed for ppid %s: %s: %s", original_ppid, type(exc).__name__, exc)
            return False
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
    p.add_argument("--ppid", default="")
    args = p.parse_args()

    os.environ["HERMES_SESSION_KEY"] = args.session_key
    os.environ["HERMES_INTERACTIVE"] = "1"

    # Prefer the spawner-supplied PID: on Windows the venv launcher stub
    # (what server.py's sys.executable resolves to under Popen) is the
    # worker's direct parent, and os.getppid() would pin us to that stub,
    # which outlives the gateway — the orphan watchdog would never fire
    # (see the native Win11 check on #100253). Fall back to getppid() when
    # the flag is absent so older spawn sites keep working.
    orig_ppid = int(args.ppid) if args.ppid else os.getppid()
    # Start before the (hundreds-of-ms) HermesCLI build — that window is itself
    # an orphan risk if the gateway dies mid-spawn.
    _start_parent_death_watchdog(orig_ppid)
    _prepare_slash_worker_runtime()

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        cli = HermesCLI(model=args.model or None, compact=True, resume=args.session_key, verbose=False)

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
            # Workers persist for the TUI session, so release allocator pages at
            # the same command boundary as other long-lived gateway processes.
            # trim_memory's shared cooldown coalesces this with nearby activity.
            try:
                from hermes_cli.mem_trim import trim_memory

                trim_memory(reason="slash worker command completion")
            except Exception as exc:
                # debug, not warning — a persistent failure would repeat on
                # every slash command forever.
                logger.debug(
                    "slash worker memory trim failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )


if __name__ == "__main__":
    main()
