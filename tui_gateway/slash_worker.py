"""Persistent slash-command worker — one HermesCLI per TUI session.

Protocol: reads JSON lines from stdin {id, command}, writes {id, ok, output|error} to stdout.
"""

# Stop a ``utils/`` (or ``proxy/``, ``ui/``) package in the launch directory from shadowing Hermes's own
# top-level modules: this worker is spawned as ``-m tui_gateway.slash_worker`` with the user's CWD, so
# ``import cli`` would otherwise resolve ``utils`` to a colliding local package and crash the child in a
# retry loop. ``hermes_bootstrap`` lives at the repo root (no collision risk), so importing it first is safe.
# ``hermes_bootstrap`` lives at the repo root, so importing it is safe before the guard runs (its name won't
# collide with a user package), and it owns the canonical path-hardening logic shared with the other entry
# points — #51693 added the guard to ``entry.py``/``acp_adapter/entry.py`` but missed this child.
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
from tui_gateway._env import env_float
from tui_gateway._stdin_recovery import handle_spurious_eof
from rich.console import Console

# Env-overridable so the integration test can drive sub-second timing.
_WATCHDOG_POLL_S = max(0.05, env_float("HERMES_SLASH_WATCHDOG_POLL_S", 2.0))
_ORPHAN_GRACE_S = max(0.0, env_float("HERMES_SLASH_WATCHDOG_GRACE_S", 5.0))
_in_flight = threading.Event()  # set while a command is executing
logger = logging.getLogger(__name__)


def _is_orphaned(original_ppid, getppid=os.getppid) -> bool:
    """Return whether this worker no longer has its original POSIX parent."""
    return getppid() != original_ppid


def _prepare_slash_worker_runtime() -> None:
    """Start bounded MCP discovery before HermesCLI snapshots tools: each slash_worker child is its
    own process — the parent ``hermes serve`` discovery thread does not populate this registry.

    See #61891.
    """
    from hermes_cli.mcp_startup import start_background_mcp_discovery, wait_for_mcp_discovery
    start_background_mcp_discovery(logger=logger, thread_name="slash-worker-mcp-discovery")
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
    buf = io.StringIO()
    # Rich Console captures its file handle at construction, so redirect_stdout won't affect it; swap
    # the console's file so self.console.print() is captured. cli._cprint is likewise redirected.
    cli.console = Console(file=buf, force_terminal=True, width=120)
    old = getattr(cli_mod, "_cprint", None)
    if old is not None:
        cli_mod._cprint = lambda text: print(text)
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            cli.process_command(cmd if cmd.startswith("/") else f"/{cmd}")
    finally:
        if old is not None:
            cli_mod._cprint = old
    # Desktop chat bubbles render plain text, not ANSI. A command that emits Rich color (e.g. /journey
    # under the gateway's inherited COLORTERM) would leak raw escapes; strip at this single choke point.
    from tools.ansi_strip import strip_ansi
    return strip_ansi(buf.getvalue().rstrip())


def _sw_log(reason: str) -> None:
    print(f"[slash-worker] {reason}", file=sys.stderr, flush=True)


def _reply(**fields) -> None:
    sys.stdout.write(json.dumps(fields) + "\n")
    sys.stdout.flush()


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--session-key", required=True)
    p.add_argument("--model", default="")
    args = p.parse_args()
    os.environ["HERMES_SESSION_KEY"] = args.session_key
    os.environ["HERMES_INTERACTIVE"] = "1"
    # Start before the (hundreds-of-ms) HermesCLI build — that window is itself an orphan risk if the
    # gateway dies mid-spawn.
    _start_parent_death_watchdog(os.getppid())
    _prepare_slash_worker_runtime()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        cli = HermesCLI(model=args.model or None, compact=True, resume=args.session_key, verbose=False)
    # Spurious stdin-EOF recovery (same shared-file-description O_NONBLOCK issue as the gateway entry
    # point — any child inheriting fd 0 can flip the flag).
    _sw_recovery_times: list[float] = []
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
            _reply(id=rid, ok=True, output=_run(cli, req.get("command", "")))
        except Exception as e:
            _reply(id=rid, ok=False, error=str(e))
        finally:
            _in_flight.clear()
            # Workers persist for the TUI session: release allocator pages at the command boundary like
            # other long-lived gateway processes (trim_memory's shared cooldown coalesces nearby activity).
            try:
                from hermes_cli.mem_trim import trim_memory
                trim_memory(reason="slash worker command completion")
            except Exception as exc:
                # debug, not warning — a persistent failure would repeat every command.
                logger.debug("slash worker memory trim failed: %s: %s", type(exc).__name__, exc)


if __name__ == "__main__":
    main()
