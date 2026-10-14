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
from typing import TYPE_CHECKING

from tui_gateway._stdin_recovery import handle_spurious_eof

if TYPE_CHECKING:
    from cli import HermesCLI

logger = logging.getLogger(__name__)


def _prepare_slash_worker_runtime() -> None:
    """Start bounded MCP discovery before HermesCLI snapshots tools: each slash_worker child is its
    own process — the parent ``hermes serve`` discovery thread does not populate this registry.

    See #61891.
    """
    from hermes_cli.mcp_startup import start_background_mcp_discovery, wait_for_mcp_discovery
    start_background_mcp_discovery(logger=logger, thread_name="slash-worker-mcp-discovery")
    wait_for_mcp_discovery()


def _slash_base(command: str) -> str:
    cmd = (command or "").strip()
    cmd = cmd.removeprefix("/")
    return (cmd.split(maxsplit=1)[0] if cmd else "").lower()


class SkillSlashRefused(RuntimeError):
    """Skill slash parks the prompt on ``_pending_input``; this worker has no reader."""

    def __init__(self, base: str):
        self.base = base
        super().__init__(f"skill command refused before process: /{base}")


def _refuse_skill_slash(command: str) -> None:
    """Refuse a skill command before ``process_command`` prints the loading banner.

    A scan failure here is not a miss the parent already handled: only a positive
    hit is refused, so a broken skill index does not block ``/status``.
    """
    base = _slash_base(command)
    if not base:
        return
    try:
        from cli import get_skill_commands
        commands = get_skill_commands()
    except Exception:
        return
    if f"/{base}" in commands:
        raise SkillSlashRefused(base)


def _run(cli: "HermesCLI", command: str) -> str:
    """Run one command; return its captured, ANSI-stripped output.

    A command like /prompt or /blueprint parks the composed text on the one-shot
    ``_pending_agent_seed`` for the interactive REPL loop (cli.py) — but this
    worker has no REPL, so the seed is harvested here onto ``cli._harvested_seed``
    and routed back to the gateway, which sends it as the next turn (#107800).
    """
    import cli as cli_mod
    from rich.console import Console

    cli._harvested_seed = ""  # one-shot: a fresh run never re-sends a stale seed
    cmd = (command or "").strip()
    if not cmd:
        return ""
    _refuse_skill_slash(cmd)
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
    output = strip_ansi(buf.getvalue().rstrip())
    cli._harvested_seed, cli._pending_agent_seed = getattr(cli, "_pending_agent_seed", None) or "", None
    return output


def _sw_log(reason: str) -> None:
    print(f"[slash-worker] {reason}", file=sys.stderr, flush=True)


def _reply(**fields) -> None:
    sys.stdout.write(json.dumps(fields) + "\n")
    sys.stdout.flush()


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--session-key", required=True)
    p.add_argument("--model", default="")
    p.add_argument("--provider", default="")
    # Accepted and ignored: a gateway started before an update still passes it to the updated worker
    # code on disk. The worker dies with its gateway through the gateway's owner-death supervisor.
    p.add_argument("--parent-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = p.parse_args()
    os.environ["HERMES_SESSION_KEY"] = args.session_key
    os.environ["HERMES_INTERACTIVE"] = "1"
    # Import cli ahead of MCP discovery: importing it loads ~/.hermes/.env and sets HERMES_QUIET,
    # which MCP ``${VAR}`` interpolation in the runtime prep depends on.
    from cli import HermesCLI
    _prepare_slash_worker_runtime()

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        # --provider pins the CLI to the parent agent's resolved provider (a MoA session's virtual
        # "moa" provider included). Without it HermesCLI re-resolves from config and dispatches the
        # MoA preset NAME to the configured real provider (#57283).
        cli = HermesCLI(model=args.model or None, provider=args.provider or None,
                        compact=True, resume=args.session_key, verbose=False)
    cli._slash_metrics_surface = None  # the TUI/Desktop client already counted the typed command
    cli.is_slash_worker = True
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
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            output = _run(cli, req.get("command", ""))
            _reply(id=rid, ok=True, output=output, seed=getattr(cli, "_harvested_seed", "") or "")
        except Exception as e:
            _reply(id=rid, ok=False, error=str(e))
        finally:
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
