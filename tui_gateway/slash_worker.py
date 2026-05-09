"""Persistent slash-command worker — one HermesCLI per TUI session.

Protocol: reads JSON lines from stdin {id, command}, writes {id, ok, output|error} to stdout.
"""

import argparse
import contextlib
import io
import json
import os
import sys

import cli as cli_mod
from cli import HermesCLI
from rich.console import Console


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


def _parse_toolsets(raw: str) -> list[str] | None:
    names = [item.strip() for item in (raw or "").split(",") if item.strip()]
    if not names:
        return _load_configured_toolsets()
    if any(name in {"all", "*"} for name in names):
        return ["all"]
    return names


def _load_configured_toolsets() -> list[str] | None:
    """Resolve the TUI slash worker's configured CLI toolsets.

    The interactive CLI entrypoint passes effective toolsets into HermesCLI.
    The TUI slash worker is a standalone subprocess, so it must resolve the
    same configured platform toolsets itself when the parent did not pass an
    explicit list. Otherwise active/config-aware commands such as /toolsets see
    ``enabled_toolsets=None`` and report an empty active list.
    """
    explicit = [
        item.strip()
        for item in os.environ.get("HERMES_TUI_TOOLSETS", "").split(",")
        if item.strip()
    ]
    if explicit:
        if any(name in {"all", "*"} for name in explicit):
            return ["all"]
        return explicit

    try:
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools

        enabled = sorted(
            _get_platform_tools(
                load_config(),
                "cli",
                include_default_mcp_servers=True,
            )
        )
        return enabled or ["all"]
    except Exception:
        return ["all"]


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--session-key", required=True)
    p.add_argument("--model", default="")
    p.add_argument("--toolsets", default="")
    args = p.parse_args()

    os.environ["HERMES_SESSION_KEY"] = args.session_key
    os.environ["HERMES_INTERACTIVE"] = "1"

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        cli = HermesCLI(
            model=args.model or None,
            compact=True,
            resume=args.session_key,
            verbose=False,
            toolsets=_parse_toolsets(args.toolsets),
        )

    for raw in sys.stdin:
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
        except Exception as e:
            sys.stdout.write(json.dumps({"id": rid, "ok": False, "error": str(e)}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
