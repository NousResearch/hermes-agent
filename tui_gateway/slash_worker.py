"""Persistent slash-command worker — one HermesCLI per TUI session.

Protocol: reads JSON lines from stdin {id, command}, writes {id, ok, output|error} to stdout.
"""

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import cli as cli_mod
from cli import HermesCLI
from rich.console import Console


def _voice_interrupt_state_path() -> Path:
    home = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
    return Path(home) / "voice_interrupt_state.json"


def _load_voice_interrupt_state() -> dict:
    """Load voice interrupt state from the shared state file.

    This is called after CLI initialization so that a fresh slash worker
    instance picks up the current interrupt/tts state from a prior session
    (e.g., when the TUI restarts and creates a new worker).
    """
    path = _voice_interrupt_state_path()
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "tts_interrupt": bool(data.get("tts_interrupt", False)),
                "tts": bool(data.get("tts", False)),
            }
    except Exception:
        pass
    return {"tts_interrupt": False, "tts": False}


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

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        cli = HermesCLI(model=args.model or None, compact=True, resume=args.session_key, verbose=False)

    # Load the persistent interrupt state so a fresh worker instance (created
    # after TUI restart) starts with the correct interrupt/tts values instead
    # of always starting from False/False.
    _saved = _load_voice_interrupt_state()
    cli._voice_tts_interrupt = _saved["tts_interrupt"]
    cli._voice_tts = _saved["tts"]

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
