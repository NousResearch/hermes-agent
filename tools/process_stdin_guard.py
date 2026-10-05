"""Command guard for text the agent writes into a running background process.

A background process can be an interactive shell or interpreter, so ``process(action="write" |
"submit")`` is a second command channel: start a harmless ``bash`` in the background, then feed it
the command ``terminal()`` would have refused. Every line the process would execute goes through
the terminal's guard first. Lines are reassembled per session across writes, with the editing keys
a shell honours applied, so ``rm -rf `` + ``$HOME\\n`` or a backspace trick is judged as the line the
shell will actually run. Based on community PR #22557 (per-write check), extended with line
reassembly. Raw text that never ends a line (single keys, ^C, TUI navigation) is forwarded as-is.
"""

import re

_ESCAPE_SEQUENCE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")
_LINE_END = re.compile(r"\r\n|\r|\n")


def _edited(raw: str) -> str:
    """The line a readline-style shell would execute from *raw* keystrokes."""
    out: list[str] = []
    for ch in _ESCAPE_SEQUENCE.sub("", raw):
        if ch in "\x03\x15":  # ^C abandons the line, ^U kills it
            out.clear()
        elif ch == "\x17":  # ^W kills the previous word
            while out and out[-1].isspace():
                out.pop()
            while out and not out[-1].isspace():
                out.pop()
        elif ch in "\x7f\x08":
            if out:
                out.pop()
        elif ch == "\t" or ch >= " ":
            out.append(ch)
    return "".join(out).strip()


def _judge(line: str) -> dict:
    from tools.terminal_tool import _check_all_guards
    return _check_all_guards(line, "local")


def refuse_stdin(session, data) -> dict | None:
    """``None`` to forward *data* to *session*; otherwise the tool response, and nothing is written."""
    text = data.decode("utf-8", "surrogateescape") if isinstance(data, bytes) else str(data)
    *completed, rest = _LINE_END.split(getattr(session, "_stdin_pending", "") + text)
    for raw in completed:
        line = _edited(raw)
        if not line:
            continue
        verdict = _judge(line)
        if not verdict.get("approved"):
            return {
                "status": verdict.get("status") or "blocked",
                "error": verdict.get("message") or "Process input refused by the command approval policy.",
                "command": line[:500],
                "description": verdict.get("description"),
                "pattern_key": verdict.get("pattern_key"),
            }
    session._stdin_pending = rest
    return None
