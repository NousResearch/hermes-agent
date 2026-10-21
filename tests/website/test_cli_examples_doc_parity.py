"""Doc-fact contract: every ``hermes ...`` example in the docs must get past argparse.

Users copy these commands at the moments that matter (moving a profile to a
new machine, rolling back after an update), and a rejected one gives them only
an argparse error. Examples drifted this way: ``hermes profile export work
./out.tar.gz`` (the path is ``-o``), ``hermes send ntfy:x "hi"`` (the target is
``--to``), ``hermes backup restore`` (no such subcommand).

Each fenced shell line and inline ``hermes ...`` code span in the English docs
is parsed with the real tree (``hermes_cli.main._build_cli_parser`` +
``_parse_cli_args``, after the same ``-p/--profile`` strip ``main`` applies).
It fails when argparse cannot place a token: unrecognized arguments, an invalid
choice, an unknown subcommand, or (fenced lines only) a missing required
argument. Placeholders (``<id>``, ``N``), shell variables and synopsis syntax
(``[--flag]``, ``a|b``, ``start/stop``) are not literal argv and are skipped.
So is any top-level word the built-in tree does not know: plugins register
their own commands, and which ones exist depends on the install.

Only the English docs are gated: i18n copies lag by design.
"""

from __future__ import annotations

import contextlib
import io
import re
import shlex
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "website" / "docs"

_FENCE_RE = re.compile(r"^[ \t]*```(\w*)[^\n]*\n(.*?)^[ \t]*```", re.MULTILINE | re.DOTALL)
_INLINE_RE = re.compile(r"(?<!`)`(hermes [^`\n]+)`(?!`)")
_SHELL_LANGS = {"", "bash", "sh", "shell", "console", "zsh"}
_COMMENT_RE = re.compile(r"(?:^|\s)#(?:\s.*)?$")
_PLACEHOLDER_RE = re.compile(r"<[^<>\n]*>")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SEPARATORS = {"|", "||", "&&", ";", "&"}
_REDIRECTS = {"<", ">", ">>", "2>", "&>"}

_UNRECOGNIZED_RE = re.compile(r"unrecognized arguments: (\S+)")
_BAD_CHOICE_RE = re.compile(r"invalid choice: '([^']*)'")
_NOT_A_COMMAND_RE = re.compile(r"'([^']*)' is not a `[^`]+` command")
_MISSING_RE = re.compile(r"the following arguments are required|expected one argument")

# (doc, rejected token) pairs the docs describe ahead of the code; drop an entry
# when the code catches up.
_KNOWN_GAPS = {
    # `hermes memory setup mem0 --mode ...` provider flags (#54945).
    ("user-guide/features/memory-providers.md", "--mode"),
    # Kanban `scheduled_at` start times (#80119).
    ("user-guide/features/kanban.md", "--scheduled-at"),
    ("user-guide/features/kanban.md", "--at"),
    # Named as a follow-up that does not exist yet.
    ("user-guide/egress/iron-proxy.md", "rotate-ca"),
}


def _fenced_commands(body: str, first_line: int):
    """Logical commands in a shell block: comments dropped, backslash continuations
    and indented synopsis wraps (``    [--json]``) joined to the line they extend."""
    pending, start = "", first_line
    for offset, raw in enumerate(body.split("\n")):
        line = _COMMENT_RE.sub("", raw).rstrip()
        wraps = pending and raw[:1].isspace() and line.lstrip()[:1] in ("[", "-", "<")
        if pending and not pending.endswith("\\") and not wraps:
            yield start, pending
            pending = ""
        if not pending:
            start = first_line + offset
        head = pending[:-1] if pending.endswith("\\") else pending
        pending = f"{head} {line.strip()}".strip()
    if pending:
        yield start, pending


def _examples():
    """(doc, line, command text, fenced) for every candidate example in the docs."""
    for md in sorted(DOCS_ROOT.rglob("*.md")):
        text = md.read_text(encoding="utf-8")
        doc = md.relative_to(DOCS_ROOT).as_posix()
        for m in _FENCE_RE.finditer(text):
            if m.group(1).lower() in _SHELL_LANGS:
                first = text.count("\n", 0, m.start(2)) + 1
                for line, command in _fenced_commands(m.group(2), first):
                    yield doc, line, command, True
        prose = _FENCE_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)
        for line, row in enumerate(prose.splitlines(), 1):
            for m in _INLINE_RE.finditer(row):
                if "→" not in m.group(1):  # a menu path, not argv
                    yield doc, line, m.group(1), False


def _hermes_argvs(command: str):
    """argv (after ``hermes``) for each ``hermes`` invocation in a shell command."""
    command = _PLACEHOLDER_RE.sub(lambda m: m.group(0).replace(" ", "_"), command)
    try:
        tokens = shlex.split(command)
    except ValueError:
        return
    segment: list[str] = []
    for token in [*tokens, ";"]:
        if token not in _SEPARATORS:
            segment.append(token)
            continue
        while segment and (segment[0] == "$" or _ENV_ASSIGN_RE.match(segment[0])):
            segment.pop(0)
        if segment[:1] == ["hermes"]:
            argv, skip = [], False
            for tok in segment[1:]:
                if skip:
                    skip = False
                elif tok in _REDIRECTS:
                    skip = True
                elif not re.match(r"^\d?>", tok):
                    argv.append(tok)
            if not any(t.startswith("[") or "..." in t or "…" in t or "$(" in t for t in argv):
                yield argv
        segment = []


def _rejected_token(parser, subparsers, argv, fenced: bool):
    """The token argparse refused, or None when the example parses or what failed
    is not literal argv (a placeholder, a shell variable, a flag named in prose)."""
    from hermes_cli.main import _parse_cli_args, _scan_profile_flag

    if not fenced and len(argv) == 1 and argv[0].startswith("-"):
        return None
    _, consume, index = _scan_profile_flag(argv)
    if consume and index is not None:
        argv = argv[:index] + argv[index + consume:]
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        try:
            _parse_cli_args(parser, subparsers, argv)
            return None
        except SystemExit as exc:
            if exc.code in (0, None):
                return None
    message = err.getvalue()
    token = None
    if m := _UNRECOGNIZED_RE.search(message):
        token = m.group(1)
    elif m := _NOT_A_COMMAND_RE.search(message):
        token = None if "/" in m.group(1) or "|" in m.group(1) else m.group(1)
    elif m := _BAD_CHOICE_RE.search(message):
        token = m.group(1)
    elif fenced and _MISSING_RE.search(message):
        return "(missing required argument)"
    if token is None or "$" in token or _PLACEHOLDER_RE.fullmatch(token):
        return None
    return token


def _built_in_parser(monkeypatch):
    import hermes_cli.main as cli

    # Plugin commands depend on what is installed/enabled; the built-in tree is deterministic.
    monkeypatch.setattr(cli, "_register_plugin_cli_commands", lambda subparsers: None)
    # A `-p <name>` placeholder must be skipped by the profile scan, not sys.exit the run.
    monkeypatch.setattr(cli, "_looks_like_hermes_invocation", lambda: False)
    return cli._build_cli_parser()


def test_documented_hermes_commands_parse(monkeypatch):
    parser, subparsers = _built_in_parser(monkeypatch)
    built_in = set(subparsers.choices)
    rejected, checked = [], 0
    for doc, line, command, fenced in _examples():
        for argv in _hermes_argvs(command):
            if argv and not argv[0].startswith("-") and argv[0] not in built_in:
                continue
            checked += 1
            token = _rejected_token(parser, subparsers, argv, fenced)
            if token and (doc, token) not in _KNOWN_GAPS:
                rejected.append(f"{doc}:{line}: hermes {shlex.join(argv)}  <- rejects {token}")
    assert checked >= 3000, (
        f"only {checked} `hermes` examples found under website/docs — the fence/inline "
        "extraction may have broken, which would make this contract vacuous."
    )
    assert not rejected, (
        "Documented `hermes` commands that the CLI rejects (fix the example, or the "
        "parser if the doc is right):\n  " + "\n  ".join(rejected)
    )
