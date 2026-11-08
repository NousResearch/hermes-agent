"""AppleScript / JXA (``osascript``) classification for the approval gate (#141346).

``osascript`` drives applications the agent never opened. The documents they hold, and the files
Finder can reach, belong to the operator: closing without saving, quitting, deleting, or running a
shell from AppleScript can destroy work the session cannot recreate. In the reported incident smart
approval auto-approved ``close active presentation saving no`` on the user's own unsaved deck.

The check fails closed instead of parsing AppleScript. An ``osascript`` run is flagged when:

* its program text (every ``-e`` argument, a here-string, a heredoc body) contains a verb from
  ``_RISKY_RE`` (close, quit, save, revert, delete, move, keystroke, click, ``do shell script``,
  raw ``«event»`` codes...). The scan is word-based and also reads string literals and comments, so a
  dialog that only *asks* "Quit without saving?" prompts as well. That costs one prompt; missing a
  real close costs the operator's document;
* the program is JXA (``-l JavaScript``), which can build any method name at run time;
* the program is not visible on the command line: a script file, stdin from a pipe or redirect,
  ``$VAR`` / ``$(...)`` / backtick expansion, or an unquoted heredoc that expands;
* ``osascript`` is run through a carrier (``sh -c``, ``eval``, ``env -S``, ``xargs``, ``ssh``,
  an interpreter's ``os.system``...), so the program cannot be bound to an argv.

A read-only call whose program is a literal ``-e`` argument (``get name of first application
process whose frontmost is true``, ``get the name of every disk``, ``activate``, ``display
notification``) is not flagged and behaves exactly as before.

Both descriptions are in ``SMART_MANUAL_ONLY_DESCRIPTIONS``: smart mode never asks the guardian
LLM about them, and a human answer covers this one command (no session/always grant).
"""

import re

OSASCRIPT_RISKY_DESCRIPTION = (
    "AppleScript can close or quit apps without saving, delete or move files, "
    "send keystrokes, or run shell commands (osascript)"
)
OSASCRIPT_OPAQUE_DESCRIPTION = (
    "osascript program comes from a script file, pipe, carrier or expansion and cannot be reviewed"
)

# Verbs whose effect can discard unsaved work, mutate files, synthesize input or escape to a shell.
_RISKY_RE = re.compile(
    r"\b(?:close|quit|delete|erase|trash|move|write|save|revert|empty|keystroke|click|restart|eject|kill)\b"
    r"|\bset\s+(?:the\s+)?(?:contents|text|value|data)\b"
    r"|\bshut\s*down\b|\blog\s*out\b|\bkey\s*code\b|\bperform\s+action\b|\bdon.?t\s+save\b"
    r"|\bdo\s+shell\s+script\b|\b(?:run|load|store)\s+script\b|\bdo\s+javascript\b|\bexecute\b"
    r"|\bopen\s+for\s+access\b|\bset\s+eof\b|\breplacing\b|\bsaving\s+no\b|\bwithout\s+saving\b"
    r"|«",
    re.IGNORECASE,
)
_JXA_RE = re.compile(r"(?:^|\s)-l\s*['\"]?\s*j(?:ava)?script\b", re.IGNORECASE)
# Commands that run a program taken from their arguments. When their command names osascript
# anywhere (quoted included), the AppleScript cannot be read from this command line.
_CARRIERS = frozenset({
    "sh", "bash", "zsh", "ksh", "dash", "fish", "csh", "tcsh", "ash", "eval", "source", ".",
    "xargs", "parallel", "ssh", "doas", "su", "watch", "script", "launchctl",
    "caffeinate", "arch", "sandbox-exec", "unbuffer", "perl", "ruby", "php", "node", "nodejs",
    "osacompile", "automator",
})
_INTERPRETER_RE = re.compile(r"python[23]?(?:\.\d+)*|perl\d*|ruby[\d.]*|pwsh|powershell")
_HEREDOC_RE = re.compile(r"(?<!<)<<(?!<)-?\s*(?P<q>['\"]?)(?P<delim>[^\s'\"<>;&|]+)(?P=q)")


def _program_name(word: str) -> str:
    """Executable basename of a command word: ``/usr/bin/osascript``, ``o\\sascript`` and
    ``osa''script`` all read ``osascript``. The raw word is split on ``/`` and ``\\`` first (the
    shell reading drops unquoted backslashes, so ``C:\\...\\x.exe`` would fuse into one word); an
    escape inside the name is caught by the plain deobfuscated basename."""
    from tools.approval_detection import _deobfuscate_shell_word_for_detection as deobfuscate
    names = {deobfuscate(re.split(r"[\\/]", word)[-1]), deobfuscate(word).rsplit("/", 1)[-1]}
    names = {n.strip("\"'").lower().removesuffix(".exe") for n in names}
    return "osascript" if "osascript" in names else min(names)


def classify_osascript(text: str) -> set[str]:
    """Descriptions that apply to *text*; empty when every ``osascript`` run is a visible,
    read-only program. A heredoc body is consumed with the command that owns it."""
    from tools.approval_detection import _iter_shell_command_word_spans, _shell_command_segment

    found: set[str] = set()
    if not _mentions_osascript(text):
        return found
    skip_until = -1
    for start, end, word in _iter_shell_command_word_spans(text):
        if start < skip_until:
            continue
        name = _program_name(word)
        segment = _shell_command_segment(text, start)
        args = segment[end - start:]
        if name != "osascript":
            if _expands(word) and (_mentions_osascript(segment) or word[:1] in ("$", "`")):
                found.add(OSASCRIPT_OPAQUE_DESCRIPTION)  # executable chosen at run time (`X=osascript; $X`)
            elif _mentions_osascript(args) and _runs_its_arguments(name, args):
                found.add(OSASCRIPT_OPAQUE_DESCRIPTION)  # `sh -c "osascript ..."`, `xargs osascript`
            continue
        program, opaque, heredoc = _osascript_program(args)
        if heredoc is not None:
            delim, quoted = heredoc
            body, skip_until = _heredoc_body(text, start + len(segment), delim)
            opaque = opaque or (not quoted and _expands(body))  # an unquoted delimiter expands the body
            program += "\n" + body
        if opaque or _expands(args):
            found.add(OSASCRIPT_OPAQUE_DESCRIPTION)
        if _JXA_RE.search(args) or _RISKY_RE.search(program):
            found.add(OSASCRIPT_RISKY_DESCRIPTION)
    return found


def _mentions_osascript(text: str) -> bool:
    """Whether ``osascript`` appears once quote and escape characters are dropped (``osa''script``)."""
    return "osascript" in re.sub(r"[\"'\\\\]", "", text).lower()


def _runs_its_arguments(name: str, args: str) -> bool:
    """Whether the command *name* executes a program named in *args* (a carrier)."""
    if name == "find":
        return bool(re.search(r"(?:^|\s)-(?:exec|execdir|ok|okdir)\b", args))
    if name == "env":
        return bool(re.search(r"(?:^|\s)(?:-\w*S|--split-string)", args))
    return name in _CARRIERS or bool(_INTERPRETER_RE.fullmatch(name))


def _expands(raw: str) -> bool:
    """True when *raw* holds ``$`` or a backtick the shell expands (anything outside single quotes)."""
    from tools.approval_detection import _scan_shell
    return any(kind != "quote" and quote != "'" and raw[i] in "$`" for kind, i, _, quote in _scan_shell(raw))


def _osascript_program(args: str) -> tuple[str, bool, tuple[str, bool] | None]:
    """``(program_text, opaque, heredoc)`` for one osascript argv tail.

    ``program_text`` joins every ``-e`` argument and here-string. ``opaque`` is True when part of the
    program is not on this command line: a script-file operand, ``-`` / no ``-e`` (stdin), a ``<``
    redirect, ``-i`` (interactive), or unparseable quoting. ``heredoc`` is ``(delimiter, quoted)``."""
    from tools.approval_detection import _shell_segment_tokens

    tokens = _shell_segment_tokens(args, 0)
    if tokens is None:
        return "", True, None
    pieces: list[str] = []
    heredoc = None
    opaque = False
    i = 0
    while i < len(tokens):
        token = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if token in (">", ">>", ">|", ">&") or (token.isdigit() and nxt.startswith(">")):
            i += 1 if token.isdigit() else 2
            continue
        if token == "<<<":
            pieces.append(nxt)
            i += 2
            continue
        if token in ("<<", "<<-"):
            raw = _HEREDOC_RE.search(args)
            heredoc = (nxt.lstrip("-"), bool(raw and raw.group("q")))
            i += 2
            continue
        if token in ("<", "<>"):
            return " ".join(pieces), True, heredoc
        if token in ("-e", "-l", "-s"):
            if token == "-e":
                pieces.append(nxt)
            i += 2
            continue
        if token.startswith("-e") and len(token) > 2:
            pieces.append(token[2:])
        elif not token.startswith("-") and (pieces or heredoc is not None):
            break  # after -e or a heredoc, operands are the program's own argv
        elif token.startswith("-i") or token == "-" or not token.startswith("-"):
            # `-i` reads an interactive session; a first operand is a script file (`-` is stdin).
            return " ".join(pieces), True, heredoc
        i += 1
    if not pieces and heredoc is None:
        opaque = True  # no -e, no heredoc: the program is read from stdin
    return "\n".join(pieces), opaque, heredoc


def _heredoc_body(text: str, after: int, delim: str) -> tuple[str, int]:
    """``(body, end_offset)`` of the heredoc starting on the line after *after*. An unterminated
    body runs to the end of *text*, as the shell reads it."""
    newline = text.find("\n", after)
    if newline < 0:
        return "", len(text)
    end = re.compile(rf"^\t*{re.escape(delim)}[ \t]*$", re.MULTILINE).search(text, newline + 1)
    if end is None:
        return text[newline + 1:], len(text)
    return text[newline + 1:end.start()], end.end()
