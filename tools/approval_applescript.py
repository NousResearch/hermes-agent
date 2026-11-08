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
# Commands that run a program taken from their arguments. When their command names osascript
# anywhere (quoted included), the AppleScript cannot be read from this command line.
_CARRIERS = frozenset({
    "sh", "bash", "zsh", "ksh", "dash", "fish", "csh", "tcsh", "ash", "eval", "source", ".",
    "xargs", "parallel", "ssh", "doas", "su", "watch", "script", "launchctl",
    "caffeinate", "arch", "sandbox-exec", "unbuffer", "perl", "ruby", "php", "node", "nodejs",
    "osacompile", "automator",
})
_HEREDOC_RE = re.compile(r"<<(?P<dash>-?)[ \t]*(?P<q>['\"]?)(?P<delim>[^\s'\"<>;&|()]+)(?P=q)(?=[\s;&|<>)]|$)")


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
    read-only program. The raw text and the backslash-newline-joined text are both read and the
    findings united: joining is how the shell reads ``osa\\<newline>script``, but a quoted heredoc
    body keeps its bytes, so joining there could forge a terminator. A variant only adds findings."""
    joined = re.sub(r"\\\r?\n", "", text)
    found = _classify(text)
    return found | _classify(joined) if joined != text else found


def _classify(text: str) -> set[str]:
    """One reading of :func:`classify_osascript`. A heredoc body is consumed with its command."""
    from tools.approval_detection import _iter_shell_command_word_spans, _shell_command_segment

    found: set[str] = set()
    if not _mentions_osascript(text):
        return found
    body_span = (-1, -1)  # a heredoc body already read; command words inside it are data
    for start, end, word in _iter_shell_command_word_spans(text):
        if body_span[0] <= start < body_span[1]:
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
        program, opaque, jxa, heredoc = _osascript_program(args)
        if heredoc is not None:
            delim, quoted, strip_tabs = heredoc
            body, body_span = _heredoc_body(text, start + len(segment), delim, strip_tabs)
            opaque = opaque or (not quoted and _expands(body))  # an unquoted delimiter expands the body
            program += "\n" + body
        if opaque or _expands(args):
            found.add(OSASCRIPT_OPAQUE_DESCRIPTION)
        if jxa or _RISKY_RE.search(program):
            found.add(OSASCRIPT_RISKY_DESCRIPTION)
    return found


def _mentions_osascript(text: str) -> bool:
    """Whether ``osascript`` appears once quote and escape characters are dropped (``osa''script``)."""
    return "osascript" in re.sub(r"[\"'\\\\]", "", text).lower()


def _runs_its_arguments(name: str, args: str) -> bool:
    """Whether the command *name* executes a program named in *args* (a carrier)."""
    from tools.approval_detection import _interpreter_family

    if name == "find":
        return bool(re.search(r"(?:^|\s)-(?:exec|execdir|ok|okdir)\b", args))
    if name == "env":
        return bool(re.search(r"(?:^|\s)(?:-\w*S|--split-string)", args))
    return name in _CARRIERS or _interpreter_family(name) is not None


def _expands(raw: str) -> bool:
    """True when *raw* holds ``$`` or a backtick the shell expands (anything outside single quotes)."""
    from tools.approval_detection import _scan_shell
    return any(kind != "quote" and quote != "'" and raw[i] in "$`" for kind, i, _, quote in _scan_shell(raw))


def _osascript_program(args: str) -> tuple[str, bool, bool, tuple[str, bool, bool] | None]:
    """``(program_text, opaque, jxa, heredoc)`` for one osascript argv tail, read from the
    shell-decoded words (so ``'-l' JavaScript`` and ``-e'...'`` count like their plain spellings).

    ``program_text`` joins every ``-e`` argument and here-string. ``opaque`` is True when part of the
    program is not on this command line: a script-file operand, ``-`` / no ``-e`` (stdin), a ``<``
    redirect, ``-i`` (interactive), a heredoc whose delimiter cannot be read, or unparseable quoting.
    ``jxa`` is True for any ``-l`` language other than AppleScript. ``heredoc`` is
    ``(delimiter, quoted, strip_tabs)``."""
    from tools.approval_detection import _shell_segment_tokens

    tokens = _shell_segment_tokens(args, 0)
    if tokens is None:
        return "", True, False, None
    pieces: list[str] = []
    heredoc = None
    jxa = opaque = False
    i = 0
    while i < len(tokens):
        token = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if not token.strip():
            # An escaped newline (`osascript \\`) is a lone token here. Option values were consumed
            # with their option below, so a real `-e ' '` argument never reaches this skip.
            i += 1
            continue
        if token in (">", ">>", ">|", ">&") or (token.isdigit() and nxt.startswith(">")):
            i += 1 if token.isdigit() else 2
            continue
        if token == "<<<":
            pieces.append(nxt)
            i += 2
            continue
        if token == "<<":
            if heredoc is not None:
                opaque = True  # two bodies: fail closed
            heredoc = _heredoc_operator(args)
            if heredoc is None:
                opaque = True  # a delimiter this reader cannot spell
            # shlex splits `<<-EOF` into `<<`, `-EOF` and `<<- EOF` into `<<`, `-`, `EOF`.
            i += 3 if nxt == "-" else 2
            continue
        if token in ("<", "<>"):
            return " ".join(pieces), True, jxa, heredoc
        if token in ("-e", "-l", "-s") or (token.startswith("-l") and len(token) > 2):
            if token == "-e":
                pieces.append(nxt)
            language = nxt if token == "-l" else token[2:] if token.startswith("-l") else ""
            jxa = jxa or bool(language and language.lower() != "applescript")
            i += 1 if len(token) > 2 else 2
            continue
        if token.startswith("-e") and len(token) > 2:
            pieces.append(token[2:])
        elif not token.startswith("-") and (pieces or heredoc is not None):
            break  # after -e or a heredoc, operands are the program's own argv
        elif token.startswith("-i") or token == "-" or not token.startswith("-"):
            # `-i` reads an interactive session; a first operand is a script file (`-` is stdin).
            return " ".join(pieces), True, jxa, heredoc
        i += 1
    if not pieces and heredoc is None:
        opaque = True  # no -e, no heredoc: the program is read from stdin
    return "\n".join(pieces), opaque, jxa, heredoc


def _heredoc_operator(args: str) -> tuple[str, bool, bool] | None:
    """``(delimiter, quoted, strip_tabs)`` of the first unquoted ``<<`` here-document operator in
    *args*, read from the raw text (shlex cannot tell ``<<-EOF`` from ``<< -EOF``); None when its
    delimiter is not a plain word."""
    from tools.approval_detection import _scan_shell

    for kind, i, _, quote in _scan_shell(args):
        if kind == "char" and quote is None and args.startswith("<<", i) and not args.startswith("<<<", i) \
                and (i == 0 or args[i - 1] != "<"):
            match = _HEREDOC_RE.match(args, i)
            if match is None:
                return None
            return match.group("delim"), bool(match.group("q")), bool(match.group("dash"))
    return None


def _heredoc_body(text: str, after: int, delim: str, strip_tabs: bool) -> tuple[str, tuple[int, int]]:
    """``(body, (body_start, body_end))`` of the heredoc that starts on the line after *after*.
    The body ends at a line that is exactly *delim* (leading tabs allowed only for ``<<-``), as
    the shell reads it; an unterminated body runs to the end of *text*."""
    newline = text.find("\n", after)
    if newline < 0:
        return "", (len(text), len(text))
    tabs = r"\t*" if strip_tabs else ""
    end = re.compile(rf"^{tabs}{re.escape(delim)}$", re.MULTILINE).search(text, newline + 1)
    if end is None:
        return text[newline + 1:], (newline + 1, len(text))
    return text[newline + 1:end.start()], (newline + 1, end.end())
