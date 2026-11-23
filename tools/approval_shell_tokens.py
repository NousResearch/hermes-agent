"""Span-keeping shell word lexer for the approval detector's quoted-grep-operand handling, and the command-word
gates of the rules whose stdlib spelling read a lookaround right after the command word.

The lexer moved verbatim from :mod:`tools.approval_detection` (size ratchet); see `_quoted_grep_pattern_spans` there.
"""

import hermes_re as re

from tools.approval_hermes_store import EndGatedMatcher, _NextStart
from tools.approval_parser_budget import _scan_shell
from tools.approval_windows_hardline import _first_from


def _backtick_end_from(segment: str, i: int) -> int | None:
    """Index of the backtick closing the one opened at ``i``, or None when none follows."""
    j = segment.find("`", i + 1)
    while j != -1 and segment[j - 1] == "\\":
        j = segment.find("`", j + 1)
    return None if j == -1 else j


def _shell_tokens_with_spans(segment: str, start: int):
    """Return shell words as ``(value, start, end, quoted)`` or ``None`` on malformed quoting.
    Deliberately small lexer that never expands shell syntax; it exists to keep source spans (which
    ``shlex`` does not expose) for deciding which quoted grep operand is data, not another command.

    Lexing stops at the end of the simple command that begins at *start*: an unquoted ``;``, ``|``,
    ``&`` or newline, or the ``)`` / backtick that closes the substitution the command sits inside.
    Without that, a grep nested as ``"$(grep … | cut …)"`` was lexed together with the enclosing
    command's closing quote, read as unbalanced quoting, and reported as a hardline block (546
    blocked turns in one run, every one a false positive; ``sed -n "$(grep -n X f | cut -d: -f1),+3p" f``
    is the canonical shape)."""
    tokens, value, token_start, quote = [], [], None, None
    depth = 0  # $(...) nesting opened AFTER start; a closer at depth 0 ends the enclosing substitution
    # A backtick opened AFTER start is an operand substitution (``grep -e `cmd` f``); the matching
    # closer belongs to it, not to an enclosing backtick the command might sit inside.
    in_backtick = False

    def flush(end: int) -> None:
        raw = segment[token_start:end]
        # Only a wholly single-quoted operand is inert shell data. Double quotes still execute $()
        # and backticks; unquoted substitutions do too.
        inert = (raw.startswith("'") and raw.endswith("'")) or ("='" in raw and raw.endswith("'"))
        tokens.append(("".join(value), token_start, end, inert))

    end_at = len(segment)
    for kind, i, _, _ in _scan_shell(segment, start):
        ch = segment[i]
        if kind == "char" and not quote:
            if ch.isspace() and ch != "\n":
                if token_start is not None:
                    flush(i)
                    value, token_start = [], None
                continue
            if segment.startswith("$(", i):
                depth += 1
            elif ch == "`":
                if in_backtick:
                    in_backtick = False
                elif depth == 0 and _backtick_end_from(segment, i) is None:
                    end_at = i  # unmatched: it closes the substitution this command sits inside
                    break
                else:
                    in_backtick = True
            elif ch == ")":
                if depth == 0:
                    end_at = i
                    break
                depth -= 1
            elif ch in ";|&\n":
                end_at = i
                break
        if token_start is None:
            token_start = i
        if kind == "quote":
            quote = None if quote else ch
        elif kind == "esc":
            value.append(segment[i + 1])
        elif ch == "\\" and not quote:
            return None  # dangling backslash
        else:
            value.append(ch)
    if quote:
        return None
    if token_start is not None:
        flush(end_at)
    return tokens


# Command-word and argument-span gates for the rules whose stdlib spelling holds a lookaround right after the command
# word (EndGatedMatcher marks the command-word ends that pass, so the rule head is matched as it was).
_FLAGS = re.IGNORECASE | re.DOTALL
_WORD_CHAR_RE = re.compile(r'\w')
_WORD_STOP_RE = re.compile(r'[\s/)=<>]')
_EXE_WORD_RE = re.compile(r'exe\b', _FLAGS)


def _command_word_gate(text: str):
    """The stdlib `_CMD_PATH` was possessive: a verb ends its word (a word boundary) and the word does not run on into a `/`
    (in `./mkfs-x/run.sh` the verb is a directory of a longer program path)."""
    stop = _NextStart(lambda t, p: (m.start() if (m := _WORD_STOP_RE.search(t, p)) else None), text)

    def passes(end: int) -> bool:
        if end < len(text) and _WORD_CHAR_RE.match(text, end):
            return False
        at = stop.at(end)
        return at == len(text) or text[at] != '/'
    return passes


def _shutdown_word_gate(text: str):
    """`_command_word_gate` plus the shutdown rule's lookahead: no `-` after the verb, and a `.` only as `.exe`
    (`reboot-notify` / `halt.sh` are other names)."""
    word_ends = _command_word_gate(text)
    return lambda end: word_ends(end) and not text.startswith('-', end) and not (
        text.startswith('.', end) and _EXE_WORD_RE.match(text, end + 1) is None)


# nc/socat: the command word ends at a blank, `<`, `>` or `&>`, then the arguments run on through any character but a
# newline, `;`, `|`, a `#` after a blank (a comment) or a background `&` (one in `2>&1`, `<&0`, `&>x` stays).
_NC_WORD_END_RE = re.compile(r'[\s<>]|&>')
_NC_SPAN_BREAK_RE = re.compile(r'[\n;|]|(?<=\s)#|(?<![<>])&(?!>)')
_NC_EXEC_FLAG_RE = re.compile(r'\s["\']?(?:(?-i:-[a-zA-Z]*[ec])\b|--(?:sh-)?exec\b)', _FLAGS)
_SOCAT_EXEC_ADDRESS_RE = re.compile(r'\b(?:exec|system):', _FLAGS)
_NC_NAME = r'(?:netcat|ncat|nc)(?:\.\w+)?'
_SOCAT_NAME = r'socat(?:\.\w+)?'


def _reverse_shell_gate(exec_re):
    """Ends where the argument span reaches *exec_re* (nc's blank + `-e`, socat's `EXEC:`) before it breaks: an exec
    hit at or after the end, no later than the first break (nc's blank may be the newline that breaks the span)."""
    def gate(text: str):
        found = [m.start() for m in exec_re.finditer(text)]
        breaks = [m.start() for m in _NC_SPAN_BREAK_RE.finditer(text)] if found else []
        # A missing hit (len + 1) must lose to a missing break (len): the hits of other commands are found too.
        return lambda end: _NC_WORD_END_RE.match(text, end) is not None and (
            _first_from(found, end, len(text) + 1) <= _first_from(breaks, end, len(text)))
    return gate


def verb_matcher(cmdpos: str, verbs: str, shutdown: bool = False) -> EndGatedMatcher:
    """``cmdpos`` + one of ``verbs`` as a whole command word (the shutdown rule's word-end test with ``shutdown``)."""
    return EndGatedMatcher(cmdpos + verbs, r'\b' + verbs, _shutdown_word_gate if shutdown else _command_word_gate)


_REVERSE_SHELLS = {"nc": (_NC_NAME, _NC_EXEC_FLAG_RE), "socat": (_SOCAT_NAME, _SOCAT_EXEC_ADDRESS_RE)}


def reverse_shell_matcher(cmdpos: str, tool: str) -> EndGatedMatcher:
    """``cmdpos`` + nc/socat (path-qualified or not) whose argument span reaches its exec flag/address."""
    name, exec_re = _REVERSE_SHELLS[tool]
    return EndGatedMatcher(cmdpos + r'(?:\S*/)?' + name, name, _reverse_shell_gate(exec_re))
