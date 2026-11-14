"""Work bounds for :mod:`tools.approval_detection`'s shell scanners, and the budgeted lexer itself.

Two limits:

- the real ``$(`` nesting of the command as its author quoted it (``substitution_depth``), a
  hardline ``command parser limit exceeded`` refusal, and
- the steps the shell scanners take during one detection call (``parser_budget``), which ends the
  scan unfinished; the approval gates then ask for a one-shot approval (``record_scan``). Normalization
  strips escapes (``\\$\\(`` becomes ``$(``, ``\\"`` a bare quote), so the text the scanners walk
  can nest, or fail to close, far deeper than the shell ever would. Counting that nesting
  syntactically either hardline-blocks ordinary scripts (one ``\\"`` per line stacked a level) or
  misses spellings that stall; charging the scanners' actual work bounds the stall itself.

``_scan_shell`` (the one quote/escape state machine behind every scanner) and its substitution-end
finders live here because they are where that work is charged.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import re
import threading

from tools.approval_ansi_c import _is_shell_comment_start

# Scanner steps allowed per detection call: one per character lexed, or handed to shlex by a word
# pass; ~3.5M steps run per second. A fixed ~1 s plus a linear allowance per command char: ordinary
# scripts cost under ~10 steps per char (a 90 KB loop script: ~0.8M), and even quote shapes whose
# normalized text never closes a `$(` (one `\"` per line) stay inside it for ~300 lines. Only the
# super-linear rescans the #7485 stalls need (thousands of steps per char) run it out.
MAX_SCAN_UNITS, SCAN_UNITS_PER_CHAR = 4_000_000, 16

_active: contextvars.ContextVar[_Budget | None] = contextvars.ContextVar("approval_parser_budget", default=None)


class ParserBudgetExceeded(Exception):
    """One detection call spent more than its scanner budget."""


class _Budget:
    __slots__ = ("memo", "remaining")

    def __init__(self, limit: int) -> None:
        self.remaining = limit
        # (fn, *args) -> result of a pure scanner call: every command-start and word pass re-asks
        # for the same substitution ends and command words, and only a fresh call is real work.
        self.memo: dict[tuple, object] = {}

    def spend(self, units: int) -> None:
        self.remaining -= units
        if self.remaining < 0:
            raise ParserBudgetExceeded


@contextlib.contextmanager
def parser_budget(command: str):
    """Bound the scanner work of one detection call on *command*; a nested call shares the outer budget."""
    if _active.get() is not None:
        yield
        return
    token = _active.set(_Budget(MAX_SCAN_UNITS + SCAN_UNITS_PER_CHAR * len(command)))
    try:
        yield
    finally:
        _active.reset(token)


def active() -> _Budget | None:
    return _active.get()


def charge(units: int) -> None:
    """Spend *units* of the active budget (no-op outside one); raises once it is spent."""
    budget = _active.get()
    if budget is not None:
        budget.spend(units)


def memo(fn, *args):
    """``fn(*args)`` for a pure *fn*, memoized for the active detection call (a plain call outside
    one). A repeat costs one unit; the work of a fresh call is charged by the scanners it runs."""
    budget = _active.get()
    if budget is None:
        return fn(*args)
    key = (fn, *args)
    if key not in budget.memo:
        budget.memo[key] = fn(*args)
    budget.spend(1)
    return budget.memo[key]


# Whether the last hardline scan of a command finished, so the dangerous scan and the approval
# gate that follow it in the same guard pass need not repeat that (up to ~1.5 s) scan.
_scans: dict[bytes, bool] = {}
_scans_lock = threading.Lock()
_MAX_RECORDED_SCANS = 256


def _scan_key(command: str) -> bytes:
    return hashlib.sha256(command.encode("utf-8", "surrogatepass")).digest()


def record_scan(command: str, complete: bool) -> None:
    """Remember whether the hardline scan of *command* read all of it."""
    with _scans_lock:
        _scans.pop(_scan_key(command), None)
        _scans[_scan_key(command)] = complete
        while len(_scans) > _MAX_RECORDED_SCANS:
            del _scans[next(iter(_scans))]


def recorded_scan(command: str) -> bool | None:
    """What :func:`record_scan` last stored for *command*, or None."""
    with _scans_lock:
        return _scans.get(_scan_key(command))


_HEREDOC_RE = re.compile(r"""<<(-?)[ \t]*((?:'[^'\n]*'|"[^"\n]*"|\\.|[^\s;&|<>()'"\\])+)""")


def _quoted_heredoc_body_spans(
    command: str, newline: int, pending: list[tuple[str, bool, bool]]
) -> list[tuple[int, int]]:
    """Bodies of quoted-delimiter heredocs, even after earlier unquoted bodies.

    Heredoc bodies follow their redirections in order. Only quoted ones are skipped by the
    substitution-depth walk: unquoted bodies still expand command substitutions. In an unquoted
    body bash joins a backslash-newline before it compares a line with the delimiter, so `a\\`
    then `D` is the body line `aD`, not the end of a `<<D` body: matching physical lines would
    end that body early and skip the rest of it, expanding nesting included, as a quoted one.
    """
    spans: list[tuple[int, int]] = []
    for delimiter, strip_tabs, quoted in pending:
        body_start = newline + 1
        line = ""
        while newline < len(command):
            end = command.find("\n", newline + 1)
            end = len(command) if end < 0 else end
            line += command[newline + 1:end]
            newline = end
            if not quoted and end < len(command) and (len(line) - len(line.rstrip("\\"))) % 2:
                line = line[:-1]  # a continuation: the next physical line belongs to this one
                continue
            if (line.lstrip("\t") if strip_tabs else line) == delimiter:
                break
            line = ""
        if quoted:
            spans.append((body_start, newline))
        if newline == len(command):
            break
    return spans


def substitution_depth(command: str) -> int:
    """Deepest ``$(`` nesting under shell quoting of the raw command: a quoted ``)`` or ``$(``
    is data, ``\\"`` inside double quotes is one literal quote, and double quotes nest per level.
    A single-quoted region is literal at its level but is also measured as its own script on top
    of the current depth (a ``-c`` payload runs it); its quote state never leaks out. A comment and
    the body of a quoted-delimiter heredoc (``<<'E'``, ``<<"E"``, ``<<\\E``) are data too; an
    unquoted-delimiter body expands ``$(`` and is counted."""
    frames = [[None, 0, False]]  # per open `$(`: [quote state, unquoted `(` count, is `$((`]
    heredocs: list[tuple[str, bool, bool]] = []  # delimiter, strip-tabs flag, quoted flag
    skip_spans: list[tuple[int, int]] = []
    deepest, i, n = 0, 0, len(command)
    while i < n:
        if skip_spans and i == skip_spans[0][0]:
            i = skip_spans.pop(0)[1]
            continue
        char, frame = command[i], frames[-1]
        # A `#` glued to a `)` that closes `$(`/`$((` is inside the same word, so it is literal and
        # what follows runs. Not telling that `)` from a subshell or case `)` here only over-counts.
        if (char == "#" and frame[0] is None and command[i - 1:i] != ")"
                and _is_shell_comment_start(command, i)):
            i = command.find("\n", i)
            if i < 0:
                break
            continue
        if command.startswith("<<<", i):  # a here-string, not a heredoc
            i += 3
            continue
        # `$((1<<'E'))` is a shift. A misread elsewhere hides nesting only from this count; the scan
        # budget still bounds what the scanners then walk.
        if char == "<" and frame[0] is None and not frame[2] and (match := _HEREDOC_RE.match(command, i)):
            word = match[2]
            quoted = any(c in word for c in "'\"\\")
            delimiter = re.sub(r"\\(.)", r"\1", word).replace("'", "").replace('"', "")
            heredocs.append((delimiter, bool(match[1]), quoted))
            i = match.end()
            continue
        if char == "\n" and frame[0] is None and heredocs:
            # A quoted body stays inert even if an earlier heredoc delimiter was unquoted.
            skip_spans.extend(_quoted_heredoc_body_spans(command, i, heredocs))
            heredocs.clear()
        if char == "'" and frame[0] is None:
            end = command.find("'", i + 1)
            end = n if end < 0 else end
            deepest = max(deepest, len(frames) - 1 + substitution_depth(command[i + 1:end]))
            i = end
        elif char == "\\":
            i += 1
        elif char == "$" and command.startswith("(", i + 1):
            frames.append([None, 0, command.startswith("((", i + 1)])
            deepest = max(deepest, len(frames) - 1)
            i += 1
        elif frame[0] == '"':
            frame[0] = None if char == '"' else '"'
        elif char == '"':
            frame[0] = char
        elif char == "(":
            frame[1] += 1
        elif char == ")":
            if frame[1]:
                frame[1] -= 1
            elif len(frames) > 1:
                frames.pop()
        i += 1
    return deepest


def _scan_shell(text: str, start: int = 0, end: int | None = None, *, subst: str = "",
                brace: bool = False, stop_unterminated: bool = False, naive_backtick: bool = False,
                comments: bool = False):
    """Yield ``(kind, i, j, quote)`` lexical steps over ``text[start:end]`` without expanding.

    The single quote/escape state machine behind every detection scanner. ``kind`` is ``"char"``
    (one char), ``"esc"`` (backslash + the char it escapes; never inside single quotes), ``"quote"``
    (an opening/closing quote char) or ``"subst"`` (a ``$(...)`` / backtick / ``${...}`` span);
    With *comments*, ``"comment"`` spans are skipped without interpreting their quote syntax.
    ``quote`` is the state the step was read in (``None``, ``'`` or ``"``). Substitutions are
    recognized unquoted when ``"u"`` is in *subst*, inside double quotes when ``"q"`` is; *brace*
    adds unquoted ``${...}``. An unterminated substitution falls through as plain chars unless
    *stop_unterminated*, which yields ``("subst", i, None, quote)`` and ends the scan (the caller
    descends to *end* itself). *naive_backtick* closes a backtick at the next backtick even if
    escaped (the quoted-prose masker's historical behavior)."""
    n = len(text) if end is None else end
    quote: str | None = None
    i = start
    budget = active()
    while i < n:
        if budget is not None:  # one unit per lexed char; inline, this loop is the hot path
            budget.remaining -= 1
            if budget.remaining < 0:
                raise ParserBudgetExceeded
        ch = text[i]
        kind, j = "char", i + 1
        if comments and quote is None and _is_shell_comment_start(text, i):
            kind, j = "comment", text.find("\n", i, n)
            if j < 0:
                j = n
        elif quote != "'" and ch == "\\" and i + 1 < n:
            kind, j = "esc", i + 2
        elif ch == quote or (quote is None and ch in "'\""):
            kind = "quote"
        elif quote != "'" and ("q" if quote else "u") in subst and (
            ch == "`" or text.startswith("$(", i) or (brace and not quote and text.startswith("${", i))
        ):
            if ch == "`":
                close = text.find("`", i + 1) + 1 or None if naive_backtick else _scan_backtick_end(text, i)
            elif text[i + 1] == "(":
                close = _scan_dollar_paren_end(text, i)
            else:
                close = text.find("}", i + 2) + 1 or None
            if close is not None:
                kind, j = "subst", close
            elif stop_unterminated:
                yield ("subst", i, None, quote)
                return
        yield (kind, i, j, quote)
        if kind == "quote":
            quote = None if quote else ch
        i = j


def _scan_dollar_paren_end(command: str, start: int) -> int | None:
    """Return the offset after a balanced ``$(...)`` command substitution."""
    return memo(_dollar_paren_end, command, start)


def _dollar_paren_end(command: str, start: int) -> int | None:
    depth = 1
    for kind, i, _, quote in _scan_shell(command, start + 2):
        if kind == "char" and not quote:
            depth += command.startswith("$(", i) - (command[i] == ")")
            if depth == 0:
                return i + 1
    return None


def _scan_backtick_end(command: str, start: int) -> int | None:
    return memo(_backtick_end, command, start)


def _backtick_end(command: str, start: int) -> int | None:
    # Backticks have no quote awareness: only a backslash escapes the next char.
    match = re.compile(r"(?:\\.|[^`\\])*`", re.DOTALL).match(command, start + 1)
    charge((match.end() if match else len(command)) - start)
    return match.end() if match else None
