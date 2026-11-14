"""Work bounds for :mod:`tools.approval_detection`'s shell scanners.

Two limits, both failing closed as ``command parser limit exceeded``:

- the real ``$(`` nesting of the command as its author quoted it (``substitution_depth``), and
- the steps the shell scanners take during one detection call (``parser_budget``). Normalization
  strips escapes (``\\$\\(`` becomes ``$(``, ``\\"`` a bare quote), so the text the scanners walk
  can nest, or fail to close, far deeper than the shell ever would. Counting that nesting
  syntactically either hardline-blocks ordinary scripts (one ``\\"`` per line stacked a level) or
  misses spellings that stall; charging the scanners' actual work bounds the stall itself.
"""

from __future__ import annotations

import contextlib
import contextvars

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


def substitution_depth(command: str) -> int:
    """Deepest ``$(`` nesting under shell quoting of the raw command: a quoted ``)`` or ``$(``
    is data, ``\\"`` inside double quotes is one literal quote, and double quotes nest per level.
    A single-quoted region is literal at its level but is also measured as its own script on top
    of the current depth (a ``-c`` payload runs it); its quote state never leaks out."""
    frames = [[None, 0]]  # per open `$(`: [quote state, unquoted `(` count]
    deepest, i, n = 0, 0, len(command)
    while i < n:
        char, frame = command[i], frames[-1]
        if char == "'" and frame[0] is None:
            end = command.find("'", i + 1)
            end = n if end < 0 else end
            deepest = max(deepest, len(frames) - 1 + substitution_depth(command[i + 1:end]))
            i = end
        elif char == "\\":
            i += 1
        elif char == "$" and command.startswith("(", i + 1):
            frames.append([None, 0])
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
