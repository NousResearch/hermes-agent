"""Same-command shell variable indirection for the detectors. Detection only: nothing is executed.

``X="rm -rf /home"; $X`` runs ``rm -rf /home``, but no pattern sees it: the literal command sits
inside a quoted assignment value (data) and the command word is ``$X``. That exact spelling got past
the hardline floor, and under ``approvals.single_query_mode: approve`` it wiped a home directory.
This module resolves ``NAME=value`` assignments (and ``for``/``select`` loop bindings) made in the
same command and substitutes them back, so the detectors see the command the shell would run. It
also names every command word that is still an unresolved expansion, and every place where the
inspection itself was incomplete, which unattended auto-approve paths use to refuse commands they
cannot read.

Resolution is per use, not per name. Each reference takes the value(s) the name holds at that
point in the text, and an assignment's own value is resolved when it is made. So a later
reassignment never erases an earlier destructive use: ``X="rm -rf /home"; $X; X=echo`` still shows
``rm -rf /home``, and ``Y=$X`` copies X's value at that moment.

A name can hold several possible values (loop words, conditional assignments). Values are combined
as a real cross product over the distinct choices, never zipped: ``for X in rm echo; do for Y in
./build /home; do $X -rf $Y`` yields ``rm -rf /home``. When the whole-command product is larger than
_MAX_VARIANTS, every SIMPLE command is still examined under every combination of its own choices.
Any bound that stops enumeration marks the resolution incomplete, and incomplete never counts as
resolved (``uninspectable_reasons``).

A binding is PROOF of a value only when it dominates the use (``approval_detection_shell_scope``):
it always runs, in the same shell, before the use, and is not a prefix-only temporary
(``X=echo true``). Any other binding (conditional, subshell, pipeline, function or loop body, a
heredoc line) only ADDS a possible value. A name no dominating binding fixes also keeps the
environment's value, the explicit ``UNKNOWN``. A builtin that writes a name the text does not show
(``read``, ``printf -v``, ``mapfile``, ``eval``, a nameref, however spelled; see
``approval_detection_clobbers``) makes that name UNKNOWN. A reader reached through a variable
(``R=read; $R X``) is classified by what the variable holds, and an unreadable one clobbers every
name. Writes that can happen LATER than where they are written (a function body, a ``trap``
handler, a ``mapfile -C`` callback, an alias, a nameref) are never overridden by an intervening
assignment: a write inside a function body runs at the CALL, which may follow any later assignment.
Redirections are dropped wherever they sit before a command is classified (``< f read X``). A value computed by a
command substitution (``X=$(cat f)``) is UNKNOWN too: the command's output is not in the text.

For detection variants this over-approximates: a wrong guess only adds a variant, and a variant can
only add blocks. For the opaque-leader classifier, UNKNOWN among a command word's values means the
command cannot be read.
"""

import functools
import itertools
import math
import os
import re
from dataclasses import dataclass

from tools.approval_detection_shell_scope import ShellScope

# The environment's (or any unreadable) value of a name. Substituted back as the reference text.
UNKNOWN = "\x00<unknown>"
# NAME=value / NAME+=value as a whole shell word (the name part is never quoted in real shell).
_ASSIGNMENT_WORD_RE = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?P<append>\+?)=(?P<value>.*)", re.DOTALL)
# NAME[subscript]=value / NAME[subscript]+=value: one array element.
_ELEMENT_ASSIGNMENT_RE = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\[[^\]]*\]\+?=(?P<value>.*)", re.DOTALL)
# Names the shell itself rewrites as commands run (`$_` is the previous command's last argument).
# A same-command assignment never proves their value.
_SHELL_MANAGED_NAMES = frozenset({
    "_", "BASH_REMATCH", "PIPESTATUS", "RANDOM", "SRANDOM", "LINENO", "SECONDS", "EPOCHSECONDS",
    "EPOCHREALTIME", "BASHPID", "BASH_COMMAND", "FUNCNAME", "PWD", "OLDPWD", "COPROC", "BASH_ARGV",
    "BASH_ARGC", "BASH_LINENO", "BASH_SOURCE", "DIRSTACK", "GROUPS", "HISTCMD", "OPTARG", "OPTIND",
})
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Declaration builtins whose operands are assignments (`export X="rm -rf /"`).
_DECLARATION_BUILTINS = frozenset({"export", "declare", "typeset", "local", "readonly"})
# Words that run the builtin named next in the current shell (`builtin declare X=v`).
_BUILTIN_WRAPPERS = frozenset({"builtin", "command"})
# Builtins whose writes can land at any LATER point: a trap handler, a nameref alias.
_LATE_WRITERS = frozenset({"trap", "declare", "typeset", "local"})
# `${NAME=word}` / `${NAME:=word}`: assigns *word* when NAME is unset (or empty).
_DEFAULT_ASSIGN_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):?=([^}]*)\}")
# Loop keywords whose NAME takes each listed word in turn.
_LOOP_BINDERS = frozenset({"for", "select"})
# $NAME, ${NAME}, ${NAME[...]}, ${NAME:-word} / ${NAME-word} / ${NAME:+word} / ${NAME:=word}.
_REFERENCE_RE = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)(?:\[[^\]}]*\])?(?:(?P<op>:?[-=?+])(?P<word>[^}]*))?\}"
    r"|(?P<bare>[A-Za-z_][A-Za-z0-9_]*))"
)
# A command word that is one variable reference or one command substitution, optionally double-quoted.
_OPAQUE_LEADER_RE = re.compile(
    r'"?(?:\$(?:\{[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]}]*\])?(?::?[-=?+][^}]*)?\}|[A-Za-z_][A-Za-z0-9_]*)'
    r'|\$\(.*\)|`.*`)"?',
    re.DOTALL,
)
# A for-loop word the resolver cannot enumerate (glob, expansion, substitution).
_UNENUMERABLE_WORD_RE = re.compile(r"[$`*?\[]")
# An assignment value computed by a command: its output is not in the text.
_COMMAND_OUTPUT_RE = re.compile(r"\$\((?!\()|`")
_CANDIDATE_ASSIGNMENT_RE = re.compile(
    r"(?<![\w$/.-])([A-Za-z_][A-Za-z0-9_]*)\+?=|\b(?:for|select)\s+([A-Za-z_][A-Za-z0-9_]*)\s")
_CANDIDATE_REFERENCE_RE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)")
# Bounds. Whole-command variants are capped (they cost a full detection pass each). Past that cap,
# each simple command gets its own variants (up to _MAX_SEGMENT_COMBOS combinations each and
# _MAX_SEGMENT_VARIANTS in total). Hitting ANY bound marks the resolution incomplete; values past
# _MAX_VALUES_PER_NAME / _MAX_LOOP_WORDS become UNKNOWN and also mark it incomplete.
_MAX_VALUES_PER_NAME = 64
_MAX_VARIANTS = 8
_MAX_SEGMENT_COMBOS = 64
_MAX_SEGMENT_VARIANTS = 512
_MAX_LOOP_WORDS = 64
_MAX_PAYLOAD_DEPTH = 3
# Passes of the dynamic-command-word fixpoint (``_Resolution._solve_dynamic_reach``).
_MAX_REACH_PASSES = 4


@dataclass(frozen=True)
class _Binding:
    """One assignment, loop binding or name-clobbering builtin. Its command starts at *start*; it
    takes effect at *end*. *name* None means every name (``eval``). *truncated*: a bound dropped
    values (they are UNKNOWN here)."""
    start: int
    end: int
    name: str | None
    raw_values: tuple[str, ...]
    append: bool = False
    persistent: bool = True
    truncated: bool = False
    command_output: bool = False
    # A command whose word is an expansion (`$R X`): its raw words from that word on. What it
    # clobbers is decided once the word's values are known (``_Resolution.clobbered``).
    dynamic: tuple[str, ...] | None = None
    # Can take effect at any later point, so no later assignment overrides it.
    late: bool = False


def _array_value(command: str, open_paren: int) -> tuple[str, int] | None:
    """``NAME=(a b c)``: the joined element text and the offset past ``)``, or None if unclosed."""
    from tools.approval_detection import _read_shell_word, _skip_shell_whitespace, _deobfuscate_shell_word_for_detection
    elements, pos = [], open_paren + 1
    while True:
        pos = _skip_shell_whitespace(command, pos)
        if pos >= len(command):
            return None
        if command[pos] == ")":
            return (" ".join(elements), pos + 1)
        start, end, word = _read_shell_word(command, pos)
        if start == end:
            return None
        elements.append(_deobfuscate_shell_word_for_detection(word))
        pos = end


def _for_loop_binding(command: str, for_start: int, pos: int) -> _Binding | None:
    """``for NAME in w1 w2 ...`` (*pos* just past ``for``/``select``): NAME bound to each literal
    word. A word no detector can enumerate adds UNKNOWN; more than _MAX_LOOP_WORDS words add UNKNOWN
    and mark the binding truncated."""
    from tools.approval_detection import _deobfuscate_shell_word_for_detection, _read_shell_word
    _, pos, name = _read_shell_word(command, pos)
    if not _NAME_RE.fullmatch(name):
        return None
    _, after, keyword = _read_shell_word(command, pos)
    if keyword != "in":
        # `for NAME; do` / `for NAME do` iterate the positional parameters: unknown.
        return _Binding(for_start, pos, name, (UNKNOWN,))
    pos = after
    words: list[str] = []
    truncated = False
    while pos < len(command):
        start, end, word = _read_shell_word(command, pos)
        if start == end or word == "do" or word.startswith(";"):
            break
        if len(words) >= _MAX_LOOP_WORDS:
            words.append(UNKNOWN)
            truncated = True
        elif _UNENUMERABLE_WORD_RE.search(word):
            words.append(UNKNOWN)
        else:
            words.append(_deobfuscate_shell_word_for_detection(word))
        pos = end
    return _Binding(for_start, pos, name, tuple(dict.fromkeys(words)) or (UNKNOWN,), truncated=truncated)


# A redirection at a word start: optional fd number or `{NAME}` (bash stores a new fd number in
# NAME), then the operator. Process substitutions `<(`/`>(` are words, not redirections.
_REDIRECTION_START_RE = re.compile(
    r"(?:[0-9]+|\{(?P<fdvar>[A-Za-z_][A-Za-z0-9_]*)\})?(?:&>>?|<<<|<<-?|<>|>>|>&|<&|>\||[<>])(?!\()")


_ARRAY_OPEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\+?=")


def _simple_command_spans(command: str, pos: int) -> tuple[list[tuple[int, int, str]], list[str]]:
    """The raw words of the simple command starting at *pos* as ``(start, end, word)``, with every
    redirection removed wherever it sits (``< in read X``, ``read X 2>/dev/null``), and the names
    ``{NAME}>f`` redirections assign. An unquoted newline ends the command."""
    from tools.approval_detection import _read_shell_word, _skip_shell_whitespace
    spans: list[tuple[int, int, str]] = []
    fd_vars: list[str] = []
    seen_any = False
    while pos < len(command):
        start = _skip_shell_whitespace(command, pos)
        if seen_any and "\n" in command[pos:start]:
            break
        redirect = _REDIRECTION_START_RE.match(command, start)
        if redirect:
            if redirect.group("fdvar"):
                fd_vars.append(redirect.group("fdvar"))
            target_start, target_end, _ = _read_shell_word(command, redirect.end())
            if target_start == target_end:
                break
            pos, seen_any = target_end, True
            continue
        start, end, word = _read_shell_word(command, start)
        if start == end:
            break
        if _ARRAY_OPEN_RE.fullmatch(word) and command.startswith("(", end):
            # `NAME=(a b c)` is one word to the shell; keep it whole (its value is re-read later).
            array = _array_value(command, end)
            if array is None:
                break
            end = array[1]
            word = command[start:end]
        spans.append((start, end, word))
        pos, seen_any = end, True
    return spans, fd_vars


def _simple_command_words(command: str, pos: int) -> list[str]:
    """The raw words of the simple command starting at *pos*, redirections removed."""
    return [word for _, _, word in _simple_command_spans(command, pos)[0]]


def _clobber_bindings(command: str, start: int, words: list[str]) -> list[_Binding]:
    """Name writes whose value the text does not show (``approval_detection_clobbers``). They take
    effect after the whole simple command (its arguments expand first). They never REPLACE a
    proven value, only add UNKNOWN to it, so a destructive value seen earlier still reaches the
    detectors."""
    from tools.approval_detection import _shell_command_segment
    from tools.approval_detection_clobbers import LATE, command_clobbers
    hit = command_clobbers(words)
    if hit == frozenset():
        return []
    end = start + len(_shell_command_segment(command, start))
    if isinstance(hit, tuple):
        return [_Binding(start, end, None, (UNKNOWN,), persistent=False, dynamic=tuple(words[hit[1]:]))]
    if hit is None or hit == LATE:
        return [_Binding(start, end, None, (UNKNOWN,), persistent=False, late=hit == LATE)]
    return [_Binding(start, end, name, (UNKNOWN,), persistent=False) for name in sorted(hit)]


def _default_assignment_bindings(command: str) -> list[_Binding]:
    """``${X:=word}`` / ``${X=word}`` may assign *word* to X (only when X is unset or empty)."""
    from tools.approval_detection import _deobfuscate_shell_word_for_detection, _scan_shell
    found = []
    for kind, i, _, quote in _scan_shell(command):
        if kind == "char" and quote != "'" and command.startswith("${", i):
            match = _DEFAULT_ASSIGN_RE.match(command, i)
            if match:
                found.append(_Binding(i, match.end(), match.group(1),
                                      (_deobfuscate_shell_word_for_detection(match.group(2)),),
                                      persistent=False))
    return found


def _collect_bindings(command: str) -> list[_Binding]:
    """Every assignment / loop binding / name clobber at a command position, in text order. Values
    are deobfuscated the way command words are (quotes and escapes removed, tiny literal
    substitutions folded), so ``X="rm -rf /home"`` binds X to ``rm -rf /home``."""
    from tools.approval_detection import (
        _deobfuscate_shell_word_for_detection, _iter_shell_command_starts, _shell_command_segment,
    )
    bindings: list[_Binding] = _default_assignment_bindings(command) if "${" in command else []
    for pos in _iter_shell_command_starts(command):
        spans, fd_vars = _simple_command_spans(command, pos)
        words = [word for _, _, word in spans]
        if words and words[0] not in _LOOP_BINDERS:
            bindings.extend(_clobber_bindings(command, pos, words))
        if fd_vars:
            # `exec {FD}<file`: bash stores the new descriptor number in FD.
            end = pos + len(_shell_command_segment(command, pos))
            bindings.extend(_Binding(pos, end, name, (UNKNOWN,), persistent=False) for name in fd_vars)
        first, declaration, wrapped = True, False, False
        pending: list[_Binding] = []
        for index, (start, end, word) in enumerate(spans):
            if first and not wrapped and word in _LOOP_BINDERS:
                loop = _for_loop_binding(command, start, end)
                if loop is not None:
                    if word == "select":
                        # select sets NAME from input (empty on a bad choice) and leaves it as it
                        # was on EOF: the words are possible values, never a proof.
                        loop = _Binding(loop.start, loop.end, loop.name, loop.raw_values + ("",),
                                        persistent=False, truncated=loop.truncated)
                        bindings.append(_Binding(start, loop.end, "REPLY", (UNKNOWN,), persistent=False))
                    bindings.append(loop)
                break
            plain = _deobfuscate_shell_word_for_detection(word) if first else word
            if first and plain in _BUILTIN_WRAPPERS:
                # `builtin declare X=v` runs the builtin in this shell.
                wrapped = True
                continue
            if first and wrapped and plain.startswith("-"):
                continue
            if first and plain in _DECLARATION_BUILTINS:
                declaration, first = True, False
                continue
            if wrapped and first:
                break   # `command ls ...`: an ordinary program
            first = False
            if declaration and word.startswith("-"):
                continue
            match = _ASSIGNMENT_WORD_RE.fullmatch(word)
            element = None if match else _ELEMENT_ASSIGNMENT_RE.fullmatch(word)
            if element:
                # `X[0]=v`: element 0 is what $X expands to; another index leaves it. Either way
                # v is a possible value, never a proof.
                pending.append(_Binding(start, end, element.group("name"),
                                        (_deobfuscate_shell_word_for_detection(element.group("value")),),
                                        persistent=False,
                                        command_output=bool(_COMMAND_OUTPUT_RE.search(element.group("value")))))
                continue
            if not match:
                # `X=v cmd ...`: the assignments only reach cmd's environment, not the shell.
                if not declaration:
                    pending = [_Binding(b.start, b.end, b.name, b.raw_values, b.append, False,
                                        b.truncated, b.command_output) for b in pending]
                break
            raw_value = match.group("value")
            open_paren = end - len(raw_value)
            if raw_value.startswith("(") and _ARRAY_OPEN_RE.fullmatch(command[start:open_paren]):
                value = _array_value(command, open_paren)[0]
            else:
                value = _deobfuscate_shell_word_for_detection(raw_value)
            pending.append(_Binding(start, end, match.group("name"), (value,), bool(match.group("append")),
                                    command_output=bool(_COMMAND_OUTPUT_RE.search(raw_value))))
        bindings.extend(pending)
    bindings.sort(key=lambda binding: binding.end)
    return bindings


def _reference_name(match) -> str:
    return match.group("braced") or match.group("bare")


def _reference_values(match, values: tuple[str, ...]) -> tuple[str, ...]:
    """What the reference *match* may expand to, given its name's possible *values*. UNKNOWN is
    kept (callers substitute the reference text for it); ``${X:-word}``-style words are added."""
    from tools.approval_detection import _deobfuscate_shell_word_for_detection
    if match.group("op") and match.group("op").lstrip(":") in "-=+":
        return tuple(dict.fromkeys(values + (_deobfuscate_shell_word_for_detection(match.group("word")),)))
    return values


# ---- expansion: references -> choices -> rendered variants -----------------------------------

@dataclass
class _Pieces:
    """*text* cut into literal runs and references. Each reference piece carries a choice key
    ``(name, values)``: references to one name with one value set hold the same value at run time
    (nothing assigns between them), so they share one choice. Different keys are independent."""
    text: str
    spans: list[tuple[int, int, tuple | None]]
    choices: dict[tuple, tuple[str, ...]]

    def render(self, pick: dict[tuple, str], lo: int = 0, hi: int | None = None) -> str:
        hi = len(self.text) if hi is None else hi
        return "".join(self.text[i:j] if key is None else pick[key]
                       for i, j, key in self.spans if lo <= i and j <= hi)

    def product(self, keys, limit: int) -> tuple[list[dict], bool]:
        """Up to *limit* picks over the full cross product of *keys*, and whether that was all."""
        keys = list(dict.fromkeys(keys))
        base = {key: values[0] for key, values in self.choices.items()}
        total = math.prod(len(self.choices[key]) for key in keys)
        picks = [{**base, **dict(zip(keys, combo))}
                 for combo in itertools.islice(itertools.product(*(self.choices[k] for k in keys)), limit)]
        return picks, total <= limit


def _reference_pieces(text: str, values_of, seps_at=None) -> _Pieces:
    """*values_of(name, offset)* returns the possible values, or None to leave the reference as is.
    *seps_at(offset)*: the non-blank IFS characters that may be in effect there. An UNQUOTED
    reference's value is word-split on them, so each such value also gets a variant with them as
    blanks (``IFS=+; X=rm+-rf+/home; $X`` runs ``rm -rf /home``)."""
    from tools.approval_detection import _scan_shell
    spans: list[tuple[int, int, tuple | None]] = []
    choices: dict[tuple, tuple[str, ...]] = {}
    skip_to = 0
    for kind, i, j, quote in _scan_shell(text):
        if i < skip_to:
            continue
        if kind == "char" and quote != "'" and text[i] == "$":
            match = _REFERENCE_RE.match(text, i)
            values = values_of(_reference_name(match), i) if match else None
            if match and values:
                values = tuple(dict.fromkeys(text[i:match.end()] if v == UNKNOWN else v
                                             for v in _reference_values(match, values)))
                seps = seps_at(i) if seps_at is not None and quote is None else ""
                if seps:
                    table = str.maketrans(seps, " " * len(seps))
                    values = tuple(dict.fromkeys(values + tuple(v.translate(table) for v in values)))
                key = (_reference_name(match), values)
                choices[key] = values
                spans.append((i, match.end(), key))
                skip_to = match.end()
                continue
        spans.append((i, j, None))
    return _Pieces(text, spans, choices)


def _expand_all(text: str, values_of, limit: int) -> tuple[list[str], bool]:
    """Every expansion of *text* (full cross product, up to *limit*) and whether that was all."""
    pieces = _reference_pieces(text, values_of)
    picks, complete = pieces.product(pieces.choices, limit)
    return list(dict.fromkeys(pieces.render(pick) for pick in picks)), complete


def _segment_end(text: str, start: int) -> int:
    """End of the simple command starting at *start* (the bound ``_shell_command_segment`` uses)."""
    from tools.approval_detection import _scan_shell
    for kind, i, _, quote in _scan_shell(text, start, subst="uq", brace=True, comments=True):
        if kind == "comment" or (kind == "char" and quote is None and text[i] in ";&|\n)`"):
            return i
    return len(text)


def _simple_command_range(text: str, starts: list[int], i: int, j: int) -> tuple[int, int]:
    for start in reversed(starts):
        if start <= i:
            end = _segment_end(text, start)
            if end >= j:
                return start, end
    return 0, len(text)


# Programs whose arguments are only printed. A combination of their argument values cannot run
# anything, so its simple command needs no cross product (each value is still seen at least once).
_PRINT_ONLY_PROGRAMS = frozenset({"echo", "printf"})
_REDIRECT_CHAR_RE = re.compile(r"[<>]")


def _inert_segment(text: str, lo: int, hi: int) -> bool:
    """The simple command ``text[lo:hi]`` is assignments only (their values are checked where they
    are USED, with their own completeness), or a print-only program whose output goes nowhere
    executable: not piped (``echo "$A $B" | sh`` runs it) and no redirection (``> $D$N``)."""
    from tools.approval_detection import _deobfuscate_shell_word_for_detection, _read_shell_word, _scan_shell
    pos = lo
    while True:
        start, end, word = _read_shell_word(text, pos)
        if start == end or start >= hi:
            return True     # assignments only
        if not _ASSIGNMENT_WORD_RE.fullmatch(word):
            break
        pos = end
    if _deobfuscate_shell_word_for_detection(word) not in _PRINT_ONLY_PROGRAMS:
        return False
    if text.startswith("|", hi) and not text.startswith("||", hi):
        return False
    if text.startswith("printf", start) and re.search(r"(?:^|\s)-v\b", text[start:hi]):
        return False
    return not any(kind == "char" and quote is None and _REDIRECT_CHAR_RE.match(text[i])
                   for kind, i, _, quote in _scan_shell(text, lo, hi))


def _expand_command(text: str, values_of, seps_at=None) -> tuple[list[str], bool]:
    """Detection variants of a whole command and whether they cover every combination.

    Up to _MAX_VARIANTS whole-command combinations cover the full product when it is that small.
    Otherwise the whole-command variants walk each choice's values in step (every value appears at
    least once), and each simple command holding a multi-valued reference is ALSO emitted alone
    under every combination of its own choices: a hardline pattern matches within one simple
    command, so that is the combination set that decides it."""
    from tools.approval_detection import _iter_shell_command_starts
    pieces = _reference_pieces(text, values_of, seps_at)
    multi = [key for key, values in pieces.choices.items() if len(values) > 1]
    picks, complete = pieces.product(multi, _MAX_VARIANTS)
    if complete:
        return list(dict.fromkeys(pieces.render(pick) for pick in picks)), True
    width = max(len(pieces.choices[key]) for key in multi)
    whole = [pieces.render({key: values[k % len(values)] for key, values in pieces.choices.items()})
             for k in range(min(width, _MAX_VARIANTS))]
    starts = sorted(_iter_shell_command_starts(text))
    segments: dict[tuple[int, int], list[tuple]] = {}
    for i, j, key in pieces.spans:
        if key is not None and len(pieces.choices[key]) > 1:
            segments.setdefault(_simple_command_range(text, starts, i, j), []).append(key)
    extra: list[str] = []
    complete = True
    for (lo, hi), keys in segments.items():
        if _inert_segment(text, lo, hi):
            continue
        seg_picks, seg_complete = pieces.product(keys, _MAX_SEGMENT_COMBOS)
        complete = complete and seg_complete
        extra.extend(pieces.render(pick, lo, hi) for pick in seg_picks)
        if len(extra) > _MAX_SEGMENT_VARIANTS:
            del extra[_MAX_SEGMENT_VARIANTS:]
            complete = False
            break
    return list(dict.fromkeys((*whole, *extra))), complete


# ---- per-use resolution -----------------------------------------------------------------------

class _Resolution:
    """Per-use values of every name in one command. *truncated* records that some value set used
    so far hit a bound (its dropped values are UNKNOWN)."""

    def __init__(self, command: str):
        self.bindings = _collect_bindings(command)
        self.scope = ShellScope(command)
        self.truncated = False
        self._memo: dict[int, tuple[str, ...]] = {}
        self._busy: set[int] = set()
        self._reach: dict[int, "frozenset[str] | str | None"] = {}
        self._solve_dynamic_reach()

    def _solve_dynamic_reach(self) -> None:
        """What each command whose command word is an expansion (``$R X``) may write. Its word's
        values depend on what the OTHER such commands write, so this is a fixpoint: start from
        "writes nothing" and recompute every reach from the previous pass's answers until none
        changes. Reaches only grow, and every pass is linear in the bindings, so a command with
        many ``$PY ...`` lines cannot recurse exponentially. If it has not settled after
        _MAX_REACH_PASSES passes, every such command writes every name (fail closed)."""
        dynamic = [k for k, b in enumerate(self.bindings) if b.dynamic is not None]
        if not dynamic:
            return
        self._reach = {k: frozenset() for k in dynamic}
        for _ in range(_MAX_REACH_PASSES):
            self._memo.clear()
            solved = {k: self._compute_reach(k) for k in dynamic}
            if solved == self._reach:
                break
            self._reach = solved
        else:
            self._reach = {k: None for k in dynamic}
            self.truncated = True
        self._memo.clear()

    def _cap(self, values) -> tuple[str, ...]:
        unique = tuple(dict.fromkeys(values))
        if len(unique) <= _MAX_VALUES_PER_NAME:
            return unique
        self.truncated = True
        return unique[:_MAX_VALUES_PER_NAME - 1] + (UNKNOWN,)

    def binding_values(self, k: int) -> tuple[str, ...]:
        if k in self._memo:
            if self.bindings[k].truncated:
                self.truncated = True
            return self._memo[k]
        if k in self._busy:
            return (UNKNOWN,)
        self._busy.add(k)
        b = self.bindings[k]
        self.truncated = self.truncated or b.truncated
        values: list[str] = [UNKNOWN] if b.command_output else []
        for raw in b.raw_values:
            if raw == UNKNOWN:
                values.append(UNKNOWN)
                continue
            expanded, complete = _expand_all(raw, lambda name, _offset: self.values_at(name, b.start),
                                             _MAX_VALUES_PER_NAME)
            values.extend(expanded)
            if not complete:
                values.append(UNKNOWN)
                self.truncated = True
        if b.append:
            old = self.values_at(b.name, b.start)
            values = [UNKNOWN if UNKNOWN in (o, n) else o + n for o in old for n in values]
        self._busy.discard(k)
        self._memo[k] = self._cap(values)
        return self._memo[k]

    def clobbers(self, k: int, name: str) -> bool:
        """Can binding *k* (name None: a clobber of unknown or dynamic reach) write *name*?"""
        b = self.bindings[k]
        if b.name is not None or b.dynamic is None:
            return b.name in (name, None)
        reach = self._reach[k]
        return reach is None or isinstance(reach, str) or name in reach

    def is_late(self, k: int) -> bool:
        b = self.bindings[k]
        return b.late or (b.dynamic is not None and self._reach[k] == "late")

    def _compute_reach(self, k: int) -> "frozenset[str] | str | None":
        """What the command at binding *k* may write when its expanded command word takes each of
        its possible values (read from the current reach answers): a name set, None (every name)
        or "late"."""
        from tools.approval_detection import _deobfuscate_shell_word_for_detection
        from tools.approval_detection_clobbers import ALL, LATE, command_clobbers
        dynamic = self.bindings[k].dynamic or ("",)
        start = self.bindings[k].start
        forms, complete = _expand_all(dynamic[0], lambda name, _offset: self.values_at(name, start),
                                      _MAX_VALUES_PER_NAME)
        if not complete:
            return ALL
        names: set[str] = set()
        for form in forms:
            # Only the resolved COMMAND word decides whether the program is known; `$` in its
            # arguments (`R="python3 $S/x.py"`) is judged by the builtin's own handler.
            hit = command_clobbers(_deobfuscate_shell_word_for_detection(form).split() + list(dynamic[1:]),
                                   include_assignments=True)
            if hit == LATE or hit is ALL or isinstance(hit, tuple):
                return LATE if hit == LATE else ALL
            names |= hit
        return frozenset(names)

    def ifs_separators(self, offset: int) -> tuple[str, bool]:
        """The non-blank characters IFS may hold at *offset*, and whether IFS may hold a value the
        text does not show. bash never imports IFS from the environment (it starts as blank,
        tab, newline), so IFS is default unless the command itself writes it."""
        writes = [k for k, b in enumerate(self.bindings) if self.clobbers(k, "IFS")
                  and (b.end <= offset or self.scope.in_loop_with(b.start, offset))]
        if not writes:
            return "", False
        values = self.values_at("IFS", offset)
        # values_at's own UNKNOWN is the default here; only a write of an unshown value is unknown:
        # one that names IFS (`IFS=$(cat f)`, `read IFS`) or a late writer (trap, eval). `source`
        # rewriting IFS is not singled out: the sourced code is unreadable and could run anything.
        unknown = UNKNOWN in values and any(
            self.is_late(k) or (self.bindings[k].name == "IFS" and UNKNOWN in self.binding_values(k))
            for k in writes)
        seps = "".join(sorted({c for v in values if v != UNKNOWN for c in v if c not in " \t\n"}))
        return seps, unknown

    def values_at(self, name: str, offset: int, *, fallback: bool = False) -> tuple[str, ...]:
        """Possible values of *name* at *offset*; UNKNOWN is among them unless a binding fixes it.
        With *fallback*, a name no earlier binding touches also takes every value it is ever given."""
        values: list[str] = [UNKNOWN]
        # Writes that run when a FUNCTION is called, not where they are written: the call may come
        # after any later assignment, so no assignment in the text replaces them.
        deferred: list[str] = []
        bound_before = False
        late = name in _SHELL_MANAGED_NAMES
        for k, b in enumerate(self.bindings):
            if not self.clobbers(k, name):
                continue
            if b.end > offset:
                # A later binding still reaches this use through a loop or a function call.
                if self.scope.in_loop_with(b.start, offset):
                    values.extend(self.binding_values(k))
                    late = late or self.is_late(k)
                continue
            bound_before = True
            late = late or self.is_late(k)
            if self.scope.floating(b.start):
                deferred.extend(self.binding_values(k))
            elif b.persistent and self.scope.dominates(b.start, offset):
                values = list(self.binding_values(k))
            else:
                values.extend(self.binding_values(k))
        values.extend(deferred)
        if late and UNKNOWN not in values:
            # A trap handler, nameref or sourced function can rewrite the name at any point.
            values.append(UNKNOWN)
        if fallback and not bound_before:
            for k, b in enumerate(self.bindings):
                if b.name == name:
                    values.extend(self.binding_values(k))
        return self._cap(values)


@functools.lru_cache(maxsize=32)
def _resolution(command: str) -> _Resolution:
    return _Resolution(command)


def _has_candidate(command: str) -> bool:
    # Cheap pre-filter: some bound NAME must also be referenced as $NAME / ${NAME before paying for
    # the quote-aware parse (most commands with `=` are flags like --opt=value or key=value data).
    if "$" not in command:
        return False
    names = {a or b for a, b in _CANDIDATE_ASSIGNMENT_RE.findall(command)}
    return bool(names) and any(ref in names for ref in _CANDIDATE_REFERENCE_RE.findall(command))


@functools.lru_cache(maxsize=32)
def _resolve(command: str) -> tuple[tuple[str, ...], bool]:
    """(resolved variants differing from *command*, whether they cover every combination)."""
    if not _has_candidate(command):
        return (), True
    resolution = _resolution(command)
    if not resolution.bindings:
        return (), True
    variants, complete = _expand_command(
        command, lambda name, offset: resolution.values_at(name, offset, fallback=True),
        lambda offset: resolution.ifs_separators(offset)[0])
    complete = complete and not resolution.truncated
    return tuple(v for v in dict.fromkeys(variants) if v != command), complete


def resolve_shell_assignment_variants(command: str) -> list[str]:
    """Every resolved form of *command* that differs from it (empty when nothing resolves)."""
    return list(_resolve(command)[0])


def resolve_shell_assignments(command: str) -> str | None:
    """The first resolved form of *command*, or None when nothing resolves."""
    variants = resolve_shell_assignment_variants(command)
    return variants[0] if variants else None


# ---- what the command's text does not fix ------------------------------------------------------

# Pathname-expansion syntax: `*`, `?`, a `[...]` bracket, or an extglob group (`@(..)`, `!(..)`, `+(..)`).
_GLOB_RE = re.compile(r"[*?]|\[[^\]]*\]|[@!+]\(")


def _leader_is_opaque(value: str) -> bool:
    """Does *value*, spliced into command position, leave the program unreadable (its first word
    is still a whole-word expansion or substitution, like an opaque command word written directly)?
    Glob syntax in an unquoted value is judged by ``_program_globs``, which tracks quoting."""
    from tools.approval_detection import _read_shell_word
    if value == UNKNOWN or not value.strip():
        return True
    start, end, first = _read_shell_word(value, 0)
    return start == end or bool(_OPAQUE_LEADER_RE.fullmatch(first)) or _basename_is_expansion(first)


# Any expansion or substitution span inside a word: `${...}`, `$(...)`, `` `...` ``, `$NAME`.
_EXPANSION_SPAN_RE = re.compile(r"\$\{[^}]*\}|\$\([^)]*\)|`[^`]*`|\$[A-Za-z_][A-Za-z0-9_]*|\$")


def _basename_is_expansion(word: str) -> bool:
    """The program name (the part of *word* after its last literal ``/``) still depends on an
    expansion: ``/bin/$X``, ``./$X``, ``${X%/*}``. A literal basename under an expanded directory
    (``$VIRTUAL_ENV/bin/python``) names its program and is readable. Quotes do not matter here."""
    if "'" in word:
        # Single-quoted parts never expand; drop them (their text is literal).
        word = re.sub(r"'[^']*'", "q", word)
    masked = _EXPANSION_SPAN_RE.sub("\x01", word.replace('"', ""))
    return "\x01" in masked.rsplit("/", 1)[-1]


# Placeholders in a glob view: a quoted character (never a glob), and a value the text does not fix.
_QUOTED_CHAR = "q"
_UNFIXED = "\x01"


def _glob_views(word: str, values_of) -> tuple[list[str], bool]:
    """What pathname expansion sees in the command word *word*, once per combination of its
    references' values: quoting is tracked per span, not by the word's first character. A quoted
    or escaped character (and every character of a quoted reference's value) becomes a
    placeholder that is never glob syntax; an unquoted literal character stays itself; an unquoted
    reference's value is spliced in as is (bash filename-expands the result of an unquoted
    expansion). So ``""$X`` with X='r?' and ``"$D"/bin/r?`` both keep their ``?``, while
    ``"$X"`` and ``'r?'`` do not. A value the text does not fix is _UNFIXED. Returns (views, whether
    every combination was produced)."""
    from tools.approval_detection import _scan_shell
    parts: list = []    # str, or (values, quoted)
    skip_to = 0
    for kind, i, j, quote in _scan_shell(word):
        if i < skip_to or kind == "quote":
            continue
        if kind == "char" and quote != "'" and word[i] == "$":
            match = _REFERENCE_RE.match(word, i)
            if match:
                values = values_of(_reference_name(match)) or (UNKNOWN,)
                values = tuple(dict.fromkeys(_UNFIXED if v == UNKNOWN else v
                                             for v in _reference_values(match, values)))
                parts.append((values, quote == '"'))
                skip_to = match.end()
                continue
        if kind == "esc":
            parts.append(word[i + 1] if word[i + 1] == "/" else _QUOTED_CHAR)
        elif quote is not None and word[i] != "/":
            parts.append(_QUOTED_CHAR)
        else:
            parts.append(word[i])
    choices = [p[0] for p in parts if isinstance(p, tuple)]
    total = math.prod(len(c) for c in choices)
    views = []
    for combo in itertools.islice(itertools.product(*choices), _MAX_VALUES_PER_NAME):
        picks = iter(combo)
        out = []
        for p in parts:
            if isinstance(p, tuple):
                value = next(picks)
                # A quoted value is neither split nor globbed: all of it is literal text.
                out.append(_QUOTED_CHAR * max(len(value), 1) if p[1] else value)
            else:
                out.append(p)
        views.append("".join(out))
    return views, total <= _MAX_VALUES_PER_NAME


def _has_unquoted_reference(word: str) -> bool:
    """Is some expansion in *word* unquoted (so word splitting applies to its value)?"""
    views, _ = _glob_views(word, lambda name: (_UNFIXED,))
    return any(_UNFIXED in view for view in views)



def _program_globs(word: str, values_of, seps: str = "") -> bool:
    """Can pathname expansion change which program the command word *word* names? Only the
    first field counts (an unquoted value is also word-split, on blanks and the non-blank IFS
    characters *seps*). An incomplete enumeration counts."""
    views, complete = _glob_views(word, values_of)
    if not complete:
        return True
    table = str.maketrans(seps, " " * len(seps))
    for view in views:
        fields = view.translate(table).split()
        if fields and _GLOB_RE.search(fields[0]):
            return True
    return False


def _partial_leader_is_opaque(word: str, start: int, resolution) -> bool:
    """A command word that EMBEDS an expansion (``/bin/$X``): opaque if, with the same-command
    values substituted, some possible form still has an expansion in its program name, or an
    unquoted glob in its program (``$D/bin/r?``, ``""$X`` with X='r?'): that runs whatever file
    matches."""
    values_of = ((lambda name: resolution.values_at(name, start)) if resolution is not None
                 else (lambda name: None))
    seps, ifs_unknown = resolution.ifs_separators(start) if resolution is not None else ("", False)
    if "$" in word and (_program_globs(word, values_of, seps) or (ifs_unknown and _has_unquoted_reference(word))):
        return True
    if not _basename_is_expansion(word):
        return False
    if resolution is None:
        return True
    pieces = _reference_pieces(word, lambda name, _offset: resolution.values_at(name, start))
    picks, complete = pieces.product(pieces.choices, _MAX_VALUES_PER_NAME)
    return not complete or any(_basename_is_expansion(form) for form in (pieces.render(pick) for pick in picks))


def _heredoc_owner_is_data(heredoc, resolution) -> bool:
    """A heredoc whose consumer is an expansion (``$S <<EOF``) is data only when every value the
    same-command bindings give that word names a program that is not a shell (``PY=python3; $PY -
    <<EOF``). An unknown value, a shell (``S=bash``), or a wrapper that runs another word
    (``S="sudo bash"``) keeps it executed."""
    from tools.approval_detection import _COMMAND_WRAPPER_WORDS, _deobfuscate_shell_word_for_detection
    from tools.approval_detection_clobbers import is_dynamic
    from tools.approval_detection_shell_scope import _HEREDOC_SHELL_CONSUMERS
    if resolution is None:
        return False
    pieces = _reference_pieces(heredoc.owner, lambda name, _offset: resolution.values_at(name, heredoc.owner_start))
    picks, complete = pieces.product(pieces.choices, _MAX_VALUES_PER_NAME)
    if not complete:
        return False
    for pick in picks:
        form = pieces.render(pick)
        words = _deobfuscate_shell_word_for_detection(form).split()
        if (is_dynamic(form) or not words or _GLOB_RE.search(words[0])
                or any(os.path.basename(word).lower() in _HEREDOC_SHELL_CONSUMERS | _COMMAND_WRAPPER_WORDS
                       for word in words)):
            return False
    return True


def _executed_heredocs(heredocs, resolution) -> list:
    return [h for h in heredocs
            if h.executed and not (h.owner_dynamic and not h.piped and _heredoc_owner_is_data(h, resolution))]


def eval_payloads(command: str) -> tuple[list[str], bool]:
    """What each ``eval`` in *command* runs: its arguments, expanded with the same-command values
    and joined (eval re-parses them as a script). Unknown values stay as their reference text.
    Returns (payloads, whether every combination was expanded)."""
    from tools.approval_detection import (
        _deobfuscate_shell_word_for_detection, _iter_shell_command_word_spans, _read_shell_word,
        _shell_command_segment,
    )
    if "eval" not in command:
        return [], True
    resolution = _resolution(command) if _has_candidate(command) else None
    # A heredoc body fed to a non-shell (`cat > f <<EOF`) is data; blanking keeps offsets intact.
    from tools.approval_detection_shell_scope import _blank_spans, _heredoc_bodies
    heredocs = _heredoc_bodies(command)
    executed = set(_executed_heredocs(heredocs, resolution))
    text = _blank_spans(command, [(h.start, h.end) for h in heredocs if h not in executed])
    payloads: list[str] = []
    all_complete = True
    for _, end, word in _iter_shell_command_word_spans(text):
        if _deobfuscate_shell_word_for_detection(word) != "eval":
            continue
        arguments = _shell_command_segment(text, end)
        values_of = (lambda name, _offset, at=end: resolution.values_at(name, at)) if resolution else (
            lambda name, _offset: None)
        expanded, complete = _expand_all(arguments, values_of, _MAX_VARIANTS)
        all_complete = all_complete and complete
        for text in expanded:
            words, pos = [], 0
            while pos < len(text):
                start, stop, arg = _read_shell_word(text, pos)
                if start == stop:
                    break
                words.append(_deobfuscate_shell_word_for_detection(arg))
                pos = stop
            if words:
                payloads.append(" ".join(words))
    return list(dict.fromkeys(payloads)), all_complete


def _inspect(command: str, depth: int, found: list, reasons: list[str]) -> None:
    from tools.approval_detection import _execution_flag_findings, _iter_shell_command_word_spans
    resolution = _resolution(command) if _has_candidate(command) else None
    scope = resolution.scope if resolution else ShellScope(command) if "<<" in command else None
    heredocs = scope.heredocs if scope else []
    for start, end, word in _iter_shell_command_word_spans(command):
        if any(h.start <= start < h.end for h in heredocs):
            continue
        if not _OPAQUE_LEADER_RE.fullmatch(word):
            # `env "PATH=$PATH" prog`: a quoted NAME=value operand of env is an assignment, not a program.
            if (not _ASSIGNMENT_WORD_RE.fullmatch(word.replace('"', ""))
                    and _partial_leader_is_opaque(word, start, resolution)):
                found.append((command, start, end, word))
            continue
        bare = word.strip('"')
        reference = _REFERENCE_RE.match(bare)
        if reference and reference.end() == len(bare):
            values = (_reference_values(reference, resolution.values_at(_reference_name(reference), start))
                      if resolution else (UNKNOWN,))
            seps, ifs_unknown = resolution.ifs_separators(start) if resolution else ("", False)
            unquoted = not word.startswith('"')
            if unquoted and seps:
                table = str.maketrans(seps, " " * len(seps))
                values = tuple(dict.fromkeys(values + tuple(v.translate(table) for v in values)))
            if (not (unquoted and ifs_unknown)
                    and not any(_leader_is_opaque(value) for value in values)
                    and not _program_globs(word, lambda name, at=start: resolution.values_at(name, at)
                                           if resolution else None, seps)):
                continue
        found.append((command, start, end, word))
    if not _resolve(command)[1]:
        reasons.append("its variables have more possible values than the resolver examines, so not "
                       "every command it can run was checked")
    payloads = [command[h.start:h.end] for h in _executed_heredocs(heredocs, resolution)]
    payloads += [payload for _, payload in _execution_flag_findings(command) if payload]
    evaluated, complete = eval_payloads(command)
    if not complete:
        reasons.append("an eval's arguments have more possible values than the resolver examines")
    payloads += evaluated
    payloads = [payload for payload in dict.fromkeys(payloads) if payload != command]
    if not payloads:
        return
    if depth >= _MAX_PAYLOAD_DEPTH:
        reasons.append(f"it nests shell payloads (bash -c, eval, a heredoc fed to a shell) more than "
                       f"{_MAX_PAYLOAD_DEPTH} levels deep, past what is inspected")
        return
    for payload in payloads:
        _inspect(payload, depth + 1, found, reasons)


@functools.lru_cache(maxsize=32)
def _inspection(command: str) -> tuple[tuple[tuple[str, int, int, str], ...], tuple[str, ...]]:
    found: list[tuple[str, int, int, str]] = []
    reasons: list[str] = []
    _inspect(command, 0, found, reasons)
    return tuple(dict.fromkeys(found)), tuple(dict.fromkeys(reasons))


def opaque_command_leaders(command: str) -> list[tuple[str, int, int, str]]:
    """Every command word whose program this command does not fix, as ``(script, start, end, word)``
    where *script* is the text the offsets index (the command, or a shell payload inside it). An
    opaque word is a variable reference that may still hold its environment (or another unreadable)
    value at that point, a variable whose possible value starts with an expansion, a substitution
    or a glob, or a command substitution (``$(cat f) args``).

    The ORIGINAL command is scanned, not a resolved form, because splicing a deobfuscated value back
    in unquotes it and invents command positions (``B=$(printf '%s' '<?php $c=1'); echo $B``). Heredoc
    bodies fed to a non-shell are data and skipped. Shell payloads (``bash -c '...'``, the body of
    ``bash <<EOF``, an ``eval``'s arguments) are scanned as scripts of their own that start with an
    unknown environment."""
    return list(_inspection(command)[0])


def uninspectable_reasons(command: str) -> list[str]:
    """Why the inspection of *command* is incomplete (a bound was hit), or [] when it is complete.
    Incomplete is never treated as resolved."""
    return list(_inspection(command)[1])
