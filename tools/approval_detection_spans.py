"""Where ``$(...)`` / ``${...}`` / backtick spans end, and which readings a command has.

One iterative pass (:func:`_analyze_substitutions`) finds the closer of every span, so every
detection scanner in :mod:`tools.approval_detection` shares one answer in linear time. No recursion
and no per-opener rescans: nesting past ``_MAX_SUBSTITUTION_NESTING`` is reported and callers fail
closed. Shells also disagree about a few quotes (:func:`_quote_readings`); detection inspects every
reading. Detection only: nothing here runs or expands anything.
"""

import functools


_MAX_SUBSTITUTION_NESTING = 48
_MAX_QUOTE_READING_ROUNDS = 24
_MAX_QUOTE_READINGS = 8


def _is_shell_comment_start(command: str, index: int, closed: "set[int] | None" = None) -> bool:
    """Whether the ``#`` at *index* begins a comment: it must begin a word, so it follows
    whitespace or a metacharacter that is not escaped and not the closer of a ``$(...)``,
    ``<(...)``, ``>(...)`` or ``a=(...)`` (there the ``#`` continues the word). A plain ``(...)`` group closer still ends the word, so ``#`` after it is a comment.
    *closed* is the set of offsets just past each closer that ends a word (see ``_word_closers``)."""
    if command[index] != "#":
        return False
    if index in _arithmetic_hashes(command):
        return False  # `$(( 1 # ))`: arithmetic has no comments
    pos = index
    while pos >= 2 and command[pos - 1] == "\n":
        j = pos - 2
        while j >= 0 and command[j] == "\\":
            j -= 1
        count = pos - 2 - j
        if not count % 2:
            return True
        pos = j + 1 if count == 1 else pos - 2
    if pos == 0:
        return True
    prev = command[pos - 1]
    if not (prev.isspace() or prev in ";&|()<>"):
        return False
    backslashes, k = 0, pos - 2
    while k >= 0 and command[k] == "\\":
        backslashes += 1
        k -= 1
    if backslashes % 2:
        return False  # the whitespace / metacharacter is escaped, so it is part of the word
    if prev == ")":
        if closed is None:
            closed = _word_closers(command)
        return pos not in closed
    return True


@functools.lru_cache(maxsize=32)
def _arithmetic_hashes(text: str) -> frozenset[int]:
    """Offsets of every ``#`` inside a closed ``$(( ... ))`` arithmetic expansion, where it is
    never a comment. One linear pass, computed once per text. ``$((`` can also be a ``$(`` command
    substitution holding a subshell; its alternative comment reading is also inspected.
    An unterminated ``$((`` marks nothing."""
    out: set[int] = set()
    pending: list[int] = []
    quote: str | None = None
    depth = 0
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if depth:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if not depth:
                    out.update(pending)
                    pending = []
            elif ch == "#":
                pending.append(i)
            i += 1
            continue
        if quote == "'":
            quote = None if ch == "'" else quote
        elif ch == "\\":
            i += 2
            continue
        elif ch == "#" and quote is None and (i == 0 or text[i - 1] in " \t\n;&|("):
            nl = text.find("\n", i)
            i = n if nl < 0 else nl
            continue
        elif ch == "'" and quote is None:
            quote = "'"
        elif ch == '"':
            quote = None if quote else '"'
        elif text.startswith("$((", i):
            depth = 2
            i += 3
            continue
        i += 1
    return frozenset(out)


@functools.lru_cache(maxsize=32)
def _word_closers(text: str) -> frozenset[int]:
    """Offsets just past every ``)`` that closes a ``$(...)``, ``<(...)``, ``>(...)`` or ``a=(...)``
    (a ``#`` right after it continues the word). A plain ``(...)`` group is not in the set. One
    linear pass, computed once per text: a stack of frames, each with its own quote state, so a
    quoted ``)`` closes nothing; comments are skipped. A pairing slip can only leave an offset out
    of the set (read as a comment: the old behaviour) or add one the shell also treats as a word."""
    out: set[int] = set()
    stack: list[list] = []  # [is_word_opener, quote] per open `(`
    quote: str | None = None
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if quote == "'":
            quote = None if ch == "'" else quote
        elif ch == "\\":
            i += 2
            continue
        elif ch == "#" and quote is None and (i == 0 or text[i - 1] in " \t\n;&|("):
            nl = text.find("\n", i)
            i = n if nl < 0 else nl
            continue
        elif ch in "\"'":
            quote = None if quote == ch else (quote or ch)
        elif ch == "(" and (quote is None or text[i - 1] == "$"):
            stack.append([text[i - 1] in "$<>=" if i else False, quote])
            quote = None
        elif ch == ")" and quote is None and stack:
            word, quote = stack.pop()
            if word:
                out.add(i + 1)
        i += 1
    return frozenset(out)


def _is_command_brace(text: str, i: int) -> bool:
    """Whether the ``${`` at *i* opens bash 5.3's command substitution (``${ cmd; }`` and
    ``${| cmd; }``) rather than a parameter expansion. Other shells reject these spellings as bad
    substitutions, so reading them as commands only errs toward asking."""
    return text.startswith("${", i) and i + 2 < len(text) and text[i + 2] in " \t\n|"


def _is_group_brace(text: str, i: int) -> bool:
    """Whether the ``{`` / ``}`` at *i* is the reserved word of a command list: a standalone word
    that, for ``}``, follows a command terminator (``echo }`` passes ``}`` as an argument)."""
    if (i + 1 < len(text) and not text[i + 1].isspace() and text[i + 1] not in ";&|)}") or (
            i > 0 and not text[i - 1].isspace() and text[i - 1] not in ";&|({"):
        return False
    if text[i] == "{":
        return True
    j = i - 1
    while j >= 0 and text[j] in " \t":
        j -= 1
    return j < 0 or text[j] in ";&|\n({}"


def _is_word_brace(text: str, i: int) -> bool:
    """Whether the span opening at *i* is a ``${...}`` parameter expansion (its body is one word)."""
    return text.startswith("${", i) and not _is_command_brace(text, i)


def _analyze_substitutions(text: str, start: int = 0, single: bool = False):
    """One iterative pass that finds the closer of every ``$(...)`` / ``${...}`` / backtick span.

    Returns ``(closes, overflow, ambiguous)``: *closes* maps each opener offset to the offset just
    past its closer, or None when it never closes; *overflow* says nesting went past
    ``_MAX_SUBSTITUTION_NESTING`` (the scan stops there and callers fail closed); *ambiguous* lists
    ``("posix", i)`` / ``("ansi", i)`` for each quote at *i* that shells read differently (see
    ``_quote_readings``). An explicit stack of frames replaces recursion, so the work is linear in
    ``len(text)`` for any input, including thousands of unterminated openers, and nothing here can
    raise ``RecursionError``.

    Every frame is a fresh command context with its own quote state, exactly as a shell parses it:
    a ``$(...)`` body is a command (comments hide their quotes and ``)``; a bare ``(`` opens a level
    its ``)`` closes), a ``${...}`` body is a word (a ``}`` closes it only outside its own quotes),
    and a backtick body ends at the next unescaped backtick. Nested spans open in any unquoted or
    double-quoted position. With *single*, analyse only the one span opening at *start*.
    """
    closes: dict[int, int | None] = {}
    ambiguous: list[tuple[str, int]] = []
    stack: list[list] = []
    cur: list = ["top", -1, None, 0]  # kind, opener offset, quote, bare-paren depth
    i, n, overflow = start, len(text), False

    def close(end: int) -> bool:
        """Close the innermost span; True when a *single*-span analysis is complete."""
        nonlocal cur
        closes[cur[1]] = end
        cur = stack.pop()
        return single and not stack

    while i < n:
        ch, kind, quote = text[i], cur[0], cur[2]
        if kind == "tick":
            if ch == "\\":
                i += 2
                continue
            if ch == "`":
                overflow = _analyze_body(text, cur[1] + 1, i, ambiguous)
                if overflow or close(i + 1):
                    break
            i += 1
            continue
        if quote == "'":
            if ch == "'":
                cur[2] = None
        elif ch == "\\":
            i += 2
            continue
        elif ch == "#" and quote is None and kind != "brace" and _is_shell_comment_start(text, i):
            i = _comment_end(text, i)
            continue
        elif ch == '"':
            cur[2] = None if quote else '"'
        elif ch == "'" and quote is None:
            cur[2] = "'"
            _note_ambiguous_quote(text, i, kind, stack, ambiguous)
        elif ch == "`" or text.startswith(("$(", "${"), i):
            if len(stack) >= _MAX_SUBSTITUTION_NESTING:
                overflow = True
                break
            width = 1 if ch == "`" else 2
            closes[i] = None
            stack.append(cur)
            cur = ["tick" if ch == "`" else "paren" if text[i + 1] == "(" else
                   "cbrace" if _is_command_brace(text, i) else "brace", i, None, 1]
            i += width
            continue
        elif quote is None and _span_closes_at(text, i, cur) and close(i + 1):
            break
        i += 1
    if cur[0] == "tick" and not overflow:
        # Unterminated backtick: consumers read its text as ordinary command text, so analyse that.
        overflow = _analyze_body(text, cur[1] + 1, n, ambiguous)
    return closes, overflow, ambiguous


def _span_closes_at(text: str, i: int, cur: list) -> bool:
    """Whether the unquoted char at *i* closes the open ``$(`` / ``${ cmd; }`` / ``${...}`` frame
    *cur*; tracks the frame's bare-paren / group-brace depth as a side effect."""
    kind, ch = cur[0], text[i]
    if kind == "paren" and ch in "()":
        cur[3] += 1 if ch == "(" else -1
        return cur[3] == 0
    if kind == "cbrace" and ch in "{}" and _is_group_brace(text, i):
        # A command list ends at a `}` in command position; `{ ...; }` groups nest inside it.
        if ch == "{":
            cur[3] += 1
        elif cur[3] > 1:
            cur[3] -= 1
        else:
            return True
        return False
    return kind == "brace" and ch == "}"


def _comment_end(text: str, i: int) -> int:
    """Offset of the newline ending the comment at *i* (the newline itself is not part of it)."""
    newline = text.find("\n", i)
    return len(text) if newline < 0 else newline


def _analyze_body(text: str, start: int, end: int, ambiguous: list[tuple[str, int]]) -> bool:
    """Analyse the backtick body ``text[start:end]`` as a command of its own: its nesting counts
    toward the limit and its shell-dependent quotes join *ambiguous* (offsets mapped back into
    *text*). An escaped backtick stays escaped in the slice, so this never recurses more than once.
    Returns whether the body overflowed."""
    _, overflow, found = _analyze_substitutions(text[start:end])
    ambiguous.extend((kind, start + offset) for kind, offset in found)
    return overflow


def _note_ambiguous_quote(text: str, i: int, kind: str, stack: list[list],
                          ambiguous: list[tuple[str, int]]) -> None:
    """Record the quote opening at *i* when shells disagree about it."""
    if i > 0 and text[i - 1] == "$":
        ambiguous.append(("ansi", i))  # `$'...'`: bash honours \' inside; dash does not
    if kind == "brace" and _inside_double_quotes(stack):
        ambiguous.append(("posix", i))  # POSIX shells read this quote as a literal (even after `$`)


def _inside_double_quotes(stack: list[list]) -> bool:
    """Whether the ``${...}`` frame on top of *stack*'s child sits in a double-quoted word."""
    for frame in reversed(stack):
        if frame[2] == '"':
            return True
        if frame[0] not in ("brace", "cbrace"):
            return False
    return False


_analyze_substitutions = functools.lru_cache(maxsize=32)(_analyze_substitutions)


def _scan_substitution_end(text: str, start: int) -> int | None:
    """Offset after the span opening at *start* (``$(``, ``${`` or a backtick), None if it never
    closes or nesting overflowed. Looks the answer up in the whole-text pass; a span the whole-text
    pass lexed inside a comment or quote (callers without comment support can land on one) is
    analysed on its own."""
    closes, overflow, _ = _analyze_substitutions(text)
    if start in closes:
        return closes[start]
    if overflow:
        raise _ParserLimit
    closes, overflow, _ = _analyze_substitutions(text, start, True)
    if overflow:
        raise _ParserLimit  # nesting only a consumer's different lexing exposes: refuse, don't skip
    return closes.get(start)


class _ParserLimit(Exception):
    """Raised inside detection when input is too deeply nested or too ambiguous to inspect; the
    public entry points turn it into the fail-closed parser-limit verdict."""


def _rewrite_ambiguous(command: str, kind: str) -> str:
    """Same-length copy of *command* read the way the other shell family reads it: for ``"posix"``
    a quote inside a double-quoted ``${...}`` is a literal (dash, busybox ash, bash --posix); for
    ``"ansi"`` the body of ``$'...'`` carries no quotes or escapes (bash honours ``\\'`` there,
    so the quote lexer would otherwise flip). Rewritten text is only ever scanned, never run."""
    done: set[int] = set()
    for _ in range(_MAX_QUOTE_READING_ROUNDS):
        todo = [i for k, i in _analyze_substitutions(command)[2] if k == kind and i not in done]
        if not todo:
            return command
        chars = list(command)
        for i in todo:
            done.add(i)
            if kind == "posix":
                chars[i] = "_"
                continue
            j = i + 1
            while j < len(chars) and chars[j] != "'":
                if chars[j] == "\\" and j + 1 < len(chars):
                    chars[j] = chars[j + 1] = "_"
                    j += 1
                j += 1
        command = "".join(chars)
    raise _ParserLimit


def _arithmetic_comment_reading(command: str) -> str:
    """Conservatively inspect the alternative command-substitution comment reading."""
    hashes = _arithmetic_hashes(command)
    if not hashes:
        return command
    chars = list(command)
    skip_to = 0
    for i in sorted(hashes):
        if i < skip_to:
            continue
        newline = command.find("\n", i)
        end = len(command) if newline < 0 else newline
        if "'" in command[i:end] or '"' in command[i:end]:
            chars[i:end] = " " * (end - i)
        skip_to = end
    return "".join(chars)


def _quote_readings(command: str) -> list[str]:
    """The command plus each distinct reading of its shell-dependent quotes (fail closed: every
    reading is inspected). Usually just ``[command]``. The readings are the closure of the
    rewrites, because reading one quote differently can turn the next into an opener (a ``$'`` that
    only opens once a posix-literal quote stops closing the string before it)."""
    if "'" not in command and ('"' not in command or "$((" not in command):
        return [command]
    readings, pending = [command], [command]
    if _substitution_nesting_exceeded(command):
        raise _ParserLimit
    while pending:
        source = pending.pop()
        for kind in ("ansi", "posix", "arithmetic"):
            reading = (_arithmetic_comment_reading(source) if kind == "arithmetic" else
                       _rewrite_ambiguous(source, kind))
            if reading not in readings:
                if len(readings) >= _MAX_QUOTE_READINGS or _substitution_nesting_exceeded(reading):
                    raise _ParserLimit
                readings.append(reading)
                pending.append(reading)
    return readings


def _substitution_nesting_exceeded(command: str) -> bool:
    return _analyze_substitutions(command)[1]
