"""Shell lexical helpers for approval detection; no shell execution occurs here."""

import re
import shlex

from tools.approval_ansi_c import _is_shell_comment_start

_PARAM_REPLACEMENT_RE = re.compile(r"\$\{[^}/\s]+/[^}/]*/(?P<replacement>[^}]*)\}")
_PARAM_DEFAULT_RE = re.compile(r"\$\{[^}:}\s]+:-(?P<default>[^}]*)\}")
_SIMPLE_SHELL_LITERAL_RE = re.compile(r"^[A-Za-z0-9_./:@%+=,-]+$")


def _splice(command: str, edits) -> str:
    """Apply sorted, non-overlapping ``(start, end, text)`` edits to *command* in one pass
    (re-slicing per edit is quadratic on 10k+ segments)."""
    parts, previous = [], 0
    for start, end, text in edits:
        parts.extend((command[previous:start], text))
        previous = end
    return "".join(parts) + command[previous:]


def _skip_shell_whitespace(command: str, pos: int) -> int:
    while pos < len(command) and command[pos].isspace():
        pos += 1
    return pos


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
    while i < n:
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
    depth = 1
    for kind, i, _, quote in _scan_shell(command, start + 2):
        if kind == "char" and not quote:
            depth += command.startswith("$(", i) - (command[i] == ")")
            if depth == 0:
                return i + 1
    return None


def _scan_backtick_end(command: str, start: int) -> int | None:
    # Backticks have no quote awareness: only a backslash escapes the next char.
    match = re.compile(r"(?:\\.|[^`\\])*`", re.DOTALL).match(command, start + 1)
    return match.end() if match else None


def _read_shell_word(command: str, pos: int) -> tuple[int, int, str]:
    """Read one shell word without executing expansions."""
    start = end = _skip_shell_whitespace(command, pos)
    for kind, i, j, quote in _scan_shell(command, start, subst="u", brace=True):
        if kind == "char" and quote is None and (command[i].isspace() or command[i] in ";&|<>()"):
            break
        end = j
    return (start, end, command[start:end])


def _literal_command_substitution_output(script: str) -> str | None:
    """Resolve tiny literal command substitutions without executing a shell."""
    try:
        tokens = shlex.split(script, posix=True)
    except ValueError:
        tokens = []
    if not tokens:
        return None
    command, args = tokens[0].lower(), tokens[1:]
    if command == "echo":
        while args and re.fullmatch(r"-[nEe]+", args[0]):
            args = args[1:]
    elif command != "printf":
        return None
    if len(args) == 1 and _SIMPLE_SHELL_LITERAL_RE.fullmatch(args[0]):
        return args[0]
    if command == "printf" and len(args) == 2 and args[0] == "%s" and _SIMPLE_SHELL_LITERAL_RE.fullmatch(args[1]):
        return args[1]
    return None


def _replace_simple_command_substitutions(word: str) -> str:
    chars: list[str] = []
    i = 0
    while i < len(word):
        opener = 2 if word.startswith("$(", i) else 1 if word[i] == "`" else 0
        end = (_scan_dollar_paren_end if opener == 2 else _scan_backtick_end)(word, i) if opener else None
        replacement = _literal_command_substitution_output(word[i + opener:end - 1]) if end is not None else None
        if replacement is None:
            replacement, end = word[i], i + 1
        chars.append(replacement)
        i = end
    return "".join(chars)


def _replace_simple_shell_expansions(word: str) -> str:
    word = _replace_simple_command_substitutions(word)
    word = _PARAM_REPLACEMENT_RE.sub(lambda match: match.group("replacement"), word)
    return _PARAM_DEFAULT_RE.sub(lambda match: match.group("default"), word)


def _strip_shell_word_syntax(word: str) -> str:
    return "".join(
        word[i + 1] if kind == "esc" else word[i]
        for kind, i, _, _ in _scan_shell(word) if kind != "quote"
    )


def _deobfuscate_shell_word_for_detection(word: str) -> str:
    """Approximate how shell syntax can spell a command word: collapses quoting/escaping plus
    simple literal command substitutions in the word itself. Intentionally narrow and non-executing."""
    for _ in range(2):
        previous = word
        word = _strip_shell_word_syntax(_replace_simple_shell_expansions(word))
        if word == previous:
            break
    return word


def _substitution_body(text: str, i: int, j: int) -> str:
    """Body of the ``$(...)`` / backtick span ``text[i:j]`` found by ``_scan_shell``."""
    return text[i + (1 if text[i] == "`" else 2):j - 1]


def _quoted_heredoc_body_spans(text: str, *, quoted_only: bool = True) -> list[tuple[int, int]]:
    """``(start, end)`` of each heredoc body whose delimiter word is quoted or escaped; with
    *quoted_only* False, of every heredoc body.
    (``<<'EOF'``, ``<<"EOF"``, ``<<\\EOF``, ``<<E'O'F``). The shell performs no expansion there, so a
    ``$(...)`` in such a body is data. Bodies are read line by line, not through the quote scanner:
    an apostrophe in heredoc text is not a quote."""
    spans, pos, n = [], 0, len(text)
    while pos < n:
        pending, body_start = [], None
        for kind, i, _, quote in _scan_shell(text, pos, subst="uq", comments=True):
            if kind != "char" or quote is not None:
                continue
            if text.startswith("<<", i) and not text.startswith("<<<", i) and (i == 0 or text[i - 1] != "<"):
                strip = text.startswith("<<-", i)
                pending.append((_read_shell_word(text, i + 2 + strip)[2], strip))
            elif text[i] == "\n" and pending:
                body_start = i + 1
                break
        if body_start is None:
            break
        pos = body_start
        for word, strip in pending:  # bodies follow the line in operator order
            delimiter, start = _strip_shell_word_syntax(word), pos
            while pos < n:
                newline = text.find("\n", pos)
                line_end = n if newline < 0 else newline
                line = text[pos:line_end]
                end, pos = pos, line_end + 1
                if (line.lstrip("\t") if strip else line) == delimiter:
                    break
            else:
                end = n
            if not quoted_only or any(ch in word for ch in "'\"\\"):
                spans.append((start, end))
    return spans


def _blank_nested_substitutions(text: str) -> tuple[str, list[str]]:
    """Return *text* with each complete ``$(...)`` / backtick span (quoted or unquoted) blanked to a
    same-length empty ``$(   )``, plus the blanked bodies. Offsets are preserved. A literal
    substitution (``$(echo bash)``) stays so command-word deobfuscation still resolves it. Callers
    scan each body separately, so every nesting level is walked once instead of once per ancestor.
    Substitutions inside a quoted heredoc body never run and are left as text."""
    literal = _quoted_heredoc_body_spans(text) if "<<" in text else []
    scanned = _splice(text, [(s, e, re.sub(r"[^\n]", " ", text[s:e])) for s, e in literal])
    parts, bodies = [], []
    for kind, i, j, _ in _scan_shell(scanned, subst="uq"):
        if kind == "subst" and j is not None and j - i >= 3:
            body = _substitution_body(text, i, j)
            if _literal_command_substitution_output(body) is None:
                parts.append("$(" + " " * (j - i - 3) + ")")
                bodies.append(body)
                continue
        parts.append(text[i:j])
    return "".join(parts), bodies


def _iter_command_substitution_bodies(command: str):
    """Yield every non-literal ``$(...)`` / backtick body at any nesting depth, each with its own
    nested substitutions blanked (they are yielded separately)."""
    pending = [command]
    while pending:
        _, bodies = _blank_nested_substitutions(pending.pop())
        for body in bodies:
            blanked, _ = _blank_nested_substitutions(body)
            yield blanked
            pending.append(body)


def _is_redirection_operator_char(text: str, i: int, previous) -> bool:
    """Whether the unquoted ``&`` / ``|`` at *i* belongs to a redirection operator (``>&``, ``<&``,
    ``&>``, ``&>>``, ``>|``) rather than separating commands. *previous* is the prior ``_scan_shell``
    step; only an unquoted plain ``<`` / ``>`` char there can own the ``&`` / ``|``. Treating the
    ``&`` in ``bash 2>&1 -c 'reboot'`` as a separator cut the command before ``-c``."""
    before = text[previous[1]] if previous and previous[0] == "char" and previous[3] is None else ""
    if text[i] == "|":
        return before == ">"
    if text[i] != "&":
        return False
    return before in ("<", ">") or (text.startswith(">", i + 1) and before not in ("&", "|"))


_REDIRECTION_OPERATOR_RE = re.compile(r"&>>?|<<<|<<-?|<>|>>|>&|<&|>\||[<>]")
_REDIRECTION_FD_PREFIX_RE = re.compile(r"(?:[0-9]+|\{[A-Za-z_][A-Za-z0-9_]*\})$")


def _blank_shell_redirections(text: str, start: int) -> str:
    """Replace every unquoted redirection (optional fd / ``{var}`` prefix, operator, target word) in
    ``text[start:]`` with spaces, keeping offsets. Redirections may appear anywhere in a simple
    command, and ``bash 2>/dev/null -c 'reboot'`` otherwise leaves ``2 > /dev/null`` in argv where
    option parsing stops at the first non-option before ``-c``. Substitutions are skipped as units
    and process substitutions ``<(...)`` / ``>(...)`` are words, not redirections."""
    edits, skip_to, previous = [], start, None
    for step in _scan_shell(text, start, subst="uq"):
        kind, i, _, quote = step
        if i < skip_to:
            continue
        match = None
        if kind == "char" and quote is None and not text.startswith(("<(", ">("), i):
            match = _REDIRECTION_OPERATOR_RE.match(text, i)
            if match and match.group().startswith("&") and previous and previous[0] == "char" \
                    and previous[3] is None and text[previous[1]] in "&|":
                match = None  # `&&>x` / `|&>x`: the `&` is a separator, the `>` is matched next
        if match is None:
            previous = step
            continue
        op_start = i
        if not match.group().startswith("&"):
            # Bounded lookback for an fd (`2>`) or `{var}>` prefix that forms its own word.
            prefix = _REDIRECTION_FD_PREFIX_RE.search(text, max(start, i - 64), i)
            if prefix and (prefix.start() == start or text[prefix.start() - 1].isspace()):
                op_start = prefix.start()
        _, target_end, _ = _read_shell_word(text, match.end())
        edits.append((op_start, target_end, " " * (target_end - op_start)))
        skip_to, previous = target_end, None
    return _splice(text, edits) if edits else text
