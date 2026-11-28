"""Shell lexical helpers for approval detection; no shell execution occurs here."""

import re
import shlex

from tools import approval_parser_budget as _parser_budget
from tools.approval_parser_budget import _scan_backtick_end, _scan_dollar_paren_end, _scan_shell
from tools.approval_quoted_newlines import _mask_quoted_newlines

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
    _parser_budget.charge(len(script))
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


# A brace word in COMMAND position expands to its comma-separated alternatives as separate words:
# `{rm,-rf,/}` runs as `rm -rf /`. Argument-position braces (`find / -{delete,print}`) never reach
# this deobfuscator, so the dynamic-word rules keyed on them still fire. No quotes/$/nesting.
_BRACE_ALTERNATION_RE = re.compile(r"\{([A-Za-z0-9_./:%@+=^-]+(?:,[A-Za-z0-9_./:%@+=^-]+)+)\}")


def _deobfuscate_shell_word_for_detection(word: str) -> str:
    """Approximate how shell syntax can spell a command word: collapses quoting/escaping plus
    simple literal command substitutions in the word itself. Intentionally narrow and non-executing."""
    # Pure, and the word walkers ask for the same words once per command start: memoize per call.
    return _parser_budget.memo(_deobfuscate_shell_word, word)


def _deobfuscate_shell_word(word: str) -> str:
    if brace := _BRACE_ALTERNATION_RE.fullmatch(word):
        return brace.group(1).replace(",", " ")
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
            quoted = any(ch in word for ch in "'\"\\")
            delimiter, start = _strip_shell_word_syntax(word), pos
            line, line_start = "", pos
            while pos < n:
                newline = text.find("\n", pos)
                line_end = n if newline < 0 else newline
                line += text[pos:line_end]
                pos = line_end + 1
                # In an expanding body bash joins a backslash-newline before it compares the line
                # with the delimiter: `a\` then `D` is the line `aD`, not the end of a `<<D` body.
                if not quoted and line_end < n and (len(line) - len(line.rstrip("\\"))) % 2:
                    line = line[:-1]
                    continue
                if (line.lstrip("\t") if strip else line) == delimiter:
                    end = line_start
                    break
                line, line_start = "", pos
            else:
                end = n
            if not quoted_only or quoted:
                spans.append((start, end))
    return spans


def _blank_nested_substitutions(text: str) -> tuple[str, list[str]]:
    """Return *text* with each complete ``$(...)`` / backtick span (quoted or unquoted) blanked to a
    same-length empty ``$(   )``, plus the blanked bodies. Offsets are preserved. A literal
    substitution (``$(echo bash)``) stays so command-word deobfuscation still resolves it. Callers
    scan each body separately, so every nesting level is walked once instead of once per ancestor.
    Substitutions inside a quoted heredoc body never run and are left as text."""
    if "$(" not in text and "`" not in text:
        return text, []
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
    nested substitutions blanked (they are yielded separately). Quoted newlines are masked first
    so the quote context that bounds a body is the one the raw command has."""
    if "$(" not in command and "`" not in command:
        return
    pending = [_mask_quoted_newlines(command)]
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


def _simple_command_end(command: str, start: int, spans: list[tuple[int, int]]) -> int:
    """End offset of the simple command starting at *start*: bounded by the innermost substitution
    body that contains it, then by the first unquoted separator or closing ``)`` at its own nesting
    level. Nested ``$(...)`` / backtick operands are skipped as units, so their quotes and
    separators never leak into (or truncate) the enclosing command's words — without this,
    ``echo "$(grep "a|b" f; bash -c 'reboot')"`` tokenized ``bash`` through the enclosing ``)"``,
    lexed as malformed, and the shell payload was silently skipped."""
    limit = len(command)
    enclosing = max((span for span in spans if span[0] <= start < span[1]), default=None)
    if enclosing is not None:
        limit = enclosing[1]
    depth = 0  # unquoted `(`: subshells, process substitutions, arrays
    previous = None
    for step in _scan_shell(command, start, limit, subst="uq", brace=True, comments=True):
        kind, i, _, quote = step
        prior, previous = previous, step
        if kind == "comment":
            return i
        if kind != "char" or quote is not None:
            continue
        ch = command[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                return i
            depth -= 1
        elif ch == "`" or (depth == 0 and ch in ";&|\n"
                           and not _is_redirection_operator_char(command, i, prior)):
            # A backtick left as a char has no closer: it ends the command it sits inside.
            return i
    return limit
