"""Span-keeping shell word lexer for the approval detector's quoted-grep-operand handling.

Moved verbatim from :mod:`tools.approval_detection` (size ratchet); see `_quoted_grep_pattern_spans` there.
"""

from tools.approval_parser_budget import _scan_shell


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
