"""Conservative heredoc masking for shell-command scanners ('&' guard, blocked-command checks,
cron lifecycle_guard) that false-positive on heredoc *bodies*. Stripping every body is unsafe the
other way (a fake ``<<`` in quotes can swallow an operator; unquoted bodies expand; ``bash <<'EOF'``
executes), so a body is masked ONLY when every delimiter is quoted, every heredoc has an exact
terminator line, the owning simple command is an allowlisted non-shell interpreter, and no list
operator follows the heredoc. Otherwise the command is returned untouched: a false positive is
acceptable, hiding shell syntax from a guard is not.
Masked bodies keep their newline count (re.MULTILINE)."""

from __future__ import annotations

import re

# Non-shell interpreters whose quoted heredoc bodies are data for THAT interpreter; optional
# VAR=... assignments, ``env`` and a path prefix allowed. Narrow on purpose: unmatched = visible.
_INERT_HEREDOC_CONSUMER_RE = re.compile(
    r"^\s*(?:[A-Z_][A-Z0-9_]*=\S+\s+)*(?:env\s+)?(?:[A-Za-z0-9_./-]+/)?"
    r"(?:python(?:3(?:\.\d+)*)?|osascript|cat)(?=\s|$)",
    re.IGNORECASE)


def _span_end(command: str, cursor: int, closer: str) -> int:
    """Index just past the backslash-aware span opened at ``cursor``."""
    end = cursor + 1
    while end < len(command):
        if command[end] == closer:
            return end + 1
        end += 2 if command[end] == "\\" and end + 1 < len(command) else 1
    return end


def _header_backtick_end(command: str, cursor: int) -> int | None:
    """Refuse backtick spans with shell comments or unresolved ownership."""
    end = _span_end(command, cursor, "`")
    span = command[cursor:end]
    if len(span) <= 1 or not span.endswith("`") or "\n" in span or "<<" in span:
        return None
    quote = None
    index = cursor + 1
    while index < end - 1:
        char = command[index]
        if char == "\\" and quote != "'":
            index += 2
            continue
        if char == quote:
            quote = None
        elif quote is None and char in "'\"":
            quote = char
        elif quote is None and char == "#" and (index == cursor + 1 or command[index - 1] in " \t;|&()"):
            return None
        index += 1
    return end


def _header_expansion_end(command: str, cursor: int) -> int | None:
    """Find a complete one-line substitution, refusing nested heredoc syntax.

    Quotes are local to each ``$(...)`` frame: the quotes in ``"$(pwd "$DIR")"``
    do not close the outer word. Multiline/nested heredoc ownership stays conservative.
    """
    if command[cursor] == "`":
        return _header_backtick_end(command, cursor)
    quotes: list[str | None] = [None]
    cursor += 2
    while cursor < len(command):
        char = command[cursor]
        if char == "\n" or command.startswith("<<", cursor):
            return None
        # A shell comment consumes apparent closers until the next line, which
        # is outside the single-line substitutions we can classify as inert.
        if char == "#" and quotes[-1] is None and command[cursor - 1] in " \t;|&()":
            return None
        if char == "\\":
            cursor += 2
        elif char == "'" and quotes[-1] is None:
            end = command.find("'", cursor + 1)
            if end == -1 or "\n" in command[cursor:end]:
                return None
            cursor = end + 1
        elif char == "`":
            end = _header_expansion_end(command, cursor)
            if end is None:
                return None
            cursor = end
        elif char == '"':
            quotes[-1] = None if quotes[-1] else '"'
            cursor += 1
        elif command.startswith("$(", cursor) or (char == "(" and quotes[-1] is None):
            quotes.append(None)
            cursor += 2 if command.startswith("$(", cursor) else 1
        elif char == ")" and quotes[-1] is None:
            quotes.pop()
            cursor += 1
            if not quotes:
                return cursor
        else:
            cursor += 1
    return None


def _mask_header_expansions(command: str) -> str:
    """Hide substitutions for ownership classification ONLY, retaining offsets.

    The returned command is never scanned or executed. The original opener remains
    visible to all guards, including commands executed while expanding a redirect.
    """
    parts = []
    cursor = 0
    quote = None
    while cursor < len(command):
        char = command[cursor]
        if char == "\\" and quote != "'":
            parts.append(command[cursor:cursor + 2])
            cursor += 2
        elif quote != "'" and (command.startswith(("$(", "<(", ">("), cursor) or char == "`"):
            end = _header_expansion_end(command, cursor)
            if end is None:
                return command
            parts.append("x" * (end - cursor))
            cursor = end
        else:
            if char == quote:
                quote = None
            elif quote is None and char in "'\"":
                quote = char
            parts.append(char)
            cursor += 1
    return "".join(parts)


def _mask_simple_quotes(command: str) -> str:
    """Blank inert quoted spans; keep ``$(``/backtick-bearing ones visible."""
    result = []
    cursor = 0
    while cursor < len(command):
        char = command[cursor]
        if char in "'\"":  # single quotes have no escapes; double quotes are backslash-aware
            end = (command.find("'", cursor + 1) + 1 if char == "'"
                   else _span_end(command, cursor, '"'))
            segment = command[cursor:end]
            if not segment.endswith(char):
                result.append(command[cursor:])
                break
            keep = char == '"' and ("$(" in segment or "`" in segment)
            result.append(segment if keep else char * 2)
            cursor = end
        elif char == "`":
            end = _span_end(command, cursor, "`")
            result.append(command[cursor:end])
            cursor = end
        else:
            result.append(char)
            cursor += 1
    return "".join(result)


def _parse_heredoc_operator(command: str, index: int):
    """Parse one ``<<`` opener -> ``(end_index, delimiter, strip_tabs, quoted)`` or None."""
    if not command.startswith("<<", index) or command.startswith("<<<", index):
        return None
    strip_tabs = command.startswith("-", index + 2)
    cursor = index + 3 if strip_tabs else index + 2
    while cursor < len(command) and command[cursor] in " \t":
        cursor += 1
    if cursor >= len(command) or command[cursor] in "\r\n":
        return None
    delimiter: list[str] = []
    quoted = False
    while cursor < len(command) and not (command[cursor].isspace() or command[cursor] in ";&|<>()"):
        char = command[cursor]
        if char == "\\":  # backslash-escaped char: quoted, literal
            if cursor + 1 >= len(command) or command[cursor + 1] in "\r\n":
                return None
            quoted = True
            delimiter.append(command[cursor + 1])
            cursor += 2
        elif char in "'\"":
            quoted = True
            cursor += 1
            while cursor < len(command) and command[cursor] != char:
                current = command[cursor]
                if current in "\r\n":
                    return None
                if char == '"' and current == "\\":
                    if cursor + 1 >= len(command):
                        return None
                    if command[cursor + 1] in '$`"\\\n':  # else backslash is literal in dquotes
                        cursor += 1
                        current = command[cursor]
                delimiter.append(current)
                cursor += 1
            if cursor >= len(command):  # unterminated quote
                return None
            cursor += 1
        else:
            delimiter.append(char)
            cursor += 1
    if not delimiter and not quoted:
        return None
    return cursor, "".join(delimiter), strip_tabs, quoted


def _is_fd_redirect_ampersand(command: str, index: int) -> bool:
    """Return whether ``&`` at ``index`` belongs to ``>&``/``<&``/``&>`` redirection."""
    before = command[index - 1] if index else ""
    after = command[index + 1] if index + 1 < len(command) else ""
    return before in "<>" or after == ">"


def _scan_heredoc_command_unit(command: str, start: int):
    """Scan one logical command.

    Return ``(end, specs, unknown_operator, post_heredoc_list_operator, owner_start)``.
    List operators before the first heredoc select the simple command that owns it. A list
    operator after a heredoc keeps the body visible because another command may consume it.
    """
    cursor = start
    quote = None
    comment = False
    specs = []
    unknown_operator = False
    post_heredoc_list_operator = False
    owner_start = start
    while cursor < len(command):
        char = command[cursor]
        if char == "\n" and (comment or quote is None):
            break
        # Backslash escapes (incl. line continuations) outside single quotes skip the next char.
        escaped = char == "\\" and quote != "'" and not comment and cursor + 1 < len(command)
        if not comment and not escaped and quote != "'" and (
            command.startswith(("$(", "<(", ">("), cursor) or (char == "`" and quote != "`")
        ):
            end = _header_expansion_end(command, cursor)
            if end is None:
                unknown_operator = True
                cursor += 1
            else:
                cursor = end
        elif comment or quote is not None or escaped:
            if char == quote:
                quote = None
            cursor += 2 if escaped else 1
        elif char in "'\"`":
            quote = char
            cursor += 1
        elif char == "#" and (cursor == start or command[cursor - 1].isspace()
                              or command[cursor - 1] in ";&|()"):
            comment = True
            cursor += 1
        elif command.startswith("<<<", cursor):
            cursor += 3
        elif command.startswith("<<", cursor):
            parsed = _parse_heredoc_operator(command, cursor)
            if parsed is None:
                unknown_operator = True
                cursor += 2
            else:
                cursor, delimiter, strip_tabs, quoted = parsed
                specs.append((delimiter, strip_tabs, quoted))
        else:
            if char in ";|&" and not (
                char == "&" and _is_fd_redirect_ampersand(command, cursor)
            ):
                if specs:
                    post_heredoc_list_operator = True
                else:
                    owner_start = cursor + 1
            cursor += 1
    return cursor, specs, unknown_operator, post_heredoc_list_operator, owner_start


def _find_heredoc_close(
        command: str, body_start: int, delimiter: str, strip_tabs: bool) -> int | None:
    """Return the position after an exact shell heredoc terminator line."""
    cursor = body_start
    while True:
        newline = command.find("\n", cursor)
        after = len(command) if newline == -1 else newline + 1
        line = command[cursor:after].removesuffix("\n").removesuffix("\r")
        candidate = line.lstrip("\t") if strip_tabs else line
        if candidate == delimiter:
            return after
        if newline == -1:
            return None
        cursor = after


def strip_inert_heredoc_bodies(command: str) -> str:
    """Mask heredoc bodies that are provably inert data (see module docstring)."""
    # Runs on every terminal call: skip the state machine when no '<<' exists; stop past the last.
    if "<<" not in command:
        return command
    last_opener_index = command.rfind("<<")
    ranges: list[tuple[int, int]] = []
    command_start = 0
    while command_start <= last_opener_index:
        (
            command_end,
            specs,
            unknown_operator,
            post_heredoc_list_operator,
            owner_start,
        ) = _scan_heredoc_command_unit(command, command_start)
        if unknown_operator:
            return command
        if not specs:
            if command_end >= len(command):
                break
            command_start = command_end + 1
            continue
        if command_end >= len(command):
            return command  # opener with no body line: unterminated — leave visible
        body_cursor = command_end + 1
        body_ranges: list[tuple[int, int]] = []
        for delimiter, strip_tabs, _quoted in specs:
            close_end = _find_heredoc_close(command, body_cursor, delimiter, strip_tabs)
            if close_end is None:
                return command  # unterminated
            body_ranges.append((body_cursor, close_end))
            body_cursor = close_end
        if (
            all(quoted for _delimiter, _strip_tabs, quoted in specs)
            and not post_heredoc_list_operator
        ):
            header = _mask_header_expansions(command[command_start:command_end])
            masked_opener = _mask_simple_quotes(header)
            masked_owner = _mask_simple_quotes(header[owner_start - command_start:])
            if not any(
                marker in masked_opener
                for marker in ("$(", "`", "<(", ">(", "(", ")", "{", "}")
            ) and _INERT_HEREDOC_CONSUMER_RE.search(masked_owner):
                ranges.extend(body_ranges)
        command_start = body_cursor
    # Single-pass rebuild (ranges are sorted and non-overlapping), bodies -> their newlines only.
    parts: list[str] = []
    previous = 0
    for start, end in ranges:
        parts += [command[previous:start], "\n" * command.count("\n", start, end)]
        previous = end
    return "".join(parts) + command[previous:]
