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
        if comment or quote is not None or escaped:
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


def _heredoc_substitution_spans(command: str, start: int, end: int):
    """Yield executable substitutions in an unquoted heredoc body.

    Quotes in a heredoc body are data, so this deliberately starts with a
    clean quote state.  Once a command substitution opens, its own shell
    syntax determines where it closes.
    """
    cursor = start
    while cursor < end:
        if command.startswith("$(", cursor):
            depth, quote, inner = 1, None, cursor + 2
            while inner < end:
                char = command[inner]
                if char == "\\" and quote != "'" and inner + 1 < end:
                    inner += 2
                    continue
                if char in "'\\\"" and (quote is None or char == quote):
                    quote = char if quote is None else None
                elif quote is None and command.startswith("$(", inner):
                    depth += 1
                    inner += 1
                elif quote is None and char == ")":
                    depth -= 1
                    if depth == 0:
                        yield cursor, inner + 1
                        cursor = inner + 1
                        break
                inner += 1
            else:
                cursor += 2
        elif command[cursor] == "`":
            inner = cursor + 1
            while inner < end and command[inner] != "`":
                inner += 2 if command[inner] == "\\" and inner + 1 < end else 1
            if inner < end:
                yield cursor, inner + 1
                cursor = inner + 1
            else:
                cursor += 1
        else:
            cursor += 1


def mask_heredoc_bodies_for_shell_scan(command: str) -> str:
    """Mask heredoc data while retaining unquoted command substitutions.

    Security scanners need command boundaries, not body text: treating body
    quotes as shell syntax can hide the command after a heredoc, while treating
    body lines as commands creates deny-rule false positives.  An unquoted
    heredoc still expands ``$(...)`` and backticks, so those spans remain for
    the caller to scan.  Offsets and newlines are retained for downstream
    projections.
    """
    if "<<" not in command:
        return command
    bodies: list[tuple[int, int, bool, bool]] = []
    command_start = 0
    while command_start < len(command):
        command_end, specs, unknown, _post_list, _owner_start = _scan_heredoc_command_unit(
            command, command_start)
        if unknown or not specs or command_end >= len(command):
            command_start = command_end + 1
            continue
        body_cursor = command_end + 1
        found_bodies: list[tuple[int, int, bool, bool]] = []
        for delimiter, strip_tabs, quoted in specs:
            close_end = _find_heredoc_close(command, body_cursor, delimiter, strip_tabs)
            if close_end is None:
                # EOF is the implicit end of an unterminated heredoc.  It has
                # no later shell command to hide, but its data must not become
                # a false command projection.
                found_bodies.append((body_cursor, len(command), not quoted, False))
                body_cursor = len(command)
                break
            found_bodies.append((body_cursor, close_end, not quoted, True))
            body_cursor = close_end
        if found_bodies:
            bodies.extend(found_bodies)
            command_start = body_cursor
        else:
            command_start = command_end + 1
    if not bodies:
        return command
    masked = list(command)
    for start, end, _unquoted, _has_terminator in bodies:
        for index in range(start, end):
            if masked[index] != "\n":
                masked[index] = " "
    for start, end, unquoted, has_terminator in bodies:
        if not unquoted:
            continue
        terminator_start = command.rfind("\n", start, end - 1) + 1 if has_terminator else end
        for sub_start, sub_end in _heredoc_substitution_spans(command, start, terminator_start):
            masked[sub_start:sub_end] = command[sub_start:sub_end]
    return "".join(masked)


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
            masked_opener = _mask_simple_quotes(command[command_start:command_end])
            masked_owner = _mask_simple_quotes(command[owner_start:command_end])
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
