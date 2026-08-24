"""Conservative heredoc masking for shell-command scanners.

Several guards scan raw command text for dangerous shell syntax (the
foreground background-'&' guard in ``tools/terminal_tool.py``, the
blocked-command checks, the gateway lifecycle guard in
``cron/lifecycle_guard.py``). Heredoc *bodies* are usually inline data —
AppleScript concatenation, Python bitwise-and, literal UI text — and
scanning them produces false positives.

Naively stripping every heredoc body is unsafe the other way: fake ``<<``
markers inside quotes or comments can swallow a *real* background operator
that follows them, and some heredoc bodies genuinely execute (unquoted
delimiters allow ``$(...)`` expansion; ``bash <<'EOF'`` runs the body as
shell). This module therefore masks a body ONLY when all of the following
hold, and otherwise leaves the command untouched:

- every heredoc delimiter on the opener is quoted (``<<'EOF'`` / ``<<"EOF"``
  / ``<<E\\OF``), so the body undergoes no shell expansion;
- every heredoc on the opener is terminated by an exact delimiter line;
- the opener composes a single command — no ``;``, ``|`` or ``&`` list or
  pipeline operators, and no nested ``$(...)``, backtick, or process
  substitution scope;
- the consuming command is an allowlisted non-shell interpreter (see
  ``_INERT_HEREDOC_CONSUMER_RE``); consumers that execute their input as
  shell (``bash``, ``sh``, ``eval``, ``ssh``, unknown commands) keep their
  bodies visible.

Conservative retention may cause a false positive (a scanner may still flag
payload text in an unquoted or unknown-consumer body), but it can never hide
a real background operator or lifecycle command from a guard.

Masked bodies are replaced by an equivalent number of newlines so line
structure — and any ``re.MULTILINE`` scanning downstream — is preserved.

Adapted from Wolfram Ravenwolf's security-hardened rework of PR #63788
(commit 69c7663c6de6b6cb05bf99203fa39673efe01ccf).
"""

from __future__ import annotations

import re

# Non-shell interpreters whose (quoted, inert) heredoc bodies are safe to
# mask: the body is program text or plain data for THAT interpreter, not
# shell syntax executed by this command line. Optional VAR=... assignments,
# an ``env`` prefix, and a path prefix are allowed. Deliberately narrow:
# anything not matched keeps its body visible (fail-closed).
_INERT_HEREDOC_CONSUMER_RE = re.compile(
    r"^\s*"
    r"(?:[A-Z_][A-Z0-9_]*=\S+\s+)*"
    r"(?:env\s+)?"
    r"(?:[A-Za-z0-9_./-]+/)?"
    r"(?:python(?:3(?:\.\d+)*)?|osascript|cat)(?=\s|$)",
    re.IGNORECASE,
)


def _mask_simple_quotes(command: str) -> str:
    """Blank inert quoted spans without erasing shell-active substitutions.

    Single-quoted spans become ``''``. Double-quoted spans become ``""``
    UNLESS they contain ``$(`` or a backtick (still executable — keep them
    visible). Backtick spans are always kept (executable).
    """
    result = []
    cursor = 0
    while cursor < len(command):
        char = command[cursor]
        if char == "'":
            closing = command.find("'", cursor + 1)
            if closing == -1:
                result.append(command[cursor:])
                break
            result.append("''")
            cursor = closing + 1
            continue
        if char == '"':
            end = cursor + 1
            while end < len(command):
                if command[end] == "\\" and end + 1 < len(command):
                    end += 2
                    continue
                if command[end] == '"':
                    end += 1
                    break
                end += 1
            if not command[cursor:end].endswith('"'):
                result.append(command[cursor:])
                break
            segment = command[cursor:end]
            result.append(segment if "$(" in segment or "`" in segment else '""')
            cursor = end
            continue
        if char == "`":
            end = cursor + 1
            while end < len(command):
                if command[end] == "\\" and end + 1 < len(command):
                    end += 2
                    continue
                if command[end] == "`":
                    end += 1
                    break
                end += 1
            result.append(command[cursor:end])
            cursor = end
            continue
        result.append(char)
        cursor += 1
    return "".join(result)


def _contains_nested_shell_scope(masked_opener: str) -> bool:
    """Return whether a quote-masked opener contains nested executable syntax."""
    return any(marker in masked_opener for marker in ("$(", "`", "<(", ">("))


def _parse_heredoc_operator(command: str, index: int):
    """Parse one active ``<<`` redirection and return its shell delimiter.

    Returns ``(end_index, delimiter, strip_tabs, quoted)`` or ``None`` when
    the token at ``index`` is not a well-formed heredoc opener (here-strings,
    trailing ``<<`` at end of line, unterminated quote in the delimiter).
    """
    if not command.startswith("<<", index) or command.startswith("<<<", index):
        return None

    cursor = index + 2
    strip_tabs = False
    if cursor < len(command) and command[cursor] == "-":
        strip_tabs = True
        cursor += 1
    while cursor < len(command) and command[cursor] in " \t":
        cursor += 1
    if cursor >= len(command) or command[cursor] in "\r\n":
        return None

    delimiter: list[str] = []
    quoted = False
    while cursor < len(command):
        char = command[cursor]
        if char.isspace() or char in ";&|<>()":
            break
        if char == "\\":
            if cursor + 1 >= len(command) or command[cursor + 1] in "\r\n":
                return None
            quoted = True
            delimiter.append(command[cursor + 1])
            cursor += 2
            continue
        if char in "'\"":
            quoted = True
            quote = char
            cursor += 1
            while cursor < len(command) and command[cursor] != quote:
                if quote == '"' and command[cursor] == "\\":
                    if cursor + 1 >= len(command):
                        return None
                    following = command[cursor + 1]
                    if following in {"$", "`", '"', "\\", "\n"}:
                        delimiter.append(following)
                        cursor += 2
                        continue
                    # In double quotes, backslash is literal before all other
                    # characters. Preserve it so the terminator stays exact.
                    delimiter.append("\\")
                    cursor += 1
                    continue
                if command[cursor] in "\r\n":
                    return None
                delimiter.append(command[cursor])
                cursor += 1
            if cursor >= len(command):
                return None
            cursor += 1
            continue
        delimiter.append(char)
        cursor += 1

    if not delimiter and not quoted:
        return None
    return cursor, "".join(delimiter), strip_tabs, quoted


def _scan_heredoc_command_unit(command: str, start: int):
    """Scan one logical shell command, ignoring markers in quotes/comments.

    Returns ``(end_index, heredoc_specs, unknown_operator, has_list_operator)``
    where each spec is ``(delimiter, strip_tabs, quoted)``.
    ``unknown_operator`` reports a ``<<`` token that could not be parsed
    soundly — the caller must fail closed and leave the command unmodified.
    ``has_list_operator`` reports an unquoted ``;``, ``|`` or ``&`` on the
    opener — the unit composes multiple shell commands.
    """
    cursor = start
    quote = None
    comment = False
    specs = []
    unknown_operator = False
    has_list_operator = False

    while cursor < len(command):
        char = command[cursor]
        if comment:
            if char == "\n":
                return cursor, specs, unknown_operator, has_list_operator
            cursor += 1
            continue

        if quote is not None:
            if quote in {'"', "`"} and char == "\\" and cursor + 1 < len(command):
                cursor += 2
                continue
            if char == quote:
                quote = None
            cursor += 1
            continue

        if char == "\\" and cursor + 1 < len(command):
            # Includes line continuations: the logical command keeps going on
            # the next physical line, so a heredoc opener there still belongs
            # to this unit.
            cursor += 2
            continue
        if char in "'\"`":
            quote = char
            cursor += 1
            continue
        if char == "#":
            previous = command[cursor - 1] if cursor > start else ""
            if cursor == start or previous.isspace() or previous in ";&|()":
                comment = True
                cursor += 1
                continue
        if char == "\n":
            return cursor, specs, unknown_operator, has_list_operator
        if command.startswith("<<<", cursor):
            cursor += 3
            continue
        if command.startswith("<<", cursor):
            parsed = _parse_heredoc_operator(command, cursor)
            if parsed is None:
                unknown_operator = True
                cursor += 2
                continue
            cursor, delimiter, strip_tabs, quoted = parsed
            specs.append((delimiter, strip_tabs, quoted))
            continue
        if char in ";|&":
            has_list_operator = True
        cursor += 1

    return len(command), specs, unknown_operator, has_list_operator


def _find_heredoc_close(
    command: str,
    body_start: int,
    delimiter: str,
    strip_tabs: bool,
) -> int | None:
    """Return the position after an exact shell heredoc terminator line."""
    cursor = body_start
    while True:
        newline = command.find("\n", cursor)
        if newline == -1:
            line = command[cursor:]
            after = len(command)
        else:
            line = command[cursor:newline]
            after = newline + 1
        if line.endswith("\r"):
            line = line[:-1]
        candidate = line.lstrip("\t") if strip_tabs else line
        if candidate == delimiter:
            return after
        if newline == -1:
            return None
        cursor = after


def strip_inert_heredoc_bodies(command: str) -> str:
    """Mask heredoc bodies that are provably inert data; keep the rest.

    See the module docstring for the qualification rules. Masked bodies are
    replaced with an equivalent number of newlines so positions of the
    surrounding real command text keep their line structure. On ANY ambiguity
    (unparseable ``<<`` token, unterminated heredoc, unquoted delimiter,
    compound opener, nested shell scope, unknown consumer) the original
    command is returned unchanged — a scanner false positive is acceptable,
    hiding real shell syntax is not.
    """
    ranges: list[tuple[int, int]] = []
    command_start = 0

    # Fast path: no '<<' anywhere means no heredoc can exist — skip the state
    # machine entirely. This function runs on every terminal tool call.
    if "<<" not in command:
        return command
    # No heredoc opener can start after the last '<<' occurrence; once the
    # scan passes it, the rest of the command needs no per-char walk.
    last_opener_index = command.rfind("<<")

    while command_start < len(command):
        if command_start > last_opener_index:
            break
        command_end, specs, unknown_operator, has_list_operator = (
            _scan_heredoc_command_unit(command, command_start)
        )
        if unknown_operator:
            return command
        if not specs:
            if command_end >= len(command):
                break
            command_start = command_end + 1
            continue
        if command_end >= len(command):
            # Opener with no following body line: nothing to mask, and the
            # heredoc is unterminated — leave everything visible.
            return command

        body_cursor = command_end + 1
        body_ranges: list[tuple[int, int]] = []
        unterminated = False
        for delimiter, strip_tabs, _quoted in specs:
            close_end = _find_heredoc_close(
                command,
                body_cursor,
                delimiter,
                strip_tabs,
            )
            if close_end is None:
                unterminated = True
                break
            body_ranges.append((body_cursor, close_end))
            body_cursor = close_end
        if unterminated:
            return command

        if all(quoted for _delimiter, _strip_tabs, quoted in specs) and not has_list_operator:
            masked_opener = _mask_simple_quotes(command[command_start:command_end])
            if not _contains_nested_shell_scope(masked_opener) and (
                _INERT_HEREDOC_CONSUMER_RE.search(masked_opener)
            ):
                ranges.extend(body_ranges)
        command_start = body_cursor

    if not ranges:
        return command
    # Single-pass rebuild: ranges are sorted and non-overlapping, so join the
    # kept segments with newline-preserving replacements (avoids a quadratic
    # full-string copy per masked range on heredoc-heavy commands).
    parts: list[str] = []
    previous = 0
    for start, end in ranges:
        parts.append(command[previous:start])
        parts.append("\n" * command.count("\n", start, end))
        previous = end
    parts.append(command[previous:])
    return "".join(parts)
