"""Conservative heredoc masking for shell-command scanners ('&' guard, blocked-command checks,
cron lifecycle_guard) that false-positive on heredoc *bodies*. Stripping every body is unsafe the
other way (a fake ``<<`` in quotes can swallow an operator; unquoted bodies expand; ``bash <<'EOF'``
executes), so a body is masked ONLY when every delimiter is quoted, every heredoc has an exact
terminator line, the simple command that owns the heredoc (after env assignments, bare ``env`` and
value-only wrappers like ``timeout 600``) is an allowlisted non-shell interpreter, nothing but pipes
follows the heredoc on its line, and every command those pipes feed is a known text filter named
bare (``| grep``, never ``| bash`` or ``| ./grep``). Any ``;``/``&&``/``||``/``&`` after the heredoc
keeps it visible, since a later command may run what the first one wrote (``cat > f; bash "$f"``).
Otherwise the body is left untouched: a false positive is acceptable, hiding shell syntax from a
guard is not.
Masked bodies keep their newline count (re.MULTILINE)."""

from __future__ import annotations

import re

# Non-shell interpreters whose quoted heredoc bodies are data for THAT interpreter; optional
# VAR=... assignments, ``env`` and a path prefix allowed. Narrow on purpose: unmatched = visible.
# ``python``/``osascript`` bodies are code, not data (``os.system``/``do shell script`` reach a
# shell at runtime), so they are masked for UX-grade scanners only: a security guard scanning for
# forbidden substrings must pass ``mask_code_consumers=False`` or the payload hides in the blanked
# body (#127698).
# Wrappers that run their argument tail and take only numeric/flag operands, so no option can
# smuggle in a command string (unlike ``env -S`` / ``su -c`` / ``sudo -s``): ``timeout 600 python3``
# still hands the heredoc to python3. A subset of lifecycle_guard's ``_TRANSPARENT_COMMAND_PREFIXES``.
_VALUE_ONLY_WRAPPERS = (
    r"(?:(?:[A-Za-z0-9_./-]+/)?(?:"
    r"timeout(?:\s+(?:-[sk]\s*[\w.]+|--(?:signal|kill-after)=[\w.]+|--?[a-z][-a-z]*))*"
    r"\s+\d+(?:\.\d+)?[smhd]?"
    r"|nice(?:\s+-n\s*-?\d+|\s+-\d+)?|nohup|stdbuf(?:\s+-[ioe]\s*[0-9LKMG]+)*"
    r"|ionice(?:\s+-[cn]\s*\d+)*)\s+)*")
_CONSUMER_PREFIX = (r"^\s*(?:[A-Z_][A-Z0-9_]*=\S+\s+)*(?:env\s+)?" + _VALUE_ONLY_WRAPPERS
                    + r"(?:[A-Za-z0-9_./-]+/)?")
_INERT_HEREDOC_CONSUMER_RE = re.compile(
    _CONSUMER_PREFIX + r"(?:python(?:3(?:\.\d+)*)?|osascript|cat)(?=\s|$)", re.IGNORECASE)
_STRICT_INERT_HEREDOC_CONSUMER_RE = re.compile(_CONSUMER_PREFIX + r"cat(?=\s|$)", re.IGNORECASE)
# Commands a heredoc owner's stdout may be piped into without the body reaching a shell: they
# print or count text and have no option that runs a command. Anything else downstream (``| bash``,
# ``| xargs sh``, ``| tee run.sh``) keeps the body visible; so do ``sort --compress-program=sh``,
# ``rg --pre CMD`` and the ``!cmd`` escapes of ``less``/``more``, which is why those are not listed.
# Bare names only: a path-qualified ``./grep`` is whatever that file is.
_PIPE_FILTER_RE = re.compile(r"^\s*(?:grep|egrep|fgrep|head|tail|uniq|wc|cut|tr|column|jq|cat)(?=\s|$)")
# The subset whose body is pure data (written out, never interpreted): `cat` only.
_DATA_HEREDOC_CONSUMER_RE = _STRICT_INERT_HEREDOC_CONSUMER_RE


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
        if char == "$":  # ``$'E'`` / ``$"E"``: bash ends at ``E``, not ``$E`` — never guess, mask nothing
            return None
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

    Return ``(end, specs, unknown_operator, operators, owner_start)``: each spec is
    ``(delimiter, strip_tabs, quoted, opener_index)``; ``operators`` lists every unquoted
    ``(index, operator)`` (``;``, ``|``, ``|&``, ``||``, ``&``, ``&&``) so a caller can find the
    simple command that owns each heredoc and the pipeline its output feeds. ``owner_start`` is
    where the simple command owning the FIRST heredoc begins.
    """
    cursor = start
    quote = None
    comment = False
    specs = []
    unknown_operator = False
    operators: list[tuple[int, str]] = []
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
                opener_index = cursor
                cursor, delimiter, strip_tabs, quoted = parsed
                specs.append((delimiter, strip_tabs, quoted, opener_index))
        elif char in ";|&" and not (char == "&" and _is_fd_redirect_ampersand(command, cursor)):
            pair = command[cursor:cursor + 2]
            operator = pair if pair in ("||", "|&", "&&") else char
            operators.append((cursor, operator))
            cursor += len(operator)
            if not specs:
                owner_start = cursor
        else:
            cursor += 1
    return cursor, specs, unknown_operator, operators, owner_start


def _heredoc_feeds_only_inert_owner(
        command: str, start: int, end: int, first_opener: int,
        operators: list[tuple[int, str]], consumer_re: re.Pattern) -> bool:
    """Whether the simple command owning the line's first heredoc is an allowlisted consumer and
    everything after it on the line is ``|``-piped text filters. One pass over the operators."""
    before = [i + len(op) for i, op in operators if i < first_opener]
    after = [(i, op) for i, op in operators if i > first_opener]
    if any(op not in ("|", "|&") for _i, op in after):
        return False
    bounds = [i for i, _op in after] + [end]
    if not consumer_re.search(_mask_simple_quotes(command[before[-1] if before else start:bounds[0]])):
        return False
    return all(_PIPE_FILTER_RE.search(_mask_simple_quotes(command[i + len(op):bounds[k + 1]]))
               for k, (i, op) in enumerate(after))


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


def iter_heredoc_bodies(command: str):
    """Yield ``(command_unit, body)`` per command line that opens heredocs, quoted or not.

    ``command_unit`` is the opener line from its owning simple command onward (so its redirects
    are visible); ``body`` joins that line's heredoc bodies without terminator lines. Stops at the
    first unparsable or unterminated heredoc: callers use this to scan MORE text, never less.
    """
    if "<<" not in command:
        return
    command_start = 0
    while command_start < len(command):
        command_end, specs, unknown_operator, _post, owner_start = _scan_heredoc_command_unit(
            command, command_start)
        if unknown_operator:
            return
        if not specs:
            command_start = command_end + 1
            continue
        body_cursor = command_end + 1
        bodies: list[str] = []
        for delimiter, strip_tabs, _quoted, _opener in specs:
            close_end = _find_heredoc_close(command, body_cursor, delimiter, strip_tabs)
            if close_end is None:
                return
            terminator_start = command.rfind("\n", body_cursor, close_end - 1) + 1
            bodies.append(command[body_cursor:max(body_cursor, terminator_start)])
            body_cursor = close_end
        yield command[owner_start:command_end], "".join(bodies)
        command_start = body_cursor


def _inert_heredoc_body_ranges(
        command: str, consumer_re: re.Pattern = _INERT_HEREDOC_CONSUMER_RE) -> list[tuple[int, int]]:
    """``(start, end)`` of every provably inert body incl. its terminator line (see module
    docstring); empty when any ``<<`` is unknown or unterminated, so nothing is masked."""
    # Runs on every terminal call: skip the state machine when no '<<' exists; stop past the last.
    if "<<" not in command:
        return []
    last_opener_index = command.rfind("<<")
    ranges: list[tuple[int, int]] = []
    command_start = 0
    while command_start <= last_opener_index:
        command_end, specs, unknown_operator, operators, _owner_start = _scan_heredoc_command_unit(
            command, command_start)
        if unknown_operator:
            return []
        if not specs:
            if command_end >= len(command):
                break
            command_start = command_end + 1
            continue
        if command_end >= len(command):
            return []  # opener with no body line: unterminated — leave visible
        body_cursor = command_end + 1
        body_ranges: list[tuple[int, int, int]] = []
        for delimiter, strip_tabs, _quoted, opener_index in specs:
            close_end = _find_heredoc_close(command, body_cursor, delimiter, strip_tabs)
            if close_end is None:
                return []  # unterminated
            body_ranges.append((body_cursor, close_end, opener_index))
            body_cursor = close_end
        masked_opener = _mask_simple_quotes(command[command_start:command_end])
        if all(quoted for _delimiter, _strip_tabs, quoted, _opener in specs) and not any(
            marker in masked_opener for marker in ("$(", "`", "<(", ">(", "(", ")", "{", "}")
        ):
            if _heredoc_feeds_only_inert_owner(command, command_start, command_end, specs[0][3], operators, consumer_re):
                ranges.extend((body_start, body_end) for body_start, body_end, _opener in body_ranges)
        command_start = body_cursor
    return ranges


def _replace_ranges(command: str, ranges: list[tuple[int, int]], keep_newlines: bool) -> str:
    # Single-pass rebuild (ranges are sorted and non-overlapping).
    parts: list[str] = []
    previous = 0
    for start, end in ranges:
        parts += [command[previous:start], "\n" * command.count("\n", start, end) if keep_newlines else ""]
        previous = end
    return "".join(parts) + command[previous:]


def strip_inert_heredoc_bodies(command: str, *, mask_code_consumers: bool = True) -> str:
    """Mask heredoc bodies that are provably inert data (see module docstring).

    ``mask_code_consumers=False`` also leaves bodies owned by code-bearing interpreters
    (``python``, ``osascript``) visible: their payload executes at runtime, so a scanner that
    looks for forbidden substrings must not have them blanked (#127698).
    """
    consumer_re = _INERT_HEREDOC_CONSUMER_RE if mask_code_consumers else _STRICT_INERT_HEREDOC_CONSUMER_RE
    return _replace_ranges(command, _inert_heredoc_body_ranges(command, consumer_re), keep_newlines=True)


def mask_heredoc_bodies_for_shell_scan(command: str, *, code_consumers: bool) -> str:
    """Drop provably inert heredoc bodies and their terminator lines for the approval scanners
    (same rule as ``strip_inert_heredoc_bodies``). ``code_consumers=False`` drops only bodies a
    data consumer (``cat``) writes out, which may replace the raw command; ``True`` also drops
    python/osascript bodies, which are code, so that copy may only be scanned IN ADDITION to the
    raw one. Removed, not blanked: long whitespace runs make ``\\s*`` command-position prefixes
    backtrack quadratically.
    """
    consumer_re = _INERT_HEREDOC_CONSUMER_RE if code_consumers else _DATA_HEREDOC_CONSUMER_RE
    # The terminator line stays: an opener left without it pairs with a later heredoc's terminator
    # and hides the commands in between from the command-start scan.
    bodies = [(start, max(start, command.rfind("\n", start, end - 1) + 1))
              for start, end in _inert_heredoc_body_ranges(command, consumer_re)]
    return _replace_ranges(command, bodies, keep_newlines=False)
