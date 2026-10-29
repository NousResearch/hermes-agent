"""Whole-record grouping of an MCP child's stderr before it is redacted and logged.

Redacting one line at a time misses a secret whose key and value sit on different lines: a
pretty-printed JSON config dump, a Python ``pprint`` that wraps a long token, a stack trace
quoting a request. :class:`StderrRecords` regroups the decoded lines into records (an open
bracket keeps the record open across lines while each line leaves the structure mid-way or is
indented; an indented line also continues the one before it), and
:func:`redact_record` strips terminal colour codes (which can split a token from its key) before
the shared redactor sees the whole record.
"""

from __future__ import annotations

import re

_ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")
_OPENERS, _CLOSERS = "{[(", "}])"
# A record past this size is written as-is (redacted) rather than held: a child printing an
# unbalanced bracket must not make the log stop growing.
RECORD_MAX_CHARS = 64 * 1024


def _bracket_delta(line: str) -> int:
    """Net open brackets on *line*, ignoring those inside double-quoted strings."""
    depth, in_string, escaped = 0, False, False
    for ch in line:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in _OPENERS:
            depth += 1
        elif ch in _CLOSERS:
            depth -= 1
    return depth


class StderrRecords:
    """Feed decoded lines; get back complete records (lists of lines)."""

    def __init__(self):
        self._lines: list[str] = []
        self._depth = 0
        self._chars = 0

    def _continues(self, line: str) -> bool:
        if not self._lines:
            return False
        stripped = line.lstrip()
        if stripped != line:  # indented: a JSON member, a wrapped repr, a traceback frame
            return True
        if self._depth == 0:
            return False
        # Inside an open bracket an unindented line still continues a structure the previous line
        # left mid-way (unindented JSON: ``{`` / ``"api_key":`` / ``"value"``). A line ending in a
        # complete statement does not hold the next one: a stray ``(`` must not stall the log.
        return stripped[:1] in _CLOSERS or self._lines[-1].rstrip()[-1:] in _OPENERS + ",:"

    def _take(self) -> list[list[str]]:
        record, self._lines, self._depth, self._chars = self._lines, [], 0, 0
        return [record] if record else []

    def feed(self, lines: list[str]) -> list[list[str]]:
        """Records completed by *lines*. A record whose brackets are balanced is released at the
        end of the batch, so ordinary log lines are never held back waiting for the next write."""
        out: list[list[str]] = []
        for raw in lines:
            line = _ANSI_RE.sub("", raw)
            if not self._continues(line):
                out += self._take()
            self._lines.append(line)
            self._depth = max(0, self._depth + _bracket_delta(line))
            self._chars += len(line) + 1
            if self._chars >= RECORD_MAX_CHARS:
                out += self._take()
        if self._depth == 0:
            out += self._take()
        return out

    def flush(self) -> list[list[str]]:
        return self._take()


def redact_record(lines: list[str]) -> list[str]:
    """The record's lines after forced redaction of the record as one text."""
    from agent.redact import redact_sensitive_text

    return redact_sensitive_text("\n".join(lines), force=True).split("\n")
