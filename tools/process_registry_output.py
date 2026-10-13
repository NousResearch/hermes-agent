"""What ProcessRegistry shows of a session's output, and how much of it is no longer kept.

``output_buffer`` is a rolling tail of the last ``max_output_chars``; ``total_output_chars``
counts everything ever ingested. Their difference is output that is gone for good, so every
view of the buffer says so instead of presenting the tail as the whole output.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.process_registry import ProcessSession


def output_tail(session: ProcessSession, n: int) -> str:
    """Last *n* chars of the session output with ANSI sequences stripped."""
    from tools.ansi_strip import strip_ansi

    return strip_ansi(session.output_buffer[-n:])


def dropped_output_chars(session: ProcessSession) -> int:
    """Chars that left the rolling buffer (0 for sessions that never overflowed it)."""
    return max(0, session.total_output_chars - len(session.output_buffer))


def completion_output(session: ProcessSession) -> dict:
    """``output`` sized by the session's ``completion_output_chars`` plus ``output_cut`` when trimmed.

    Shared by the completion notification AND the wait/poll/kill snapshots: a bot in an api_server
    or one-shot session cannot receive notifications and polls instead, so the polled result must
    carry the same whole reply and the same cut marker (#115334). ``output_cut`` counts every char
    not shown; ``output_dropped`` is the part of it that the log no longer holds either."""
    from tools.process_registry import COMPLETION_OUTPUT_CHARS

    limit = session.completion_output_chars or COMPLETION_OUTPUT_CHARS
    dropped = dropped_output_chars(session)
    cut = max(0, len(session.output_buffer) - limit) + dropped
    return {
        "output": output_tail(session, limit),
        **({"output_cut": cut} if cut > 0 else {}),
        **({"output_dropped": dropped} if dropped > 0 else {}),
    }


def log_lines(session: ProcessSession) -> tuple[list[str], int]:
    """``(lines, dropped_chars)`` of the kept log. After an overflow the buffer starts mid-line,
    so that partial first line is dropped too rather than shown as line 0."""
    from tools.ansi_strip import strip_ansi

    dropped = dropped_output_chars(session)
    output = strip_ansi(session.output_buffer)
    if dropped:
        first_break = output.find("\n")
        output = output[first_break + 1:] if first_break != -1 else ""
    return output.splitlines(), dropped
