"""Position/span helpers shared by the approval detectors (`tools.approval_detection`).

They live here so that facade stays inside its line cap. It is a pure ``str`` helper with no
detector state of its own, so moving it changes nothing about what a detector decides.
"""

from __future__ import annotations


def _backtick_end_from(segment: str, i: int) -> int | None:
    """Index of the backtick closing the one opened at ``i``, or None when none follows."""
    j = segment.find("`", i + 1)
    while j != -1 and segment[j - 1] == "\\":
        j = segment.find("`", j + 1)
    return None if j == -1 else j
