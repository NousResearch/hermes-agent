"""Sibling-backend child for the turn-marker cross-process race test (spawned, so importable)."""

from __future__ import annotations


def record_after_parent_moves(home: str, session_key: str, loaded, proceed) -> None:
    """Start a turn on *session_key*, pausing between the marker file's load and its store until
    *proceed* is set — the window a second backend on the same HERMES_HOME races into."""
    from tui_gateway import turn_marker

    original_load = turn_marker._load

    def paused_load(path):
        entries = original_load(path)
        loaded.set()
        proceed.wait(30)
        return entries

    turn_marker._load = paused_load
    turn_marker.record_turn_start(home, session_key, "second prompt")
