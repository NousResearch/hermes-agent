"""``process_manage(action="wait", until=...)``: end a wait as soon as the process prints the awaited text.

Without it, waiting on a dev server or a watch-mode build meant either sleeping a guessed number of seconds
or holding the full ``timeout`` even though "ready" scrolled past in the first second. ``until`` is a plain
substring (same matching as ``watch_patterns``), checked against output the process has printed that no
earlier ``until`` wait on it already consumed: the first wait sees everything since spawn (a server that
printed "listening" before the wait began matches at once), and a later wait for the same text (the next
"Compiled" of a rebuild) only matches new output.
"""

from __future__ import annotations

import threading
import time

_MAX_TRACKED = 512
# session id -> total_output_chars position up to which an earlier ``until`` wait consumed the output
_consumed: dict[str, int] = {}
_consumed_lock = threading.Lock()


def until_match(session, until: str) -> bool:
    """True once ``until`` appears in output after this session's consumed mark; advances the mark past it."""
    from tools.ansi_strip import strip_ansi

    with session._lock:
        buffer, total = session.output_buffer, session.total_output_chars
    buffer_start = total - len(buffer)  # the buffer is a rolling tail of everything ever ingested
    with _consumed_lock:
        mark = _consumed.get(session.id, 0)
        unseen = buffer[max(mark - buffer_start, 0):]
        if (pos := unseen.find(until)) >= 0:
            end = total - len(unseen) + pos + len(until)
        elif until in strip_ansi(unseen):  # colour codes inside the awaited text
            end = total
        else:
            return False
        _consumed[session.id] = end
        while len(_consumed) > _MAX_TRACKED:
            _consumed.pop(next(iter(_consumed)))
    return True


def matched_result(session, until: str, output: str) -> dict:
    return {
        "status": "matched", "command": session.command, "output": output, "process_running": True,
        "note": f"Output contains {until!r}; the process is still running."}


def timeout_result(session, effective_timeout: int, timeout_note: str | None, output: str, until: str | None) -> dict:
    """``wait`` result when the window elapsed with the process still running (not an error)."""
    # Not a failure — models re-issued identical waits after misreading this as an error.
    result = {"status": "timeout", "command": session.command, "output": output, "process_running": True}
    base_note = f"Wait window of {effective_timeout}s elapsed — the process is still running. This is not an error."
    if until:
        base_note += f" {until!r} has not appeared in its output yet."
    if session.started_at:
        base_note += f" Uptime: {int(time.time() - session.started_at)}s."
    base_note += (
        " notify_on_complete is set: you will be notified on exit — do more work instead of waiting again."
        if session.notify_on_complete else
        " Poll again later or use terminal(background=true, notify_on_complete=true) next time for automatic notification.")
    result["timeout_note"] = f"{timeout_note}. {base_note}" if timeout_note else base_note
    return result
