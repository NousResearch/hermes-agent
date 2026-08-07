"""Tests for Windows OSError recovery on TUI gateway stdin reads (#78820).

On Windows, a child process inheriting the stdio pipe can corrupt its state,
surfacing as ``OSError(EINVAL/EBADF/EPIPE)`` on ``sys.stdin.readline()``
instead of the empty-read that POSIX produces.  Without recovery, this kills
the gateway child mid-session, losing the in-flight turn.

These tests exercise the ``handle_stdin_oserror`` recovery function and verify
that both TUI entry points (entry.py, slash_worker.py) wire it into their
stdin read loops.
"""

import errno
import os
import time

import pytest

from tui_gateway._stdin_recovery import (
    MAX_RECOVERIES_PER_MINUTE,
    handle_stdin_oserror,
)


# ---------------------------------------------------------------------------
# Unit tests for handle_stdin_oserror (real production function)
# ---------------------------------------------------------------------------

class TestHandleStdinOserror:
    """Exercise every branch of the OSError recovery function."""

    def test_non_recoverable_errno_returns_none(self):
        """An OSError whose errno is NOT in the recoverable set must signal
        re-raise (return None) so unexpected errors propagate normally."""
        exc = OSError(errno.ECONNRESET, "Connection reset")
        assert handle_stdin_oserror(exc, [], lambda r: None) is None

    def test_einval_recoverable_under_limit(self):
        """EINVAL is the primary Windows symptom — recoverable, returns True."""
        exc = OSError(errno.EINVAL, os.strerror(errno.EINVAL))
        times: list[float] = []
        assert handle_stdin_oserror(exc, times, lambda r: None) is True
        assert len(times) == 1

    def test_ebadf_recoverable(self):
        """EBADF (child closed inherited handle) is also recoverable."""
        exc = OSError(errno.EBADF, os.strerror(errno.EBADF))
        assert handle_stdin_oserror(exc, [], lambda r: None) is True

    def test_epipe_recoverable(self):
        """EPIPE on a read path is recoverable on Windows."""
        exc = OSError(errno.EPIPE, os.strerror(errno.EPIPE))
        assert handle_stdin_oserror(exc, [], lambda r: None) is True

    def test_rate_limit_exceeded_returns_false(self):
        """More than MAX_RECOVERIES_PER_MINUTE recoveries → graceful break."""
        exc = OSError(errno.EINVAL, os.strerror(errno.EINVAL))
        # Already at the cap with fresh timestamps.
        now = time.time()
        times = [now] * MAX_RECOVERIES_PER_MINUTE
        assert handle_stdin_oserror(exc, times, lambda r: None) is False

    def test_rate_limit_logs_before_breaking(self):
        """The log_fn must be called so the crash-log explains the exit."""
        exc = OSError(errno.EINVAL, os.strerror(errno.EINVAL))
        now = time.time()
        times = [now] * MAX_RECOVERIES_PER_MINUTE
        messages: list[str] = []
        handle_stdin_oserror(exc, times, messages.append)
        assert any("rate" in m.lower() or "exceed" in m.lower() for m in messages)

    def test_recovery_times_pruned(self):
        """Recovery timestamps older than 60s are pruned on each call."""
        exc = OSError(errno.EINVAL, os.strerror(errno.EINVAL))
        old = time.time() - 120  # 2 minutes ago — outside the window
        times = [old, old]
        handle_stdin_oserror(exc, times, lambda r: None)
        # Old entries removed, new entry appended.
        assert all(t > old for t in times)
        assert len(times) == 1

    def test_recoverable_logs_retry_message(self):
        """A successful recovery logs a diagnostic line."""
        exc = OSError(errno.EINVAL, os.strerror(errno.EINVAL))
        messages: list[str] = []
        handle_stdin_oserror(exc, [], messages.append)
        assert len(messages) == 1
        assert "EINVAL" in messages[0] or "retry" in messages[0].lower()

    def test_shared_rate_limit_with_spurious_eof(self):
        """OSError recovery and spurious-EOF recovery share the same
        recovery_times list so a single rate-limit budget covers both."""
        exc = OSError(errno.EINVAL, os.strerror(errno.EINVAL))
        times = [time.time()]  # one spurious-EOF recovery already consumed
        handle_stdin_oserror(exc, times, lambda r: None)
        assert len(times) == 2  # shared list grew


# ---------------------------------------------------------------------------
# Source-level verification: both entry points wire up OSError handling
# ---------------------------------------------------------------------------

def _source(filename: str) -> str:
    import pathlib
    here = pathlib.Path(__file__).resolve()
    repo_root = here.parent.parent.parent
    return (repo_root / "tui_gateway" / filename).read_text(encoding="utf-8")


def test_entry_wraps_readline_in_oserror_catch():
    """entry.py must catch OSError around sys.stdin.readline() and route it
    through handle_stdin_oserror instead of crashing (#78820)."""
    source = _source("entry.py")
    assert "handle_stdin_oserror" in source, (
        "entry.py must import and call handle_stdin_oserror"
    )
    assert "except OSError" in source, (
        "entry.py must catch OSError on the stdin read"
    )
    # The except must be near readline, not in an unrelated block.
    readline_idx = source.index("sys.stdin.readline()")
    except_idx = source.index("except OSError")
    assert except_idx > readline_idx, (
        "except OSError must follow the readline() call in entry.py"
    )


def test_slash_worker_wraps_readline_in_oserror_catch():
    """slash_worker.py must catch OSError around sys.stdin.readline() too."""
    source = _source("slash_worker.py")
    assert "handle_stdin_oserror" in source, (
        "slash_worker.py must import and call handle_stdin_oserror"
    )
    assert "except OSError" in source, (
        "slash_worker.py must catch OSError on the stdin read"
    )
    readline_idx = source.index("sys.stdin.readline()")
    except_idx = source.index("except OSError")
    assert except_idx > readline_idx, (
        "except OSError must follow the readline() call in slash_worker.py"
    )


# ---------------------------------------------------------------------------
# Loop-pattern integration test (mirrors the exact entry.py loop structure)
# ---------------------------------------------------------------------------

def test_loop_pattern_recovers_then_reads_line():
    """Exercise the loop pattern used by entry.py: an OSError on the first
    read is caught and retried, then a valid JSON line is read successfully.

    This mirrors the production loop's control flow (try/except OSError →
    handle_stdin_oserror → continue/break/raise) using the real function.
    """
    import json

    class _FakeStdin:
        """Yields OSError once, then a JSON line, then EOF."""
        def __init__(self):
            self._n = 0

        def readline(self):
            self._n += 1
            if self._n == 1:
                raise OSError(errno.EINVAL, os.strerror(errno.EINVAL))
            if self._n == 2:
                return json.dumps({"jsonrpc": "2.0", "method": "ping"}) + "\n"
            return ""  # genuine EOF

    fake = _FakeStdin()
    recovery_times: list[float] = []
    lines_read: list[str] = []

    # Exact mirror of the entry.py / slash_worker.py stdin loop.
    while True:
        try:
            raw = fake.readline()
        except OSError as exc:
            action = handle_stdin_oserror(exc, recovery_times, lambda r: None)
            if action is None:
                raise
            if action:
                continue
            break
        if not raw:
            break
        line = raw.strip()
        if not line:
            continue
        lines_read.append(line)

    assert len(lines_read) == 1
    parsed = json.loads(lines_read[0])
    assert parsed["method"] == "ping"


def test_loop_pattern_non_recoverable_errno_propagates():
    """An unexpected OSError errno must propagate (not be swallowed)."""
    class _FakeStdin:
        def readline(self):
            raise OSError(errno.ECONNRESET, "Connection reset")

    fake = _FakeStdin()

    with pytest.raises(OSError):
        while True:
            try:
                raw = fake.readline()
            except OSError as exc:
                action = handle_stdin_oserror(exc, [], lambda r: None)
                if action is None:
                    raise
                if action:
                    continue
                break
            if not raw:
                break
