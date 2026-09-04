"""Regression test: the TUI gateway entry point survives an ``OSError`` raised
directly out of ``sys.stdin.readline()``, instead of crashing unhandled.

Orca ADE's WebGL-rendered PTY on Windows raises ``OSError: [Errno 22] Invalid
argument`` out of ``readline()`` on the first reply, instead of returning an
empty string like a normal EOF. Before this fix nothing in the read loop
caught it, so it propagated to ``sys.excepthook`` (installed in
``tui_gateway/server.py``) and killed the gateway subprocess (#92284).

Harness: same style as tests/tui_gateway/test_entry_picker_prewarm.py.
"""

from __future__ import annotations

from tui_gateway import entry
from tui_gateway._stdin_recovery import MAX_RECOVERIES_PER_MINUTE, handle_stdin_read_error


class _FlakyStdin:
    """Raises ``OSError`` from ``readline()`` a fixed number of times, then EOFs."""

    def __init__(self, raise_count: int):
        self._remaining = raise_count
        self.read_attempts = 0

    def readline(self):
        self.read_attempts += 1
        if self._remaining > 0:
            self._remaining -= 1
            raise OSError(22, "Invalid argument")
        return ""


def _run_main(monkeypatch, stdin):
    monkeypatch.setattr(entry, "_install_sidecar_publisher", lambda: None)
    monkeypatch.setattr(entry, "ensure_mcp_discovery_started", lambda: None)
    monkeypatch.setattr(entry, "resolve_skin", lambda: "default")
    monkeypatch.setattr(entry.server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(entry, "write_json", lambda payload: True)
    monkeypatch.setattr(entry, "_log_exit", lambda reason: None)
    monkeypatch.setattr(entry, "_recovery_times", [])
    monkeypatch.setattr(entry.sys, "stdin", stdin)


def test_main_survives_oserror_from_readline(monkeypatch):
    """A transient OSError out of readline() must be retried, not crash.

    Returning at all (rather than propagating the OSError) proves the read
    loop caught it; the attempt count proves it actually retried the read
    instead of e.g. treating the exception as an immediate EOF.
    """
    stdin = _FlakyStdin(raise_count=2)
    _run_main(monkeypatch, stdin)

    entry.main()  # must not raise

    assert stdin.read_attempts == 3  # 2 failing reads + the EOF that ends the loop


def test_handle_stdin_read_error_gives_up_past_rate_limit():
    """A caller that keeps hitting the OSError must eventually be told to
    stop retrying rather than spin forever."""
    recovery_times: list[float] = []
    logged: list[str] = []
    exc = OSError(22, "Invalid argument")

    for _ in range(MAX_RECOVERIES_PER_MINUTE):
        assert handle_stdin_read_error(exc, recovery_times, logged.append) is True

    assert handle_stdin_read_error(exc, recovery_times, logged.append) is False
    assert "rate exceeded" in logged[-1]
