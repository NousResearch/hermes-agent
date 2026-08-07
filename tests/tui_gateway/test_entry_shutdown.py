"""Regression tests for TUI gateway exit finalization."""

import tui_gateway.entry as entry


def test_log_exit_finalizes_sessions_before_return(monkeypatch, tmp_path, capsys):
    """stdin EOF/broken-pipe exits must not rely only on atexit.

    Regression for #60286: when the TUI command pipe closes after a tool result
    but before the final assistant response, relying on interpreter atexit can
    leave the live session to be classified later by the WS orphan reaper.  The
    signal path already finalizes sessions explicitly; clean gateway-exit paths
    should do the same before returning/exiting.
    """

    calls = []
    crash_log = tmp_path / "tui_gateway_crash.log"
    monkeypatch.setattr(entry, "_CRASH_LOG", str(crash_log))
    monkeypatch.setattr(entry.server, "_shutdown_sessions", lambda: calls.append("shutdown"))

    entry._log_exit("stdin EOF (TUI closed the command pipe)")

    assert calls == ["shutdown"]
    assert "stdin EOF (TUI closed the command pipe)" in crash_log.read_text()
    assert "[gateway-exit] stdin EOF" in capsys.readouterr().err
