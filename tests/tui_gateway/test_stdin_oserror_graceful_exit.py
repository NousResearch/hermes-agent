"""Regression test: severed stdin pipe must exit cleanly, not crash.

On Windows the TUI gateway child's stdio pipe can be invalidated under a live
read loop (parent restarted/reloaded and orphaned it, console torn down) and
``sys.stdin.readline()`` raises ``OSError: [Errno 22] Invalid argument``.
Unhandled that is a traceback + exit 1, which the client surfaces as
"gateway exited — recovering your session" and instantly respawns into the
same broken pipe (reconnect storm). ``entry.main()`` must log forensics and
return normally instead; same for the slash worker loop (covered here via the
shared shape: both loops must survive ``OSError`` from ``readline``).

Harness: same style as test_entry_picker_prewarm.py — import the modules and
monkeypatch I/O collaborators (no subprocess, no real gateway).
"""

from __future__ import annotations

import io

from tui_gateway import entry
from hermes_cli import model_switch_providers


class _SeveredStdin(io.StringIO):
    """stdin whose pipe was invalidated: every read raises like Windows."""

    def readline(self, *args):
        raise OSError(22, "Invalid argument")


def _stub_entry_main(monkeypatch, stdin):
    monkeypatch.setattr(entry, "_install_sidecar_publisher", lambda: None)
    monkeypatch.setattr(entry, "ensure_mcp_discovery_started", lambda: None)
    monkeypatch.setattr(entry, "resolve_skin", lambda: "default")
    monkeypatch.setattr(entry.server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(entry, "handle_spurious_eof", lambda *a: False)
    monkeypatch.setattr(entry, "write_json", lambda payload: True)
    monkeypatch.setattr(
        model_switch_providers, "prewarm_picker_cache_async", lambda: None
    )
    monkeypatch.setattr(entry.sys, "stdin", stdin)


def test_main_returns_normally_on_stdin_oserror(monkeypatch):
    """main() must survive OSError from readline: log + return, never raise."""
    _stub_entry_main(monkeypatch, _SeveredStdin())

    reasons: list[str] = []
    monkeypatch.setattr(entry, "_log_exit", lambda reason: reasons.append(reason))
    appended: list[str] = []
    monkeypatch.setattr(
        entry, "_append_crash_log", lambda header, dump=None: appended.append(header)
    )

    entry.main()  # returning at all proves the OSError was handled

    assert reasons, "expected a [gateway-exit] reason for the severed stdin"
    assert any("stdin" in r for r in reasons), f"reason should name stdin: {reasons!r}"
    assert appended, "expected a crash-log forensics entry"
    assert any("stdin" in h for h in appended), f"forensics should name stdin: {appended!r}"


def test_main_still_returns_on_genuine_eof(monkeypatch):
    """The OSError guard must not change the genuine-EOF path (empty read)."""
    _stub_entry_main(monkeypatch, io.StringIO(""))
    monkeypatch.setattr(entry, "_log_exit", lambda reason: None)
    monkeypatch.setattr(entry, "_append_crash_log", lambda header, dump=None: None)

    entry.main()  # empty stdin -> EOF -> returns
