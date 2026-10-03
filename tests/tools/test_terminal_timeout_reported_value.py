"""Reported terminal timeout must be the one the runtime enforces (#85809).

``/config`` printed the raw ``TERMINAL_TIMEOUT`` env value, so a user who set the documented
"0 = no timeout" misunderstanding saw "Timeout: 0s" — a value the terminal tool never uses (it
falls back to the default, which is why the real symptom was every command reporting an instantly
misleading "timed out after 0s"). The display must resolve the value the same way the tool does
and warn about the invalid setting instead of echoing it.
"""

import contextlib
import datetime
import io
import logging

import cli


def _stub_cli() -> "cli.HermesCLI":
    """Minimal object carrying only what show_config() reads (no REPL bootstrap)."""
    obj = object.__new__(cli.HermesCLI)
    obj.model = "test-model"
    obj.base_url = "http://localhost"
    obj.api_key = "x" * 20
    obj.max_turns = 90
    obj.enabled_toolsets = None
    obj.verbose = False
    obj.session_start = datetime.datetime.now()
    obj.agent = None
    return obj


def _timeout_line(out: str) -> str:
    matches = [line for line in out.splitlines() if "Timeout:" in line]
    assert len(matches) == 1, out
    return matches[0]


def test_config_display_reports_the_effective_timeout_not_an_invalid_raw_value(monkeypatch, caplog):
    monkeypatch.setenv("TERMINAL_TIMEOUT", "0")
    buf = io.StringIO()
    with caplog.at_level(logging.WARNING), contextlib.redirect_stdout(buf):
        _stub_cli().show_config()

    line = _timeout_line(buf.getvalue())
    assert line.split()[-1] == "180s", f"display must show the enforced default, got: {line!r}"

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "TERMINAL_TIMEOUT" in m and "must be > 0" in m and "180" in m for m in messages
    ), f"an invalid value must be reported loudly, got: {messages}"


def test_config_display_passes_a_valid_timeout_through_unchanged(monkeypatch):
    monkeypatch.setenv("TERMINAL_TIMEOUT", "600")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _stub_cli().show_config()
    assert _timeout_line(buf.getvalue()).strip().endswith("600s")
