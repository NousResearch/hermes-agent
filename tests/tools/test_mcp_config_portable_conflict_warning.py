"""A same-name portable/native MCP conflict is a configuration the user chose once, but
``_portable_mcp_servers`` runs on every ``_load_mcp_config`` (about once a minute in a gateway), so the
warning repeated 1k+ times a day in ``logs/errors.log`` and buried real warnings. Pin one warning per
process per server, and prove the repeat still reaches the debug log.
"""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools import mcp_tool_config as _mcp_config
from tools.mcp_tool_config import _portable_mcp_servers

PORTABLE = {"rea": {"command": "npx", "args": ["-y", "rea-agents@6.7.0", "mcp"]}}
NATIVE = {"rea": {"command": "python3", "args": ["s.py"]}}


@pytest.fixture(autouse=True)
def _reset_dedupe():
    # getattr: the sabotage run (source reverted, this test kept) has no dedupe set to clear, and a
    # bare attribute access would turn every case into a setup ERROR instead of the behavioural assert.
    _clear = getattr(_mcp_config, "_portable_conflict_warned", set()).clear
    _clear()
    yield
    _clear()


def _run(safe_servers, portable=PORTABLE, rejected=frozenset()):
    manager = SimpleNamespace(get_portable_mcp_servers=lambda: {k: dict(v) for k, v in portable.items()})
    with patch("hermes_cli.plugins.discover_plugins"), \
            patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        _portable_mcp_servers(safe_servers, rejected)


def _conflict_records(caplog):
    return [r for r in caplog.records if "conflicts with native config" in r.getMessage()]


def test_native_override_warns_once_per_process(caplog):
    with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
        _run(dict(NATIVE))
        _run(dict(NATIVE))
    warnings = [r for r in _conflict_records(caplog) if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "skipping" in warnings[0].getMessage()
    # The repeat is still visible at DEBUG, so -v logging keeps the every-load trace.
    assert [r for r in _conflict_records(caplog) if r.levelno == logging.DEBUG]


def test_carried_package_costs_warn_once(caplog):
    portable = {"rea": {**PORTABLE["rea"], "tool_costs": {"lookup": [1, 2]}}}
    safe = {}
    with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
        # Each load rebuilds the native entry from config.yaml, so the carry repeats every time.
        for _ in range(2):
            safe = dict(NATIVE)
            _run(safe, portable=portable)
    # Behaviour unchanged: the package's declared costs still ride the native entry.
    assert safe["rea"]["tool_costs"] == {"lookup": [1, 2]}
    warnings = [r for r in _conflict_records(caplog) if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "declared" in warnings[0].getMessage()


def test_rejected_native_entry_warns_once(caplog):
    portable = {"rea": PORTABLE["rea"]}
    with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
        safe = {}
        _run(safe, portable=portable, rejected=frozenset({"rea"}))
        _run(safe, portable=portable, rejected=frozenset({"rea"}))
    assert "rea" not in safe
    records = [r for r in caplog.records if "native entry of that name was rejected" in r.getMessage()]
    assert [r.levelno for r in records].count(logging.WARNING) == 1


def test_each_conflicting_server_warns_on_its_own(caplog):
    portable = {"rea": PORTABLE["rea"], "other": {"command": "npx", "args": ["other"]}}
    with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
        safe = {"rea": dict(NATIVE["rea"]), "other": {"command": "python3", "args": ["o.py"]}}
        _run(safe, portable=portable)
    warnings = [r for r in _conflict_records(caplog) if r.levelno == logging.WARNING]
    assert len(warnings) == 2


def test_config_reloads_log_the_conflict_once(tmp_path, monkeypatch, caplog):
    """End to end through the real load path: a gateway reload sees the merge, the operator sees one line."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager = SimpleNamespace(get_portable_mcp_servers=lambda: {k: dict(v) for k, v in PORTABLE.items()})
    with patch("hermes_cli.config.load_config", return_value={"mcp_servers": dict(NATIVE)}), \
            patch("hermes_cli.plugins.discover_plugins"), \
            patch("hermes_cli.plugins.get_plugin_manager", return_value=manager), \
            caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
        first = _mcp_config._load_mcp_config()
        second = _mcp_config._load_mcp_config()
    assert first["rea"]["command"] == "python3" and second["rea"]["command"] == "python3"
    warnings = [r for r in _conflict_records(caplog) if r.levelno == logging.WARNING]
    assert len(warnings) == 1
