"""A portable MCP server that clashes with a native entry warns once, not on every config load.

The same clash is found again on every ``_load_mcp_config()`` (each discovery pass and status read), so a
gateway logged it about once a minute for a user whose plugin and ``hermes mcp add`` both named the server.
"""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools import mcp_tool_config as _mcp_config

PORTABLE = {"letsfg": {"url": "https://letsfg.co/mcp", "strict_redirect_headers": True}}
NATIVE = {"letsfg": {"url": "https://letsfg.co/mcp", "auth": "oauth"}}


@pytest.fixture(autouse=True)
def _reset_dedupe():
    _mcp_config._portable_clash_warned.clear()
    yield
    _mcp_config._portable_clash_warned.clear()


def _load(portable, native, caplog):
    manager = SimpleNamespace(get_portable_mcp_servers=lambda: dict(portable))
    with (
        patch("hermes_cli.config.load_config", return_value={"mcp_servers": native}),
        patch("hermes_cli.plugins.discover_plugins"),
        patch("hermes_cli.plugins.get_plugin_manager", return_value=manager),
        caplog.at_level(logging.WARNING, logger="tools.mcp_tool"),
    ):
        return _mcp_config._load_mcp_config()


def _clash_warnings(caplog):
    return [r.getMessage() for r in caplog.records if "conflicts with native config" in r.getMessage()]


def test_repeated_loads_warn_once_and_keep_the_native_entry(caplog):
    for _ in range(3):
        servers = _load(PORTABLE, NATIVE, caplog)
        assert servers["letsfg"] == NATIVE["letsfg"]
    assert _clash_warnings(caplog) == ["Portable MCP server 'letsfg' conflicts with native config; skipping"]


def test_a_changed_outcome_warns_again(caplog):
    _load(PORTABLE, NATIVE, caplog)
    costed = {"letsfg": {**PORTABLE["letsfg"], "spend_limit": 50}}
    for _ in range(2):
        servers = _load(costed, NATIVE, caplog)
    assert servers["letsfg"]["spend_limit"] == 50
    assert len(_clash_warnings(caplog)) == 2
    assert "declared spend_limit" in _clash_warnings(caplog)[1]


def test_each_server_warns_for_itself(caplog):
    portable = {**PORTABLE, "other": {"url": "https://example.com/mcp"}}
    native = {**NATIVE, "other": {"url": "https://example.com/mcp", "auth": "oauth"}}
    _load(portable, native, caplog)
    _load(portable, native, caplog)
    assert len(_clash_warnings(caplog)) == 2
