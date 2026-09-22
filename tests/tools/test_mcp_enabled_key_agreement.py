"""One parser for an MCP server's ``enabled`` key: every reader must agree.

Invariant under test
--------------------
Seven call paths answer "is this server enabled?" from the same
``mcp_servers.<name>`` block, and each used to carry its own reading of the value:

* ``tools/mcp_tool_common._parse_boolish`` — the MCP client, discovery and
  registration. A *numeric* value fell through to the fallback, so a
  hand-written ``enabled: 0`` read as **on**.
* ``hermes_cli/tools_config._parse_enabled_flag`` — the platform/cron/oneshot/
  TUI resolvers, where ``int`` was falsey and ``enabled: 0`` read as **off**.
* ``hermes_cli/mcp_catalog.server_enabled`` and the ``hermes mcp list`` row —
  a string was matched against ``{"true", "1", "yes"}``, so ``enabled: "on"``
  read as **disabled** while every other reader called it enabled.
* ``tui_gateway/mcp_rpc_helpers.summarize_server``,
  ``hermes_cli/web_server_mcp._mcp_server_summary`` and ``acp_adapter/session``
  compared with ``is not False``, so ``enabled: 0`` read as **on** there too.

All of them now call :func:`utils.parse_boolish`: one rule for the key, one
fallback (``default``, with a warning) for a value that isn't boolean-ish. These
tests pin the *agreement* between the readers rather than any single
implementation's output, so a future reader that grows its own rule fails here.
"""

from __future__ import annotations

import re

import pytest

from hermes_cli.mcp_catalog import server_enabled
from hermes_cli.tools_config import _parse_enabled_flag, enabled_mcp_server_names
from hermes_cli.web_server_mcp import _mcp_server_summary
from tools.mcp_tool_discovery import _enabled as discovery_enabled
from tools.mcp_tool_registration import _server_enabled as registration_enabled
from tui_gateway.mcp_rpc_helpers import summarize_server
from utils import parse_boolish

# Every shape a hand-edited (or JSON-ish) config.yaml can hold for this key:
# YAML booleans, the numbers YAML 1.1 users write for on/off, and the word
# strings that reach the same code path through JSON or a shell-written config.
BOOLISH_VALUES = [
    (True, True),
    (False, False),
    (0, False),
    (1, True),
    (2, True),
    (0.0, False),
    (1.0, True),
    (None, True),
    ("true", True),
    ("TRUE", True),
    (" 1 ", True),
    ("yes", True),
    ("on", True),
    ("false", False),
    ("0", False),
    ("no", False),
    ("off", False),
]


@pytest.mark.parametrize("value,expected", BOOLISH_VALUES)
def test_every_reader_of_the_enabled_key_agrees(value, expected):
    assert parse_boolish(value, default=True) is expected
    assert _parse_enabled_flag(value, default=True) is expected
    assert discovery_enabled({"enabled": value}) is expected
    assert registration_enabled({"enabled": value}) is expected
    assert server_enabled({"enabled": value}) is expected
    assert summarize_server("srv", {"enabled": value})["enabled"] is expected
    assert _mcp_server_summary("srv", {"enabled": value})["enabled"] is expected
    names = enabled_mcp_server_names({"mcp_servers": {"srv": {"enabled": value}}})
    assert ("srv" in names) is expected


def test_enabled_zero_is_disabled_for_the_client_and_the_resolvers():
    """The reported drift: the MCP client read ``enabled: 0`` as on, resolvers as off."""
    assert discovery_enabled({"enabled": 0}) is False
    assert registration_enabled({"enabled": 0}) is False
    assert parse_boolish(0, default=True) is False
    assert _parse_enabled_flag(0, default=True) is False
    assert ("srv" in enabled_mcp_server_names({"mcp_servers": {"srv": {"enabled": 0}}})) is False


def test_hermes_mcp_list_shows_the_state_the_runtime_uses(monkeypatch, capsys):
    """``hermes mcp list`` printed ``enabled: "on"`` as disabled (only true/1/yes matched)."""
    from hermes_cli import mcp_config

    monkeypatch.setattr(
        mcp_config,
        "_get_mcp_servers",
        lambda config=None: {"srv": {"url": "https://example.invalid/mcp", "enabled": "on"}},
    )
    mcp_config.cmd_mcp_list()

    row = next(line for line in capsys.readouterr().out.splitlines() if "srv" in line)
    assert "enabled" in row
    assert not re.search(r"disabled", row)


def test_unrecognized_value_falls_back_to_the_default():
    assert parse_boolish("maybe", default=True) is True
    assert parse_boolish("maybe", default=False) is False
    assert parse_boolish({"nested": True}, default=True) is True
    assert parse_boolish(None, default=False) is False
