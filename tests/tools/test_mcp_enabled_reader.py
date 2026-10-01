"""Every reader of ``mcp_servers.<name>.enabled`` gives one answer per value.

The MCP client, the toolset resolver, the profile editor, the catalog and the list/status
surfaces used to parse the key four different ways, so ``enabled: 0`` was off for the resolver
and on for the client, and ``enabled: "false"`` was on in the server list. The case table is
shared with the desktop (``apps/desktop/src/lib/mcp-enabled-cases.json``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

CASES = json.loads(
    (Path(__file__).resolve().parents[2] / "apps/desktop/src/lib/mcp-enabled-cases.json").read_text(encoding="utf-8"))


def _readers():
    from hermes_cli.mcp_catalog import server_enabled
    from hermes_cli.tools_config import enabled_mcp_server_names
    from tools.mcp_tool_common import mcp_server_enabled
    from tui_gateway.mcp_rpc_helpers import summarize_server
    from tui_gateway.methods_profiles import _mcp_entry_enabled

    return {
        "mcp client": mcp_server_enabled,
        "toolset resolver": lambda cfg: "s" in enabled_mcp_server_names({"mcp_servers": {"s": cfg}}),
        "profile editor": _mcp_entry_enabled,
        "catalog": server_enabled,
        "server list": lambda cfg: summarize_server("s", cfg)["enabled"],
    }


@pytest.mark.parametrize("case", CASES, ids=lambda c: repr(c.get("enabled", "<absent>")))
def test_every_reader_agrees_with_the_shared_table(case):
    entry = {k: v for k, v in case.items() if k != "on"}
    answers = {name: bool(read({"command": "x", **entry})) for name, read in _readers().items()}

    assert answers == dict.fromkeys(answers, case["on"])
