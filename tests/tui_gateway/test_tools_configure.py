"""``tools.configure`` accepts exactly the toolsets that ``hermes tools enable|disable`` accepts on ``cli``."""
from __future__ import annotations

import pytest


@pytest.mark.parametrize("action", ["enable", "disable"])
def test_tools_configure_rejects_toolsets_that_cli_does_not_allow(action):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.tools_config import _get_platform_tools
    from hermes_cli.tools_config_mcp import toolset_rejections
    from hermes_cli.toolset_scope import _TOOLSET_PLATFORM_RESTRICTIONS
    from tui_gateway.server import _methods

    restricted = sorted(name for name, platforms in _TOOLSET_PLATFORM_RESTRICTIONS.items() if "cli" not in platforms)
    assert restricted
    save_config({"platform_toolsets": {"cli": ["file"] if action == "enable" else ["file", "web"]}})
    names = [*restricted, "web", "no_such_toolset"]

    response = _methods["tools.configure"]("configure", {"action": action, "names": names})

    result = response["result"]
    enabled = sorted(_get_platform_tools(load_config(), "cli", include_default_mcp_servers=False))
    assert result == {
        "changed": ["web"],
        "enabled_toolsets": enabled,
        "info": None,
        "missing_servers": [],
        "rejected": toolset_rejections(restricted, "cli"),
        "reset": False,
        "unknown": ["no_such_toolset"],
    }
    assert ("web" in enabled) == (action == "enable")
