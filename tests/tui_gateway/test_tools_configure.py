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


@pytest.mark.parametrize("action", ["enable", "disable"])
def test_tools_configure_classifies_targets_from_one_catalogue(monkeypatch, action):
    from hermes_cli import tools_config
    from hermes_cli.config import save_config
    from tui_gateway.server import _methods

    reads = 0

    def discovered_keys():
        nonlocal reads
        reads += 1
        return {"late_plugin"} if reads >= 3 else set()

    monkeypatch.setattr(tools_config, "_get_plugin_toolset_keys", discovered_keys)
    save_config({"platform_toolsets": {"cli": ["file"]}})
    response = _methods["tools.configure"]("catalogue", {"action": action, "names": ["late_plugin"]})
    result = response["result"]
    assert {key: result[key] for key in ("changed", "unknown", "rejected")} == {
        "changed": [], "unknown": ["late_plugin"], "rejected": {},
    }
