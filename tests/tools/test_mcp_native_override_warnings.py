"""A native portable-server override is reported once per home and cost shape."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools import mcp_tool_config


@pytest.mark.parametrize("costed", [False, True])
def test_native_override_is_reported_once_for_each_profile(monkeypatch, caplog, costed):
    monkeypatch.setattr(mcp_tool_config, "_REPORTED_NATIVE_OVERRIDES", set())
    home = ["profile-a"]
    monkeypatch.setattr("hermes_constants.hermes_home_key", lambda: home[0])
    portable = {"command": "python3", "args": ["server.py"]}
    if costed:
        portable["tool_costs"] = {"render": 2}
    manager = SimpleNamespace(get_portable_mcp_servers=lambda: {"srv": dict(portable)})
    with patch("hermes_cli.plugins.discover_plugins"), patch("hermes_cli.plugins.get_plugin_manager", return_value=manager), \
            caplog.at_level("WARNING", logger="tools.mcp_tool"):
        for profile in ("profile-a", "profile-b"):
            home[0] = profile
            for _ in range(3):
                native = {"srv": {"command": "uvx", "args": ["srv-mcp"]}}
                mcp_tool_config._portable_mcp_servers(native)
                assert native["srv"]["command"] == "uvx"
                if costed:
                    assert native["srv"]["tool_costs"] == {"render": 2}
            assert len([r for r in caplog.records if "conflicts with native config" in r.message]) == (1 if profile == "profile-a" else 2)
