"""Tests for portable plugin MCP server environment variable interpolation (#120526)."""

import os
from types import SimpleNamespace
from unittest.mock import patch


def test_portable_servers_interpolate_env_placeholders():
    """Portable plugin MCP server declarations interpolate ${VAR} from environment (#120526)."""
    portable = {
        "plugin__custom": {
            "command": "node",
            "args": ["server.js"],
            "env": {"API_KEY": "${PORTABLE_API_KEY}"},
            "headers": {"Authorization": "Bearer ${PORTABLE_TOKEN}"},
        }
    }
    manager = SimpleNamespace(get_portable_mcp_servers=lambda: portable)
    with (
        patch("hermes_cli.config.load_config", return_value={"mcp_servers": {}}),
        patch("hermes_cli.plugins.discover_plugins"),
        patch("hermes_cli.plugins.get_plugin_manager", return_value=manager),
        patch.dict(os.environ, {"PORTABLE_API_KEY": "key-12345", "PORTABLE_TOKEN": "token-abc"}),
    ):
        from tools.mcp_tool_config import _load_mcp_config

        result = _load_mcp_config()

    server = result["plugin__custom"]
    assert server["env"]["API_KEY"] == "key-12345"
    assert server["headers"]["Authorization"] == "Bearer token-abc"
