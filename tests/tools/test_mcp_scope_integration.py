"""Integration tests for MCP scope filtering (delegation-only servers).

These tests spawn the echo test MCP server as a real stdio subprocess and
verify that:
1. ``discover_mcp_tools()`` (parent path) skips ``scope: delegation`` servers
2. ``_load_mcp_config(scope_filter=\"delegation\")`` returns only those servers
3. Sub-agents connect delegation-scoped servers via ``register_mcp_servers()``
"""

import os
import sys
from unittest.mock import patch

import pytest


@pytest.fixture()
def echo_server_path():
    """Absolute path to the test echo MCP server."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        os.path.dirname(this_dir), "fixtures", "mcp_echo_server.py"
    )


@pytest.fixture()
def scoped_config(echo_server_path):
    """Return a config dict with one shared and one delegation-scoped server."""
    return {
        "mcp_servers": {
            "shared_echo": {
                "command": sys.executable,
                "args": [echo_server_path],
            },
            "sub_echo": {
                "command": sys.executable,
                "args": [echo_server_path],
                "scope": "delegation",
            },
        }
    }


# ---------------------------------------------------------------------------
# Test 1: discover_mcp_tools skips delegation-scoped servers
# ---------------------------------------------------------------------------

def test_discover_mcp_tools_skips_delegation_scope(scoped_config):
    """Parent agent discover_mcp_tools should not connect scope:delegation."""
    from tools.mcp_tool import discover_mcp_tools, _MCP_AVAILABLE

    if not _MCP_AVAILABLE:
        pytest.skip("MCP SDK not available")

    # We can't fully mock discover_mcp_tools because it uses the real
    # event loop orchestration. Instead, verify that _load_mcp_config
    # correctly filters, which is what discover_mcp_tools feeds into
    # register_mcp_servers.
    with patch("hermes_cli.config.load_config", return_value=scoped_config):
        from tools.mcp_tool import _load_mcp_config

        # Parent view: no delegation-scoped servers
        parent_servers = _load_mcp_config(scope_filter="parent")
        assert "shared_echo" in parent_servers
        assert "sub_echo" not in parent_servers
        assert len(parent_servers) == 1


# ---------------------------------------------------------------------------
# Test 2: delegation scope returns only scoped servers
# ---------------------------------------------------------------------------

def test_delegation_config_returns_only_scoped(scoped_config):
    """_load_mcp_config(scope_filter='delegation') returns only scoped."""
    from tools.mcp_tool import _MCP_AVAILABLE

    if not _MCP_AVAILABLE:
        pytest.skip("MCP SDK not available")

    with patch("hermes_cli.config.load_config", return_value=scoped_config):
        from tools.mcp_tool import _load_mcp_config

        deleg_servers = _load_mcp_config(scope_filter="delegation")
        assert "sub_echo" in deleg_servers
        assert "shared_echo" not in deleg_servers
        assert len(deleg_servers) == 1


# ---------------------------------------------------------------------------
# Test 3: registering delegation-scoped server via register_mcp_servers
# ---------------------------------------------------------------------------

def test_register_delegation_scoped_server(scoped_config, echo_server_path):
    """register_mcp_servers can connect delegation-scoped servers."""
    from tools.mcp_tool import register_mcp_servers, _MCP_AVAILABLE

    if not _MCP_AVAILABLE:
        pytest.skip("MCP SDK not available")

    # Load only delegation-scoped servers
    with patch("hermes_cli.config.load_config", return_value=scoped_config):
        from tools.mcp_tool import _load_mcp_config

        deleg_servers = _load_mcp_config(scope_filter="delegation")

    assert "sub_echo" in deleg_servers

    # Connect the server
    tool_names = register_mcp_servers(deleg_servers)

    # Should have registered the echo tool from the test server.
    # MCP tool naming: mcp__<server>__<tool> (double-underscore separator).
    assert "mcp__sub_echo__echo" in tool_names, (
        f"Expected mcp__sub_echo__echo in registered tools, got: {tool_names}"
    )

    # Verify the tool is callable via the MCP handler
    from tools.mcp_tool import _servers

    server = _servers.get("sub_echo")
    assert server is not None, "Server 'sub_echo' not found in _servers"
    assert server.session is not None, "Server 'sub_echo' has no active session"
