"""Regression tests for user-facing MCP status summaries."""

from __future__ import annotations

import types

import tools.mcp_tool as mcp_tool


def test_get_mcp_status_keeps_disabled_servers_labeled_disabled(monkeypatch):
    monkeypatch.setattr(
        mcp_tool,
        "_load_mcp_config",
        lambda: {
            "enabled_http": {"url": "https://example.test/mcp", "enabled": True},
            "disabled_stdio": {"command": "npx", "args": ["demo"], "enabled": False},
        },
    )

    with mcp_tool._lock:
        saved_servers = dict(mcp_tool._servers)
        mcp_tool._servers.clear()
        mcp_tool._servers["enabled_http"] = types.SimpleNamespace(
            session=object(),
            _registered_tool_names=["mcp_enabled_http_demo"],
            _tools=[],
            _sampling=None,
        )

    try:
        status = mcp_tool.get_mcp_status()
    finally:
        with mcp_tool._lock:
            mcp_tool._servers.clear()
            mcp_tool._servers.update(saved_servers)

    assert [entry["name"] for entry in status] == ["enabled_http", "disabled_stdio"]
    assert status[0]["connected"] is True
    assert status[0]["disabled"] is False
    assert status[0]["tools"] == 1
    assert status[1]["connected"] is False
    assert status[1]["disabled"] is True
    assert status[1]["tools"] == 0
