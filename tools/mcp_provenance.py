"""Compatibility-preserving MCP tool provenance helpers."""

import logging

logger = logging.getLogger(__name__)


def _track_mcp_tool_server(tool_name: str, server_name: str) -> None:
    """Remember the exact raw MCP server that registered *tool_name*."""
    from tools.mcp_tool import _lock, _mcp_tool_server_names
    with _lock:
        _mcp_tool_server_names[tool_name] = server_name


def _forget_mcp_tool_server(tool_name: str) -> None:
    """Forget MCP server provenance for a deregistered tool."""
    from tools.mcp_tool import _lock, _mcp_tool_server_names
    with _lock:
        _mcp_tool_server_names.pop(tool_name, None)
