"""Application requirements compose with live, recycled, and lazy MCP connections."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from hermes_platform import declaration


@pytest.mark.parametrize("connection", ["live", "recycled", "lazy", "absent"])
def test_check_fn_composes_connection_and_registered_requirement(tmp_path, monkeypatch, connection):
    from tools import mcp_tool, mcp_tool_handlers
    from tools.mcp_tool_scope import _resolve_server_key

    monkeypatch.setattr(declaration, "_REGISTRY", {})
    core = mcp_tool
    monkeypatch.setattr(core, "_servers", {})
    monkeypatch.setattr(core, "_lazy_server_configs", {})
    monkeypatch.setattr(core, "_server_tool_scopes", {})
    name = "declaration-test-server"
    key = _resolve_server_key(name)
    if connection == "lazy":
        core._lazy_server_configs[key] = {}
    elif connection != "absent":
        core._servers[key] = SimpleNamespace(
            session=object() if connection == "live" else None,
            _is_recycled_stdio=lambda: connection == "recycled",
        )
    check = mcp_tool_handlers._make_check_fn(name)
    connected = connection != "absent"
    assert check() is connected

    location = tmp_path / "application.exe"
    decl = declaration.parse_declaration(
        name, {sys.platform: {"presence": "executable", "location": str(location)}},
        {"app": True}, where="test-plugin/plugin.yaml",
    )
    declaration.register(name, decl)
    assert check() is False
    location.write_text("presence fixture", encoding="utf-8")
    assert check() is connected
    location.unlink()
    assert check() is False
    declaration.clear()
    assert check() is connected
