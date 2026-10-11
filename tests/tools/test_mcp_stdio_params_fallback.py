"""Invariant: ``StdioServerParameters`` binding falls back to its defining submodule.

``mcp 2.0.0`` split the protocol types into ``mcp_types`` and rebuilt ``mcp/__init__.py`` on
top of it. On some 2.0 builds the top-level re-export is fragile: importing ``mcp`` succeeds
while ``mcp.StdioServerParameters`` still trips an intermediate import, so a flat
``_import_sdk_names("mcp", ("StdioServerParameters",))`` loses the name and every later
``tools.mcp_tool.StdioServerParameters`` access raises ``AttributeError`` — which surfaces as
"Failed to connect to MCP server ...: module 'tools.mcp_tool' has no attribute
'StdioServerParameters'" and parks the server. ``_bind_stdio_server_parameters`` must retry
against the defining submodule ``mcp.client.stdio``, which is stable across 1.x and 2.x.
"""

from tools import mcp_tool


def test_falls_back_to_defining_submodule_when_top_level_missing(monkeypatch):
    """A fragile top-level re-export must not lose StdioServerParameters."""
    calls = []

    def fake_import(module, names, missing_msg=None):
        calls.append(module)
        if module == "mcp.client.stdio" and "StdioServerParameters" in names:
            mcp_tool.StdioServerParameters = object()  # what the real call binds
            return True
        return False  # the top-level re-export is missing, as on the fragile build

    monkeypatch.setattr(mcp_tool, "_import_sdk_names", fake_import)
    assert mcp_tool._bind_stdio_server_parameters() is True
    assert calls == ["mcp", "mcp.client.stdio"]
    assert hasattr(mcp_tool, "StdioServerParameters")


def test_prefers_the_top_level_export_when_present(monkeypatch):
    """The normal path binds from ``mcp`` and never consults the submodule."""
    calls = []

    def fake_import(module, names, missing_msg=None):
        calls.append(module)
        if module == "mcp":
            mcp_tool.StdioServerParameters = object()
            return True
        return False

    monkeypatch.setattr(mcp_tool, "_import_sdk_names", fake_import)
    assert mcp_tool._bind_stdio_server_parameters() is True
    assert calls == ["mcp"]


def test_reports_failure_when_neither_path_yields_it(monkeypatch):
    """Both paths missing (no mcp installed) is a clean False, not an AttributeError."""
    monkeypatch.setattr(mcp_tool, "_import_sdk_names", lambda *a, **k: False)
    monkeypatch.delattr(mcp_tool, "StdioServerParameters", raising=False)
    assert mcp_tool._bind_stdio_server_parameters() is False
