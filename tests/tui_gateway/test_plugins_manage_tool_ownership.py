"""``plugins.manage list`` names the tools each package's own code produces (``tools`` /
``tool_prefixes``): Desktop lets a plugin's desktop half render media only from those, so the
answer must come from what the loader actually registered, never from a manifest claim.

Real discovery against the sandbox ``HERMES_HOME``: two native plugins and one portable MCP package.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.config import load_config, save_config
from tui_gateway import server

_NATIVE = """
def _handler(args, **_kw):
    return "{}"

def register(ctx):
    schema = {"name": "%(tool)s", "description": "d", "parameters": {"type": "object", "properties": {}}}
    ctx.register_tool(name="%(tool)s", toolset="%(name)s", schema=schema, handler=_handler)
    %(extra)s
"""

_OVERRIDE_TTS = (
    'ctx.register_tool(name="text_to_speech", toolset="%(name)s", override=True, handler=_handler, '
    'schema={"name": "text_to_speech", "description": "d", "parameters": {"type": "object", "properties": {}}})'
)


def _native_plugin(root: Path, name: str, tool: str, extra: str = "") -> None:
    plugin = root / name
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(f"name: {name}\nversion: '1.0'\ndescription: d\n")
    (plugin / "__init__.py").write_text(_NATIVE % {"name": name, "tool": tool, "extra": extra % {"name": name}})


def _portable_plugin(root: Path) -> None:
    from hermes_cli.agent_plugins import MCP_SCHEMA_V1, PLUGIN_SCHEMA_V1
    plugin = root / "meshy-ish"
    plugin.mkdir(parents=True)
    (plugin / "plugin.json").write_text(json.dumps({"$schema": PLUGIN_SCHEMA_V1, "name": "meshy-ish"}))
    (plugin / "mcp.json").write_text(json.dumps({
        "$schema": MCP_SCHEMA_V1, "mcpServers": {"meshy-3d": {"type": "stdio", "command": "python"}}}))


def test_list_reports_only_tools_the_package_itself_registered():
    root = Path(os.environ["HERMES_HOME"]) / "plugins"
    _native_plugin(root, "sfx", "sfx_make", _OVERRIDE_TTS)
    _native_plugin(root, "other", "other_tool")
    _portable_plugin(root)
    cfg = load_config()
    cfg["plugins"] = {"enabled": ["sfx", "other", "meshy-ish"],
                      "entries": {"sfx": {"allow_tool_override": True}}}
    save_config(cfg)

    # The package's MCP server as the MCP client records it once connected (no real server here),
    # plus a config server with a look-alike name that must not count for the package.
    from tools.mcp_tool_common import _core
    from tools.mcp_tool_scope import _resolve_server_key
    connected = {
        _resolve_server_key("meshy-3d"): SimpleNamespace(_registered_tool_names=["mcp__meshy_3d__download"]),
        _resolve_server_key("meshy_3d_x"): SimpleNamespace(_registered_tool_names=["mcp__meshy_3d_x__steal"]),
    }
    with patch.dict(_core._servers, connected), \
         patch("hermes_cli.plugins_cmd_catalog.catalog_rows_maps", return_value=({}, {}, {})):
        resp = server.handle_request({"id": "1", "method": "plugins.manage", "params": {"action": "list"}})

    rows = {row["name"]: row for row in resp["result"]["plugins"]}
    from hermes_cli.plugins import get_plugin_manager
    assert "text_to_speech" in get_plugin_manager()._plugins["sfx"].tools_registered  # the override took
    # The core voice-reply tool stays core even after an approved override; another plugin's tool
    # is never listed on this row.
    assert rows["sfx"]["tools"] == ["sfx_make"]
    assert rows["other"]["tools"] == ["other_tool"]
    # A portable package owns exactly the tools its own MCP server registered.
    assert rows["meshy-ish"]["tools"] == ["mcp__meshy_3d__download"]
