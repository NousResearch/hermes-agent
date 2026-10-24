"""Per-server rows of an installed portable plugin for ``plugins.manage list`` (``methods_tools._plugin_rows``).

Bodies are rebound onto server.py's globals (method_ctx.bind_module) and reference them bare.
"""

from __future__ import annotations

from pathlib import Path

from .method_ctx import bind_module


def _plugin_server_rows(
    plugin_dir: Path | None, key: str, *, portable: bool,
    catalog_titles: dict[str, str] | None = None,
) -> list[dict]:
    if not portable or plugin_dir is None:
        return []
    package = _tools_mod("hermes_cli.agent_plugins").load_agent_plugin(plugin_dir, plugin_dir)
    namespace = package.manifest.get("extensions", {}).get("com.nousresearch.hermes", {})
    declared = namespace.get("servers", {})
    if not isinstance(declared, dict):
        return []
    server_name_for = _tools_mod("hermes_cli.plugins_manifest").portable_mcp_server_name
    liveness = _tools_mod("tools.mcp_liveness")
    core = _tools_mod("tools.mcp_tool_common")._core
    resolve_key = _tools_mod("tools.mcp_tool_scope")._resolve_server_key
    # The server sentence's app name: the curated catalog title when the package is a catalog
    # install, else the manifest name, else the server slug the declaration carries — a raw
    # slug reads like an error code (#119975). *catalog_titles* is pre-resolved by the caller
    # (one live-catalog resolution per listing): a per-plugin ``get_live_catalog_entry`` would
    # re-resolve the whole catalog once per installed plugin.
    display_name = str(package.manifest.get("name") or "") or None
    sidecar = _tools_mod("hermes_cli.plugins_cmd_catalog").catalog_install_record(plugin_dir)
    if sidecar:
        title = (catalog_titles or {}).get(str(sidecar.get("catalog_name") or ""))
        if title:
            display_name = title
    rows = []
    for name in sorted(declared):
        internal_name = server_name_for(key, name)
        connection_key = resolve_key(internal_name)
        server = core._servers.get(connection_key)
        connected = server is not None and (server.session is not None or server._is_recycled_stdio())
        if connected:
            rows.append({"name": name, "state": "connected", "sentence": ""})
            continue
        decl = _tools_mod("hermes_platform.declaration").lookup(internal_name)
        status = liveness.status(internal_name)
        if decl is None or status is None:
            rows.append({"name": name, "state": "unknown", "sentence": ""})
            continue
        rows.append({
            "name": name,
            "state": status.state,
            "sentence": liveness.describe(decl, status.availability, status.state, display_name),
        })
    return rows


def register(server) -> None:
    bind_module(globals(), server)
