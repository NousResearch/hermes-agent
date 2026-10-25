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


def _server_identity(cfg: object) -> tuple:
    """What a server config runs: a same-name native entry that runs the same thing is the user
    widening the package server's trust; anything else is the user's own, different server. Both
    sides compare after the connect-time ``${VAR}`` expansion (``load_config`` already expanded the
    native entry; the package's declaration keeps its placeholders until connect)."""
    cfg = _tools_mod("tools.mcp_tool_config")._interpolate_env_vars(cfg if isinstance(cfg, dict) else {})
    args = cfg.get("args") or ()
    return (str(cfg.get("command") or ""), tuple(str(a) for a in (args if isinstance(args, list) else [args])),
            str(cfg.get("url") or ""))


def _portable_server_tools(manager, native: dict) -> dict[str, set[str]]:
    """Tools each portable package's own MCP servers registered: live connections plus lazily
    cached (not yet connected) ones. A name the user's native config claims for a different server
    credits no package: the native entry is the one that runs (``_portable_mcp_servers``)."""
    core = _tools_mod("tools.mcp_tool_common")._core
    resolve_key = _tools_mod("tools.mcp_tool_scope")._resolve_server_key
    declared = manager.get_portable_mcp_servers()
    listed: dict[str, set[str]] = {}
    for server_name, owner in manager.get_portable_mcp_server_plugins().items():
        names = listed.setdefault(owner, set())
        if server_name in native and _server_identity(native[server_name]) != _server_identity(
                declared.get(server_name)):
            continue
        key = resolve_key(server_name)
        names.update(getattr(core._servers.get(key), "_registered_tool_names", None) or ())
        names.update(core._lazy_server_tool_names.get(key) or ())
    return listed


def _plugin_tool_ownership() -> dict[str, list[str]]:
    """The exact tool names each loaded plugin's OWN code registered, by manifest key.

    Desktop lets a plugin's desktop half render media only from these, so the answer comes
    from the loader's and the MCP client's records, never from a manifest claim or a name prefix.
    Native: names the plugin registered, minus built-in tool names (an approved override of
    ``text_to_speech`` is still the core voice reply to the user). Portable: the tools each MCP
    server the loader attributed to the package registered (``_portable_server_tools``; a server
    that never connected or cached its tools owns nothing yet). A name two packages list belongs
    to neither."""
    plugins_mod = _tools_mod("hermes_cli.plugins")
    toolsets = _tools_mod("toolsets")
    plugins_mod.discover_plugins()
    manager = plugins_mod.get_plugin_manager()
    native = _tools_mod("hermes_cli.config").load_config().get("mcp_servers")
    builtin = set(toolsets._HERMES_CORE_TOOLS)
    for entry in toolsets.TOOLSETS.values():
        builtin.update(entry.get("tools", ()))
    listed = {key: set(loaded.tools_registered) - builtin for key, loaded in manager._plugins.items()}
    for owner, names in _portable_server_tools(manager, native if isinstance(native, dict) else {}).items():
        listed.setdefault(owner, set()).update(names)
    seen: dict[str, int] = {}
    for names in listed.values():
        for name in names:
            seen[name] = seen.get(name, 0) + 1
    return {key: sorted(n for n in names if seen[n] == 1) for key, names in listed.items()}


def register(server) -> None:
    bind_module(globals(), server)
