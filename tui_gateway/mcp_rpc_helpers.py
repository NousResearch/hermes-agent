"""Shared helpers for the per-profile MCP lifecycle RPCs (mcp.servers.*).

Published onto ``tui_gateway.server`` as ``_mcp_summarize_server`` so the rebound handler
bodies in methods_tools resolve it.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


def server_configs_with_sources(config_servers: Mapping[str, dict]) -> tuple[Dict[str, dict], Dict[str, str | None]]:
    servers = {name: dict(cfg) for name, cfg in config_servers.items() if isinstance(cfg, dict)}
    plugins: Dict[str, str | None] = {name: None for name in servers}
    try:
        from hermes_cli.plugins import discover_plugins, get_plugin_manager
        from tools.mcp_tool_config import _filter_suspicious_mcp_servers

        discover_plugins()
        manager = get_plugin_manager()
        portable = _filter_suspicious_mcp_servers(manager.get_portable_mcp_servers())
        owners = manager.get_portable_mcp_server_plugins()
        for name, cfg in portable.items():
            if name not in servers:
                servers[name] = dict(cfg)
                plugins[name] = owners.get(name)
    except Exception:
        pass
    return servers, plugins


def summarize_server(name: str, cfg: dict, plugin: str | None = None) -> Dict[str, Any]:
    from hermes_cli.mcp_config import _oauth_tokens_present
    from tools.mcp_tool_common import mcp_server_enabled

    cfg = cfg if isinstance(cfg, dict) else {}
    transport = "http" if cfg.get("url") else ("stdio" if cfg.get("command") else "unknown")
    auth = cfg.get("auth")
    headers = cfg.get("headers") or {}
    if not auth and isinstance(headers, dict) and any(str(key).lower() == "authorization" for key in headers):
        auth = "header"
    return {
        "name": name,
        "transport": transport,
        "url": cfg.get("url"),
        "command": cfg.get("command"),
        "args": list(cfg.get("args") or []),
        "env": sorted(str(k) for k in (cfg.get("env") or {})),
        "auth": auth,
        "oauth_tokens_present": _oauth_tokens_present(name) if auth == "oauth" else None,
        "enabled": mcp_server_enabled(cfg),
        "tools": cfg.get("tools"),
        "source": "plugin" if plugin is not None else "config",
        "plugin": plugin}


def record_mcp_add(entry: Any, server_config: Mapping[str, Any], saved: bool) -> None:
    """Count an ``mcp.servers.add`` as an MCP extension install. A save fails only on a suspicious
    command/args configuration (``_save_mcp_server`` returns False)."""
    from hermes_cli.mcp_catalog import record_mcp_install

    source = "catalog" if entry is not None else ("url" if server_config.get("url") else "local")
    record_mcp_install(source, entry.name if entry is not None else None, "success" if saved else "failed",
                       failure_class=None if saved else "config_rejected")
