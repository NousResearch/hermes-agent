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
    from utils import parse_boolish

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
        "enabled": parse_boolish(cfg.get("enabled", True), default=True),
        "tools": cfg.get("tools"),
        "source": "plugin" if plugin is not None else "config",
        "plugin": plugin}


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Optional  # noqa: F401,E402
from typing import Tuple  # noqa: F401,E402

def resolve_profile(rid, params, err_fn) -> Tuple[Optional[Any], Optional[dict]]:
    """Resolve the optional ``profile`` param to a HERMES_HOME override token.

    Returns ``(token, error)``: ``token`` is None for the launch profile (no
    override) or an opaque reset token; ``error`` is a JSON-RPC error dict
    (built via ``err_fn``) when the named profile doesn't exist. Callers reset
    ``token`` in a finally via :func:`reset_profile`.
    """
    profile = str(params.get("profile") or "").strip()
    if not profile:
        return None, None
    from hermes_cli.profiles import get_profile_dir
    from hermes_constants import set_hermes_home_override

    try:
        profile_dir = get_profile_dir(profile)
    except ValueError:
        return None, err_fn(rid, 4064, f"profile '{profile}' not found")
    if not profile_dir or not profile_dir.is_dir():
        return None, err_fn(rid, 4064, f"profile '{profile}' not found")
    return set_hermes_home_override(str(profile_dir)), None
# ---- END PLUGIN-COMPAT ----
