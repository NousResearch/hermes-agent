"""Shared helpers for the per-profile MCP lifecycle RPCs (mcp.servers.*).

These live in their own module (not methods_tools) because methods_tools
handlers are rebound onto ``tui_gateway.server``'s globals at install time
(see method_ctx.HandlerRegistry.install); a plain module-level def in
methods_tools would not be reachable from a rebound handler body. Handlers
import these at call time instead.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


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

    profile_dir = get_profile_dir(profile)
    if not profile_dir or not profile_dir.is_dir():
        return None, err_fn(rid, 4064, f"profile '{profile}' not found")
    return set_hermes_home_override(str(profile_dir)), None


def reset_profile(token) -> None:
    if token is not None:
        try:
            from hermes_constants import reset_hermes_home_override

            reset_hermes_home_override(token)
        except Exception:
            pass


def summarize_server(name: str, cfg: dict) -> Dict[str, Any]:
    """Serialize one server's config for a UI (no secret values).

    Mirrors web_server._mcp_server_summary plus an ``oauth_tokens_present``
    flag so a UI can tell an OAuth server that still needs authentication from
    one already authenticated.
    """
    from hermes_cli.mcp_config import _oauth_tokens_present

    cfg = cfg if isinstance(cfg, dict) else {}
    transport = "http" if cfg.get("url") else ("stdio" if cfg.get("command") else "unknown")
    auth = cfg.get("auth")
    headers = cfg.get("headers") or {}
    if not auth and isinstance(headers, dict) and any(
        str(key).lower() == "authorization" for key in headers
    ):
        auth = "header"
    tokens_present = _oauth_tokens_present(name) if auth == "oauth" else None
    return {
        "name": name,
        "transport": transport,
        "url": cfg.get("url"),
        "command": cfg.get("command"),
        "args": list(cfg.get("args") or []),
        "env": sorted(str(k) for k in (cfg.get("env") or {})),
        "auth": auth,
        "oauth_tokens_present": tokens_present,
        "enabled": cfg.get("enabled", True) is not False,
        "tools": cfg.get("tools"),
    }
