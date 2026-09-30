"""Session tool configuration JSON-RPC handler."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_callbacks import _reset_session_agent
    from .methods_tools import _mcp_plugin_write_error, _mcp_server_rows, _tools_mod
    from .server import (
        ProfileUnavailableError,
        _bind_build_profile_scopes,
        _err,
        _methods,
        _ok,
        _release_build_profile_scopes,
        _sess_nowait,
    )

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


@method("tools.configure")
def _(rid, params: dict) -> dict:
    try:
        return _tools_configure_request(rid, params)
    except ProfileUnavailableError:
        raise
    except Exception as exc:
        return _err(rid, 5035, str(exc))


def _tools_configure_request(rid, params: dict) -> dict:
    sid = params.get("session_id", "")
    session = None
    if sid:
        session, err = _sess_nowait(params, rid)
        if err:
            return err
    # The client sends session_id, not profile; the live session is authoritative.
    home = (session or {}).get("profile_home")
    scopes = _bind_build_profile_scopes(home)
    try:
        return _configure_session_tools(rid, params, sid, session)
    finally:
        _release_build_profile_scopes(scopes)


def _mcp_excluded_tools(config: dict) -> dict[str, frozenset[str]]:
    return {name: frozenset((server.get("tools") or {}).get("exclude") or [])
            for name, server in (config.get("mcp_servers") or {}).items()}


def _configure_session_tools(rid, params: dict, sid: str, session) -> dict:
    action = str(params.get("action", "") or "").strip().lower()
    targets = [str(name).strip() for name in params.get("names", []) or [] if str(name).strip()]
    if action not in {"disable", "enable"}:
        return _err(rid, 4017, f"unknown tools action: {action}")
    if not targets:
        return _err(rid, 4018, "names required")
    hc, tc = _tools_mod("hermes_cli.config"), _tools_mod("hermes_cli.tools_config")
    cfg = hc.load_config()
    enabled_before = tc._get_platform_tools(cfg, "cli", include_default_mcp_servers=False)
    excluded_before = _mcp_excluded_tools(cfg)
    valid_toolsets = {ts_key for ts_key, _, _ in tc.CONFIGURABLE_TOOLSETS} | tc._get_plugin_toolset_keys()
    mcp_targets = [name for name in targets if ":" in name]
    rejections = _tools_mod("hermes_cli.tools_config_mcp").toolset_rejections(
        [name for name in targets if ":" not in name], "cli")
    unknown = [name for name in targets if ":" not in name and name not in valid_toolsets]
    rejected = {name: message for name, message in rejections.items() if name in valid_toolsets}
    toolset_targets = [name for name in targets if ":" not in name and name not in rejections]
    if toolset_targets:
        tc._apply_toolset_change(cfg, "cli", toolset_targets, action)
    plugins = _mcp_server_rows()[1]
    for target in mcp_targets:
        server_name = target.split(":", 1)[0]
        if err := _mcp_plugin_write_error(rid, server_name, plugins):
            return err
    missing_servers = tc._apply_mcp_change(cfg, mcp_targets, action) if mcp_targets else set()
    hc.save_config(cfg)
    enabled = sorted(tc._get_platform_tools(hc.load_config(), "cli", include_default_mcp_servers=False))
    selection_changed = set(enabled) != enabled_before or _mcp_excluded_tools(cfg) != excluded_before
    reset = bool(session) and selection_changed
    info = _reset_session_agent(sid, session) if reset else None
    changed = [
        name for name in targets
        if name not in rejections and (":" not in name or name.split(":", 1)[0] not in missing_servers)]
    return _ok(rid, {
        "changed": changed, "enabled_toolsets": enabled, "info": info,
        "missing_servers": sorted(missing_servers), "rejected": rejected, "reset": reset, "unknown": unknown})


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
