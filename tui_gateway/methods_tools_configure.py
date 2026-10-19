"""``tools.configure``'s toolset/MCP config write for one session (``methods_tools`` split).

Moved verbatim from ``tui_gateway/methods_tools.py`` (file-line ratchet): the body closes over
server.py globals (``_sessions``, ``_err``, ``_profile_home_lease``, ...) through
``method_ctx.bind_module``; publication runs from the parent's ``register()``.
"""

from .method_ctx import bind_module


def _configure_session_tools(rid, params: dict, sid: str, session, home, incarnation) -> dict:
    action = str(params.get("action", "") or "").strip().lower()
    targets = [str(name).strip() for name in params.get("names", []) or [] if str(name).strip()]
    if action not in {"disable", "enable"}:
        return _err(rid, 4017, f"unknown tools action: {action}")
    if not targets:
        return _err(rid, 4018, "names required")
    # Lease the read/modify/write, not the potentially slow agent build below.
    with _profile_home_lease(home, incarnation) if session is not None else contextlib.nullcontext():
        with _sessions_lock:
            if session is not None and (
                    not _session_slot_current(sid, session)
                    or not _session_profile_identity_matches(session, home, incarnation)):
                return _err(rid, 4001, "session changed during tools.configure")
        hc, tc = _tools_mod("hermes_cli.config"), _tools_mod("hermes_cli.tools_config")
        cfg = hc.load_config()
        valid_toolsets = {ts_key for ts_key, _, _ in tc.CONFIGURABLE_TOOLSETS} | tc._get_plugin_toolset_keys()
        mcp_targets = [name for name in targets if ":" in name]
        unknown = [name for name in targets if ":" not in name and name not in valid_toolsets]
        toolset_targets = [name for name in targets if ":" not in name and name in valid_toolsets]
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
    info = _reset_session_agent(sid, session) if session else None
    changed = [
        name for name in targets
        if name not in unknown and (":" not in name or name.split(":", 1)[0] not in missing_servers)]
    return _ok(rid, {
        "changed": changed, "enabled_toolsets": enabled, "info": info,
        "missing_servers": sorted(missing_servers), "reset": bool(session), "unknown": unknown})


def register(server) -> None:
    """Publish this module's helpers onto ``server`` (rebound to its globals)."""
    bind_module(globals(), server, skip=("_",))
