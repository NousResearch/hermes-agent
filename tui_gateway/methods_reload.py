"""``reload.all`` (``/reload``, #159205) for the chats this sidecar hosts.

``.env`` → plugins → MCP servers → skills (:mod:`hermes_cli.chat_reload`), then one rebuild of
the live sessions' tool snapshots with ``preserve_prefix=False`` (the consented cache break; the
MCP pool and the plugin registry are process-global, as for ``reload.mcp``). The whole pass holds
``_mcp_reload_lock``, so a ``reload.mcp`` (or a second ``/reload``) never tears the registry down
mid-pass. Lock order: ``_mcp_reload_lock`` → the plugin go-live lock → the plugin manager's
discovery lock; nothing takes ``_mcp_reload_lock`` under either of the other two.

Bodies are rebound onto server.py's globals (method_ctx.bind_module) and reference them bare.
"""

from __future__ import annotations

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


def _reload_all_mcp_pass() -> dict:
    """``reload.mcp``'s full pass (launch profile, then every hosted profile home) returning the
    launch profile's server diff; marks the loaded revision so ``reload.mcp`` callers coalesce on it."""
    global _mcp_reload_gen, _mcp_reload_loaded_rev
    chat_reload = _tools_mod("hermes_cli.chat_reload")
    with _session_profile_runtime_scope({"profile_home": None}):
        diff = chat_reload.reload_mcp_servers()
        loaded = _compute_mcp_rev()
    with _sessions_lock:
        homes = {sess.get("profile_home") for sess in _sessions.values() if sess.get("agent") is not None}
    for home in sorted(homes - {None}):
        try:
            with _session_profile_runtime_scope({"profile_home": home}):
                _tools_mod("tools.mcp_tool_discovery").discover_mcp_tools()
        except Exception:
            logger.warning("MCP rediscovery failed for profile %s", home, exc_info=True)
    _mcp_reload_loaded_rev = loaded
    _mcp_reload_gen += 1
    return diff


@method("reload.all")
def _reload_all_rpc(rid, params: dict) -> dict:
    t = _tools_mod("agent.i18n").t
    chat_reload = _tools_mod("hermes_cli.chat_reload")
    session = _sessions.get(str(params.get("session_id") or ""))
    if not params.get("confirm") and _mcp_reload_confirm_required():
        return _ok(rid, {"status": "confirm_required", "message": t("cli.reload_all.confirm_required")})
    if session is not None and session.get("running"):
        return _ok(rid, {"status": "busy", "message": t("cli.reload_all.busy")})
    if session is not None and _session_uses_compute_host(session):
        return _err(rid, 5019, "/reload is not available for compute-host sessions; use /reload-mcp")
    agent = (session or {}).get("agent")
    report = chat_reload.ReloadReport()
    before = chat_reload.tool_names(agent)
    # Each step reports its own failure; _refresh_live_sessions logs a session it could not refresh.
    with _mcp_reload_lock:
        chat_reload.run_steps(report, mcp=_reload_all_mcp_pass,
                              scope=lambda: _session_profile_runtime_scope(session or {"profile_home": None}))
        _refresh_live_sessions(preserve_prefix=False)
    if agent is not None:
        chat_reload.record_tools(report, agent, before)
        if note := chat_reload.model_note(report):
            # The live-notes queue: the next tool result or user turn carries it, never a row of its own.
            _tools_mod("agent.plugin_live_notes").queue_plugin_notice(agent, "reload", note)
    if params.get("always"):
        try:
            from cli import save_config_value
            save_config_value("approvals.mcp_reload_confirm", False)
        except Exception:
            logger.warning("Failed to persist mcp_reload_confirm=false", exc_info=True)
    return _ok(rid, {"status": "reloaded", "output": "\n".join(chat_reload.summary_lines(report)),
                     "report": report.to_dict()})


def register(server) -> None:
    bind_module(globals(), server)
