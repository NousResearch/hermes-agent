"""Session-owned connector UI RPCs; authorization is not consent to read app data.

Both calls run on the RPC pool. WS upgrade authentication (including legacy
local/SSH tokens) and live transport membership are the authority, not a
renderer-supplied profile or identity. No agent build, browser, or wait loop.
"""

import contextvars

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_CONNECTOR_RPC_METHODS = frozenset({"connectors.list", "connectors.connect"})
_connector_rpc_origin: contextvars.ContextVar[tuple | None] = contextvars.ContextVar("connector_rpc_origin", default=None)


def _capture_connector_rpc_owner(params):
    sid = params.get("session_id")
    _, owner = _current_session_steer_authority(sid if isinstance(sid, str) else "")
    _connector_rpc_origin.set((owner, owner.get("profile_home") if owner is not None else None))


def _connector_rpc_error(rid, code, reason, message):
    return _err(rid, code, message, data={"reason": reason})


def _connector_owner_matches(sid, owner, profile_home):
    _, current = _current_session_steer_authority(sid)
    return (current is owner and not owner.get("_finalized")
            and owner.get("profile_home") == profile_home)


def _connector_rpc(rid, params, action):
    sid = params.get("session_id")
    if not isinstance(sid, str) or not sid.strip():
        return _connector_rpc_error(rid, 4000, "INVALID_PARAMS", "session_id required")
    _, owner = _current_session_steer_authority(sid)
    origin = _connector_rpc_origin.get()
    if (owner is None or owner.get("_finalized")
            or origin is not None and (origin[0] is not owner or origin[1] != owner.get("profile_home"))):
        return _connector_rpc_error(rid, 4001, "NOT_OWNER", "session not found or not owned by this transport")
    if _session_uses_compute_host(owner):
        return _connector_rpc_error(rid, 5033, "UNSUPPORTED_RUNTIME", "Connectors must be managed on the session's compute host.")
    allowed = {"session_id"} if action == "status" else {"session_id", "connectors", "reconnect"}
    # Shared-primary routing adds this metadata; the live transport above owns authorization.
    allowed.add("profile")
    if set(params) - allowed:
        return _connector_rpc_error(rid, 4000, "INVALID_PARAMS", "unsupported connector parameters")
    args = {"action": action}
    if action != "status":
        import re
        slugs = params.get("connectors")
        if (not isinstance(slugs, list) or not slugs
                or any(not isinstance(slug, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", slug) is None for slug in slugs)
                or not isinstance(params.get("reconnect", False), bool)):
            return _connector_rpc_error(rid, 4000, "INVALID_PARAMS", "connectors must be nonempty slugs; reconnect must be boolean")
        args.update(action="reconnect" if params.get("reconnect", False) else "connect", connectors=slugs)
    profile_home = owner.get("profile_home")
    runtime_token = _current_runtime_session_record.set(owner)
    try:
        # Bind launch explicitly too: an ambient sibling-profile override must not
        # leak into a session whose profile_home=None means the launch profile.
        scope = {"profile_home": profile_home or str(_hermes_home)}
        with _session_profile_runtime_scope(scope):
            tokens = _set_session_context(owner["session_key"], cwd=_session_cwd(owner), ui_session_id=sid)
            try:
                result = _dispatch_connector_rpc(rid, sid, owner, profile_home, args)
            finally:
                _clear_session_context(tokens)
        if not _connector_owner_matches(sid, owner, profile_home):
            return _connector_rpc_error(rid, 4001, "NOT_OWNER", "session ownership changed")
        return result
    except Exception:
        # Do not send exception strings: HTTP errors can contain headers/tokens.
        return _connector_rpc_error(rid, 5034, "CONNECTOR_REQUEST_FAILED", "Connector request failed. Try again explicitly.")
    finally:
        _current_runtime_session_record.reset(runtime_token)


def _dispatch_connector_rpc(rid, sid, owner, profile_home, args):
    import model_tools
    from tools.tool_gateway.config import connectors_available
    from tui_gateway.connector_payload import connector_ui_payload

    agent = owner.get("agent")
    # A cold session has no cached grant yet. Resolve exactly as _make_agent
    # does, under that session's profile/cwd, without constructing an LLM.
    enabled = (agent.enabled_toolsets if agent is not None
               else _load_enabled_toolsets(_resolve_agent_platform(_session_source(owner))))
    disabled = agent.disabled_toolsets if agent is not None else None
    if ("manage_connections" not in model_tools._select_tool_names(enabled, disabled, quiet_mode=True)
            or not connectors_available()):
        if args["action"] == "status":
            return _ok(rid, {"available": False, "connectors": []})
        return _connector_rpc_error(rid, 4031, "CONNECTORS_UNAVAILABLE", "Connectors are not available in this session.")
    if not _connector_owner_matches(sid, owner, profile_home):
        return _connector_rpc_error(rid, 4001, "NOT_OWNER", "session ownership changed")
    raw = model_tools.handle_function_call(
        "manage_connections", args, task_id=owner["session_key"],
        session_id=getattr(agent, "session_id", None) or owner["session_key"],
        tool_call_id=f"connector-ui-{uuid.uuid4().hex}",
        enabled_toolsets=enabled, disabled_toolsets=disabled,
    )
    data = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(data, dict) or "error" in data:
        return _connector_rpc_error(rid, 5034, "CONNECTOR_REQUEST_FAILED", "Connector request failed or was refused by policy.")
    key = "connectors" if args["action"] == "status" else "results"
    if not isinstance(data.get(key), list) or any(not isinstance(row, dict) for row in data[key]):
        return _connector_rpc_error(rid, 5034, "INVALID_CONNECTOR_RESPONSE", "Connector service returned an invalid response.")
    if key == "connectors":
        return _ok(rid, {"available": True, "connectors": connector_ui_payload(data[key])})
    if not data["results"] or not isinstance(data.get("summary"), dict):
        return _connector_rpc_error(rid, 5034, "INVALID_CONNECTOR_RESPONSE", "Connector service returned no authorization results.")
    return _ok(rid, connector_ui_payload(data))


@method("connectors.list")
def _(rid, params):
    """{session_id} -> {available, connectors}; raw metadata additions survive."""
    return _connector_rpc(rid, params, "status")


@method("connectors.connect")
def _(rid, params):
    """{session_id, connectors, reconnect?} -> manage_connections' results/summary."""
    return _connector_rpc(rid, params, "connect")


def register(server):
    bind_module(globals(), server, skip=("_",))
    server._LONG_HANDLERS = server._LONG_HANDLERS | _CONNECTOR_RPC_METHODS
