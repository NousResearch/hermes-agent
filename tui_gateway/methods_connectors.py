"""Session-scoped connector RPCs and the connection-operation bridge.

Live transport ownership, not renderer-supplied profile or identity, authorizes requests.
The operation itself lives in ``tools.connectors.live``; this module reads and drives it and
pushes every transition to the session as ``connection.update``.
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


def _owned_session(rid, params):
    """(owner, None) for a session this transport owns; (None, error reply) otherwise."""
    sid = params.get("session_id")
    if not isinstance(sid, str) or not sid.strip():
        return None, _connector_rpc_error(rid, 4000, "INVALID_PARAMS", "session_id required")
    _, owner = _current_session_steer_authority(sid)
    origin = _connector_rpc_origin.get()
    if (owner is None or owner.get("_finalized")
            or origin is not None and (origin[0] is not owner or origin[1] != owner.get("profile_home"))):
        return None, _connector_rpc_error(rid, 4001, "NOT_OWNER", "session not found or not owned by this transport")
    if _session_uses_compute_host(owner):
        return None, _connector_rpc_error(rid, 5033, "UNSUPPORTED_RUNTIME", "Connectors must be managed on the session's compute host.")
    return owner, None


def _connector_rpc(rid, params, action):
    owner, error = _owned_session(rid, params)
    if error:
        return error
    sid = params["session_id"]
    allowed = {"session_id"} if action == "status" else {"session_id", "connectors", "reconnect"}
    # ``profile`` is routing metadata, never authorization.
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
        # Bind the launch profile to prevent ambient sibling-profile leakage.
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
        # Do not expose exception strings: HTTP errors can contain credentials.
        return _connector_rpc_error(rid, 5034, "CONNECTOR_REQUEST_FAILED", "Connector request failed. Try again explicitly.")
    finally:
        _current_runtime_session_record.reset(runtime_token)


def _dispatch_connector_rpc(rid, sid, owner, profile_home, args):
    import model_tools
    from tools.connectors import connectors_available, live
    from tui_gateway.connector_payload import connector_ui_payload

    agent = owner.get("agent")
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
    if args["action"] != "status" and (operation := live.current(owner["session_key"])) is not None:
        # The card's Try again / Connect while the model's operation is open: reissue on that op.
        return _reissue(rid, operation, args)
    raw = model_tools.handle_function_call(
        "manage_connections", args, task_id=owner["session_key"],
        session_id=getattr(agent, "session_id", None) or owner["session_key"],
        tool_call_id=f"connector-ui-{uuid.uuid4().hex}",
        enabled_toolsets=enabled, disabled_toolsets=disabled,
    )
    data = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(data, dict) or "error" in data:
        return _connector_rpc_error(rid, 5034, "CONNECTOR_REQUEST_FAILED", "Connector request failed or was refused by policy.")
    if args["action"] == "status":
        if not isinstance(data.get("connectors"), list) or any(not isinstance(row, dict) for row in data["connectors"]):
            return _connector_rpc_error(rid, 5034, "INVALID_CONNECTOR_RESPONSE", "Connector service returned an invalid response.")
        return _ok(rid, {"available": True, "connectors": connector_ui_payload(data["connectors"])})
    if not isinstance(data.get("targets"), list):
        return _connector_rpc_error(rid, 5034, "INVALID_CONNECTOR_RESPONSE", "Connector service returned no authorization results.")
    return _ok(rid, connector_ui_payload(data))


def _reissue(rid, operation, args):
    """Re-mint links for the named targets on the open operation (user actor)."""
    from tools.connectors.contract import Actor, TargetState
    from tools.connectors.gateway.client import ConnectorClient
    from tools.connectors.managed import mint
    from tui_gateway.connector_payload import connector_ui_payload

    # Only a dead link is re-minted. A waiting target already holds its link (minted up front);
    # the card re-opens that one and never calls here for it.
    targets = [operation.target(n) for n in args["connectors"]]
    if any(t is None for t in targets):
        return _connector_rpc_error(rid, 4004, "UNKNOWN_TARGET", "no such target on the open operation")
    stale = [t.name for t in targets if t.state in (TargetState.failed, TargetState.expired)]
    if len(stale) != len(targets):
        return _connector_rpc_error(rid, 4002, "LINK_STILL_VALID",
                                    "only a failed or expired target can be re-minted; reopen the stored link")
    mint(ConnectorClient(), operation, stale, reinitiate=True, actor=Actor.user)
    return _ok(rid, connector_ui_payload(_operation_view(operation)))


def _live_operation(rid, params, owner):
    from tools.connectors import live

    op_id = params.get("op_id")
    if not isinstance(op_id, str) or not op_id:
        return None, _connector_rpc_error(rid, 4000, "INVALID_PARAMS", "op_id required")
    operation = live.get(owner["session_key"], op_id)
    if operation is None:
        return None, _connector_rpc_error(rid, 4004, "UNKNOWN_OPERATION", "no open operation with that op_id in this session")
    return operation, None


@method("connectors.list")
def _(rid, params):
    return _connector_rpc(rid, params, "status")


@method("connectors.connect")
def _(rid, params):
    return _connector_rpc(rid, params, "connect")


@method("connectors.operation.status")
def _(rid, params):
    from tui_gateway.connector_payload import connector_ui_payload

    owner, error = _owned_session(rid, params)
    if error:
        return error
    operation, error = _live_operation(rid, params, owner)
    if error:
        return error
    return _ok(rid, connector_ui_payload(_operation_view(operation)))


@method("connection.respond")
def _(rid, params):
    """The card's answer for the operation named by ``op_id``: per-target user / renderer-flow
    transitions and an optional Continue. The contract decides what the card may claim."""
    from tools.connectors import live
    from tools.connectors.contract import SettleReason
    from tools.connectors.mcp import apply_answer
    from tools.connectors.operation import IllegalTransition

    owner, error = _owned_session(rid, params)
    if error:
        return error
    operation, error = _live_operation(rid, params, owner)
    if error:
        return error
    try:
        apply_answer(operation, json.dumps(params["result"]))
    except IllegalTransition as exc:
        return _connector_rpc_error(rid, 4002, "ILLEGAL_TRANSITION", str(exc))
    if not operation.settled and operation.all_resolved:
        operation.settle(SettleReason.all_resolved)
    if operation.settled:
        live.close(operation)
    return _ok(rid, {"status": "ok", "settled": operation.settled})


def _operation_view(operation):
    return {**operation.result(), "settled": operation.settled}


def _connection_update(operation, change=None):
    """Emit ``connection.update`` for one transition, a link refresh, or settlement. Every frame
    carries the full target snapshot so the renderer never reconstructs state from deltas."""
    from tui_gateway import server

    with server._sessions_lock:
        sid = next((s for s, c in server._sessions.items() if c.get("session_key") == operation.session_key), None)
    if sid is None:
        return
    payload = _operation_view(operation)
    if change:
        payload.update(change)
    server._emit("connection.update", sid, payload)


def _install_update_hook():
    """Route every operation change through ``_connection_update``. Idempotent: ``register`` can run
    more than once (reload, tests) and must not stack wrappers."""
    from tools.connectors import operation as op_module

    if getattr(op_module.ConnectionOperation, "_update_hook_installed", False):
        return
    op_module.ConnectionOperation._update_hook_installed = True
    op_module.ConnectionOperation.on_change = staticmethod(_connection_update)


def register(server):
    bind_module(globals(), server, skip=("_",))
    server._LONG_HANDLERS = server._LONG_HANDLERS | _CONNECTOR_RPC_METHODS
    _install_update_hook()
