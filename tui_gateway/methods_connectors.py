"""Session-scoped connector RPCs and the connection-operation bridge.

Live transport ownership, not renderer-supplied profile or identity, authorizes requests.
The operation itself lives in ``tools.connectors.live``; this module reads and drives it and
pushes every transition to the session as ``connection.update``.
"""

import contextvars

from .contracts.config_free_tier_control import (
    ConnectorsConnectParams,
    ConnectorsConnectResult,
    ConnectorsListParams,
    ConnectorsListResult,
)
from .contracts.connectors_operation import (
    ConnectionOperationParams,
    ConnectionOperationStatus,
    ConnectionRespondParams,
    ConnectionRespondResult,
    ConnectionUpdatePayload,
)
from .method_ctx import HandlerRegistry, bind_module
import json
import uuid

_registry = HandlerRegistry()
method = _registry.method
_CONNECTOR_RPC_METHODS = frozenset({"connectors.list", "connectors.connect"})
_connector_rpc_origin: contextvars.ContextVar[tuple | None] = contextvars.ContextVar("connector_rpc_origin", default=None)


def _capture_connector_rpc_owner(params):
    # ``params`` is a dict: this runs on the raw request frame, before the method wrapper validates it.
    sid = params.get("session_id")
    _, owner = srv._current_session_steer_authority(sid if isinstance(sid, str) else "")
    srv._connector_rpc_origin.set((owner, owner.get("profile_home") if owner is not None else None))


def _connector_rpc_error(rid, code, reason, message):
    return srv._err(rid, code, message, data={"reason": reason})


def _connector_owner_matches(sid, owner, profile_home):
    _, current = srv._current_session_steer_authority(sid)
    return (current is owner and not owner.get("_finalized")
            and owner.get("profile_home") == profile_home)


def _owned_session(rid, params):
    """(owner, None) for a session this transport owns; (None, error reply) otherwise."""
    sid = params.session_id
    if not sid.strip():
        return None, srv._connector_rpc_error(rid, 4000, "INVALID_PARAMS", "session_id required")
    _, owner = srv._current_session_steer_authority(sid)
    origin = srv._connector_rpc_origin.get()
    if (owner is None or owner.get("_finalized")
            or origin is not None and (origin[0] is not owner or origin[1] != owner.get("profile_home"))):
        return None, srv._connector_rpc_error(rid, 4001, "NOT_OWNER", "session not found or not owned by this transport")
    if srv._session_uses_compute_host(owner):
        return None, srv._connector_rpc_error(rid, 5033, "UNSUPPORTED_RUNTIME", "Connectors must be managed on the session's compute host.")
    return owner, None


def _connector_rpc(rid, params, action):
    owner, error = srv._owned_session(rid, params)
    if error:
        return error
    sid = params.session_id
    args = {"action": action}
    if action != "status":
        import re

        if not params.connectors or any(
            re.fullmatch(r"[a-z0-9][a-z0-9_-]*", connector) is None
            for connector in params.connectors
        ):
            return srv._connector_rpc_error(rid, 4000, "INVALID_PARAMS", "connectors must be nonempty slugs")
        args.update(action="reconnect" if params.reconnect else "connect", connectors=params.connectors)
    profile_home = owner.get("profile_home")
    runtime_token = srv._current_runtime_session_record.set(owner)
    try:
        # Bind the launch profile to prevent ambient sibling-profile leakage.
        scope = {"profile_home": profile_home or str(srv._hermes_home)}
        with srv._session_profile_runtime_scope(scope):
            tokens = srv._set_session_context(owner["session_key"], cwd=srv._session_cwd(owner), ui_session_id=sid)
            try:
                result = srv._dispatch_connector_rpc(rid, sid, owner, profile_home, args)
            finally:
                srv._clear_session_context(tokens)
        if not srv._connector_owner_matches(sid, owner, profile_home):
            return srv._connector_rpc_error(rid, 4001, "NOT_OWNER", "session ownership changed")
        return result
    except Exception:
        # Do not expose exception strings: HTTP errors can contain credentials.
        return srv._connector_rpc_error(rid, 5034, "CONNECTOR_REQUEST_FAILED", "Connector request failed. Try again explicitly.")
    finally:
        srv._current_runtime_session_record.reset(runtime_token)


def _dispatch_connector_rpc(rid, sid, owner, profile_home, args):
    import model_tools
    from tools.connectors import connectors_available, live
    from tui_gateway.connector_payload import connector_ui_payload

    agent = owner.get("agent")
    enabled = (agent.enabled_toolsets if agent is not None
               else srv._load_enabled_toolsets(srv._resolve_agent_platform(srv._session_source(owner))))
    disabled = agent.disabled_toolsets if agent is not None else None
    if ("manage_connections" not in model_tools._select_tool_names(enabled, disabled, quiet_mode=True)
            or not connectors_available()):
        if args["action"] == "status":
            return ConnectorsListResult(available=False, connectors=[])
        return srv._connector_rpc_error(rid, 4031, "CONNECTORS_UNAVAILABLE", "Connectors are not available in this session.")
    if not srv._connector_owner_matches(sid, owner, profile_home):
        return srv._connector_rpc_error(rid, 4001, "NOT_OWNER", "session ownership changed")
    if args["action"] != "status" and (operation := live.current(owner["session_key"])) is not None:
        # The card's Try again / Connect while the model's operation is open: reissue on that op.
        return srv._reissue(rid, operation, args)
    raw = model_tools.handle_function_call(
        "manage_connections", args, task_id=owner["session_key"],
        session_id=getattr(agent, "session_id", None) or owner["session_key"],
        tool_call_id=f"connector-ui-{uuid.uuid4().hex}",
        enabled_toolsets=enabled, disabled_toolsets=disabled,
    )
    data = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(data, dict) or "error" in data:
        return srv._connector_rpc_error(rid, 5034, "CONNECTOR_REQUEST_FAILED", "Connector request failed or was refused by policy.")
    if args["action"] == "status":
        if not isinstance(data.get("connectors"), list) or any(not isinstance(row, dict) for row in data["connectors"]):
            return srv._connector_rpc_error(rid, 5034, "INVALID_CONNECTOR_RESPONSE", "Connector service returned an invalid response.")
        return ConnectorsListResult(available=True, connectors=connector_ui_payload(data["connectors"]))
    if not isinstance(data.get("targets"), list):
        return srv._connector_rpc_error(rid, 5034, "INVALID_CONNECTOR_RESPONSE", "Connector service returned no authorization results.")
    # ``ConnectionOperation.result()`` carries ``settled_at``; the wire snapshot names ``settled`` too.
    data.setdefault("settled", data.get("settled_at") is not None)
    return ConnectorsConnectResult.model_validate(connector_ui_payload(data))


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
        return srv._connector_rpc_error(rid, 4004, "UNKNOWN_TARGET", "no such target on the open operation")
    stale = [t.name for t in targets if t.state in (TargetState.failed, TargetState.expired)]
    if len(stale) != len(targets):
        return srv._connector_rpc_error(rid, 4002, "LINK_STILL_VALID",
                                    "only a failed or expired target can be re-minted; reopen the stored link")
    mint(ConnectorClient(), operation, stale, reinitiate=True, actor=Actor.user)
    return ConnectorsConnectResult.model_validate(connector_ui_payload(srv._operation_view(operation)))


def _live_operation(rid, params, owner):
    from tools.connectors import live

    if not params.op_id:
        return None, srv._connector_rpc_error(rid, 4000, "INVALID_PARAMS", "op_id required")
    operation = live.get(owner["session_key"], params.op_id)
    if operation is None:
        return None, srv._connector_rpc_error(rid, 4004, "UNKNOWN_OPERATION", "no open operation with that op_id in this session")
    return operation, None


@method("connectors.list")
def _(rid, params: ConnectorsListParams) -> ConnectorsListResult | dict:
    """Return connector catalog + connection state for one owned session."""
    return srv._connector_rpc(rid, params, "status")


@method("connectors.connect")
def _(rid, params: ConnectorsConnectParams) -> ConnectorsConnectResult | dict:
    """Start or re-initiate authorization for named connectors."""
    return srv._connector_rpc(rid, params, "connect")


@method("connectors.operation.status")
def _(rid, params: ConnectionOperationParams) -> ConnectionOperationStatus | dict:
    from tui_gateway.connector_payload import connector_ui_payload

    owner, error = srv._owned_session(rid, params)
    if error:
        return error
    operation, error = srv._live_operation(rid, params, owner)
    if error:
        return error
    return ConnectionOperationStatus.model_validate(connector_ui_payload(srv._operation_view(operation)))


@method("connection.respond")
def _(rid, params: ConnectionRespondParams) -> ConnectionRespondResult | dict:
    """The card's answer for the operation named by ``op_id``: per-target user / renderer-flow
    transitions and an optional Continue. The contract decides what the card may claim."""
    from tools.connectors import live
    from tools.connectors.contract import SettleReason
    from tools.connectors.mcp import apply_answer
    from tools.connectors.operation import IllegalTransition

    owner, error = srv._owned_session(rid, params)
    if error:
        return error
    operation, error = srv._live_operation(rid, params, owner)
    if error:
        return error
    try:
        apply_answer(operation, params.result.model_dump_json(exclude_none=True))
    except IllegalTransition as exc:
        return srv._connector_rpc_error(rid, 4002, "ILLEGAL_TRANSITION", str(exc))
    if not operation.settled and operation.all_resolved:
        operation.settle(SettleReason.all_resolved)
    if operation.settled:
        live.close(operation)
    return ConnectionRespondResult(status="ok", settled=operation.settled)


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
    server._emit("connection.update", sid, ConnectionUpdatePayload.model_validate({**srv._operation_view(operation), **(change or {})}))


def _install_update_hook():
    """Route every operation change through ``_connection_update``. Idempotent: ``register`` can run
    more than once (reload, tests) and must not stack wrappers."""
    from tools.connectors import operation as op_module

    if getattr(op_module.ConnectionOperation, "_update_hook_installed", False):
        return
    op_module.ConnectionOperation._update_hook_installed = True
    op_module.ConnectionOperation.on_change = staticmethod(srv._connection_update)


def register(server):
    bind_module(globals(), server, skip=("_",))
    server._LONG_HANDLERS = server._LONG_HANDLERS | srv._CONNECTOR_RPC_METHODS
    srv._install_update_hook()

# Bound last, after every definition, so importing this module first (tests, the gateway process)
# lets server.py's own tail import see a complete module — the same tail-import idiom server.py uses.
from tui_gateway import server as srv  # noqa: E402
