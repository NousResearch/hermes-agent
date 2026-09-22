import contextlib
import contextvars

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_CONNECTOR_RPC_METHODS = frozenset({"connectors.list", "connectors.connect", "connection.respond"})
_connector_rpc_origin: contextvars.ContextVar[tuple | None] = contextvars.ContextVar("connector_rpc_origin", default=None)


def _capture_connector_rpc_owner(params):
    owner = params.get("owner") if isinstance(params, dict) else None
    sid = owner.get("session_id") if isinstance(owner, dict) and owner.get("type") == "session" else ""
    _, session = _current_session_steer_authority(sid if isinstance(sid, str) else "")
    _connector_rpc_origin.set((session, session.get("profile_home") if session is not None else None))


def _connector_rpc_error(rid, code, reason, message):
    return _err(rid, code, message, data={"reason": reason})


def _connector_auth_error(rid, exc):
    from tui_gateway.contracts.connectors import ConnectorErrorReason

    if exc.code in {"no_access", "ORG_ACCESS_DENIED"}:
        return _connector_rpc_error(
            rid, 4030, ConnectorErrorReason.org_access_denied,
            "This account cannot manage connectors for this organization.",
        )
    if exc.status == 401 or exc.code in {"invalid_token", "INVALID_TOKEN", "NO_TOKEN"}:
        return _connector_rpc_error(rid, 4032, ConnectorErrorReason.needs_nous_auth, "Sign in to use connectors.")
    return _connector_rpc_error(
        rid, 4030, ConnectorErrorReason.forbidden_scope, "Connector access is not permitted for this account."
    )


def _connector_guard(fn):
    def handler(rid, params):
        from tools.connectors.gateway.errors import GatewayAuthError
        from tui_gateway.contracts.connectors import ConnectorErrorReason

        try:
            return fn(rid, params)
        except ProfileUnavailableError:
            raise
        except GatewayAuthError as exc:
            return _connector_auth_error(rid, exc)
        except Exception:
            return _connector_rpc_error(
                rid, 5034, ConnectorErrorReason.connector_request_failed, "Connector request failed. Try again explicitly."
            )

    return handler


def _connector_owner_matches(sid, owner, profile_home):
    _, current = _current_session_steer_authority(sid)
    return current is owner and not owner.get("_finalized") and owner.get("profile_home") == profile_home


def _session_owner(rid, owner):
    from tui_gateway.contracts.connectors import ConnectorErrorReason

    sid = owner.session_id
    _, session = _current_session_steer_authority(sid)
    origin = _connector_rpc_origin.get()
    if (session is None or session.get("_finalized")
            or origin is not None and (origin[0] is not session or origin[1] != session.get("profile_home"))):
        return None, _connector_rpc_error(
            rid, 4001, ConnectorErrorReason.not_owner, "session not found or not owned by this transport"
        )
    if _session_uses_compute_host(session):
        return None, _connector_rpc_error(
            rid, 5033, ConnectorErrorReason.unsupported_runtime, "Connectors must be managed on the session's compute host."
        )
    return session, None


def _account_gate_closed(rid):
    from tools.connectors import connectors_available
    from tui_gateway.contracts.connectors import ConnectorErrorReason

    if connectors_available():
        return None
    return _connector_rpc_error(rid, 4031, ConnectorErrorReason.connectors_unavailable, "Connectors are not available.")


def _parse_params(rid, params, model):
    from pydantic import ValidationError

    from tui_gateway.contracts.connectors import ConnectorErrorReason

    try:
        return model.model_validate(params), None
    except ValidationError:
        return None, _connector_rpc_error(rid, 4000, ConnectorErrorReason.invalid_params, "Connector parameters are invalid.")


def _connector_params(rid, params, model):
    request, error = _parse_params(rid, params, model)
    if error:
        return None, None, error
    if request.owner.type == "account":
        return request, None, None
    session, error = _session_owner(rid, request.owner)
    return request, session, error


def _session_connector_gate(rid, session, action):
    import model_tools
    from tools.connectors import connectors_available

    from tui_gateway.contracts.connectors import ConnectorErrorReason

    agent = session.get("agent")
    enabled = agent.enabled_toolsets if agent is not None else _load_enabled_toolsets(_resolve_agent_platform(_session_source(session)))
    disabled = agent.disabled_toolsets if agent is not None else None
    if "manage_connections" not in model_tools._select_tool_names(enabled, disabled, quiet_mode=True) or not connectors_available():
        if action == "status":
            return enabled, disabled, _ok(rid, {"available": False, "connectors": []})
        return enabled, disabled, _connector_rpc_error(
            rid, 4031, ConnectorErrorReason.connectors_unavailable, "Connectors are not available in this session."
        )
    return enabled, disabled, None


def _session_connector_rpc(rid, request, session, action):
    import json
    import model_tools
    import uuid

    from tools.connectors import live
    from tui_gateway.connector_payload import connector_ui_payload
    from tui_gateway.contracts.connectors import ConnectorErrorReason

    sid = request.owner.session_id
    enabled, disabled, error = _session_connector_gate(rid, session, action)
    if error:
        return error
    if not _connector_owner_matches(sid, session, session.get("profile_home")):
        return _connector_rpc_error(rid, 4001, ConnectorErrorReason.not_owner, "session ownership changed")
    args = {"action": "reconnect" if action == "connect" and request.reconnect else action}
    if action == "connect":
        args["connectors"] = request.connectors
    if action == "connect":
        operation = live.current(session["session_key"], profile_home=session.get("profile_home"))
        if operation is None:
            return _connector_rpc_error(
                rid, 4004, ConnectorErrorReason.unknown_operation, "No open connection operation for this session."
            )
        return _reissue(rid, operation, args)
    raw = model_tools.handle_function_call(
        "manage_connections",
        args,
        task_id=session["session_key"],
        session_id=getattr(session.get("agent"), "session_id", None) or session["session_key"],
        tool_call_id=f"connector-ui-{uuid.uuid4().hex}",
        enabled_toolsets=enabled,
        disabled_toolsets=disabled,
    )
    data = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(data, dict) or "error" in data:
        return _connector_rpc_error(rid, 5034, ConnectorErrorReason.connector_request_failed, "Connector request failed. Try again explicitly.")
    if action == "status":
        if not isinstance(data.get("connectors"), list) or any(not isinstance(row, dict) for row in data["connectors"]):
            return _connector_rpc_error(rid, 5034, ConnectorErrorReason.invalid_connector_response, "Connector service returned an invalid response.")
        return _ok(rid, {"available": True, "connectors": _connector_rows(data["connectors"])})
    if not isinstance(data.get("targets"), list):
        return _connector_rpc_error(rid, 5034, ConnectorErrorReason.invalid_connector_response, "Connector service returned no authorization results.")
    return _ok(rid, connector_ui_payload(data))


def _connector_rows(rows):
    from tools.connectors.gateway.wire import ConnectorListItem
    from tui_gateway.connector_payload import connector_ui_payload

    return connector_ui_payload([
        ConnectorListItem.model_validate(row).model_dump(mode="json", by_alias=False)
        for row in rows
    ])


def _account_connector_list(rid):
    from tools.connectors.managed import managed_client

    return _ok(rid, {"available": True, "connectors": _connector_rows(managed_client().list_connectors())})


def _account_connector_connect(rid, request):
    from tools.connectors import account
    from tui_gateway.connector_payload import connector_ui_payload
    from tui_gateway.contracts.connectors import ConnectorErrorReason

    action = "reconnect" if request.reconnect else "connect"
    try:
        start = account.find_or_start_operation(request.connectors, action=action, profile_home=_account_home(request))
        if not start.started:
            from tools.connectors.contract import TargetState

            targets = [start.operation.target(name) for name in request.connectors]
            if any(target.state in (TargetState.failed, TargetState.expired) for target in targets):
                return _reissue(rid, start.operation, {"connectors": request.connectors})
            return _ok(rid, connector_ui_payload(_operation_view(start.operation)))
        if not account.wait_for_prepare(start):
            return _ok(rid, connector_ui_payload(_operation_view(start.operation)))
        if start.failed:
            return _connector_rpc_error(rid, 5034, ConnectorErrorReason.connector_request_failed, "Connector request failed. Try again explicitly.")
        return _ok(rid, connector_ui_payload(_operation_view(start.operation)))
    except ValueError:
        return _connector_rpc_error(rid, 4000, ConnectorErrorReason.invalid_params, "Connector parameters are invalid.")


def _connector_rpc(rid, params, action):
    from tui_gateway.contracts.connectors import ConnectorErrorReason, ConnectorsConnectParams, ConnectorsListParams

    model = ConnectorsListParams if action == "status" else ConnectorsConnectParams
    request, session, error = _connector_params(rid, params, model)
    if error:
        return error
    if request.owner.type == "account":
        with _account_scope(request):
            if closed := _account_gate_closed(rid):
                return _ok(rid, {"available": False, "connectors": []}) if action == "status" else closed
            return _account_connector_list(rid) if action == "status" else _account_connector_connect(rid, request)

    runtime_token = _current_runtime_session_record.set(session)
    try:
        profile_home = session.get("profile_home")
        with _session_profile_runtime_scope({"profile_home": profile_home or str(_hermes_home)}):
            tokens = _set_session_context(session["session_key"], cwd=_session_cwd(session), ui_session_id=request.owner.session_id)
            try:
                result = _session_connector_rpc(rid, request, session, action)
            finally:
                _clear_session_context(tokens)
        if not _connector_owner_matches(request.owner.session_id, session, profile_home):
            return _connector_rpc_error(rid, 4001, ConnectorErrorReason.not_owner, "session ownership changed")
        return result
    finally:
        _current_runtime_session_record.reset(runtime_token)


def _reissue(rid, operation, args):
    from tools.connectors.run import (
        LINK_STILL_VALID,
        MIXED_KINDS,
        NOT_ALLOWED,
        REFUSED,
        SETTLED,
        UNKNOWN_TARGET,
        reissue,
    )
    from tui_gateway.connector_payload import connector_ui_payload
    from tui_gateway.contracts.connectors import ConnectorErrorReason

    reason = reissue(operation, args["connectors"])
    if reason == UNKNOWN_TARGET:
        return _connector_rpc_error(rid, 4004, ConnectorErrorReason.unknown_target, "No such target on the open operation.")
    if reason == MIXED_KINDS:
        return _connector_rpc_error(rid, 4000, ConnectorErrorReason.invalid_params, "One target kind per request.")
    if reason == LINK_STILL_VALID:
        return _connector_rpc_error(rid, 4002, ConnectorErrorReason.link_still_valid, "Reopen the stored link.")
    if reason == SETTLED:
        return _connector_rpc_error(rid, 4002, ConnectorErrorReason.reissue_refused, "The operation has settled.")
    if reason == NOT_ALLOWED:
        return _connector_rpc_error(rid, 4002, ConnectorErrorReason.reissue_refused, "This target cannot be run again.")
    if reason == REFUSED:
        return _connector_rpc_error(rid, 4002, ConnectorErrorReason.reissue_refused, "The target cannot be run again.")
    return _ok(rid, connector_ui_payload(_operation_view(operation)))


def _operation_params(rid, params):
    from tui_gateway.contracts.connectors_operation import ConnectionOperationParams

    return _connector_params(rid, params, ConnectionOperationParams)


@method("connectors.list")
@_connector_guard
def _(rid, params):
    return _connector_rpc(rid, params, "status")


@method("connectors.connect")
@_connector_guard
def _(rid, params):
    return _connector_rpc(rid, params, "connect")


def _account_home(request):
    home = _profile_home(request.profile)
    return str(home) if home else None


@contextlib.contextmanager
def _account_scope(request):
    with _session_profile_runtime_scope({"profile_home": _account_home(request)}):
        tokens = _set_session_context("")
        try:
            yield
        finally:
            _clear_session_context(tokens)


def _operation_for_request(rid, request, session):
    from tools.connectors import live
    from tui_gateway.contracts.connectors import ConnectorErrorReason

    if request.owner.type == "session":
        operation = live.get(session["session_key"], request.op_id, profile_home=session.get("profile_home"))
    else:
        with _account_scope(request):
            if closed := _account_gate_closed(rid):
                return None, closed
            operation = live.get_by_op_id(request.op_id, profile_home=_account_home(request))
    if operation is None:
        return None, _connector_rpc_error(rid, 4004, ConnectorErrorReason.unknown_operation, "No open operation with that op_id.")
    return operation, None


@method("connectors.operation.status")
@_connector_guard
def _(rid, params):
    from tui_gateway.connector_payload import connector_ui_payload

    request, session, error = _operation_params(rid, params)
    if error:
        return error
    operation, error = _operation_for_request(rid, request, session)
    return error or _ok(rid, connector_ui_payload(_operation_view(operation)))


@method("connectors.operation.wake")
@_connector_guard
def _(rid, params):
    request, session, error = _operation_params(rid, params)
    if error:
        return error
    operation, error = _operation_for_request(rid, request, session)
    if error:
        return error
    operation.wake.set()
    return _ok(rid, {"status": "ok"})


@method("connection.respond")
@_connector_guard
def _(rid, params):
    from pydantic import ValidationError

    from tui_gateway.contracts.connectors import ConnectorErrorReason
    from tui_gateway.contracts.connectors_operation import ConnectionAnswer

    envelope = {key: value for key, value in params.items() if key != "result"}
    request, session, error = _operation_params(rid, envelope)
    if error:
        return error
    try:
        answer = ConnectionAnswer.model_validate(params.get("result"))
    except ValidationError:
        return _connector_rpc_error(rid, 4002, ConnectorErrorReason.invalid_answer, "Connection answer is invalid.")
    operation, error = _operation_for_request(rid, request, session)
    if error:
        return error
    if request.owner.type == "account":
        with _account_scope(request):
            return _apply_connection_answer(rid, answer, operation)
    with _session_profile_runtime_scope({"profile_home": session.get("profile_home") or str(_hermes_home)}):
        return _apply_connection_answer(rid, answer, operation)


def _apply_connection_answer(rid, answer, operation):
    from tools.connectors import live
    from tools.connectors.contract import SettleReason
    from tools.connectors.mcp import apply_answer
    from tools.connectors.operation import IllegalTransition
    from tui_gateway.contracts.connectors import ConnectorErrorReason

    try:
        apply_answer(operation, answer.model_dump_json(exclude_none=True))
    except IllegalTransition:
        return _connector_rpc_error(rid, 4002, ConnectorErrorReason.invalid_answer, "Connection answer is invalid.")
    if not operation.settled and operation.all_resolved:
        operation.settle(SettleReason.all_resolved)
    if operation.settled:
        live.close(operation)
    return _ok(rid, {"status": "ok", "settled": operation.settled})


def _snapshot_view(snapshot):
    return {**snapshot, "settled": snapshot.get("settled_at") is not None}


def _operation_view(operation):
    return _snapshot_view(operation.snapshot())


def _connection_update(operation, change, snapshot):
    from hermes_constants import get_process_hermes_home, hermes_home_key
    from tui_gateway import server
    from tui_gateway.connector_payload import connector_ui_payload

    payload = _snapshot_view(snapshot)
    if change:
        payload.update(change)
    if operation.session_key.startswith("account:"):
        payload["targets"] = [
            {key: value for key, value in target.items() if key not in ("connect_url", "connection_id")}
            for target in payload["targets"]
        ]
        payload["owner"] = {"type": "account"}
        server._broadcast_global_event("connection.update", connector_ui_payload(payload))
        return
    with server._sessions_lock:
        sid = next(
            (
                sid
                for sid, session in server._sessions.items()
                if session.get("session_key") == operation.session_key
                and hermes_home_key(session.get("profile_home") or get_process_hermes_home()) == operation.profile_key
            ),
            None,
        )
    if sid is not None:
        payload["owner"] = {"type": "session", "session_id": sid}
        server._emit("connection.update", sid, connector_ui_payload(payload))


def _install_update_hook():
    from tools.connectors import operation as op_module

    if getattr(op_module.ConnectionOperation, "_update_hook_installed", False):
        return
    op_module.ConnectionOperation._update_hook_installed = True
    op_module.ConnectionOperation.on_change = staticmethod(_connection_update)


def register(server):
    bind_module(globals(), server, skip=("_",))
    server._LONG_HANDLERS = server._LONG_HANDLERS | _CONNECTOR_RPC_METHODS
    _install_update_hook()
