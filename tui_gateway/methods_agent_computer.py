"""JSON-RPC surface for persistent Agent Computers.

Handlers are rebound onto server.py globals (see method_ctx.py). Module
helpers therefore cannot see ``_ok`` / ``current_transport``; handlers
resolve those names in-body after install.
"""

from __future__ import annotations

from gateway.agent_computer import get_contract
from gateway.agent_computer.contract import agent_from_profile, error_payload, owner_principal
from gateway.agent_computer.errors import AgentComputerError

from .method_ctx import HandlerRegistry
from .methods_browser_control import _is_authenticated_identity

_registry = HandlerRegistry()
method = _registry.method

_ERR_FORBIDDEN = 4403
_ERR_CONFLICT = 4409
_ERR_NOT_FOUND = 4404


def _principal_from(params: dict, identity, profile_name: str, *, owner_only: bool = False) -> str:
    """Owner from a ticket-authenticated identity; agent from the session profile.

    Client ``profile_id`` is never used for authorization. The embedded TUI
    PTY identity (``server-internal``) is not an owner — same gate as
    ``browser.controller.*``.
    """
    if _is_authenticated_identity(identity):
        return owner_principal()
    if owner_only:
        raise AgentComputerError("owner authentication required")
    profile = str(profile_name or "default").strip() or "default"
    return agent_from_profile(profile)


def _rpc_error(rid, exc: AgentComputerError, err_fn):
    code = _ERR_FORBIDDEN if exc.http_status in (401, 403) else _ERR_CONFLICT
    if exc.http_status == 404:
        code = _ERR_NOT_FOUND
    return err_fn(rid, code, str(exc), error_payload(exc))


@method("computer.ensure")
def _(rid, params: dict, _contract=get_contract, _who=_principal_from, _fail=_rpc_error) -> dict:
    params = params or {}
    try:
        transport = current_transport()  # type: ignore[name-defined]
        identity = getattr(transport, "auth_identity", None) if transport is not None else None
        profile = str(_current_profile_name() or "default")  # type: ignore[name-defined]
        principal = _who(params, identity, profile)
        if principal.startswith("agent:"):
            profile = principal.split(":", 1)[1]
        return _ok(rid, _contract().ensure(profile, principal))  # type: ignore[name-defined]
    except AgentComputerError as exc:
        return _fail(rid, exc, _err)  # type: ignore[name-defined]


@method("computer.status")
def _(rid, params: dict, _contract=get_contract, _who=_principal_from, _fail=_rpc_error) -> dict:
    params = params or {}
    try:
        transport = current_transport()  # type: ignore[name-defined]
        identity = getattr(transport, "auth_identity", None) if transport is not None else None
        principal = _who(params, identity, _current_profile_name())  # type: ignore[name-defined]
        return _ok(rid, _contract().status(str(params.get("computer_id") or ""), principal))  # type: ignore[name-defined]
    except AgentComputerError as exc:
        return _fail(rid, exc, _err)  # type: ignore[name-defined]


@method("computer.list")
def _(rid, params: dict, _contract=get_contract, _who=_principal_from, _fail=_rpc_error) -> dict:
    params = params or {}
    try:
        transport = current_transport()  # type: ignore[name-defined]
        identity = getattr(transport, "auth_identity", None) if transport is not None else None
        principal = _who(params, identity, _current_profile_name())  # type: ignore[name-defined]
        return _ok(rid, _contract().list_computers(principal))  # type: ignore[name-defined]
    except AgentComputerError as exc:
        return _fail(rid, exc, _err)  # type: ignore[name-defined]


@method("computer.wake")
def _(rid, params: dict, _contract=get_contract, _who=_principal_from, _fail=_rpc_error) -> dict:
    params = params or {}
    try:
        transport = current_transport()  # type: ignore[name-defined]
        identity = getattr(transport, "auth_identity", None) if transport is not None else None
        principal = _who(params, identity, _current_profile_name())  # type: ignore[name-defined]
        return _ok(rid, _contract().wake(str(params.get("computer_id") or ""), principal))  # type: ignore[name-defined]
    except AgentComputerError as exc:
        return _fail(rid, exc, _err)  # type: ignore[name-defined]


@method("computer.observe")
def _(rid, params: dict, _contract=get_contract, _who=_principal_from, _fail=_rpc_error) -> dict:
    params = params or {}
    try:
        transport = current_transport()  # type: ignore[name-defined]
        identity = getattr(transport, "auth_identity", None) if transport is not None else None
        principal = _who(params, identity, _current_profile_name())  # type: ignore[name-defined]
        return _ok(  # type: ignore[name-defined]
            rid,
            _contract().observe(
                str(params.get("computer_id") or ""),
                principal,
                lease_id=str(params.get("lease_id") or ""),
                fencing_epoch=int(params.get("fencing_epoch") or 0),
            ),
        )
    except AgentComputerError as exc:
        return _fail(rid, exc, _err)  # type: ignore[name-defined]


@method("computer.act")
def _(rid, params: dict, _contract=get_contract, _who=_principal_from, _fail=_rpc_error) -> dict:
    params = params or {}
    try:
        transport = current_transport()  # type: ignore[name-defined]
        identity = getattr(transport, "auth_identity", None) if transport is not None else None
        principal = _who(params, identity, _current_profile_name())  # type: ignore[name-defined]
        return _ok(  # type: ignore[name-defined]
            rid,
            _contract().act(
                str(params.get("computer_id") or ""),
                principal,
                lease_id=str(params.get("lease_id") or ""),
                fencing_epoch=int(params.get("fencing_epoch") or 0),
                kind=str(params.get("kind") or ""),
                target=str(params.get("target") or ""),
                text=str(params.get("text") or ""),
                action_class=str(params.get("action_class") or ""),
                x=params.get("x"),
                y=params.get("y"),
                key=str(params.get("key") or ""),
                code=str(params.get("code") or ""),
                delta_x=float(params.get("delta_x") or 0),
                delta_y=float(params.get("delta_y") or 0),
            ),
        )
    except AgentComputerError as exc:
        return _fail(rid, exc, _err)  # type: ignore[name-defined]


@method("computer.takeover")
def _(rid, params: dict, _contract=get_contract, _who=_principal_from, _fail=_rpc_error) -> dict:
    params = params or {}
    try:
        transport = current_transport()  # type: ignore[name-defined]
        identity = getattr(transport, "auth_identity", None) if transport is not None else None
        principal = _who(params, identity, _current_profile_name(), owner_only=True)  # type: ignore[name-defined]
        computer_id = str(params.get("computer_id") or "")
        result = _contract().request_takeover(
            computer_id,
            principal,
            reason=str(params.get("reason") or ""),
        )
        _contract().service.bind_owner_transport(computer_id, transport)
        return _ok(rid, result)  # type: ignore[name-defined]
    except AgentComputerError as exc:
        return _fail(rid, exc, _err)  # type: ignore[name-defined]


@method("computer.takeover.connect")
def _(rid, params: dict, _contract=get_contract, _who=_principal_from, _fail=_rpc_error) -> dict:
    params = params or {}
    try:
        transport = current_transport()  # type: ignore[name-defined]
        identity = getattr(transport, "auth_identity", None) if transport is not None else None
        principal = _who(params, identity, _current_profile_name(), owner_only=True)  # type: ignore[name-defined]
        computer_id = str(params.get("computer_id") or "")
        result = _contract().connect_takeover(
            computer_id,
            principal,
            takeover_token=str(params.get("takeover_token") or ""),
        )
        _contract().service.bind_owner_transport(computer_id, transport)
        return _ok(rid, result)  # type: ignore[name-defined]
    except AgentComputerError as exc:
        return _fail(rid, exc, _err)  # type: ignore[name-defined]


@method("computer.give_back")
def _(rid, params: dict, _contract=get_contract, _who=_principal_from, _fail=_rpc_error) -> dict:
    params = params or {}
    try:
        transport = current_transport()  # type: ignore[name-defined]
        identity = getattr(transport, "auth_identity", None) if transport is not None else None
        principal = _who(params, identity, _current_profile_name(), owner_only=True)  # type: ignore[name-defined]
        return _ok(  # type: ignore[name-defined]
            rid,
            _contract().give_back(
                str(params.get("computer_id") or ""),
                principal,
                lease_id=str(params.get("lease_id") or ""),
                fencing_epoch=int(params.get("fencing_epoch") or 0),
            ),
        )
    except AgentComputerError as exc:
        return _fail(rid, exc, _err)  # type: ignore[name-defined]


@method("computer.identity.create")
def _(rid, params: dict, _contract=get_contract, _who=_principal_from, _fail=_rpc_error) -> dict:
    params = params or {}
    try:
        transport = current_transport()  # type: ignore[name-defined]
        identity = getattr(transport, "auth_identity", None) if transport is not None else None
        principal = _who(params, identity, _current_profile_name(), owner_only=True)  # type: ignore[name-defined]
        ownership = params.get("ownership") or []
        if isinstance(ownership, str):
            ownership = [ownership]
        return _ok(  # type: ignore[name-defined]
            rid,
            _contract().create_identity(
                principal,
                ownership=[str(x) for x in ownership],
                metadata=params.get("metadata") if isinstance(params.get("metadata"), dict) else {},
            ),
        )
    except AgentComputerError as exc:
        return _fail(rid, exc, _err)  # type: ignore[name-defined]


@method("computer.identity.attach")
def _(rid, params: dict, _contract=get_contract, _who=_principal_from, _fail=_rpc_error) -> dict:
    params = params or {}
    try:
        transport = current_transport()  # type: ignore[name-defined]
        identity = getattr(transport, "auth_identity", None) if transport is not None else None
        principal = _who(params, identity, _current_profile_name())  # type: ignore[name-defined]
        return _ok(  # type: ignore[name-defined]
            rid,
            _contract().attach_identity(
                str(params.get("computer_id") or ""),
                str(params.get("identity_id") or ""),
                principal,
            ),
        )
    except AgentComputerError as exc:
        return _fail(rid, exc, _err)  # type: ignore[name-defined]


@method("computer.owner_disconnect")
def _(rid, params: dict, _contract=get_contract, _who=_principal_from, _fail=_rpc_error) -> dict:
    params = params or {}
    try:
        transport = current_transport()  # type: ignore[name-defined]
        identity = getattr(transport, "auth_identity", None) if transport is not None else None
        principal = _who(params, identity, _current_profile_name(), owner_only=True)  # type: ignore[name-defined]
        return _ok(  # type: ignore[name-defined]
            rid,
            _contract().owner_disconnect(str(params.get("computer_id") or ""), principal),
        )
    except AgentComputerError as exc:
        return _fail(rid, exc, _err)  # type: ignore[name-defined]


def register(server) -> None:
    _registry.install(server)
