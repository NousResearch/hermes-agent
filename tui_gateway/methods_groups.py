"""Hosted-room JSON-RPC contract: durable room identity, replay, and the process-owned
same-gateway Discussion driver; ``groups.capabilities`` keeps that boundary machine-readable.
Server-free on purpose: the gateway process imports this module for the
hosted-room lifecycle, so error frames come from ``rpc_frames`` and module-private helpers are passed
through keyword defaults. ``_room_method`` is the shared envelope."""

from .method_ctx import HandlerRegistry
from .rpc_frames import err_frame

import contextlib
import importlib
import os
import threading

from .contracts.groups_bot_relay import (
    GroupsApproveParams, GroupsApproveResult, GroupsCapabilitiesParams, GroupsCapabilitiesResult,
    GroupsCreateParams, GroupsCreateResult, GroupsDemoteParams, GroupsDemoteResult,
    GroupsDisbandParams, GroupsDisbandResult, GroupsListParams, GroupsListResult,
    GroupsLogParams, GroupsLogResult, GroupsPeerInviteParams, GroupsPeerInviteResult,
    GroupsPeerRegisterParams, GroupsPeerRegisterResult, GroupsPeerRevokeParams,
    GroupsPeerRevokeResult, GroupsPromoteParams, GroupsPromoteResult, GroupsReplicateParams,
    GroupsReplicateResult, GroupsReplicaStateParams, GroupsReplicaStateResult, GroupsRenameParams,
    GroupsRenameResult, GroupsRetryParams, GroupsRetryResult, GroupsSendParams, GroupsSendPayload, GroupsSendResult,
    GroupsStateParams, GroupsStateResult, GroupsStopParams, GroupsStopResult,
)

_registry = HandlerRegistry()
method = _registry.method

#: Wire order of ``groups.capabilities.methods``; every one runs on the RPC pool.
_METHODS = (
    "groups.capabilities", "groups.list", "groups.create", "groups.state", "groups.send",
    "groups.rename", "groups.log", "groups.disband", "groups.replicate", "groups.replica_state",
    "groups.promote", "groups.demote", "groups.stop", "groups.retry", "groups.approve",
    "groups.peer.invite", "groups.peer.revoke", "groups.peer.register")
LONG_HANDLERS = frozenset(_METHODS)

_service_lock = threading.Lock()
_run_store_lock = threading.Lock()
_bound_server = None
_service = None

_WORKER_UNAVAILABLE = "Group Chat worker is unavailable. Restart the Hermes gateway and try again."
_DRIVER_UNAVAILABLE = "hosted room driver is unavailable"


def bind_server(server) -> None:
    """Bind the fully initialized server module without starting a worker."""
    global _bound_server
    _bound_server = server
    server._profile_execution_policy = _profile_execution_policy


def start_hosted_room_service():
    """Start one process-owned hosted room service idempotently."""
    global _service
    if _bound_server is None:
        return None
    from gateway.hosted_rooms import default_db_path
    from tui_gateway.hosted_room_service import HostedRoomService
    db_path = default_db_path()
    with _service_lock:
        if _service is not None and _service.db_path != db_path:
            _service.stop(timeout=1.0)
            _service = None
        if _service is None:
            _service = HostedRoomService(_bound_server, db_path=db_path)
        _service.start()
        return _service


def stop_hosted_room_service(*, timeout: float = 5.0) -> bool:
    """Stop the process-owned worker without interrupting accepted turns."""
    global _service
    with _service_lock:
        service = _service
        if service is None:
            return True
        stopped = service.stop(timeout=timeout)
        if stopped and _service is service:
            _service = None
        return stopped


def get_hosted_room_service():
    """Return the active service, if its lifecycle owner started it."""
    service = _service
    if service is None:
        return None
    try:
        status = service.runtime.status()
    except Exception:
        return None
    return service if status.get("running") and not status.get("stopping") else None


def _profile_name() -> str:
    return (os.getenv("HERMES_PROFILE") or "default").strip() or "default"


def _current_profile() -> str:
    return str(_bound_server._current_profile_name() or "").strip()


def _foreign_profile_home(profile: str):
    """Home of a routed profile other than the process's own, or ``ValueError``."""
    home = _bound_server._profile_home(profile)
    if home is None:
        raise ValueError(f"profile '{profile}' is unavailable")
    return home


def _requested_profile(requested: str | None) -> str:
    requested = (requested or "").strip()
    if not requested:
        return _profile_name()
    if _bound_server is None:
        raise ValueError("profile routing is unavailable")
    if requested == _current_profile():
        return requested
    _foreign_profile_home(requested)
    return str(_bound_server._response_profile_name(requested) or requested)


def _api_server_key(profile: str | None = None) -> str:
    # Published onto the server by methods_bot_relay.register (an explicit routed profile is
    # authoritative: never borrow the process profile's key on a multiplexed gateway).
    if profile and _bound_server is not None and profile != _current_profile():
        from agent.secret_scope import build_profile_secret_scope
        home = _bound_server._profile_home(profile)
        if home is None:
            return ""
        return str(build_profile_secret_scope(home).get("API_SERVER_KEY") or "").strip()
    scoped = ""
    with contextlib.suppress(Exception):
        from agent.secret_scope import get_secret
        scoped = (get_secret("API_SERVER_KEY", "") or "").strip()
    return scoped or (os.getenv("API_SERVER_KEY") or "").strip()


def _profile_execution_policy(profile: str) -> dict:
    """Resolve execution policy under the exact multiplexed profile's FULL runtime scope: the policy
    reads provider credentials (``_xai_credentials_present`` -> ``get_env_value``), which under a
    home-only override resolved from the launch process env (or raised once hosting fails closed)."""
    from gateway.hosted_room_execution_policy import execution_policy_mapping
    if _bound_server is None:
        return execution_policy_mapping(target_profile=profile)
    home = None if profile in {_current_profile(), _profile_name()} else _foreign_profile_home(profile)
    with _bound_server._session_profile_runtime_scope({"profile_home": str(home) if home else None}):
        return execution_policy_mapping(target_profile=profile)


def _room_link_run_storage_durable() -> bool:
    """Return whether peer-run replay survives this gateway process."""
    if _bound_server is None:
        return True
    store = getattr(_bound_server, "_run_idempotency_store", None)
    if store is None:
        from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
        with _run_store_lock:
            store = getattr(_bound_server, "_run_idempotency_store", None)
            if store is None:
                store = _bound_server._run_idempotency_store = RunIdempotencyStore()
    return bool(getattr(store, "durable", False))


def _local_catalog(installation_id: str, profile: str, execution_policy: dict) -> dict:
    """Advertise this gateway's direct-only, text-only RoomLink catalog."""
    from gateway.hosted_room_peer import PROTOCOL_VERSION, local_catalog_mapping
    return local_catalog_mapping(
        installation_id=installation_id, protocol_versions=(PROTOCOL_VERSION,),
        link_modes=("direct",), text=True, attachments=False, target_profile=profile,
        execution_policy=execution_policy)


def _grant_expiry(claims: dict) -> float:
    return float(claims.get("status_expires_at", claims["expires_at"]))


def _room_error_class(replica_only: bool) -> type:
    if replica_only:
        from gateway.hosted_room_replicas import ReplicaError
        return ReplicaError
    from gateway.hosted_rooms import HostedRoomError
    return HostedRoomError


def _room_method(
    name: str, *, code: int, room_code: int | None = None, replica_only: bool = False,
    with_reason: bool = True, service_code: int | None = None,
    service_message: str = _DRIVER_UNAVAILABLE, db: bool = False,
):
    """Register ``fn`` under ``name`` with the shared hosted-room error envelope."""
    error_class = _room_error_class

    def dec(fn):
        def handler(rid, params):
            args = (rid, params)
            if service_code is not None:
                service = get_hosted_room_service()
                if service is None:
                    return err_frame(rid, service_code, service_message)
                args += (service,)
            if db:
                from gateway.hosted_rooms import default_db_path
                args += (default_db_path(),)
            try:
                return fn(*args)
            except Exception as exc:
                if room_code is not None and isinstance(exc, error_class(replica_only)):
                    reason = getattr(exc, "reason", None) if with_reason else None
                    return err_frame(rid, room_code, str(exc), {"reason": reason} if reason else None)
                return err_frame(rid, code, str(exc))
        handler.__doc__ = fn.__doc__
        return method(name)(handler)
    return dec


@method("groups.capabilities")
def _(rid, params: GroupsCapabilitiesParams, _catalog=_local_catalog, _methods=_METHODS) -> GroupsCapabilitiesResult:
    """Describe the hosted-room protocol implemented by this gateway."""
    from gateway.hosted_rooms import MAX_LOG_LIMIT, PROTOCOL_VERSION, local_authority_gateway_id
    service = get_hosted_room_service()
    driver_ready = bool(service and service.runtime.status()["running"])
    try:
        from gateway.hosted_room_peer import gateway_room_grant_secret
        profile = _requested_profile(params.profile)
        if not _room_link_run_storage_durable():
            raise ValueError("durable run idempotency storage is required")
        gateway_room_grant_secret()
        policy = _profile_execution_policy(profile)
        catalog = _catalog(local_authority_gateway_id(), profile, policy)
        room_link = {"enabled": True, "profile": profile, "catalog": catalog, "endpoint": catalog["endpoint"]}
    except Exception:
        room_link = {"enabled": False, "reason": (
            "durable_run_storage_required" if not _room_link_run_storage_durable()
            else "gateway_roomlink_secret_unavailable")}
    return GroupsCapabilitiesResult(
        protocol_version=PROTOCOL_VERSION, driver=driver_ready,
        persistent_process=bool(room_link.get("catalog", {}).get("persistent_process", False)),
        authority_gateway_id=local_authority_gateway_id(), room_link=room_link,
        features=[
            "authority_epoch", "coordinator_fencing", "room_identity", "monotonic_log",
            "idempotent_send", "replayable_disband", "typed_events", "actor_identity",
            "log_replication", "authority_takeover"],
        methods=list(_methods), max_log_limit=MAX_LOG_LIMIT,
    )


@_room_method("groups.peer.invite", code=4120, db=True)
def _(rid, params: GroupsPeerInviteParams, db_path, _catalog=_local_catalog, _expiry=_grant_expiry) -> GroupsPeerInviteResult:
    """Mint one target-issued room/profile grant for a prospective home."""
    from gateway.hosted_room_peer import decode_room_grant, gateway_room_grant_secret, issue_room_grant
    from gateway.hosted_rooms import local_authority_gateway_id, reserve_peer_room
    if not _room_link_run_storage_durable():
        raise ValueError("durable run idempotency storage is required")
    installation_id = local_authority_gateway_id()
    profile = _requested_profile(params.profile)
    ttl = params.ttl_seconds if params.ttl_seconds is not None else 3600
    if not 60 <= ttl <= 24 * 60 * 60:
        raise ValueError("ttl_seconds must be between 60 and 86400")
    grant_secret = gateway_room_grant_secret()
    execution_policy = _profile_execution_policy(profile)
    token = issue_room_grant(
        grant_secret, grant_id=params.grant_id or f"grant-{os.urandom(16).hex()}", room_id=params.room_id or "",
        home_install_id=params.home_install_id or "", authority_gateway_id=params.authority_gateway_id or "",
        authority_epoch=params.authority_epoch or 0, member_id=params.member_id or "", target_install_id=installation_id,
        target_profile=profile, execution_policy_digest=execution_policy["policy_digest"], ttl_seconds=ttl)
    claims = decode_room_grant(grant_secret, token, permission="status")
    reserve_peer_room(db_path, claims=claims, expires_at=_expiry(claims))
    catalog = _catalog(installation_id, profile, execution_policy)
    return GroupsPeerInviteResult(grant=token, target_profile=profile, catalog=catalog, endpoint=catalog["endpoint"])


@_room_method("groups.peer.revoke", code=4122, db=True)
def _(rid, params: GroupsPeerRevokeParams, db_path, _expiry=_grant_expiry) -> GroupsPeerRevokeResult:
    """Revoke one target-issued grant using its exact profile scope."""
    from gateway.hosted_room_peer import decode_room_grant, gateway_room_grant_secret
    from gateway.hosted_rooms import local_authority_gateway_id, revoke_room_grant_scope
    profile = _requested_profile(params.profile)
    claims = decode_room_grant(gateway_room_grant_secret(), params.grant, permission="status")
    if claims["target_profile"] != profile or claims["target_install_id"] != local_authority_gateway_id():
        raise ValueError("room grant target does not match this profile")
    revoke_room_grant_scope(db_path, claims=claims, expires_at=_expiry(claims))
    return GroupsPeerRevokeResult(revoked=True)


@_room_method("groups.peer.register", code=5120, service_code=4121)
def _(rid, params: GroupsPeerRegisterParams, service) -> GroupsPeerRegisterResult:
    """Register and probe one scoped target route on the room home."""
    from gateway.hosted_room_peer import GatewayRoomCatalog, PROTOCOL_VERSION as ROOM_LINK_PROTOCOL_VERSION, validate_room_link_url
    from gateway.hosted_rooms import local_authority_gateway_id, room_state
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
    target_url, transport_security = validate_room_link_url(params.target_url)
    catalog = GatewayRoomCatalog.from_mapping(params.catalog.as_mapping())
    if ROOM_LINK_PROTOCOL_VERSION not in catalog.protocol_versions:
        raise ValueError(f"target does not support RoomLink protocol v{ROOM_LINK_PROTOCOL_VERSION}")
    if "direct" not in catalog.link_modes:
        raise ValueError("target does not support a direct RoomLink")
    client = PeerRunsHTTPClient(base_url=target_url, api_key="", receipt_db_path=service.db_path)
    probe = client.probe(grant=params.grant)
    if GatewayRoomCatalog.from_mapping(probe.get("catalog")) != catalog:
        raise ValueError("target capability catalog changed during setup")
    home_install_id = local_authority_gateway_id()
    home_room = room_state(service.db_path, room_id=params.room_id)
    expected_scope = {
        "room_id": params.room_id, "home_install_id": home_install_id,
        "authority_gateway_id": home_room.get("authority_gateway_id"),
        "member_id": params.member_id, "target_profile": params.target_profile}
    if (any(probe.get(key) != value for key, value in expected_scope.items())
            or int(probe.get("authority_epoch") or 0) != int(home_room.get("authority_epoch") or 0)):
        raise ValueError("room grant scope does not match this route")
    route = PeerMemberRoute(
        home_install_id=home_install_id, member_id=params.member_id, target_install_id=catalog.installation_id,
        target_profile=params.target_profile, capability_digest=catalog.catalog_digest,
        execution_policy_digest=catalog.execution_policy.policy_digest,
        cancellation_scope_id=params.cancellation_scope_id or f"cancel-{params.room_id}",
        trace_id=params.trace_id or f"trace-{os.urandom(16).hex()}", grant=params.grant)
    service.register_peer_route(
        room_id=params.room_id, member_id=params.member_id, route=route, client=client, target_url=target_url,
        catalog=catalog)
    return GroupsPeerRegisterResult(
        registered=True, mode="direct", transport_security=transport_security,
        target_install_id=catalog.installation_id, target_profile=params.target_profile)


@_room_method("groups.list", code=5110, db=True)
def _(rid, params: GroupsListParams, db_path) -> GroupsListResult:
    """List rooms hosted by this gateway."""
    from gateway.hosted_rooms import MAX_ROOM_LIST_LIMIT, list_rooms
    limit = params.limit if params.limit is not None else MAX_ROOM_LIST_LIMIT
    offset = params.offset if params.offset is not None else 0
    rooms = list_rooms(db_path, include_disbanded=params.include_disbanded is True, limit=limit, offset=offset)
    next_offset = offset + limit if len(rooms) == limit else None
    return GroupsListResult(rooms=rooms, next_offset=next_offset)


@_room_method("groups.create", code=5111, room_code=4110, service_code=4123, service_message=_WORKER_UNAVAILABLE)
def _(rid, params: GroupsCreateParams, service) -> GroupsCreateResult:
    """Create a hosted room idempotently; authority is this gateway's stable install identity."""
    room = service.create_room(room_id=params.room_id, name=params.name, members=[member.as_mapping() for member in params.members])
    return GroupsCreateResult(room=room)


@_room_method("groups.state", code=5115, room_code=4114, db=True)
def _(rid, params: GroupsStateParams, db_path) -> GroupsStateResult:
    """Return one hosted room's replay cursor and fenced authority state."""
    from gateway.hosted_rooms import room_state
    room = room_state(db_path, room_id=params.room_id, include_disbanded=params.include_disbanded is True)
    service = get_hosted_room_service()
    driver_status = service.status(room["room_id"]) if service is not None and room.get("disbanded_at") is None else None
    return GroupsStateResult(room=room, driver_status=driver_status)


@_room_method("groups.send", code=5112, room_code=4111, service_code=4123, service_message=_WORKER_UNAVAILABLE)
def _(rid, params: GroupsSendParams, service) -> GroupsSendResult:
    """Append one typed event idempotently (inert ``message.user`` only; actor is server-owned)."""
    from gateway.hosted_rooms import user_event_id
    client_event_id = params.event_id
    payload = params.payload.model_dump(mode="json") if isinstance(params.payload, GroupsSendPayload) else params.payload
    event = service.send(room_id=params.room_id, event_id=user_event_id(client_event_id), payload=payload)
    return GroupsSendResult(event=event, client_event_id=client_event_id, accepted=True, driver_started=True)


@_room_method("groups.disband", code=5114, room_code=4113, service_code=4123, service_message=_WORKER_UNAVAILABLE)
def _(rid, params: GroupsDisbandParams, service) -> GroupsDisbandResult:
    """Permanently tombstone a hosted room id."""
    from gateway.hosted_rooms import AuthorityConflictError, RoomHistoryExpiredError, disband_room, local_authority_gateway_id, room_state

    def disband_with_state(state: dict | None = None) -> GroupsDisbandResult:
        local_gateway_id = local_authority_gateway_id()
        if state is not None and str(state["authority_gateway_id"]) != local_gateway_id:
            raise AuthorityConflictError("This Group Chat is managed by another gateway.")
        tombstone = disband_room(
            service.db_path, room_id=params.room_id, expected_gateway_id=str(local_gateway_id),
            expected_epoch=int(state["authority_epoch"] if state is not None else 1))
        return GroupsDisbandResult(tombstone=tombstone)
    try:
        existing = room_state(service.db_path, room_id=params.room_id, include_disbanded=True)
    except RoomHistoryExpiredError:
        return disband_with_state()
    if existing.get("disbanded_at") is not None:
        return disband_with_state(existing)
    service.stop_room(params.room_id, cancel_id=params.cancel_id or "room-disbanded", require_acknowledged=True)
    service.revoke_room_routes(params.room_id)
    return disband_with_state(existing)


@_room_method("groups.stop", code=5116, service_code=4115)
def _(rid, params: GroupsStopParams, service) -> GroupsStopResult:
    """Durably cancel queued or running work for one hosted room."""
    count = service.stop_room(params.room_id, cancel_id=params.cancel_id or "desktop-stop")
    return GroupsStopResult(cancelled=count)


@_room_method("groups.approve", code=5119, service_code=4115)
def _(rid, params: GroupsApproveParams, service) -> GroupsApproveResult:
    """Resolve one exact approval requested by a local or peer room member."""
    result = service.approve_room_task(
        params.room_id, member_id=params.member_id, task_id=params.task_id,
        execution_generation=params.execution_generation, choice=str(params.choice), request_id=params.request_id)
    return GroupsApproveResult(approved=True, result=result)


@_room_method("groups.retry", code=5118, service_code=4115)
def _(rid, params: GroupsRetryParams, service) -> GroupsRetryResult:
    """Retry one indeterminate room task after explicit user confirmation."""
    task = service.retry_room_task(params.room_id, task_id=params.task_id)
    if not isinstance(task, dict):
        task = {}
    identity = task.get("identity")
    receipt = {
        **{field: str(getattr(identity, field, "") or "") for field in ("room_id", "task_id", "thread_id", "turn_id")},
        "status": str(task.get("status") or ""), "execution_generation": int(task.get("execution_generation") or 0),
        "cancel_generation": int(task.get("cancel_generation") or 0)}
    return GroupsRetryResult(retried=True, task=receipt)


def _passthrough(
    name: str, module: str, fn_name: str, doc: str, *, code: int, room_code: int, result_type: type,
    param_values, replica_only: bool = False, wrap: str | None = None,
) -> None:
    """Register a typed method whose result is ``module.fn(db_path, **param_values(params))``."""
    @_room_method(name, code=code, room_code=room_code, replica_only=replica_only, with_reason=not replica_only, db=True)
    def handler(rid, params_in, db_path, _import=importlib.import_module):
        result = getattr(_import(module), fn_name)(db_path, **param_values(params_in))
        return result_type(**({wrap: result} if wrap else result))
    handler.__doc__ = doc


_passthrough(
    "groups.rename", "gateway.hosted_rooms", "rename_room", """Rename one hosted room atomically with its replay event.""",
    code=5117, room_code=4117, result_type=GroupsRenameResult,
    param_values=lambda params: {"room_id": params.room_id, "event_id": params.event_id, "name": params.name}, wrap="room")
_passthrough(
    "groups.log", "gateway.hosted_rooms", "read_events", """Return a monotonic room-log delta after ``since_seq``.""",
    code=5113, room_code=4112, result_type=GroupsLogResult,
    param_values=lambda params: {
        "room_id": params.room_id, "since_seq": params.since_seq if params.since_seq is not None else 0,
        "limit": params.limit if params.limit is not None else 100, "include_disbanded": params.include_disbanded is True})
_passthrough(
    "groups.replicate", "gateway.hosted_room_replicas", "ingest_page", """Persist one authority-stamped replay page into the local replica store.""",
    code=5116, room_code=4116, result_type=GroupsReplicateResult, replica_only=True,
    param_values=lambda params: {
        "room_id": params.room_id, "room_name": params.room_name, "members": params.members,
        "page": params.page.as_mapping()})
_passthrough(
    "groups.replica_state", "gateway.hosted_room_replicas", "replica_state", """Report the local replica's coverage and authority lineage.""",
    code=5117, room_code=4117, result_type=GroupsReplicaStateResult, replica_only=True,
    param_values=lambda params: {"room_id": params.room_id})


@_room_method("groups.promote", code=5118, room_code=4118, with_reason=False, db=True)
def _(rid, params: GroupsPromoteParams, db_path) -> GroupsPromoteResult | dict:
    """Continue a replicated room on THIS gateway at ``epoch + 1``."""
    from gateway.hosted_room_replicas import promote_replica
    if params.confirm is not True:
        return err_frame(rid, 4118, "promotion requires confirm=true acknowledging the previous authority can no longer commit")
    return GroupsPromoteResult(**promote_replica(
        db_path, room_id=params.room_id, reason=params.reason or "authority-unreachable"))


_passthrough(
    "groups.demote", "gateway.hosted_room_replicas", "demote_room", """Fence this gateway's stale room authority.""",
    code=5119, room_code=4119, result_type=GroupsDemoteResult, replica_only=True,
    param_values=lambda params: {
        "room_id": params.room_id, "observed_gateway_id": params.observed_gateway_id,
        "observed_epoch": params.observed_epoch})


def register(server) -> None:
    _registry.install(server, globals())
