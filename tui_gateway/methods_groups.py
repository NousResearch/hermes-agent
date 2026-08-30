"""Hosted-room JSON-RPC contract.

These methods expose durable room identity, replay, and the process-owned
same-gateway Discussion driver. ``groups.capabilities`` keeps that boundary
machine-readable so older clients stay on the renderer-owned room path.
"""

from .method_ctx import HandlerRegistry

import os
import threading

_registry = HandlerRegistry()
method = _registry.method

LONG_HANDLERS = frozenset({
    "groups.list",
    "groups.capabilities",
    "groups.create",
    "groups.state",
    "groups.send",
    "groups.log",
    "groups.disband",
    "groups.replicate",
    "groups.replica_state",
    "groups.promote",
    "groups.demote",
    "groups.stop",
    "groups.retry",
    "groups.approve",
})

_service_lock = threading.Lock()
_bound_server = None
_service = None


def bind_server(server) -> None:
    """Bind the fully initialized server module without starting a worker."""

    global _bound_server
    _bound_server = server


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


_WORKER_UNAVAILABLE = (
    "Group Chat worker is unavailable. Restart the Hermes gateway and try again."
)


@method("groups.capabilities")
def _(rid, params: dict) -> dict:
    """Describe the hosted-room protocol implemented by this gateway."""
    from gateway.hosted_rooms import (
        MAX_LOG_LIMIT,
        PROTOCOL_VERSION,
        local_authority_gateway_id,
    )

    service = get_hosted_room_service()
    driver_ready = bool(service and service.runtime.status()["running"])
    return _ok(
        rid,
        {
            "protocol_version": PROTOCOL_VERSION,
            "driver": driver_ready,
            "persistent_process": os.getenv("HERMES_DESKTOP") != "1",
            "authority_gateway_id": local_authority_gateway_id(),
            "features": [
                "authority_epoch",
                "coordinator_fencing",
                "room_identity",
                "monotonic_log",
                "idempotent_send",
                "replayable_disband",
                "typed_events",
                "actor_identity",
                "log_replication",
                "authority_takeover",
            ],
            "methods": [
                "groups.capabilities",
                "groups.list",
                "groups.create",
                "groups.state",
                "groups.send",
                "groups.log",
                "groups.disband",
                "groups.replicate",
                "groups.replica_state",
                "groups.promote",
                "groups.demote",
                "groups.stop",
                "groups.retry",
                "groups.approve",
            ],
            "max_log_limit": MAX_LOG_LIMIT,
        },
    )


@method("groups.list")
def _(rid, params: dict) -> dict:
    """List rooms hosted by this gateway."""
    try:
        from gateway.hosted_rooms import (
            MAX_ROOM_LIST_LIMIT,
            default_db_path,
            list_rooms,
        )

        limit = params.get("limit", MAX_ROOM_LIST_LIMIT)
        offset = params.get("offset", 0)
        rooms = list_rooms(
            default_db_path(),
            include_disbanded=params.get("include_disbanded") is True,
            limit=limit,
            offset=offset,
        )

        return _ok(
            rid,
            {
                "rooms": rooms,
                "next_offset": offset + limit if len(rooms) == limit else None,
            },
        )
    except Exception as exc:
        return _err(rid, 5110, str(exc))


@method("groups.create")
def _(rid, params: dict) -> dict:
    """Create a hosted room idempotently.

    Required params: ``room_id``, ``name``, and ``members``. Authority is
    derived from this gateway's stable install identity, never from the client.
    """
    from gateway.hosted_rooms import HostedRoomError

    try:
        service = get_hosted_room_service()
        if service is None:
            return _err(rid, 4123, _WORKER_UNAVAILABLE)
        room = service.create_room(
            room_id=params.get("room_id"),
            name=params.get("name"),
            members=params.get("members"),
        )
        return _ok(rid, {"room": room})
    except HostedRoomError as exc:
        reason = getattr(exc, "reason", None)
        return _err(rid, 4110, str(exc), {"reason": reason} if reason else None)
    except Exception as exc:
        return _err(rid, 5111, str(exc))


@method("groups.state")
def _(rid, params: dict) -> dict:
    """Return one hosted room's replay cursor and fenced authority state."""
    from gateway.hosted_rooms import HostedRoomError, default_db_path, room_state

    try:
        room = room_state(
            default_db_path(),
            room_id=params.get("room_id"),
            include_disbanded=params.get("include_disbanded") is True,
        )
        service = get_hosted_room_service()
        return _ok(
            rid,
            {
                "room": room,
                **(
                    {"driver_status": service.status(str(room["room_id"]))}
                    if service is not None and room.get("disbanded_at") is None
                    else {}
                ),
            },
        )
    except HostedRoomError as exc:
        reason = getattr(exc, "reason", None)
        return _err(rid, 4114, str(exc), {"reason": reason} if reason else None)
    except Exception as exc:
        return _err(rid, 5115, str(exc))


@method("groups.send")
def _(rid, params: dict) -> dict:
    """Append one typed event to a hosted room idempotently.

    Required params: ``room_id``, ``event_id``, and object ``payload``. Only
    inert ``message.user`` events are accepted through this client-facing
    method. The actor is server-owned rather than trusted from params.
    Admission is durable; no Bot turn is started by this slice.
    """
    from gateway.hosted_rooms import HostedRoomError, user_event_id

    try:
        client_event_id = params.get("event_id")
        service = get_hosted_room_service()
        if service is None:
            return _err(rid, 4123, _WORKER_UNAVAILABLE)
        event = service.send(
            room_id=params.get("room_id"),
            event_id=user_event_id(client_event_id),
            payload=params.get("payload"),
        )
        return _ok(
            rid,
            {
                "event": event,
                "client_event_id": client_event_id,
                "accepted": True,
                "driver_started": True,
            },
        )
    except HostedRoomError as exc:
        reason = getattr(exc, "reason", None)
        return _err(rid, 4111, str(exc), {"reason": reason} if reason else None)
    except Exception as exc:
        return _err(rid, 5112, str(exc))


@method("groups.disband")
def _(rid, params: dict) -> dict:
    """Permanently tombstone a hosted room id."""
    from gateway.hosted_rooms import (
        AuthorityConflictError,
        HostedRoomError,
        RoomHistoryExpiredError,
        disband_room,
        local_authority_gateway_id,
        room_state,
    )

    try:
        service = get_hosted_room_service()
        if service is None:
            return _err(rid, 4123, _WORKER_UNAVAILABLE)

        def disband_with_state(state: dict | None = None) -> dict:
            local_gateway_id = local_authority_gateway_id()
            if state is not None and (
                str(state["authority_gateway_id"]) != local_gateway_id
            ):
                raise AuthorityConflictError(
                    "This Group Chat is managed by another gateway."
                )
            return disband_room(
                service.db_path,
                room_id=params.get("room_id"),
                expected_gateway_id=str(
                    local_gateway_id
                ),
                expected_epoch=int(
                    state["authority_epoch"] if state is not None else 1
                ),
            )

        try:
            existing = room_state(
                service.db_path,
                room_id=params.get("room_id"),
                include_disbanded=True,
            )
        except RoomHistoryExpiredError:
            tombstone = disband_with_state()
            return _ok(rid, {"tombstone": tombstone})
        if existing.get("disbanded_at") is not None:
            tombstone = disband_with_state(existing)
            return _ok(rid, {"tombstone": tombstone})
        service.stop_room(
            str(params.get("room_id") or ""),
            cancel_id=str(params.get("cancel_id") or "room-disbanded"),
            require_acknowledged=True,
        )
        tombstone = disband_with_state(existing)
        return _ok(rid, {"tombstone": tombstone})
    except HostedRoomError as exc:
        reason = getattr(exc, "reason", None)
        return _err(rid, 4113, str(exc), {"reason": reason} if reason else None)
    except Exception as exc:
        return _err(rid, 5114, str(exc))


@method("groups.stop")
def _(rid, params: dict) -> dict:
    """Durably cancel queued or running work for one hosted room."""

    service = get_hosted_room_service()
    if service is None:
        return _err(rid, 4115, "hosted room driver is unavailable")
    try:
        count = service.stop_room(
            str(params.get("room_id") or ""),
            cancel_id=str(params.get("cancel_id") or "desktop-stop"),
        )
        return _ok(rid, {"cancelled": count})
    except Exception as exc:
        return _err(rid, 5116, str(exc))


@method("groups.approve")
def _(rid, params: dict) -> dict:
    """Resolve one exact approval requested by a local room member."""

    service = get_hosted_room_service()
    if service is None:
        return _err(rid, 4115, "hosted room driver is unavailable")
    try:
        result = service.approve_room_task(
            str(params.get("room_id") or ""),
            member_id=str(params.get("member_id") or ""),
            task_id=str(params.get("task_id") or ""),
            execution_generation=int(params.get("execution_generation") or 0),
            choice=str(params.get("choice") or ""),
            request_id=str(params.get("request_id") or ""),
        )
        return _ok(rid, {"approved": True, "result": result})
    except Exception as exc:
        return _err(rid, 5119, str(exc))


@method("groups.retry")
def _(rid, params: dict) -> dict:
    """Retry one indeterminate room task after explicit user confirmation."""

    service = get_hosted_room_service()
    if service is None:
        return _err(rid, 4115, "hosted room driver is unavailable")
    try:
        task = service.retry_room_task(
            str(params.get("room_id") or ""),
            task_id=str(params.get("task_id") or ""),
        )
        identity = task.get("identity") if isinstance(task, dict) else None
        receipt = {
            "room_id": str(getattr(identity, "room_id", "") or ""),
            "task_id": str(getattr(identity, "task_id", "") or ""),
            "thread_id": str(getattr(identity, "thread_id", "") or ""),
            "turn_id": str(getattr(identity, "turn_id", "") or ""),
            "status": str(task.get("status") or "") if isinstance(task, dict) else "",
            "execution_generation": int(task.get("execution_generation") or 0)
            if isinstance(task, dict)
            else 0,
            "cancel_generation": int(task.get("cancel_generation") or 0)
            if isinstance(task, dict)
            else 0,
        }
        return _ok(rid, {"retried": True, "task": receipt})
    except Exception as exc:
        return _err(rid, 5118, str(exc))


@method("groups.log")
def _(rid, params: dict) -> dict:
    """Return a monotonic room-log delta after ``since_seq``."""
    from gateway.hosted_rooms import HostedRoomError, default_db_path, read_events

    try:
        delta = read_events(
            default_db_path(),
            room_id=params.get("room_id"),
            since_seq=params.get("since_seq", 0),
            limit=params.get("limit", 100),
            include_disbanded=params.get("include_disbanded") is True,
        )
        return _ok(rid, delta)
    except HostedRoomError as exc:
        reason = getattr(exc, "reason", None)
        return _err(rid, 4112, str(exc), {"reason": reason} if reason else None)
    except Exception as exc:
        return _err(rid, 5113, str(exc))


@method("groups.replicate")
def _(rid, params: dict) -> dict:
    """Persist one authority-stamped replay page into the local replica store.

    ``page`` is the verbatim ``groups.log`` result read from the room's
    authority gateway; ingest is idempotent and refuses sequence gaps and
    authority-epoch regressions.
    """
    from gateway.hosted_room_replicas import ReplicaError, ingest_page
    from gateway.hosted_rooms import default_db_path

    try:
        result = ingest_page(
            default_db_path(),
            room_id=params.get("room_id"),
            room_name=params.get("room_name"),
            members=params.get("members"),
            page=params.get("page"),
        )
        return _ok(rid, result)
    except ReplicaError as exc:
        return _err(rid, 4116, str(exc))
    except Exception as exc:
        return _err(rid, 5116, str(exc))


@method("groups.replica_state")
def _(rid, params: dict) -> dict:
    """Report the local replica's coverage and authority lineage."""
    from gateway.hosted_room_replicas import ReplicaError, replica_state
    from gateway.hosted_rooms import default_db_path

    try:
        return _ok(rid, replica_state(default_db_path(), room_id=params.get("room_id")))
    except ReplicaError as exc:
        return _err(rid, 4117, str(exc))
    except Exception as exc:
        return _err(rid, 5117, str(exc))


@method("groups.promote")
def _(rid, params: dict) -> dict:
    """Continue a replicated room on THIS gateway at ``epoch + 1``.

    Requires ``confirm: true`` — the caller asserts the previous authority can
    no longer commit (explicit user action; a lease/quorum driver later).
    """
    from gateway.hosted_room_replicas import ReplicaError, promote_replica
    from gateway.hosted_rooms import HostedRoomError, default_db_path

    if params.get("confirm") is not True:
        return _err(
            rid,
            4118,
            "promotion requires confirm=true acknowledging the previous "
            "authority can no longer commit",
        )
    try:
        result = promote_replica(
            default_db_path(),
            room_id=params.get("room_id"),
            reason=params.get("reason", "authority-unreachable"),
        )
        return _ok(rid, result)
    except ReplicaError as exc:
        return _err(rid, 4118, str(exc))
    except HostedRoomError as exc:
        return _err(rid, 4118, str(exc))
    except Exception as exc:
        return _err(rid, 5118, str(exc))


@method("groups.demote")
def _(rid, params: dict) -> dict:
    """Fence this gateway's stale room authority against a proven newer epoch."""
    from gateway.hosted_room_replicas import ReplicaError, demote_room
    from gateway.hosted_rooms import default_db_path

    try:
        result = demote_room(
            default_db_path(),
            room_id=params.get("room_id"),
            observed_gateway_id=params.get("observed_gateway_id"),
            observed_epoch=params.get("observed_epoch"),
        )
        return _ok(rid, result)
    except ReplicaError as exc:
        return _err(rid, 4119, str(exc))
    except Exception as exc:
        return _err(rid, 5119, str(exc))


def register(server) -> None:
    _registry.install(server)
