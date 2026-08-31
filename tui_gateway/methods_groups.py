"""Hosted-room JSON-RPC contract.

These methods expose durable room identity and an append-only, monotonic room
log. They deliberately do not drive Bot turns yet. ``groups.capabilities``
makes that boundary machine-readable so a hosted-aware Desktop cannot mistake
the log prototype for a complete gateway-side orchestrator.
"""

from .method_ctx import HandlerRegistry

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
})


@method("groups.capabilities")
def _(rid, params: dict) -> dict:
    """Describe the hosted-room protocol implemented by this gateway."""
    from gateway.hosted_rooms import (
        MAX_LOG_LIMIT,
        PROTOCOL_VERSION,
        local_authority_gateway_id,
    )

    return _ok(
        rid,
        {
            "protocol_version": PROTOCOL_VERSION,
            "driver": False,
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
    from gateway.hosted_rooms import (
        HostedRoomError,
        create_room,
        default_db_path,
        local_authority_gateway_id,
    )

    try:
        room = create_room(
            default_db_path(),
            room_id=params.get("room_id"),
            name=params.get("name"),
            members=params.get("members"),
            authority_gateway_id=local_authority_gateway_id(),
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
        return _ok(
            rid,
            {
                "room": room_state(
                    default_db_path(),
                    room_id=params.get("room_id"),
                    include_disbanded=params.get("include_disbanded") is True,
                )
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
    from gateway.hosted_rooms import (
        AuthorityConflictError,
        HostedRoomError,
        append_event,
        default_db_path,
        local_authority_gateway_id,
        room_state,
        user_event_id,
    )

    try:
        room = room_state(default_db_path(), room_id=params.get("room_id"))
        local_gateway_id = local_authority_gateway_id()
        if str(room["authority_gateway_id"]) != local_gateway_id:
            raise AuthorityConflictError(
                "This Group Chat is managed by another gateway."
            )
        client_event_id = params.get("event_id")
        event = append_event(
            default_db_path(),
            room_id=params.get("room_id"),
            event_id=user_event_id(client_event_id),
            kind="message.user",
            actor={"kind": "user", "id": "desktop"},
            payload=params.get("payload"),
            authority_gateway_id=local_gateway_id,
            authority_epoch=int(room["authority_epoch"]),
        )
        return _ok(
            rid,
            {
                "event": event,
                "client_event_id": client_event_id,
                "accepted": True,
                "driver_started": False,
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
        default_db_path,
        disband_room,
        local_authority_gateway_id,
        room_state,
    )

    try:
        local_gateway_id = local_authority_gateway_id()
        try:
            room = room_state(
                default_db_path(),
                room_id=params.get("room_id"),
                include_disbanded=True,
            )
        except RoomHistoryExpiredError:
            room = {
                "authority_gateway_id": local_gateway_id,
                "authority_epoch": 1,
            }
        if str(room["authority_gateway_id"]) != local_gateway_id:
            raise AuthorityConflictError(
                "This Group Chat is managed by another gateway."
            )
        tombstone = disband_room(
            default_db_path(),
            room_id=params.get("room_id"),
            expected_gateway_id=local_gateway_id,
            expected_epoch=int(room["authority_epoch"]),
        )
        return _ok(rid, {"tombstone": tombstone})
    except HostedRoomError as exc:
        reason = getattr(exc, "reason", None)
        return _err(rid, 4113, str(exc), {"reason": reason} if reason else None)
    except Exception as exc:
        return _err(rid, 5114, str(exc))


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
