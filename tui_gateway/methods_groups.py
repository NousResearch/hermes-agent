"""Hosted-room JSON-RPC contract.

These methods expose durable room identity, replay, and the process-owned
same-gateway Discussion driver. ``groups.capabilities`` keeps that boundary
machine-readable so older clients stay on the renderer-owned room path.
"""

from .method_ctx import HandlerRegistry

import os
import threading
import time

_registry = HandlerRegistry()
method = _registry.method

LONG_HANDLERS = frozenset({
    "groups.list",
    "groups.capabilities",
    "groups.create",
    "groups.state",
    "groups.send",
    "groups.rename",
    "groups.log",
    "groups.disband",
    "groups.replica_state",
    "groups.stop",
    "groups.retry",
    "groups.approve",
    "groups.peer.invite",
    "groups.peer.revoke_exact",
    "groups.peer.revoke",
    "groups.peer.register",
    "groups.control.invite",
    "groups.control.register",
    "groups.control.revoke",
})

_service_lock = threading.Lock()
_run_store_lock = threading.Lock()
_bound_server = None
_service = None


def bind_server(server) -> None:
    """Bind the fully initialized server module without starting a worker."""

    global _bound_server
    _bound_server = server
    server._profile_execution_policy = _profile_execution_policy
    server._revoke_peer_room_control = _revoke_peer_room_control


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


def _profile_name() -> str:
    return (os.getenv("HERMES_PROFILE") or "default").strip() or "default"


def _requested_profile(params: dict) -> str:
    requested = str(params.get("profile") or "").strip()
    if not requested:
        return _profile_name()
    if _bound_server is None:
        raise ValueError("profile routing is unavailable")
    current = str(_bound_server._current_profile_name() or "").strip()
    if requested == current:
        return current
    home = _bound_server._profile_home(requested)
    if home is None:
        raise ValueError(f"profile '{requested}' is unavailable")
    return str(_bound_server._response_profile_name(requested) or requested)


def _api_server_key(profile: str | None = None) -> str:
    if profile and _bound_server is not None:
        current = str(_bound_server._current_profile_name() or "").strip()
        if profile != current:
            from agent.secret_scope import build_profile_secret_scope

            home = _bound_server._profile_home(profile)
            if home is None:
                return ""
            # An explicit routed profile is authoritative. Never borrow the
            # process/default profile's API key on a multiplexed gateway.
            return str(
                build_profile_secret_scope(home).get("API_SERVER_KEY") or ""
            ).strip()
    try:
        from agent.secret_scope import get_secret

        scoped = (get_secret("API_SERVER_KEY", "") or "").strip()
        if scoped:
            return scoped
    except Exception:
        pass
    return (os.getenv("API_SERVER_KEY") or "").strip()


def _revoke_peer_room_control(room_id: str, member_id: str) -> int:
    from gateway.hosted_room_control_client import revoke_stored_peer_control
    from gateway.hosted_rooms import default_db_path

    return revoke_stored_peer_control(
        default_db_path(), room_id=room_id, member_id=member_id
    )


def _profile_execution_policy(profile: str) -> dict:
    """Resolve execution policy under the exact multiplexed profile home."""

    from gateway.hosted_room_execution_policy import execution_policy_mapping
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    token = None
    if _bound_server is not None:
        current = str(_bound_server._current_profile_name() or "").strip()
        if profile not in {current, _profile_name()}:
            home = _bound_server._profile_home(profile)
            if home is None:
                raise ValueError(f"profile '{profile}' is unavailable")
            token = set_hermes_home_override(str(home))
    try:
        return execution_policy_mapping(target_profile=profile)
    finally:
        if token is not None:
            reset_hermes_home_override(token)


def _room_link_run_storage_durable() -> bool:
    """Return whether peer-run replay survives this gateway process."""

    if _bound_server is None:
        # Direct method-contract tests and embedded callers without a bound API
        # server do not expose peer-run transport. The production server always
        # binds before advertising capabilities.
        return True
    store = getattr(_bound_server, "_run_idempotency_store", None)
    if store is None:
        # The dashboard/TUI process owns groups.* but does not construct the
        # API adapter that normally owns this store. Open the same shared
        # SQLite-backed store lazily so capability negotiation reflects the
        # real /v1/runs replay boundary instead of depending on test-only
        # injection. A separately enabled API adapter uses the same file.
        from gateway.platforms.api_server import RunIdempotencyStore

        with _run_store_lock:
            store = getattr(_bound_server, "_run_idempotency_store", None)
            if store is None:
                store = RunIdempotencyStore()
                _bound_server._run_idempotency_store = store
    return bool(getattr(store, "durable", False))


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
    try:
        from gateway.hosted_room_peer import (
            PROTOCOL_VERSION as ROOM_LINK_PROTOCOL_VERSION,
            gateway_room_grant_secret,
            local_catalog_mapping,
        )

        profile = _requested_profile(params)
        if not _room_link_run_storage_durable():
            raise ValueError("durable run idempotency storage is required")
        gateway_room_grant_secret()
        catalog = local_catalog_mapping(
            installation_id=local_authority_gateway_id(),
            protocol_versions=(ROOM_LINK_PROTOCOL_VERSION,),
            link_modes=("direct",),
            text=True,
            attachments=False,
            target_profile=profile,
            execution_policy=_profile_execution_policy(profile),
        )
        room_link = {
            "enabled": True,
            "profile": profile,
            "catalog": catalog,
            "endpoint": catalog["endpoint"],
        }
    except Exception:
        room_link = {
            "enabled": False,
            "reason": (
                "durable_run_storage_required"
                if not _room_link_run_storage_durable()
                else "gateway_roomlink_secret_unavailable"
            ),
        }
    return _ok(
        rid,
        {
            "protocol_version": PROTOCOL_VERSION,
            "driver": driver_ready,
            "persistent_process": bool(
                room_link.get("catalog", {}).get("persistent_process", False)
            ),
            "authority_gateway_id": local_authority_gateway_id(),
            "room_link": room_link,
            "features": [
                "authority_epoch",
                "coordinator_fencing",
                "desktop_compatibility_mailbox",
                "reciprocal_room_control",
                "room_identity",
                "monotonic_log",
                "idempotent_send",
                "replayable_disband",
                "typed_events",
                "actor_identity",
                "peer_route_grant_fingerprint",
            ],
            "methods": [
                "groups.capabilities",
                "groups.list",
                "groups.create",
                "groups.state",
                "groups.send",
                "groups.rename",
                "groups.log",
                "groups.disband",
                "groups.replica_state",
                "groups.stop",
                "groups.retry",
                "groups.approve",
                "groups.peer.invite",
                "groups.peer.revoke_exact",
                "groups.peer.revoke",
                "groups.peer.register",
                "groups.desktop.claim",
                "groups.desktop.presence",
                "groups.desktop.renew",
                "groups.desktop.complete",
                "groups.control.invite",
                "groups.control.register",
                "groups.control.revoke",
            ],
            "max_log_limit": MAX_LOG_LIMIT,
        },
    )


@method("groups.desktop.claim")
def _(rid, params: dict) -> dict:
    """Advertise classic rooms and lease pending messaging commands."""

    try:
        from gateway.desktop_room_mailbox import claim_commands, default_db_path

        commands = claim_commands(
            default_db_path(),
            consumer_id=params.get("consumer_id"),
            room_authorities=params.get("room_authorities", []),
            actions=params.get("actions"),
            limit=params.get("limit", 8),
        )
        return _ok(rid, {"commands": commands})
    except Exception as exc:
        return _err(rid, 4130, str(exc))


@method("groups.desktop.presence")
def _(rid, params: dict) -> dict:
    """Renew classic-room ownership without claiming pending commands."""

    try:
        from gateway.desktop_room_mailbox import default_db_path, refresh_presence

        room_ids = refresh_presence(
            default_db_path(),
            consumer_id=params.get("consumer_id"),
            room_authorities=params.get("room_authorities", []),
        )
        return _ok(rid, {"room_ids": room_ids})
    except Exception as exc:
        return _err(rid, 4137, str(exc))


@method("groups.desktop.complete")
def _(rid, params: dict) -> dict:
    """Commit the outcome of one classic-room compatibility command."""

    try:
        from gateway.desktop_room_mailbox import complete_command, default_db_path

        command = complete_command(
            default_db_path(),
            consumer_id=params.get("consumer_id"),
            command_id=params.get("command_id"),
            lease_token=params.get("lease_token"),
            success=params.get("success") is True,
            result=params.get("result", {}),
        )
        return _ok(rid, {"command": command})
    except Exception as exc:
        return _err(rid, 4131, str(exc))


@method("groups.desktop.renew")
def _(rid, params: dict) -> dict:
    """Renew one live classic-room command lease while its turn settles."""

    try:
        from gateway.desktop_room_mailbox import default_db_path, renew_command

        command = renew_command(
            default_db_path(),
            consumer_id=params.get("consumer_id"),
            command_id=params.get("command_id"),
            lease_token=params.get("lease_token"),
        )
        return _ok(rid, {"command": command})
    except Exception as exc:
        return _err(rid, 4132, str(exc))


@method("groups.peer.invite")
def _(rid, params: dict) -> dict:
    """Mint one target-issued room/profile grant for a prospective home."""
    try:
        from gateway.hosted_room_peer import (
            PROTOCOL_VERSION as ROOM_LINK_PROTOCOL_VERSION,
            decode_room_grant,
            gateway_room_grant_secret,
            issue_room_grant,
            local_catalog_mapping,
        )
        from gateway import hosted_rooms

        if not _room_link_run_storage_durable():
            raise ValueError("durable run idempotency storage is required")
        installation_id = hosted_rooms.local_authority_gateway_id()
        profile = _requested_profile(params)
        ttl = float(params.get("ttl_seconds", 3600))
        if not 60 <= ttl <= 24 * 60 * 60:
            raise ValueError("ttl_seconds must be between 60 and 86400")
        grant_secret = gateway_room_grant_secret()
        execution_policy = _profile_execution_policy(profile)
        token = issue_room_grant(
            grant_secret,
            grant_id=str(params.get("grant_id") or f"grant-{os.urandom(16).hex()}"),
            room_id=str(params.get("room_id") or ""),
            home_install_id=str(params.get("home_install_id") or ""),
            authority_gateway_id=str(
                params.get("authority_gateway_id") or ""
            ),
            authority_epoch=int(params.get("authority_epoch") or 0),
            member_id=str(params.get("member_id") or ""),
            target_install_id=installation_id,
            target_profile=profile,
            execution_policy_digest=execution_policy["policy_digest"],
            ttl_seconds=ttl,
        )
        claims = decode_room_grant(grant_secret, token, permission="status")
        hosted_rooms.reserve_peer_room(
            hosted_rooms.default_db_path(),
            claims=claims,
            expires_at=float(claims.get("status_expires_at", claims["expires_at"])),
        )
        catalog = local_catalog_mapping(
            installation_id=installation_id,
            protocol_versions=(ROOM_LINK_PROTOCOL_VERSION,),
            link_modes=("direct",),
            text=True,
            attachments=False,
            target_profile=profile,
            execution_policy=execution_policy,
        )
        return _ok(
            rid,
            {
                "grant": token,
                "target_profile": profile,
                "catalog": catalog,
                "endpoint": catalog["endpoint"],
            },
        )
    except Exception as exc:
        return _err(rid, 4120, str(exc))


@method("groups.peer.revoke")
def _(rid, params: dict) -> dict:
    """Revoke one target-issued grant using its exact profile scope."""
    try:
        from gateway import hosted_rooms
        from gateway.hosted_room_peer import decode_room_grant, gateway_room_grant_secret

        profile = _requested_profile(params)
        claims = decode_room_grant(
            gateway_room_grant_secret(),
            str(params.get("grant") or ""),
            permission="status",
            allow_expired_for_revocation=True,
        )
        if (
            claims["target_profile"] != profile
            or claims["target_install_id"]
            != hosted_rooms.local_authority_gateway_id()
        ):
            raise ValueError("room grant target does not match this profile")
        hosted_rooms.revoke_room_grant_scope(
            hosted_rooms.default_db_path(),
            claims=claims,
            expires_at=float(
                claims.get("status_expires_at", claims["expires_at"])
            ),
        )
        _revoke_peer_room_control(
            str(claims["room_id"]), str(claims["member_id"])
        )
        return _ok(rid, {"revoked": True})
    except Exception as exc:
        return _err(rid, 4122, str(exc))


@method("groups.peer.revoke_exact")
def _(rid, params: dict) -> dict:
    """Revoke only this bearer grant, preserving concurrent replacements."""
    try:
        from gateway import hosted_rooms
        from gateway.hosted_room_peer import decode_room_grant, gateway_room_grant_secret

        profile = _requested_profile(params)
        claims = decode_room_grant(
            gateway_room_grant_secret(),
            str(params.get("grant") or ""),
            permission="status",
            allow_expired_for_revocation=True,
        )
        if (
            claims["target_profile"] != profile
            or claims["target_install_id"]
            != hosted_rooms.local_authority_gateway_id()
        ):
            raise ValueError("room grant target does not match this profile")
        hosted_rooms.revoke_room_grant_id(
            hosted_rooms.default_db_path(),
            claims=claims,
            expires_at=float(
                claims.get("status_expires_at", claims["expires_at"])
            ),
        )
        return _ok(rid, {"revoked": True})
    except Exception as exc:
        return _err(rid, 4122, str(exc))


@method("groups.peer.register")
def _(rid, params: dict) -> dict:
    """Register and probe one scoped target route on the room home."""
    service = get_hosted_room_service()
    if service is None:
        return _err(rid, 4121, "hosted room driver is unavailable")
    try:
        from gateway.hosted_room_peer import (
            GatewayRoomCatalog,
            PROTOCOL_VERSION as ROOM_LINK_PROTOCOL_VERSION,
            validate_room_link_url,
        )
        from gateway.hosted_rooms import local_authority_gateway_id, room_state
        from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
        from tui_gateway.hosted_room_peer_transport import PeerMemberRoute

        target_url, transport_security = validate_room_link_url(
            params.get("target_url")
        )
        catalog = GatewayRoomCatalog.from_mapping(params.get("catalog"))
        if ROOM_LINK_PROTOCOL_VERSION not in catalog.protocol_versions:
            raise ValueError(
                f"target does not support RoomLink protocol v{ROOM_LINK_PROTOCOL_VERSION}"
            )
        if "direct" not in catalog.link_modes:
            raise ValueError("target does not support a direct RoomLink")
        target_profile = str(params.get("target_profile") or "")
        grant = str(params.get("grant") or "")
        expected_grant_sha256 = None
        if "expected_grant_sha256" in params:
            expected_grant_sha256 = str(params.get("expected_grant_sha256") or "")
            if expected_grant_sha256 and (
                len(expected_grant_sha256) != 64
                or any(character not in "0123456789abcdef" for character in expected_grant_sha256)
            ):
                raise ValueError("expected_grant_sha256 must be a sha256 digest")
        client = PeerRunsHTTPClient(
            base_url=target_url,
            api_key="",
            receipt_db_path=service.db_path,
        )
        probe = client.probe(grant=grant)
        live_catalog = GatewayRoomCatalog.from_mapping(probe.get("catalog"))
        if live_catalog != catalog:
            raise ValueError("target capability catalog changed during setup")
        if (
            ROOM_LINK_PROTOCOL_VERSION not in live_catalog.protocol_versions
            or "direct" not in live_catalog.link_modes
        ):
            raise ValueError("target RoomLink capability is incompatible")
        room_id = str(params.get("room_id") or "")
        member_id = str(params.get("member_id") or "")
        home_install_id = local_authority_gateway_id()
        home_room = room_state(service.db_path, room_id=room_id)
        if (
            probe.get("room_id") != room_id
            or probe.get("home_install_id") != home_install_id
            or probe.get("authority_gateway_id")
            != home_room.get("authority_gateway_id")
            or int(probe.get("authority_epoch") or 0)
            != int(home_room.get("authority_epoch") or 0)
            or probe.get("member_id") != member_id
            or probe.get("target_profile") != target_profile
        ):
            raise ValueError("room grant scope does not match this route")
        route = PeerMemberRoute(
            home_install_id=home_install_id,
            member_id=member_id,
            target_install_id=catalog.installation_id,
            target_profile=target_profile,
            capability_digest=catalog.catalog_digest,
            execution_policy_digest=catalog.execution_policy.policy_digest,
            cancellation_scope_id=str(
                params.get("cancellation_scope_id")
                or f"cancel-{params.get('room_id') or ''}"
            ),
            trace_id=str(params.get("trace_id") or f"trace-{os.urandom(16).hex()}"),
            grant=grant,
        )
        service.register_peer_route(
            room_id=room_id,
            member_id=member_id,
            route=route,
            client=client,
            target_url=target_url,
            catalog=catalog,
            **(
                {"expected_grant_sha256": expected_grant_sha256}
                if expected_grant_sha256 is not None
                else {}
            ),
        )
        return _ok(
            rid,
            {
                "registered": True,
                "mode": "direct",
                "transport_security": transport_security,
                "target_install_id": catalog.installation_id,
                "target_profile": target_profile,
            },
        )
    except Exception as exc:
        return _err(rid, 5120, str(exc))


@method("groups.control.invite")
def _(rid, params: dict) -> dict:
    """Issue one durable return-control credential to a room participant."""

    try:
        from gateway import hosted_room_controls
        from gateway.hosted_room_peer import (
            PROTOCOL_VERSION as ROOM_LINK_PROTOCOL_VERSION,
            local_catalog_mapping,
        )
        from gateway.hosted_rooms import local_authority_gateway_id, room_state

        service = get_hosted_room_service()
        if service is None:
            return _err(rid, 4123, _WORKER_UNAVAILABLE)
        room = room_state(service.db_path, room_id=params.get("room_id"))
        member_id = str(params.get("member_id") or "")
        caller_install_id = str(params.get("caller_install_id") or "")
        member = next(
            (
                item
                for item in room["members"]
                if str(item.get("member_id") or "") == member_id
            ),
            None,
        )
        target = member.get("target") if isinstance(member, dict) else None
        if (
            not isinstance(target, dict)
            or target.get("kind") != "peer"
            or str(target.get("installation_id") or "") != caller_install_id
        ):
            raise ValueError("control participant does not match the frozen room member")
        profile = _requested_profile(params)
        catalog = local_catalog_mapping(
            installation_id=local_authority_gateway_id(),
            protocol_versions=(ROOM_LINK_PROTOCOL_VERSION,),
            link_modes=("direct",),
            text=True,
            attachments=False,
            target_profile=profile,
            execution_policy=_profile_execution_policy(profile),
        )
        endpoint = catalog.get("endpoint")
        home_url = (
            str(endpoint.get("url") or "")
            if isinstance(endpoint, dict) and endpoint.get("available") is True
            else ""
        )
        if not home_url:
            raise ValueError("room authority has no reachable control endpoint")
        request_id = str(params.get("request_id") or "").strip()
        if not request_id:
            raise ValueError("room control invitation requires request_id")
        now = time.time()
        issued = hosted_room_controls.issue_home_control_token(
            service.db_path,
            room_id=room["room_id"],
            member_id=member_id,
            authority_gateway_id=room["authority_gateway_id"],
            authority_epoch=int(room["authority_epoch"]),
            expires_at=hosted_room_controls.ROOM_LIFETIME_EXPIRES_AT,
            request_id=request_id,
            now=now,
        )
        return _ok(
            rid,
            {
                "room_id": issued.room_id,
                "member_id": issued.member_id,
                "authority_gateway_id": issued.authority_gateway_id,
                "authority_epoch": issued.authority_epoch,
                "room_name": str(room.get("name") or room["room_id"]),
                "member_count": len(room["members"]),
                "control_token": issued.control_token,
                "home_url": home_url,
                "expires_at": issued.expires_at,
            },
        )
    except Exception as exc:
        return _err(rid, 4150, str(exc))


@method("groups.control.register")
def _(rid, params: dict) -> dict:
    """Persist one private return route on the participating gateway."""

    try:
        from gateway import hosted_room_controls
        from gateway.hosted_room_control_client import RoomControlHTTPClient
        from gateway.hosted_rooms import default_db_path

        profile = _requested_profile(params)
        if not hosted_room_controls.peer_reservation_matches(
            default_db_path(),
            room_id=params.get("room_id"),
            member_id=params.get("member_id"),
            target_profile=profile,
            authority_gateway_id=params.get("authority_gateway_id"),
            authority_epoch=int(params.get("authority_epoch") or 0),
        ):
            raise ValueError("room control route has no matching live reservation")
        saved = hosted_room_controls.save_peer_control_link(
            default_db_path(),
            room_id=params.get("room_id"),
            member_id=params.get("member_id"),
            home_url=params.get("home_url"),
            authority_gateway_id=params.get("authority_gateway_id"),
            authority_epoch=int(params.get("authority_epoch") or 0),
            room_name=params.get("room_name"),
            member_count=params.get("member_count"),
            control_token=params.get("control_token"),
            expires_at=params.get("expires_at"),
            allow_rotation=True,
        )
        try:
            summary = RoomControlHTTPClient(saved.link).summary()
            summary_room = summary.get("room") if isinstance(summary, dict) else None
            if (
                not isinstance(summary_room, dict)
                or str(summary_room.get("room_id") or "") != saved.link.room_id
                or str(summary_room.get("authority_gateway_id") or "")
                != saved.link.authority_gateway_id
                or int(summary_room.get("authority_epoch") or 0)
                != saved.link.authority_epoch
            ):
                raise ValueError("room control authority returned mismatched scope")
        except Exception:
            hosted_room_controls.delete_peer_control_links(
                default_db_path(),
                room_id=saved.link.room_id,
                member_id=saved.link.member_id,
            )
            raise
        return _ok(
            rid,
            {
                "registered": True,
                "idempotent": saved.idempotent,
                "room_id": saved.link.room_id,
                "member_id": saved.link.member_id,
            },
        )
    except Exception as exc:
        return _err(rid, 4151, str(exc))


@method("groups.control.revoke")
def _(rid, params: dict) -> dict:
    """Revoke a participant's private return route idempotently."""

    try:
        room_id = str(params.get("room_id") or "")
        member_id = str(params.get("member_id") or "")
        removed = _revoke_peer_room_control(room_id, member_id)
        return _ok(rid, {"revoked": removed})
    except Exception as exc:
        return _err(rid, 4152, str(exc))


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
        driver_status = (
            service.status_with_grant_fingerprints(str(room["room_id"]))
            if service is not None and room.get("disbanded_at") is None
            else None
        )
        return _ok(
            rid,
            {
                "room": room,
                **({"driver_status": driver_status} if driver_status else {}),
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


@method("groups.rename")
def _(rid, params: dict) -> dict:
    """Rename one hosted room atomically with its replay event."""
    from gateway.hosted_rooms import HostedRoomError, default_db_path, rename_room

    try:
        renamed = rename_room(
            default_db_path(),
            room_id=params.get("room_id"),
            event_id=params.get("event_id"),
            name=params.get("name"),
        )
        return _ok(rid, {"room": renamed})
    except HostedRoomError as exc:
        reason = getattr(exc, "reason", None)
        return _err(rid, 4117, str(exc), {"reason": reason} if reason else None)
    except Exception as exc:
        return _err(rid, 5117, str(exc))


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

        def disband_with_controls(state: dict | None = None) -> dict:
            from gateway import hosted_room_controls

            local_gateway_id = local_authority_gateway_id()
            if state is not None and (
                str(state["authority_gateway_id"]) != local_gateway_id
            ):
                raise AuthorityConflictError(
                    "This Group Chat is managed by another gateway."
                )
            tombstone = disband_room(
                service.db_path,
                room_id=params.get("room_id"),
                expected_gateway_id=str(
                    local_gateway_id
                ),
                expected_epoch=int(
                    state["authority_epoch"] if state is not None else 1
                ),
            )
            hosted_room_controls.revoke_home_control_tokens(
                service.db_path,
                room_id=params.get("room_id"),
            )
            return tombstone

        try:
            existing = room_state(
                service.db_path,
                room_id=params.get("room_id"),
                include_disbanded=True,
            )
        except RoomHistoryExpiredError:
            tombstone = disband_with_controls()
            return _ok(rid, {"tombstone": tombstone})
        if existing.get("disbanded_at") is not None:
            tombstone = disband_with_controls(existing)
            return _ok(rid, {"tombstone": tombstone})
        service.stop_room(
            str(params.get("room_id") or ""),
            cancel_id=str(params.get("cancel_id") or "room-disbanded"),
            require_acknowledged=True,
        )
        service.revoke_room_routes(str(params.get("room_id") or ""))
        tombstone = disband_with_controls(existing)
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
    """Resolve one exact approval requested by a local or peer room member."""

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
            retry_id=str(params.get("command_id") or "") or None,
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
    """Fail closed until replica ingest is bound to verified RoomLink claims."""
    return _err(
        rid,
        4116,
        "Group Chat replication requires a verified RoomLink grant.",
        {"reason": "replica_provenance_required"},
    )


@method("groups.replica_state")
def _(rid, params: dict) -> dict:
    """Report the local replica's coverage and authority lineage."""
    from gateway.hosted_room_replicas import ReplicaError, replica_state
    from gateway.hosted_rooms import default_db_path

    try:
        return _ok(rid, replica_state(default_db_path(), room_id=params.get("room_id")))
    except ReplicaError as exc:
        reason = getattr(exc, "reason", None)
        return _err(rid, 4117, str(exc), {"reason": reason} if reason else None)
    except Exception as exc:
        return _err(rid, 5117, str(exc))


@method("groups.promote")
def _(rid, params: dict) -> dict:
    """Fail closed for clients that cached the retired takeover method."""
    return _err(
        rid,
        4118,
        "Group Chat takeover is disabled until Hermes can select one globally "
        "exclusive authority.",
        {"reason": "authority_takeover_disabled"},
    )


@method("groups.demote")
def _(rid, params: dict) -> dict:
    """Fail closed for clients that cached the retired demotion method."""
    return _err(
        rid,
        4119,
        "Group Chat authority changes require a verified takeover decision.",
        {"reason": "authority_takeover_disabled"},
    )


def register(server) -> None:
    _registry.install(server)
