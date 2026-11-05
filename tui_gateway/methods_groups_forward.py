"""Standalone backend -> owning gateway for Group Chat calls (WA-19).

A standalone ``hermes serve``/``hermes dashboard`` has no session authority: on its own it keeps
rooms in the root ``shared-state.db`` and drives them here. Once a ready gateway serves the
profile, that gateway's store is the one Group Chat list native Desktop and the Webapp share, so
the room calls a Desktop makes go to it instead (the owner-forward path prompts already use:
``tui_gateway/owner_forward.py``). The gateway imports the rooms this backend already hosts once
(``gateway/hosted_rooms_gateway_import.py``) and fences them here, so each room has one driver.

Routing, so no view ever loses a room it listed before:

- a room only the standalone store holds runs locally (a gateway cannot have it);
- a room handed to the gateway goes to the ROOT home's gateway, the only one that imports;
- anything else (``groups.create``, rooms the gateway made) goes to the selected profile's gateway;
- ``groups.list`` is that gateway's list plus the standalone rooms not handed over, plus, for a
  named profile, its handed rooms read from the root gateway.

The call runs locally only when no gateway serves the profile (``absent``, or an ``incompatible``
one that never imports) and the room was not handed over: a handed room is never written to the
standalone store, so with its gateway down every call on it is a retryable refusal. A
``starting``/``draining`` gateway is waited for and any other present one is a retryable refusal: a local write then would be hidden from the gateway's list. RoomLink's
gateway-to-gateway methods (``groups.peer.*``, replication, promote/demote) stay local.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def forwarded() -> frozenset[str]:
    """Every room method the gateway route serves (its ``GROUP_METHODS``), so a room the gateway
    owns answers the same calls on this wire; a hand list here fell behind each new verb (#145960)."""
    from gateway.session_group_controls import GROUP_METHODS
    return frozenset(GROUP_METHODS)

#: The owner was found but did not answer: a verdict for this call, never a local write instead.
OWNER_LOST_CODE = 5124
#: Gateway states with no owner that will import rooms written here later: run the call locally.
#: ``incompatible`` is an older gateway without the session authority; it never imports, and the
#: first current gateway imports what was written here on its start.
_LOCAL_STATES = frozenset({"absent", "incompatible"})
#: States on the way to (or back to) a ready owner: wait for it, never write locally meanwhile.
_TRANSIENT_STATES = frozenset({"starting", "draining"})
#: How long one call waits for a starting or restarting gateway before refusing (retryable).
READY_WAIT_S = 20.0
_READY_POLL_S = 0.25


class _OwnerBusy(Exception):
    """A gateway serves (or is about to serve) the home but cannot answer now."""


def _home(server, params: dict):
    from pathlib import Path
    profile = str(params.get("profile") or "").strip()
    if profile and profile != str(server._current_profile_name() or "").strip():
        home = server._profile_home(profile)
        if home is None:
            raise ValueError(f"profile '{profile}' is unavailable")
        return Path(home)
    from hermes_constants import get_hermes_home
    return Path(get_hermes_home())


def _root_home():
    from pathlib import Path
    from gateway.hosted_rooms import default_db_path
    return Path(default_db_path()).parent  # the home whose gateway imports the standalone rooms


def _discover(home):
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    return discover_gateway_endpoint(home, timeout=5)


def owner_endpoint(home):
    """The ready gateway's endpoint for ``home``, or ``None`` when none serves it (run locally).

    A gateway that is starting or restarting is waited for (bounded by :data:`READY_WAIT_S`); one
    that is present but unreachable (slow identify, conflict) raises :class:`_OwnerBusy`. Either way
    nothing is written to the standalone store while a gateway may import from it."""
    import time
    deadline = time.monotonic() + READY_WAIT_S
    while True:
        discovery = _discover(home)
        if discovery.state == "ready" and discovery.endpoint is not None:
            return discovery.endpoint
        if discovery.state in _LOCAL_STATES:
            return None
        if discovery.state not in _TRANSIENT_STATES or time.monotonic() >= deadline:
            raise _OwnerBusy(discovery.reason_code or discovery.state)
        time.sleep(_READY_POLL_S)


def _standalone_holder(room_id: str) -> str | None:
    """Where the standalone store says ``room_id`` lives: ``"gateway"`` (handed to the root
    gateway's store), ``"local"`` (any other room it holds: only this backend drives or mirrors
    it) or ``None`` (not a standalone room)."""
    from contextlib import closing
    from gateway.hosted_rooms import _read_connection, default_db_path
    from gateway.hosted_rooms_gateway_import import handoff_gateway_id
    source = default_db_path()
    if not source.is_file():
        return None
    with closing(_read_connection(source)) as conn:
        row = conn.execute("SELECT authority_gateway_id FROM hosted_rooms WHERE room_id=?",
                           (room_id,)).fetchone()
    if row is None:
        return None
    return "gateway" if str(row[0]) == handoff_gateway_id() else "local"


def _standalone_rooms(params: dict | None = None) -> list[dict]:
    """The standalone store's rooms that were not handed to the gateway (the gateway-less list)."""
    from gateway.hosted_rooms import MAX_ROOM_LIST_LIMIT, default_db_path, list_rooms
    if not default_db_path().is_file():
        return []
    return local_rooms(list_rooms(default_db_path(), limit=MAX_ROOM_LIST_LIMIT,
                                  include_disbanded=bool(params and params.get("include_disbanded") is True)))


def _call(server, rid, home, endpoint, method: str, params: dict) -> dict:
    from hermes_cli.gateway_client import GatewayRPCError
    from tui_gateway import owner_forward
    try:
        result = owner_forward.forward_rpc(home, endpoint, method, params)
    except GatewayRPCError as exc:
        return server._err(rid, exc.code if isinstance(exc.code, int) else 5000, str(exc), exc.data or None)
    except Exception as exc:  # health: allow BLE001 -- transport boundary: any dial/IO failure is one refusal
        logger.warning("Group Chat %s: the gateway that owns it did not answer: %s", method, exc)
        return server._err(rid, OWNER_LOST_CODE, "The Hermes gateway did not answer. Try again.")
    result = result if isinstance(result, dict) else {}
    if method == "groups.capabilities" and isinstance(result.get("methods"), list):
        # Advertise only what this wire can carry: the room methods it forwards.
        carried = forwarded()
        result = {**result, "methods": [name for name in result["methods"] if name in carried]}
    return server._ok(rid, result)


def _busy(server, rid, why) -> dict:
    return server._err(rid, OWNER_LOST_CODE,
                       f"The Hermes gateway is not ready yet ({why}). Try again in a moment.")


def forward(server, rid, method: str, params: dict) -> dict | None:
    """The owning gateway's answer to ``method`` as this wire's envelope, or ``None`` when the call
    runs locally: no gateway serves the profile, or the room lives only in the standalone store."""
    if method not in forwarded():
        return None
    try:
        home = _home(server, params)
        room_id = str(params.get("room_id") or "")
        holder = _standalone_holder(room_id) if room_id else None
        if holder == "local":
            return None  # only this backend holds it: never ask a gateway that cannot have it
        if holder == "gateway":
            # The room was handed to the root gateway's store, whichever view asks; that gateway
            # answers for its own profile, so the view's profile does not travel with the call.
            home = _root_home()
            params = {key: value for key, value in params.items() if key != "profile"}
        if method == "groups.list":
            return _list(server, rid, home, params)
        endpoint = owner_endpoint(home)
    except _OwnerBusy as exc:
        return _busy(server, rid, exc)
    except ValueError as exc:
        return server._err(rid, 4000, str(exc))
    if endpoint is None:
        if holder == "gateway":
            # Only the gateway's store holds this room; the standalone copy is frozen and never
            # imported again, so a local call would acknowledge a change the owner never sees.
            return server._err(rid, OWNER_LOST_CODE, "The Hermes gateway that holds this Group Chat "
                               "is not running. Start it and try again.")
        return None
    return _call(server, rid, home, endpoint, method, params)


def _list(server, rid, home, params: dict) -> dict | None:
    """One Group Chat list that hides no existing room: the profile's gateway rooms, plus (on the
    first page) the rooms only the standalone store holds and, for a named profile, the standalone
    rooms already handed to the root gateway."""
    from gateway.hosted_rooms import MAX_ROOM_LIST_LIMIT, default_db_path
    from gateway.hosted_rooms_gateway_import import handed_room_ids
    endpoint = owner_endpoint(home)
    root = _root_home()
    handed = handed_room_ids(default_db_path()) if default_db_path().is_file() else set()
    root_endpoint = owner_endpoint(root) if handed and home.resolve() != root.resolve() else None
    if endpoint is None and root_endpoint is None:
        return None  # the local listing (gateway-less) is the whole list
    if endpoint is None:
        reply = server._ok(rid, {"rooms": [], "next_offset": None})
    else:
        reply = _call(server, rid, home, endpoint, "groups.list", params)
    if "error" in reply or int(params.get("offset") or 0) != 0:
        return reply
    rooms = list(reply["result"].get("rooms") or [])
    extra = _standalone_rooms(params)
    if root_endpoint is not None:
        page = _call(server, rid, root, root_endpoint, "groups.list",
                     {"include_disbanded": params.get("include_disbanded") is True,
                      "limit": MAX_ROOM_LIST_LIMIT})
        if "error" in page:
            return page
        extra += [room for room in page["result"].get("rooms") or [] if str(room.get("room_id")) in handed]
    seen = {str(room.get("room_id")) for room in rooms}
    for room in extra:
        if str(room.get("room_id")) not in seen:
            seen.add(str(room.get("room_id")))
            rooms.append(room)
    return server._ok(rid, {**reply["result"], "rooms": rooms})


def forward_bound(rid, method: str, params: dict) -> dict | None:
    """:func:`forward` on the server ``methods_groups`` is bound to (``None``: none bound yet)."""
    from tui_gateway import methods_groups
    server = methods_groups._bound_server
    return None if server is None else forward(server, rid, method, params)


def local_rooms(rooms: list[dict]) -> list[dict]:
    """A gateway-less listing without the rooms already handed to the gateway's store."""
    from gateway.hosted_rooms_gateway_import import handoff_gateway_id
    handoff = handoff_gateway_id()
    return [room for room in rooms if room.get("authority_gateway_id") != handoff]
