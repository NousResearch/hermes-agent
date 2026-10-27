"""Standalone backend -> owning gateway for Group Chat calls (WA-19).

A standalone ``hermes serve``/``hermes dashboard`` has no session authority: on its own it keeps
rooms in the root ``shared-state.db`` and drives them here. Once a ready gateway serves the
profile, that gateway's store is the one Group Chat list native Desktop and the Webapp share, so
the room calls a Desktop makes go to it instead (the owner-forward path prompts already use:
``tui_gateway/owner_forward.py``). The gateway imports the rooms this backend already hosts once
(``gateway/hosted_rooms_gateway_import.py``) and fences them here, so each room has one driver.

With no gateway running, the calls run locally exactly as before. RoomLink's gateway-to-gateway
methods (``groups.peer.*``, replication, promote/demote) are not Desktop calls and stay local.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: The Desktop's room calls this wire serves; the gateway route serves each (GROUP_METHODS).
FORWARDED = frozenset({
    "groups.capabilities", "groups.list", "groups.create", "groups.state", "groups.log",
    "groups.send", "groups.rename", "groups.disband", "groups.stop", "groups.retry",
    "groups.approve"})
#: The owner was found but did not answer: a verdict for this call, never a local write instead.
OWNER_LOST_CODE = 5124


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


def forward(server, rid, method: str, params: dict) -> dict | None:
    """The owning gateway's answer to ``method`` as this wire's envelope, or ``None`` when no ready
    gateway serves the profile (the caller runs it locally)."""
    from tui_gateway import owner_forward
    if method not in FORWARDED:
        return None
    try:
        home = _home(server, params)
        endpoint = owner_forward.discover_owner(home)
    except owner_forward.OwnerNotRunning:
        return None
    except ValueError as exc:
        return server._err(rid, 4000, str(exc))
    from hermes_cli.gateway_client import GatewayRPCError
    try:
        result = owner_forward.forward_rpc(home, endpoint, method, params)
    except GatewayRPCError as exc:
        return server._err(rid, exc.code if isinstance(exc.code, int) else 5000, str(exc), exc.data or None)
    except Exception as exc:  # health: allow BLE001 -- transport boundary: any dial/IO failure is one refusal
        logger.warning("Group Chat %s: the gateway that owns it did not answer: %s", method, exc)
        return server._err(rid, OWNER_LOST_CODE, "The Hermes gateway did not answer. Try again.")
    result = result if isinstance(result, dict) else {}
    if method == "groups.capabilities" and isinstance(result.get("methods"), list):
        # Advertise only what this wire can carry (attachments and Discard have no contract here).
        result = {**result, "methods": [name for name in result["methods"] if name in FORWARDED]}
    return server._ok(rid, result)


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
