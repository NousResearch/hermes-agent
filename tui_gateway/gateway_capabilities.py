"""Gateway ``gateway.ready`` capability negotiation (issue #130702).

A mobile client cannot tell "no recovery guarantee" from "guaranteed recovery"
unless the greeting says so. This module is the single source of truth for what
the greeting advertises: a ``capabilities`` map plus a ``protocol`` version
object. Both ``tui_gateway/entry.py`` (stdio) and ``tui_gateway/ws.py``
(WebSocket) build the first frame from here so the two transports stay byte
compatible in the fields clients branch on.
"""

from __future__ import annotations

from typing import Any

# First versioned greeting. Bump when a capability's meaning changes in a way
# an old client would misread (a purely additive capability needs no bump).
GATEWAY_PROTOCOL_VERSION = 1


def gateway_capabilities() -> dict[str, Any]:
    """Capabilities this backend guarantees at greeting time.

    - ``event_replay``: ``session.events.since`` + per-session ``seq`` + ``replay_epoch``.
    - ``change_events``: ``*.changed`` broadcasts (clients may demote legacy polls).
    - ``turn_lease``: durable per-session turn fencing (``session_turn_leases``).
    - ``turn_recovery``: ``False`` — this backend offers NO guaranteed turn
      recovery across disconnect/restart; a client must not assume a lost turn
      will resume. The explicit ``False`` (rather than an absent key) is the
      whole point: absent vs false must not be ambiguous.
    - ``client_turn_id``: ``prompt.submit`` accepts ``client_turn_id`` for
      idempotency/turn identity.
    """
    return {
        "event_replay": True,
        "change_events": True,
        "turn_lease": True,
        "turn_recovery": False,
        "client_turn_id": True,
    }


def gateway_protocol() -> dict[str, Any]:
    """Version envelope for the greeting so clients can gate on shape."""
    return {"version": GATEWAY_PROTOCOL_VERSION}


def gateway_ready_payload(
    skin: dict[str, Any],
    replay_epoch: str,
    *,
    heartbeat: bool | None = None,
) -> dict[str, Any]:
    """Build a ``gateway.ready`` payload (contract-validated by the caller).

    ``heartbeat`` is WebSocket-only; stdio omits it (``None`` stays absent so
    the old shape validates unchanged).
    """
    payload: dict[str, Any] = {
        "skin": skin,
        "change_events": True,
        "replay_epoch": replay_epoch,
        "capabilities": gateway_capabilities(),
        "protocol": gateway_protocol(),
    }
    if heartbeat is not None:
        payload["heartbeat"] = heartbeat
    return payload
