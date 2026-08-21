"""Matrix room-admin agent tools.

Three opt-in tools that complete the Matrix room lifecycle, wiring the
already-present-but-unused ``MATRIX_TOOLS_ALLOW_ROOM_CREATE`` gate:

  * ``matrix_create_room`` — create a room (preset / topic / invite / encryption)
  * ``matrix_leave_room``  — leave (unjoin) a room
  * ``matrix_delete_room`` — leave **+ forget** (account-level delete)

They are registered under the ``hermes-matrix`` toolset (see ``toolsets.py``)
and are NOT part of ``_HERMES_CORE_TOOLS``, so they are offered only to
Matrix sessions — never to the CLI / other messaging platforms. That matches
the Matrix scoping contract in
``website/docs/user-guide/messaging/matrix.md`` ("these tools are scoped to
Matrix contexts and are not available in non-Matrix toolsets").

Implementation note — why a raw Client-Server API call and NOT
``adapter.create_room()``:
  * The agent tool loop runs on a DIFFERENT asyncio event loop than the live
    MatrixAdapter's mautrix client. Awaiting the adapter's coroutine (which
    drives the client's aiohttp session) cross-loop raises
    "Timeout context manager should be used inside a task".
  * ``adapter.create_room()`` also eagerly does ``self._joined_rooms.add(id)``,
    which makes the gateway's ``_join_room_by_id`` guard skip a proper join of
    the freshly-created room (the "self-created room is dead for dispatch" bug).
  A fresh aiohttp POST to ``/_matrix/client/v3/createRoom`` runs cleanly on the
  agent loop and leaves ``_joined_rooms`` untouched, so the live client sees the
  new room through its normal sync path. This mirrors the raw-HTTP
  Client-Server pattern used by the Matrix sender in tools/send_message_tool.py.

Authorization: leave/forget are destructive and must respect the Matrix room
boundary (``gateway/session.py`` — "a turn is scoped to the current Matrix
room/thread only"). ``_authorize_room`` therefore refuses to act on a room the
agent was never in, not just any non-empty id it happens to be handed. It
fails CLOSED: if no scope can be established at all (no current room bound
AND no ``MATRIX_ALLOWED_ROOMS`` allowlist), a destructive action is denied
rather than silently allowed — an operator who genuinely needs a room-less
cron/standalone run to act cross-room sets ``MATRIX_TOOLS_ALLOW_ANY_ROOM=true``
to opt in explicitly.

Cache reconciliation: a raw leave/forget bypasses the sync loop, which only
*adds* room ids to ``_joined_rooms``. After a successful leave we call
``MatrixAdapter.reconcile_left_room`` to evict the id so a later
``_join_room_by_id`` re-joins rather than trusting a stale cache entry.
"""
import os
from typing import Optional

from tools.registry import registry, tool_error, tool_result

MATRIX_CREATE_ROOM_SCHEMA = {
    "name": "matrix_create_room",
    "description": (
        "Create a new Matrix room and return its room_id. Requires the matrix "
        "platform to be configured. Rooms are private by default; pass invite to "
        "add users (full Matrix IDs like '@alice:matrix.example.org')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Room display name."},
            "topic": {"type": "string", "description": "Room topic/description."},
            "invite": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Matrix user IDs to invite, e.g. ['@alice:matrix.example.org'].",
            },
            "is_direct": {"type": "boolean", "description": "Mark as a direct (DM) room. Default false."},
            "preset": {
                "type": "string",
                "enum": ["private_chat", "trusted_private_chat", "public_chat"],
                "description": "Visibility preset. Default private_chat. public_chat needs MATRIX_ALLOW_PUBLIC_ROOMS=true.",
            },
            "encrypted": {
                "type": "boolean",
                "description": "Create the room end-to-end encrypted (megolm). Default false.",
            },
        },
        "required": [],
    },
}

MATRIX_LEAVE_ROOM_SCHEMA = {
    "name": "matrix_leave_room",
    "description": (
        "Leave (unjoin) a Matrix room you are a member of. The room keeps "
        "existing for its other members; you simply stop participating. Pass the "
        "room_id (e.g. '!abc123:matrix.example.org'). You may only leave a room "
        "you are actually in — this turn's room, a room listed in "
        "MATRIX_ALLOWED_ROOMS, or a room the bot is still joined to."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "room_id": {
                "type": "string",
                "description": "Room to leave, e.g. '!abc123:matrix.example.org'.",
            },
            "reason": {
                "type": "string",
                "description": "Optional human-readable reason recorded in the leave event.",
            },
        },
        "required": ["room_id"],
    },
}

MATRIX_DELETE_ROOM_SCHEMA = {
    "name": "matrix_delete_room",
    "description": (
        "Delete a Matrix room from your account: leave it and then forget it, so "
        "it disappears from your room list. For a room you created and are the only "
        "member of, this effectively tears it down. NOTE: Matrix has no true "
        "server-side delete for regular users — any other members keep their own "
        "copy; a full server purge requires a homeserver admin. You may only delete "
        "a room you are actually in (this turn's room, a MATRIX_ALLOWED_ROOMS "
        "entry, or a room the bot is still joined to). Pass the room_id."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "room_id": {
                "type": "string",
                "description": "Room to delete (leave + forget), e.g. '!abc123:matrix.example.org'.",
            },
            "reason": {
                "type": "string",
                "description": "Optional reason recorded in the leave event.",
            },
        },
        "required": ["room_id"],
    },
}


def _check_matrix_room_admin() -> bool:
    """Gate for all three room-admin tools, reusing the documented
    ``MATRIX_TOOLS_ALLOW_ROOM_CREATE`` capability flag (off by default).
    If the agent may create Matrix rooms, it may also leave/forget them — one
    'room admin' capability, no extra env wiring."""
    return os.getenv("MATRIX_TOOLS_ALLOW_ROOM_CREATE", "").lower() in ("true", "1", "yes")


def _live_adapter():
    """Return the running gateway's live Matrix adapter, or None.

    Shared by the credential lookup, the room-authorization check, and the
    post-leave cache reconciliation so all three consult the *same* live
    instance the gateway actually authenticated with. Best-effort: any lookup
    failure (CLI / cron / a gateway without Matrix) yields None.
    """
    try:
        from gateway.run import _gateway_runner_ref
        from gateway.config import Platform

        runner = _gateway_runner_ref()
    except Exception:
        return None
    if runner is None:
        return None
    try:
        return runner.adapters.get(Platform.MATRIX)
    except Exception:
        return None


def _matrix_creds():
    """Return (homeserver, token), preferring the live adapter's connected
    values, falling back to env. Keeps us in lock-step with whatever the
    running gateway actually authenticated with."""
    homeserver = ""
    token = ""
    adapter = _live_adapter()
    if adapter is not None:
        homeserver = getattr(adapter, "_homeserver", "") or ""
        token = getattr(adapter, "_access_token", "") or ""
    homeserver = (homeserver or os.getenv("MATRIX_HOMESERVER", "")).rstrip("/")
    token = token or os.getenv("MATRIX_ACCESS_TOKEN", "")
    return homeserver, token


def _current_room() -> str:
    """Return the room id this turn is scoped to, or "".

    The gateway binds the current Matrix room to
    ``HERMES_SESSION_CHAT_ID`` (``_set_session_env`` in gateway/run.py), which
    is the room boundary documented in ``gateway/session.py`` ("a turn is
    scoped to the current Matrix room/thread only"). Reading it via
    ``get_session_env`` uses the task-local ContextVar first, so concurrent
    turns in different rooms never see each other's room.
    """
    try:
        from gateway.session_context import get_session_env
    except Exception:
        return ""
    return (get_session_env("HERMES_SESSION_CHAT_ID", "") or "").strip()


def _allowed_room_ids():
    """Operator-configured cross-room allowlist (``MATRIX_ALLOWED_ROOMS``)."""
    return {r.strip() for r in os.getenv("MATRIX_ALLOWED_ROOMS", "").split(",") if r.strip()}


def _joined_rooms() -> Optional[set]:
    """The set of rooms the live adapter is a member of, or None when the
    adapter isn't available. A room in this set is one the agent *was in* —
    so it is a legitimate leave/delete target even if it isn't the current room.

    ``None`` (adapter unavailable) is distinct from an *empty* set: it means we
    have no positive membership evidence, NOT that every room is in scope.
    Callers must treat ``None`` as "unknown", never as "allow"."""
    adapter = _live_adapter()
    if adapter is None:
        return None
    try:
        return set(adapter._joined_rooms)
    except Exception:
        return None


def _allow_any_room() -> bool:
    """Explicit operator opt-in to act on arbitrary rooms from a room-less
    context (``MATRIX_TOOLS_ALLOW_ANY_ROOM=true``).

    Only consulted when NO scope could be established at all (no current room
    bound AND no ``MATRIX_ALLOWED_ROOMS`` allowlist) — the otherwise
    fail-closed path. Setting an allowlist still acts as a whitelist; this
    flag only lifts the denial when there is nothing to scope against.
    """
    return os.getenv("MATRIX_TOOLS_ALLOW_ANY_ROOM", "").lower() in ("true", "1", "yes")


def _authorize_room(room_id: str):
    """Enforce the Matrix room boundary for destructive leave/forget actions.

    Returns None when *room_id* is a legitimate target, or a ``tool_error``
    string when it is not. A room is legitimate when it is any of:
      * in the operator's ``MATRIX_ALLOWED_ROOMS`` allowlist, or
      * this turn's room (``_current_room``), or
      * a room the live adapter is still joined to (the agent was in it) —
        a *positive* membership match, so the adapter being unavailable
        (``_joined_rooms() is None``) never counts as "in every room".

    Everything else fails CLOSED:
      * current room bound but the room differs from it — cross-room action,
        exactly what the session boundary forbids;
      * an allowlist is set and the room isn't in it — strict whitelist;
      * no current room and no allowlist (room-less cron/standalone run) —
        denied unless ``MATRIX_TOOLS_ALLOW_ANY_ROOM=true`` is set explicitly.
    Failing closed means a bug that unsets the session room mid-Matrix-session
    degrades to "tool says no" rather than "tool silently acts on any room".
    """
    current = _current_room()
    allowlist = _allowed_room_ids()
    if room_id in allowlist:
        return None
    if current and room_id == current:
        return None
    joined = _joined_rooms()
    if joined is not None and room_id in joined:
        return None
    if current:
        return tool_error(
            f"Room {room_id} is outside this turn's scope (this turn is in {current}). "
            "To act on another room, run the tool from that room's conversation or "
            "add its room_id to MATRIX_ALLOWED_ROOMS."
        )
    if allowlist:
        return tool_error(
            f"Room {room_id} is not in MATRIX_ALLOWED_ROOMS ({sorted(allowlist)})."
        )
    # No current room bound and no allowlist: there is no scope to check
    # against, and an unavailable adapter is no evidence of membership.
    # Fail closed unless the operator opted in explicitly.
    if _allow_any_room():
        return None
    return tool_error(
        f"No scope established for {room_id}: this turn has no bound Matrix room "
        "and MATRIX_ALLOWED_ROOMS is not set. Destructive room actions fail closed "
        "without a scope. Run the tool from the room's own conversation, set "
        "MATRIX_ALLOWED_ROOMS, or set MATRIX_TOOLS_ALLOW_ANY_ROOM=true to allow "
        "arbitrary rooms from a room-less context."
    )


def _reconcile_adapter(room_id: str) -> None:
    """Best-effort: evict *room_id* from the live adapter's membership caches
    after a raw leave/forget so a later ``_join_room_by_id`` re-joins instead
    of trusting a stale entry. No-op when the gateway isn't running or the
    adapter lacks the method (e.g. an older adapter).
    """
    adapter = _live_adapter()
    if adapter is None:
        return
    try:
        adapter.reconcile_left_room(room_id)
    except Exception:
        pass


async def _matrix_room_action(homeserver, token, room_id, action, body=None):
    """POST /_matrix/client/v3/rooms/{room_id}/{action} (action = leave|forget)
    via a fresh aiohttp session on the agent loop. Returns (status, text)."""
    import aiohttp
    from urllib.parse import quote

    url = f"{homeserver}/_matrix/client/v3/rooms/{quote(room_id, safe='')}/{action}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        async with session.post(url, headers=headers, json=body or {}) as resp:
            return resp.status, await resp.text()


async def _matrix_create_room_api(homeserver, token, body):
    """POST /_matrix/client/v3/createRoom via a fresh aiohttp session on the
    agent loop. Returns (status, parsed_json)."""
    import aiohttp

    url = f"{homeserver}/_matrix/client/v3/createRoom"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        async with session.post(url, headers=headers, json=body) as resp:
            text = await resp.text()
            try:
                data = await resp.json()
            except Exception:
                data = {"_raw": text}
            return resp.status, data


async def _handle_matrix_create_room(args, **kwargs):
    homeserver, token = _matrix_creds()
    if not homeserver or not token:
        return tool_error(
            "Matrix not configured (MATRIX_HOMESERVER + MATRIX_ACCESS_TOKEN required)."
        )

    preset = args.get("preset", "private_chat") or "private_chat"
    if preset == "public_chat" and os.getenv("MATRIX_ALLOW_PUBLIC_ROOMS", "").lower() not in (
        "true",
        "1",
        "yes",
    ):
        return tool_error("Refusing to create a public room without MATRIX_ALLOW_PUBLIC_ROOMS=true.")

    body = {"preset": preset}
    if args.get("name"):
        body["name"] = str(args["name"])
    if args.get("topic"):
        body["topic"] = str(args["topic"])
    invite = args.get("invite") or []
    if invite:
        body["invite"] = [str(u) for u in invite]
    if args.get("is_direct"):
        body["is_direct"] = True
    if args.get("encrypted"):
        # Turn on megolm at creation via initial state. The room is encrypted
        # server-side immediately; the gateway's mautrix client must then set up
        # its outbound megolm session for the room (the genuinely-untested path).
        body["initial_state"] = [
            {
                "type": "m.room.encryption",
                "state_key": "",
                "content": {"algorithm": "m.megolm.v1.aes-sha2"},
            }
        ]

    try:
        status, data = await _matrix_create_room_api(homeserver, token, body)
    except Exception as exc:
        return tool_error(f"matrix_create_room request failed: {exc}")
    if status not in {200, 201}:
        return tool_error(f"Matrix createRoom error ({status}): {str(data)[:300]}")

    room_id = data.get("room_id")
    if not room_id:
        return tool_error(f"createRoom returned no room_id: {str(data)[:200]}")
    return tool_result(
        success=True,
        room_id=room_id,
        invited=invite,
        preset=preset,
        encrypted=bool(args.get("encrypted")),
    )


def _require_room(args):
    """Shared validation + authorization for leave/forget.

    Returns (homeserver, token, room_id) or (None, tool_error). Enforces the
    Matrix room boundary via ``_authorize_room``: a non-empty room_id is not
    enough — the agent must actually be in that room (this turn's room, a
    MATRIX_ALLOWED_ROOMS entry, or a room the bot is still joined to).
    """
    homeserver, token = _matrix_creds()
    if not homeserver or not token:
        return None, tool_error(
            "Matrix not configured (MATRIX_HOMESERVER + MATRIX_ACCESS_TOKEN required)."
        )
    room_id = str(args.get("room_id") or "").strip()
    if not room_id:
        return None, tool_error("room_id is required (e.g. '!abc123:matrix.example.org').")
    err = _authorize_room(room_id)
    if err is not None:
        return None, err
    return (homeserver, token, room_id), None


async def _handle_matrix_leave_room(args, **kwargs):
    ctx, err = _require_room(args)
    if err is not None or ctx is None:
        return err
    homeserver, token, room_id = ctx
    body = {"reason": str(args["reason"])} if args.get("reason") else {}
    try:
        status, text = await _matrix_room_action(homeserver, token, room_id, "leave", body)
    except Exception as exc:
        return tool_error(f"matrix_leave_room request failed: {exc}")
    if status != 200:
        return tool_error(f"Matrix leave error ({status}): {text[:300]}")
    _reconcile_adapter(room_id)
    return tool_result(success=True, room_id=room_id, action="leave")


async def _handle_matrix_delete_room(args, **kwargs):
    ctx, err = _require_room(args)
    if err is not None or ctx is None:
        return err
    homeserver, token, room_id = ctx
    body = {"reason": str(args["reason"])} if args.get("reason") else {}

    # 1) leave — tolerate "already not a member" (M_FORBIDDEN) as effectively-left
    try:
        lstatus, ltext = await _matrix_room_action(homeserver, token, room_id, "leave", body)
    except Exception as exc:
        return tool_error(f"matrix_delete_room leave failed: {exc}")
    already_gone = lstatus == 403 and "M_FORBIDDEN" in ltext
    if lstatus != 200 and not already_gone:
        return tool_error(f"Matrix leave (during delete) error ({lstatus}): {ltext[:300]}")
    _reconcile_adapter(room_id)

    # 2) forget — removes the room from this account's room list (requires having left)
    try:
        fstatus, ftext = await _matrix_room_action(homeserver, token, room_id, "forget", {})
    except Exception as exc:
        return tool_error(f"matrix_delete_room forget failed: {exc}")
    if fstatus != 200:
        return tool_error(f"Matrix forget error ({fstatus}): {ftext[:300]}")

    return tool_result(
        success=True,
        room_id=room_id,
        action="leave+forget",
        note=(
            "Left and forgotten — removed from your room list. Matrix has no true "
            "server-side delete for regular users; any other members keep their "
            "copy, and a full server purge requires a homeserver admin."
        ),
    )


# All three tools are registered under the ``hermes-matrix`` toolset only (see
# toolsets.py), so they reach Matrix sessions and never the CLI or other
# messaging platforms. Each is gated on MATRIX_TOOLS_ALLOW_ROOM_CREATE.
registry.register(
    name="matrix_create_room",
    toolset="hermes-matrix",
    schema=MATRIX_CREATE_ROOM_SCHEMA,
    handler=_handle_matrix_create_room,
    check_fn=_check_matrix_room_admin,
    is_async=True,
    emoji="\U0001F3E0",
    description="Create a Matrix room via the Client-Server API.",
)

registry.register(
    name="matrix_leave_room",
    toolset="hermes-matrix",
    schema=MATRIX_LEAVE_ROOM_SCHEMA,
    handler=_handle_matrix_leave_room,
    check_fn=_check_matrix_room_admin,
    is_async=True,
    emoji="\U0001F6AA",  # door
    description="Leave (unjoin) a Matrix room via the Client-Server API.",
)

registry.register(
    name="matrix_delete_room",
    toolset="hermes-matrix",
    schema=MATRIX_DELETE_ROOM_SCHEMA,
    handler=_handle_matrix_delete_room,
    check_fn=_check_matrix_room_admin,
    is_async=True,
    emoji="\U0001F5D1",  # wastebasket
    description="Delete a Matrix room (leave + forget) via the Client-Server API.",
)
