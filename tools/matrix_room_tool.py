"""Matrix room-admin agent tools.

Exposes a `matrix_create_room` agent tool so an agent can create Matrix rooms on
demand. Gated by MATRIX_TOOLS_ALLOW_ROOM_CREATE (finally implementing the
documented-but-unwired flag noted in gateway/platforms/matrix.py).

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
"""
import os

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


def _check_matrix_create_room() -> bool:
    return os.getenv("MATRIX_TOOLS_ALLOW_ROOM_CREATE", "").lower() in ("true", "1", "yes")


def _matrix_creds():
    """Return (homeserver, token), preferring the live adapter's connected
    values, falling back to env. Keeps us in lock-step with whatever the
    running gateway actually authenticated with."""
    homeserver = ""
    token = ""
    try:
        from gateway.run import _gateway_runner_ref
        from gateway.config import Platform

        runner = _gateway_runner_ref()
        adapter = runner.adapters.get(Platform.MATRIX) if runner is not None else None
        if adapter is not None:
            homeserver = getattr(adapter, "_homeserver", "") or ""
            token = getattr(adapter, "_access_token", "") or ""
    except Exception:
        pass
    homeserver = (homeserver or os.getenv("MATRIX_HOMESERVER", "")).rstrip("/")
    token = token or os.getenv("MATRIX_ACCESS_TOKEN", "")
    return homeserver, token


async def _handle_matrix_create_room(args, **kwargs):
    try:
        import aiohttp
    except ImportError:
        return tool_error("aiohttp not installed. Run: pip install aiohttp")

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

    url = f"{homeserver}/_matrix/client/v3/createRoom"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(url, headers=headers, json=body) as resp:
                text = await resp.text()
                if resp.status not in {200, 201}:
                    return tool_error(f"Matrix createRoom error ({resp.status}): {text[:300]}")
                data = await resp.json()
    except Exception as exc:
        return tool_error(f"matrix_create_room request failed: {exc}")

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


registry.register(
    name="matrix_create_room",
    toolset="hermes-matrix",
    schema=MATRIX_CREATE_ROOM_SCHEMA,
    handler=_handle_matrix_create_room,
    check_fn=_check_matrix_create_room,
    is_async=True,
    emoji="\U0001F3E0",
    description="Create a Matrix room via the Client-Server API.",
)


# ===========================================================================
# matrix_leave_room + matrix_delete_room  (room-admin lifecycle completion)
#
# Same raw Client-Server API pattern as matrix_create_room above: a fresh
# aiohttp POST on the agent's own event loop (the live MatrixAdapter client
# runs on a different loop). leave/forget mirror create so the full room
# lifecycle (create -> leave -> delete) is available wherever create_room is.
# ===========================================================================

MATRIX_LEAVE_ROOM_SCHEMA = {
    "name": "matrix_leave_room",
    "description": (
        "Leave (unjoin) a Matrix room you are a member of. The room keeps "
        "existing for its other members; you simply stop participating. Pass the "
        "room_id (e.g. '!abc123:matrix.example.org') as returned by "
        "matrix_create_room. Use matrix_delete_room instead if you also want the "
        "room removed from your room list."
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
        "copy; a full server purge requires a homeserver admin. Pass the room_id."
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
    """Gate for leave/delete. Reuses the room-create capability flag: if the agent
    may create Matrix rooms, she may also leave/forget them. One 'room admin'
    capability, no extra env wiring."""
    return os.getenv("MATRIX_TOOLS_ALLOW_ROOM_CREATE", "").lower() in ("true", "1", "yes")


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


def _require_room(args):
    """Shared validation: returns (homeserver, token, room_id) or a tool_error."""
    homeserver, token = _matrix_creds()
    if not homeserver or not token:
        return None, tool_error(
            "Matrix not configured (MATRIX_HOMESERVER + MATRIX_ACCESS_TOKEN required)."
        )
    room_id = str(args.get("room_id") or "").strip()
    if not room_id:
        return None, tool_error("room_id is required (e.g. '!abc123:matrix.example.org').")
    return (homeserver, token, room_id), None


async def _handle_matrix_leave_room(args, **kwargs):
    ctx, err = _require_room(args)
    if err is not None:
        return err
    homeserver, token, room_id = ctx
    body = {"reason": str(args["reason"])} if args.get("reason") else {}
    try:
        status, text = await _matrix_room_action(homeserver, token, room_id, "leave", body)
    except Exception as exc:
        return tool_error(f"matrix_leave_room request failed: {exc}")
    if status != 200:
        return tool_error(f"Matrix leave error ({status}): {text[:300]}")
    return tool_result(success=True, room_id=room_id, action="leave")


async def _handle_matrix_delete_room(args, **kwargs):
    ctx, err = _require_room(args)
    if err is not None:
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
