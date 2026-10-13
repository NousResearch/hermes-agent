"""Matrix room-admin tools (matrix_create_room / matrix_leave_room / matrix_delete_room).

Let the agent create, leave, and forget Matrix rooms on demand. All three are
gated by MATRIX_TOOLS_ALLOW_ROOM_CREATE (off by default — opt-in), so nothing
here is reachable unless the operator explicitly enables it.

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
  new room through its normal sync path. This mirrors the proven
  ``_send_matrix_via_adapter`` raw-HTTP pattern in tools/send_message_tool.py.
"""
import asyncio
import json
from urllib.parse import quote

from agent.secret_scope import get_secret_str
from gateway.session_context import get_session_env
from tools.registry import registry, tool_error, tool_result

MATRIX_CREATE_ROOM_SCHEMA = {
    "name": "matrix_create_room",
    "description": (
        "Create a new Matrix room and return its room_id. Requires the matrix "
        "platform to be configured. Rooms are private by default; pass invite to "
        "add users (full Matrix IDs like '@alice:example.org')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Room display name."},
            "topic": {"type": "string", "description": "Room topic/description."},
            "invite": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Matrix user IDs to invite, e.g. ['@alice:example.org'].",
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


def _flag(name: str) -> bool:
    """An operator on/off switch, read through the active profile's secret scope."""
    return get_secret_str(name).strip().lower() in ("true", "1", "yes")


def _check_matrix_create_room() -> bool:
    return _flag("MATRIX_TOOLS_ALLOW_ROOM_CREATE")


class MatrixRoomRequestError(Exception):
    """The Client-Server API request could not be made or did not complete."""


async def _matrix_post(homeserver, token, path, body):
    """POST *body* to ``{homeserver}/_matrix/client/v3{path}`` on a fresh aiohttp
    session (agent loop, see the module docstring). Returns (status, text);
    transport failures raise MatrixRoomRequestError."""
    try:
        import aiohttp
    except ImportError as exc:
        raise MatrixRoomRequestError("aiohttp not installed. Run: pip install aiohttp") from exc
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(f"{homeserver}/_matrix/client/v3{path}", headers=headers, json=body) as resp:
                return resp.status, await resp.text()
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        raise MatrixRoomRequestError(str(exc) or type(exc).__name__) from exc


def _live_matrix_adapter():
    """The running gateway's MatrixAdapter, or None (CLI, gateway not up)."""
    try:
        from gateway.run import _gateway_runner_ref
    except ImportError:  # gateway extras not installed: no live adapter to consult
        return None
    from gateway.config import Platform

    runner = _gateway_runner_ref()
    return runner.adapters.get(Platform.MATRIX) if runner is not None else None


def _matrix_creds():
    """Return (homeserver, token), preferring the live adapter's connected
    values, falling back to env. Keeps us in lock-step with whatever the
    running gateway actually authenticated with."""
    homeserver = ""
    token = ""
    adapter = _live_matrix_adapter()
    if adapter is not None:
        homeserver = getattr(adapter, "_homeserver", "") or ""
        token = getattr(adapter, "_access_token", "") or ""
    homeserver = (homeserver or get_secret_str("MATRIX_HOMESERVER")).rstrip("/")
    token = token or get_secret_str("MATRIX_ACCESS_TOKEN")
    return homeserver, token


async def _handle_matrix_create_room(args, **kwargs):
    homeserver, token = _matrix_creds()
    if not homeserver or not token:
        return tool_error(
            "Matrix not configured (MATRIX_HOMESERVER + MATRIX_ACCESS_TOKEN required)."
        )

    preset = args.get("preset", "private_chat") or "private_chat"
    if preset == "public_chat" and not _flag("MATRIX_ALLOW_PUBLIC_ROOMS"):
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
        status, text = await _matrix_post(homeserver, token, "/createRoom", body)
    except MatrixRoomRequestError as exc:
        return tool_error(f"matrix_create_room request failed: {exc}")
    if status not in {200, 201}:
        return tool_error(f"Matrix createRoom error ({status}): {text[:300]}")
    try:
        data = json.loads(text)
    except ValueError:
        return tool_error(f"createRoom returned invalid JSON: {text[:200]}")

    room_id = data.get("room_id") if isinstance(data, dict) else None
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
        "room_id (e.g. '!abc123:example.org') as returned by "
        "matrix_create_room, or omit it to leave the room this conversation is in. "
        "A room other than the current one is only accepted when the operator "
        "allows cross-room room admin. Use matrix_delete_room instead if you also "
        "want the room removed from your room list."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "room_id": {
                "type": "string",
                "description": "Room to leave, e.g. '!abc123:example.org'. Defaults to the current room.",
            },
            "reason": {
                "type": "string",
                "description": "Optional human-readable reason recorded in the leave event.",
            },
        },
        "required": [],
    },
}

MATRIX_DELETE_ROOM_SCHEMA = {
    "name": "matrix_delete_room",
    "description": (
        "Delete a Matrix room from your account: leave it and then forget it, so "
        "it disappears from your room list. For a room you created and are the only "
        "member of, this effectively tears it down. NOTE: Matrix has no true "
        "server-side delete for regular users — any other members keep their own "
        "copy; a full server purge requires a homeserver admin. Defaults to the "
        "room this conversation is in; another room_id is only accepted when the "
        "operator allows cross-room room admin."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "room_id": {
                "type": "string",
                "description": "Room to delete (leave + forget), e.g. '!abc123:example.org'. Defaults to the current room.",
            },
            "reason": {
                "type": "string",
                "description": "Optional reason recorded in the leave event.",
            },
        },
        "required": [],
    },
}


def _check_matrix_room_admin() -> bool:
    """Gate for leave/delete. Reuses the room-create capability flag: if the
    operator allows the agent to create Matrix rooms, it may also leave/forget
    them. One 'room admin' capability, no extra env wiring."""
    return _flag("MATRIX_TOOLS_ALLOW_ROOM_CREATE")


async def _matrix_room_action(homeserver, token, room_id, action, body=None):
    """POST /rooms/{room_id}/{action} (action = leave|forget). Returns (status, text)."""
    return await _matrix_post(homeserver, token, f"/rooms/{quote(room_id, safe='')}/{action}", body or {})


def _current_matrix_room() -> str:
    """The room this turn is bound to, or "" outside a Matrix session.

    Matrix sessions are room/thread-bound: the gateway binds HERMES_SESSION_PLATFORM
    and HERMES_SESSION_CHAT_ID (the room_id) per turn."""
    if get_session_env("HERMES_SESSION_PLATFORM", "").strip().lower() != "matrix":
        return ""
    return get_session_env("HERMES_SESSION_CHAT_ID", "").strip()


def _cross_room_allowed() -> bool:
    """Operator opt-in for acting on a room other than the current one (and for
    leave/delete outside a Matrix session, e.g. from the CLI). Off by default."""
    return _flag("MATRIX_TOOLS_ALLOW_CROSS_ROOM")


def _require_room(args):
    """Shared validation: returns (homeserver, token, room_id) or a tool_error.

    The room defaults to the one this turn is bound to. Any other room, or any
    room outside a Matrix session, needs MATRIX_TOOLS_ALLOW_CROSS_ROOM."""
    homeserver, token = _matrix_creds()
    if not homeserver or not token:
        return None, tool_error(
            "Matrix not configured (MATRIX_HOMESERVER + MATRIX_ACCESS_TOKEN required)."
        )
    current = _current_matrix_room()
    room_id = str(args.get("room_id") or "").strip() or current
    if not room_id:
        return None, tool_error("room_id is required (e.g. '!abc123:example.org').")
    if room_id != current and not _cross_room_allowed():
        where = f"the current room ({current})" if current else "a Matrix conversation's own room"
        return None, tool_error(
            f"Refusing to act on {room_id}: room admin is limited to {where}. "
            "The operator can allow other rooms with MATRIX_TOOLS_ALLOW_CROSS_ROOM=true."
        )
    return (homeserver, token, room_id), None


def _reconcile_adapter_after_leave(room_id: str) -> None:
    """Drop *room_id* from the live MatrixAdapter's membership caches.

    The leave goes through a separate HTTP client, and incremental sync only ever
    adds joined rooms, so without this the adapter would keep treating the room as
    joined (and ``_join_room_by_id`` would skip a later re-join). Plain set and
    dict removals: safe to run from the agent loop."""
    adapter = _live_matrix_adapter()
    if adapter is None:
        return
    joined = getattr(adapter, "_joined_rooms", None)
    if joined is not None:
        joined.discard(room_id)
    dm_rooms = getattr(adapter, "_dm_rooms", None)
    if dm_rooms is not None:
        dm_rooms.pop(room_id, None)


async def _handle_matrix_leave_room(args, **kwargs):
    ctx, err = _require_room(args)
    if err is not None:
        return err
    homeserver, token, room_id = ctx
    body = {"reason": str(args["reason"])} if args.get("reason") else {}
    try:
        status, text = await _matrix_room_action(homeserver, token, room_id, "leave", body)
    except MatrixRoomRequestError as exc:
        return tool_error(f"matrix_leave_room request failed: {exc}")
    if status != 200:
        return tool_error(f"Matrix leave error ({status}): {text[:300]}")
    _reconcile_adapter_after_leave(room_id)
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
    except MatrixRoomRequestError as exc:
        return tool_error(f"matrix_delete_room leave failed: {exc}")
    already_gone = lstatus == 403 and "M_FORBIDDEN" in ltext
    if lstatus != 200 and not already_gone:
        return tool_error(f"Matrix leave (during delete) error ({lstatus}): {ltext[:300]}")
    # Membership is gone from here on, whatever the forget below does.
    _reconcile_adapter_after_leave(room_id)

    # 2) forget — removes the room from this account's room list (requires having left)
    try:
        fstatus, ftext = await _matrix_room_action(homeserver, token, room_id, "forget", {})
    except MatrixRoomRequestError as exc:
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
