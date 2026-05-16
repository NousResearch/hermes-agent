"""Feishu Message History Tool -- read recent messages from local Hermes OS session store.

Provides ``feishu_message_history`` for reading conversation history.
History is read from Hermes OS local session storage (SQLite via hermes_state),
which works reliably for both P2P DM and group chats — no Feishu API limitation.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# hermes_state integration
# ---------------------------------------------------------------------------

_HERMES_STATE_AVAILABLE = False
_SessionDB = None


def _get_session_db():
    """Lazily import and return SessionDB instance."""
    global _SessionDB, _HERMES_STATE_AVAILABLE
    if _SessionDB is None:
        try:
            import sys
            # Try gateway path first
            agent_path = Path(__file__).resolve().parents[2]
            if str(agent_path) not in sys.path:
                sys.path.insert(0, str(agent_path))
            from hermes_state import SessionDB

            _SessionDB = SessionDB()
            _HERMES_STATE_AVAILABLE = True
        except Exception as e:
            logger.warning("hermes_state unavailable: %s", e)
            _HERMES_STATE_AVAILABLE = False
    return _SessionDB


def _get_hermes_home() -> Path:
    hermes_home = os.getenv("HERMES_HOME", "")
    if hermes_home:
        p = Path(hermes_home)
        if p.exists():
            return p
    return Path.home() / ".hermes"


def _get_sessions_index() -> dict:
    """Load sessions.json index."""
    sessions_index = _get_hermes_home() / "sessions" / "sessions.json"
    if sessions_index.exists():
        try:
            with open(sessions_index) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _find_current_session_id(chat_id: str) -> str | None:
    """Find the current session ID for a chat_id from sessions.json."""
    index = _get_sessions_index()
    session_key = f"agent:main:feishu:dm:{chat_id}"
    entry = index.get(session_key, {})
    return entry.get("session_id") or None


def _get_chat_user_id(chat_id: str) -> str | None:
    """Get the user_id (open_id) for a chat from sessions.json."""
    index = _get_sessions_index()
    session_key = f"agent:main:feishu:dm:{chat_id}"
    entry = index.get(session_key, {})
    origin = entry.get("origin", {})
    return origin.get("user_id") or None


def _read_local_history(chat_id: str, limit: int = 20, start_time: str = "") -> tuple[str, str]:
    """
    Read message history from hermes_state (SQLite).

    Strategy:
    1. Find current session_id from sessions.json
    2. Use user_id to find ALL past sessions for this user (same user_id = same person)
    3. Merge messages from all sessions, newest first

    Returns (content, source).
    """
    db = _get_session_db()
    if db is None:
        return "", "unavailable"

    # Get current session info
    current_session_id = _find_current_session_id(chat_id)
    user_id = _get_chat_user_id(chat_id)

    if not user_id:
        return "", "no_user"

    # Find all sessions for this user (same user_id across all sessions)
    all_sessions = db.list_sessions_rich()
    user_sessions = [s for s in all_sessions if s.get("user_id") == user_id]
    user_sessions.sort(key=lambda s: s.get("started_at", 0))

    if not user_sessions:
        return "", "no_sessions"

    # Parse start_time filter
    start_dt = None
    if start_time:
        try:
            dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            start_dt = dt.astimezone(timezone.utc) if dt.tzinfo is None else dt
        except (ValueError, OSError):
            pass

    # Collect all messages from all user sessions
    all_messages = []
    for session in user_sessions:
        session_id = session["id"]
        messages = db.get_messages(session_id)
        for msg in messages:
            # Apply start_time filter
            if start_dt and msg.get("timestamp"):
                try:
                    import time
                    msg_dt = datetime.fromtimestamp(msg["timestamp"], tz=timezone.utc)
                    if msg_dt < start_dt:
                        continue
                except (ValueError, OSError):
                    pass

            all_messages.append((session.get("started_at", 0), msg))

    if not all_messages:
        return "", "no_messages"

    # Sort by session start time, then by timestamp within session
    all_messages.sort(key=lambda x: (x[0], x[1].get("timestamp", 0)))

    # Take last N messages
    recent = all_messages[-limit:]
    formatted = _format_messages(recent)
    return formatted, "hermes_state"


def _format_messages(messages: list) -> str:
    """Format (timestamp, msg_dict) tuples into a readable conversation log."""
    if not messages:
        return "(No messages found)"

    lines = []
    for _, msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content") or ""

        # Skip injected context blocks
        if content.strip().startswith("<current_user>"):
            continue

        # Truncate very long messages
        if len(content) > 400:
            content = content[:400] + "..."

        if role == "user":
            lines.append(f"[You]: {content}")
        elif role == "assistant":
            lines.append(f"[Bot]: {content}")
        elif role == "system":
            continue  # Skip system messages
        else:
            lines.append(f"[{role}]: {content[:200]}")

    if not lines:
        return "(No messages found)"

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool schema and handler
# ---------------------------------------------------------------------------

FEISHU_MESSAGE_HISTORY_SCHEMA = {
    "name": "feishu_message_history",
    "description": (
        "Read recent messages from the conversation history. "
        "Use this when the user asks to recall earlier messages, "
        "check conversation history, or reference what they said before.\n\n"
        "Reads from local Hermes OS session storage (works for both P2P DM and group chats).\n\n"
        "Args:\n"
        "  chat_id (str, optional): The Feishu chat ID. Defaults to the current session's chat ID.\n"
        "  limit (int, optional): Max messages to return, default 20.\n"
        "  start_time (str, optional): ISO8601 timestamp to start from.\n"
        "Returns a formatted plain-text conversation log."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "chat_id": {
                "type": "string",
                "description": (
                    "The Feishu chat ID. For DM chats: starts with 'oc_'. "
                    "Defaults to current session's chat ID."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of messages to return (default 20).",
                "minimum": 1,
                "maximum": 100,
            },
            "start_time": {
                "type": "string",
                "description": (
                    "ISO8601 timestamp string (e.g. '2026-04-01T00:00:00Z'). "
                    "Only messages after this time will be returned."
                ),
            },
        },
        "required": [],
    },
}


def _check_feishu():
    return True  # Always available


def _handle_feishu_message_history(args: dict, **kw) -> str:
    """Handle feishu_message_history tool calls."""
    chat_id = args.get("chat_id", "").strip()
    if not chat_id:
        chat_id = os.getenv("HERMES_SESSION_CHAT_ID", "").strip()
    if not chat_id:
        return tool_error(
            "chat_id is required. Provide it as an argument, or make sure this tool "
            "is called within an active Feishu session (HERMES_SESSION_CHAT_ID not set)."
        )

    limit = args.get("limit", 20)
    limit = max(1, min(100, int(limit)))

    start_time = args.get("start_time", "").strip()

    content, source = _read_local_history(chat_id, limit=limit, start_time=start_time)

    if source == "unavailable":
        return tool_error(
            "Hermes OS session store is unavailable. "
            "Make sure hermes_state is accessible from the hermes-agent path."
        )

    if source == "no_user":
        return tool_error(
            f"No user information found for chat '{chat_id}'. "
            f"This chat may not have an active Hermes OS session."
        )

    if source == "no_sessions":
        return tool_error(
            f"No session history found for chat '{chat_id}'."
        )

    if source == "no_messages" or not content.strip():
        return tool_result(success=True, content="(No messages found)")

    return tool_result(success=True, content=content)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="feishu_message_history",
    toolset="feishu",
    schema=FEISHU_MESSAGE_HISTORY_SCHEMA,
    handler=_handle_feishu_message_history,
    check_fn=_check_feishu,
    emoji="💬",
)
