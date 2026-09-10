"""Persist command output for the UI without adding anything to model history."""

import logging
from uuid import UUID

from tools.ansi_strip import strip_ansi

logger = logging.getLogger(__name__)
PERSISTENCE_ERROR = (
    "Command finished, but its report could not be saved to chat history. "
    "The output below is still available in this window. Do not rerun the command to save it."
)


def persist_command_output(session: dict, params: dict, command: str, payload: dict) -> dict:
    """An opt-in display write must never turn a completed command into an RPC retry."""
    event_id = params.get("display_event_id")
    output = payload.get("output")
    if not event_id or not isinstance(output, str) or not output.strip():
        return payload
    from tui_gateway import server

    try:
        event_id = str(UUID(event_id))
        if not session or not session.get("session_key"):
            raise ValueError("missing_session")
        if server._ensure_session_db_row(session) is False:
            raise RuntimeError("session_store_unavailable")
        body = f"warning: {payload['warning']}\n{output}" if payload.get("warning") else output
        with server._session_db(session) as db:
            if db is None:
                raise RuntimeError("session_store_unavailable")
            event = db.append_display_event(
                session["session_key"], event_id, command.lstrip("/").split()[0], strip_ansi(body).strip())
            if session.get("pending_title"):
                server._title_read(session, db, session["session_key"])
        return {**payload, "display_event": event}
    except Exception as exc:
        # Exception strings may include command data. Keep diagnostics structural.
        logger.warning("Command display persistence failed (%s)", type(exc).__name__)
        return {**payload, "persistence_error": PERSISTENCE_ERROR}


def with_display_events(messages: list, db, session_key: str) -> list:
    """Augment an outgoing UI projection only; never mutate the live/resumed history."""
    reader = getattr(db, "get_display_events", None)
    if not callable(reader):
        return messages
    events = reader(session_key, include_ancestors=True)
    if not events:
        return messages
    projected = [
        {"role": "system", "text": event["content"], "row_id": event["id"],
         "timestamp": event["timestamp"], "display_kind": "command_result"}
        for event in events
    ]
    # Compaction can carry an older tail after a newer summary. Preserve that
    # transcript order when interleaving events, without changing source times.
    ordered = []
    effective_time = float("-inf")
    for index, message in enumerate(messages):
        effective_time = max(effective_time, message.get("timestamp") or 0)
        ordered.append(((effective_time, 0, index), message))
    ordered.extend(((event["timestamp"], 1, index), event) for index, event in enumerate(projected))
    return [row for _, row in sorted(ordered, key=lambda item: item[0])]
