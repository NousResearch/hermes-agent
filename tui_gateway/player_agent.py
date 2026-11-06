"""Session-bound host routing for public desktop player capabilities."""

from __future__ import annotations

import json
import logging

from .method_ctx import bind_module

logger = logging.getLogger(__name__)


def request_player_agent_for_session(session_id: str, operation: dict) -> str:
    """Forward one closed player operation through the live turn's renderer request callback."""
    from gateway.session_context import get_session_env
    from pydantic import TypeAdapter, ValidationError
    from tui_gateway.contracts.server_requests import PlayerAgentOperation

    durable_sid = str(session_id or "").strip()
    if not durable_sid:
        return json.dumps({"error": "session-unavailable"})
    try:
        validated = TypeAdapter(PlayerAgentOperation).validate_python(operation)
    except ValidationError:
        return json.dumps({"error": "invalid-operation"})

    # The agent's durable conversation id differs from the live UI runtime id.
    # Bind both identities to the active transport and exact record; never scan
    # sessions or let plugin/model input choose a renderer route.
    ui_sid = get_session_env("HERMES_UI_SESSION_ID", "").strip()
    transport, session = _current_session_steer_authority(ui_sid)
    with _sessions_lock:
        if (transport is None or session is None or session is not _current_runtime_session_record.get()
                or _sessions.get(ui_sid) is not session or session.get("_finalized", False)):
            return json.dumps({"error": "desktop-player-unavailable"})
        agent = session.get("agent")
        if agent is None or str(getattr(agent, "session_id", "") or "") != durable_sid:
            return json.dumps({"error": "desktop-player-unavailable"})
        callback = getattr(agent, "player_agent_callback", None)
    if not callable(callback):
        return json.dumps({"error": "desktop-player-unavailable"})
    try:
        return callback(validated.model_dump(mode="json", exclude_none=True))
    except Exception:
        logger.info("Desktop player request unavailable", exc_info=True)
        return json.dumps({"error": "desktop-player-unavailable"})


def register(server):
    bind_module(globals(), server)
