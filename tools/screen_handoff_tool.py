"""Agent-facing, non-blocking request for a human to take over Bot Desktop."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tools.registry import registry

logger = logging.getLogger(__name__)


def _public_url() -> str:
    from hermes_cli.dashboard_auth.prefix import resolve_public_url
    return str(resolve_public_url() or "").rstrip("/")


def _check_screen_handoff() -> bool:
    """Expose the tool only when the configured web service and Bot Desktop exist."""
    if not _public_url():
        return False
    try:
        from tools.bot_desktop.runtime import status
        # Starting the screen belongs to the explicit request, never to tool
        # discovery or to opening the private link.  ``installed`` is the
        # service capability; the handler performs the idempotent start.
        return bool(status().installed)
    except Exception:
        return False


def _session_source(session_id: str) -> tuple[str, str, str] | None:
    """Return (origin_json, profile_home, session_key) from the durable session row."""
    if not session_id:
        return None
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    home = Path(get_hermes_home())
    db = SessionDB(home / "state.db", read_only=True)
    try:
        row = db.get_session(str(session_id))
    finally:
        db.close()
    origin = str((row or {}).get("origin_json") or "")
    if not origin:
        return None
    try:
        parsed = json.loads(origin)
    except (TypeError, ValueError):
        return None
    platform = str(parsed.get("platform") or "").lower()
    if platform not in {"telegram", "discord"} or not parsed.get("user_id"):
        return None
    session_key = str((row or {}).get("session_key") or "")
    return origin, str((row or {}).get("profile_home") or home), session_key


def request_screen_access(args: dict[str, Any], *, session_id: str = "", **_kwargs: Any) -> str:
    reason = str((args or {}).get("reason") or "").strip()
    if not reason:
        return json.dumps({"success": False, "error": "reason is required"})
    if not _public_url():
        return json.dumps({"success": False, "error": "screen handoff is not configured"})
    from gateway.screen_handoff import (
        ScreenHandoffStore, has_screen_handoff_notify, notify_screen_handoff,
    )

    source_info = _session_source(session_id)
    if source_info is None:
        return json.dumps({"success": False, "error": "screen handoff requires a Telegram or Discord private identity"})
    origin_json, profile_home, session_key = source_info
    if not session_key or not has_screen_handoff_notify(session_key):
        return json.dumps({"success": False, "error": "screen handoff is unavailable for this turn"})
    try:
        from tools.bot_desktop.runtime import ensure_started_for_tool, status
        ensure_started_for_tool()
        if not status().running:
            return json.dumps({"success": False, "error": "Bot Desktop could not be started"})
    except Exception:
        return json.dumps({"success": False, "error": "Bot Desktop status is unavailable"})
    store = ScreenHandoffStore(profile_home)
    try:
        handoff, created = store.create_or_get(session_id=session_id, source_json=origin_json, reason=reason)
        if created:
            public_url = _public_url()
            delivered = notify_screen_handoff(session_key, {
                **handoff.public(), "invite_token": handoff.invite_token,
                "confirmation_code": handoff.confirmation_code,
                "invite_url": f"{public_url}/screen-handoff/{handoff.invite_token}",
            })
            if not delivered:
                store.revoke(handoff.request_id)
                return json.dumps({"success": False, "error": "private delivery is unavailable"})
        return json.dumps({
            "success": True, "state": handoff.state, "request_id": handoff.request_id,
            "expires_at": handoff.invite_expires_at, "delivery": "private",
            "reused": not created,
        })
    except Exception as exc:
        logger.exception("screen handoff request failed")
        return json.dumps({"success": False, "error": f"screen handoff failed: {type(exc).__name__}"})


registry.register(
    name="request_screen_access",
    toolset="computer_use",
    schema={
        "name": "request_screen_access",
        "description": (
            "Ask the authenticated user to take over the Bot Desktop browser in a private Telegram "
            "or Discord message. This returns immediately; never ask for a password in chat."
        ),
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
    },
    handler=request_screen_access,
    check_fn=_check_screen_handoff,
    requires_env=[],
    description="Request a short-lived private human screen takeover without blocking the agent turn.",
)
