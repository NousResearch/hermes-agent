"""set_theme's desktop bridge: ``theme.apply`` for values (the renderer answers with what it applied)
or ``theme.pick`` for a card that waits for the user."""

from __future__ import annotations

import json
import time

# Applying a theme is a store write the renderer answers in ms; a picker waits for a person.
THEME_TIMEOUT_S = 15
THEME_PICK_TIMEOUT_S = 300

_UNANSWERED = {"success": False, "error": "No Hermes Desktop window answered the theme request."}


def theme_request(sid: str, payload: dict) -> dict:
    """The renderer's answer; a dismissed picker and an expired one are told apart."""
    from tui_gateway import server_requests
    from tui_gateway.server import _ask

    if "picker" not in payload:
        answer = _ask("theme.apply", sid, dict(payload), timeout=THEME_TIMEOUT_S)
        return json.loads(answer) if answer else dict(_UNANSWERED)
    started = time.monotonic()
    result = server_requests.send("theme.pick", sid, dict(payload), timeout=THEME_PICK_TIMEOUT_S)
    if result is None:
        return {"timed_out": True} if time.monotonic() - started >= THEME_PICK_TIMEOUT_S else {"cancelled": True}
    if isinstance(result.get("picked"), dict):
        return {"success": True, "picked": result["picked"]}
    if isinstance(result.get("said"), str) and result["said"].strip():
        return {"said": result["said"].strip()}
    return {"cancelled": True}
