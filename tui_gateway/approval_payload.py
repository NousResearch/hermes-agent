"""Small, side-effect-light payload builder for approval requests."""

from __future__ import annotations


def approval_request_payload(data: dict | None) -> dict:
    """Build the client-safe representation of a pending approval."""
    payload = dict(data or {})
    if "choices" not in payload:
        choices = ["once"]
        if not payload.get("smart_denied") and payload.get("allow_session") is not False:
            choices.append("session")
            if payload.get("allow_permanent") is not False:
                choices.append("always")
        payload["choices"] = choices + ["deny"]
    if "command" in payload:
        from agent.redact import redact_sensitive_text
        payload["command"] = redact_sensitive_text(
            str(payload.get("command") or ""), force=True
        )
    return payload
