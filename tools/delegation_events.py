"""Presentation-neutral identity for terminal delegation events, never routing authority."""
from typing import Any


def internal_event_envelope(delegation_id: Any) -> dict[str, Any]:
    """Keep legacy ids on replay; new delegations mint full UUIDs at dispatch."""
    if not isinstance(delegation_id, str) or not delegation_id.strip():
        raise ValueError("delegation_id is required for an internal event envelope")
    return {
        "event_schema": "hermes.internal_event.v1",
        "event_id": f"async_delegation:{delegation_id}:terminal",
        "event_kind": "workflow.async_delegation.terminal",
        "workflow_id": f"delegation:{delegation_id}",
        "display_kind": "internal_event",
        "user_originated": False,
        "terminal": True,
    }


def internal_event_persistence(event: Any, *, trusted_internal: bool = False) -> tuple[str | None, dict[str, Any] | None]:
    """Validate an explicit producer envelope; never infer provenance from result text.

    This identifies a unit's terminal result, not an early failed-child notice from a still-running
    batch. Callers must separately prove ownership/admission before treating any input as internal.
    The compatibility keyword ``trusted_internal`` never relaxes validation.
    """
    if (not isinstance(event, dict) or event.get("type", "async_delegation") != "async_delegation"
            or event.get("task_failure_notice")):
        return None, None
    delegation_id = event.get("delegation_id")
    if not isinstance(delegation_id, str) or not delegation_id.strip():
        return None, None
    expected = internal_event_envelope(delegation_id)
    if any(event.get(key) != value or type(event.get(key)) is not type(value) for key, value in expected.items()):
        return None, None
    return "internal_event", {key: value for key, value in {**expected, "delegation_id": delegation_id}.items()
                              if key != "display_kind"}
