"""Read-only client projection for Workstation-capable surfaces.

The Electron controller owns browser/page identity and the canonical journal
owns durable execution history. This module only transports the controller's
typed resource and event snapshots to Python clients; it never persists task
state.
"""

from __future__ import annotations

from typing import Any

RESOURCE_SCHEMA_VERSION = 1
EVENT_SCHEMA_VERSION = 1


def _unavailable(error: str) -> dict[str, Any]:
    return {
        "available": False,
        "schema_version": RESOURCE_SCHEMA_VERSION,
        "runtime": None,
        "generated_at": None,
        "resources": [],
        "error": error[:500],
    }


def _unavailable_events(error: str) -> dict[str, Any]:
    return {
        "available": False,
        "schema_version": EVENT_SCHEMA_VERSION,
        "runtime": None,
        "generated_at": None,
        "task_id": None,
        "events": [],
        "error": error[:500],
    }


def get_workstation_resources() -> dict[str, Any]:
    """Return a bounded, safe-to-render resource snapshot.

    Controller loss is an explicit degraded state for clients, not a reason
    to invent an in-process fallback store or fail the whole Dashboard.
    """
    try:
        from tools.browser_workstation import workstation_controller_resources

        response = workstation_controller_resources()
    except Exception as exc:
        return _unavailable(str(exc))

    if not isinstance(response, dict):
        return _unavailable("Workstation controller returned an invalid resource envelope")

    schema_version = response.get("schema_version")
    resources = response.get("resources")
    runtime = response.get("runtime")
    generated_at = response.get("generated_at")
    if schema_version != RESOURCE_SCHEMA_VERSION or runtime != "electron-chromium" or not isinstance(resources, list):
        return _unavailable("Workstation resource protocol mismatch")

    normalized: list[dict[str, Any]] = []
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        if not isinstance(resource.get("resource_type"), str) or not isinstance(resource.get("resource_id"), str):
            continue
        if not isinstance(resource.get("state"), dict):
            continue
        normalized.append(
            {
                "resource_type": resource["resource_type"],
                "resource_id": resource["resource_id"],
                "task_id": resource.get("task_id"),
                "session_id": resource.get("session_id"),
                "permissions": [str(item) for item in resource.get("permissions", []) if isinstance(item, str)],
                "state": resource["state"],
                "updated_at": str(resource.get("updated_at") or generated_at or ""),
            }
        )

    return {
        "available": True,
        "schema_version": RESOURCE_SCHEMA_VERSION,
        "runtime": runtime,
        "generated_at": generated_at,
        "resources": normalized,
        "error": None,
    }


def get_workstation_events(
    *,
    task_id: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    """Return bounded journal events from the canonical browser controller.

    This is a transport adapter only. The Electron runtime and Hermes
    ExecutionJournal remain authoritative; Dashboard and TUI callers receive
    the same normalized read-only projection.
    """

    try:
        bounded_limit = min(200, max(1, int(limit)))
    except (TypeError, ValueError):
        return _unavailable_events("Workstation event limit must be an integer")

    if task_id is not None and not isinstance(task_id, str):
        return _unavailable_events("Workstation event task_id must be a string")

    clean_task_id = (task_id.strip() or None) if task_id is not None else None
    try:
        from tools.browser_workstation import workstation_controller_events

        response = workstation_controller_events(task_id=clean_task_id, limit=bounded_limit)
    except Exception as exc:
        return _unavailable_events(str(exc))

    if not isinstance(response, dict):
        return _unavailable_events("Workstation controller returned an invalid event envelope")

    schema_version = response.get("schema_version")
    events = response.get("events")
    runtime = response.get("runtime")
    generated_at = response.get("generated_at")
    response_task_id = response.get("task_id")
    if (
        schema_version != EVENT_SCHEMA_VERSION
        or runtime != "electron-chromium"
        or not isinstance(events, list)
        or (response_task_id is not None and not isinstance(response_task_id, str))
        or (clean_task_id is not None and response_task_id != clean_task_id)
    ):
        return _unavailable_events("Workstation event protocol mismatch")

    normalized: list[dict[str, Any]] = []
    for event in events[:bounded_limit]:
        if not isinstance(event, dict):
            continue
        required = ("event_id", "kind", "task_id", "session_id", "message", "timestamp")
        if not all(isinstance(event.get(field), str) and event[field].strip() for field in required):
            continue
        item = {field: event[field] for field in required}
        for field in ("elapsed_seconds", "url", "browser_tab_id", "risk"):
            if field in event:
                item[field] = event[field]
        if isinstance(event.get("metadata"), dict):
            item["metadata"] = dict(event["metadata"])
        if isinstance(event.get("evidence"), list):
            item["evidence"] = [evidence_item for evidence_item in event["evidence"] if isinstance(evidence_item, dict)]
        normalized.append(item)

    return {
        "available": True,
        "schema_version": EVENT_SCHEMA_VERSION,
        "runtime": runtime,
        "generated_at": generated_at,
        "task_id": response_task_id,
        "events": normalized,
        "error": None,
    }
