"""Projection for relayed ``subagent.*`` progress events.

TUI gateway server owns transport and child-session mirroring. This module owns
wire payload shaping so delegation outcome/evidence fields do not regrow server
monolith whenever logical-result contract expands.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any


def build_subagent_progress_payload(
    event_type: str,
    name: str | None,
    preview: str | None,
    values: Mapping[str, Any],
) -> dict[str, object]:
    """Build one bounded, JSON-safe ``subagent.*`` event payload."""

    payload: dict[str, object] = {
        "goal": str(values.get("goal") or ""),
        "task_count": int(values.get("task_count") or 1),
        "task_index": int(values.get("task_index") or 0),
    }

    for key in ("subagent_id", "parent_id", "child_session_id"):
        if values.get(key):
            payload[key] = str(values[key])
    if values.get("depth") is not None:
        payload["depth"] = int(values["depth"])
    if values.get("model"):
        payload["model"] = str(values["model"])
    if values.get("tool_count") is not None:
        payload["tool_count"] = int(values["tool_count"])
    if values.get("toolsets"):
        payload["toolsets"] = [str(toolset) for toolset in values["toolsets"]]

    for key in (
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "api_calls",
    ):
        value = values.get(key)
        if value is not None:
            try:
                payload[key] = int(value)
            except (TypeError, ValueError):
                pass

    if values.get("files_read"):
        payload["files_read"] = [str(path) for path in values["files_read"]]
    if values.get("files_written"):
        payload["files_written"] = [str(path) for path in values["files_written"]]
    if values.get("output_tail"):
        payload["output_tail"] = list(values["output_tail"])
    if name:
        payload["tool_name"] = str(name)
    if preview:
        payload["text"] = str(preview)
    if values.get("status"):
        payload["status"] = str(values["status"])
    if values.get("outcome"):
        payload["outcome"] = str(values["outcome"])
    if values.get("exit_reason"):
        payload["exit_reason"] = str(values["exit_reason"])
    if values.get("interrupted") is not None:
        payload["interrupted"] = bool(values["interrupted"])
    if values.get("tool_error_count") is not None:
        try:
            payload["tool_error_count"] = int(values["tool_error_count"])
        except (TypeError, ValueError):
            pass
    if values.get("summary"):
        payload["summary"] = str(values["summary"])
    if values.get("duration_seconds") is not None:
        payload["duration_seconds"] = float(values["duration_seconds"])

    # Parent-side output_schema validation is authoritative evidence. Keep
    # lifecycle status untouched while carrying separate logical verdict.
    if "schema_valid" in values:
        payload["schema_valid"] = bool(values["schema_valid"])
    if values.get("schema_retries") is not None:
        try:
            payload["schema_retries"] = int(values["schema_retries"])
        except (TypeError, ValueError):
            pass
    schema_errors = values.get("schema_errors")
    if isinstance(schema_errors, list) and schema_errors:
        payload["schema_errors"] = [str(error) for error in schema_errors]
    if values.get("error"):
        payload["error"] = str(values["error"])
    if values.get("error_authoritative") is not None:
        payload["error_authoritative"] = bool(values["error_authoritative"])

    if preview and event_type == "subagent.tool":
        payload["tool_preview"] = str(preview)
        payload["text"] = str(preview)

    return payload


def install_server_overlay(server: Any) -> None:
    """Install bounded subagent projection over server's legacy callback.

    ``tui_gateway.server`` remains the transport/lifecycle owner. Its original
    callback still handles every non-subagent event; this overlay intercepts
    only ``subagent.*`` so logical outcome evidence can evolve in this bounded
    module without modifying the oversized compatibility file.
    """

    marker = "_subagent_progress_overlay_installed"
    if getattr(server, marker, False):
        return
    original = getattr(server, "_on_tool_progress", None)
    if not callable(original):
        return

    @wraps(original)
    def on_tool_progress(
        sid: str,
        event_type: str,
        name: str | None = None,
        preview: str | None = None,
        args: dict | None = None,
        **values: Any,
    ) -> Any:
        if not event_type.startswith("subagent."):
            return original(sid, event_type, name, preview, args, **values)
        if not server._tool_progress_enabled(sid):
            return None

        payload = build_subagent_progress_payload(event_type, name, preview, values)
        if event_type != "subagent.text":
            server._emit(event_type, sid, payload)
        server._mirror_subagent_to_child(event_type, payload)
        return None

    server._on_tool_progress = on_tool_progress
    setattr(server, marker, True)
