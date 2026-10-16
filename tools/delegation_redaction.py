"""Delegation-owned output projections, using the existing forced text redactor.

Child output is scrubbed before it is shortened or handed to another consumer.
Parent-authored instructions and delivery coordinates retain their exact values.
Live streamed text needs its own buffering policy and does not pass through here.
"""

from __future__ import annotations

from typing import Any

from agent.redact import REDACTION_UNAVAILABLE, redact_for_egress

_PARENT_EVENT_FIELDS = frozenset(
    {
        "type",
        "delegation_id",
        "session_key",
        "origin_ui_session_id",
        "origin_session_id",
        "parent_session_id",
        "goal",
        "goals",
        "context",
        "toolsets",
        "role",
        "model",
        "group",
        "scope_id",
        "user_id",
        "user_name",
        "dispatched_at",
        "completed_at",
    }
)
_RESULT_METADATA = frozenset(
    {"task_index", "model", "group", "_child_role", "status", "exit_reason"}
)


def redact_child_value(value: Any, *, _depth: int = 0) -> Any:
    """Copy JSON-shaped child data; unsupported or excessively deep values are withheld."""
    if _depth > 64:
        return REDACTION_UNAVAILABLE
    if isinstance(value, str):
        return redact_for_egress(value)
    if isinstance(value, dict):
        return {
            redact_for_egress(str(key)): redact_child_value(item, _depth=_depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_child_value(item, _depth=_depth + 1) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return REDACTION_UNAVAILABLE


def redact_child_result(result: dict[str, Any], output_schema=None) -> dict[str, Any]:
    """Project a result or batch, preserving accounting and host-owned model/task identity.

    Revalidation here is only for a summary changed by redaction. General schema
    revalidation after annotations/truncation belongs to the output-schema owner.
    Old durable entries do not retain the schema, so their previous positive verdict
    cannot certify a newly redacted summary.
    """
    cleaned = {}
    for key, value in result.items():
        if key == "results" and isinstance(value, list):
            cleaned[key] = [
                redact_child_result(entry)
                if isinstance(entry, dict)
                else redact_child_value(entry)
                for entry in value
            ]
        else:
            cleaned[key] = (
                value if key in _RESULT_METADATA else redact_child_value(value)
            )
    if "schema_valid" in result and cleaned.get("summary") != result.get("summary"):
        valid, errors = (
            False,
            [
                "The redacted summary has not been validated against the original output_schema."
            ],
        )
        if isinstance(output_schema, dict):
            from tools.delegation_output_schema import validate_output

            try:
                summary = cleaned.get("summary")
                valid, errors = validate_output(
                    summary if isinstance(summary, str) else "", output_schema
                )
            except Exception:  # health: allow BLE001 -- validator faults must not restore raw output or render a secret-bearing exception
                errors = ["Output validation was unavailable after secret redaction."]
        cleaned["schema_valid"] = valid
        cleaned.pop("schema_errors", None)
        cleaned.pop("schema_note", None)
        if not valid:
            cleaned["schema_errors"] = redact_child_value(errors)
            cleaned["schema_note"] = (
                "Secret redaction changed the delivered summary; do not treat it as validated output."
            )
    return cleaned


def redact_completion_event(event: dict[str, Any]) -> dict[str, Any]:
    """Scrub output while leaving parent instructions and routing outside the projection."""
    cleaned = redact_child_result(
        {key: value for key, value in event.items() if key not in _PARENT_EVENT_FIELDS}
    )
    cleaned.update(
        {key: value for key, value in event.items() if key in _PARENT_EVENT_FIELDS}
    )
    return cleaned
