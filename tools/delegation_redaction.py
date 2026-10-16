"""Delegation-owned output projections, using the existing text redactor.

Child output is scrubbed before it is shortened or handed to another consumer.
Parent-authored instructions and delivery coordinates retain their exact values.
Live streamed text needs its own buffering policy and does not pass through here.

Policy (``force``): copies that leave the process or persist (spill files, the
async_delegations rows and their replay/recovery, worker IPC, memory providers,
lifecycle hooks) are always scrubbed (``force=True``, the default). The parent
model's own in-process copy and live progress honour ``security.redact_secrets:
false`` (``force=False``), the same as any other tool result the parent reads.
Registered vault values are scrubbed in both modes.
"""

from __future__ import annotations

import json
from typing import Any

import agent.redact as _redact
from agent.redact import (
    REDACTION_UNAVAILABLE,
    is_secret_field_name,
    redact_for_egress,
    redact_sensitive_text,
)

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


def redact_child_text(text: Any, *, force: bool = True) -> str:
    """Scrub one child-produced string. ``force=False`` honours the user's opt-out."""
    if force or _redact._redact_enabled():
        return redact_for_egress(text)
    try:  # opted out: vault values only (still a hard boundary)
        return redact_sensitive_text(str(text or ""))
    except Exception:  # health: allow BLE001 -- fail closed, never return raw text on redactor faults
        return REDACTION_UNAVAILABLE


def _redact_field(key: str, value: str, force: bool) -> str:
    """Redact a string value with its key as context: once a JSON summary is parsed apart, the
    key-gated ``"api_key": "<opaque>"`` rule no longer sees the key, so credential-named fields
    get that pass on a one-field document. Fails closed for those fields only."""
    cleaned = redact_child_text(value, force=force)
    if cleaned != value or not is_secret_field_name(key):
        return cleaned
    raw = json.dumps({key: value})
    scrubbed = redact_child_text(raw, force=force)
    if scrubbed == raw:
        return cleaned
    try:
        probe = json.loads(scrubbed)
    except ValueError:
        return REDACTION_UNAVAILABLE
    only = next(iter(probe.values()), None) if isinstance(probe, dict) and len(probe) == 1 else None
    return only if isinstance(only, str) else REDACTION_UNAVAILABLE


def redact_child_value(value: Any, *, force: bool = True, _depth: int = 0) -> Any:
    """Copy JSON-shaped child data; unsupported or excessively deep values are withheld."""
    if _depth > 64:
        return REDACTION_UNAVAILABLE
    if isinstance(value, str):
        return redact_child_text(value, force=force)
    if isinstance(value, dict):
        return {
            redact_child_text(str(key), force=force): (
                _redact_field(str(key), item, force)
                if isinstance(item, str)
                else redact_child_value(item, force=force, _depth=_depth + 1)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_child_value(item, force=force, _depth=_depth + 1) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return REDACTION_UNAVAILABLE


def _redact_summary(summary: Any, structured: bool, force: bool) -> Any:
    """Structured (schema) summaries that parse as JSON are redacted per value and
    re-serialized: a text pass can eat JSON delimiters (``KEY=***`` swallowing the
    closing quote and comma) and break an answer that was valid. Unchanged JSON keeps
    its original formatting; anything that does not parse gets the text pass."""
    if structured and isinstance(summary, str):
        try:
            parsed = json.loads(summary)
        except ValueError:
            pass
        else:
            cleaned = redact_child_value(parsed, force=force)
            return summary if cleaned == parsed else json.dumps(cleaned, ensure_ascii=False)
    return redact_child_value(summary, force=force)


def redact_child_result(result: dict[str, Any], output_schema=None, *, force: bool = True) -> dict[str, Any]:
    """Project a result or batch, preserving accounting and host-owned model/task identity.

    Revalidation here is only for a summary changed by redaction. General schema
    revalidation after annotations/truncation belongs to the output-schema owner.
    Old durable entries do not retain the schema, so their previous positive verdict
    cannot certify a newly redacted summary.
    """
    structured = isinstance(output_schema, dict) or "schema_valid" in result
    cleaned = {}
    for key, value in result.items():
        if key == "results" and isinstance(value, list):
            cleaned[key] = [
                redact_child_result(entry, force=force)
                if isinstance(entry, dict)
                else redact_child_value(entry, force=force)
                for entry in value
            ]
        elif key == "summary":
            cleaned[key] = _redact_summary(value, structured, force)
        else:
            cleaned[key] = (
                value if key in _RESULT_METADATA else redact_child_value(value, force=force)
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
            cleaned["schema_errors"] = redact_child_value(errors, force=force)
            cleaned["schema_note"] = (
                "Secret redaction changed the delivered summary; do not treat it as validated output."
            )
    return cleaned


def redact_completion_event(event: dict[str, Any], *, force: bool = True) -> dict[str, Any]:
    """Scrub output while leaving parent instructions and routing outside the projection."""
    cleaned = redact_child_result(
        {key: value for key, value in event.items() if key not in _PARENT_EVENT_FIELDS}, force=force
    )
    cleaned.update(
        {key: value for key, value in event.items() if key in _PARENT_EVENT_FIELDS}
    )
    return cleaned
