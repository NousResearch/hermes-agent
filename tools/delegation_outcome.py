"""Bounded owner for delegation lifecycle/outcome policy and evidence shaping.

Oversized compatibility surfaces (``delegate_tool`` and ``process_registry``)
call these pure helpers instead of owning authority-bearing outcome behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence


_FATAL_DELEGATION_EXIT_REASONS = frozenset(
    {
        "all_retries_exhausted_no_response",
        "code_skew_detected",
        "empty_response_exhausted",
        "ollama_runtime_context_too_small",
    }
)

_FATAL_DELEGATION_EXIT_PREFIXES = (
    "code_skew_attribute_error(",
    "error_near_max_iterations(",
    "local_processing_error(",
)

_PARTIAL_DELEGATION_EXIT_REASONS = frozenset(
    {
        "budget_exhausted",
        "max_iterations",
        "partial_stream_recovery",
    }
)

_PARTIAL_DELEGATION_EXIT_PREFIXES = ("max_iterations_reached(",)

OUTCOME_ICONS = {
    "unverified": "⚠",
    "partial": "◐",
    "failed": "✗",
    "unknown": "⚠",
}

DELEGATION_OUTCOME_TOOL_GUIDANCE = (
    "- 'status' tracks lifecycle; completed means only child loop ended. "
    "'outcome': partial, unverified, unknown, or failed. Verify evidence "
    "before completion claims.\n"
)


@dataclass(frozen=True)
class DelegationResultClassification:
    """Authoritative two-axis classification of one terminal child attempt."""

    completed: bool
    interrupted: bool
    runtime_error: Any
    turn_exit_reason: str
    runtime_failed: bool
    runtime_partial: bool
    schema_failed: bool
    usable_summary: bool
    status: str
    outcome: str
    exit_reason: str


def is_fatal_delegation_exit_reason(value: object) -> bool:
    """Return whether *value* proves a fatal child-runtime exit."""
    reason = str(value or "").strip()
    return reason in _FATAL_DELEGATION_EXIT_REASONS or reason.startswith(
        _FATAL_DELEGATION_EXIT_PREFIXES
    )


def is_partial_delegation_exit_reason(value: object) -> bool:
    """Return whether *value* proves incomplete but potentially usable work."""
    reason = str(value or "").strip()
    return reason in _PARTIAL_DELEGATION_EXIT_REASONS or reason.startswith(
        _PARTIAL_DELEGATION_EXIT_PREFIXES
    )


def delegation_schema_retry_allowed(result: Mapping[str, Any]) -> bool:
    """Allow schema repair only after a normally completed first attempt."""
    exit_reason = str(
        result.get("turn_exit_reason") or result.get("exit_reason") or ""
    ).strip()
    outcome = str(result.get("outcome") or "").strip().lower()
    return (
        bool(result.get("completed", False))
        and not bool(result.get("failed", False))
        and not bool(result.get("interrupted", False))
        and not bool(result.get("partial", False))
        and not bool(result.get("error"))
        and not is_fatal_delegation_exit_reason(exit_reason)
        and not is_partial_delegation_exit_reason(exit_reason)
        and outcome not in {"failed", "partial", "unknown"}
    )


def classify_delegation_result(
    terminal_result: Mapping[str, Any],
    *,
    aggregate_result: Mapping[str, Any],
    summary: object,
    schema_requested: bool,
    schema_valid: Optional[bool],
) -> DelegationResultClassification:
    """Classify terminal runtime evidence without trusting child summary prose."""
    summary_text = str(summary or "")
    completed = bool(terminal_result.get("completed", False))
    failed = bool(terminal_result.get("failed", False))
    partial = bool(terminal_result.get("partial", False))
    runtime_error = terminal_result.get("error")
    interrupted = bool(terminal_result.get("interrupted", False))
    turn_exit_reason = str(
        terminal_result.get("turn_exit_reason")
        or terminal_result.get("exit_reason")
        or ""
    ).strip()
    explicit_outcome = str(terminal_result.get("outcome") or "").strip().lower()
    runtime_unknown = explicit_outcome == "unknown" or turn_exit_reason == "unknown"
    runtime_failed = (
        failed
        or bool(runtime_error)
        or is_fatal_delegation_exit_reason(turn_exit_reason)
    )
    runtime_partial = partial or is_partial_delegation_exit_reason(turn_exit_reason)
    schema_failed = schema_requested and schema_valid is False
    empty_sentinel = summary_text.strip() == "(empty)"
    usable_summary = bool(summary_text.strip()) and not empty_sentinel

    if runtime_failed:
        status = "failed"
    elif interrupted:
        status = "interrupted"
    elif summary_text and not empty_sentinel:
        status = "completed"
    else:
        status = "failed"

    if runtime_failed or schema_failed:
        outcome = "failed"
    elif runtime_unknown:
        outcome = "unknown"
    elif interrupted:
        outcome = "partial" if usable_summary else "failed"
    elif usable_summary:
        outcome = "unverified" if completed and not runtime_partial else "partial"
    else:
        outcome = "failed"

    if turn_exit_reason:
        exit_reason = turn_exit_reason
    elif runtime_failed:
        exit_reason = str(terminal_result.get("exit_reason") or "error")
    elif interrupted:
        exit_reason = "interrupted"
    elif aggregate_result.get("failed") or aggregate_result.get("error"):
        exit_reason = "error"
    elif completed:
        exit_reason = "completed"
    else:
        exit_reason = "max_iterations"

    return DelegationResultClassification(
        completed=completed,
        interrupted=interrupted,
        runtime_error=runtime_error,
        turn_exit_reason=turn_exit_reason,
        runtime_failed=runtime_failed,
        runtime_partial=runtime_partial,
        schema_failed=schema_failed,
        usable_summary=usable_summary,
        status=status,
        outcome=outcome,
        exit_reason=exit_reason,
    )


def terminal_tool_error_count(
    result: Mapping[str, Any],
    *,
    stringify_tool_content: Callable[[Any], str],
    looks_like_error_output: Callable[[str], bool],
) -> int:
    """Return terminal-attempt tool failures, preferring explicit telemetry."""
    explicit = result.get("tool_error_count")
    if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit >= 0:
        return explicit
    messages = result.get("messages") or []
    if not isinstance(messages, list):
        return 0
    return sum(
        1
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "tool"
        and looks_like_error_output(stringify_tool_content(message.get("content", "")))
    )


def apply_schema_evidence(
    target: MutableMapping[str, Any],
    *,
    classification: DelegationResultClassification,
    schema_requested: bool,
    schema_valid: Optional[bool],
    schema_retries: int,
    schema_errors: Sequence[str],
) -> None:
    """Attach schema verdict and authoritative failure evidence in one place."""
    if not schema_requested:
        return
    target["schema_valid"] = bool(schema_valid)
    if schema_retries:
        target["schema_retries"] = schema_retries
    if schema_valid is False and schema_errors:
        target["schema_errors"] = list(schema_errors)
    if not classification.schema_failed:
        return
    if not classification.runtime_failed:
        target["error"] = (
            "Final answer does not satisfy the declared output_schema "
            "after the bounded retry."
            if schema_retries
            else "Final answer does not satisfy the declared output_schema."
        )
    target["error_authoritative"] = True


def schema_evidence_payload(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Copy schema evidence from a result entry into progress/public envelopes."""
    keys = (
        "schema_valid",
        "schema_retries",
        "schema_errors",
        "error_authoritative",
    )
    payload = {key: entry[key] for key in keys if key in entry}
    if entry.get("error_authoritative") is True and "error" in entry:
        payload["error"] = entry["error"]
    return payload


def delegation_stop_evidence(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Build logical-outcome fields for the public ``subagent_stop`` hook."""
    return {
        "child_outcome": entry.get("outcome"),
        "child_schema_valid": entry.get("schema_valid"),
        "child_schema_errors": entry.get("schema_errors"),
        "child_schema_retries": entry.get("schema_retries"),
        "child_error_authoritative": entry.get("error_authoritative", False),
        "child_exit_reason": entry.get("exit_reason"),
        "child_interrupted": entry.get("interrupted", False),
        "child_tool_error_count": entry.get("tool_error_count", 0),
        "child_terminal_tool_error_count": entry.get("terminal_tool_error_count"),
    }


def failed_delegation_evidence(
    *,
    status: str = "failed",
    exit_reason: str = "error",
    interrupted: bool = False,
) -> dict[str, Any]:
    """Return complete fail-closed evidence for synthetic failure envelopes."""
    return {
        "status": status,
        "outcome": "failed",
        "exit_reason": exit_reason,
        "interrupted": interrupted,
        "tool_error_count": 0,
    }


def delegation_batch_icon(entry: Mapping[str, Any]) -> str:
    """Render batch progress without false-green completion signals."""
    outcome = entry.get("outcome")
    status = entry.get("status")
    if outcome in OUTCOME_ICONS:
        return OUTCOME_ICONS[str(outcome)]
    if status != "completed":
        return "✗"
    return "⚠" if entry.get("summary") else "✗"


def derive_result_outcome(result: Mapping[str, Any]) -> str:
    """Return logical outcome for durable/current results, failing closed."""
    status = str(result.get("status") or "").strip().lower()
    exit_reason = result.get("exit_reason") or result.get("turn_exit_reason")
    if (
        (bool(result.get("error")) and result.get("error_authoritative") is not False)
        or status in ("error", "timeout", "failed")
        or is_fatal_delegation_exit_reason(exit_reason)
    ):
        return "failed"
    if result.get("schema_valid") is False:
        return "failed"
    summary = result.get("summary")
    has_summary = (
        isinstance(summary, str)
        and bool(summary.strip())
        and summary.strip() != "(empty)"
    )
    outcome = result.get("outcome")
    if outcome == "failed":
        return "failed"
    if outcome == "unknown" or status == "unknown":
        return "unknown"
    if status == "interrupted" or is_partial_delegation_exit_reason(exit_reason):
        return "partial" if has_summary else "failed"
    if isinstance(outcome, str) and outcome in OUTCOME_ICONS:
        return outcome
    return "unverified" if has_summary else "failed"


def delegation_evidence_fields(result: Mapping[str, Any]) -> list[str]:
    """Return concise runtime evidence fields for parent verification."""
    fields: list[str] = []
    if result.get("exit_reason") is not None:
        exit_reason = str(result["exit_reason"]).replace("\n", " ")[:80]
        fields.append(f"exit_reason={exit_reason}")
    if "interrupted" in result and result.get("interrupted") is not None:
        fields.append(f"interrupted={str(bool(result['interrupted'])).lower()}")
    if result.get("tool_error_count") is not None:
        fields.append(f"tool_errors={result['tool_error_count']}")
    if result.get("schema_valid") is False:
        fields.append("schema_valid=false")
        schema_errors = result.get("schema_errors")
        if isinstance(schema_errors, list) and schema_errors:
            first_error = str(schema_errors[0]).replace("\n", " ")[:120]
            fields.append(f"schema_error={first_error}")
    return fields
