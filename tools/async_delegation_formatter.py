"""Bounded renderer for parent-facing async delegation completion envelopes."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from tools.delegation_outcome import (
    OUTCOME_ICONS,
    delegation_evidence_fields,
    derive_result_outcome,
)


def _default_format_age(seconds: float) -> str:
    try:
        total = max(0, int(seconds))
    except (TypeError, ValueError):
        return "?"
    if total < 60:
        return f"{total}s"
    minutes, _seconds = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes}m" if minutes else f"{hours}h"


def _no_model_notice(_results: Sequence[dict[str, Any]]) -> None:
    return None


def format_async_delegation(
    evt: dict[str, Any],
    *,
    format_age: Callable[[float], str] = _default_format_age,
    model_not_found_notice: Callable[[Sequence[dict[str, Any]]], list[str] | None]
    = _no_model_notice,
) -> str:
    """Render self-contained async result with fail-closed logical outcomes."""
    deleg_id = evt.get("delegation_id", "unknown")
    goal = evt.get("goal", "") or ""
    context = evt.get("context")
    toolsets = evt.get("toolsets")
    role = evt.get("role") or "leaf"
    model = evt.get("model") or "?"
    status = evt.get("status") or "completed"
    summary = evt.get("summary")
    error = evt.get("error")
    api_calls = evt.get("api_calls", 0)
    duration = evt.get("duration_seconds", "?")
    truncated = evt.get("truncated") or evt.get("exit_reason") == "max_iterations"
    dispatched_at = evt.get("dispatched_at")
    completed_at = evt.get("completed_at") or time.time()

    batch_results = evt.get("results")
    if evt.get("is_batch") or isinstance(batch_results, list):
        results = batch_results or []
        goals = evt.get("goals") or []
        n = len(results) if results else len(goals)
        total_duration = evt.get("total_duration_seconds", duration)
        lines = [
            f"[ASYNC DELEGATION BATCH COMPLETE — {deleg_id}]",
            f"A background fan-out of {n} subagent(s) you dispatched earlier "
            "has finished. All ran in parallel and waited on each other; their "
            "consolidated results are below. You may have moved on since "
            "dispatching — act on these or re-dispatch if things have changed.",
            "NOTE: a subagent finishing and producing output is NOT task "
            "acceptance. Each task below is marked unverified/partial/failed — "
            "verify the returned handles/evidence yourself before relying on it.",
            "",
        ]
        if isinstance(dispatched_at, (int, float)):
            timestamp = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(dispatched_at)
            )
            age = f" ({format_age(completed_at - dispatched_at)} ago)"
            lines.append(f"Dispatched: {timestamp}{age}")
        if context:
            lines.append(f"Context you provided: {context}")
        if toolsets:
            lines.append(f"Toolsets: {', '.join(toolsets)}")
        lines.append(
            f"Role: {role}   Model: {model}   Total duration: {total_duration}s"
        )
        if error and not results:
            lines.append("--- ERROR ---")
            lines.append(f"The batch did not complete successfully: {error}")
            return "\n".join(lines)
        notice = model_not_found_notice(results)
        if notice:
            lines.append("")
            lines.extend(notice)
        for result in sorted(results, key=lambda item: item.get("task_index", 0)):
            index = result.get("task_index", 0)
            result_status = result.get("status", "?")
            result_outcome = derive_result_outcome(result)
            result_summary = result.get("summary")
            result_error = result.get("error")
            result_goal = (
                goals[index] if index < len(goals) else result.get("goal", "")
            )
            result_truncated = (
                result.get("truncated")
                or result.get("exit_reason") == "max_iterations"
            )
            icon = OUTCOME_ICONS.get(result_outcome, "⚠")
            lines.append("")
            header = f"--- {icon} TASK {index + 1}/{n}"
            if result_goal:
                header += f": {result_goal}"
            header += f"  (outcome={result_outcome}, status={result_status}"
            if result.get("api_calls"):
                header += f", api_calls={result['api_calls']}"
            if result.get("duration_seconds") is not None:
                header += f", {result['duration_seconds']}s"
            if result_truncated:
                header += ", TRUNCATED: hit max_iterations — work may be incomplete"
            evidence = delegation_evidence_fields(result)
            if evidence:
                header += ", " + ", ".join(evidence)
            lines.append(header + ") ---")
            if result_outcome == "unverified" and result_summary:
                lines.append(
                    "Unverified summary (subagent self-report — verify before "
                    "relying on it):"
                )
                lines.append(result_summary)
            elif result_summary:
                if result_truncated:
                    lines.append(
                        "[TRUNCATED — subagent hit its iteration cap; the "
                        "summary below may be incomplete. Verify before relying "
                        "on it, or re-dispatch the unfinished part.]"
                    )
                if result_error:
                    lines.append(f"({result_outcome}: {result_error})")
                lines.append(
                    f"Partial/unverified output (outcome={result_outcome}):"
                )
                lines.append(result_summary)
            else:
                lines.append(
                    f"(no summary — outcome={result_outcome}, status={result_status}"
                    + (f": {result_error}" if result_error else "")
                    + ")"
                )
            live_transcript = result.get("live_transcript")
            if live_transcript:
                lines.append(
                    "Full live transcript (complete tool/assistant trace): "
                    f"{live_transcript}"
                )
        return "\n".join(lines)

    age = ""
    if isinstance(dispatched_at, (int, float)):
        age = f" ({format_age(completed_at - dispatched_at)} ago)"

    outcome = derive_result_outcome(evt)
    lines = [
        f"[ASYNC DELEGATION COMPLETE — {deleg_id}]",
        "A background subagent you dispatched earlier has finished. You may "
        "have moved on since dispatching it; the full task source is below so "
        "you can act on the result or re-dispatch if things have changed.",
        "NOTE: the subagent finishing and producing output is NOT task "
        "acceptance. Verify the returned handles/evidence yourself before "
        "relying on this result.",
        "",
    ]
    if isinstance(dispatched_at, (int, float)):
        timestamp = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(dispatched_at)
        )
        lines.append(f"Dispatched: {timestamp}{age}")
    lines.append(f"Original goal: {goal}")
    if context:
        lines.append(f"Context you provided: {context}")
    if toolsets:
        lines.append(f"Toolsets: {', '.join(toolsets)}")
    lines.append(f"Role: {role}   Model: {model}")
    notice = model_not_found_notice([evt])
    if notice:
        lines.append("")
        lines.extend(notice)
    truncation_notice = (
        " [TRUNCATED: hit max_iterations — work may be incomplete]"
        if truncated
        else ""
    )
    lines.append(
        f"Outcome: {outcome}   Status: {status}   API calls: {api_calls}   "
        f"Duration: {duration}s{truncation_notice}"
    )
    evidence = delegation_evidence_fields(evt)
    if evidence:
        lines.append("Runtime evidence: " + ", ".join(evidence))
    lines.append("--- RESULT ---")
    if outcome == "unverified" and summary:
        lines.append(
            "Unverified summary (subagent self-report — verify before relying on it):"
        )
        lines.append(summary)
    elif outcome == "partial" and summary:
        if truncated:
            lines.append(
                "[TRUNCATED — subagent hit its iteration cap; the summary below "
                "may be incomplete. Verify before relying on it, or re-dispatch "
                "the unfinished part.]"
            )
        lines.append(
            "The subagent produced partial output (ran out of iterations or "
            "was interrupted)."
            + (f" {error}" if error else "")
        )
        lines.append(f"Partial/unverified output (outcome={outcome}):")
        lines.append(summary)
    else:
        lines.append(
            f"The subagent did not complete successfully (outcome={outcome}, "
            f"status={status})."
            + (f"\n{error}" if error else "")
        )
        if summary:
            lines.append(f"Partial/unverified output (outcome={outcome}):")
            lines.append(summary)
    return "\n".join(lines)
