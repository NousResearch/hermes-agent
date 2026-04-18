"""Shared helpers for manual group-report delivery flows."""

from __future__ import annotations

import json
from typing import Any, Callable

from tool_result_validation import normalize_tool_failure_result


def resolve_manual_report_delivery_target(
    value,
    *,
    policy: dict[str, Any],
    resolve_delivery_target: Callable[[Any], str | None],
    current_chat_delivery_target: Callable[[], str],
) -> str:
    resolved = resolve_delivery_target(value)
    if resolved is None:
        stored_target = str(policy.get("manual_report_target") or "").strip()
        if stored_target:
            return stored_target
        return current_chat_delivery_target()
    return resolved


def manual_report_delivery_key(target: str) -> str:
    return f"manual:{str(target or '').strip()}"


def deliver_manual_group_report(
    *,
    report: dict[str, Any],
    policy: dict[str, Any],
    reporting: dict[str, Any],
    explicit_delivery_target,
    resolve_delivery_target: Callable[[Any], str | None],
    current_chat_delivery_target: Callable[[], str],
    format_report: Callable[..., str],
    send_message: Callable[[dict[str, Any]], str],
    failure_prefix: str,
    record_delivery: Callable[..., Any],
    record_delivery_kwargs: dict[str, Any],
) -> dict[str, Any]:
    delivery_target = resolve_manual_report_delivery_target(
        explicit_delivery_target,
        policy=policy,
        resolve_delivery_target=resolve_delivery_target,
        current_chat_delivery_target=current_chat_delivery_target,
    )
    message = format_report(
        report,
        group_name=policy.get("group_name"),
    )
    send_result = json.loads(
        send_message(
            {
                "action": "send",
                "target": delivery_target,
                "message": message,
            }
        )
    )
    failed_delivery = normalize_tool_failure_result(
        send_result,
        failure_prefix=failure_prefix,
    )
    delivery_state = record_delivery(
        delivery_key=manual_report_delivery_key(delivery_target),
        target=delivery_target,
        error=str((failed_delivery or {}).get("error") or "").strip() or None,
        **record_delivery_kwargs,
    )
    payload = {
        "report": report,
        "reporting": reporting,
        "report_control": dict(reporting.get("report_control") or {}),
        "delivery": {
            "target": delivery_target,
            "result": send_result,
            "state": delivery_state,
        },
    }
    if failed_delivery:
        return {
            **failed_delivery,
            **payload,
        }
    return {
        "success": True,
        **payload,
    }
