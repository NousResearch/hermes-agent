"""Shared helpers for validating downstream tool result payloads."""

from __future__ import annotations

from typing import Any


def normalize_tool_result_payload(
    result: Any,
    *,
    invalid_result_detail: str = "工具未返回有效结果",
    missing_success_detail: str = "工具未返回成功结果",
) -> dict[str, Any]:
    """Coerce arbitrary tool output into an explicit success/failure payload."""

    if not isinstance(result, dict):
        return {
            "success": False,
            "detail": invalid_result_detail,
        }

    payload: dict[str, Any] = dict(result)
    error = str(payload.get("error") or "").strip()
    if error:
        payload["success"] = False
        return payload

    if payload.get("success") is True:
        return payload

    detail = str(
        payload.get("message")
        or payload.get("detail")
        or payload.get("status")
        or ""
    ).strip()
    if not detail:
        payload["detail"] = missing_success_detail
    payload["success"] = False
    return payload


def tool_result_failure_text(result: Any, *, failure_prefix: str) -> str | None:
    """Return a normalized failure string unless the tool result is explicit success."""

    payload = normalize_tool_result_payload(result)
    error = str(payload.get("error") or "").strip()
    if error:
        return error

    if payload.get("success") is True:
        return None

    detail = str(
        payload.get("message")
        or payload.get("detail")
        or payload.get("status")
        or ""
    ).strip()
    if detail:
        return f"{failure_prefix}：{detail}"
    return f"{failure_prefix}：工具未返回成功结果"


def normalize_tool_failure_result(result: Any, *, failure_prefix: str) -> dict[str, Any] | None:
    """Return a normalized failure payload, or None when the tool result is explicit success."""

    failure = tool_result_failure_text(result, failure_prefix=failure_prefix)
    if failure is None:
        return None

    payload: dict[str, Any] = dict(result) if isinstance(result, dict) else {}
    payload["success"] = False
    payload["error"] = failure
    return payload
