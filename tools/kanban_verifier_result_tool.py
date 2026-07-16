"""Verify-only Kanban result emitter tool.

This tool is only exposed inside the protected-merge verifier subprocess. It
does not open the Kanban database and does not accept a task/run target from
the model; those fields are bound by the supervisor through environment.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any

from tools.registry import registry, tool_error


_CONTRACT_ID = "protected-merge:v1"
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_TEXT_LIMIT = 4096
_EMIT_LOCK = threading.Lock()
_EMITTED = False


def _check_verify_only() -> bool:
    return os.environ.get("HERMES_KANBAN_VERIFY_ONLY") == "1"


def _load_contract() -> dict[str, Any]:
    raw = os.environ.get("HERMES_KANBAN_VERIFY_CONTRACT")
    if not raw:
        raise RuntimeError("missing verifier contract")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("verifier contract must be an object")
    required = {
        "contract_id",
        "contract_hash",
        "task_id",
        "run_id",
        "claim_lock",
        "pr_url",
        "approved_head",
        "approved_base",
        "nonce",
    }
    unknown = sorted(set(payload) - required)
    if unknown:
        raise RuntimeError(f"unsupported verifier contract field: {unknown[0]}")
    missing = sorted(field for field in required if field not in payload)
    if missing:
        raise RuntimeError(f"missing verifier contract field: {missing[0]}")
    if payload["contract_id"] != _CONTRACT_ID:
        raise RuntimeError("unexpected verifier contract id")
    if not isinstance(payload["run_id"], int) or payload["run_id"] <= 0:
        raise RuntimeError("invalid verifier run id")
    for field in required - {"run_id"}:
        if not isinstance(payload[field], str) or not payload[field]:
            raise RuntimeError(f"invalid verifier contract field: {field}")
    if not _SHA256_HEX_RE.fullmatch(payload["contract_hash"]):
        raise RuntimeError("invalid verifier contract hash")
    expected_hash = os.environ.get("HERMES_KANBAN_VERIFY_CONTRACT_HASH")
    if expected_hash != payload["contract_hash"]:
        raise RuntimeError("verifier contract hash mismatch")
    return payload


def _result_fd() -> int:
    raw_fd = os.environ.get("HERMES_KANBAN_VERIFY_RESULT_FD")
    if raw_fd:
        fd = int(raw_fd)
        if fd <= 2:
            raise RuntimeError("invalid verifier result fd")
        return fd
    raw_handle = os.environ.get("HERMES_KANBAN_VERIFY_RESULT_HANDLE")
    if raw_handle:
        if os.name != "nt":
            raise RuntimeError("verifier result handle is Windows-only")
        import msvcrt

        handle = int(raw_handle)
        if handle <= 0:
            raise RuntimeError("invalid verifier result handle")
        return int(msvcrt.open_osfhandle(handle, os.O_WRONLY))
    raise RuntimeError("missing verifier result endpoint")


def _bounded_text(value: Any, *, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise RuntimeError(f"{field} must be a string")
    text = value.strip()
    if len(text.encode("utf-8")) > _TEXT_LIMIT:
        raise RuntimeError(f"{field} too large")
    return text


def _handle_verifier_result(
    verdict: Any,
    summary: str = "",
    reason: str = "",
    **_: Any,
) -> str:
    global _EMITTED
    if not _check_verify_only():
        return tool_error("kanban_verifier_result is only available in verify-only mode")
    if isinstance(verdict, dict):
        args = verdict
        verdict = args.get("verdict")
        summary = args.get("summary", "")
        reason = args.get("reason", "")
    if not isinstance(verdict, str):
        return tool_error(
            "verdict must be one of: approved, changes_requested, comment_only, superseded"
        )
    verdict_to_action = {
        "approved": "complete",
        "changes_requested": "re-block",
        "comment_only": "re-block",
        "superseded": "re-block",
    }
    action = verdict_to_action.get(verdict)
    if action is None:
        return tool_error(
            "verdict must be one of: approved, changes_requested, comment_only, superseded"
        )
    try:
        payload = _load_contract()
        summary_text = _bounded_text(summary, field="summary")
        reason_text = _bounded_text(reason, field="reason")
        if verdict != "approved" and not reason_text:
            return tool_error(f"{verdict} verdict requires a reason")
        frame = {
            "version": 1,
            "task_id": payload["task_id"],
            "run_id": payload["run_id"],
            "claim_lock": payload["claim_lock"],
            "contract_id": payload["contract_id"],
            "contract_hash": payload["contract_hash"],
            "pr_url": payload["pr_url"],
            "approved_head": payload["approved_head"],
            "approved_base": payload["approved_base"],
            "nonce": payload["nonce"],
            "action": action,
        }
        if summary_text:
            frame["summary"] = summary_text
        if reason_text:
            frame["reason"] = reason_text
        raw = (json.dumps(frame, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        with _EMIT_LOCK:
            if _EMITTED:
                return tool_error("verifier result already emitted")
            fd = _result_fd()
            try:
                os.write(fd, raw)
            finally:
                try:
                    os.close(fd)
                except OSError:
                    pass
            _EMITTED = True
    except Exception as exc:
        return tool_error(f"kanban_verifier_result: {exc}")
    return json.dumps({"ok": True, "emitted": True, "verdict": verdict})


KANBAN_VERIFIER_RESULT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "kanban_verifier_result",
        "description": (
            "Emit the single bound protected-merge verifier outcome. This tool "
            "can only emit once and cannot target an arbitrary task or run."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["approved", "changes_requested", "comment_only", "superseded"],
                    "description": (
                        "approved completes the bound task. changes_requested, "
                        "comment_only, and superseded re-block the bound task "
                        "for human input."
                    ),
                },
                "summary": {
                    "type": "string",
                    "description": "Short verifier summary for an approved verdict.",
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Required explanation for changes_requested, comment_only, "
                        "and superseded verdicts."
                    ),
                },
            },
            "required": ["verdict"],
        },
    },
}


registry.register(
    name="kanban_verifier_result",
    toolset="kanban_verifier",
    schema=KANBAN_VERIFIER_RESULT_SCHEMA,
    handler=_handle_verifier_result,
    check_fn=_check_verify_only,
    emoji="✔",
)
