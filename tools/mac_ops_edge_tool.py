"""Service-gated read-only Mac/browser handoff tools.

These tools appear only when the privileged edge is explicitly configured.
They carry the model-authored contract without interpreting it; the edge owns
the GitLab credential and the local Mac worker remains the execution boundary.
"""

from __future__ import annotations

import json
from typing import Any

from gateway.mac_ops_edge_client import (
    MacOpsEdgeClientError,
    mac_ops_edge_configured,
    privileged_mac_ops_edge_client,
)
from gateway.mac_ops_edge_protocol import MacOpsReadOnlyClass
from tools.registry import registry, tool_error


def _submit(args: dict[str, Any], **_kwargs: Any) -> str:
    try:
        response = privileged_mac_ops_edge_client().submit_readonly(
            title=args.get("title"),
            task_class=args.get("task_class"),
            contract=args.get("contract"),
            idempotency_key=args.get("idempotency_key"),
        )
        return json.dumps(response, ensure_ascii=False, sort_keys=True)
    except (MacOpsEdgeClientError, TypeError, ValueError) as exc:
        code = getattr(exc, "code", "mac_ops_readonly_submit_invalid")
        uncertain = bool(getattr(exc, "dispatch_uncertain", False))
        return tool_error(
            json.dumps(
                {
                    "error": code,
                    "dispatch_uncertain": uncertain,
                    "instruction": (
                        "Read the same idempotency key before considering another "
                        "submission. Never create a new key to bypass uncertain state."
                        if uncertain
                        else "Correct the explicit request fields or report the blocker."
                    ),
                },
                sort_keys=True,
            )
        )


def _read(args: dict[str, Any], **_kwargs: Any) -> str:
    try:
        response = privileged_mac_ops_edge_client().read_task(
            issue_iid=args.get("issue_iid"),
            idempotency_key=args.get("idempotency_key"),
        )
        return json.dumps(response, ensure_ascii=False, sort_keys=True)
    except (MacOpsEdgeClientError, TypeError, ValueError) as exc:
        code = getattr(exc, "code", "mac_ops_task_read_invalid")
        return tool_error(json.dumps({"error": code}, sort_keys=True))


SUBMIT_SCHEMA = {
    "name": "mac_ops_readonly_submit",
    "description": (
        "Submit an exact model-authored read-only task contract to the separately "
        "authenticated Mac edge. Use it when the required evidence exists only "
        "in an explicitly selected local browser session, reviewed local files, "
        "or local CLI. The edge does not decide what the task means. It never "
        "accepts mutations or self-asserted approval. Preserve the idempotency key "
        "for reconciliation and then use mac_ops_task_read."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "maxLength": 240,
                "description": "Concise title for the exact handoff.",
            },
            "task_class": {
                "type": "string",
                "enum": [item.value for item in MacOpsReadOnlyClass],
                "description": "Explicit mechanical permission class for this read-only handoff.",
            },
            "contract": {
                "type": "string",
                "description": (
                    "Complete task contract containing Objective, Allowed scope, "
                    "Forbidden actions, Secrets handling, Verification, and Expected "
                    "report. Author the actual meaning here; do not include credentials."
                ),
            },
            "idempotency_key": {
                "type": "string",
                "pattern": "^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$",
                "description": "Stable unique key reused for every retry of this exact contract.",
            },
        },
        "required": ["title", "task_class", "contract", "idempotency_key"],
        "additionalProperties": False,
    },
}


READ_SCHEMA = {
    "name": "mac_ops_task_read",
    "description": (
        "Read live state and bounded non-system evidence notes for one previously "
        "submitted Mac task. Interpret the evidence yourself. An open issue is not "
        "success; a completed claim must be supported by the returned external state "
        "and receipt."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "issue_iid": {
                "type": "integer",
                "minimum": 1,
                "description": "Exact confidential GitLab issue IID returned by submission.",
            },
            "idempotency_key": {
                "type": "string",
                "pattern": "^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$",
                "description": "Stable key for this exact read observation.",
            },
        },
        "required": ["issue_iid", "idempotency_key"],
        "additionalProperties": False,
    },
}


registry.register(
    name="mac_ops_readonly_submit",
    toolset="mac_ops",
    schema=SUBMIT_SCHEMA,
    handler=_submit,
    check_fn=mac_ops_edge_configured,
    emoji="🖥️",
    max_result_size_chars=256_000,
)

registry.register(
    name="mac_ops_task_read",
    toolset="mac_ops",
    schema=READ_SCHEMA,
    handler=_read,
    check_fn=mac_ops_edge_configured,
    emoji="🖥️",
    max_result_size_chars=256_000,
)


__all__ = ["READ_SCHEMA", "SUBMIT_SCHEMA"]
