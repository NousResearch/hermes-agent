#!/usr/bin/env python3
"""Prototype PR workflow guardrail tool for Wave 6.

This module intentionally avoids importing Wave 5-only tools/task_tool.py or
agent/task_scheduler.py. Until those APIs exist on the target branch, task
creation is represented as a structured remediation request only.
"""

from __future__ import annotations

import json
from typing import Any

from agent.pr_workflow import (
    DISALLOWED_PR_ACTIONS,
    GhCliError,
    GhPrPollingAdapter,
    RemediationRequest,
    _base_contract,
    append_ai_signature,
    redact_sensitive_text,
)
from tools.registry import registry


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _unsupported_action(action: str) -> str:
    return _json(
        {
            "ok": False,
            "error": "unsupported_pr_workflow_action",
            "message": (
                f"Wave 6 prototype does not merge, auto-merge, close, or delete PR branches. "
                f"Requested action: {action}"
            ),
        }
    )


def _require_repo_pr(repo: str | None, pr_number: int | None) -> str | None:
    if not str(repo or "").strip() or not pr_number:
        return _json({"ok": False, "error": "repo_and_pr_number_required", "message": "repo and pr_number are required before polling or sending PR workflow actions."})
    return None


def pr_workflow_tool(
    *,
    action: str,
    repo: str | None = None,
    pr_number: int | None = None,
    body: str | None = None,
    reason: str | None = None,
    title: str | None = None,
    summary: str | None = None,
    local_verification_passed: bool | None = None,
) -> str:
    """Run a Wave 6 PR workflow prototype action.

    Supported actions are intentionally read-only/format-only except for
    returning structured remediation specs. No GitHub token is accepted or
    persisted, and merge-like actions fail closed.
    """
    normalized = str(action or "").strip().lower().replace("-", "_")
    if not normalized:
        return _json({"ok": False, "error": "action_required"})
    if normalized in DISALLOWED_PR_ACTIONS:
        return _unsupported_action(normalized)
    if normalized == "format_comment":
        return _json({"ok": True, "body": append_ai_signature(body or "")})
    if normalized == "poll_pr":
        error = _require_repo_pr(repo, pr_number)
        if error:
            return error
        try:
            result = GhPrPollingAdapter().poll_pull_request(str(repo), int(pr_number), local_verification_passed=local_verification_passed)
        except GhCliError as exc:
            return _json({"ok": False, "error": exc.__class__.__name__, "message": redact_sensitive_text(str(exc))})
        except Exception as exc:
            return _json({"ok": False, "error": "pr_workflow_poll_failed", "message": redact_sensitive_text(str(exc))})
        return _json({"ok": True, "poll_result": result.model_dump(mode="json")})
    if normalized == "create_remediation_task":
        error = _require_repo_pr(repo, pr_number)
        if error:
            return error
        pr = {"url": f"https://github.com/{repo}/pull/{pr_number}", "headRefName": ""}
        contract = _base_contract(str(repo), int(pr_number), pr, "pr_workflow_tool")
        contract.update(
            {
                "task": title or f"Remediate PR #{pr_number}",
                "expected_outcome": summary or "A verified remediation request for the PR workflow loop.",
            }
        )
        request = RemediationRequest(
            reason=reason or "manual",
            title=title or f"Remediate PR #{pr_number}",
            summary=summary or "Scheduler-ready remediation request created; enqueue via the task tool when mutation permission is granted.",
            task_contract=contract,
        )
        return _json(
            {
                "ok": True,
                "mode": "scheduler_ready_remediation_request",
                "remediation_request": request.model_dump(mode="json"),
            }
        )
    return _json({"ok": False, "error": "unknown_pr_workflow_action", "message": f"Unsupported action: {normalized}"})


PR_WORKFLOW_SCHEMA = {
    "name": "pr_workflow",
    "description": "Prototype Wave 6 PR workflow guardrails: poll PR state, format AI-disclosed comments, and return remediation task specs without merging or storing GitHub tokens.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "One of: poll_pr, format_comment, create_remediation_task. Merge-like actions are refused."},
            "repo": {"type": "string", "description": "GitHub owner/repo for PR polling or remediation specs."},
            "pr_number": {"type": "integer", "description": "Pull request number."},
            "body": {"type": "string", "description": "Comment body to format with mandatory AI signature."},
            "reason": {"type": "string", "description": "Remediation reason such as ci_failure or review_feedback."},
            "title": {"type": "string", "description": "Remediation title."},
            "summary": {"type": "string", "description": "Remediation summary."},
            "local_verification_passed": {"type": "boolean", "description": "Whether local verification gate has passed."},
        },
        "required": ["action"],
    },
}


registry.register(
    name="pr_workflow",
    toolset="github",
    schema=PR_WORKFLOW_SCHEMA,
    handler=lambda args, **kw: pr_workflow_tool(
        action=args.get("action", ""),
        repo=args.get("repo"),
        pr_number=args.get("pr_number"),
        body=args.get("body"),
        reason=args.get("reason"),
        title=args.get("title"),
        summary=args.get("summary"),
        local_verification_passed=args.get("local_verification_passed"),
    ),
    check_fn=lambda: True,
    emoji="🔁",
)
