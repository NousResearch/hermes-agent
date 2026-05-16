#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


SCHEMA = "hermes.autonomy.crypto_bot_evidence_issue.v1"
DEFAULT_STATE_ROOT = Path("/Users/preston/.local/state/hermes-operator")
ISSUE_DIR_NAME = "evidence-issues"
ACTIVE_STATUSES = {"active", "repair_attempted"}
RESOLVED_STATUSES = {"repaired", "invalidated", "superseded"}
VALID_STATUSES = ACTIVE_STATUSES | RESOLVED_STATUSES
VALID_TYPES = {
    "completion_gate_failure",
    "unsupported_completion_claim",
    "stale_sidecar",
    "validator_failure",
}


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def issue_root(state_root: Path = DEFAULT_STATE_ROOT) -> Path:
    return state_root / ISSUE_DIR_NAME


def slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return normalized.strip("-") or "evidence-issue"


def issue_path(root: Path, issue_id: str) -> Path:
    return root / f"{slug(issue_id)}.json"


def empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def normalize_issue(raw: dict[str, Any]) -> dict[str, Any]:
    issue = dict(raw)
    if issue.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported evidence issue schema: {issue.get('schema')}")
    if not issue.get("issue_id"):
        raise ValueError("Evidence issue missing issue_id")
    if issue.get("type") not in VALID_TYPES:
        raise ValueError(f"Unsupported evidence issue type: {issue.get('type')}")
    if issue.get("status") not in VALID_STATUSES:
        raise ValueError(f"Unsupported evidence issue status: {issue.get('status')}")
    issue.setdefault("task_id", None)
    issue.setdefault("session_id", None)
    issue.setdefault("branch_alias", None)
    issue.setdefault("claim_id", None)
    issue.setdefault("repo_path", None)
    issue.setdefault("branch", None)
    issue.setdefault("base", None)
    issue.setdefault("bad_head", None)
    issue.setdefault("repaired_head", None)
    issue.setdefault("gate_report_path", None)
    issue.setdefault("invalidation_reason", None)
    issue.setdefault("evidence_paths", [])
    issue.setdefault("operator_or_gate_basis", None)
    issue.setdefault("created_at", utc_timestamp())
    issue.setdefault("updated_at", issue["created_at"])
    return issue


def load_gate_report(path_value: str | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def validate_issue_for_write(issue: dict[str, Any]) -> None:
    if (
        issue.get("type") == "completion_gate_failure"
        and issue.get("status") == "repaired"
    ):
        gate_report_path = issue.get("gate_report_path")
        if not gate_report_path:
            raise ValueError("repaired completion issue requires gate_report_path")
        report = load_gate_report(str(gate_report_path))
        if report is None:
            raise ValueError(
                "repaired completion issue requires a readable gate report"
            )
        if report.get("gate_passed") is not True or report.get("conclusion") != "PASS":
            raise ValueError(
                "repaired completion issue requires a passing completion gate"
            )
        repaired_head = issue.get("repaired_head")
        if not repaired_head:
            raise ValueError("repaired completion issue requires repaired_head")
        if report.get("target_full_head") != repaired_head:
            raise ValueError(
                "repaired_head must match gate report target_full_head"
            )
        if issue.get("branch") and report.get("target_branch") != issue.get("branch"):
            raise ValueError("branch must match gate report target_branch")
        if issue.get("bad_head") and issue.get("bad_head") == repaired_head:
            raise ValueError("repaired_head must differ from bad_head")

    if (
        issue.get("type") == "unsupported_completion_claim"
        and issue.get("status") == "invalidated"
        and not issue.get("invalidation_reason")
    ):
        raise ValueError(
            "invalidated unsupported completion claim requires invalidation_reason"
        )


def load_issue(path: Path) -> dict[str, Any]:
    return normalize_issue(json.loads(path.read_text()))


def list_issues(state_root: Path = DEFAULT_STATE_ROOT) -> list[dict[str, Any]]:
    root = issue_root(state_root)
    if not root.exists():
        return []
    issues: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            issue = load_issue(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            issues.append(
                {
                    "schema": SCHEMA,
                    "issue_id": path.stem,
                    "status": "active",
                    "type": "validator_failure",
                    "repo_path": None,
                    "evidence_paths": [str(path)],
                    "operator_or_gate_basis": f"Unreadable registry issue: {exc}",
                    "created_at": utc_timestamp(),
                    "updated_at": utc_timestamp(),
                    "registry_error": str(exc),
                }
            )
            continue
        issue["_path"] = str(path)
        issues.append(issue)
    return issues


def active_issues(state_root: Path = DEFAULT_STATE_ROOT) -> list[dict[str, Any]]:
    return [
        issue
        for issue in list_issues(state_root)
        if issue.get("status") in ACTIVE_STATUSES
    ]


def resolved_issues(state_root: Path = DEFAULT_STATE_ROOT) -> list[dict[str, Any]]:
    return [
        issue
        for issue in list_issues(state_root)
        if issue.get("status") in RESOLVED_STATUSES
    ]


def write_issue(issue: dict[str, Any], state_root: Path = DEFAULT_STATE_ROOT) -> Path:
    root = issue_root(state_root)
    root.mkdir(parents=True, exist_ok=True)
    normalized = normalize_issue(issue)
    validate_issue_for_write(normalized)
    path = issue_path(root, str(normalized["issue_id"]))
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")
    return path


def build_issue(
    *,
    issue_id: str,
    issue_type: str,
    status: str,
    task_id: str | None = None,
    session_id: str | None = None,
    branch_alias: str | None = None,
    claim_id: str | None = None,
    repo_path: str | None = None,
    branch: str | None = None,
    base: str | None = None,
    bad_head: str | None = None,
    repaired_head: str | None = None,
    gate_report_path: str | None = None,
    invalidation_reason: str | None = None,
    evidence_paths: list[str] | None = None,
    operator_or_gate_basis: str | None = None,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = utc_timestamp()
    created_at = existing.get("created_at", now) if existing else now
    return normalize_issue(
        {
            "schema": SCHEMA,
            "issue_id": issue_id,
            "task_id": empty_to_none(task_id),
            "session_id": empty_to_none(session_id),
            "branch_alias": empty_to_none(branch_alias),
            "claim_id": empty_to_none(claim_id),
            "type": issue_type,
            "status": status,
            "repo_path": empty_to_none(repo_path),
            "branch": empty_to_none(branch),
            "base": empty_to_none(base),
            "bad_head": empty_to_none(bad_head),
            "repaired_head": empty_to_none(repaired_head),
            "gate_report_path": empty_to_none(gate_report_path),
            "invalidation_reason": empty_to_none(invalidation_reason),
            "created_at": created_at,
            "updated_at": now,
            "evidence_paths": evidence_paths or [],
            "operator_or_gate_basis": empty_to_none(operator_or_gate_basis),
        }
    )


def find_issue(
    *,
    state_root: Path,
    issue_id: str,
) -> dict[str, Any] | None:
    path = issue_path(issue_root(state_root), issue_id)
    if not path.exists():
        return None
    return load_issue(path)


def matching_claim_issues(
    *,
    state_root: Path,
    claim_id: str,
    statuses: set[str] | None = None,
) -> list[dict[str, Any]]:
    return [
        issue
        for issue in list_issues(state_root)
        if issue.get("claim_id") == claim_id
        and (statuses is None or issue.get("status") in statuses)
    ]


def matching_task_issues(
    *,
    state_root: Path,
    task_id: str,
    statuses: set[str] | None = None,
) -> list[dict[str, Any]]:
    return [
        issue
        for issue in list_issues(state_root)
        if issue.get("task_id") == task_id
        and (statuses is None or issue.get("status") in statuses)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--format", choices=("json",), default="json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list")

    upsert = subparsers.add_parser("upsert")
    upsert.add_argument("--issue-id", required=True)
    upsert.add_argument("--type", required=True, choices=sorted(VALID_TYPES))
    upsert.add_argument("--status", required=True, choices=sorted(VALID_STATUSES))
    upsert.add_argument("--task-id")
    upsert.add_argument("--session-id")
    upsert.add_argument("--branch-alias")
    upsert.add_argument("--claim-id")
    upsert.add_argument("--repo-path")
    upsert.add_argument("--branch")
    upsert.add_argument("--base")
    upsert.add_argument("--bad-head")
    upsert.add_argument("--repaired-head")
    upsert.add_argument("--gate-report-path")
    upsert.add_argument("--invalidation-reason")
    upsert.add_argument("--evidence-path", action="append", default=[])
    upsert.add_argument("--basis")

    invalidate = subparsers.add_parser("invalidate-claim")
    invalidate.add_argument("--claim-id", required=True)
    invalidate.add_argument("--issue-id")
    invalidate.add_argument("--repo-path")
    invalidate.add_argument("--reason", required=True)
    invalidate.add_argument("--evidence-path", action="append", default=[])

    args = parser.parse_args()

    if args.command == "list":
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "state_root": str(args.state_root),
                    "issue_root": str(issue_root(args.state_root)),
                    "issues": list_issues(args.state_root),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "invalidate-claim":
        issue_id = args.issue_id or f"{args.claim_id}-invalidated"
        existing = find_issue(state_root=args.state_root, issue_id=issue_id)
        issue = build_issue(
            issue_id=issue_id,
            issue_type="unsupported_completion_claim",
            status="invalidated",
            claim_id=args.claim_id,
            repo_path=args.repo_path,
            invalidation_reason=args.reason,
            evidence_paths=args.evidence_path,
            operator_or_gate_basis="explicit machine-readable invalidation",
            existing=existing,
        )
        path = write_issue(issue, args.state_root)
        print(json.dumps({"schema": SCHEMA, "path": str(path), "issue": issue}))
        return 0

    existing = find_issue(state_root=args.state_root, issue_id=args.issue_id)
    issue = build_issue(
        issue_id=args.issue_id,
        issue_type=args.type,
        status=args.status,
        task_id=args.task_id,
        session_id=args.session_id,
        branch_alias=args.branch_alias,
        claim_id=args.claim_id,
        repo_path=args.repo_path,
        branch=args.branch,
        base=args.base,
        bad_head=args.bad_head,
        repaired_head=args.repaired_head,
        gate_report_path=args.gate_report_path,
        invalidation_reason=args.invalidation_reason,
        evidence_paths=args.evidence_path,
        operator_or_gate_basis=args.basis,
        existing=existing,
    )
    path = write_issue(issue, args.state_root)
    print(json.dumps({"schema": SCHEMA, "path": str(path), "issue": issue}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
