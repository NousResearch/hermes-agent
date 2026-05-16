#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import crypto_bot_evidence_issue_registry as evidence_issues
import crypto_bot_remote_lifecycle_common as remote_common


SCHEMA = "hermes.autonomy.crypto_bot_kanban_import_preview.v1"
DEFAULT_STATE_ROOT = Path("/Users/preston/.local/state/hermes-operator")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yaml_scalar(path: Path, key: str, default: str | None = None) -> str | None:
    if not path.exists():
        return default
    match = re.search(
        rf"^\s*{re.escape(key)}\s*:\s*([^\n#]+)\s*$",
        path.read_text(encoding="utf-8", errors="replace"),
        re.M,
    )
    if not match:
        return default
    return match.group(1).strip().strip("'\"") or default


def state_json_files(state_root: Path, dirname: str) -> list[Path]:
    root = state_root / dirname
    if not root.exists():
        return []
    return sorted(root.glob("*.json"))


def completion_gates(state_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    gates: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in state_json_files(state_root, "completion-gates"):
        try:
            payload = read_json(path)
        except json.JSONDecodeError:
            continue
        task_id = str(payload.get("task_id") or payload.get("session_id") or "")
        if not task_id:
            continue
        if (
            payload.get("gate_passed") is not True
            or payload.get("conclusion") != "PASS"
        ):
            continue
        gates[task_id] = (path, payload)
    return gates


def pr_evidence_packets(state_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    packets: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in state_json_files(state_root, "pr-evidence"):
        try:
            payload = read_json(path)
        except json.JSONDecodeError:
            continue
        task_id = str(payload.get("task_id") or "")
        if not task_id or payload.get("pr_evidence_ready") is not True:
            continue
        packets[task_id] = (path, payload)
    return packets


def remote_branch_head(repo_root: Path, branch: str | None) -> str | None:
    if not branch:
        return None
    local = remote_common.git_stdout(
        repo_root,
        ["rev-parse", f"refs/remotes/origin/{branch}"],
    )
    if local:
        return local
    result = remote_common.run_git(
        repo_root,
        ["ls-remote", "--heads", "origin", branch],
    )
    if result["exit_code"] != 0:
        return None
    lines = str(result["stdout"]).splitlines()
    line = lines[0] if lines else ""
    return line.split()[0] if line.split() else None


def list_open_prs(
    repo_root: Path,
    api_get: remote_common.ApiGet = remote_common.api_get_json,
) -> tuple[bool, list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    remote_raw = remote_common.git_stdout(repo_root, ["remote", "get-url", "origin"])
    derived = remote_common.derive_gitea_from_remote(remote_raw)
    api_base = derived.get("api_base")
    owner = derived.get("owner")
    repo = derived.get("repo")
    if not api_base or not owner or not repo:
        return False, [], ["Unable to derive loopback Gitea PR API from origin"]
    response = api_get(f"{api_base}/repos/{owner}/{repo}/pulls")
    if response.get("status") != 200:
        warnings.append(f"Gitea pull request list returned {response.get('status')}")
        return False, [], warnings
    data = response.get("data")
    if not isinstance(data, list):
        return False, [], ["Gitea pull request list did not return a list"]
    return True, [item for item in data if isinstance(item, dict)], warnings


def matching_pr(
    pulls: list[dict[str, Any]],
    branch: str | None,
    head: str | None,
) -> dict[str, Any] | None:
    if not branch and not head:
        return None
    for pull in pulls:
        pull_head = pull.get("head") if isinstance(pull.get("head"), dict) else {}
        refs = {
            str(pull_head.get("ref") or ""),
            str(pull_head.get("label") or "").split(":")[-1],
        }
        shas = {
            str(pull_head.get("sha") or ""),
            str(pull.get("head_sha") or ""),
        }
        if branch in refs or (head and head in shas):
            return pull
    return None


def lane_for(session: dict[str, Any], status: str, reason: str) -> str:
    worker = str(session.get("worker") or "")
    if status == "review_required":
        lowered_reason = reason.lower()
        return (
            "crypto-ci-triage"
            if "pr" in lowered_reason or "ci" in lowered_reason
            else "crypto-reviewer"
        )
    if "audit" in str(session.get("session_type") or "").lower():
        return "crypto-codex-audit"
    if "codex-security" in worker:
        return "crypto-reviewer"
    if "codex-observability" in worker:
        return "crypto-ci-triage"
    if "codex-" in worker or session.get("session_type") == "implementation":
        return "crypto-implementer"
    return "crypto-pm-orchestrator"


def is_underspecified(session: dict[str, Any]) -> bool:
    return not session.get("title") or not session.get("allowed_write_scope")


def evidence_metadata(
    *,
    repo_root: Path,
    task_id: str,
    gate: tuple[Path, dict[str, Any]] | None,
    pr_packet: tuple[Path, dict[str, Any]] | None,
    pulls: list[dict[str, Any]],
) -> dict[str, Any]:
    gate_path: Path | None = None
    gate_payload: dict[str, Any] = {}
    if gate:
        gate_path, gate_payload = gate
    pr_path: Path | None = None
    pr_payload: dict[str, Any] = {}
    if pr_packet:
        pr_path, pr_payload = pr_packet
    branch = str(
        gate_payload.get("target_branch")
        or pr_payload.get("source_branch")
        or ""
    ) or None
    head = str(
        gate_payload.get("target_full_head")
        or pr_payload.get("source_head")
        or ""
    ) or None
    pr = matching_pr(pulls, branch, head)
    sidecar = gate_payload.get("sidecar_result")
    sidecar_path = None
    if isinstance(sidecar, dict):
        sidecar_path = sidecar.get("path")
    return {
        "completion_gate_path": str(gate_path) if gate_path else None,
        "completion_gate_passed": bool(gate_payload),
        "sidecar_path": sidecar_path,
        "pr_evidence_packet_path": str(pr_path) if pr_path else None,
        "pr_evidence_ready": bool(pr_payload),
        "pr_url": (pr or {}).get("html_url") or (pr or {}).get("url"),
        "pr_number": (pr or {}).get("number") or (pr or {}).get("id"),
        "pr_exists": pr is not None,
        "branch": branch,
        "head": head,
        "remote_branch_head": remote_branch_head(repo_root, branch),
        "task_id": task_id,
    }


def build_preview(
    *,
    repo_root: Path,
    project_descriptor: Path,
    state_root: Path = DEFAULT_STATE_ROOT,
    api_get: remote_common.ApiGet = remote_common.api_get_json,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    plan_rel = yaml_scalar(
        project_descriptor,
        "strategic_plan_path",
        "docs/planning/autoresearch_runpod_to_live_trade/plan.json",
    )
    board_slug = yaml_scalar(project_descriptor, "native_kanban_board", "crypto_bot")
    plan_path = repo_root / str(plan_rel)
    if not repo_root.exists():
        blockers.append(f"repo_root missing: {repo_root}")
    if not project_descriptor.exists():
        blockers.append(f"project descriptor missing: {project_descriptor}")
    if not plan_path.exists():
        blockers.append(f"strategic plan missing: {plan_path}")
    worktree_clean = (
        remote_common.worktree_clean(repo_root) if repo_root.exists() else False
    )
    if not worktree_clean:
        blockers.append("crypto_bot worktree is dirty")
    if blockers:
        return {
            "schema": SCHEMA,
            "generated_at": utc_now(),
            "board_slug": board_slug,
            "repo_root": str(repo_root),
            "project_descriptor": str(project_descriptor),
            "source_plan": str(plan_path),
            "total_sessions": 0,
            "cards_to_import": 0,
            "cards": [],
            "dependencies": [],
            "unresolved_claims": {},
            "import_safety": {
                "live_import_allowed": False,
                "ready_to_request_import_approval": False,
                "blockers": blockers,
                "warnings": warnings,
            },
        }

    plan = read_json(plan_path)
    sessions = plan.get("sessions")
    if not isinstance(sessions, list):
        blockers.append("strategic plan sessions must be a list")
        sessions = []
    by_id = {
        str(session.get("session_id")): session
        for session in sessions
        if isinstance(session, dict) and session.get("session_id")
    }
    if len(by_id) != len(sessions):
        blockers.append("strategic plan contains duplicate or missing session IDs")
    for session in by_id.values():
        for parent in session.get("depends_on") or []:
            if parent not in by_id:
                blockers.append(
                    f"session {session['session_id']} depends on missing {parent}"
                )

    gate_by_task = completion_gates(state_root)
    pr_by_task = pr_evidence_packets(state_root)
    pulls_readable, pulls, pr_warnings = list_open_prs(repo_root, api_get=api_get)
    warnings.extend(pr_warnings)
    if not pulls_readable:
        warnings.append("Open PR existence could not be proven from read-only Gitea")

    active_issues = evidence_issues.active_issues(state_root)
    active_issue_tasks = {
        str(issue.get("task_id") or issue.get("claim_id") or "")
        for issue in active_issues
    }

    done_ids: set[str] = set()
    dependencies: list[dict[str, str]] = []
    unresolved_completed_without_gate: list[str] = []
    unresolved_completed_without_pr_evidence: list[str] = []
    cards: list[dict[str, Any]] = []

    for session in sessions:
        task_id = str(session.get("session_id"))
        deps = [str(item) for item in session.get("depends_on") or []]
        dependencies.extend({"parent": dep, "child": task_id} for dep in deps)
        gate = gate_by_task.get(task_id)
        pr_packet = pr_by_task.get(task_id)
        evidence = evidence_metadata(
            repo_root=repo_root,
            task_id=task_id,
            gate=gate,
            pr_packet=pr_packet,
            pulls=pulls,
        )
        plan_status = str(session.get("status") or "unknown")
        status = "triage"
        reason = "planned card requires specification"
        unresolved_claims: list[str] = []

        if task_id in active_issue_tasks:
            status = "blocked"
            reason = "active evidence issue blocks import as ready"
        elif is_underspecified(session):
            status = "triage"
            reason = "missing title or allowed_write_scope"
        elif plan_status in {
            "completed_current_evidence_backed",
            "not_required_current_evidence_backed",
        }:
            if not gate:
                status = "blocked"
                reason = "plan_completed_claim_lacks_completion_gate"
                unresolved_completed_without_gate.append(task_id)
                unresolved_claims.append(reason)
                if not pr_packet:
                    unresolved_completed_without_pr_evidence.append(task_id)
                    unresolved_claims.append("plan_completed_claim_lacks_pr_evidence")
            elif evidence["pr_exists"]:
                status = "done"
                reason = "completion gate PASS and PR exists"
                done_ids.add(task_id)
            else:
                status = "review_required"
                reason = "blocked_remote_pr_missing"
        elif any(dep not in done_ids for dep in deps):
            status = "blocked"
            reason = "dependencies_not_done"
        elif plan_status == "planned_next":
            status = "ready"
            reason = "planned_next with satisfied dependencies"
        elif plan_status == "planned":
            status = "todo"
            reason = "planned with satisfied dependencies"
        else:
            status = "triage"
            reason = f"unmapped plan status: {plan_status}"

        card = {
            "card_id": task_id,
            "title": session.get("title") or task_id,
            "summary": session.get("summary") or session.get("title") or task_id,
            "plan_status": plan_status,
            "dependencies": deps,
            "parent_links": [
                {"parent": dep, "child": task_id}
                for dep in deps
            ],
            "initial_status": status,
            "status_reason": reason,
            "assignee_lane": lane_for(session, status, reason),
            "workspace_strategy": {
                "root": (
                    "/Users/preston/.local/state/hermes-operator/"
                    "kanban-workspaces/crypto_bot"
                ),
                "per_card": "isolated native Kanban worker workspace",
                "product_repo": str(repo_root),
                "branch_prefix": "hermes/",
                "no_live_dispatch": True,
            },
            "evidence_metadata": evidence,
            "unresolved_claims": unresolved_claims,
            "operator_approval_required": bool(
                session.get("operator_approval_required")
            ),
            "allowed_write_scope": session.get("allowed_write_scope") or [],
            "forbidden_in_session": session.get("forbidden_in_session") or [],
            "validation_expectations": session.get("validation_expectations") or [],
        }
        cards.append(card)

    if unresolved_completed_without_gate:
        warnings.append(
            "Plan-completed sessions lack current completion-gate evidence: "
            + ", ".join(unresolved_completed_without_gate)
        )
    if unresolved_completed_without_pr_evidence:
        warnings.append(
            "Plan-completed sessions lack current PR evidence packets: "
            + ", ".join(unresolved_completed_without_pr_evidence)
        )

    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "board_slug": board_slug,
        "repo_root": str(repo_root),
        "project_descriptor": str(project_descriptor),
        "source_plan": str(plan_path),
        "source_plan_sha256": sha256_file(plan_path),
        "total_sessions": len(sessions),
        "cards_to_import": len(cards),
        "card_ids": [card["card_id"] for card in cards],
        "cards": cards,
        "dependencies": dependencies,
        "unresolved_claims": {
            "plan_completed_lacking_gate_evidence": (
                unresolved_completed_without_gate
            ),
            "plan_completed_lacking_pr_evidence": (
                unresolved_completed_without_pr_evidence
            ),
        },
        "import_safety": {
            "live_import_allowed": False,
            "writes_live_kanban_cards": False,
            "worker_dispatch_allowed": False,
            "product_file_writes_allowed": False,
            "gitea_mutation_allowed": False,
            "pr_creation_allowed": False,
            "merge_allowed": False,
            "ready_to_request_import_approval": (
                not blockers and len(cards) == len(sessions)
            ),
            "blockers": blockers,
            "warnings": warnings,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--project-descriptor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("json",), default="json")
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    args = parser.parse_args()

    payload = build_preview(
        repo_root=args.repo_root,
        project_descriptor=args.project_descriptor,
        state_root=args.state_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["import_safety"]["ready_to_request_import_approval"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
