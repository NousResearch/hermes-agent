#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import crypto_bot_remote_lifecycle_common as common


SCHEMA = "hermes.autonomy.crypto_bot_merge_readiness.v1"


def _load_ci_evidence(path: Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return common.read_json(path)


def _checks_from_ci(ci: dict[str, Any] | None) -> tuple[bool, bool, str | None]:
    if not ci:
        return False, False, None
    head = ci.get("source_head") or ci.get("head_sha") or ci.get("sha")
    checks = ci.get("checks") or ci.get("statuses") or ci.get("check_runs")
    if isinstance(checks, dict) and "statuses" in checks:
        checks = checks["statuses"]
    if not isinstance(checks, list) or not checks:
        return False, False, str(head) if head else None
    values: list[str] = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        values.append(
            str(
                check.get("conclusion")
                or check.get("status")
                or check.get("state")
                or ""
            ).lower()
        )
    passed = bool(values) and all(value in {"success", "passed"} for value in values)
    return True, passed, str(head) if head else None


def evaluate_merge_readiness(
    *,
    repo_root: Path,
    pr_evidence_packet: Path,
    gitea_pr_number: str | None = None,
    gitea_pr_url: str | None = None,
    ci_evidence: dict[str, Any] | None = None,
    hermes_root: Path = common.DEFAULT_HERMES_ROOT,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []

    packet = common.read_json(pr_evidence_packet)
    packet_blockers = [str(item) for item in packet.get("blockers", [])]
    if packet.get("schema") != "hermes.autonomy.crypto_bot_pr_evidence.v1":
        blockers.append("PR evidence packet has unexpected schema")
    if packet_blockers:
        blockers.append("PR evidence packet is not clean")

    local_gate_pass = bool(packet.get("gate_pass"))
    if not local_gate_pass:
        blockers.append("Local completion gate did not PASS")

    pr_source_branch_head_matches_gate = bool(packet.get("gate_branch_head_match"))
    if not pr_source_branch_head_matches_gate:
        blockers.append("PR source branch/head do not match gate")

    source_head = str(packet.get("source_head") or "")
    source_branch = str(packet.get("source_branch") or "")
    source_ref_head = common.git_stdout(
        repo_root,
        ["rev-parse", f"refs/heads/{source_branch}"],
    )
    if source_ref_head != source_head:
        blockers.append("Local source branch no longer resolves to packet source head")

    target_branch = str(packet.get("target_branch") or "")
    target_branch_protected = common.is_protected_branch_name(target_branch)
    merge_would_mutate_protected_branch = target_branch_protected

    checks_present, checks_passed, ci_head = _checks_from_ci(ci_evidence)
    checks_current_for_source_head = bool(ci_head and ci_head == source_head)
    if not checks_present:
        blockers.append("CI/check evidence is missing")
    elif not checks_passed:
        blockers.append("CI/check evidence is not passing")
    if checks_present and not checks_current_for_source_head:
        blockers.append("CI/check evidence is stale or for the wrong source head")

    branch_protection_satisfied = not target_branch_protected
    if target_branch_protected:
        warnings.append(
            "Target branch is protected by Hermes policy; merge needs future gate"
        )

    policy = common.load_policy_flags(hermes_root)
    merge_authority_enabled = policy["merge_authority_enabled"]
    if not merge_authority_enabled:
        blockers.append("Merge authority is disabled by policy")

    merge_candidate_validated = (
        not packet_blockers
        and local_gate_pass
        and pr_source_branch_head_matches_gate
        and checks_present
        and checks_passed
        and checks_current_for_source_head
    )
    merge_ready = (
        merge_candidate_validated
        and merge_authority_enabled
        and branch_protection_satisfied
        and not merge_would_mutate_protected_branch
    )

    return {
        "schema": SCHEMA,
        "repo_path": str(repo_root),
        "pr_evidence_packet_path": str(pr_evidence_packet),
        "gitea_pr_number": gitea_pr_number,
        "gitea_pr_url": gitea_pr_url,
        "local_gate_pass": local_gate_pass,
        "pr_source_branch_head_matches_gate": pr_source_branch_head_matches_gate,
        "target_branch": target_branch,
        "target_branch_protected": target_branch_protected,
        "ci_check_evidence_present": checks_present,
        "checks_passed": checks_passed,
        "checks_current_for_source_head": checks_current_for_source_head,
        "branch_protection_satisfied": branch_protection_satisfied,
        "merge_would_mutate_protected_branch": merge_would_mutate_protected_branch,
        "merge_authority_enabled": merge_authority_enabled,
        "merge_candidate_validated": merge_candidate_validated,
        "merge_ready": merge_ready,
        "policy": policy,
        "blockers": blockers,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--pr-evidence-packet", type=Path, required=True)
    parser.add_argument("--gitea-pr-number")
    parser.add_argument("--gitea-pr-url")
    parser.add_argument("--ci-check-evidence", type=Path)
    parser.add_argument("--hermes-root", type=Path, default=common.DEFAULT_HERMES_ROOT)
    parser.add_argument("--format", choices=("json",), default="json")
    args = parser.parse_args()

    payload = evaluate_merge_readiness(
        repo_root=args.repo_root,
        pr_evidence_packet=args.pr_evidence_packet,
        gitea_pr_number=args.gitea_pr_number,
        gitea_pr_url=args.gitea_pr_url,
        ci_evidence=_load_ci_evidence(args.ci_check_evidence),
        hermes_root=args.hermes_root,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["merge_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
