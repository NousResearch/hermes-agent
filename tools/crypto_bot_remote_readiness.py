#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import crypto_bot_remote_lifecycle_common as common


SCHEMA = "hermes.autonomy.crypto_bot_remote_readiness.v1"


def _local_evidence_ready(repo_root: Path, state_root: Path) -> tuple[bool, list[str]]:
    try:
        import crypto_bot_autonomy_readiness as local_readiness
    except ImportError:
        return False, ["Local evidence readiness tool is not importable"]
    payload = local_readiness.run_checks(
        crypto_bot_repo=repo_root,
        state_root=state_root,
    )
    blockers = [str(item) for item in payload.get("blockers", [])]
    return bool(payload.get("local_evidence_ready")), blockers


def _api_url(api_base: str, owner: str, repo: str, suffix: str) -> str:
    return f"{api_base}/repos/{owner}/{repo}{suffix}"


def _status_state(data: Any) -> tuple[bool, bool]:
    if not isinstance(data, dict):
        return False, False
    total_count = int(data.get("total_count") or 0)
    state = str(data.get("state") or "").lower()
    return total_count > 0, state == "success"


def evaluate_remote_readiness(
    *,
    repo_root: Path,
    base_branch: str = "main",
    gitea_url: str | None = None,
    state_root: Path = common.DEFAULT_STATE_ROOT,
    hermes_root: Path = common.DEFAULT_HERMES_ROOT,
    local_evidence_ready_override: bool | None = None,
    api_get: common.ApiGet = common.api_get_json,
    remote_reachable_override: bool | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []

    remote_raw = common.git_stdout(repo_root, ["remote", "get-url", "origin"])
    remote_url = common.sanitize_url(remote_raw)
    if common.remote_contains_userinfo(remote_raw):
        warnings.append("Remote URL contained userinfo and was redacted in output")

    local_branch = common.git_stdout(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    local_head = common.git_stdout(repo_root, ["rev-parse", "HEAD"])
    worktree_clean = common.worktree_clean(repo_root)
    if not worktree_clean:
        blockers.append("Local worktree is dirty; remote integration is blocked")

    if remote_reachable_override is None:
        ls_remote = common.run_git(repo_root, ["ls-remote", "--heads", "origin"])
        remote_reachable = ls_remote["exit_code"] == 0
    else:
        remote_reachable = remote_reachable_override
    if not remote_reachable:
        blockers.append("origin is not reachable with read-only git ls-remote")

    derived = common.derive_gitea_from_remote(remote_raw)
    api_base = common.normalize_gitea_api_base(gitea_url) or derived["api_base"]
    owner = str(derived["owner"] or "")
    repo_name = str(derived["repo"] or "")
    if gitea_url and (not owner or not repo_name):
        owner = "preston"
        repo_name = repo_root.name

    gitea_read_api_reachable = False
    gitea_auth_required = False
    branch_protection_readable = False
    default_branch_protected: bool | None = None
    pull_request_api_readable = False
    actions_or_checks_readable = False
    runner_status_readable = False
    default_branch = base_branch
    checks_present = False
    checks_passed = False
    api_probe_summary: dict[str, dict[str, Any]] = {}

    if not api_base or not owner or not repo_name:
        blockers.append(
            "Gitea read API was not derived from a loopback/local HTTP remote"
        )
    else:
        version = api_get(f"{api_base}/version")
        api_probe_summary["version"] = {
            "status": version.get("status"),
            "error": version.get("error"),
        }
        gitea_read_api_reachable = version.get("status") == 200
        if not gitea_read_api_reachable:
            blockers.append("Gitea read API is unavailable")

        repo_meta = api_get(_api_url(api_base, owner, repo_name, ""))
        api_probe_summary["repo"] = {
            "status": repo_meta.get("status"),
            "error": repo_meta.get("error"),
        }
        if isinstance(repo_meta.get("data"), dict):
            default_branch = str(
                repo_meta["data"].get("default_branch") or default_branch
            )
        if repo_meta.get("status") == 401:
            gitea_auth_required = True

        branch = api_get(
            _api_url(api_base, owner, repo_name, f"/branches/{default_branch}")
        )
        api_probe_summary["default_branch"] = {
            "status": branch.get("status"),
            "error": branch.get("error"),
        }
        if isinstance(branch.get("data"), dict) and "protected" in branch["data"]:
            branch_protection_readable = True
            default_branch_protected = bool(branch["data"].get("protected"))
        if branch.get("status") == 401:
            gitea_auth_required = True

        protections = api_get(
            _api_url(api_base, owner, repo_name, "/branch_protections")
        )
        api_probe_summary["branch_protections"] = {
            "status": protections.get("status"),
            "error": protections.get("error"),
        }
        if protections.get("status") == 200:
            branch_protection_readable = True
        if protections.get("status") == 401:
            gitea_auth_required = True
            warnings.append("Branch protection collection requires authentication")

        pulls = api_get(_api_url(api_base, owner, repo_name, "/pulls"))
        api_probe_summary["pulls"] = {
            "status": pulls.get("status"),
            "error": pulls.get("error"),
        }
        pull_request_api_readable = pulls.get("status") == 200
        if not pull_request_api_readable:
            blockers.append("Pull request API is not readable")
        if pulls.get("status") == 401:
            gitea_auth_required = True

        if local_head:
            statuses = api_get(
                _api_url(api_base, owner, repo_name, f"/statuses/{local_head}")
            )
            combined = api_get(
                _api_url(api_base, owner, repo_name, f"/commits/{local_head}/status")
            )
            api_probe_summary["statuses"] = {
                "status": statuses.get("status"),
                "count": len(statuses.get("data") or [])
                if isinstance(statuses.get("data"), list)
                else None,
                "error": statuses.get("error"),
            }
            api_probe_summary["combined_status"] = {
                "status": combined.get("status"),
                "error": combined.get("error"),
            }
            actions_or_checks_readable = statuses.get("status") == 200 or (
                combined.get("status") == 200
            )
            if combined.get("status") == 200:
                checks_present, checks_passed = _status_state(combined.get("data"))
            elif isinstance(statuses.get("data"), list):
                checks_present = bool(statuses["data"])
                status_values = [
                    str(item.get("status") or "").lower()
                    for item in statuses["data"]
                    if isinstance(item, dict)
                ]
                checks_passed = bool(status_values) and all(
                    status in {"success", "passed"} for status in status_values
                )

        tasks = api_get(_api_url(api_base, owner, repo_name, "/actions/tasks"))
        api_probe_summary["actions_tasks"] = {
            "status": tasks.get("status"),
            "error": tasks.get("error"),
            "total_count": tasks.get("data", {}).get("total_count")
            if isinstance(tasks.get("data"), dict)
            else None,
        }
        if tasks.get("status") == 200:
            actions_or_checks_readable = True
        if tasks.get("status") == 401:
            gitea_auth_required = True

        runners = api_get(_api_url(api_base, owner, repo_name, "/actions/runners"))
        api_probe_summary["actions_runners"] = {
            "status": runners.get("status"),
            "error": runners.get("error"),
        }
        runner_status_readable = runners.get("status") == 200
        if runners.get("status") == 401:
            gitea_auth_required = True
            warnings.append("Runner status requires authentication")

    workflow_files = common.find_workflow_files(repo_root)
    workflow_risks = common.workflow_risk_findings(repo_root, workflow_files)
    if workflow_files:
        warnings.append("Workflow files discovered as read-only controlled surfaces")

    if local_evidence_ready_override is None:
        local_evidence_ready, local_evidence_blockers = _local_evidence_ready(
            repo_root, state_root
        )
    else:
        local_evidence_ready = local_evidence_ready_override
        local_evidence_blockers = []
    if not local_evidence_ready:
        warnings.extend(
            f"Local evidence readiness blocker: {item}"
            for item in local_evidence_blockers[:5]
        )

    matching_gate = common.find_matching_completion_gate(
        state_root,
        branch=local_branch,
        head=local_head,
    )
    pr_packet_probe: dict[str, Any] | None = None
    pr_evidence_ready = False
    pr_tool = hermes_root / "tools/crypto_bot_pr_evidence_contract.py"
    if matching_gate and pr_tool.exists() and local_branch and local_head:
        try:
            import crypto_bot_pr_evidence_contract as pr_contract

            gate_data = common.read_json(matching_gate)
            pr_packet_probe = pr_contract.evaluate_pr_evidence(
                repo_root=repo_root,
                task_id=str(gate_data.get("task_id") or "unknown"),
                base_ref=str(gate_data.get("base_ref") or default_branch),
                target_branch=default_branch,
                source_branch=local_branch,
                source_head=local_head,
                completion_gate=matching_gate,
                sidecar_result=None,
                state_root=state_root,
                write_artifacts=False,
            )
            pr_evidence_ready = bool(pr_packet_probe.get("pr_evidence_ready"))
            if not pr_evidence_ready:
                warnings.extend(
                    "PR evidence probe blocker: " + str(item)
                    for item in pr_packet_probe.get("blockers", [])[:5]
                )
        except Exception as exc:  # noqa: BLE001 - readiness records probe failure
            warnings.append(f"PR evidence packet generator probe failed: {exc}")
    elif not matching_gate:
        warnings.append("No passing completion gate matches current branch and HEAD")
    elif not pr_tool.exists():
        warnings.append("PR evidence packet generator tool is missing")

    if not actions_or_checks_readable:
        blockers.append("CI/check evidence is not readable")
    if not runner_status_readable:
        blockers.append("Runner status is not readable without additional authority")
    if not checks_present:
        blockers.append("No current CI/check evidence found for local HEAD")
    elif not checks_passed:
        blockers.append("Current CI/check evidence is not passing for local HEAD")

    policy = common.load_policy_flags(hermes_root)
    remote_readiness_ready = (
        remote_reachable and gitea_read_api_reachable and pull_request_api_readable
    )
    ci_evidence_ready = actions_or_checks_readable and checks_present and checks_passed
    pr_creation_ready = (
        local_evidence_ready
        and remote_readiness_ready
        and pr_evidence_ready
        and policy["controlled_remote_branch_push_enabled"]
        and policy["one_pr_creation_pilot_enabled"]
    )
    ready_to_request_controlled_one_pr_pilot = (
        local_evidence_ready
        and remote_readiness_ready
        and pr_evidence_ready
        and worktree_clean
        and bool(local_branch)
        and not common.is_protected_branch_name(local_branch, default_branch)
    )
    merge_readiness_supported = (
        branch_protection_readable
        and runner_status_readable
        and ci_evidence_ready
        and policy["merge_authority_enabled"]
    )
    ready_for_merge_autonomy = merge_readiness_supported

    if not local_evidence_ready:
        next_action = "Resolve local evidence readiness before remote lifecycle work."
    elif not remote_readiness_ready:
        next_action = "Keep remote integration paused and report remote probe blockers."
    elif ready_to_request_controlled_one_pr_pilot and not pr_creation_ready:
        next_action = (
            "Local PR evidence is clean; request exact Operator approval before "
            "one non-force push and one PR creation."
        )
    elif not pr_evidence_ready:
        next_action = "Generate or repair the local PR evidence packet before pilot."
    elif not ci_evidence_ready:
        next_action = (
            "CI evidence remains unavailable until a remote branch or PR exists."
        )
    elif not pr_creation_ready:
        next_action = (
            "Remote read probe completed; push and PR creation remain policy-gated."
        )
    else:
        next_action = "Remote PR pilot prerequisites are staged; await Operator policy."

    return {
        "schema": SCHEMA,
        "repo_path": str(repo_root),
        "remote_url": remote_url,
        "default_branch": default_branch,
        "local_branch": local_branch,
        "local_head": local_head,
        "worktree_clean": worktree_clean,
        "remote_reachable": remote_reachable,
        "gitea_read_api_reachable": gitea_read_api_reachable,
        "gitea_auth_required": gitea_auth_required,
        "branch_protection_readable": branch_protection_readable,
        "default_branch_protected": default_branch_protected,
        "pull_request_api_readable": pull_request_api_readable,
        "actions_or_checks_readable": actions_or_checks_readable,
        "runner_status_readable": runner_status_readable,
        "workflow_files": workflow_files,
        "workflow_risk_findings": workflow_risks,
        "local_evidence_ready": local_evidence_ready,
        "remote_ready": remote_readiness_ready,
        "pr_creation_ready": pr_creation_ready,
        "ci_evidence_ready": ci_evidence_ready,
        "merge_readiness_supported": merge_readiness_supported,
        "remote_readiness_ready": remote_readiness_ready,
        "pr_evidence_ready": pr_evidence_ready,
        "ready_for_pr_evidence_packet": pr_evidence_ready,
        "merge_readiness_ready": merge_readiness_supported,
        "ready_for_local_autonomy": local_evidence_ready,
        "ready_for_remote_pr_pilot": pr_creation_ready,
        "ready_to_request_controlled_one_pr_pilot": (
            ready_to_request_controlled_one_pr_pilot
        ),
        "ready_for_merge_autonomy": ready_for_merge_autonomy,
        "matching_completion_gate": str(matching_gate) if matching_gate else None,
        "pr_evidence_probe": pr_packet_probe,
        "policy": policy,
        "api_probe_summary": api_probe_summary,
        "blockers": blockers,
        "warnings": warnings,
        "next_action": next_action,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=common.DEFAULT_REPO_ROOT)
    parser.add_argument("--base-branch", default="main")
    parser.add_argument("--format", choices=("json",), default="json")
    parser.add_argument("--gitea-url")
    parser.add_argument("--state-root", type=Path, default=common.DEFAULT_STATE_ROOT)
    parser.add_argument("--hermes-root", type=Path, default=common.DEFAULT_HERMES_ROOT)
    args = parser.parse_args()

    payload = evaluate_remote_readiness(
        repo_root=args.repo_root,
        base_branch=args.base_branch,
        gitea_url=args.gitea_url,
        state_root=args.state_root,
        hermes_root=args.hermes_root,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["remote_readiness_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
