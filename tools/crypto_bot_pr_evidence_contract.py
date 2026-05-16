#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import crypto_bot_remote_lifecycle_common as common


SCHEMA = "hermes.autonomy.crypto_bot_pr_evidence.v1"
REPORT_DIR_NAME = "pr-evidence"


def _list_diff(repo: Path, left: str, right: str) -> tuple[list[str], dict[str, Any]]:
    result = common.run_git(repo, ["diff", "--name-only", f"{left}..{right}"])
    if result["exit_code"] != 0:
        return [], result
    return [line for line in str(result["stdout"]).splitlines() if line], result


def _resolve_sidecar_path(
    gate_report: dict[str, Any],
    sidecar_result: Path | None,
) -> Path | None:
    if sidecar_result:
        return sidecar_result
    sidecar = gate_report.get("sidecar_result")
    if isinstance(sidecar, dict) and sidecar.get("path"):
        return Path(str(sidecar["path"]))
    return None


def _validators_from_gate(gate_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "git_diff_check": gate_report.get("git_diff_check"),
        "ruff_check": gate_report.get("ruff_check"),
        "targeted_tests": gate_report.get("targeted_tests"),
        "validator_commands": gate_report.get("validator_commands"),
    }


def _gate_allowlist(gate_report: dict[str, Any]) -> tuple[list[str], list[str]]:
    allowlist = gate_report.get("allowlist")
    if not isinstance(allowlist, dict):
        return [], []
    paths = [str(path) for path in allowlist.get("paths", []) if str(path)]
    patterns = [
        str(pattern)
        for pattern in allowlist.get("patterns", [])
        if str(pattern)
    ]
    return paths, patterns


def _dedupe_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for finding in findings:
        key = json.dumps(finding, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return deduped


def _render_pr_body(
    *,
    task_id: str,
    repo_root: Path,
    target_branch: str,
    source_branch: str,
    source_head: str,
    base_ref: str,
    merge_base: str,
    changed_files: list[str],
    completion_gate: Path,
    sidecar_result: Path,
    validators: dict[str, Any],
    blocked_surface_proof: str,
    task_source: Any,
) -> str:
    changed = "\n".join(f"- `{path}`" for path in changed_files) or "- none"
    validator_lines = [
        "- `git diff --check`: "
        + str((validators.get("git_diff_check") or {}).get("exit_code")),
        "- `ruff check`: "
        + str((validators.get("ruff_check") or {}).get("status", "not_applicable")),
        "- targeted tests: " + str(validators.get("targeted_tests")),
    ]
    return (
        f"# {task_id}: autonomous branch evidence\n\n"
        "## Scope\n\n"
        f"- Task source: `{task_source}`\n"
        f"- Repository: `{repo_root}`\n"
        f"- Target branch: `{target_branch}`\n"
        f"- Source branch: `{source_branch}`\n"
        f"- Source full SHA: `{source_head}`\n"
        f"- Completion gate base ref: `{base_ref}`\n"
        f"- Target merge base: `{merge_base}`\n\n"
        "## Changed Files\n\n"
        f"{changed}\n\n"
        "## Evidence\n\n"
        f"- Completion gate JSON path: `{completion_gate}`\n"
        f"- Codex sidecar result path: `{sidecar_result}`\n"
        "- Validators:\n"
        + "\n".join(validator_lines)
        + "\n"
        f"- Blocked-surface proof: `{blocked_surface_proof}`\n\n"
        "## Authority\n\n"
        "- This packet does not push, create a PR, update PR metadata, run CI, "
        "start runners, deploy, or merge.\n"
        "- PR creation is separate from merge. Merge-to-main requires a future "
        "merge gate and explicit merge authority.\n"
    )


def evaluate_pr_evidence(
    *,
    repo_root: Path,
    task_id: str,
    base_ref: str,
    target_branch: str,
    source_branch: str,
    source_head: str,
    completion_gate: Path,
    sidecar_result: Path | None,
    state_root: Path = common.DEFAULT_STATE_ROOT,
    title_output: Path | None = None,
    body_output: Path | None = None,
    packet_output: Path | None = None,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []

    gate_report: dict[str, Any] = {}
    if not completion_gate.exists():
        blockers.append(f"Completion gate report missing: {completion_gate}")
    else:
        try:
            gate_report = common.read_json(completion_gate)
        except json.JSONDecodeError as exc:
            blockers.append(f"Completion gate report is invalid JSON: {exc}")

    gate_pass = (
        gate_report.get("conclusion") == "PASS"
        and gate_report.get("gate_passed") is True
    )
    if not gate_pass:
        blockers.append("Completion gate report is not PASS")

    gate_branch_head_match = (
        gate_report.get("target_branch") == source_branch
        and gate_report.get("target_full_head") == source_head
    )
    if not gate_branch_head_match:
        blockers.append("Source branch/head do not match completion gate report")

    gate_repo = gate_report.get("repo_path")
    if gate_repo and Path(str(gate_repo)) != repo_root:
        blockers.append("Repository path does not match completion gate report")

    worktree_clean = common.worktree_clean(repo_root)
    if not worktree_clean:
        blockers.append("Worktree is dirty")

    if common.is_protected_branch_name(source_branch, target_branch):
        blockers.append("Source branch is default/protected and cannot be a PR source")

    source_ref_head = common.git_stdout(
        repo_root,
        ["rev-parse", f"refs/heads/{source_branch}"],
    )
    if source_ref_head != source_head:
        blockers.append("Local source branch ref does not resolve to source head")

    target_ref_head = common.git_stdout(repo_root, ["rev-parse", target_branch])
    if not target_ref_head:
        blockers.append(f"Target branch/ref is not resolvable: {target_branch}")

    base_full = common.git_stdout(repo_root, ["rev-parse", base_ref])
    if not base_full:
        blockers.append(f"Completion gate base ref is not resolvable: {base_ref}")

    merge_base = common.git_stdout(
        repo_root,
        ["merge-base", target_branch, source_head],
    )
    if not merge_base:
        blockers.append("Unable to compute merge-base with target branch")
        merge_base = ""

    gate_changed_files = [
        str(path) for path in gate_report.get("changed_files", []) if str(path)
    ]
    source_changed_files, source_diff = _list_diff(repo_root, base_ref, source_head)
    if source_diff["exit_code"] != 0:
        blockers.append("Unable to compute source diff from completion gate base")
    if source_changed_files != gate_changed_files:
        blockers.append(
            "Changed files from completion gate base do not match gate report"
        )

    pr_changed_files: list[str] = []
    if merge_base:
        pr_changed_files, pr_diff = _list_diff(repo_root, merge_base, source_head)
        if pr_diff["exit_code"] != 0:
            blockers.append("Unable to compute target PR diff from merge base")
        if pr_changed_files != gate_changed_files:
            blockers.append(
                "Target PR changed files do not match completion gate changed files"
            )

    sidecar_path = _resolve_sidecar_path(gate_report, sidecar_result)
    if not sidecar_path:
        blockers.append("Sidecar result path missing")
    elif not sidecar_path.exists():
        blockers.append(f"Sidecar result path does not exist: {sidecar_path}")

    allowlisted_paths, allowlisted_patterns = _gate_allowlist(gate_report)
    gate_scan = common.scan_blocked_surfaces(
        gate_changed_files,
        allowlisted_paths=allowlisted_paths,
        allowlisted_patterns=allowlisted_patterns,
    )
    pr_scan = common.scan_blocked_surfaces(
        pr_changed_files,
        allowlisted_paths=allowlisted_paths,
        allowlisted_patterns=allowlisted_patterns,
    )
    scan_findings = _dedupe_findings(
        gate_scan + [hit for hit in pr_scan if hit not in gate_scan]
    )
    if gate_report.get("blocked_surface_scan"):
        scan_findings = _dedupe_findings(
            scan_findings
            + [
                hit
                for hit in gate_report["blocked_surface_scan"]
                if isinstance(hit, dict)
            ]
        )
    blocked_surface_findings = common.block_findings(scan_findings)
    if blocked_surface_findings:
        blockers.append("Blocked surfaces are present in gate or target PR diff")

    validators = _validators_from_gate(gate_report)
    task_source = gate_report.get("task_source") or task_id
    blocked_surface_proof = (
        "PASS: no blocked surfaces in completion-gate changed files or target PR diff"
        if not blocked_surface_findings
        else "BLOCKED: blocked surfaces found"
    )
    title = f"{task_id}: validated autonomous branch evidence"
    body = ""
    secret_findings: list[str] = []
    if sidecar_path:
        body = _render_pr_body(
            task_id=task_id,
            repo_root=repo_root,
            target_branch=target_branch,
            source_branch=source_branch,
            source_head=source_head,
            base_ref=base_ref,
            merge_base=merge_base,
            changed_files=gate_changed_files,
            completion_gate=completion_gate,
            sidecar_result=sidecar_path,
            validators=validators,
            blocked_surface_proof=blocked_surface_proof,
            task_source=task_source,
        )
        secret_findings = common.secret_findings_in_text(body)
        if secret_findings:
            blockers.append("Generated PR body contains secret-looking content")

    pr_evidence_ready = not blockers
    output_dir = state_root / REPORT_DIR_NAME
    short_head = source_head[:7]
    stamp = common.utc_timestamp()
    packet_path = packet_output or output_dir / f"{stamp}-{task_id}-{short_head}.json"
    title_path = (
        title_output or output_dir / f"{stamp}-{task_id}-{short_head}-title.txt"
    )
    body_path = body_output or output_dir / f"{stamp}-{task_id}-{short_head}-body.md"

    packet: dict[str, Any] = {
        "schema": SCHEMA,
        "task_id": task_id,
        "repo_path": str(repo_root),
        "target_branch": target_branch,
        "base_ref": base_ref,
        "target_merge_base": merge_base or None,
        "source_branch": source_branch,
        "source_head": source_head,
        "completion_gate_json_path": str(completion_gate),
        "sidecar_result_path": str(sidecar_path) if sidecar_path else None,
        "gate_pass": gate_pass,
        "gate_branch_head_match": gate_branch_head_match,
        "worktree_clean": worktree_clean,
        "source_branch_non_protected": not common.is_protected_branch_name(
            source_branch, target_branch
        ),
        "target_branch_head": target_ref_head,
        "completion_gate_changed_files": gate_changed_files,
        "source_changed_files_from_gate_base": source_changed_files,
        "pr_changed_files_from_target_merge_base": pr_changed_files,
        "changed_files_match_gate": (
            source_changed_files == gate_changed_files
            and pr_changed_files == gate_changed_files
        ),
        "blocked_surface_findings": blocked_surface_findings,
        "blocked_surface_scan": scan_findings,
        "blocked_surface_proof": blocked_surface_proof,
        "validators": validators,
        "pr_title": title,
        "pr_title_path": str(title_path) if pr_evidence_ready else None,
        "pr_body_path": str(body_path) if pr_evidence_ready else None,
        "packet_path": str(packet_path) if write_artifacts else None,
        "secret_findings": secret_findings,
        "policy": common.load_policy_flags(),
        "pr_evidence_ready": pr_evidence_ready,
        "ready_for_pr_evidence_packet": pr_evidence_ready,
        "blockers": blockers,
        "warnings": warnings,
    }

    if write_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
        if pr_evidence_ready:
            title_path.write_text(title + "\n")
            body_path.write_text(body)

    return packet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--target-branch", default="main")
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--completion-gate", type=Path, required=True)
    parser.add_argument("--sidecar-result", type=Path)
    parser.add_argument("--state-root", type=Path, default=common.DEFAULT_STATE_ROOT)
    parser.add_argument("--title-output", type=Path)
    parser.add_argument("--body-output", type=Path)
    parser.add_argument("--packet-output", type=Path)
    parser.add_argument("--format", choices=("json",), default="json")
    args = parser.parse_args()

    packet = evaluate_pr_evidence(
        repo_root=args.repo_root,
        task_id=args.task_id,
        base_ref=args.base_ref,
        target_branch=args.target_branch,
        source_branch=args.source_branch,
        source_head=args.source_head,
        completion_gate=args.completion_gate,
        sidecar_result=args.sidecar_result,
        state_root=args.state_root,
        title_output=args.title_output,
        body_output=args.body_output,
        packet_output=args.packet_output,
    )
    print(json.dumps(packet, indent=2, sort_keys=True))
    return 0 if packet["pr_evidence_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
