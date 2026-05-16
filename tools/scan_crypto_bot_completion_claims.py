#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import crypto_bot_evidence_issue_registry as issue_registry


SCHEMA = "hermes.autonomy.crypto_bot_completion_claim_scan.v1"
DEFAULT_REPO_ROOT = Path("/Users/preston/robinhood/crypto_bot")
DEFAULT_STATE_ROOT = Path("/Users/preston/.local/state/hermes-operator")
SKIP_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".log"}
SENSITIVE_NAME_PARTS = (
    ".env",
    "token",
    "secret",
    "credential",
    "cookie",
    "private",
    "keychain",
)


def run_git(repo: Path, args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": ["git", *args],
        "cwd": str(repo),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def safe_state_files(root: Path, claim_id: str) -> list[Path]:
    if not root.exists():
        return []
    matches: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            lowered = path.relative_to(root).as_posix().lower()
        except ValueError:
            lowered = path.name.lower()
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if any(part in lowered for part in SENSITIVE_NAME_PARTS):
            continue
        if claim_id.lower() in lowered:
            matches.append(path)
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if claim_id.lower() in text.lower():
            matches.append(path)
    return sorted(set(matches))


def passing_gate_files(paths: list[Path]) -> list[Path]:
    passing: list[Path] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("gate_passed") is True and payload.get("conclusion") == "PASS":
            passing.append(path)
    return passing


def gate_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def claim_proving_gate_files(paths: list[Path], claim_id: str) -> list[Path]:
    proving: list[Path] = []
    for path in paths:
        payload = gate_payload(path)
        if not payload:
            continue
        if (
            payload.get("gate_passed") is not True
            or payload.get("conclusion") != "PASS"
        ):
            continue
        if payload.get("claim_id") == claim_id:
            proving.append(path)
            continue
        if payload.get("task_id") == claim_id or payload.get("task_source") == claim_id:
            proving.append(path)
    return proving


def scan_claim(
    *,
    repo_root: Path,
    claim_id: str,
    state_root: Path = DEFAULT_STATE_ROOT,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    git_commands: list[dict[str, Any]] = []

    if not repo_root.exists():
        blockers.append(f"repo root missing: {repo_root}")
        return {
            "schema": SCHEMA,
            "claim_id": claim_id,
            "repo_path": str(repo_root),
            "state_root": str(state_root),
            "classification": "STATE_ARTIFACT_MISSING",
            "supported": False,
            "blockers": blockers,
            "warnings": warnings,
        }

    branch_cmd = run_git(
        repo_root,
        ["branch", "--all", "--list", f"*{claim_id}*"],
    )
    show_ref_cmd = run_git(repo_root, ["show-ref"])
    log_claim_cmd = run_git(
        repo_root,
        ["log", "--oneline", "--decorate", "--all", f"--grep={claim_id}"],
    )
    git_commands.extend([branch_cmd, show_ref_cmd, log_claim_cmd])

    branch_evidence = [
        line for line in branch_cmd["stdout"].splitlines() if line.strip()
    ]
    ref_evidence = [
        line
        for line in show_ref_cmd["stdout"].splitlines()
        if claim_id.lower() in line.lower()
    ]
    log_evidence = [
        line for line in log_claim_cmd["stdout"].splitlines() if line.strip()
    ]
    state_files = safe_state_files(state_root, claim_id)
    sidecar_files = [
        path for path in state_files if "codex-sidecar-audits" in str(path)
    ]
    gate_files = [
        path for path in state_files if "completion-gates" in str(path)
    ]
    passing_gates = passing_gate_files(gate_files)
    claim_proving_gates = claim_proving_gate_files(gate_files, claim_id)
    invalidations = issue_registry.matching_claim_issues(
        state_root=state_root,
        claim_id=claim_id,
        statuses={"invalidated"},
    )
    git_claim_evidence = bool(branch_evidence or ref_evidence or log_evidence)

    if invalidations and claim_proving_gates:
        blockers.append(
            f"invalidation for {claim_id} conflicts with passing claim gate evidence"
        )
        classification = "INVALIDATION_CONFLICT"
        supported = False
        invalidated = False
        blocks_readiness = True
    elif invalidations:
        classification = (
            "INVALIDATED_WITH_DISTINCT_TASK_ALIAS"
            if git_claim_evidence
            else "INVALIDATED"
        )
        supported = False
        invalidated = True
        blocks_readiness = False
        if git_claim_evidence:
            warnings.append(
                f"local git evidence mentions {claim_id}, but no passing gate "
                "claims to prove that invalidated completion claim"
            )
    else:
        invalidated = False
        blocks_readiness = True

    if not invalidations:
        if not branch_evidence:
            blockers.append(f"no local branch evidence for {claim_id}")
        if not ref_evidence:
            blockers.append(f"no git ref evidence for {claim_id}")
        if not log_evidence:
            blockers.append(f"no git commit/log evidence for {claim_id}")
        if not sidecar_files:
            blockers.append(f"no sidecar evidence for {claim_id}")
        if not passing_gates:
            blockers.append(f"no passing completion-gate evidence for {claim_id}")

        supported = not blockers
        blocks_readiness = not supported
        if supported:
            classification = "SUPPORTED_COMPLETION"
        elif not branch_evidence and not ref_evidence and not log_evidence:
            classification = "UNSUPPORTED_TELEGRAM_ONLY_CLAIM"
        elif branch_evidence and not log_evidence:
            classification = "LOST_LOCAL_BRANCH_OR_COMMIT"
        else:
            classification = "STATE_ARTIFACT_MISSING"

    return {
        "schema": SCHEMA,
        "claim_id": claim_id,
        "repo_path": str(repo_root),
        "state_root": str(state_root),
        "branch_evidence": branch_evidence,
        "ref_evidence": ref_evidence,
        "log_evidence": log_evidence,
        "sidecar_evidence": [str(path) for path in sidecar_files],
        "completion_gate_evidence": [str(path) for path in gate_files],
        "passing_completion_gate_evidence": [str(path) for path in passing_gates],
        "claim_proving_gate_evidence": [
            str(path) for path in claim_proving_gates
        ],
        "invalidation_evidence": [
            issue.get("_path") for issue in invalidations if issue.get("_path")
        ],
        "state_evidence": [str(path) for path in state_files],
        "git_commands": git_commands,
        "classification": classification,
        "supported": supported,
        "invalidated": invalidated,
        "blocks_readiness": blocks_readiness,
        "blockers": blockers,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--claim-id", required=True)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--format", choices=("json",), default="json")
    args = parser.parse_args()

    report = scan_claim(
        repo_root=args.repo_root,
        claim_id=args.claim_id,
        state_root=args.state_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not report["blocks_readiness"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
