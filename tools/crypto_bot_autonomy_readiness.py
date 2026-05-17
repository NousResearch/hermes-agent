#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import crypto_bot_evidence_issue_registry as evidence_issues
import crypto_bot_kanban_import_audit as kanban_import_audit
import crypto_bot_pr_ci_audit as pr_ci_audit
import crypto_bot_remote_lifecycle_common as remote_common
import runtime_asset_parity
import scan_crypto_bot_completion_claims as completion_claims

SCHEMA = "hermes.autonomy.crypto_bot_readiness.v2"
DEFAULT_HERMES_ROOT = Path("/Users/preston/.hermes/hermes-agent")
DEFAULT_CRYPTO_BOT_REPO = Path("/Users/preston/robinhood/crypto_bot")
DEFAULT_WRAPPER = Path("/Users/preston/.local/bin/hermes-codex-audit")
DEFAULT_SKILLS_ROOT = Path("/Users/preston/.hermes/skills")
DEFAULT_PLUGINS_ROOT = Path("/Users/preston/.hermes/plugins")
DEFAULT_STATE_ROOT = Path("/Users/preston/.local/state/hermes-operator")
DEFAULT_PR_CI_AUDIT_TOOL = DEFAULT_HERMES_ROOT / "tools/crypto_bot_pr_ci_audit.py"
DEFAULT_KANBAN_PREVIEW = (
    DEFAULT_STATE_ROOT / "kanban-import-previews/crypto_bot-preview.json"
)
EXPECTED_KANBAN_DEPENDENCIES = 101
KANBAN_TASK_LINE = re.compile(
    r"^\s*(?:\S+\s+)?(?P<id>S\d{3}[A-Z]?|dev13-\d+)\b(?:\s+(?P<status>\S+))?"
)

HISTORICAL_BASELINE_ARTIFACTS = {
    "docs/development/hermes_coding_work_packet_template.md": {
        "classification": "historical_evidence_missing_warning",
        "reason": (
            "Historical dev13 migration template; Hermes-side "
            "docs/autonomy/crypto_bot_target_loop_v2.md is the current "
            "source-of-truth."
        ),
    },
    "docs/implementation/hermes_pm_checkpoint_13b_plan.md": {
        "classification": "historical_evidence_missing_warning",
        "reason": (
            "Historical dev13 checkpoint evidence; not a current global "
            "safety prerequisite for Tenacity-native autonomy."
        ),
    },
}

TASK_SCOPED_VALIDATORS = {
    "scripts/validation/validate-security-evidence-wrapper.py": {
        "classification": "task_scoped_validator_missing_warning",
        "reason": (
            "Advisory dev13 validator for non-runtime security-evidence "
            "script work; required only when the selected task explicitly "
            "needs that validator."
        ),
        "required_task_ids": {"dev13-002"},
    }
}

GLOBAL_VALIDATORS = (
    "scripts/validation/validate-secrets-discipline.sh",
    "scripts/validation/validate-governance-baseline.sh",
)

REQUIRED_BASELINE_TRACKED = (
    *HISTORICAL_BASELINE_ARTIFACTS.keys(),
    *TASK_SCOPED_VALIDATORS.keys(),
)

REQUIRED_VALIDATORS = (
    *TASK_SCOPED_VALIDATORS.keys(),
    *GLOBAL_VALIDATORS,
)

HERMES_RELATED_PREFIXES = (
    "docs/pm/",
    "docs/hermes/",
    "docs/development/hermes_",
    "docs/architecture/hermes_",
    "docs/ci/hermes_",
    "docs/implementation/hermes_",
    "docs/operations/hermes_",
    "scripts/hermes_pm/",
    "scripts/hermes_operator/",
    "tests/hermes_pm/",
    "tests/hermes_operator/",
)

STALE_DEV13_004_BRANCHES = (
    "hermes/dev13-004-hermes-completion-verifier",
)

STALE_DEV13_004_PATH_PATTERNS = (
    re.compile(r"(^|/)hermes_pm_dev13-004_rollback\.md$"),
    re.compile(r"(^|/)hermes_pm_checkpoint_13d_plan\.md$"),
    re.compile(r"(^|/)verify_completion\.py$"),
    re.compile(r"(^|/)test_hermes_completion_verifier\.py$"),
)

def run_git(repo: Path, args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def simple_yaml_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or line.lstrip().startswith("#") or line.startswith(" "):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


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


def selected_task_requires_validator(
    selected_task_id: str | None,
    validator_relpath: str,
) -> bool:
    if not selected_task_id:
        return False
    meta = TASK_SCOPED_VALIDATORS.get(validator_relpath, {})
    task_ids = {str(item) for item in meta.get("required_task_ids", set())}
    return selected_task_id in task_ids


def tracked_files(repo: Path) -> list[str]:
    code, out, _ = run_git(repo, ["ls-files"])
    if code != 0:
        return []
    return out.splitlines()


def managed_files_equal(src: Path, dest: Path) -> bool:
    return runtime_asset_parity.managed_files_equal(src, dest)


def hermes_related_crypto_bot_paths(repo: Path) -> list[str]:
    return [
        path
        for path in tracked_files(repo)
        if path.startswith(HERMES_RELATED_PREFIXES)
    ]


def stale_dev13_004_paths(repo: Path) -> list[str]:
    return [
        path
        for path in tracked_files(repo)
        if any(pattern.search(path) for pattern in STALE_DEV13_004_PATH_PATTERNS)
    ]


def stale_dev13_004_branches(repo: Path) -> list[str]:
    code, branches, _ = run_git(
        repo,
        [
            "branch",
            "--format",
            "%(refname:short)",
            "--list",
            *STALE_DEV13_004_BRANCHES,
        ],
    )
    if code != 0:
        return []
    return [branch for branch in branches.splitlines() if branch.strip()]


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration_ms": round((time.time() - started) * 1000),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "exit_code": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "command timed out",
            "duration_ms": round((time.time() - started) * 1000),
        }


def tenacity_feature_readiness(
    hermes_root: Path,
    override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if override is not None:
        return override
    tool = hermes_root / "tools/hermes_tenacity_feature_readiness.py"
    if not tool.exists():
        return {
            "native_control_plane_ready": False,
            "blockers": [f"Tenacity feature readiness tool missing: {tool}"],
            "warnings": [],
        }
    result = _run(["python3", str(tool), "--format", "json"], cwd=hermes_root)
    try:
        payload = json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {
            "native_control_plane_ready": False,
            "blockers": [
                "Tenacity feature readiness tool did not emit valid JSON"
            ],
            "warnings": [result["stderr"].strip()] if result["stderr"].strip() else [],
            "exit_code": result["exit_code"],
        }
    if result["exit_code"] != 0 and payload.get("native_control_plane_ready") is True:
        payload["native_control_plane_ready"] = False
        payload.setdefault("blockers", []).append(
            "Tenacity feature readiness command exited non-zero"
        )
    return payload


def native_kanban_status(
    *,
    board_slug: str,
    expected_sessions: int,
    override: bool | None = None,
) -> dict[str, Any]:
    if override is not None:
        return {
            "board_slug": board_slug,
            "board_exists": override,
            "card_count": expected_sessions if override else None,
            "status_counts": {},
            "expected_sessions": expected_sessions,
            "native_kanban_ready": override,
            "source": "override",
            "warnings": [],
        }
    boards = _run(["hermes", "kanban", "boards", "list"], timeout=20)
    board_exists = False
    warnings: list[str] = []
    if boards["exit_code"] != 0:
        warnings.append("Hermes Kanban board list failed")
    else:
        for line in boards["stdout"].splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0].strip("●") == board_slug or (
                len(parts) > 1 and parts[1] == board_slug
            ):
                board_exists = True
                break
    card_count: int | None = None
    status_counts: dict[str, int] = {}
    if board_exists:
        listing = _run(["hermes", "kanban", "--board", board_slug, "list"], timeout=20)
        if listing["exit_code"] == 0:
            card_ids: set[str] = set()
            for line in listing["stdout"].splitlines():
                match = KANBAN_TASK_LINE.match(line)
                if not match:
                    continue
                card_ids.add(match.group("id"))
                status = match.group("status")
                if status:
                    status_counts[status] = status_counts.get(status, 0) + 1
            card_count = len(card_ids)
        else:
            warnings.append("Hermes Kanban card list failed")
    native_ready = board_exists and card_count == expected_sessions
    if board_exists and card_count != expected_sessions:
        warnings.append(
            "Native crypto_bot Kanban card count mismatch: "
            f"expected {expected_sessions}, found {card_count}"
        )
    return {
        "board_slug": board_slug,
        "board_exists": board_exists,
        "card_count": card_count,
        "status_counts": status_counts,
        "expected_sessions": expected_sessions,
        "native_kanban_ready": native_ready,
        "source": "hermes kanban boards list",
        "warnings": warnings,
    }


def migration_inventory_missing_paths(repo: Path, inventory_path: Path) -> list[str]:
    if not inventory_path.exists():
        return hermes_related_crypto_bot_paths(repo)
    body = inventory_path.read_text()
    return [
        path
        for path in hermes_related_crypto_bot_paths(repo)
        if f"`{path}`" not in body
    ]


def has_secret_looking_values(path: Path) -> list[str]:
    suspicious: list[str] = []
    ignored_dirs = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "node_modules",
        "venv",
        "__pycache__",
    }
    secret_value = re.compile(
        r"(?i)(api[_-]?key|token|password|passwd|private[_ -]?key|secret|credential)"
        r"\s*[:=]\s*['\"]?([A-Za-z0-9_./+=-]{12,})"
    )
    for candidate in path.rglob("*"):
        if any(part in ignored_dirs for part in candidate.parts):
            continue
        if not candidate.is_file():
            continue
        if candidate.suffix not in {".yaml", ".yml", ".json", ".toml"}:
            continue
        text = candidate.read_text(errors="ignore")
        if "BEGIN PRIVATE KEY" in text or re.search(r"sk-[A-Za-z0-9]{20,}", text):
            suspicious.append(str(candidate))
            continue
        for match in secret_value.finditer(text):
            value = match.group(2).lower()
            if value in {"redacted", "placeholder", "example", "changeme"}:
                continue
            if re.fullmatch(r"[a-z_][a-z0-9_.]*", value):
                continue
            suspicious.append(str(candidate))
            break
    return sorted(set(suspicious))


def verify_repaired_completion_issue(issue: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    gate_report_path = issue.get("gate_report_path")
    if not gate_report_path:
        return [
            f"Repaired issue {issue['issue_id']} is missing gate_report_path"
        ]
    report_path = Path(str(gate_report_path))
    if not report_path.exists():
        return [
            f"Repaired issue {issue['issue_id']} gate report is missing: "
            f"{report_path}"
        ]
    try:
        report = json.loads(report_path.read_text())
    except json.JSONDecodeError as exc:
        return [
            f"Repaired issue {issue['issue_id']} gate report is invalid JSON: {exc}"
        ]
    if report.get("gate_passed") is not True or report.get("conclusion") != "PASS":
        blockers.append(
            f"Repaired issue {issue['issue_id']} gate report is not PASS"
        )
    repaired_head = issue.get("repaired_head")
    if not repaired_head:
        blockers.append(
            f"Repaired issue {issue['issue_id']} is missing repaired_head"
        )
    elif report.get("target_full_head") != repaired_head:
        blockers.append(
            f"Repaired issue {issue['issue_id']} repaired_head does not match "
            "gate target_full_head"
        )
    if issue.get("bad_head") and issue.get("bad_head") == repaired_head:
        blockers.append(
            f"Repaired issue {issue['issue_id']} repaired_head equals bad_head"
        )
    if issue.get("branch") and report.get("target_branch") != issue.get("branch"):
        blockers.append(
            f"Repaired issue {issue['issue_id']} branch does not match gate report"
        )
    return blockers


def _json_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("*.json"))


def latest_completion_gate(
    state_root: Path,
    task_id: str,
) -> tuple[Path | None, dict[str, Any] | None]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in _json_files(state_root / "completion-gates"):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if str(payload.get("task_id") or payload.get("session_id")) != task_id:
            continue
        if (
            payload.get("gate_passed") is not True
            or payload.get("conclusion") != "PASS"
        ):
            continue
        matches.append((path, payload))
    return matches[-1] if matches else (None, None)


def latest_pr_evidence(
    state_root: Path,
    task_id: str,
    head: str | None = None,
) -> tuple[Path | None, dict[str, Any] | None]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in _json_files(state_root / "pr-evidence"):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if str(payload.get("task_id")) != task_id:
            continue
        if payload.get("pr_evidence_ready") is not True:
            continue
        if head and payload.get("source_head") != head:
            continue
        matches.append((path, payload))
    return matches[-1] if matches else (None, None)


def open_pr_for_branch(
    repo: Path,
    branch: str | None,
    head: str | None,
) -> dict[str, Any]:
    if not branch:
        return {"readable": False, "exists": False, "url": None, "number": None}
    remote_raw = remote_common.git_stdout(repo, ["remote", "get-url", "origin"])
    derived = remote_common.derive_gitea_from_remote(remote_raw)
    api_base = derived.get("api_base")
    owner = derived.get("owner")
    repo_name = derived.get("repo")
    if not api_base or not owner or not repo_name:
        return {"readable": False, "exists": False, "url": None, "number": None}
    pulls = remote_common.api_get_json(
        f"{api_base}/repos/{owner}/{repo_name}/pulls"
    )
    if pulls.get("status") != 200:
        return {
            "readable": False,
            "exists": False,
            "url": None,
            "number": None,
            "status": pulls.get("status"),
        }
    for pull in pulls.get("data") or []:
        if not isinstance(pull, dict):
            continue
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
            return {
                "readable": True,
                "exists": True,
                "url": pull.get("html_url") or pull.get("url"),
                "number": pull.get("number") or pull.get("id"),
            }
    return {"readable": True, "exists": False, "url": None, "number": None}


def remote_branch_head(repo: Path, branch: str | None) -> str | None:
    if not branch:
        return None
    code, out, _ = run_git(repo, ["ls-remote", "origin", f"refs/heads/{branch}"])
    if code == 0 and out.strip():
        first_field = out.splitlines()[0].split()[0]
        if first_field:
            return first_field
    code, out, _ = run_git(repo, ["rev-parse", f"refs/remotes/origin/{branch}"])
    if code == 0 and out.strip():
        return out.strip()
    return None


def s006_pr_pilot_status(repo: Path, state_root: Path) -> dict[str, Any]:
    gate_path, gate = latest_completion_gate(state_root, "S006")
    head = str(gate.get("target_full_head") or "") if gate else None
    branch = str(gate.get("target_branch") or "") if gate else None
    pr_path, pr = latest_pr_evidence(state_root, "S006", head=head)
    remote_head = remote_branch_head(repo, branch)
    pr_probe = open_pr_for_branch(repo, branch, head)
    blockers: list[str] = []
    warnings: list[str] = []
    if not gate:
        blockers.append("S006 has no current passing completion gate")
    if not pr:
        blockers.append("S006 has no ready PR evidence packet")
    if head and remote_head != head:
        blockers.append("S006 remote branch is missing or not at the gated head")
    if not pr_probe.get("readable"):
        warnings.append("S006 PR list is not readable")
    if pr_probe.get("exists"):
        blockers.append("S006 PR already exists; PR pilot request is not needed")
    ready = not blockers and bool(pr_probe.get("readable"))
    return {
        "ready_to_request_s006_pr_pilot": ready,
        "completion_gate_path": str(gate_path) if gate_path else None,
        "pr_evidence_path": str(pr_path) if pr_path else None,
        "branch": branch,
        "head": head,
        "remote_branch_head": remote_head,
        "pr_exists": bool(pr_probe.get("exists")),
        "pr_url": pr_probe.get("url"),
        "pr_number": pr_probe.get("number"),
        "blockers": blockers,
        "warnings": warnings,
    }


def run_checks(
    *,
    hermes_root: Path = DEFAULT_HERMES_ROOT,
    crypto_bot_repo: Path = DEFAULT_CRYPTO_BOT_REPO,
    wrapper_path: Path = DEFAULT_WRAPPER,
    skills_root: Path = DEFAULT_SKILLS_ROOT,
    plugins_root: Path = DEFAULT_PLUGINS_ROOT,
    state_root: Path = DEFAULT_STATE_ROOT,
    selected_task_id: str | None = None,
    tenacity_feature_override: dict[str, Any] | None = None,
    native_kanban_ready_override: bool | None = None,
    kanban_audit_override: dict[str, Any] | None = None,
    pr_ci_audit_override: dict[str, Any] | None = None,
    pr_ci_audit_tool_path: Path | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    source_runtime_blockers: list[str] = []
    source_runtime_warnings: list[str] = []
    tenacity_blockers: list[str] = []
    board_import_blockers: list[str] = []
    board_import_warnings: list[str] = []
    kanban_audit_blockers: list[str] = []
    pr_ci_audit_blockers: list[str] = []
    legacy_artifact_classification: dict[str, dict[str, Any]] = {}
    checked_paths: dict[str, str] = {}
    evidence_loop_blockers: list[str] = []
    evidence_loop_warnings: list[str] = []
    active_evidence_issues: list[dict[str, Any]] = []
    resolved_evidence_issues: list[dict[str, Any]] = []
    completion_claim_reports: list[dict[str, Any]] = []
    plan_valid = False
    plan_session_count = 0

    descriptor = hermes_root / "projects/crypto_bot/crypto_bot.project.yaml"
    target_loop = hermes_root / "docs/autonomy/crypto_bot_target_loop_v2.md"
    inventory = (
        hermes_root / "docs/autonomy/crypto_bot_hermes_asset_migration_inventory.md"
    )
    startup = hermes_root / "projects/crypto_bot/autonomous_startup_message.md"
    gate_tool = hermes_root / "tools/crypto_bot_completion_gate.py"
    prompt_generator = hermes_root / "tools/render_crypto_bot_sidecar_audit_prompt.py"
    claim_scanner = hermes_root / "tools/scan_crypto_bot_completion_claims.py"
    issue_registry_tool = hermes_root / "tools/crypto_bot_evidence_issue_registry.py"
    remote_readiness_tool = hermes_root / "tools/crypto_bot_remote_readiness.py"
    pr_evidence_tool = hermes_root / "tools/crypto_bot_pr_evidence_contract.py"
    merge_readiness_tool = hermes_root / "tools/crypto_bot_merge_readiness.py"

    for label, path in {
        "hermes_root": hermes_root,
        "crypto_bot_repo": crypto_bot_repo,
        "project_descriptor": descriptor,
        "target_loop": target_loop,
        "migration_inventory": inventory,
        "startup_message": startup,
        "sidecar_wrapper": wrapper_path,
        "completion_gate_tool": gate_tool,
        "sidecar_prompt_generator": prompt_generator,
        "completion_claim_scanner": claim_scanner,
        "evidence_issue_registry_tool": issue_registry_tool,
        "remote_readiness_tool": remote_readiness_tool,
        "pr_evidence_contract_tool": pr_evidence_tool,
        "merge_readiness_tool": merge_readiness_tool,
        "evidence_issue_root": evidence_issues.issue_root(state_root),
        "state_root": state_root,
    }.items():
        checked_paths[label] = str(path)

    if not hermes_root.exists():
        blockers.append(f"Hermes repo path missing: {hermes_root}")
    if not descriptor.exists():
        blockers.append(f"Managed project descriptor missing: {descriptor}")
    if not target_loop.exists():
        blockers.append(f"Target loop spec missing: {target_loop}")
    if not startup.exists():
        blockers.append(f"Autonomous startup message missing: {startup}")
    if not gate_tool.exists():
        evidence_loop_blockers.append(f"Completion gate tool missing: {gate_tool}")
    if not prompt_generator.exists():
        evidence_loop_blockers.append(
            f"Sidecar prompt generator missing: {prompt_generator}"
        )
    if not claim_scanner.exists():
        evidence_loop_blockers.append(
            f"Completion claim scanner missing: {claim_scanner}"
        )
    if not issue_registry_tool.exists():
        evidence_loop_blockers.append(
            f"Evidence issue registry tool missing: {issue_registry_tool}"
        )
    for label, path in {
        "remote readiness tool": remote_readiness_tool,
        "PR evidence contract tool": pr_evidence_tool,
        "merge readiness tool": merge_readiness_tool,
    }.items():
        if not path.exists():
            warnings.append(
                f"{label} missing; remote lifecycle readiness is unavailable"
            )

    descriptor_values: dict[str, str] = {}
    if descriptor.exists():
        descriptor_values = simple_yaml_values(descriptor)
        for key in (
            "project_id",
            "repo_path",
            "gitea_remote",
            "strategic_plan_path",
            "default_branch",
            "autonomy_profile",
        ):
            if not descriptor_values.get(key):
                blockers.append(f"Descriptor missing required key: {key}")

    if not crypto_bot_repo.exists():
        blockers.append(f"crypto_bot repo path missing: {crypto_bot_repo}")
    else:
        code, top, err = run_git(crypto_bot_repo, ["rev-parse", "--show-toplevel"])
        if code != 0:
            blockers.append(f"crypto_bot path is not a git repo: {err.strip()}")
        elif Path(top.strip()) != crypto_bot_repo:
            blockers.append(f"crypto_bot path is not repo root: {top.strip()}")
        code, status, _ = run_git(crypto_bot_repo, ["status", "--short"])
        if code != 0:
            blockers.append("Unable to read crypto_bot git status")
        elif status.strip():
            blockers.append("crypto_bot worktree is dirty")

        plan_rel = descriptor_values.get(
            "strategic_plan_path",
            "docs/planning/autoresearch_runpod_to_live_trade/plan.json",
        )
        plan_path = crypto_bot_repo / plan_rel
        checked_paths["strategic_plan"] = str(plan_path)
        if not plan_path.exists():
            blockers.append(f"Strategic plan missing: {plan_path}")
        else:
            try:
                plan = json.loads(plan_path.read_text())
                sessions = plan.get("sessions", [])
                plan_session_count = len(sessions)
                if len(sessions) != 90:
                    blockers.append(
                        f"Strategic plan expected 90 sessions, found {len(sessions)}"
                    )
                else:
                    plan_valid = True
            except json.JSONDecodeError as exc:
                blockers.append(f"Strategic plan is not valid JSON: {exc}")

        tracked = set(tracked_files(crypto_bot_repo))
        for rel, meta in HISTORICAL_BASELINE_ARTIFACTS.items():
            present = rel in tracked
            classification = str(meta["classification"])
            legacy_artifact_classification[rel] = {
                "present": present,
                "classification": classification,
                "blocks_readiness": False,
                "reason": meta["reason"],
            }
            if rel not in tracked:
                warnings.append(
                    f"Historical dev13 baseline artifact not tracked: {rel}"
                )
        for rel, meta in TASK_SCOPED_VALIDATORS.items():
            present = (crypto_bot_repo / rel).exists()
            required_for_selected_task = selected_task_requires_validator(
                selected_task_id,
                rel,
            )
            classification = str(meta["classification"])
            blocks_readiness = required_for_selected_task and not present
            legacy_artifact_classification[rel] = {
                "present": present,
                "classification": classification,
                "blocks_readiness": blocks_readiness,
                "required_for_selected_task": required_for_selected_task,
                "selected_task_id": selected_task_id,
                "reason": meta["reason"],
            }
            if not present and blocks_readiness:
                blockers.append(
                    "Required task-scoped validator missing for selected task "
                    f"{selected_task_id}: {rel}"
                )
            elif not present:
                warnings.append(f"Task-scoped validator not present: {rel}")
        for rel in GLOBAL_VALIDATORS:
            if not (crypto_bot_repo / rel).exists():
                legacy_artifact_classification[rel] = {
                    "present": False,
                    "classification": "required_global_validator_blocker",
                    "blocks_readiness": True,
                    "reason": "Current global safety validator is required.",
                }
                blockers.append(f"Required global validator missing: {rel}")
            else:
                legacy_artifact_classification[rel] = {
                    "present": True,
                    "classification": "required_global_validator_blocker",
                    "blocks_readiness": False,
                    "reason": "Current global safety validator is required.",
                }

        stale_branches = stale_dev13_004_branches(crypto_bot_repo)
        if stale_branches:
            blockers.append(
                "Stale dev13-004 branch evidence is present: "
                + ", ".join(stale_branches)
            )
        stale_paths = stale_dev13_004_paths(crypto_bot_repo)
        if stale_paths:
            blockers.append(
                "Stale dev13-004 file evidence is tracked: "
                + ", ".join(stale_paths[:10])
            )

        missing_inventory_paths = migration_inventory_missing_paths(
            crypto_bot_repo, inventory
        )
        if missing_inventory_paths:
            blockers.append(
                "Migration inventory missing Hermes-related paths: "
                + ", ".join(missing_inventory_paths[:10])
            )
            if len(missing_inventory_paths) > 10:
                warnings.append(
                    "Migration inventory has "
                    f"{len(missing_inventory_paths)} missing paths"
                )

    pm_skill = skills_root / "project-management/crypto-bot-pm/SKILL.md"
    sidecar_skill = skills_root / "development/codex-sidecar/SKILL.md"
    plugin_yaml = plugins_root / "crypto-bot-pm/plugin.yaml"
    plugin_tools = plugins_root / "crypto-bot-pm/tools.py"

    installed_skills = {
        "crypto-bot-pm": {
            "path": str(pm_skill),
            "installed": pm_skill.exists(),
            "source_backed": pm_skill.exists()
            and "/Users/preston/.hermes/hermes-agent" in pm_skill.read_text(errors="ignore"),
        },
        "codex-sidecar": {
            "path": str(sidecar_skill),
            "installed": sidecar_skill.exists(),
            "source_backed": sidecar_skill.exists()
            and "/Users/preston/.hermes/hermes-agent" in sidecar_skill.read_text(errors="ignore"),
        },
    }
    installed_plugins = {
        "crypto-bot-pm": {
            "path": str(plugin_yaml.parent),
            "installed": plugin_yaml.exists() and plugin_tools.exists(),
        }
    }

    for name, status in installed_skills.items():
        if not status["installed"]:
            source_runtime_blockers.append(f"Installed skill missing: {name}")
        elif not status["source_backed"]:
            source_runtime_warnings.append(
                f"Installed skill is not clearly Hermes-source-backed: {name}"
            )
    if not installed_plugins["crypto-bot-pm"]["installed"]:
        source_runtime_blockers.append("Installed crypto-bot-pm plugin missing")

    wrapper_exists = wrapper_path.exists()
    wrapper_executable = wrapper_exists and os.access(wrapper_path, os.X_OK)
    sidecar_status = {
        "wrapper_path": str(wrapper_path),
        "wrapper_exists": wrapper_exists,
        "wrapper_executable": wrapper_executable,
        "mode": "audit-readonly",
        "available_or_runnable": wrapper_executable,
    }
    if not wrapper_executable:
        source_runtime_blockers.append(
            f"Codex sidecar wrapper missing or not executable: {wrapper_path}"
        )

    suspicious = has_secret_looking_values(hermes_root)
    if suspicious:
        blockers.append("Secret-looking values found in generated configs")
        warnings.extend(f"Suspicious config: {path}" for path in suspicious[:10])

    source_runtime_pairs = {
        "crypto-bot-pm skill": (
            hermes_root / "skills/project-management/crypto-bot-pm",
            skills_root / "project-management/crypto-bot-pm",
        ),
        "codex-sidecar skill": (
            hermes_root / "skills/development/codex-sidecar",
            skills_root / "development/codex-sidecar",
        ),
        "crypto-bot-pm plugin": (
            hermes_root / "plugins/crypto-bot-pm",
            plugins_root / "crypto-bot-pm",
        ),
        "hermes-codex-audit wrapper": (
            hermes_root / "wrappers/hermes-codex-audit",
            wrapper_path,
        ),
    }
    runtime_asset_status: dict[str, dict[str, Any]] = {}
    for label, (src, dest) in source_runtime_pairs.items():
        status = runtime_asset_parity.compare_paths(src, dest)
        matches = status["matches_source"]
        runtime_asset_status[label] = status
        if not matches:
            source_runtime_blockers.append(
                f"Installed runtime asset diverges from Hermes source: {label}"
            )

    active_evidence_issues = evidence_issues.active_issues(state_root)
    resolved_evidence_issues = evidence_issues.resolved_issues(state_root)
    for issue in active_evidence_issues:
        subject = issue.get("task_id") or issue.get("claim_id") or issue["issue_id"]
        evidence_loop_blockers.append(
            f"Active evidence issue {issue['issue_id']}: "
            f"{issue['type']} for {subject}"
        )

    for issue in resolved_evidence_issues:
        if (
            issue.get("status") == "repaired"
            and issue.get("type") == "completion_gate_failure"
        ):
            evidence_loop_blockers.extend(verify_repaired_completion_issue(issue))
        if (
            issue.get("status") == "invalidated"
            and issue.get("type") == "unsupported_completion_claim"
            and not issue.get("invalidation_reason")
        ):
            evidence_loop_blockers.append(
                f"Invalidated issue {issue['issue_id']} is missing "
                "invalidation_reason"
            )

    if crypto_bot_repo.exists() and claim_scanner.exists():
        invalidated_claim_ids = sorted(
            {
                str(issue.get("claim_id"))
                for issue in resolved_evidence_issues
                if issue.get("type") == "unsupported_completion_claim"
                and issue.get("status") == "invalidated"
                and issue.get("claim_id")
            }
        )
        for claim_id in invalidated_claim_ids:
            report = completion_claims.scan_claim(
                repo_root=crypto_bot_repo,
                claim_id=claim_id,
                state_root=state_root,
            )
            completion_claim_reports.append(
                {
                    "claim_id": claim_id,
                    "classification": report["classification"],
                    "supported": report["supported"],
                    "invalidated": report["invalidated"],
                    "blocks_readiness": report["blocks_readiness"],
                    "blockers": report["blockers"],
                }
            )
            if report["blocks_readiness"]:
                evidence_loop_blockers.append(
                    f"Completion claim registry conflict {claim_id}: "
                    f"{report['classification']}"
                )
            elif report["classification"] == "INVALIDATED":
                evidence_loop_warnings.append(
                    f"Completion claim {claim_id} is explicitly invalidated"
                )

    migration_status = {
        "inventory_path": str(inventory),
        "inventory_exists": inventory.exists(),
        "missing_paths_count": 0,
    }
    if crypto_bot_repo.exists():
        migration_status["missing_paths_count"] = len(
            migration_inventory_missing_paths(crypto_bot_repo, inventory)
        )

    tenacity_payload = tenacity_feature_readiness(
        hermes_root,
        override=tenacity_feature_override,
    )
    native_control_plane_ready = bool(
        tenacity_payload.get("native_control_plane_ready")
    )
    if not native_control_plane_ready:
        payload_blockers = tenacity_payload.get("blockers") or []
        if payload_blockers:
            tenacity_blockers.extend(str(item) for item in payload_blockers)
        else:
            tenacity_blockers.append("Tenacity feature readiness is not green")

    board_slug = yaml_scalar(descriptor, "native_kanban_board", "crypto_bot")
    kanban_source = yaml_scalar(descriptor, "kanban_source_of_truth", "planned")
    native_lifecycle_required = kanban_source in {"planned", "native", "required"}
    effective_kanban_override = native_kanban_ready_override
    if (
        effective_kanban_override is None
        and crypto_bot_repo.resolve() != DEFAULT_CRYPTO_BOT_REPO.resolve()
    ):
        effective_kanban_override = False
    native_kanban = native_kanban_status(
        board_slug=str(board_slug or "crypto_bot"),
        expected_sessions=plan_session_count,
        override=effective_kanban_override,
    )
    native_kanban_ready = bool(native_kanban.get("native_kanban_ready"))
    if native_kanban.get("warnings"):
        board_import_warnings.extend(str(item) for item in native_kanban["warnings"])
    if native_lifecycle_required and not native_kanban_ready:
        board_import_warnings.append(
            "Native crypto_bot Kanban board has not been imported yet"
        )

    kanban_audit_payload: dict[str, Any] | None = None
    kanban_audit_ready = bool(effective_kanban_override)
    if kanban_audit_override is not None:
        kanban_audit_payload = kanban_audit_override
        kanban_audit_ready = (
            kanban_audit_payload.get("classification")
            in kanban_import_audit.VALID_IMPORT_CLASSIFICATIONS
        )
        if native_lifecycle_required and not kanban_audit_ready:
            native_kanban_ready = False
            reason = "Native Kanban import audit failed"
            payload_blockers = kanban_audit_payload.get("blockers") or []
            if payload_blockers:
                reason += ": " + "; ".join(str(item) for item in payload_blockers[:3])
            kanban_audit_blockers.append(reason)
    elif effective_kanban_override is None:
        preview_path = state_root / "kanban-import-previews/crypto_bot-preview.json"
        try:
            kanban_audit_payload = kanban_import_audit.evaluate_kanban_import_audit(
                preview_path=preview_path,
                board_slug=str(board_slug or "crypto_bot"),
                expected_card_count=plan_session_count or 90,
                expected_dependency_count=EXPECTED_KANBAN_DEPENDENCIES,
            )
            kanban_audit_ready = (
                kanban_audit_payload.get("classification")
                in kanban_import_audit.VALID_IMPORT_CLASSIFICATIONS
            )
        except Exception as exc:  # noqa: BLE001 - readiness fails closed
            kanban_audit_payload = {
                "classification": "KANBAN_AUDIT_ERROR",
                "blockers": [str(exc)],
            }
            kanban_audit_ready = False
        if native_lifecycle_required and not kanban_audit_ready:
            native_kanban_ready = False
            reason = "Native Kanban import audit failed"
            payload_blockers = kanban_audit_payload.get("blockers") or []
            if payload_blockers:
                reason += ": " + "; ".join(str(item) for item in payload_blockers[:3])
            kanban_audit_blockers.append(reason)

    s006_pr_status: dict[str, Any] = {
        "ready_to_request_s006_pr_pilot": False,
        "blockers": [],
        "warnings": [],
    }
    if crypto_bot_repo.exists():
        s006_pr_status = s006_pr_pilot_status(crypto_bot_repo, state_root)

    pr_ci_tool = pr_ci_audit_tool_path or (
        hermes_root / "tools/crypto_bot_pr_ci_audit.py"
    )
    pr_ci_payload: dict[str, Any] | None = None
    if pr_ci_audit_override is not None:
        pr_ci_payload = pr_ci_audit_override
    elif pr_ci_tool.exists():
        try:
            pr_ci_payload = pr_ci_audit.evaluate_pr_ci_audit(write_artifact=False)
        except Exception as exc:  # noqa: BLE001 - readiness fails closed
            pr_ci_payload = {
                "pr_exists": False,
                "ci_state": "inaccessible",
                "s006_remote_lifecycle_state": "pr_ci_audit_error",
                "blockers": [str(exc)],
            }
            pr_ci_audit_blockers.append(f"PR/CI audit failed: {exc}")
    elif s006_pr_status.get("pr_exists"):
        pr_ci_audit_blockers.append(
            "S006 PR exists but crypto_bot_pr_ci_audit.py is missing"
        )

    s006_pr_exists = bool(
        (pr_ci_payload or {}).get("pr_exists", s006_pr_status.get("pr_exists", False))
    )
    ci_evidence_ready = bool((pr_ci_payload or {}).get("ci_evidence_ready"))
    merge_ready = bool((pr_ci_payload or {}).get("merge_ready"))
    s006_remote_lifecycle_state = str(
        (pr_ci_payload or {}).get(
            "s006_remote_lifecycle_state",
            "pr_exists_unknown" if s006_pr_status.get("pr_exists") else "pr_absent",
        )
    )
    ready_to_request_s006_pr_pilot = (
        bool(s006_pr_status.get("ready_to_request_s006_pr_pilot"))
        and not s006_pr_exists
        and native_kanban_ready
        and kanban_audit_ready
    )

    blockers.extend(source_runtime_blockers)
    blockers.extend(tenacity_blockers)
    blockers.extend(kanban_audit_blockers)
    blockers.extend(pr_ci_audit_blockers)
    warnings.extend(source_runtime_warnings)
    warnings.extend(board_import_warnings)

    source_runtime_ready = not source_runtime_blockers
    basic_installation_ready = not blockers
    evidence_loop_ready = not evidence_loop_blockers
    local_evidence_ready = evidence_loop_ready
    ready_for_local_autonomy = (
        basic_installation_ready
        and source_runtime_ready
        and local_evidence_ready
        and native_control_plane_ready
    )
    board_import_ready = (
        basic_installation_ready
        and source_runtime_ready
        and local_evidence_ready
        and native_control_plane_ready
        and plan_valid
        and native_lifecycle_required
        and not native_kanban_ready
    )
    if not plan_valid:
        board_import_blockers.append("Strategic plan is not valid for board import")
    if not native_lifecycle_required:
        board_import_blockers.append(
            "Managed project does not require native Kanban lifecycle truth"
        )
    ready_to_request_board_import = board_import_ready and not board_import_blockers
    s006_remote_lifecycle_complete = s006_remote_lifecycle_state in {
        "remote_lifecycle_complete",
        "merged",
    }
    s006_remote_lifecycle_blocks_next_task = (
        native_lifecycle_required
        and native_kanban_ready
        and not s006_remote_lifecycle_complete
    )
    ready_for_next_task = (
        ready_for_local_autonomy
        and (native_kanban_ready or not native_lifecycle_required)
        and not s006_remote_lifecycle_blocks_next_task
    )
    remote_lifecycle_readiness = {
        "local_evidence_ready": local_evidence_ready,
        "remote_readiness_ready": False,
        "pr_evidence_ready": False,
        "ready_for_pr_evidence_packet": False,
        "ci_evidence_ready": ci_evidence_ready,
        "merge_readiness_ready": merge_ready,
        "merge_ready": merge_ready,
        "ready_for_local_autonomy": ready_for_local_autonomy,
        "ready_for_remote_pr_pilot": False,
        "ready_to_request_controlled_one_pr_pilot": False,
        "ready_to_request_s006_pr_pilot": ready_to_request_s006_pr_pilot,
        "ready_for_merge_autonomy": False,
        "blocks_next_task": s006_remote_lifecycle_blocks_next_task,
        "s006_pr_exists": s006_pr_exists,
        "s006_remote_lifecycle_state": s006_remote_lifecycle_state,
        "remote_probe_required": str(remote_readiness_tool),
        "note": (
            "Local evidence readiness is intentionally independent from "
            "remote, PR, CI, and merge readiness."
        ),
    }
    blockers.extend(evidence_loop_blockers)
    warnings.extend(evidence_loop_warnings)
    next_autonomous_action = (
        "Resolve readiness blockers before autonomous crypto_bot work."
    )
    if blockers:
        next_autonomous_action = (
            "Resolve readiness blockers before autonomous crypto_bot work."
        )
    elif ready_to_request_board_import:
        next_autonomous_action = (
            "Request exact Operator approval for native Kanban board import "
            "from the machine-verifiable preview; do not run S007A yet."
        )
    elif native_lifecycle_required and not native_kanban_ready:
        next_autonomous_action = (
            "Native Kanban lifecycle truth is required before next-task work."
        )
    elif s006_remote_lifecycle_blocks_next_task and ready_to_request_s006_pr_pilot:
        next_autonomous_action = (
            "Request exact Operator approval for the controlled S006 PR pilot "
            "retry; do not dispatch S007A until PR, CI, and merge evidence "
            "close S006 remote lifecycle."
        )
    elif s006_remote_lifecycle_blocks_next_task:
        next_autonomous_action = (
            "Hold next-task dispatch until S006 remote lifecycle evidence is "
            "complete."
        )
    elif ready_for_next_task:
        next_autonomous_action = "Proceed only with gated branch-local work."
    return {
        "schema": SCHEMA,
        "ready": ready_for_next_task,
        "basic_installation_ready": basic_installation_ready,
        "evidence_loop_ready": evidence_loop_ready,
        "source_runtime_ready": source_runtime_ready,
        "local_evidence_ready": local_evidence_ready,
        "native_control_plane_ready": native_control_plane_ready,
        "tenacity_feature_readiness": tenacity_payload,
        "native_kanban_ready": native_kanban_ready,
        "native_kanban_status": native_kanban,
        "board_import_ready": board_import_ready,
        "ready_to_request_board_import": ready_to_request_board_import,
        "remote_readiness_ready": False,
        "pr_evidence_ready": False,
        "ready_for_pr_evidence_packet": False,
        "ci_evidence_ready": ci_evidence_ready,
        "merge_ready": merge_ready,
        "merge_readiness_ready": merge_ready,
        "ready_for_local_autonomy": ready_for_local_autonomy,
        "ready_for_remote_pr_pilot": False,
        "ready_to_request_controlled_one_pr_pilot": False,
        "ready_to_request_s006_pr_pilot": ready_to_request_s006_pr_pilot,
        "ready_for_merge_autonomy": False,
        "ready_for_next_task": ready_for_next_task,
        "s006_remote_lifecycle_blocks_next_task": (
            s006_remote_lifecycle_blocks_next_task
        ),
        "blockers": blockers,
        "basic_installation_blockers": [
            blocker
            for blocker in blockers
            if blocker
            not in evidence_loop_blockers
            + source_runtime_blockers
            + tenacity_blockers
        ],
        "source_runtime_blockers": source_runtime_blockers,
        "source_runtime_warnings": source_runtime_warnings,
        "tenacity_blockers": tenacity_blockers,
        "board_import_blockers": board_import_blockers,
        "board_import_warnings": board_import_warnings,
        "kanban_audit_ready": kanban_audit_ready,
        "kanban_audit": kanban_audit_payload,
        "kanban_audit_blockers": kanban_audit_blockers,
        "pr_ci_audit": pr_ci_payload,
        "pr_ci_audit_blockers": pr_ci_audit_blockers,
        "s006_pr_exists": s006_pr_exists,
        "s006_remote_lifecycle_state": s006_remote_lifecycle_state,
        "evidence_loop_blockers": evidence_loop_blockers,
        "warnings": warnings,
        "evidence_loop_warnings": evidence_loop_warnings,
        "legacy_artifact_classification": legacy_artifact_classification,
        "task_scoped_validator_warnings": [
            rel
            for rel, meta in legacy_artifact_classification.items()
            if meta["classification"] == "task_scoped_validator_missing_warning"
            and not meta["present"]
            and not meta["blocks_readiness"]
        ],
        "checked_paths": checked_paths,
        "crypto_bot_repo_path": str(crypto_bot_repo),
        "hermes_repo_path": str(hermes_root),
        "installed_skills": installed_skills,
        "installed_plugins": installed_plugins,
        "runtime_asset_status": runtime_asset_status,
        "sidecar_status": sidecar_status,
        "active_evidence_issues": active_evidence_issues,
        "resolved_evidence_issues": resolved_evidence_issues,
        "completion_claim_reports": completion_claim_reports,
        "migration_status": migration_status,
        "remote_lifecycle_readiness": remote_lifecycle_readiness,
        "s006_pr_pilot_status": s006_pr_status,
        "next_autonomous_action": next_autonomous_action,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("json",), default="json")
    parser.add_argument("--hermes-root", type=Path, default=DEFAULT_HERMES_ROOT)
    parser.add_argument("--crypto-bot-repo", type=Path, default=DEFAULT_CRYPTO_BOT_REPO)
    parser.add_argument("--wrapper-path", type=Path, default=DEFAULT_WRAPPER)
    parser.add_argument("--skills-root", type=Path, default=DEFAULT_SKILLS_ROOT)
    parser.add_argument("--plugins-root", type=Path, default=DEFAULT_PLUGINS_ROOT)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--selected-task-id")
    args = parser.parse_args()

    payload = run_checks(
        hermes_root=args.hermes_root,
        crypto_bot_repo=args.crypto_bot_repo,
        wrapper_path=args.wrapper_path,
        skills_root=args.skills_root,
        plugins_root=args.plugins_root,
        state_root=args.state_root,
        selected_task_id=args.selected_task_id,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
