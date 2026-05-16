#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import crypto_bot_autonomy_readiness as readiness
import crypto_bot_kanban_import_audit as kanban_audit
import crypto_bot_pr_ci_audit as pr_ci_audit
import runtime_asset_parity

SCHEMA = "hermes.autonomy.crypto_bot_control_plane_self_check.v1"
HERMES_ROOT = Path("/Users/preston/.hermes/hermes-agent")
CRYPTO_BOT_REPO = Path("/Users/preston/robinhood/crypto_bot")
HERMES_HOME = Path("/Users/preston/.hermes")
STATE_ROOT = Path("/Users/preston/.local/state/hermes-operator")
EXPECTED_CARDS = 90
EXPECTED_DEPENDENCIES = 101
VALID_PR_CI_STATES = {"absent", "pending", "failed", "passed", "stale"}

CRITICAL_FILES = (
    "tools/crypto_bot_kanban_import_audit.py",
    "tools/crypto_bot_pr_ci_audit.py",
    "tools/crypto_bot_autonomy_readiness.py",
    "tools/crypto_bot_control_plane_self_check.py",
    "tools/crypto_bot_gitea_pr_pilot.py",
    "tools/crypto_bot_kanban_import_execute.py",
    "projects/crypto_bot/crypto_bot.project.yaml",
    "projects/crypto_bot/autonomous_startup_message.md",
    "skills/project-management/crypto-bot-pm/SKILL.md",
    "skills/development/codex-sidecar/SKILL.md",
    "docs/autonomy/crypto_bot_native_kanban_goal_loop.md",
    "docs/autonomy/crypto_bot_hooks_policy.md",
    "docs/autonomy/crypto_bot_remote_lifecycle_contract.md",
    "docs/autonomy/crypto_bot_gitea_ci_pr_target_loop.md",
    "docs/autonomy/crypto_bot_completion_evidence_contract.md",
)

RUNTIME_PAIRS = {
    "crypto-bot-pm skill": (
        HERMES_ROOT / "skills/project-management/crypto-bot-pm",
        HERMES_HOME / "skills/project-management/crypto-bot-pm",
    ),
    "codex-sidecar skill": (
        HERMES_ROOT / "skills/development/codex-sidecar",
        HERMES_HOME / "skills/development/codex-sidecar",
    ),
    "crypto-bot-pm plugin": (
        HERMES_ROOT / "plugins/crypto-bot-pm",
        HERMES_HOME / "plugins/crypto-bot-pm",
    ),
    "hermes-codex-audit wrapper": (
        HERMES_ROOT / "wrappers/hermes-codex-audit",
        Path("/Users/preston/.local/bin/hermes-codex-audit"),
    ),
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def run_git(repo: Path, args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def tracked(repo: Path, relpath: str) -> bool:
    result = run_git(repo, ["ls-files", "--error-unmatch", relpath])
    return result["exit_code"] == 0


def critical_file_status(repo: Path) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for rel in CRITICAL_FILES:
        path = repo / rel
        status[rel] = {
            "path": str(path),
            "exists": path.exists(),
            "tracked": tracked(repo, rel),
            "sha256": sha256_file(path),
        }
    return status


def managed_files_equal(src: Path, dest: Path) -> bool:
    return runtime_asset_parity.managed_files_equal(src, dest)


def runtime_parity() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for label, (src, dest) in RUNTIME_PAIRS.items():
        result[label] = runtime_asset_parity.compare_paths(src, dest)
    return result


def durable_self_check_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"{timestamp}-crypto-bot-control-plane-self-check.json"
    if not path.exists():
        return path
    for idx in range(1, 100):
        candidate = output_dir / (
            f"{timestamp}-crypto-bot-control-plane-self-check-{idx}.json"
        )
        if not candidate.exists():
            return candidate
    raise RuntimeError("unable to allocate durable self-check path")


def build_report(
    *,
    hermes_root: Path,
    crypto_bot_repo: Path,
    state_root: Path,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    branch = run_git(hermes_root, ["branch", "--show-current"])["stdout"].strip()
    head = run_git(hermes_root, ["rev-parse", "HEAD"])["stdout"].strip()
    status = run_git(hermes_root, ["status", "--short", "--branch"])["stdout"]
    files = critical_file_status(hermes_root)
    for rel, item in files.items():
        if not item["exists"]:
            blockers.append(f"Critical file missing: {rel}")
        if not item["tracked"]:
            blockers.append(f"Critical file is not tracked: {rel}")

    parity = runtime_parity()
    for label, item in parity.items():
        if not item["matches_source"]:
            blockers.append(f"Installed runtime asset diverges from source: {label}")

    preview = state_root / "kanban-import-previews/crypto_bot-preview.json"
    try:
        kanban = kanban_audit.evaluate_kanban_import_audit(
            preview_path=preview,
            expected_card_count=EXPECTED_CARDS,
            expected_dependency_count=EXPECTED_DEPENDENCIES,
        )
        kanban_audit.write_durable_audit(kanban, state_root)
    except Exception as exc:  # noqa: BLE001 - self-check fails closed
        kanban = {"classification": "KANBAN_AUDIT_ERROR", "blockers": [str(exc)]}
    if kanban.get("classification") not in kanban_audit.VALID_IMPORT_CLASSIFICATIONS:
        blockers.append("Kanban audit failed")
    if kanban.get("card_count") != EXPECTED_CARDS:
        blockers.append("Kanban card count mismatch")
    if kanban.get("dependency_count") != EXPECTED_DEPENDENCIES:
        blockers.append("Kanban dependency count mismatch")

    pr_ci_tool = hermes_root / "tools/crypto_bot_pr_ci_audit.py"
    if not pr_ci_tool.exists():
        pr_ci = {"ci_state": "inaccessible", "blockers": ["PR/CI audit tool missing"]}
        blockers.append("PR/CI audit tool missing")
    else:
        try:
            pr_ci = pr_ci_audit.evaluate_pr_ci_audit(write_artifact=True)
        except Exception as exc:  # noqa: BLE001 - self-check fails closed
            pr_ci = {
                "ci_state": "inaccessible",
                "blockers": [str(exc)],
                "pr_exists": False,
            }
    if pr_ci.get("ci_state") not in VALID_PR_CI_STATES:
        blockers.append("PR/CI audit did not produce a stable CI state")

    try:
        readiness_payload = readiness.run_checks()
    except Exception as exc:  # noqa: BLE001 - self-check fails closed
        readiness_payload = {"ready": False, "blockers": [str(exc)]}
        blockers.append(f"Autonomy readiness failed: {exc}")

    if readiness_payload.get("native_kanban_ready") != (
        kanban.get("classification") in kanban_audit.VALID_IMPORT_CLASSIFICATIONS
    ):
        blockers.append("Readiness native_kanban_ready disagrees with Kanban audit")
    if readiness_payload.get("s006_pr_exists") != pr_ci.get("pr_exists"):
        blockers.append("Readiness S006 PR existence disagrees with PR/CI audit")
    if readiness_payload.get("s006_remote_lifecycle_state") != pr_ci.get(
        "s006_remote_lifecycle_state"
    ):
        blockers.append(
            "Readiness S006 remote lifecycle state disagrees with PR/CI audit"
        )

    crypto_status = run_git(crypto_bot_repo, ["status", "--short", "--branch"])
    if crypto_status["exit_code"] != 0:
        blockers.append("Unable to read crypto_bot git status")
    elif "\n" in crypto_status["stdout"].strip():
        blockers.append("crypto_bot worktree is dirty")

    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "hermes_repo_path": str(hermes_root),
        "hermes_home": str(HERMES_HOME),
        "hermes_branch": branch,
        "hermes_head": head,
        "hermes_git_status": status,
        "crypto_bot_repo_path": str(crypto_bot_repo),
        "crypto_bot_git_status": crypto_status,
        "critical_files": files,
        "installed_runtime_manifest": runtime_asset_parity.latest_manifest(
            state_root / "user-asset-install-manifests"
        ),
        "runtime_asset_status": parity,
        "kanban_audit": kanban,
        "pr_ci_audit": pr_ci,
        "autonomy_readiness": readiness_payload,
        "blockers": sorted(set(blockers)),
        "warnings": warnings,
        "conclusion": "FAIL" if blockers else "PASS",
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hermes crypto_bot control-plane source/runtime/audit self-check."
    )
    parser.add_argument("--format", choices=["json"], default="json")
    parser.add_argument("--hermes-root", type=Path, default=HERMES_ROOT)
    parser.add_argument("--crypto-bot-repo", type=Path, default=CRYPTO_BOT_REPO)
    parser.add_argument("--state-root", type=Path, default=STATE_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STATE_ROOT / "control-plane-self-checks",
    )
    parser.add_argument("--no-write", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_report(
        hermes_root=args.hermes_root,
        crypto_bot_repo=args.crypto_bot_repo,
        state_root=args.state_root,
    )
    if not args.no_write:
        path = durable_self_check_path(args.output_dir)
        report["self_check_json_path"] = str(path)
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["conclusion"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
