#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import crypto_bot_autonomy_readiness as autonomy_readiness


SCHEMA = "hermes.autonomy.tenacity_feature_readiness.v1"
HERMES_AGENT = Path("/Users/preston/.hermes/hermes-agent")
HERMES_ROOT = Path("/Users/preston/.hermes/hermes-agent")
CUSTOM_ASSETS = {
    "crypto-bot-pm skill": (
        HERMES_ROOT / "skills/project-management/crypto-bot-pm",
        Path("/Users/preston/.hermes/skills/project-management/crypto-bot-pm"),
    ),
    "codex-sidecar skill": (
        HERMES_ROOT / "skills/development/codex-sidecar",
        Path("/Users/preston/.hermes/skills/development/codex-sidecar"),
    ),
    "crypto-bot-pm plugin": (
        HERMES_ROOT / "plugins/crypto-bot-pm",
        Path("/Users/preston/.hermes/plugins/crypto-bot-pm"),
    ),
    "hermes-codex-audit wrapper": (
        HERMES_ROOT / "wrappers/hermes-codex-audit",
        Path("/Users/preston/.local/bin/hermes-codex-audit"),
    ),
}


def run(args: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        args,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def hermes(*args: str) -> tuple[int, str, str]:
    return run(["hermes", *args])


def git_head(path: Path) -> str | None:
    if not path.exists():
        return None
    code, out, _ = run(["git", "-C", str(path), "rev-parse", "HEAD"])
    return out.strip() if code == 0 else None


def first_line(text: str) -> str:
    return text.splitlines()[0] if text.splitlines() else ""


def custom_skill_parity() -> dict[str, bool]:
    return {
        label: autonomy_readiness.managed_files_equal(src, dest)
        for label, (src, dest) in CUSTOM_ASSETS.items()
    }


def build_payload() -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []

    version_code, version_out, version_err = hermes("version")
    kanban_code, kanban_out, _ = hermes("kanban", "--help")
    hooks_code, hooks_out, _ = hermes("hooks", "--help")
    gateway_code, gateway_out, _ = hermes("gateway", "status")
    profile_code, profile_out, _ = hermes("profile", "list")

    goals_doc = HERMES_AGENT / "website/docs/user-guide/features/goals.md"
    slash_doc = HERMES_AGENT / "website/docs/reference/slash-commands.md"
    hooks_doc = HERMES_AGENT / "website/docs/user-guide/features/hooks.md"
    goal_available = (
        goals_doc.exists()
        and "/goal" in goals_doc.read_text(errors="ignore")
        and slash_doc.exists()
        and "/goal" in slash_doc.read_text(errors="ignore")
    )
    kanban_available = (
        kanban_code == 0 and "Durable SQLite-backed task board" in kanban_out
    )
    hooks_available = (
        hooks_code == 0
        and "shell-script hooks" in hooks_out
        and hooks_doc.exists()
        and "pre_tool_call" in hooks_doc.read_text(errors="ignore")
    )
    worker_lanes_available = (
        Path(
            "/Users/preston/.hermes/skills/devops/kanban-worker/SKILL.md"
        ).exists()
        and Path(
            "/Users/preston/.hermes/skills/devops/kanban-orchestrator/SKILL.md"
        ).exists()
        and profile_code == 0
        and "default" in profile_out
    )
    codex_runtime_available = os.access(
        "/Users/preston/.local/bin/hermes-codex-audit",
        os.X_OK,
    )
    gateway_loaded = gateway_code == 0 and "Gateway service is loaded" in gateway_out
    parity = custom_skill_parity()

    checks = {
        "goal_available": goal_available,
        "kanban_available": kanban_available,
        "hooks_available": hooks_available,
        "worker_lanes_available": worker_lanes_available,
        "codex_runtime_available": codex_runtime_available,
        "gateway_loaded": gateway_loaded,
    }
    for key, ok in checks.items():
        if not ok:
            blockers.append(f"{key} is false")
    for label, ok in parity.items():
        if not ok:
            blockers.append(f"custom asset parity failed: {label}")

    if version_code != 0:
        warnings.append(f"hermes version failed: {version_err.strip()}")

    return {
        "schema": SCHEMA,
        "hermes_version": first_line(version_out),
        "hermes_agent_head": git_head(HERMES_AGENT),
        **checks,
        "custom_skill_parity": parity,
        "native_control_plane_ready": not blockers,
        "blockers": blockers,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("json",), default="json")
    args = parser.parse_args()
    _ = args

    payload = build_payload()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["native_control_plane_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
