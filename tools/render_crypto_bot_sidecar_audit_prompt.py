#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import crypto_bot_completion_gate as completion_gate


DEFAULT_REPO_ROOT = Path("/Users/preston/robinhood/crypto_bot")
DEFAULT_HERMES_ROOT = Path("/Users/preston/.hermes/hermes-agent")
SIDECAR_AUDIT_SCHEMA = "hermes.autonomy.crypto_bot_sidecar_audit.v1"


def resolve_ref(repo_root: Path, ref: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", ref],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError:
        return ref
    if proc.returncode != 0:
        return ref
    return proc.stdout.strip()


def render_prompt(
    *,
    repo_root: Path,
    base: str,
    head: str,
    branch: str,
    task_id: str | None = None,
    hermes_root: Path = DEFAULT_HERMES_ROOT,
) -> str:
    task_line = f"- Task ID: {task_id}\n" if task_id else ""
    base_full = resolve_ref(repo_root, base)
    head_full = resolve_ref(repo_root, head)
    canonical_range = f"{base_full}..{head_full}"
    docs_allowlist = {"paths": [], "patterns": [], "sources": []}
    if task_id:
        docs_allowlist = completion_gate.merge_allowlists(
            completion_gate.load_strategic_plan_allowlist(repo_root, task_id),
            completion_gate.parse_yaml_allowlist_section(
                hermes_root / "projects/crypto_bot/crypto_bot.project.yaml",
                task_id,
            ),
        )
    allowlist_lines = "\n".join(
        [
            *(f"  - path: {path}" for path in docs_allowlist["paths"]),
            *(f"  - pattern: {pattern}" for pattern in docs_allowlist["patterns"]),
        ]
    ) or "  - none"
    return f"""You are Codex in bounded audit-readonly mode for Hermes.

Repository: {repo_root}
{task_line}- Target branch: {branch}
- Base ref: {base}
- Base full SHA: {base_full}
- Expected final HEAD/ref: {head}
- Expected final full HEAD: {head_full}
- Canonical base/head range: {canonical_range}

Rules:
- Use only read-only local commands.
- Do not modify files, commit, push, create PRs, mutate Gitea, deploy, start
  services, run workflows or runners, inspect secrets, use ruff format, or touch
  broker/trading/financial/runtime surfaces.
- Do not infer pass from clean worktree. Do not accept prefilled conclusions.
- Do not run full verification, broad repository scans, test suites, app
  servers, workflows, runners, or any command outside the bounded list below.
- If a validator fails, a command times out, or the audit is incomplete, report
  FAIL or BLOCKED.

Required commands to run from the repository root, exactly in this audit:

```bash
cd {repo_root}
git status --short --branch
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git diff --name-only {canonical_range}
git diff --check {canonical_range}
```

Also perform a read-only blocked-surface path scan over the changed files from
`git diff --name-only {canonical_range}`.

Task docs allowlist for daemon/service wording only:
{allowlist_lines}

Treat these as BLOCKED regardless of docs wording or allowlists: `.gitea/workflows`,
secret-like paths (`.env`, token, key, private-key, cookie, credential), runtime
DBs, logs, broker/trading/financial/live-market/order/account/position/wallet
paths, deployment/GitOps paths, workflow/runner paths, and executable
service-start/runtime scripts or code paths. A safe docs path under
`docs/contracts/*.md`, `docs/development/*.md`, or `docs/architecture/*.md` may
mention daemon/service concepts only when it is listed above and no executable,
config, runtime, workflow, deploy, secret, trading, or broker path changed.

Report only this markdown structure. Do not leave any machine-evidence field
blank, and do not use prose-only substitutes for the machine evidence:

# Codex Sidecar Final Branch-Local Audit

## Commands Run
For each command: working directory, command, exit code, stdout/stderr summary.

## Machine Evidence
- Schema: {SIDECAR_AUDIT_SCHEMA}
- Branch observed: <branch from git rev-parse --abbrev-ref HEAD>
- Full HEAD observed: <40-char sha from git rev-parse HEAD>
- Base/head range audited: {canonical_range}
- Changed files:
  - <path from git diff --name-only, or none>
- Worktree status: <clean or dirty, from git status --short --branch>
- git diff --check exit code: <integer exit code>
- Blocked-surface scan: <PASS or BLOCKED, with basis; include allowlisted docs
  basis when relevant>
- Final conclusion: <PASS|FAIL|BLOCKED>

## Notes
Briefly explain the conclusion. The final conclusion field above must contain
exactly one of PASS, FAIL, or BLOCKED.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--hermes-root", type=Path, default=DEFAULT_HERMES_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompt = render_prompt(
        repo_root=args.repo_root,
        base=args.base,
        head=args.head,
        branch=args.branch,
        task_id=args.task_id,
        hermes_root=args.hermes_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
