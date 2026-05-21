---
title: "Kanban Adversarial Reviewer — Independent review gate for implementation tasks"
sidebar_label: "Kanban Adversarial Reviewer"
description: "Independent Kanban review gate for implementation tasks"
---

{/* This page mirrors the bundled skill source. Edit skills/devops/kanban-adversarial-reviewer/SKILL.md when updating the workflow. */}

# Kanban Adversarial Reviewer

Independent Kanban review gate for implementation tasks. Verifies target-workspace reality, git state, acceptance criteria, and scratch-vs-repo mismatches before downstream cards promote.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/devops/kanban-adversarial-reviewer` |
| Version | `1.0.0` |
| Platforms | linux, macos, windows |
| Tags | `kanban`, `review`, `verification`, `adversarial`, `code-review` |
| Related skills | [`kanban-worker`](/docs/user-guide/skills/bundled/devops/devops-kanban-worker), [`kanban-orchestrator`](/docs/user-guide/skills/bundled/devops/devops-kanban-orchestrator), `requesting-code-review` |

## Reference: full SKILL.md

# Kanban Adversarial Reviewer

Use this when a Kanban task is the dedicated review gate after an implementation card.

Your job is not to be agreeable. Your job is to verify reality.

## Core stance

- Do not trust worker summaries by default.
- Treat comments, summaries, and metadata as claims that require verification.
- PASS only when you have direct evidence from the target workspace / repo / deployment surface.
- If evidence is missing or contradictory, FAIL.

## Minimum review procedure

1. Run `kanban_show()` on the current review task and inspect:
   - parent implementation task summary/metadata
   - comments containing changed files, repo path, commit SHA, commands run, artifact URLs
   - prior failed reviews or corrections
2. Work in the actual task workspace (`$HERMES_KANBAN_WORKSPACE`).
3. Verify claimed artifacts exist in the target workspace itself.
   - Fail if files only exist in scratch dirs, temp dirs, or a different worktree.
   - Hello Hermes regression to catch: implementation claims success, but the main repo does not contain the changes.
4. Verify git reality:
   - current branch
   - `git status --short`
   - claimed commit SHA exists and matches when one was claimed
   - push/deploy claims have evidence, not just prose
5. Verify acceptance criteria directly:
   - rerun tests/build/lint/smoke checks when feasible
   - otherwise collect concrete evidence explaining why rerun was impossible
6. Verify no placeholder config/TODO scaffolding remains unless explicitly allowed by the task body.

## PASS / FAIL contract

PASS:
- Add a `kanban_comment` with concrete evidence.
- Include changed files actually observed, commands rerun, and exact outputs or statuses.
- Then `kanban_complete(summary=..., metadata=...)`.

FAIL:
- Add a `kanban_comment` with concrete findings and evidence.
- Prefer creating a correction task assigned to the implementer, with this review card as parent, so the audit trail stays explicit.
- If a human decision is required, `kanban_block(reason="...")` with the exact missing decision.

## Suggested evidence fields

```json
{
  "verdict": "pass or fail",
  "repo_path": "/abs/path",
  "observed_changed_files": ["..."],
  "git": {
    "branch": "...",
    "status_summary": "...",
    "commit": "..."
  },
  "checks_rerun": [
    {"command": "pytest -q", "exit_code": 0}
  ],
  "findings": []
}
```

## Failure examples

- Claimed file path does not exist in the target repo.
- Claimed commit SHA is absent or does not contain the stated change.
- Tests were claimed but cannot be reproduced.
- Deployment/live URL claim has no evidence.
- Scratch workspace contains artifacts that never landed in the real repo.

## Anti-patterns

- “Looks plausible” PASS.
- Trusting `changed_files` from metadata without checking the filesystem.
- Treating a green scratch run as proof of repo state.
- Completing the review gate without evidence in comments/metadata.
