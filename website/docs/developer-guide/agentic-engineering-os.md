---
title: Agentic Engineering OS
sidebar_position: 2
---

# Agentic Engineering OS

Hermes supports a lightweight operating system for AI-assisted engineering. It separates context by how often it is needed:

```text
Permanent repo memory -> AGENTS.md
On-demand workflows   -> skills/
Fresh upstream facts   -> .references/
Quality control        -> review loops
```

This prevents a common failure mode: stuffing every rule, workflow, and reference into one giant prompt. Agents should load the right layer at the right time.

## AGENTS.md: always-on rules

`AGENTS.md` is for context every coding agent needs on every task:

- architecture map
- test commands
- important entry points
- security constraints
- branch and review discipline
- repo-specific pitfalls

Keep it compact. Do not add long optional procedures to `AGENTS.md`; make a skill instead.

## Skills: on-demand procedures

Skills live under `skills/<category>/<name>/SKILL.md`. Their descriptions are indexed up front, while the full body is loaded only when relevant.

Hermes ships three software-development skills for the agentic engineering loop:

- `code-structure` — restructure messy AI-generated code into clean boundaries without changing behavior.
- `gpt-loop` — run implement/test/review/fix/re-review loops until blockers are resolved.
- `code-simplifier` — reduce accidental complexity while preserving behavior and safety checks.

These complement existing skills such as `writing-plans`, `test-driven-development`, `subagent-driven-development`, and `requesting-code-review`.

## .references: local source/docs snapshots

Models may not know the latest behavior of fast-moving APIs or libraries. Store local snapshots in the repo root `.references/` directory:

```text
.references/openai-agents-sdk/
.references/anthropic-docs/
.references/supabase-mcp/
.references/telegram-bot-api/
```

Rules:

- `.references/*` is ignored by Git.
- Only `.references/README.md` and `.references/.gitkeep` are tracked.
- Never store secrets, auth files, logs, or live `.env` files there.
- Ask the agent to search the specific reference folder it needs instead of loading the whole tree.

This is separate from skill-local `references/` folders. Skill references are committed support docs for a skill; root `.references/` is an ignored workspace for fresh external material.

## Review loops: quality control

For non-trivial changes, do not ship the first implementation pass. Use the loop:

```text
implement -> run checks -> review -> fix feedback -> re-run checks -> re-review -> ship
```

Review feedback can come from:

- a Hermes reviewer subagent
- CI/test/lint output
- GitHub PR review comments
- external review tools already configured in the environment
- a human reviewer
- a local checklist when no reviewer is available

Use `gpt-loop` while iterating and `requesting-code-review` as the final pre-commit gate.

## Typical workflow

1. Read `AGENTS.md` for repo constants.
2. Load a skill when a task matches its trigger:
   - structure issue -> `code-structure`
   - review/iteration issue -> `gpt-loop`
   - complexity issue -> `code-simplifier`
3. Search `.references/<tool-or-api>/` only when current upstream facts are needed.
4. Run targeted validation.
5. Review and fix feedback.
6. Commit only after the diff is scoped, validated, and secret-clean.

## Maintenance checklist

When extending this system:

- Keep `AGENTS.md` short and permanent.
- Put class-level procedures in skills.
- Put helper files under skill-local `references/`, `templates/`, or `scripts/`.
- Keep downloaded upstream references out of Git.
- Add or update tests for new built-in skills.
