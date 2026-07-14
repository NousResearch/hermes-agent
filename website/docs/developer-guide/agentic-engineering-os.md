---
title: Agentic Engineering OS
sidebar_position: 2
description: "A lightweight operating model for reliable agent-assisted engineering in Hermes without growing the model tool schema."
---

# Agentic Engineering OS

Agentic Engineering OS is a lightweight operating model for humans and AI
agents contributing to Hermes. The goal is simple: make agent-assisted work more
reliable while preserving Hermes' narrow-waist design.

It separates context by **half-life**:

```text
Permanent repo invariants -> AGENTS.md
Reusable procedures       -> skills/
Fresh upstream facts      -> .references/
Quality control           -> review loops
```

## Value proposition

Hermes provides a repeatable quality system for AI-assisted
engineering with minimal runtime footprint:

- **Smaller permanent prompts:** `AGENTS.md` stays focused on durable repo
  invariants instead of absorbing every optional workflow.
- **Reusable procedures:** workflows become skills that load only when their
  descriptions match the task.
- **Current external knowledge:** local source/docs snapshots can be searched
  from `.references/` without committing vendor trees or bloating prompts.
- **Higher-quality contributions:** non-trivial work follows an explicit
  implement → validate → review → fix → revalidate loop instead of shipping the
  first draft.
- **No core-surface expansion:** this adds no model tools, no runtime hooks, no
  provider changes, and no core code paths. The prompt impact is limited to a
  small cached repo/skill-index footprint.

In short: it improves contributor reliability at the documentation/skills layer,
which matches Hermes' preference for growing capability at the edges instead of
widening the core.

## The four layers

### 1. `AGENTS.md`: always-on repo memory

`AGENTS.md` is for context every coding agent needs on every task:

- architecture map and load-bearing files
- test commands and validation expectations
- branch, review, and contribution discipline
- security constraints
- Hermes-specific pitfalls and invariants

Keep it compact. Do not add long optional procedures to `AGENTS.md`; make a
skill instead.

### 2. Skills: on-demand procedures

Skills live under `skills/<category>/<name>/SKILL.md`. Hermes indexes their
frontmatter descriptions up front, while the full body is loaded only when the
skill is relevant.

Hermes uses these software-development skills for the agentic
engineering loop:

- `code-structure` — restructure messy AI-generated code into clean boundaries
  without changing behavior.
- `review-loop` — run validate/review/fix/revalidate loops until blockers are
  resolved.
- `simplify-code` — the existing parallel cleanup pass for reducing accidental
  complexity while preserving behavior and safety checks.

These complement existing skills such as `writing-plans`,
`test-driven-development`, `subagent-driven-development`, and
`requesting-code-review`.

### 3. `.references/`: local source/docs snapshots

Models may not know the latest behavior of fast-moving APIs or libraries. Store
local snapshots in the repo root `.references/` directory:

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
- Search the specific reference folder needed for the task instead of loading a
  whole reference tree into context.

This is separate from skill-local `references/` folders. Skill references are
committed support docs for a skill; root `.references/` is an ignored workspace
for fresh external material.

### 4. Review loops: quality control

For non-trivial changes, do not ship the first implementation pass. Use the
loop:

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

Use `review-loop` while iterating and `requesting-code-review` as the final
pre-commit gate.

## Typical workflow

1. Read `AGENTS.md` for repo constants.
2. Load a skill when a task matches its trigger:
   - structure issue -> `code-structure`
   - review/iteration issue -> `review-loop`
   - complexity issue -> `simplify-code`
3. Search `.references/<tool-or-api>/` only when current upstream facts are
   needed.
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
