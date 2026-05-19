---
title: Context Hygiene Contracts
---

# Context Hygiene Contracts

Hermes keeps five context layers separate so useful context compounds without turning the prompt into landfill.

This page is an operator contract, not a personality rewrite. Do not use it to bulk-edit `SOUL.md` or delete skills/memory automatically.

## Layers

| Layer | Source | Purpose | Must not contain |
| --- | --- | --- | --- |
| SOUL | `SOUL.md` | Identity, voice, autonomy, verification, completion standards | transient task status, commit IDs, issue numbers, project-specific TODOs |
| Project context | `.hermes.md`, `HERMES.md`, `AGENTS.md`, `agents.md`, `CLAUDE.md`, `claude.md`, `.cursorrules`, `.cursor/rules/*.mdc` under/above the working directory according to prompt-builder priority | Repo map, local commands, project-specific rules | global personality policy, durable personal preferences |
| Skills | `skills/**/SKILL.md` | Repeatable procedures with frontmatter, steps, pitfalls, verification | one-off session logs, stale task progress, secrets |
| Memory/profile | `MEMORY.md`, `USER.md`, memory providers | Stable facts and preferences that should survive sessions | PR numbers, commit SHAs, “fixed X”, phase status, raw credentials |
| Sessions/traces | session DB/files and `harness/*.jsonl` | What happened, replay evidence, trace/failure metadata | raw prompts, raw tool args, raw tool errors, raw secret-bearing output |

## Control-plane audit

Tier 4 adds a metadata-only context hygiene audit:

- Python: `agent.context_hygiene.audit_context_hygiene()`
- Harness facade: `HermesHarness().control_plane.context_hygiene()`
- Dashboard/API: `GET /api/harness/context-hygiene`

The audit returns counts, hashes, layer presence, and issue codes only. It does not return raw memory rows, prompt text, skill bodies, session content, or local filesystem paths.

Issue codes are intentionally boring and machine-readable:

- `soul_missing`
- `project_context_missing`
- `skill_frontmatter_incomplete`
- `skill_duplicate_names`
- `memory_contains_task_progress`
- `memory_contains_procedure`

These are prompts for a human/operator to inspect the source layer. They are not permission for the agent to rewrite identity, delete skills, or erase memory without explicit approval.

## Operating rules

1. Put identity/policy in SOUL.
2. Put repo-specific rules in AGENTS/project context.
3. Put reusable workflows in skills.
4. Put stable facts/preferences in memory.
5. Put historical execution evidence in sessions/traces.
6. If a fact will be stale in a week, it does not belong in memory.
7. If a procedure has steps, commands, or pitfalls, it belongs in a skill.
8. If a trace needs to explain failure, persist structure/counts/hashes, not raw private content.

## Verification

Use focused tests when changing this surface:

```bash
scripts/run_tests.sh tests/agent/test_context_hygiene.py \
  tests/agent/test_hermes_harness.py::test_control_plane_harness_exposes_context_hygiene \
  tests/hermes_cli/test_web_server.py::TestWebServerEndpoints::test_harness_trace_replay_endpoints_are_content_safe -q
```

For broader harness changes, also run the harness control-plane tests and the dashboard harness endpoint tests.
