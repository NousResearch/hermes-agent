---
title: "Kanban Operations"
sidebar_label: "Kanban Operations"
description: "Hermes Kanban umbrella: orchestrator decomposition, worker execution, task claiming, summaries, dependencies, and board hygiene"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Kanban Operations

Hermes Kanban umbrella: orchestrator decomposition, worker execution, task claiming, summaries, dependencies, and board hygiene.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/devops/kanban-operations` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Kanban Operations

Use this skill for Hermes Kanban workflows where work is decomposed into board tasks and executed by specialized workers.

## Orchestrator workflow

1. Understand the goal and success criteria.
2. Sketch the dependency graph before creating cards.
3. Create small, independently verifiable tasks with clear inputs, outputs, owners/profiles, and blockers.
4. Link dependencies explicitly and keep board state current.
5. Integrate worker results only after checking returned artifacts or evidence.

## Worker workflow

1. Claim only cards you can actually see and identify.
2. Read the card, dependencies, and attached context before acting.
3. Work in the assigned workspace; preserve tenant/profile isolation.
4. Update status with concise progress and blockers.
5. Finish with a summary that includes changed paths, commands run, evidence, and follow-up needs.

## Anti-patterns

- Creating huge cards that require another decomposition pass.
- Claiming cards by guessed IDs.
- Reporting success without a test/log/artifact handle.
- Mixing unrelated tasks into one worker run.
- Losing dependency information in free-form comments.

## Good card shape

- Title: imperative and scoped.
- Body: context, acceptance criteria, explicit non-goals, verification command, expected artifact.
- Metadata: repo/workdir, branch policy, profile/tool requirements, dependencies.
## Support files

- `references/absorbed-skills.md` — list of original skill packages consolidated into this umbrella and where to recover full archived content.
