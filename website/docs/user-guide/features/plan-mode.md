---
sidebar_position: 17
title: "Plan Mode"
description: "Code-enforced plan-then-execute: Hermes may plan and inspect but cannot mutate until you approve the plan."
---

# Plan Mode (`/plan`)

`/plan` puts Hermes into a **code-enforced** planning state: it can read, search, and write a plan document, but every mutating tool — `terminal`, `write_file`/`patch` (outside the plan file), `patch`, code execution, delegation, memory writes, cron, browser actions, and more — is **blocked** until you explicitly approve the plan. This is stronger than asking the model nicely to "just plan": the restriction is enforced in the tool-dispatch path, so the model *cannot* mutate even if it tries.

It complements the prompt-level [`plan` skill](../skills/) (which teaches Hermes *how* to write a good plan). Plan mode is the guardrail *underneath* that skill — the skill writes the plan; plan mode makes sure nothing else happens until you say go.

## When to use it

- You want a plan reviewed **before** any files change or commands run.
- You're handing Hermes a risky or large change and want a checkpoint.
- You want "plan → approve → execute" as a hard workflow, not a suggestion.

## Quick start

```
/plan
```

What happens:

1. **Plan mode on** — `📝 Plan mode on. I'll plan only — mutating tools are blocked until you approve.`
2. **Hermes plans** — it inspects the repo with read-only tools and writes a markdown plan under `.hermes/plans/`.
3. **Approval prompt** — when the plan is ready Hermes calls `plan_ready`, which asks you **Approve** / **Keep planning** (rendered as buttons on chat platforms, an arrow-key picker in the CLI).
4. **You decide** —
   - **Approve** → the mutating-tool restriction lifts on the next turn and Hermes starts executing.
   - **Keep planning** → Hermes stays in plan mode and receives your feedback to refine the plan.

## Commands

| Command | What it does |
|---|---|
| `/plan` | Enter plan mode. Hermes plans only until you approve. |
| `/plan status` | Show the current plan-mode state (planning / awaiting approval / approved / off). |
| `/plan show` | Print the path of the plan file Hermes has written. |
| `/plan approve` | Approve the plan yourself — lifts the restriction on the next turn. |
| `/plan reject [feedback]` | Stay in plan mode; the optional feedback is passed back to Hermes. |
| `/plan exit` | Leave plan mode, **discarding** the pending plan. |

Works on the CLI and every gateway platform. On Slack, `/plan` is reached via `/hermes plan` (Slack caps apps at 50 native slash commands).

:::warning `/plan exit` never approves
`/plan exit` throws the pending plan away and unlocks mutations **without executing it** — it is *not* a shortcut for "approve". Use `/plan approve` (or the **Approve** button) when you want the plan to run.
:::

## How enforcement works

Plan mode is enforced in two independent layers, both **fail-closed**:

1. **Toolset restriction** — entering plan mode removes the mutating toolsets from the model's view for the session (they simply aren't in the tool schema the model sees), and exposes the `plan_ready` approval tool. The agent is rebuilt with this restricted toolset on the next turn.
2. **Dispatch guard** — before *any* tool executes, Hermes re-checks the persisted plan state. If the session is planning (or awaiting approval) and the tool is mutating, the call is blocked and a structured message is returned to the model instead of running. The only mutation allowed is writing the plan markdown itself, and only when the target path is under `.hermes/plans/`. If the plan state cannot be read, mutating tools are blocked (fail-closed) rather than allowed.

Plan state is stored per session and survives restarts, `/resume`, and context compaction, so the guardrail holds across the whole conversation — not just the turn you typed `/plan`.

## Approval via clarify

The approval prompt rides the same [clarify](./overview.md) machinery Hermes uses for any interactive question. On chat platforms the **Approve** / **Keep planning** options render as native buttons; in the CLI they're an arrow-key picker; and you can always type your own response, which Hermes treats as "keep planning" plus feedback.

## Config: always-on plan mode

Set the top-level `plan_mode` key in `config.yaml` to make **every new session** start in plan mode:

```yaml
plan_mode: always
```

New sessions begin in planning and require an explicit `/plan approve` (or the **Approve** button) before any mutation. An explicit `/plan exit` still lets an individual session leave plan mode.

## Relationship to the `plan` skill

- The **`plan` skill** is prompt-level craft: it tells Hermes to write a bite-sized, path-exact, testable plan under `.hermes/plans/` and not to implement.
- **Plan mode** is the enforcement: even if the model ignores the skill, the mutating tools are gone and the dispatch guard blocks them. Use them together — the skill for plan *quality*, plan mode for plan *safety*.
