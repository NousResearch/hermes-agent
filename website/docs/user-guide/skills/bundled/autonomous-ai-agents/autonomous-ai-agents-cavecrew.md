---
title: "Cavecrew — Delegate work with compact subagent outputs"
sidebar_label: "Cavecrew"
description: "Delegate work with compact subagent outputs"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Cavecrew

Delegate work with compact subagent outputs.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/autonomous-ai-agents/cavecrew` |
| Version | `1.0.0` |
| Author | Julius Brussee (@JuliusBrussee), adapted by Hermes Agent. |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `delegation`, `subagents`, `context`, `review` |
| Related skills | `autonomous-ai-agents/hermes-agent`, `software-development/subagent-driven-development` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Cavecrew Skill

Cavecrew is a delegation style for using subagents without flooding the main
conversation with long reports. It is adapted from the Caveman/Cavecrew Claude
Code workflow, but translated to Hermes concepts and tool names.

It does not add new model tools or install Claude hooks. It gives Hermes a
default decision policy for when to use compact investigation, edit, and review
delegation.

## When to Use

Use this skill when the work benefits from a focused subagent but the main
conversation must stay small.

Good fits:

- locating definitions, callers, tests, or config ownership
- making a surgical one or two file edit
- reviewing a diff for bugs and regressions
- splitting broad investigation into independent angles
- preserving context during long coding sessions

Avoid it when the user needs a narrative explanation, product thinking, or
large cross-cutting implementation. In those cases, keep the work in the main
thread or use the normal subagent-driven-development workflow.

## Prerequisites

No external Caveman installation is required.

Expected Hermes capabilities:

- `delegate_task` or equivalent subagent delegation
- `search_files` for locating code
- `read_file` for exact file context
- `patch` for bounded edits
- `terminal` for focused verification commands

## How to Run

When the task is broad, launch compact investigators in parallel with separate
questions:

- definitions and entry points
- callers and side effects
- existing tests and fixtures

When the task is bounded to one or two files, hand exact paths and constraints
to a compact builder.

When code has changed, ask a compact reviewer for findings only.

## Quick Reference

Investigator output:

```text
topic:
- path:line - `symbol` - short note
totals: counts.
```

Builder output:

```text
path:line-range - change under 10 words.
verified: re-read OK.
```

Reviewer output:

```text
path:line: severity: problem. fix.
totals: critical high medium questions.
```

Refusal tokens:

- `too-big.` split into smaller tasks
- `needs-confirm.` destructive or risky operation
- `ambiguous.` one blocking question
- `regressed.` verification failed inside the bounded scope

## Procedure

1. Classify the task before delegating.
2. Use investigator for "where is X" and "what touches Y" questions.
3. Use builder only when the write scope is already clear and at most two files.
4. Use reviewer after a diff exists, and ask for bugs before style.
5. Keep subagent output structured and terse.
6. Expand the terse result for the user only when the user needs prose.

## Pitfalls

- Do not send a vague feature request to the compact builder.
- Do not use compact review for architectural brainstorming.
- Do not ask investigators to read the whole repo.
- Do not hide destructive steps behind terse output.
- Do not prioritize compactness over safety warnings.

## Verification

For code edits, verify in the main thread after integrating subagent output:

- re-read touched files
- inspect `git diff`
- run the smallest relevant test or check
- report skipped tests explicitly

Attribution: adapted from `JuliusBrussee/caveman` Cavecrew concepts.
