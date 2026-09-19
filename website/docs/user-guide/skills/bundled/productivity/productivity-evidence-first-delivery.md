---
title: "Evidence First Delivery — Deliver multi-step work with verified outcomes"
sidebar_label: "Evidence First Delivery"
description: "Deliver multi-step work with verified outcomes"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Evidence First Delivery

Deliver multi-step work with verified outcomes.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/productivity/evidence-first-delivery` |
| Version | `0.1.0` |
| Author | Ryan (ryankr), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Workflow`, `Verification`, `Delivery`, `Planning` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Evidence-First Delivery Skill

Use a small delivery lifecycle for work that needs more than a direct answer. It keeps planning, execution, and verification distinct without replacing the specialized skill that performs the domain work.

## When to Use

- A request has multiple steps, a deliverable, a code or configuration change, delegated work, or an external action.
- The user needs a reliable completion claim, decision, report, implementation, or verification.
- The work can be split into independent investigation, build, review, or validation lanes.

Don't use for: a simple factual answer or a short transformation with no artifact or external effect.

## Prerequisites

- Identify the requested deliverable, constraints, side effects, and completion criteria before acting.
- Use the domain-specific skill for the work itself; use `delegate_task` only for independent reasoning-heavy lanes.

## How to Run

Apply this lifecycle while using the relevant Hermes tools:

- Use `read_file`, `search_files`, `web_extract`, or `web_search` to gather evidence.
- Use `write_file` or `patch` only after the intended artifact scope is clear.
- Use `delegate_task` for non-overlapping lanes and retain each lane's artifact or test evidence.
- Use `browser_exec` or the relevant API/browser tool when the requested result is rendered or externally visible.

## Quick Reference

- **Planned:** scope and acceptance checks exist; relevant work has not run.
- **Executed, unverified:** a tool or executor reported completion without an independent check.
- **Verified:** the final artifact or target state was inspected and every named acceptance check has evidence.
- **Blocked or partial:** a specific prerequisite, failed check, or remaining scope prevents completion.

## Procedure

### 1. Frame the delivery contract

Extract explicit and implied deliverables, constraints, side effects, and acceptance checks. Map each requested result to one verification method or a stated limitation. Done when every named outcome has a checkable completion condition.

### 2. Choose the smallest execution shape

Use direct Hermes tools for mechanical work. Parallelize only independent reads, research, reviews, or non-overlapping implementation lanes; give each `delegate_task` lane exclusive scope and a required evidence output. Done when no lanes can overwrite the same artifact or make contradictory external changes.

### 3. Execute with durable evidence

Write artifacts to canonical locations and retain relevant command, test, API, or browser output. Treat a subagent report as unverified until its cited artifact, test result, URL, identifier, or external state is inspected. Done when every claimed completed action has concrete evidence.

### 4. Run the acceptance gate

Inspect the final artifact itself, not only source code or intermediate output. For external writes, read the exact target back. For code or UI work, run the repository's documented checks and inspect the requested API or rendered surface when applicable. Done when every deliverable is classified as verified, unverified, or blocked.

### 5. Deliver the result precisely

Lead with the decision or result. Separate completed-and-verified work, completed-but-unverified work, and blocked or unavailable work; include concrete evidence paths, test results, URLs, or identifiers. Done when the final answer makes no completion claim unsupported by observed evidence.

## Pitfalls

- A green build does not prove a requested UI, document, API payload, or deployment behavior is present.
- A successful API response does not prove durable external state; read the target back.
- Do not make a plan, a plausible output, or an executor self-report sound like verification.
- Avoid over-delegation when coordination cost or conflicting assumptions exceed the value of parallelism.
- Stop in-flight work immediately when the user issues a stop instruction; do not resume without a new explicit request.

## Verification

- [ ] Every requested deliverable is marked verified, unverified, or blocked.
- [ ] Each verified claim has an inspected artifact or observed tool result.
- [ ] External writes were checked with a read-after-write operation.
- [ ] Final reporting distinguishes facts, assumptions, risks, and recommendations where relevant.
