---
title: "Agent Routable Consumer Gate"
sidebar_label: "Agent Routable Consumer Gate"
description: "Require agent-routable/proposal/backlog labels to have real consumers that claim, execute, verify, and retire work"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Agent Routable Consumer Gate

Require agent-routable/proposal/backlog labels to have real consumers that claim, execute, verify, and retire work. Use when reviewing attention queues, improvement backlogs, proposal_required, agent_routable, repair candidates, or autonomous work queues.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/software-development/agent-routable-consumer-gate` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Tags | `queues`, `agents`, `work-routing`, `autonomy`, `anti-theater` |
| Related skills | [`autonomy-verb-proof-gate`](/docs/user-guide/skills/bundled/software-development/software-development-autonomy-verb-proof-gate), [`behavioral-verifier-gate`](/docs/user-guide/skills/bundled/software-development/software-development-behavioral-verifier-gate) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Agent-Routable Consumer Gate

## Purpose

Stop queue labels from masquerading as autonomous handling. `agent_routable` means nothing unless an agent/runtime can consume the item and close it with evidence.

## Required consumer contract

For each routable item class, identify:

- consumer name/entrypoint;
- claim/lock mechanism;
- input schema;
- authority policy;
- action executor;
- verifier;
- retry/rollback behavior;
- terminal states;
- retirement/update record;
- monitoring/eval that detects stuck items.

## Required lifecycle

```text
queued
-> claimed
-> policy_checked
-> executed | deferred | rejected
-> verified
-> retired | updated | escalated
```

If the lifecycle is absent, the item is not agent-routable; it is merely labeled.

## Failure conditions

Fail when:

- queue has open items but no consumer;
- consumer exists but cannot execute the recommended next action;
- verifier is artifact-only;
- item remains open across runs without escalation;
- dashboard hides backlog because no new proposals were created this run.

## Review output

```text
Queue/item:
Claimed routing:
Consumer present:
Lifecycle state:
Verifier:
Stuck/retired/escalated:
Verdict: routable | label-only | stuck | needs HIL | invalid
```

## Pitfall

Do not count “0 new proposals this run” as an empty backlog. Existing unresolved items remain open work.
