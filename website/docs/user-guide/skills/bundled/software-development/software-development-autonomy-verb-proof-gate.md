---
title: "Autonomy Verb Proof Gate"
sidebar_label: "Autonomy Verb Proof Gate"
description: "Gate autonomy/self-healing/self-improvement claims by requiring real behavior-changing verbs before accepting artifacts"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Autonomy Verb Proof Gate

Gate autonomy/self-healing/self-improvement claims by requiring real behavior-changing verbs before accepting artifacts. Use when a system claims autonomy, agentic behavior, self-healing, self-improvement, loop closure, or production readiness.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/software-development/autonomy-verb-proof-gate` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Tags | `autonomy`, `verification`, `anti-theater`, `control-loops` |
| Related skills | [`karpathy-autonomy-persona`](/docs/user-guide/skills/bundled/software-development/software-development-karpathy-autonomy-persona), [`behavioral-verifier-gate`](/docs/user-guide/skills/bundled/software-development/software-development-behavioral-verifier-gate) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Autonomy Verb Proof Gate

## Purpose

Prevent autonomy theater by refusing to treat nouns as verbs. A system proves autonomy only through behavior-changing action under constraints, not through dashboards, traces, registries, proposals, or labels.

## Trigger

Use when reviewing or building anything described as:

- autonomous;
- self-healing;
- self-improving;
- agentic;
- closed loop;
- production control plane;
- kernel proving a loop.

## Required proof chain

Before accepting the claim, identify each concrete link:

```text
real sensed condition
-> policy/objective/authority decision
-> bounded action selection
-> actuator executes in environment
-> independent verifier observes changed behavior
-> recurrence or burden delta improves on later run/replay
```

If any link is missing, say the claim is not proven.

## Reject substitutions

Do not accept:

- trace written -> behavior changed;
- proposal created -> decision made;
- registry entry -> capability deployed;
- dashboard green -> system healthy;
- cron executed -> autonomy;
- real adapter observed -> control loop acted;
- tests passed -> real behavior improved.

## Output format

```text
Claim:
Verdict: proven | unproven | artifact-only | telemetry-only
Missing verb link(s):
Smallest proof experiment:
Required verifier:
```

## Common pitfall

Do not propose a smaller artifact patch as if it satisfies a behavioral pass condition. If you recommend an interim containment step, label exactly which original requirements remain unsatisfied.
