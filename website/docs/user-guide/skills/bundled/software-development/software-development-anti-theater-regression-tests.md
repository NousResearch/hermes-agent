---
title: "Anti Theater Regression Tests"
sidebar_label: "Anti Theater Regression Tests"
description: "Design regression tests that catch autonomy theater including artifact-only proof, stale projections, fake deployment, missing actuators, and self-referentia..."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Anti Theater Regression Tests

Design regression tests that catch autonomy theater including artifact-only proof, stale projections, fake deployment, missing actuators, and self-referential verification. Use when adding tests/evals for autonomous agents, control planes, self-healing, self-improvement, or production readiness.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/software-development/anti-theater-regression-tests` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Tags | `testing`, `evals`, `anti-theater`, `autonomy`, `regression` |
| Related skills | [`behavioral-verifier-gate`](/docs/user-guide/skills/bundled/software-development/software-development-behavioral-verifier-gate), [`run-scoped-causality-gate`](/docs/user-guide/skills/bundled/software-development/software-development-run-scoped-causality-gate) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Anti-Theater Regression Tests

## Purpose

Write tests that fail when the system only looks autonomous. The tests should falsify common theater surfaces before any positive claim is accepted.

## Required negative tests

Add focused tests for these failure modes when relevant:

1. Registry says `deployed`, but no actuator/entrypoint exists -> fail.
2. `verified` means only record exists -> fail.
3. Projection exists, but high-water mark is stale -> fail.
4. Required event types exist only in previous runs -> fail.
5. `synthetic: false`, but source kind is fixture/manual outside test mode -> fail.
6. `agent_routable` item has no consumer lifecycle -> fail.
7. HIL approval exists in YAML/chat/dashboard, but not canonical event store -> fail.
8. `watch_clean` without same-run policy/action/verifier chain -> fail.
9. Self-healing only refreshes dashboard/projection -> fail if claimed as operational repair.
10. Self-improvement writes proposal/spec only -> fail if claimed as capability improvement.

## Positive test requirement

Pair every negative test with one minimal positive kernel:

```text
real/fixture-labeled input -> bounded action -> independent verifier -> measured behavior/state delta
```

For unit tests, fixtures are allowed only if named as fixtures and unable to satisfy production acceptance.

## Test naming

Use names that encode the lie being prevented:

```text
test_deployed_capability_requires_actuator
test_watch_clean_requires_same_run_verifier_chain
test_yaml_hil_decision_does_not_grant_authority
test_record_exists_is_not_behavioral_verifier
```

## Review checklist

- [ ] Does the test fail on artifact-only implementation?
- [ ] Does it require an independent verifier?
- [ ] Does it separate unit fixtures from production acceptance?
- [ ] Does it prove current-run causality when evaluating runs?
- [ ] Does it prevent the exact theater surface seen before?
