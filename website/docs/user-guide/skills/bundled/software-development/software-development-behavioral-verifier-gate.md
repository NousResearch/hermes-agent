---
title: "Behavioral Verifier Gate"
sidebar_label: "Behavioral Verifier Gate"
description: "Reject weak verifiers that only prove records/artifacts exist and replace them with independent behavioral checks"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Behavioral Verifier Gate

Reject weak verifiers that only prove records/artifacts exist and replace them with independent behavioral checks. Use when reviewing tests, evals, closers, verifiers, loop closure, production gates, or claims that something is verified.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/software-development/behavioral-verifier-gate` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Tags | `verification`, `tests`, `evals`, `anti-theater`, `autonomy` |
| Related skills | [`autonomy-verb-proof-gate`](/docs/user-guide/skills/bundled/software-development/software-development-autonomy-verb-proof-gate), [`anti-theater-regression-tests`](/docs/user-guide/skills/bundled/software-development/software-development-anti-theater-regression-tests) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Behavioral Verifier Gate

## Purpose

Stop record-existence checks from masquerading as proof. A verifier must observe the effect of an action on behavior or state outside the artifact that claims success.

## Forbidden as closure verifiers

These may be audit checks, but they are not closure proof:

```text
*_record_exists
*_report_exists
*_proposal_exists
*_registry_entry_exists
*_dashboard_contains
status == pass
label == verified
file exists
```

## Required verifier properties

A valid verifier has:

- target behavior or state being checked;
- source independent from the artifact under test;
- before/after or expected/observed comparison;
- falsifiable failure condition;
- replay or live reproduction command;
- no dependence on self-reported status as the only evidence.

## Rewrite pattern

Bad:

```text
learning_update_record_exists
```

Good:

```text
Replay the same failure class before/after the learning update and show the future context/action differs in the expected direction.
```

Bad:

```text
capability_registry_entry_exists
```

Good:

```text
Invoke the capability entrypoint on a held-out case and verify the declared effect occurred with no boundary violation.
```

## Review output

```text
Claim being verified:
Current verifier:
Verdict: behavioral | artifact-only | self-referential
Independent source needed:
Replacement verifier:
Regression test:
```

## Pitfall

Do not delete audit records. Reclassify them correctly: audit evidence supports traceability, not behavioral proof.
