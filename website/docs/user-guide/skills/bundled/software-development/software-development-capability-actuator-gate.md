---
title: "Capability Actuator Gate"
sidebar_label: "Capability Actuator Gate"
description: "Prevent capability registries or specs from being treated as deployed capabilities without executable actuators and effect verification"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Capability Actuator Gate

Prevent capability registries or specs from being treated as deployed capabilities without executable actuators and effect verification. Use when reviewing capability registries, self-improvement outputs, plugins, tools, shadow capabilities, or deployment claims.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/software-development/capability-actuator-gate` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Tags | `capabilities`, `deployment`, `actuators`, `self-improvement`, `anti-theater` |
| Related skills | [`behavioral-verifier-gate`](/docs/user-guide/skills/bundled/software-development/software-development-behavioral-verifier-gate), [`autonomy-verb-proof-gate`](/docs/user-guide/skills/bundled/software-development/software-development-autonomy-verb-proof-gate) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Capability Actuator Gate

## Purpose

A capability is not deployed because it appears in a registry. Deployment requires an invocation path and verified effect.

## Deployment minimum

Do not call a capability deployed unless it has:

- executable entrypoint or callable tool;
- declared input/output contract;
- authority boundary and effect surface;
- run command or invocation example;
- independent effect verifier;
- rollback/disable path;
- at least one successful run trace on a non-trivial case;
- tests for boundary violations and failure modes.

## State vocabulary

Use precise states:

- `proposed`: idea/spec exists;
- `registered`: metadata exists;
- `implemented`: code/entrypoint exists;
- `invocable`: caller can execute it;
- `verified_shadow`: effect proven in bounded/shadow mode;
- `deployed`: used by the real runtime under policy;
- `retired`: disabled with rollback record.

A registry row alone is at most `registered`.

## Review output

```text
Capability claim:
Registry/spec present: yes/no
Actuator present: yes/no
Invocation path:
Effect verifier:
Run trace:
Rollback path:
Correct state:
```

## Anti-theater check

If the declared effect surface only writes registry/evaluation files, the capability is not built. It is documentation about a possible capability.
