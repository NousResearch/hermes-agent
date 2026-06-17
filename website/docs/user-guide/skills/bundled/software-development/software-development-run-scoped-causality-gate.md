---
title: "Run Scoped Causality Gate"
sidebar_label: "Run Scoped Causality Gate"
description: "Validate production autonomy runs by requiring same-run causal event chains instead of historical event presence"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Run Scoped Causality Gate

Validate production autonomy runs by requiring same-run causal event chains instead of historical event presence. Use when reviewing event stores, production evals, run logs, dashboards, watch_clean/green status, or source-truth projections.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/software-development/run-scoped-causality-gate` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Tags | `event-store`, `evals`, `causality`, `production`, `autonomy` |
| Related skills | [`autonomy-verb-proof-gate`](/docs/user-guide/skills/bundled/software-development/software-development-autonomy-verb-proof-gate), [`behavioral-verifier-gate`](/docs/user-guide/skills/bundled/software-development/software-development-behavioral-verifier-gate) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Run-Scoped Causality Gate

## Purpose

Prevent production evals from passing because the log contains the right labels somewhere in history. Acceptance must be scoped to one run and prove causal sequence.

## Required run fields

Every production event used for acceptance needs:

- `run_id`;
- monotonic timestamp or sequence;
- event type;
- source/adapter;
- payload schema version;
- causal parent or prior event reference where applicable.

## Required same-run chain

For autonomy claims, require this chain in the same `run_id`:

```text
sensor.*
-> policy.decision_recorded
-> closer.executed OR hil.decision_requested
-> verifier.completed
-> projection.rebuilt(high_water_event_id == latest_event_id)
```

For HIL flows add:

```text
hil.decision_received -> hil.decision_applied
```

## Failure conditions

Fail the eval when:

- required event types exist only in previous runs;
- projection high-water mark is stale;
- latest event is after projection rebuild;
- policy decision is missing;
- verifier event is missing;
- synthetic flag is false but source kind is fixture/manual without explicit test mode;
- dashboard status is derived from projection state alone.

## Review output

```text
Run id:
Claimed status:
Observed same-run chain:
Missing causal links:
Projection high-water check:
Verdict: valid run | stale/historical pass | telemetry-only
```

## Pitfall

A clean telemetry run is not an autonomy run. If the chain stops at observation/projection, call it telemetry-only.
