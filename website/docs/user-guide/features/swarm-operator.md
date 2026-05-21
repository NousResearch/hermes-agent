---
title: Jeeves Swarm Operator
---

# Jeeves Swarm Operator

Jeeves Swarm Operator is an experimental coordination layer for routing a user request through direct work, bounded child-agent fan-out, durable jobs, scripts, or execution pipes, then returning one parent-owned response.

It is **disabled by default**. The current implementation is safe shadow/dry-run infrastructure: state, routing records, status formatting, a no-op Honcho summary adapter, and verification planning helpers. It does not enable live interception, restart the gateway, or dispatch child agents unless explicitly wired later.

## Default configuration

The repository defaults are:

```yaml
swarm_operator:
  enabled: false
  dry_run: true
  max_children: 3
  persist_to_honcho: false
  honcho_summary_enabled: false
```

Do not enable this in a live profile until the shadow rollout checklist passes.

## Shadow state

When the gateway hook is explicitly enabled in dry-run mode, it records job state under the active Hermes profile:

- `${HERMES_HOME}/state/swarm_operator_state.json`
- `${HERMES_HOME}/state/swarm_operator_metrics.jsonl`

The store uses `get_hermes_home()` so profiles remain isolated.

## Status

`/status` can include a compact swarm status section when persisted swarm jobs exist. It shows active jobs, tasks, blockers, permission requests, and the latest audit event. Secret-looking metadata keys are redacted.

## Honcho summaries

Honcho persistence is parent-owned and inert unless enabled. The adapter writes only compact final summaries through an injected/optional writer and strips scratchpads, raw transcripts, reasoning, and secret-looking values. If Honcho is unavailable, persistence is a structured no-op.

## Verification planner

The verification helper builds deterministic checklists for direct/swarm/script/pipe work. It records explicit eval results on the `SwarmJob`; it does not run external tests or tools by itself.

## Rollout checklist

Before enabling locally:

1. Run focused tests:

   ```bash
   python -m pytest tests/swarm -q -o 'addopts='
   ```

2. Review shadow state after a test message.
3. Confirm normal gateway replies are unchanged.
4. Confirm no external sends, deploys, restarts, or destructive actions occur without explicit approval.
5. Only then consider setting `swarm_operator.enabled: true` in the profile config and restarting the gateway manually.
