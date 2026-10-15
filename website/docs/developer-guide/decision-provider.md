---
title: Decision provider and Jev shadow layer
sidebar_position: 18
---

# Decision provider and Jev shadow layer

Hermes has a small provider-neutral decision seam in `agent/decision_provider.py`.
It is intentionally local and dependency-free. A Jev adapter is optional: configure
an audited JSON subprocess only when its source and privacy behavior have been
reviewed. Hermes does not install `hermes-jev-skills`, Winnow, JevCal, or another
third-party package as part of this layer.

## Default behavior

The default configuration is:

```yaml
decision_provider:
  mode: shadow
  provider: hermes-bridge-jev
  adapter_path: ~/.hermes/workspaces/hermes-bridge/scripts/hermes_jev_adapter.cjs
  model: typesafe/jev-1.13
  adapter_command: []
```

The consumers are `pre_action`, `completion`, `loop`, `context_sieve`,
`failure_classification`, `retry`, `specialist`, and `notification`. Each has
its own confidence threshold, timeout, and `fail_mode`. The defaults are
shadow-only: a provider result is logged as a candidate and never authorizes,
blocks, publishes, spends money, changes permissions, or replaces a Hermes
guardrail decision.

To connect a local adapter, use an explicit argument vector rather than a shell
string:

```yaml
decision_provider:
  mode: shadow
  provider: hermes-bridge-jev
  adapter_path: ~/.hermes/workspaces/hermes-bridge/scripts/hermes_jev_adapter.cjs
  model: typesafe/jev-1.13
  adapter_command: []       # optional explicit argv; overrides adapter_path
```

The adapter receives one JSON object on stdin containing `consumer`, `request_id`,
the privacy-screened `payload`, and the non-verbatim hard-constraint ledger. The
existing `hermes-bridge/scripts/hermes_jev_adapter.cjs` dispatches these requests
through one `HermesJevController` and returns normalized JSON such as
`{"decision":"retry_same","confidence":0.91}`.

## Safety and privacy boundary

Before any provider call, Hermes deterministically records user constraints and
approval boundaries without copying their text into the provider request. Secret
patterns, secret-bearing paths, private keys, and private file/code payloads fail
closed and are not sent to the adapter. Pre-action arguments are projected to
types, sizes, and hashes. The local replay log stores hashes,
metadata, decisions, thresholds, and error codes—not raw tool output, arguments,
or provider rationale.

The existing scope checks, destructive-command approvals, checkpoints, financial
and permission approvals, tool-loop guardrails, and stop gates remain the owners
of enforcement. This layer observes them and includes their state in shadow
records; it does not duplicate or bypass them.

## Context retention and replay

For large tool outputs, the `context_sieve` consumer produces a shadow candidate
with line ranges and explicit preservation markers for exact errors, active file
paths, unresolved failures, approval evidence, and non-reproducible evidence.
Only hashes and retention metadata leave Hermes; sampled output remains local.
The existing tool-result spillover, incident checkpoint, child-session rotation,
and context compressor still own the actual transcript. No context is dropped by
this first implementation.

Records are written to `$HERMES_HOME/decision_logs/replay.jsonl` by default with
0600 permissions. Add labeled `outcome` values offline and calibrate thresholds
per consumer:

```python
from agent.decision_provider import calibrate_thresholds, replay_records

records = replay_records("/path/to/replay.jsonl")
print(calibrate_thresholds(records))
```

This is calibration scaffolding, not an automatic threshold rewrite. Keep the
layer in shadow mode until a replay review demonstrates that a consumer’s labels,
privacy behavior, and failure policy are reliable.
