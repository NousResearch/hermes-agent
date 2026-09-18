# CLI Shadow Observer

## Status

- Implementation: CLI completion-path integration
- Mode: opt-in `shadow` only
- Strict enforcement: not implemented
- Gateway/Discord/cron: not connected

The observer runs after a completed CLI turn and invokes an explicitly configured
local adapter. It does not alter the response, delivery path, session state, or
model request. The response body is passed transiently to the adapter process and
is not logged or persisted by the observer.

## Configuration

Add the following non-secret configuration to the active Hermes `config.yaml`.
The evaluator adapter path and Evidence path must be chosen explicitly by the
operator; they are not accepted from a user prompt.

```yaml
evaluation:
  shadow:
    enabled: true
    adapter_command:
      - python3
      - /absolute/path/to/agent-level-evaluator/scripts/hermes_shadow_adapter.py
    evidence_output: /absolute/path/to/isolated/operational-evidence.jsonl
    timeout_seconds: 10
    policy:
      required_patterns: []
      forbidden_patterns: []
```

The setting takes effect on a new CLI process. `enabled: false`, a missing
section, or an invalid configuration preserves normal CLI behavior and does not
invoke the adapter.

## Recorded boundary

The observer sends a `completed_response` event with:

- `trigger_origin=hermes`
- `execution_mode=shadow`
- CLI runtime surface
- opaque request ID
- hashed agent/evaluator configuration identities
- bounded trace counts

The adapter result is reduced to status metadata. Adapter stdout, stderr, and
response content are not displayed or written to the Hermes session transcript.
An adapter timeout, malformed response, or non-zero exit is returned as an
inconclusive observation while the normal CLI result remains available.

## Verification boundary

The following are separate claims:

- `observer implemented`: verified by focused tests
- `CLI runtime seam wired`: verified in the feature worktree, not installed runtime
- `real runtime observation`: one-shot CLI probe verified externally in the evaluator repository
- `continuous observation`: pending installation/activation in a controlled CLI process
- `operational evidence`: not yet accumulated over an observation window

Do not enable this configuration in the running/default Hermes checkout until the
feature worktree is reviewed, committed, and installed or otherwise selected
explicitly.
