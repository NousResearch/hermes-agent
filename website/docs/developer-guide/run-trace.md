---
title: Run trace
---

# Run trace

Hermes run traces are an opt-in, metadata-only observability record for one
agent turn. They are intentionally smaller and safer than full trajectories:
they capture execution shape without persisting raw prompts, tool arguments,
tool results, or assistant text.

Run traces are designed as a substrate for future eval, review, and replay
workflows. The first implementation is observe-only and fail-open: if trace
writing fails, the agent turn continues normally.

## Enable

Run tracing is disabled by default. Enable it in `~/.hermes/config.yaml`:

```yaml
observability:
  run_trace_enabled: true
  run_trace_path: run_traces/run_traces.jsonl
```

`run_trace_path` is relative to `HERMES_HOME`; absolute or escaping paths fall
back to the default path under `HERMES_HOME`.

## Storage

By default traces are appended as JSON Lines:

```text
~/.hermes/run_traces/run_traces.jsonl
```

Each line is one turn-level record.

## Schema

Current schema version: `hermes_run_trace_v1`.

```json
{
  "schema_version": "hermes_run_trace_v1",
  "run_id": "sha256:...",
  "session_id": "sha256:...",
  "turn_id": "sha256:...",
  "task_id": "sha256:...",
  "model": "anthropic/claude-sonnet-4.6",
  "provider": "openrouter",
  "source": "cli",
  "status": "completed",
  "exit_reason": "text_response",
  "api_call_count": 2,
  "started_at_ms": 1770000000000,
  "ended_at_ms": 1770000001234,
  "duration_ms": 1234,
  "tool_calls": [
    {
      "name": "read_file",
      "tool_call_id": "sha256:...",
      "status": "requested",
      "duration_ms": null,
      "error_type": "",
      "error_message": ""
    }
  ]
}
```

## Privacy boundary

Run traces must not store:

- raw user prompts;
- raw system prompts;
- raw assistant text;
- raw tool arguments;
- raw tool outputs;
- full message history;
- credentials or secret values.

Only metadata is persisted. Metadata strings are still passed through forced
secret redaction before writing. All run, session, turn, task, and tool-call
identifiers are stored as short SHA-256 fingerprints instead of raw text. This
keeps generated IDs correlatable while preventing slug-shaped caller text from
bypassing the metadata-only boundary.

`exit_reason` and per-tool `error_message` are controlled codes, not raw
exception strings. This avoids persisting exception text that might contain
prompts, tool arguments, tool outputs, URLs, or other non-credential private
data.

## Relationship to trajectories

Trajectories are detailed debugging artifacts and may include the content needed
to reconstruct model/tool interactions. Run traces are not a replacement for
trajectories. They are a low-risk observability surface for aggregate review and
future eval tooling.

Use trajectories when you need full execution detail and have an appropriate
privacy boundary. Use run traces when you only need run shape, status, timing,
model/provider, and tool names.

## Failure behavior

Trace persistence is best-effort. Writer errors are logged at debug level and
return `False` internally; they do not fail the agent turn.
