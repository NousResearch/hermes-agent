# Tool Loop Detection

Hermes detects when an agent enters a degenerate tool-calling loop and
automatically intervenes to break the cycle.

## The Problem

Autoregressive language models generate tool call names token by token.
When a model calls the wrong tool repeatedly, each failed attempt adds
more occurrences of the wrong name to context, making it increasingly
likely to generate the same wrong name on the next attempt. This creates
a self-reinforcing failure mode that the model cannot escape through
reasoning alone — even when its internal thinking explicitly identifies
the correct tool.

## How It Works

Three detection strategies run after every tool call:

| Detector | Triggers When |
|----------|---------------|
| `generic_repeat` | Same tool + identical arguments called N times consecutively |
| `poll_no_progress` | Same tool returns identical results N times consecutively |
| `ping_pong` | Agent alternates between exactly 2 tool states (A-B-A-B) |

### Severity Levels

- **Warning** (default: 3 consecutive): Logged to console. No intervention.
- **Critical** (default: 5 consecutive): Repeated call/response pairs are
  pruned from context and replaced with a summary message.

### Context Pruning

The critical intervention is **removing the repeated wrong-tool-name
occurrences from context**. This directly changes the token probability
distribution, making it possible for the model to generate a different
tool name on the next attempt.

When the model's reasoning content mentions a different tool name than
the one it actually called (a "reasoning-intent mismatch"), the summary
message includes explicit guidance to try that tool instead.

## Configuration

In `config.yaml`:

```yaml
tool_loop_detection:
  warning_threshold: 3    # Log a warning after this many consecutive repeats
  critical_threshold: 5   # Prune context after this many consecutive repeats
  window_size: 30         # Sliding window of recent tool calls to track
```

All values are optional; defaults apply when omitted.

## Relationship to File Read Loop Detection

The `read_file` and `search` tools have their own loop detection
(warn at 3, block at 4 consecutive identical reads). That mechanism
operates at the tool level and blocks execution.

Tool loop detection operates at the orchestrator level and **prunes
context** — a fundamentally different intervention that addresses
token-level anchoring rather than just preventing execution.
