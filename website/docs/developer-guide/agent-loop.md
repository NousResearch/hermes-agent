---
sidebar_position: 3
title: "Agent Loop Internals"
description: "Detailed walkthrough of AIAgent execution, API modes, tools, callbacks, and fallback behavior"
---

# Agent Loop Internals

The core orchestration engine is `run_agent.py`'s `AIAgent`.

## Core responsibilities

`AIAgent` is responsible for:

- assembling the effective prompt and tool schemas
- selecting the correct provider/API mode
- making interruptible model calls
- executing tool calls (sequentially or concurrently)
- maintaining session history
- handling compression, retries, and fallback models

## API modes

Hermes currently supports three API execution modes:

| API mode | Used for |
|----------|----------|
| `chat_completions` | OpenAI-compatible chat endpoints, including OpenRouter and most custom endpoints |
| `codex_responses` | OpenAI Codex / Responses API path |
| `anthropic_messages` | Native Anthropic Messages API |

The mode is resolved from explicit args, provider selection, and base URL heuristics.

## Turn lifecycle

```text
run_conversation()
  -> generate effective task_id
  -> append current user message
  -> load or build cached system prompt
  -> maybe preflight-compress
  -> build api_messages
  -> inject ephemeral prompt layers
  -> apply prompt caching if appropriate
  -> make interruptible API call
  -> if tool calls: execute them, append tool results, loop
  -> if final text: persist, cleanup, return response
```

## Interruptible API calls

Hermes wraps API requests so they can be interrupted from the CLI or gateway.

This matters because:

- the agent may be in a long LLM call
- the user may send a new message mid-flight
- background systems may need cancellation semantics

## Tool execution modes

Hermes uses two execution strategies:

- sequential execution for single or interactive tools
- concurrent execution for multiple non-interactive tools

Concurrent tool execution preserves message/result ordering when reinserting tool responses into conversation history.

## Callback surfaces

`AIAgent` supports platform/integration callbacks such as:

- `tool_progress_callback`
- `thinking_callback`
- `reasoning_callback`
- `clarify_callback`
- `step_callback`
- `message_callback`

These are how the CLI, gateway, and ACP integrations stream intermediate progress and interactive approval/clarification flows.

## Budget and fallback behavior

Hermes tracks a shared iteration budget across parent and subagents. It also injects budget pressure hints near the end of the available iteration window.

Fallback model support allows the agent to switch providers/models when the primary route fails in supported failure paths.

## Compression and persistence

Before and during long runs, Hermes may:

- flush memory before context loss
- compress middle conversation turns
- split the session lineage into a new session ID after compression
- preserve recent context and structural tool-call/result consistency

## Key files to read next

- `run_agent.py`
- `agent/prompt_builder.py`
- `agent/context_compressor.py`
- `agent/prompt_caching.py`
- `model_tools.py`

## Related docs

- [Provider Runtime Resolution](./provider-runtime.md)
- [Prompt Assembly](./prompt-assembly.md)
- [Context Compression & Prompt Caching](./context-compression-and-caching.md)
- [Tools Runtime](./tools-runtime.md)
