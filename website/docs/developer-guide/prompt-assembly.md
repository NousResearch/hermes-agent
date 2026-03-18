---
sidebar_position: 5
title: "Prompt Assembly"
description: "How Hermes builds the system prompt, preserves cache stability, and injects ephemeral layers"
---

# Prompt Assembly

Hermes deliberately separates:

- **cached system prompt state**
- **ephemeral API-call-time additions**

This is one of the most important design choices in the project because it affects:

- token usage
- prompt caching effectiveness
- session continuity
- memory correctness

Primary files:

- `run_agent.py`
- `agent/prompt_builder.py`
- `tools/memory_tool.py`

## Cached system prompt layers

The cached system prompt is assembled in roughly this order:

1. agent identity — `SOUL.md` from `HERMES_HOME` when available, otherwise falls back to `DEFAULT_AGENT_IDENTITY` in `prompt_builder.py`
2. tool-aware behavior guidance
3. Honcho static block (when active)
4. optional system message
5. frozen MEMORY snapshot
6. frozen USER profile snapshot
7. skills index
8. context files (`AGENTS.md`, `.cursorrules`, `.cursor/rules/*.mdc`) — SOUL.md is **not** included here when it was already loaded as the identity in step 1
9. timestamp / optional session ID
10. platform hint

When `skip_context_files` is set (e.g., subagent delegation), SOUL.md is not loaded and the hardcoded `DEFAULT_AGENT_IDENTITY` is used instead.

## API-call-time-only layers

These are intentionally *not* persisted as part of the cached system prompt:

- `ephemeral_system_prompt`
- prefill messages
- gateway-derived session context overlays
- later-turn Honcho recall injected into the current-turn user message

This separation keeps the stable prefix stable for caching.

## Memory snapshots

Local memory and user profile data are injected as frozen snapshots at session start. Mid-session writes update disk state but do not mutate the already-built system prompt until a new session or forced rebuild occurs.

## Context files

`agent/prompt_builder.py` scans and sanitizes:

- `AGENTS.md`
- `.cursorrules`
- `.cursor/rules/*.mdc`

`SOUL.md` is loaded separately via `load_soul_md()` for the identity slot. When it loads successfully, `build_context_files_prompt(skip_soul=True)` prevents it from appearing twice.

Long files are truncated before injection.

## Skills index

The skills system contributes a compact skills index to the prompt when skills tooling is available.

## Why prompt assembly is split this way

The architecture is intentionally optimized to:

- preserve provider-side prompt caching
- avoid mutating history unnecessarily
- keep memory semantics understandable
- let gateway/ACP/CLI add context without poisoning persistent prompt state

## Related docs

- [Context Compression & Prompt Caching](./context-compression-and-caching.md)
- [Session Storage](./session-storage.md)
- [Gateway Internals](./gateway-internals.md)
