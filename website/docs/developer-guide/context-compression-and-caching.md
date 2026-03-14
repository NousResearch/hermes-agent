---
sidebar_position: 6
title: "Context Compression & Prompt Caching"
description: "How Hermes compresses long conversations and applies provider-side prompt caching"
---

# Context Compression & Prompt Caching

Hermes manages long conversations with two complementary mechanisms:

- prompt caching
- context compression

Primary files:

- `agent/prompt_caching.py`
- `agent/context_compressor.py`
- `run_agent.py`

## Prompt caching

For Anthropic/native and Claude-via-OpenRouter flows, Hermes applies Anthropic-style cache markers.

Current strategy:

- cache the system prompt
- cache the last 3 non-system messages
- default TTL is 5 minutes unless explicitly extended

This is implemented in `agent/prompt_caching.py`.

## Why prompt stability matters

Prompt caching only helps when the stable prefix remains stable. That is why Hermes avoids rebuilding or mutating the core system prompt mid-session unless it has to.

## Compression trigger

Hermes can compress context when conversations become large. Configuration defaults live in `config.yaml`, and the compressor also has runtime checks based on actual prompt token counts.

## Compression algorithm

The compressor protects:

- the first N turns
- the last N turns

and summarizes the middle section.

It also cleans up structural issues such as orphaned tool-call/result pairs so the API never receives invalid conversation structure after compression.

## Pre-compression memory flush

Before compression, Hermes can give the model one last chance to persist memory so facts are not lost when middle turns are summarized away.

## Session lineage after compression

Compression can split the session into a new session ID while preserving parent lineage in the state DB.

This lets Hermes continue operating with a smaller active context while retaining a searchable ancestry chain.

## Re-injected state after compression

After compression, Hermes may re-inject compact operational state such as:

- todo snapshot
- prior-read-files summary

## Related docs

- [Prompt Assembly](./prompt-assembly.md)
- [Session Storage](./session-storage.md)
- [Agent Loop Internals](./agent-loop.md)
