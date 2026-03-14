---
sidebar_position: 8
title: "Session Storage"
description: "How Hermes stores sessions in SQLite, maintains lineage, and exposes recall/search"
---

# Session Storage

Hermes uses a SQLite-backed session store as the main source of truth for historical conversation state.

Primary files:

- `hermes_state.py`
- `gateway/session.py`
- `tools/session_search_tool.py`

## Main database

The primary store lives at:

```text
~/.hermes/state.db
```

It contains:

- sessions
- messages
- metadata such as token counts and titles
- lineage relationships
- full-text search indexes

## What is stored per session

Examples of important session metadata:

- session ID
- source/platform
- title
- created/updated timestamps
- token counts
- tool call counts
- stored system prompt snapshot
- parent session ID after compression splits

## Lineage

When Hermes compresses a conversation, it can continue in a new session ID while preserving ancestry via `parent_session_id`.

This means resuming/searching can follow session families instead of treating each compressed shard as unrelated.

## Gateway vs CLI persistence

- CLI uses the state DB directly for resume/history/search
- gateway keeps active-session mappings and may also maintain additional platform transcript/state files
- some legacy JSON/JSONL artifacts still exist for compatibility, but SQLite is the main historical store

## Session search

The `session_search` tool uses the session DB's search features to retrieve and summarize relevant past work.

## Related docs

- [Gateway Internals](./gateway-internals.md)
- [Prompt Assembly](./prompt-assembly.md)
- [Context Compression & Prompt Caching](./context-compression-and-caching.md)
