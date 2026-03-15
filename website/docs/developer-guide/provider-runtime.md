---
sidebar_position: 4
title: "Provider Runtime Resolution"
description: "How Hermes resolves providers, credentials, API modes, and auxiliary models at runtime"
---

# Provider Runtime Resolution

Hermes has a shared provider runtime resolver used across:

- CLI
- gateway
- cron jobs
- ACP
- auxiliary model calls

Primary implementation:

- `hermes_cli/runtime_provider.py`
- `hermes_cli/auth.py`
- `agent/auxiliary_client.py`

If you are trying to add a new first-class inference provider, read [Adding Providers](./adding-providers.md) alongside this page.

## Resolution precedence

At a high level, provider resolution uses:

1. explicit CLI/runtime request
2. environment variables
3. `config.yaml` model/provider config
4. provider-specific defaults or auto resolution

## Providers

Current provider families include:

- OpenRouter
- Nous Portal
- OpenAI Codex
- Anthropic (native)
- Z.AI
- Kimi / Moonshot
- MiniMax
- MiniMax China
- custom OpenAI-compatible endpoints

## Output of runtime resolution

The runtime resolver returns data such as:

- `provider`
- `api_mode`
- `base_url`
- `api_key`
- `source`
- provider-specific metadata like expiry/refresh info

## Why this matters

This resolver is the main reason Hermes can share auth/runtime logic between:

- `hermes chat`
- gateway message handling
- cron jobs running in fresh sessions
- ACP editor sessions
- auxiliary model tasks

## OpenRouter vs custom OpenAI-compatible base URLs

Hermes contains logic to avoid leaking the wrong API key to a custom endpoint when both `OPENROUTER_API_KEY` and `OPENAI_API_KEY` exist.

That distinction is especially important for:

- local model servers
- non-OpenRouter OpenAI-compatible APIs
- switching providers without re-running setup

## Native Anthropic path

Anthropic is not just "via OpenRouter" anymore.

When provider resolution selects `anthropic`, Hermes uses:

- `api_mode = anthropic_messages`
- the native Anthropic Messages API
- `agent/anthropic_adapter.py` for translation

Credential resolution for native Anthropic now prefers refreshable Claude Code credentials over copied env tokens when both are present. In practice that means:

- Claude Code credential files are treated as the preferred source when they include refreshable auth
- manual `ANTHROPIC_TOKEN` / `CLAUDE_CODE_OAUTH_TOKEN` values still work as explicit overrides
- Hermes preflights Anthropic credential refresh before native Messages API calls
- Hermes still retries once on a 401 after rebuilding the Anthropic client, as a fallback path

## OpenAI Codex path

Codex uses a separate Responses API path:

- `api_mode = codex_responses`
- dedicated credential resolution and auth store support

## Auxiliary model routing

Auxiliary tasks such as:

- vision
- web extraction summarization
- context compression summaries
- session search summarization
- skills hub operations
- MCP helper operations
- memory flushes

can use their own provider/model routing rather than the main conversational model.

## Fallback models

Hermes also supports a configured fallback model/provider, allowing runtime failover in supported error paths.

## Related docs

- [Agent Loop Internals](./agent-loop.md)
- [ACP Internals](./acp-internals.md)
- [Context Compression & Prompt Caching](./context-compression-and-caching.md)
