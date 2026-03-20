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
2. `config.yaml` model/provider config
3. environment variables
4. provider-specific defaults or auto resolution

That ordering matters because Hermes treats the saved model/provider choice as the source of truth for normal runs. This prevents a stale shell export from silently overriding the endpoint a user last selected in `hermes model`.

## Providers

Current provider families include:

- AI Gateway (Vercel)
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

## AI Gateway

Set `AI_GATEWAY_API_KEY` in `~/.hermes/.env` and run with `--provider ai-gateway`. Hermes fetches available models from the gateway's `/models` endpoint, filtering to language models with tool-use support.

## OpenRouter, AI Gateway, and custom OpenAI-compatible base URLs

Hermes contains logic to avoid leaking the wrong API key to a custom endpoint when multiple provider keys exist (e.g. `OPENROUTER_API_KEY`, `AI_GATEWAY_API_KEY`, and `OPENAI_API_KEY`).

Each provider's API key is scoped to its own base URL:

- `OPENROUTER_API_KEY` is only sent to `openrouter.ai` endpoints
- `AI_GATEWAY_API_KEY` is only sent to `ai-gateway.vercel.sh` endpoints
- `OPENAI_API_KEY` is used for custom endpoints and as a fallback

Hermes also distinguishes between:

- a real custom endpoint selected by the user
- the OpenRouter fallback path used when no custom endpoint is configured

That distinction is especially important for:

- local model servers
- non-OpenRouter/non-AI Gateway OpenAI-compatible APIs
- switching providers without re-running setup
- config-saved custom endpoints that should keep working even when `OPENAI_BASE_URL` is not exported in the current shell

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

When an auxiliary task is configured with provider `main`, Hermes resolves that through the same shared runtime path as normal chat. In practice that means:

- env-driven custom endpoints still work
- custom endpoints saved via `hermes model` / `config.yaml` also work
- auxiliary routing can tell the difference between a real saved custom endpoint and the OpenRouter fallback

## Fallback models

Hermes supports a configured fallback model/provider pair, allowing runtime failover when the primary model encounters errors.

### How it works internally

1. **Storage**: `AIAgent.__init__` stores the `fallback_model` dict and sets `_fallback_activated = False`.

2. **Trigger points**: `_try_activate_fallback()` is called from three places in the main retry loop in `run_agent.py`:
   - After max retries on invalid API responses (None choices, missing content)
   - On non-retryable client errors (HTTP 401, 403, 404)
   - After max retries on transient errors (HTTP 429, 500, 502, 503)

3. **Activation flow** (`_try_activate_fallback`):
   - Returns `False` immediately if already activated or not configured
   - Calls `resolve_provider_client()` from `auxiliary_client.py` to build a new client with proper auth
   - Determines `api_mode`: `codex_responses` for openai-codex, `anthropic_messages` for anthropic, `chat_completions` for everything else
   - Swaps in-place: `self.model`, `self.provider`, `self.base_url`, `self.api_mode`, `self.client`, `self._client_kwargs`
   - For anthropic fallback: builds a native Anthropic client instead of OpenAI-compatible
   - Re-evaluates prompt caching (enabled for Claude models on OpenRouter)
   - Sets `_fallback_activated = True` — prevents firing again
   - Resets retry count to 0 and continues the loop

4. **Config flow**:
   - CLI: `cli.py` reads `CLI_CONFIG["fallback_model"]` → passes to `AIAgent(fallback_model=...)`
   - Gateway: `gateway/run.py._load_fallback_model()` reads `config.yaml` → passes to `AIAgent`
   - Validation: both `provider` and `model` keys must be non-empty, or fallback is disabled

### What does NOT support fallback

- **Subagent delegation** (`tools/delegate_tool.py`): subagents inherit the parent's provider but not the fallback config
- **Cron jobs** (`cron/`): run with a fixed provider, no fallback mechanism
- **Auxiliary tasks**: use their own independent provider auto-detection chain (see Auxiliary model routing above)

### Test coverage

See `tests/test_fallback_model.py` for comprehensive tests covering all supported providers, one-shot semantics, and edge cases.

## Related docs

- [Agent Loop Internals](./agent-loop.md)
- [ACP Internals](./acp-internals.md)
- [Context Compression & Prompt Caching](./context-compression-and-caching.md)
