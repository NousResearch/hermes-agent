---
title: Prompt Caching
description: Reuse your stable system prompt across requests to cut input token cost and latency.
sidebar_label: Prompt Caching
sidebar_position: 9
---

# Prompt Caching

Prompt caching lets the model reuse your stable conversation prefix — most importantly your system prompt and `SOUL.md` — across requests instead of re-reading it every turn, cutting input token cost and shaving latency off each reply.

It's set-and-forget: enable it with a single config line, and Hermes attaches the right cache markers automatically. There's no per-model tuning to do.

## Enabling It

Add a `prompt_caching` section to your `~/.hermes/config.yaml`:

```yaml
prompt_caching:
  cache_ttl: "5m"   # use "1h" for long sessions with pauses between turns
```

> **Config path:** `~/.hermes/config.yaml` is the default. If `HERMES_HOME` is set (e.g. on a server install), the file is at `$HERMES_HOME/config.yaml` instead.

`cache_ttl` is the only setting. The cache prefix is reused for this long after each request; subsequent requests within the window get a cache hit. Only `"5m"` and `"1h"` are valid — any other value falls back to `"5m"`.

Hermes handles the rest: it sends the provider's `cache_control` markers automatically and only when the active model supports caching. You don't configure anything per model.

## Supported Providers

`cache_ttl` takes effect for providers where Hermes injects `cache_control` markers:

| Provider / Model | `cache_ttl` effect | Notes |
|------------------|--------------------|-------|
| **Anthropic Claude** — native API, or via OpenRouter / Nous Portal | Yes | Default path; no extra config |
| **Qwen / Alibaba** — on OpenCode / DashScope | Yes | |
| **DeepSeek** — `deepseek-v3.2`, `deepseek-v4-flash` via OpenRouter | Yes | Requires [#36984](https://github.com/NousResearch/hermes-agent/pull/36984) — see below |
| **Other providers** (Llama, Mistral, etc.) | No effect | Provider-side caching only |
| **Cerebras** | No effect | Automatic server-side KV caching, independent of this config |

For supported providers, the CLI confirms caching at startup:

```
💾 Prompt caching: ENABLED (Claude via OpenRouter, 5m TTL)
```

This message only appears for Claude and Qwen paths. Cerebras and DeepSeek don't emit it — that's expected.

### DeepSeek via OpenRouter

`deepseek/deepseek-v4-flash` and `deepseek/deepseek-v3.2` accept `cache_control` markers on OpenRouter and return real cache hits when they are present. Without them, the full prompt is re-billed on every turn.

This requires the models to be in `_OPENROUTER_EXPLICIT_CACHE_CONTROL_MODEL_IDS` in `agent/agent_runtime_helpers.py`. [PR #36984](https://github.com/NousResearch/hermes-agent/pull/36984) (open, pending merge) adds `deepseek-v4-flash` and `deepseek-v4-flash-20260423`, building on [#20945](https://github.com/NousResearch/hermes-agent/pull/20945) which covers `deepseek-v3.2` and the Qwen family. Once either is merged, `cache_ttl` controls DeepSeek caching the same way it does Claude.

**Measured on production Hermes (2026-05-31, `deepseek/deepseek-v4-flash` via OpenRouter):**

| Scenario | Cache hit rate |
|----------|---------------|
| Warmup → identical repeat | 98% |
| Realistic 5-turn growing conversation | 87–89% (warm turns) |
| Aggregate over multi-turn session | ~64% |
| Baseline before patch | ~1% |

### Cerebras

Cerebras caches KV state of repeated system-prompt prefixes automatically at the infrastructure level. No `cache_control` markers are sent and `cache_ttl` has no effect — the caching happens regardless. An orchestrator with a stable `SOUL.md` typically sees a **91–99% cache hit rate** with **1–4 seconds** of latency savings per request.

## Why It Helps

The biggest win is on the parts of the prompt that never change between turns — your system prompt and `SOUL.md`. As long as those stay stable, the model reads them from cache instead of reprocessing them every request. The savings compound over a long session.

## Tips

- **Leave it on.** There's no downside on unsupported models, and no per-model setup.
- **Keep your system prompt stable.** Editing `SOUL.md` or the system prompt mid-session invalidates the cached prefix, and the cache rebuilds over the next turn or two.
- **Use `"1h"` for slow conversations.** If you take long breaks between messages, `cache_ttl: "1h"` keeps the prefix warm past the 5-minute default.
