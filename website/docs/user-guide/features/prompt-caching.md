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

`cache_ttl` is the only setting. The cache prefix is reused for this long after each request; subsequent requests within the window get a cache hit. Only `"5m"` and `"1h"` are valid — any other value falls back to `"5m"`.

Hermes handles the rest: it sends the provider's `cache_control` markers automatically and only when the active model supports caching. You don't configure anything per model.

## Supported Providers

Caching activates automatically when your model and provider support it:

| Provider | Caching |
|----------|---------|
| **Anthropic Claude** (native API or via OpenRouter) | Yes — controlled by `cache_ttl` |
| **DeepSeek** | Yes |
| **Cerebras** | Yes — automatic, no config needed |

When caching is active, the CLI shows it at startup:

```
💾 Prompt caching: ENABLED (Claude via OpenRouter, 5m TTL)
```

If your model doesn't support caching, the setting is simply ignored — it's safe to leave in your config.

## Why It Helps

The biggest win is on the parts of the prompt that never change between turns — your system prompt and `SOUL.md`. As long as those stay stable, the model reads them from cache instead of reprocessing them every request.

In practice, the orchestrator running on Cerebras sees a **91–99% cache hit rate** when the system prompt is stable, saving roughly **1–4 seconds per request**. The savings compound over a long session: every turn that hits the cache is cheaper and faster than the one before it would have been.

## Tips

- **Leave it on.** There's no downside on unsupported models, and no per-model setup.
- **Keep your system prompt stable.** Editing `SOUL.md` or the system prompt mid-session invalidates the cached prefix, and the cache rebuilds over the next turn or two.
- **Use `"1h"` for slow conversations.** If you take long breaks between messages (e.g. a chat assistant you ping a few times an hour), `cache_ttl: "1h"` keeps the prefix warm past the 5-minute default.
