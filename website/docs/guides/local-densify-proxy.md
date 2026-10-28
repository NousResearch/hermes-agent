---
sidebar_position: 18
title: "Cut Paid Token Spend with a Local Densify Proxy"
description: "Route Hermes through a local OpenAI-compatible proxy that densifies outbound prompts before they hit paid APIs — complementary to Hermes context compression"
---

# Cut Paid Token Spend with a Local Densify Proxy

Hermes already **compacts long conversations** when you approach the model context
limit (`compression:` / the built-in `ContextCompressor`). That is mid-session
history management.

This guide covers a different layer: **pre-request densification**. A small
local proxy rewrites verbose user prompts (fluff, repetition, noisy tool dumps)
*before* they leave your machine for a paid provider. The two layers stack —
leave Hermes compression on.

```text
Hermes Agent
      │
      ▼
 Local densify proxy  ← free rules + optional local LLM (Ollama / MLX)
      │  fewer input tokens
      ▼
 Paid API (xAI, OpenRouter, Anthropic, …)
```

One open-source implementation of this pattern is
[prompt-codec](https://github.com/jwaynelowry/prompt-codec) (Rust, MIT). Any
OpenAI-compatible densify proxy on loopback works the same way.

:::info Not a replacement for context compression
Do **not** turn off `compression.enabled` for this. Densify proxies shrink each
outbound request; Hermes compression still summarizes older turns when the
session is large.
:::

## What you need

| Piece | Notes |
|-------|-------|
| Hermes Agent | Working install with a paid (or Portal) provider |
| Local densify proxy | e.g. prompt-codec listening on `127.0.0.1:8787` |
| Optional local LLM | Ollama / MLX for stronger savings; rules-only mode works without one |
| Upstream API key | Same key Hermes already uses for the target provider |

## Quick start (prompt-codec)

### 1. Run the proxy

```bash
# Install / build — see the project's README
cargo install --path /path/to/prompt-codec   # or: cargo build --release

export X_API_KEY=...   # whatever the proxy's upstream_api_key_env expects
prompt-codec proxy
# → http://127.0.0.1:8787/v1
```

Confirm the local model (if configured) is reachable:

```bash
prompt-codec health
```

### 2. Add a Hermes provider

In `~/.hermes/config.yaml`:

```yaml
providers:
  prompt_codec:
    api: http://127.0.0.1:8787/v1
    name: prompt_codec
    api_key: ${X_API_KEY}
    transport: chat_completions
```

Or as a named custom endpoint via `custom_providers:` — same URL and key pattern.
See [AI Providers](/integrations/providers) for the full custom-endpoint options.

### 3. Point Hermes at it when you want savings

Use `hermes model` (or set `model.provider`) to select the densify provider for
sessions where input tokens dominate cost. Keep your direct Portal / xAI /
OpenRouter provider for turns where you want **zero encode latency**.

## Latency and failure modes

| Concern | Expected behavior |
|---------|-------------------|
| Encode latency | Typically ~1–5s with a warm small local model; rules-only is milliseconds |
| Local model down / timeout | Good proxies degrade to deterministic rules (or pass-through) — Hermes should still get a normal upstream response |
| Streaming / errors | Upstream status, headers, and SSE should pass through unchanged (429 stays 429) |
| Prompt caching | Prefer densify logic that keeps resent history **byte-stable** (e.g. only rewrite the latest user turn on a cache miss) so Anthropic / provider prompt caches stay warm |

## How this differs from running Hermes entirely on Ollama

| | [Local Ollama as the main model](./local-ollama-setup.md) | Densify proxy in front of a paid model |
|--|----------------------------------------------------------|----------------------------------------|
| Who answers the task | Local model | Paid / Portal model |
| Goal | Zero API $ / full privacy | Fewer **input** tokens on paid calls |
| Quality | Bound by local model | Bound by your cloud model |
| Hermes compression | Still useful | Still useful |

Many people run both: densify + paid model for daily agent work; full-local
Ollama for offline / privacy sessions.

## Security notes

- Bind the densify proxy to **loopback only** (`127.0.0.1`).
- Prefer a host-header guard so browser DNS-rebinding cannot drive the proxy.
- Never put long-lived upstream secrets in a world-readable config file — use
  env vars (`X_API_KEY`, Hermes `.env`, etc.).

## See also

- [Context compression & caching](/developer-guide/context-compression-and-caching) — Hermes mid-session compaction
- [AI Providers](/integrations/providers) — custom endpoints and `providers:`
- [Run Hermes locally with Ollama](./local-ollama-setup.md) — full-local inference
- [prompt-codec](https://github.com/jwaynelowry/prompt-codec) — reference densify proxy
