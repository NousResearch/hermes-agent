---
sidebar_position: 4
title: "Semantic Recall"
description: "Per-turn embedding-based recall of similar past turns in the same session (jcode feature adoption)."
---

# Semantic Recall

Semantic recall is a per-session, per-turn surface that automatically surfaces earlier turns in the same session that are semantically similar to the user's current message. It is independent of [Persistent Memory](./memory) (MEMORY.md / USER.md) — recall is *within* a session, not across them.

Adopted from [jcode](https://github.com/1jehuang/jcode)'s memory architecture.

## What it does

When semantic recall is enabled:

1. Each user and assistant turn is embedded (turned into a vector) at the moment it lands in the conversation.
2. On each subsequent turn, the user's message is embedded too.
3. The top-K (default 5) most similar past turns are returned by cosine similarity.
4. Those turns are appended to the user message as a `<recalled_context>` block at API-call time only — never persisted, never injected into the cached system prompt, never visible in session transcripts.

The model sees background context from earlier in the same session without having to call a tool to fetch it.

## Enable it

```bash
hermes config set memory.semantic_recall.enabled true
hermes config set memory.semantic_recall.backend numpy
```

Then start a new session. The first turn will trigger a one-time download of the embedding model (`all-MiniLM-L6-v2`, ~90 MB).

## Backends

| Backend | Cost | Notes |
|---|---|---|
| `noop` | Free | Default if explicitly chosen. Disables embedding. Top-K is always empty. No-op fallback when other backends fail to load. |
| `fastembed` | ~25 MB model download on first use, no torch | **Recommended default.** Uses `fastembed` + `BAAI/bge-small-en-v1.5` (384-dim). CPU-fast (~5–20 ms per text). Fully offline after first download. Pure ONNX runtime — no torch dependency. |
| `numpy` | ~500 MB total via torch | Uses `sentence-transformers` + `all-MiniLM-L6-v2`. Cosine sim over real vectors. Heavy install — only use if you already have torch. |

If the configured backend is unavailable (e.g. `fastembed` not installed, network down on first use), recall silently degrades to noop — no error, no injection. Run `hermes doctor` to see the current state.

## Configuration

All knobs live under `memory.semantic_recall.*` in `~/.hermes/config.yaml`:

```yaml
memory:
  semantic_recall:
    enabled: false           # turn the feature on
    backend: "fastembed"     # "noop" | "fastembed" | "numpy"
    model: "BAAI/bge-small-en-v1.5"
    top_k: 5                 # max turns recalled per turn
    max_turns: 200           # sliding-window size of the on-disk store
    max_tokens: 1500         # hard cap on the recall block size
```

## What it doesn't do

- **Not cross-session.** Recall only sees turns from the current session. Cross-session memory is what [Persistent Memory](./memory) is for.
- **Not persistent.** The recall block is ephemeral — it is injected at API-call time and never written to the session DB. Session resume does not replay it.
- **Not a tool.** The model cannot query recall directly. It is purely a passive injection surface.
- **Not prompt-cached.** The cached system prompt stays stable across turns — only the per-turn user message gets the recall block appended. This preserves prefix caching.

## Disabling

```bash
hermes config set memory.semantic_recall.enabled false
```

To wipe the on-disk embedding store:

```bash
rm ~/.hermes/profiles/default/recall.db
```

## Inspecting

`hermes doctor` shows the current recall state — whether the feature is enabled, which backend is active, whether the backend is healthy, and how many embeddings are stored.

## Performance

| Operation | Typical latency |
|---|---|
| First call (model load) | 1–3 s |
| Embed a short turn | 5–50 ms |
| Top-K cosine over 200 turns | <1 ms |
| Total per-turn overhead (enabled) | 5–100 ms |

The injection adds ~0–1500 tokens to the user message per turn, depending on `max_tokens` and how much of the recalled content fits.

## See also

- [Persistent Memory](./memory) — durable cross-session memory (MEMORY.md, USER.md)
- [Session search](./sessions) — full-text search over past session transcripts
