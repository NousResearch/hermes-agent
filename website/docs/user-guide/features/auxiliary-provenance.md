---
title: Auxiliary Provenance
description: Per-call metadata showing which provider served auxiliary calls, the full fallback chain walked, and any errors.
sidebar_label: Auxiliary Provenance
sidebar_position: 10
---

# Auxiliary Provenance

When Hermes resolves an auxiliary task (vision, compression, skills hub, MCP, approval, title generation, review, or triage specifier) it walks a provider chain — primary, then per-task fallback, then main-agent fallback, then built-in discovery — until one succeeds. Auxiliary provenance records that entire journey so you can inspect it after the fact.

## What Provenance Shows

Provenance is per-call metadata. It tells you:

- **`served_by`** — the provider that actually served the call
- **`served_model`** — the model used for the service
- **`fallback_chain_used`** — the ordered list of providers/models tried
- **`fallback_count`** — how many fallbacks were needed before success
- **`attempts[]`** — each attempt in the chain: provider, model, status (`ok`, `failed`, `skipped`), any `failure` message, and `latency_ms`
- **`final_status`** — `ok` if the call eventually succeeded, or the final failure reason

This is useful for debugging extraction quality when fallback kicks in: if a vision summary or compression result looks off, you can see whether it came from the primary model or a deeper fallback with a different capability profile.

## Enabling Provenance

Provenance is **opt-in** and off by default. Add to `~/.hermes/config.yaml`:

```yaml
auxiliary:
  expose_provenance: true
```

When off (default), there is zero behavior change — no extra data is collected, no provider routing info is exposed, and no overhead is added.

## Reading Provenance

When `expose_provenance` is enabled, callers can read provenance after an auxiliary call:

```python
from hermes_agent.auxiliary import _get_auxiliary_provenance

prov = _get_auxiliary_provenance()
# Returns a dict with:
# {
#   "served_by": "openrouter",
#   "served_model": "google/gemini-3-flash-preview",
#   "fallback_chain_used": ["main", "openrouter", "nous"],
#   "fallback_count": 1,
#   "attempts": [
#     {"provider": "main", "model": "anthropic/claude-sonnet-4", "status": "failed", "failure": "capacity exceeded", "latency_ms": 1240},
#     {"provider": "openrouter", "model": "google/gemini-3-flash-preview", "status": "ok", "failure": None, "latency_ms": 890}
#   ],
#   "final_status": "ok"
# }
```

Each attempt entry includes the provider name, model string, status, optional failure text, and latency in milliseconds.

## Privacy and Cost

- **Default off** — no data is captured unless you explicitly enable it.
- **Zero behavior change when off** — the feature is a no-op; nothing leaks, nothing slows down.
- **Opt-in by design** — exposing which providers served which calls reveals your routing configuration. Keeping it off avoids leaking that info in shared logs or multi-user environments.

## Use Case

Per [@Rashadamom](https://github.com/NousResearch/hermes-agent/issues/22201#issuecomment-...) on #22201: when a fallback provider serves an auxiliary task like vision or compression, the output quality can differ from the primary model. Provenance lets you confirm whether a poor extraction came from the intended provider or a deeper fallback — making root-cause analysis faster.

## Related

- Issue: [#36797](https://github.com/NousResearch/hermes-agent/issues/36797) — auxiliary provenance feature
- Related docs: [Fallback Providers](./fallback-providers.md)
