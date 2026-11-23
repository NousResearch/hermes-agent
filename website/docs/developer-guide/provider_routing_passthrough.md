# OpenRouter provider_routing passthrough

The `provider_routing` block in `config.yaml` supports **general
passthrough**: any key that is not one of the typed preference keys
(`sort`, `only`, `ignore`, `order`, `require_parameters`,
`data_collection`, and the per-model `models` map) is forwarded verbatim
into the OpenRouter request's `provider` object. This holds for both the
flat block and each `provider_routing.models.<model-id>` entry.

## Why

OpenRouter's provider preference object gains fields faster than any
wrapper can type them: `zdr` (Zero Data Retention), `quantizations`,
`max_price`, `preferred_min_throughput`, `preferred_max_latency`,
`allow_fallbacks`, `enforce_distillable_text`, provider-specific
`headers`, and more (see the
[provider routing docs](https://openrouter.ai/docs/features/provider-routing)).
Until now Hermes silently dropped any such key, so users could not opt
into routing guarantees — most notably ZDR — without code changes.

With passthrough, current and future OpenRouter routing fields work
without a Hermes release per new field.

## Configuration

```yaml
# ~/.hermes/config.yaml
provider_routing:
  # Typed keys keep their existing attribute-based semantics.
  sort: throughput
  # Everything else is passed through verbatim:
  zdr: true                      # Zero Data Retention routing
  quantizations: [fp8, fp16]     # endpoint quantization filter
  max_price:
    prompt: 0.5                  # USD per 1M prompt tokens
    completion: 1.5              # USD per 1M completion tokens
  preferred_min_throughput: 50   # tokens/s per 1M-prompt endpoints
  allow_fallbacks: false         # strict: only listed providers

  models:
    # Per-model entries overlay the flat block for that model, passthrough
    # keys included (e.g. kimi-k3's native mxfp4 must not inherit an fp8 floor).
    "moonshotai/kimi-k3":
      quantizations: [mxfp4]
```

### Semantics

- **Typed keys win on conflict.** A passthrough value for a typed key is
  ignored (logged at `DEBUG`): the typed config keys remain the
  authoritative spelling of those preferences.
- **Per-model entries outrank the flat block.** An unknown key set on a
  `models.<id>` entry wins over the same key set flat; the flat
  passthrough fills only what the per-model entry leaves unset.
- **Unknown keys warn at agent construction**: each unrecognized key
  logs a `WARNING` (`provider_routing: forwarding unrecognized key ...`)
  so typos surface immediately — a mistyped `zdr` should never route
  silently to a non-ZDR provider.
- **Explicit values are sent verbatim; unset is never sent.** A passthrough
  key set to a value — including `false` or `0`, since
  `allow_fallbacks: false` is a meaningful OpenRouter pin — is forwarded
  as-is; a key left unset (or `None`) emits no wire key.
- **Scoped to the provider object.** Only `extra_body["provider"]` is
  touched; nothing else in the request body changes.
- **Fail-closed on ZDR.** With `zdr: true`, OpenRouter routes only to
  ZDR-capable endpoints and errors if none exist rather than silently
  degrading to a data-retaining provider.

### Scope

OpenRouter only. The typed keys continue to work for the Nous profile;
passthrough keys are an OpenRouter-specific extension of the provider
preference object and are guarded separately (cf. #89430: the Nous API
400s on unknown provider keys).

### What it covers (OpenRouter provider-object fields)

- `zdr` — zero-data-retention routing (the motivating case)
- `quantizations`, `max_price` — replace one-off PRs #91800 and #72102/#72118/#94742
- `preferred_min_throughput`, `preferred_max_latency` — SLA-style routing
- `enforce_distillable_text`, `partition`, `allow_fallbacks` — policy controls
- any future field OpenRouter adds — zero Hermes-side changes needed

## Precedent

- [Nanocoder](https://github.com/Nanocoder/AI-CLI) OpenRouter provider:
  typed fields + shallow-merge `extra_body` passthrough.
- #68679 adds ZDR via a policy-aware catalog + Desktop toggle —
  complementary: it is the ZDR *feature*; this is the generic *transport*
  that would also have carried it (and retires the one-off field PRs).

## Related PRs / issues

- #102341 — the original passthrough PR this continues (closed by its author)
- #68679 — request-time ZDR enforcement (open, complementary)
- #17247 — closed in favor of #68679
- #91800 — quantization passthrough (one-off)
- #72102 / #72118 / #94742 — max_price passthrough attempts (one-offs)
- #89430 / #99830 — Nous API 400s on unknown provider keys (why this is
  OpenRouter-scoped)
- #32757 — original zdr preference request
- #104159 — per-model `provider_routing.models` overlay (this feature
  extends it with passthrough keys)
