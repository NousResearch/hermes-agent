---
sidebar_position: 11
title: Model Catalog
description: Remotely-hosted manifest that keeps curated model picker lists current between releases.
---

# Model Catalog

Hermes fetches curated model lists from a JSON manifest hosted alongside the docs site. This lets maintainers update picker lists without shipping a new `hermes-agent` release.

The manifest is read in two ways:

- **OpenRouter and Nous Portal** take their whole curated list from it.
- **Every other built-in provider** (Anthropic, OpenAI API, Bedrock, Gemini, OpenCode Zen and the rest) gets *added* rows. A model published after your release shows up in the picker; nothing your install already lists is removed.

When the manifest is unreachable (offline, network blocked, hosting failure), Hermes silently falls back to the in-repo snapshot that ships with the CLI. The manifest never breaks the picker — worst case you see whatever list was bundled with your installed version.

## Live manifest URL

```
https://hermes-agent.nousresearch.com/docs/api/model-catalog.json
```

Published on every merge to `main` via the existing `deploy-site.yml` GitHub Pages pipeline. The source of truth lives in the repo at `website/static/api/model-catalog.json`.

## Schema

```json
{
  "version": 1,
  "updated_at": "2026-04-25T22:00:00Z",
  "metadata": {},
  "providers": {
    "openrouter": {
      "metadata": {},
      "models": [
        {"id": "z-ai/glm-5.3",         "description": "default", "default": true},
        {"id": "moonshotai/kimi-k3",   "description": "recommended", "metadata": {}},
        {"id": "openai/gpt-5.4",       "description": ""}
      ]
    },
    "nous": {
      "metadata": {},
      "models": [
        {"id": "z-ai/glm-5.3", "default": true},
        {"id": "anthropic/claude-opus-4.7"},
        {"id": "moonshotai/kimi-k3"}
      ]
    },
    "anthropic": {
      "models": [
        {"id": "claude-opus-5-5"},
        {"id": "claude-sonnet-5-5"}
      ]
    }
  }
}
```

Field notes:

- **`version`** — integer schema version. Future schemas bump this; Hermes refuses manifests with versions it doesn't understand and falls back to the hardcoded snapshot.
- **`metadata`** — free-form dict at the manifest, provider, and model level. Any keys. Hermes ignores unknown fields, so you can annotate entries (`"tier": "paid"`, `"tags": [...]`, etc.) without coordinating a schema change.
- **`description`** — OpenRouter-only. Drives picker badge text (`"recommended"`, `"free"`, `"default"`, or empty). Nous Portal doesn't use this.
- **`default`** — exactly one entry per provider may carry `"default": true`. That model is the **silent default**: what Hermes lands on when the user never selected a model (GUI onboarding confirm card, `provider` configured with no `model`, empty `model.default`). Read cache-only at runtime (`get_default_model_from_cache`) so hot resolution paths never hit the network; when no cached manifest exists, Hermes falls back to the in-repo `PREFERRED_SILENT_DEFAULT_MODEL` constant, which must match the labeled entry. This lets maintainers rotate the silent default without shipping a release. It is deliberately a capable low-cost model, never the priciest flagship. Only the OpenRouter and Nous Portal blocks carry the label; on any other block it is ignored.
- **`requires`** — optional list on a model entry, for the added-row providers. It names the model sets (below) the model cannot be driven correctly without. A build that does not read one of them skips the entry, so an older release never lists a model it cannot drive.
- **Pricing:** the manifest's `pricing` section can fill a missing standard or fast-tier rate without replacing a bundled rate for that tier. See [Pricing for new models](#pricing-for-new-models) for the lookup order.
- **Context length and capabilities** still come from live provider APIs and models.dev first. The manifest's `context_lengths` and `model_metadata` tables extend the last fallback; see below for how context keys match and which entry wins.

## Providers other than OpenRouter and Nous Portal

Each built-in provider row in `_PROVIDER_MODELS` is published as a `{"models": [{"id": ...}]}` block under the provider's Hermes id. On the client the block only adds to the row the install ships: the ids the install does not have come first, in manifest order, then the shipped row exactly as it is. The manifest does not reorder or respell a shipped id.

The merge feeds the places that *list* models: the `/model` picker on every surface, `hermes model`, and the validation that decides whether a typed model id is known. An added id is accepted on a provider whose picker offers it, and not on one whose live listing stands alone.

It deliberately does not feed anything that *chooses* a model for you:

- The default model for a provider with no model configured and the short aliases (`sonnet`, `opus`) read the built-in list.
- The Desktop onboarding recommendation chooses from the list the picker would have had without the manifest's rows, in that order, for the requested profile.
- Provider auto-detection for a bare model name (`/model some-model`) treats an added row as something to offer, not as proof the current provider serves it. The provider's model cache records which of its ids are there only because of the manifest, as of the last listing the provider answered, so a discovery outage does not change where a name goes either.

So a manifest edit can never change what an unconfigured install runs, what it costs, or which provider a typed name lands on.

Other rules:

- The manifest only extends a list. A provider with no built-in row gets nothing from it, which keeps third-party provider plugins on their own lists, including a plugin whose name matches a published block.
- `openai-codex`, `xai`, `xai-oauth` and `moa` are not published and are not read: their rows come from the account or from another source. On `openai` and `openai-api`, an id that only the account's own `/models` listing may advertise (`gpt-6-astra`) is never taken from the manifest.
- Vercel AI Gateway shows an added id once the gateway's live catalog serves it, after the shipped rows.
- These rows are read from the disk cache only and never trigger a fetch of their own. The fetch points in the table below keep the cache warm.
- Per-provider override URLs (below) apply to OpenRouter and Nous Portal only.
- Six providers (`zai`, `nvidia`, `moonshot`, `deepseek`, `opencode-zen`, `huggingface`) show their plugin's own fallback list while their live `/models` probe is unavailable. The manifest rows join once the probe answers.

Not published, on purpose: `openai-codex` (only the account's live Codex catalog may advertise a model), `xai` and `xai-oauth` (their rows come from the models.dev cache) and `moa` (a preset name, not a model list).

## Metadata and context windows for new models

Two top-level tables publish what the built-in tables hold on `main`:

```json
{
  "context_lengths": {
    "default":     {"gpt-6.2-sol": 1050000},
    "bedrock":     {"anthropic.claude-opus-5": 1000000},
    "codex_oauth": {"gpt-6.2-sol": 500000}
  },
  "model_metadata": {
    "openai": {
      "gpt-6.2-sol": {
        "limit": {"context": 1050000, "output": 128000},
        "modalities": {"input": ["text", "image"], "output": ["text"]},
        "tool_call": true,
        "reasoning": true
      }
    }
  }
}
```

- **`context_lengths`** mirrors the three hardcoded context-window tables (the general one, Bedrock's, and the Codex OAuth fallback). Keys match the way the built-in tables do: a key is a substring of the model id and the longest matching key wins.
- **`model_metadata`** mirrors the built-in metadata entries, keyed by models.dev provider id and lowercased model id, in the models.dev entry shape.

Both only fill gaps. A context-table key or a provider/model metadata entry your install ships keeps its own value, under any casing, even when the manifest disagrees. What the manifest adds is the row for a model the install has never heard of.

One consequence is worth spelling out. Context keys match as substrings and the longest key wins, so a new, more specific key (`glm-9-preview`) does take over from a shipped family catch-all (`glm`) for the models it names. That is the gap being filled: the catch-all was only a guess for that model. A model no key matches still gets the generic 256K window, and a model no metadata source knows stays "unknown" for vision and reasoning.

A malformed row is dropped rather than half-used: a context value that is not a positive integer, a blank key, or a metadata entry without a valid `limit.context`. A `cost` field in a metadata entry is ignored; prices never come from this table.

The order of sources is unchanged: an explicit `model.context_length`, live provider endpoints and models.dev all still come first. These tables sit where the hardcoded tables already sat, at the end.

## Model sets: request handling for new models

Some request handling is keyed on an exact list of model ids in the code: which models take the `low`..`max` effort ladder, which accept a fast tier, which must go through a different API. A new model that behaves like an existing one used to need a release just to join those lists. `model_sets` publishes the lists:

```json
{
  "model_sets": {
    "openai_gpt62_contract":    ["gpt-6.2-astra", "gpt-6.2-sol"],
    "bedrock_openai_responses": ["openai.gpt-6.1-sol"],
    "anthropic_fast_mode":      ["claude-opus-5-5"],
    "openai_ultrafast":         ["gpt-6.1-sol", "gpt-6.2-sol"],
    "openai_fast_eu":           ["gpt-6.1-sol", "gpt-6.2-sol"]
  }
}
```

| Set | What an id in it gets |
|---|---|
| `openai_gpt62_contract` | The GPT-6.2 request contract: the `low`..`max` effort ladder (no `none`), the strict request sanitizer on official OpenAI, Codex and US/EU residency routes, account discovery before the Codex picker lists it, and no Ultrafast on the EU host |
| `bedrock_openai_responses` | Treated as a Bedrock-hosted OpenAI model (Mantle Responses for the bare id) |
| `anthropic_fast_mode` | Accepts Anthropic fast mode (`speed: "fast"`) |
| `openai_ultrafast` | Permits `service_tier: "ultrafast"` for an otherwise fast-eligible OpenAI model and route |
| `openai_fast_eu` | Permits an otherwise eligible fast tier on `eu.api.openai.com`; the GPT-6.2 EU Ultrafast exclusion still applies |

Rules:

- **Ids can be added, sets cannot.** A set exists only because code reads it. A name your build does not know is ignored, and a model that needs a new set still needs a release.
- **A name means one behaviour, for good.** `requires` checks the name, not a revision of the code behind it. If the request handling for a set changes, the release that changes it gives the new behaviour a new set name and uses that name in `requires`. Older builds then skip the row instead of driving the model the old way.
- **A tier set is one condition, not the whole decision.** Being in `openai_ultrafast` or `openai_fast_eu` does not make a model fast-capable on its own: the built-in family and route checks still have to pass.
- **Nothing is removed.** An id your install ships in a set stays in it.
- **Hosts stay in the code.** A set decides which *model* may use a tier. Which *endpoint* may receive it is still decided by the built-in route checks, so the manifest cannot send a billable tier to a host the code does not already allow.
- **A fast tier is still opt-in.** A set only makes `/fast` available for a model. Nothing is billed at a higher tier unless you ask for it.
- **Ids are matched the way the built-in list is:** lowercased and bare, after the same prefix and suffix stripping the code already does.

A picker entry for a model in a set marked as required for driving it (`openai_gpt62_contract`, `bedrock_openai_responses`) carries that name under `requires`. That is what keeps a build from listing a model whose contract it has not learned yet. When publishing a new id by hand, add it to the matching `model_sets` list in the same update: a `requires` entry checks that the build understands the set name, it does not add the id to that set.

The publisher derives `requires` from these exact-id sets only. When a model's handling lives somewhere else (a provider plugin, a prefix check), nothing derives it. The release that adds that handling should register a required set for the model's ids, even if the code keeps matching by prefix, so that the picker row is gated the same way.

## Pricing for new models

`pricing` publishes the bundled rate snapshots on `main`, so the cost of a model released after your install does not read "unknown" until the next release:

```json
{
  "pricing": {
    "official_docs": {
      "openai": {
        "gpt-6.2-sol": {
          "input_cost_per_million": "2.00",
          "output_cost_per_million": "10.00",
          "cache_read_cost_per_million": "0.10",
          "cache_write_cost_per_million": "2.50",
          "tier_threshold_tokens": 272000,
          "input_cost_per_million_above": "4.00",
          "source_url": "https://developers.openai.com/api/docs/models/gpt-6.2-sol",
          "pricing_version": "openai-gpt-6.2-2026-11"
        }
      }
    },
    "anthropic_fast_mode": {"claude-opus-5-5": {"input_cost_per_million": "8.00", "output_cost_per_million": "40.00"}},
    "openai_ultrafast":    {"gpt-6.1-sol":     {"input_cost_per_million": "12.00", "output_cost_per_million": "60.00"}}
  }
}
```

- **`official_docs`** is the standard-tier snapshot, keyed by billing provider (`openai`, `anthropic`, `google`, `bedrock`, ...) and lowercased model id. Field names are the ones the code uses for a rate entry; rates are decimal strings. Token rates are USD per million tokens; `request_cost` is USD per request.
- **`anthropic_fast_mode`** and **`openai_ultrafast`** are the fast-tier snapshots, keyed by model id. They are used when the response reports it was served at that tier.

Rules:

- **A bundled rate is never replaced.** The manifest is consulted only after the install's own snapshot has no row for the model, under the same name normalizing (dotted Anthropic versions, Bedrock region prefixes). Price corrections still ship with a release.
- **Standard rates are tried in order:** the install's bundled snapshot, the configured endpoint's `/models` price, the published manifest snapshot, then models.dev on eligible first-party routes. If a relay supplies no price, a vendor snapshot can still produce an estimate; it does not establish what the relay charges.
- **Malformed rates are ignored.** Each rate must be a finite, non-negative decimal, and a manifest entry needs both input and output token rates. Hermes discards the whole entry if it fails validation. The cost is unknown only if no usable rate is available for the usage being estimated.
- OpenRouter costs and subscription-included routes do not read this section.
- GPT-6.2's fast-tier rates are computed from its standard rates in code (they carry a dated launch discount), so they are not in these tables. A later model on the same contract uses the same calculation, but only on a vendor snapshot rate (bundled or published). When the standard rate came from somewhere else, such as an endpoint's own listing or models.dev, the Priority, Fast and Ultrafast costs show as unknown: that rate says nothing about the tier price.

## Fetch behavior

| When | What happens |
|---|---|
| `/model` or `hermes model` | Fetches if disk cache is stale, else uses cache |
| Gateway running | Background refresh every `ttl_minutes` (default 20), so the picker never lags the published manifest by more than one window |
| Disk cache fresh (< TTL) | No network hit |
| Network failure with cache | Silent fallback to cache, one log line |
| Network failure, no cache | Silent fallback to in-repo snapshot |
| Manifest fails schema validation | Treated as unreachable |

Cache location: `~/.hermes/cache/model_catalog.json`.

### Per-provider model lists in the GUI picker

The Desktop, TUI and dashboard pickers (`model.options`) build each provider's row from the disk-cached live catalog (`~/.hermes/provider_models_cache.json`) or, when nothing is cached yet, the curated list. Opening the picker never waits on a provider's `/v1/models` probe or on an auth probe: stale or missing catalogs are refreshed in a background thread and land on the next open, so one slow, rate-limited or unreachable provider cannot hold the whole picker on its loading state. **Refresh models** (or `/model --refresh`) is the explicit action that busts the cache and probes every provider live.

## Config

```yaml
model_catalog:
  enabled: true
  url: https://hermes-agent.nousresearch.com/docs/api/model-catalog.json
  ttl_minutes: 20
  providers: {}
```

Set `enabled: false` to disable remote fetch entirely and always use the in-repo snapshot (this also disables the gateway's background refresh). `ttl_minutes` sets both the cache lifetime and the gateway refresh cadence; the legacy `ttl_hours` key is still honoured if you set it explicitly.

### Per-provider override URLs

Third parties can self-host their own curation list using the same schema. Point a provider at a custom URL:

```yaml
model_catalog:
  providers:
    openrouter:
      url: https://example.com/my-openrouter-curation.json
```

The overriding manifest only needs to populate the provider block(s) it cares about. Other providers continue to resolve against the master URL.

### Hiding providers from the picker

`excluded_providers` lets you hide specific providers from the `/model` picker even when valid credentials exist. Useful when credentials are present for legacy or testing providers that shouldn't appear in normal use (e.g. an old Copilot or OpenRouter token still cached in `auth.json` or discovered via the `gh` CLI).

```yaml
model_catalog:
  excluded_providers:
    - copilot
    - openrouter
    - openai
```

The exclusion is matched case-insensitively against every key a provider can surface under — the Hermes id and models.dev id (built-in mapped providers), the overlay pid and resolved Hermes slug (overlay providers), and the canonical slug (canonical providers) — so a single entry like `copilot` hides the provider regardless of which section emits it. It is honored by every `/model` picker surface: the gateway interactive/text pickers, the TUI picker, and the interactive `hermes model` CLI picker. An empty list (or omitting the key) has no effect.

## Updating the manifest

Maintainers:

```bash
# Re-generate from the in-repo hardcoded lists (keeps manifest in sync after
# editing OPENROUTER_MODELS or any _PROVIDER_MODELS row in hermes_cli/models_catalog_static.py).
python scripts/build_model_catalog.py
```

Then PR the resulting change to `website/static/api/model-catalog.json` to `main`. The docs site auto-deploys on merge and the new manifest is live within a few minutes.

You can also hand-edit the JSON directly for fine-grained metadata changes that don't belong in the in-repo snapshot — the generator script is a convenience, not the single source of truth.
