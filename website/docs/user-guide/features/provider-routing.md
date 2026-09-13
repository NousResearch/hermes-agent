---
title: Provider Routing
description: Configure OpenRouter provider preferences to optimize for cost, speed, or quality.
sidebar_label: Provider Routing
sidebar_position: 7
---

# Provider Routing

When using [OpenRouter](https://openrouter.ai) as your LLM provider, Hermes Agent supports **provider routing** — fine-grained control over which underlying AI providers handle your requests and how they're prioritized.

OpenRouter routes requests to many providers (e.g., Anthropic, Google, AWS Bedrock, Together AI). Provider routing lets you optimize for cost, speed, quality, or enforce specific provider requirements.

:::note
[Nous Portal](/integrations/nous-portal) decides routing centrally per model and does not accept caller-supplied provider preferences; Hermes never sends the `provider` object to Portal, so `provider_routing` is simply ignored there.
:::

## Configuration

Add a `provider_routing` section to your `~/.hermes/config.yaml`:

```yaml
provider_routing:
  sort: "price"           # How to rank providers
  only: []                # Whitelist: only use these providers
  ignore: []              # Blacklist: never use these providers
  order: []               # Explicit provider priority order
  require_parameters: false  # Only use providers that support all parameters
  data_collection: null   # Control data collection ("allow" or "deny")
```

:::info
Provider routing only applies when using OpenRouter. It has no effect on Nous Portal or direct provider connections (e.g., connecting directly to the Anthropic API).
:::

## Options

### `sort`

Controls how OpenRouter ranks available providers for your request.

| Value | Description |
|-------|-------------|
| `"price"` | Cheapest provider first |
| `"throughput"` | Fastest tokens-per-second first |
| `"latency"` | Lowest time-to-first-token first |

```yaml
provider_routing:
  sort: "price"
```

### `only`

Whitelist of provider slugs. When set, **only** these providers will be used. All others are excluded. Use the lowercase slug shown by OpenRouter for each provider.

```yaml
provider_routing:
  only:
    - "anthropic"
    - "google"
```

### `ignore`

Blacklist of provider names. These providers will **never** be used, even if they offer the cheapest or fastest option.

```yaml
provider_routing:
  ignore:
    - "together"
    - "deepinfra"
```

### `order`

Explicit priority order. Providers listed first are preferred. Unlisted providers are used as fallbacks.

```yaml
provider_routing:
  order:
    - "anthropic"
    - "google"
    - "amazon-bedrock"
```

Combining `:nitro` or `:floor` on the model id with `provider_routing.order` disables that variant's tier-admission. OpenRouter replaces the variant sort with your explicit `order`, so priority/flex endpoints are no longer admitted automatically. Hermes still sends `order` unchanged — the combination is valid when you list a tier-suffixed slug such as `openai/priority`, `openai/fast`, or `google-vertex/flex`. Sticky pin is not applied to `:nitro` / `:floor` models for the same reason.

### `require_parameters`

When `true`, OpenRouter will only route to providers that support **all** parameters in your request (like `temperature`, `top_p`, `tools`, etc.). This avoids silent parameter drops.

```yaml
provider_routing:
  require_parameters: true
```

### `data_collection`

Controls whether providers can use your prompts for training. Options are `"allow"` or `"deny"`.

```yaml
provider_routing:
  data_collection: "deny"
```

### Per-model overrides (`models`)

Pin a different provider set per model. Keys under `models` are model ids; each entry takes the same
`sort` / `only` / `ignore` / `order` / `require_parameters` / `data_collection` keys and overrides the
flat value for that model only. Anything you don't set per model falls through to the flat defaults.

```yaml
provider_routing:
  sort: "price"                      # applies to every model
  models:
    "openai/gpt-6-astra":
      only: ["openai"]               # never let a reseller serve this one
    "anthropic/claude-fable-5.1":
      only: ["anthropic"]
    "moonshotai/kimi-k2.6":
      order: ["moonshotai", "together"]
      sort: "throughput"
```

Matching is spelling-tolerant like `agent.reasoning_overrides` (`claude-fable-5.1` / `claude-fable-5-1`,
with or without the `openrouter/` prefix). The override follows the model the agent is *currently* on, so
`/model` switches, fallback activation, cron jobs, and delegated subagents on another model each get their
own pins. Pinning a provider per model also keeps OpenRouter's prompt cache warm: repeatedly hitting the
same upstream provider preserves the cached prefix, while load-balancing across providers loses it.
Edit `config.yaml` directly for these keys: model ids contain dots, which `hermes config set`
reads as path separators. `provider_routing` itself is an open dict, so `hermes config set provider_routing.sticky_order.enabled true` and `provider_routing.models.*` paths are accepted without `--force`.

## Practical Examples

### Optimize for Cost

Route to the cheapest available provider. Good for high-volume usage and development:

```yaml
provider_routing:
  sort: "price"
```

### Optimize for Speed

Prioritize low-latency providers for interactive use:

```yaml
provider_routing:
  sort: "latency"
```

### Optimize for Throughput

Best for long-form generation where tokens-per-second matters:

```yaml
provider_routing:
  sort: "throughput"
```

### Lock to Specific Providers

Ensure all requests go through a specific provider for consistency:

```yaml
provider_routing:
  only:
    - "anthropic"
```

### Avoid Specific Providers

Exclude providers you don't want to use (e.g., for data privacy):

```yaml
provider_routing:
  ignore:
    - "together"
    - "lepton"
  data_collection: "deny"
```

### Preferred Order with Fallbacks

Try your preferred providers first, fall back to others if unavailable:

```yaml
provider_routing:
  order:
    - "anthropic"
    - "google"
  require_parameters: true
```

### Sticky Order

A manual `order` turns off [OpenRouter's own sticky routing](https://openrouter.ai/docs/guides/best-practices/prompt-caching). Without a pin, the aggregator may hop to a different upstream provider on every request. Prompt cache is **not** shared across those providers, so a long agent session can repay the full prefill on each hop.

`sticky_order` is an opt-in pin (default **off**). Hermes sends only the current slug from the resolved `order` — intersected with `only` if you set one — and rotates to the next slug when that provider fails. Per-model overlay (`models.<id>`) is applied at request time first; the pin then lands **inside** that already-overlaid pool. Turn it on for long agent sessions that use `order` and pay a meaningful prefill.

A **logical request** is each model API call — including every tool-loop round inside a user turn, and the iteration-limit summary. An idle gap longer than `ttl_seconds` between any two of those calls resets the pin to `pool[0]`. In-request retry backoff (retries of the same API call) does not count as idle.

```yaml
provider_routing:
  order: ["z-ai/fp8", "novita/fp8"]
  sticky_order:
    enabled: true      # default false — opt-in
    ttl_seconds: 600   # default 600
```

| Event | What happens |
|-------|----------------|
| Timeout or overload | Rotate the pin, then retry the same logical request on the next slug. These two classes also use the deferred eager-fallback gate: model fallback waits until every pool slug has produced one of these errors |
| Server error (5xx) | Rotate the pin and retry on the next slug. `server_error` is **not** added to the transport-failure eager-fallback class — after retries it follows the normal (non-deferred) fallback path |
| Rate limit (429) or empty/invalid response | Stay on the current slug. Normal retry / model-fallback still applies — these errors do not walk the pin pool |
| Every slug has failed with timeout or overload | Eager transport-failure model fallback may fire (not before the last slug has failed). This delays OpenRouter-level fallback until the local pool is exhausted |
| Idle longer than `ttl_seconds` between logical requests | Reset to the first slug (every provider's cache is already cold). In-request retry backoff does not count as idle |
| A single eligible slug (pool of one) | Pins without rotation |

If `order ∩ only` is empty, `sticky_order` silently disables with a warning in the log. With no `order` at all, the feature is a no-op — it does not rotate over a bare `only` list.

The pin applies on full-agent paths that actually resolve a sticky pool (`providers_order` / `only`): the main conversation, session summary, subagents, and cron. Batch workers do not receive the flat constructor routing args (`providers_order`, `only`, and the rest) from `config.yaml` — those land only when the batch CLI passes them. The request-time per-model overlay (`provider_routing.models.<id>`) and sticky bind still apply from config when configured. State is per-agent and is never persisted to SessionDB or written into the prompt. A delegated child is a new agent and never inherits the parent's pin object.

Transient connection drops retry on the same provider first (bounded `HERMES_STREAM_RETRIES` micro-retries) so a warm prompt cache is not thrown away. The pin rotates only when the failure classifies as provider-unhealthy (`timeout` / `overloaded` / `server_error`) after those retries are exhausted.

Sticky is live only on the native OpenRouter **chat-completions** path. Nous Portal is excluded: the Portal ignores or rejects the `provider` object. `custom:` endpoints are not live — sticky requires the transport to actually send a `provider` object. Any other `api_mode` (`anthropic_messages`, `codex_responses`, and future modes) and direct provider connections are a no-op.

Sticky is not applied to `@preset/` models. Request-level provider routing for those models is unchanged. Sticky is also not applied to `:nitro` / `:floor` models — `provider.order` disables their tier-admission. Configured `order` is still sent unchanged; to keep tier endpoints eligible, name a tier-suffixed slug in `order` (see [`order`](#order)).

Sticky is also off for OpenRouter speed-tier models (`openai/gpt-6-astra`, `openai/gpt-6-astra-pro`, and their `-fast` / `-flex` slugs). The OpenRouter plugin owns `provider.only` for those endpoints; a sticky pin would make `order` and `only` disjoint.

Light auxiliary calls through `agent/auxiliary_client.py` (title generation, vision, compression, and other `auxiliary.<task>` work) never receive the sticky pin. Configure those independently under `auxiliary.<task>.extra_body`.

Precedence: per-model `provider_routing.models.<model>` wins over the flat `provider_routing` keys; sticky then narrows `order` to the active slug **inside** that already-resolved pool. When you also configured `only`, sticky narrows that list to the active slug as well.

## How It Works

Provider routing preferences (`order`, `only`, and the rest of the `provider` object) are passed via the `extra_body.provider` field on agent chat requests and iteration-limit summaries wherever the active profile emits provider prefs. (`extra_body` is the OpenAI Python SDK argument; it becomes the top-level `provider` object in the JSON request.) Per-model `provider_routing.models.<id>` overlays those prefs at request time from the current `agent.model`.

The sticky pin itself (`order=[active_slug]`, `allow_fallbacks=false`) is applied only on native OpenRouter. Regular (non-sticky) `provider_routing` `order`/`only` is unchanged and is still emitted wherever the profile already sends provider preferences.

Light auxiliary tasks that go through `agent/auxiliary_client.py` are configured independently under `auxiliary.<task>.extra_body`. Those calls never receive the sticky pin.

- **CLI mode** — configured in `~/.hermes/config.yaml`, loaded at startup
- **Gateway mode** — same config file, loaded when the gateway starts

The routing config is read from `config.yaml` and passed as parameters when creating the `AIAgent`:

```
providers_allowed  ← from provider_routing.only
providers_ignored  ← from provider_routing.ignore
providers_order    ← from provider_routing.order
provider_sort      ← from provider_routing.sort
provider_require_parameters ← from provider_routing.require_parameters
provider_data_collection    ← from provider_routing.data_collection
```

Per-model `models.<id>` overlays those constructor values again at request time. Sticky then narrows the overlaid `order` (and `only`, when set) to the active slug.

:::tip
You can combine multiple options. For example, sort by price but exclude certain providers and require parameter support:

```yaml
provider_routing:
  sort: "price"
  ignore: ["together"]
  require_parameters: true
  data_collection: "deny"
```
:::

## Default Behavior

When no `provider_routing` section is configured (the default), the aggregator uses its own default routing logic, which generally balances cost and availability automatically.

:::tip Provider Routing vs. Fallback Models
Provider routing controls which **sub-providers behind OpenRouter** handle your requests. For automatic failover to an entirely different provider when your primary model fails, see [Fallback Providers](/user-guide/features/fallback-providers).
:::
