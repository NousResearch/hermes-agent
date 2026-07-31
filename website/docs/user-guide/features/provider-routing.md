---
title: Provider Routing
description: Configure OpenRouter or Nous Portal provider preferences to optimize for cost, speed, or quality.
sidebar_label: Provider Routing
sidebar_position: 7
---

# Provider Routing

When using [OpenRouter](https://openrouter.ai), [EU Router](/integrations/providers#eu-router), or [Nous Portal](/integrations/nous-portal) as your LLM provider, Hermes Agent supports **provider routing** — fine-grained control over which underlying AI providers handle your requests and how they're prioritized.

These aggregators route requests to many providers (e.g., Anthropic, Google, AWS Bedrock, Together AI, Scaleway, Mistral). Provider routing lets you optimize for cost, speed, quality, or enforce specific provider requirements — and, on EU Router, enforce EU-only data residency.

:::tip
Traffic routed through Nous Portal respects the same provider preferences — and Portal subscribers get 10% off token-billed providers.
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

  # EU Router only — silently dropped if your active model is on
  # OpenRouter or Nous Portal instead (see "How It Works" below).
  data_residency: null    # Restrict to a region: "eu", "eea", "de", "fr", ...
  eu_owned: false         # Restrict to EEA-owned providers only
  max_retention_days: null  # Max days a provider may retain request data (0 = none)
```

:::info
Provider routing only applies when using OpenRouter, EU Router, or Nous Portal. It has no effect with direct provider connections (e.g., connecting directly to the Anthropic API). The three EU-only fields (`data_residency`, `eu_owned`, `max_retention_days`) apply only when your active model is on EU Router — see [EU Router](/integrations/providers#eu-router) for setup.
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

### `data_residency` (EU Router only)

Restrict routing to providers hosted in a specific region: `"eu"`, `"eea"`, `"de"`, `"fr"`, and similar region codes. See [EU Router's routing docs](https://www.eurouter.ai/docs/concepts/routing) for the full list of supported values.

```yaml
provider_routing:
  data_residency: "eu"
```

### `eu_owned` (EU Router only)

When `true`, restricts routing to providers that are themselves EEA-owned — a stricter guarantee than `data_residency` alone (which only constrains where the infrastructure is physically located).

```yaml
provider_routing:
  eu_owned: true
```

### `max_retention_days` (EU Router only)

Caps how many days a provider is allowed to retain your request data. `0` means no retention at all — this is a meaningful, explicit value, not "unset".

```yaml
provider_routing:
  max_retention_days: 0
```

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

### EU-Only Routing (GDPR / Data Residency)

Force every request through EU-owned infrastructure — requires [EU Router](/integrations/providers#eu-router) as your model provider:

```yaml
model:
  provider: "eurouter"
  default: "mistral-large-3"

provider_routing:
  data_residency: "eu"
  eu_owned: true
  max_retention_days: 0
```

## How It Works

Provider routing preferences are passed to OpenRouter, EU Router, or Nous Portal on agent chat requests and iteration-limit summaries via the `extra_body.provider` field. (`extra_body` is the OpenAI Python SDK argument; it becomes the top-level `provider` object in the JSON request.) Auxiliary tasks such as compression and title generation are configured independently under `auxiliary.<task>.extra_body`.

- **CLI mode** — configured in `~/.hermes/config.yaml`, loaded at startup
- **Gateway mode** — same config file, loaded when the gateway starts
- **Desktop mode** — same config file, read by the bundled backend

The routing config is read from `config.yaml` and passed as parameters when creating the `AIAgent`:

```
providers_allowed  ← from provider_routing.only
providers_ignored  ← from provider_routing.ignore
providers_order    ← from provider_routing.order
provider_sort      ← from provider_routing.sort
provider_require_parameters ← from provider_routing.require_parameters
provider_data_collection    ← from provider_routing.data_collection
provider_data_residency     ← from provider_routing.data_residency      (EU Router only)
provider_eu_owned           ← from provider_routing.eu_owned            (EU Router only)
provider_max_retention_days ← from provider_routing.max_retention_days  (EU Router only)
```

:::info Per-provider filtering
Every field in `provider_routing` is read once and shared across all aggregator-style providers, but each provider only receives the keys it actually understands: `data_residency`, `eu_owned`, and `max_retention_days` are filtered out before a request goes to OpenRouter or Nous Portal, and never sent in the first place if your active model isn't on EU Router. Setting these globally is safe even if you switch between EU Router and other providers.
:::

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
Provider routing controls which **sub-providers behind OpenRouter or Nous Portal** handle your requests. For automatic failover to an entirely different provider when your primary model fails, see [Fallback Providers](/user-guide/features/fallback-providers).
:::
