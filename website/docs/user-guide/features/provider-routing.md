---
title: Provider Routing
description: Configure OpenRouter provider preferences to optimize for cost, speed, or quality.
sidebar_label: Provider Routing
sidebar_position: 7
---

# Provider Routing

When using [OpenRouter](https://openrouter.ai) as your LLM provider, Hermes Agent supports **provider routing** — fine-grained control over which underlying AI providers handle your requests and how they're prioritized.

OpenRouter routes requests to many providers (e.g., Anthropic, Google, AWS Bedrock, Together AI). Provider routing lets you optimize for cost, speed, quality, or enforce specific provider requirements. Comparing the two gateways? [Nous Portal vs OpenRouter](https://portal.nousresearch.com/compare/openrouter) covers pricing, free models and plans.

:::note
[Nous Portal](../../integrations/nous-portal.md) decides routing centrally per model and does not accept caller-supplied provider preferences; Hermes never sends the `provider` object to Portal, so `provider_routing` is simply ignored there.
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
own pins. Edit `config.yaml` directly for these keys: model ids contain dots, which `hermes config set`
reads as path separators.

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

## OpenAI Fast/Flex tier aliases

For OpenAI models served by multiple OpenRouter service tiers, Hermes accepts
`openai/<model>-flex` and `openai/<model>-fast` aliases. They are not separate
`/models` entries: Hermes rewrites the wire model to `openai/<model>` and pins
`provider.only` to `openai/flex` or `openai/fast` respectively. For example:

```
hermes chat --cli --provider openrouter -m openai/gpt-6.1-terra-flex
```

sends model `openai/gpt-6.1-terra` with `provider.only: ["openai/flex"]`.

This support is for the built-in `openrouter` provider. A custom `providers:` entry
whose base URL is OpenRouter uses a generic profile and does not apply the rewrite,
so select `--provider openrouter` when using a tier alias.

Only `openai/*` ids are interpreted. Other vendors use different endpoint tags,
and some ship real model ids ending in `-fast`, so those are left untouched.
Malformed (`openai/-flex`), chained (`...-fast-flex`) and routing-variant
(`...:floor`) ids are also left untouched rather than silently rewritten.

Preference and fallback semantics:

- The tier alias's `only` overrides a conflicting user `only`; other preferences
  (`ignore`, `order`, `sort`, `require_parameters`, `data_collection`) survive.
- `ignore: ["openai/flex"]` conflicts with a Flex alias and can leave no eligible
  endpoint. Do not combine them.
- Tier pins are exact: `allow_fallbacks` does not expand a one-element `only`
  pool to standard endpoints. That is intentional — a Flex request is a request
  for Flex pricing, not "Flex if convenient, otherwise standard price".
- Fallback models resolve independently; a tier pin does not leak to another
  model in the fallback chain.
- Raw `request_overrides` are applied after the profile and can replace the
  provider object or model. Treat that as an expert escape hatch.
- Local cost estimates for a tier alias may use the base model's `/models`
  pricing and therefore overstate Flex cost. OpenRouter bills the endpoint that
  actually serves the request; check the response `service_tier` and your
  OpenRouter usage for the charged rate.

## How It Works

Provider routing preferences are passed to OpenRouter on agent chat requests and iteration-limit summaries via the `extra_body.provider` field. (`extra_body` is the OpenAI Python SDK argument; it becomes the top-level `provider` object in the JSON request.) Auxiliary tasks such as compression and title generation are configured independently under `auxiliary.<task>.extra_body`.

- **CLI mode** — configured in `~/.hermes/config.yaml`, loaded at startup
- **Gateway mode** — same config file. Messaging-platform turns use the copy loaded when the gateway starts. A local `hermes chat` / TUI session reads it from its own profile when the gateway creates the session: an edit applies to the next new session without a gateway restart, and open sessions keep the values they were created with

The routing config is read from `config.yaml` and passed as parameters when creating the `AIAgent`:

```
providers_allowed  ← from provider_routing.only
providers_ignored  ← from provider_routing.ignore
providers_order    ← from provider_routing.order
provider_sort      ← from provider_routing.sort
provider_require_parameters ← from provider_routing.require_parameters
provider_data_collection    ← from provider_routing.data_collection
```

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
Provider routing controls which **sub-providers behind OpenRouter** handle your requests. For automatic failover to an entirely different provider when your primary model fails, see [Fallback Providers](./fallback-providers.md).
:::
