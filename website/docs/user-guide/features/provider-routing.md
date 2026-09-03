---
title: Provider Routing
description: Configure OpenRouter or Nous Portal provider preferences to optimize for cost, speed, or quality.
sidebar_label: Provider Routing
sidebar_position: 7
---

# Provider Routing

When using [OpenRouter](https://openrouter.ai) or [Nous Portal](/integrations/nous-portal) as your LLM provider, Hermes Agent supports **provider routing** — fine-grained control over which underlying AI providers handle your requests and how they're prioritized.

OpenRouter routes requests to many providers (e.g., Anthropic, Google, AWS Bedrock, Together AI). Provider routing lets you optimize for cost, speed, quality, or enforce specific provider requirements.

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
```

:::info
Provider routing only applies when using OpenRouter or Nous Portal. It has no effect with direct provider connections (e.g., connecting directly to the Anthropic API).
:::

## Scope settings to the selected main provider

Use `main_provider_policies.<provider>` when auxiliary, fallback, context, or
provider-routing settings should activate only while that provider is the
agent's **primary** route. The base profile remains authoritative for every
other main provider, so switching back restores its original behavior without
rewriting config.

```yaml
main_provider_policies:
  openrouter:
    enabled: true
    model_overrides:
      z-ai/glm-5.3-flash:
        context_length: 1048576
    provider_routing:
      require_parameters: true
      data_collection: deny
    auxiliary:
      compression:
        provider: openrouter
        model: deepseek/deepseek-v4-flash-0731
        reasoning_effort: low
      review:
        provider: anthropic
        model: claude-opus-5
    fallback_providers:
      - provider: openai-codex
        model: gpt-5.6-sol-900k
      - provider: anthropic
        model: claude-opus-5
```

The provider policy is resolved from the live main route when an agent is
created, including a session `/model` switch, delegated child, gateway session,
or cron agent. It is **not** activated merely because OpenRouter appears as a
fallback or auxiliary provider. Within `auxiliary`, the scalar
`free_only` and `openrouter_model` settings govern OpenRouter auto-fallback for
that live main route alongside task-specific entries such as `compression` and
`review`.

Administrator-managed config remains authoritative after policy projection.
For example, a managed `provider_routing.data_collection: deny` or managed
auxiliary/compression leaf cannot be relaxed by a user-defined provider policy.

`model_overrides` is keyed by exact model ID. Use it for model-specific values
such as `context_length`; a pin for one OpenRouter model never leaks to a
smaller model on the same aggregator. Policies cannot change `model.provider`
or `model.default` — model selection remains explicit through `hermes model` or
`/model`.

Set `enabled: false` to keep a policy configured but dormant. `hermes config
get` continues to show the base profile values; the overlay is a runtime
projection and is never written over those base values.

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

## How It Works

Provider routing preferences are passed to OpenRouter or Nous Portal on agent chat requests and iteration-limit summaries via the `extra_body.provider` field. (`extra_body` is the OpenAI Python SDK argument; it becomes the top-level `provider` object in the JSON request.) Auxiliary tasks such as compression and title generation are configured independently under `auxiliary.<task>.extra_body`.

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
