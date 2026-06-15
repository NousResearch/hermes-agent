---
title: Provider Routing
description: Configure OpenRouter provider preferences to optimize for cost, speed, or quality.
sidebar_label: Provider Routing
sidebar_position: 7
---

# Provider Routing

When using [OpenRouter](https://openrouter.ai) as your LLM provider, Hermes Agent supports **provider routing** — fine-grained control over which underlying AI providers handle your requests and how they're prioritized.

OpenRouter routes requests to many providers (e.g., Anthropic, Google, AWS Bedrock, Together AI). Provider routing lets you optimize for cost, speed, quality, or enforce specific provider requirements.

:::tip
Traffic routed through [Nous Portal](/integrations/nous-portal) still respects per-model routing and priority configs — and Portal subscribers get 10% off token-billed providers.
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
Provider routing only applies when using OpenRouter. It has no effect with direct provider connections (e.g., connecting directly to the Anthropic API).
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

Whitelist of provider names. When set, **only** these providers will be used. All others are excluded.

```yaml
provider_routing:
  only:
    - "Anthropic"
    - "Google"
```

### `ignore`

Blacklist of provider names. These providers will **never** be used, even if they offer the cheapest or fastest option.

```yaml
provider_routing:
  ignore:
    - "Together"
    - "DeepInfra"
```

### `order`

Explicit priority order. Providers listed first are preferred. Unlisted providers are used as fallbacks.

```yaml
provider_routing:
  order:
    - "Anthropic"
    - "Google"
    - "AWS Bedrock"
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
    - "Anthropic"
```

### Avoid Specific Providers

Exclude providers you don't want to use (e.g., for data privacy):

```yaml
provider_routing:
  ignore:
    - "Together"
    - "Lepton"
  data_collection: "deny"
```

### Preferred Order with Fallbacks

Try your preferred providers first, fall back to others if unavailable:

```yaml
provider_routing:
  order:
    - "Anthropic"
    - "Google"
  require_parameters: true
```

## Cost-aware model routing checklist

Use telemetry before changing routing. For delegated work, Hermes writes handoff telemetry events that include `trace_id`, parent/session correlation (`parent_session_id`, `parent_task_id`, `parent_subagent_id`, `subagent_id`), `model`, `provider`, `api_mode`, token breakdowns, `duration_seconds`, `cost.estimated_usd`, `cost.status`, `cost.source`, `status`, `exit_reason`, and `result`. Review those signals after a few representative tasks so routing decisions come from observed volume, latency, cost, and success/failure patterns rather than pricing tables alone.

Recommended lanes:

- **Cheap/default lane** — keep routine chat, summarization, extraction, simple code edits, and broad subagent fan-out on the configured cheap default such as `gpt-5.4-mini` or the current low-cost equivalent. This is the right lane when telemetry shows modest `tokens`, acceptable `duration_seconds`, low `cost.estimated_usd`, and successful `result.status` / `result.summary` outcomes.
- **Stronger model lane** — escalate to a stronger model when recent telemetry for the cheap lane shows repeated poor `result` quality, retries, tool-loop churn, high reasoning complexity, failed coding/debugging attempts, or tasks whose expected output cost is lower than the cost of rework.
- **Review lane** — use a review-oriented model or dedicated reviewer for security-sensitive changes, large refactors, production-impacting config, PR review, or any task where a cheap first pass succeeded but the `trace_id` should be auditable before merge/deploy.
- **Fallback/provider lane** — use `fallback_providers` for reliability failures and provider outages; use OpenRouter `provider_routing` (`sort: "price"`, `latency`, `throughput`, `order`, `only`, `ignore`) only to choose the underlying OpenRouter provider for the same model request.

Practical operating loop:

1. Start with the cheap default for exploratory or parallel work.
2. Inspect telemetry grouped by `model` and `trace_id`: total `tokens`, `duration_seconds`, `cost.estimated_usd`, and `result` status/quality.
3. Escalate only the task type or trace pattern that justifies it; do not globally raise `model.default` because one task was hard.
4. Keep review lanes explicit and auditable by preserving the original `trace_id` in notes, PRs, or summaries.
5. Re-check telemetry after the change; keep the cheaper lane if stronger routing does not reduce failures, latency, or total cost.

## How It Works

Provider routing preferences are passed to the OpenRouter API via the `extra_body.provider` field on every API call. This applies to both:

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
  ignore: ["Together"]
  require_parameters: true
  data_collection: "deny"
```
:::

## Default Behavior

When no `provider_routing` section is configured (the default), OpenRouter uses its own default routing logic, which generally balances cost and availability automatically.

:::tip Provider Routing vs. Fallback Models
Provider routing controls which **sub-providers within OpenRouter** handle your requests. For automatic failover to an entirely different provider when your primary model fails, see [Fallback Providers](/user-guide/features/fallback-providers).
:::
