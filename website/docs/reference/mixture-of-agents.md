---
title: Mixture of Agents (MoA)
---

Mixture of Agents (MoA) runs several reference models in parallel, then sends their responses to a separate aggregator model that synthesizes the final answer.

Use it for genuinely hard prompts where diversity helps: multi-step reasoning, architecture tradeoffs, difficult debugging, complex coding tasks, or synthesis across viewpoints.

Configuration lives under `moa:` in `config.yaml`.

## Shipped default config

```yaml
moa:
  enabled: true
  reference_models:
    - model: "anthropic/claude-opus-4.7"
      provider: "openrouter"
      reasoning: "xhigh"
    - model: "google/gemini-2.5-pro"
      provider: "openrouter"
      reasoning: "xhigh"
    - model: "openai/gpt-5.5-pro"
      provider: "openrouter"
      reasoning: "xhigh"
    - model: "deepseek/deepseek-v3.2"
      provider: "openrouter"
      reasoning: "xhigh"
  aggregator_model:
    model: "anthropic/claude-opus-4.7"
    provider: "openrouter"
    reasoning: "xhigh"
  reference_temperature: 0.6
  aggregator_temperature: 0.4
```

This is the fallback roster Hermes ships with when your `moa:` block is absent.
`min_successful_references` is omitted by default and resolves to `2` for this
four-model roster.

Notes:

- `enabled: false` disables the tool without deleting the roster.
- String shorthand is allowed for convenience:

```yaml
moa:
  reference_models:
    - "anthropic/claude-opus-4.7"
  aggregator_model: "anthropic/claude-opus-4.7"
```

Those shorthand entries default to `provider: openrouter`.

## Provider routing

Each entry chooses its own provider.

That lets you mix:

- OpenRouter for breadth and reliability
- OpenAI Codex for GPT-5 via a Codex subscription
- Anthropic direct for native Claude routing
- Other configured providers supported by Hermes

Preflight is roster-aware: Hermes checks the providers actually referenced by
your MoA config. A Codex-only roster does not require `OPENROUTER_API_KEY`,
while a mixed roster needs credentials for every provider it touches.

If a provider/model pairing is not in the known provider catalog, Hermes logs a warning but does not block the call. This is intentional: catalogs can lag behind newly released model slugs.

## Reasoning per model

Each entry may optionally set `reasoning`.

Accepted values are `minimal`, `low`, `medium`, `high`, `xhigh`, and `none`.
`max` is not accepted for MoA entries.

Behavior:

- omitted / `null` / `""` => do not send any reasoning parameter; use the provider default
- `none` => send an explicit reasoning-disabled config when supported
- invalid value => load-time config error

Reasoning is translated per provider:

- OpenRouter => `extra_body.reasoning` only for known reasoning-capable model families
- Nous / AI Gateway => `extra_body.reasoning`; Nous omits explicit disabled reasoning
- Copilot / Copilot ACP => `extra_body.reasoning` with effort remapped to the model's supported set
- OpenAI Codex / Anthropic => `reasoning_config` passed to the adapter layer
- Custom OpenAI-compatible endpoints => top-level `reasoning_effort`, or `extra_body.think=false` for `none`
- Kimi / Moonshot => top-level `reasoning_effort` plus `extra_body.thinking`; `minimal` maps to `low`, and `xhigh` maps to `high`
- Tencent TokenHub => top-level `reasoning_effort`, limited to `low`, `medium`, and `high`
- LM Studio => top-level `reasoning_effort` only when the model advertises reasoning support

:::note Silent-accept providers
OpenRouter proxies to many backends that silently accept and drop `extra_body.reasoning` without error; if a reasoning setting seems inert, cross-check against provider-side logs or billing.
:::

## Request compatibility

MoA aligns its direct chat-completions requests with Hermes' provider contracts:

- Kimi / Moonshot models omit `temperature`; the endpoint chooses the mode-specific default.
- Anthropic models that reject sampling parameters omit `temperature`.
- Custom endpoints hosted at `api.openai.com` receive `max_completion_tokens`; other OpenAI-compatible endpoints receive `max_tokens`.
- Nous requests include the `product=hermes-agent` attribution tag.
- If a provider rejects `temperature` or reasoning parameters with a clear unsupported-parameter error, MoA retries immediately without that parameter and remembers the drop for the same provider/model/endpoint for the rest of the process.

## Operational notes

- The aggregator is the final answer. If the aggregator fails, the whole MoA call fails.
- `min_successful_references` is validated against the roster size.
- When the key is omitted, Hermes uses an adaptive default: `2` for multi-model rosters, `1` for single-model rosters.

## `/model` and `/reasoning_effort` do not cascade into MoA

MoA has its own roster and per-entry reasoning configuration.

Important:

- `/model` changes the main agent model, not the MoA reference roster or aggregator
- `/reasoning_effort` changes the main agent loop, not the `moa.*.reasoning` entries

If you want MoA to track your main model or reasoning preferences, update both places.

## Codex warning

Routing one or more MoA slots through `openai-codex` can burn through a Codex plan quickly.

Because MoA fans out in parallel, routing multiple reference models through Codex multiplies consumption against the plan's daily cap.

If the aggregator itself is Codex-routed and fails, the whole MoA tool fails.

## Related docs

- [Tools Reference](./tools-reference.md)
- [Toolsets Reference](./toolsets-reference.md)
- [Providers](../integrations/providers.md)
