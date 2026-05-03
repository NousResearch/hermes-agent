---
title: Mixture of Agents
description: Run a hard prompt through multiple frontier LLMs in parallel, then synthesize the responses with an aggregator.
sidebar_label: Mixture of Agents
sidebar_position: 16
---

# Mixture of Agents

The **Mixture of Agents** (MoA) tool runs your prompt through several frontier models in parallel (the *reference* layer), then feeds the collected answers to a single aggregator model that synthesizes a final response. It is most useful for hard one-shot questions where you want to trade latency and token spend for higher answer quality.

MoA is part of the `moa` toolset. It is OFF by default; enable it from the setup checklist or by adding `moa` under `enabled_toolsets` in `config.yaml`.

## Configuring the roster

MoA is configured entirely from `~/.hermes/config.yaml` under the `moa:` key. The module constants inside `tools/mixture_of_agents_tool.py` are fallbacks; do not edit them.

```yaml
moa:
  enabled: true
  reference_models:
    - {model: anthropic/claude-opus-4.7,       provider: openrouter, reasoning: xhigh}
    - {model: google/gemini-2.5-pro,           provider: openrouter, reasoning: xhigh}
    - {model: openai/gpt-5.5-pro,              provider: openrouter, reasoning: xhigh}
    - {model: deepseek/deepseek-v3.2,          provider: openrouter, reasoning: xhigh}
  aggregator_model:
    model: anthropic/claude-opus-4.7
    provider: openrouter
    reasoning: xhigh
  reference_temperature: 0.6
  aggregator_temperature: 0.4
  min_successful_references: 2
```

Every entry in `reference_models` and `aggregator_model` accepts:

- `model` — the provider-native model slug.
- `provider` — any provider accepted by the main agent's router (`openrouter`, `openai-codex`, `nous`, `anthropic`, `copilot`, `ai-gateway`, `custom`, …). Defaults to `openrouter` if omitted.
- `reasoning` *(optional)* — one of `minimal`, `low`, `medium`, `high`, `xhigh`, `none`. Omit the key (or set `null` / `""`) to send no MoA-specific reasoning override. `max` is not accepted for MoA entries.

A bare string is accepted as shorthand for `{model: <string>, provider: openrouter}`.

MoA translates `reasoning` to the provider's request shape. OpenRouter, Nous,
AI Gateway, and Copilot use `extra_body.reasoning` where supported; Codex and
Anthropic receive adapter-facing `reasoning_config`; Kimi / Moonshot and
Tencent TokenHub use top-level `reasoning_effort` (Kimi maps `minimal` to
`low` and `xhigh` to `high`); LM Studio only receives a reasoning field when
the model advertises support.

MoA also applies provider-specific request compatibility rules: Kimi /
Moonshot and no-sampling Anthropic models omit `temperature`, direct OpenAI
custom endpoints receive `max_completion_tokens`, and Nous requests include the
`product=hermes-agent` tag. Clear unsupported-parameter errors trigger one
immediate retry without the rejected temperature or reasoning field and are
cached for that provider/model/endpoint for the rest of the process.

### Kill switch

Set `moa.enabled: false` to disable the tool without touching `enabled_toolsets`. Preflight will refuse to run, and calls to the tool return a structured failure (`error: "MoA disabled via moa.enabled=false"`). Invalid config under a disabled block still raises at load time — errors are not deferred.

### `min_successful_references`

Defaults to `min(2, len(reference_models))` so a typical 3–4 model roster tolerates one reference failure. Set it explicitly to override. Values outside `[1, len(reference_models)]` fail validation.

## Interaction with `/model` and `/reasoning_effort`

MoA keeps its own roster — **`/model` does not cascade into MoA**. Switching the main agent's model has no effect on the reference or aggregator models used by the tool.

Similarly, **`/reasoning_effort` only affects the main agent loop**. MoA entries each carry their own `reasoning` field, and the tool does not read the session-level reasoning effort. If you want MoA to track a change, update both `/reasoning_effort` *and* the relevant `reasoning:` fields under `moa:`.

## Provider cost warning

Routing multiple reference models through `openai-codex` multiplies consumption against the Codex plan's daily cap. The tool logs a one-time warning per Codex-routed config shape; take it seriously if you're on a metered plan.

## Credentials

`check_moa_requirements` is config-aware: it asks Hermes' provider resolver to create a client for every `provider` in your roster (including the aggregator). An all-Codex roster with Codex credentials will pass even when `OPENROUTER_API_KEY` is unset. A mixed roster needs credentials for every provider it touches.

## Debug mode

Set `MOA_TOOLS_DEBUG=1` in your environment to dump per-call metadata (provider, resolved model, reasoning kwargs, reference successes/failures, final response length) to the Hermes debug directory.
