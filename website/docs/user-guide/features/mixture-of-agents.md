---
sidebar_position: 7
title: "Mixture of Agents"
description: "Create named MoA presets that appear as selectable models under the Mixture of Agents provider"
---

# Mixture of Agents

Mixture of Agents is a virtual model provider. Each named MoA preset appears as a selectable model under the `moa` provider.

When you select a MoA preset, the preset's aggregator is the acting model. It is the model that writes the assistant response and emits tool calls. Reference models run first and provide analysis for the aggregator to use.

Use MoA when a hard task benefits from multiple model perspectives but still needs Hermes' normal agent loop: tool calls, follow-up iterations, interrupts, transcript persistence, and the same session context as any other message.

## Select a MoA preset as your model

You can select a preset through the normal model picker surfaces:

```bash
/model default --provider moa
/model review --provider moa
```

The Dashboard, TUI, and Desktop model pickers also show a `Mixture of Agents` provider row. Its models are your configured preset names.

## Slash command shortcut

`/moa` is convenience sugar over model selection:

```bash
/moa
```

Switches the current session to the default MoA preset.

```bash
/moa review
```

If `review` exactly matches a preset name, switches the current session to provider `moa`, model `review`.

```bash
/moa design and implement a migration plan for this flaky test cluster
```

If the text does not exactly match a preset name, Hermes treats it as a one-shot prompt. It temporarily switches to the default MoA preset for that turn, sends the prompt, then restores the previous model afterward.

Preset matching is exact on purpose. Hermes does not fuzzy-match preset names, so normal prompts cannot accidentally become model switches.

## How it works in the agent loop

For each main model call when provider `moa` is selected, Hermes:

1. resolves the selected preset by name;
2. runs the configured reference models without tool schemas (they receive only the conversation's user/assistant text — not the Hermes system prompt or tool-call transcript — so reference calls stay cheap and avoid strict-provider rejections);
3. appends the reference outputs as private context for the aggregator;
4. calls the configured aggregator with the normal Hermes tool schema;
5. treats the aggregator response as the real model response;
6. if the aggregator calls tools, Hermes executes those tools normally;
7. on the next model iteration, the same MoA process runs again over the updated conversation, including tool results.

Because MoA is selected through the normal model system, it composes automatically with `/goal`, gateway sessions, TUI sessions, and Desktop chat.

## Configure presets

You can configure named MoA presets from:

- Dashboard → Models → Model Settings → Mixture of Agents
- Desktop app → Settings → Model → Mixture of Agents
- `hermes moa configure [name]`
- `config.yaml`

The config stores explicit provider/model pairs, so you can mix providers and use multiple models from the same provider:

```yaml
moa:
  default_preset: default
  presets:
    default:
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
        - provider: openrouter
          model: deepseek/deepseek-v4-pro
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
      reference_temperature: 0.6
      aggregator_temperature: 0.4
      max_tokens: 4096
      enabled: true
```

Default preset:

- reference: `openai-codex:gpt-5.5`
- reference: `openrouter:deepseek/deepseek-v4-pro`
- aggregator / acting model: `openrouter:anthropic/claude-opus-4.8`

## Terminal preset management

```bash
hermes moa list
hermes moa configure              # update the default preset
hermes moa configure review       # create or update a named preset
hermes moa delete review
```

## Benchmarks

On HermesBench, a two-model MoA preset — `claude-opus-4.8` aggregating over a `gpt-5.5` reference — outscores either model run on its own:

| Model | HermesBench score |
|---|---|
| **Opus aggregator (opus-4.8 + gpt-5.5 reference) — MoA** | **0.8202** |
| `anthropic/claude-opus-4.8` | 0.7607 |
| `openai/gpt-5.5` | 0.7412 |

The MoA configuration beats its strongest component (opus-4.8) by ~6 points, confirming that aggregating a second perspective lifts quality on hard tasks rather than just averaging the two.

## Notes

- MoA is no longer listed under `hermes tools`; there is no `moa` toolset to enable.
- Setting `enabled: false` on a preset disables the reference fan-out for that preset: the aggregator acts alone, exactly as if you selected it as a plain model. This is the per-preset off switch surfaced in the dashboard and desktop settings.
- A preset's aggregator cannot be another MoA preset. Recursive MoA trees are intentionally blocked.
- Credential failures on one reference model do not abort the turn. Hermes includes the failure in the reference context and continues with whatever models returned.
- MoA increases model-call count. A single model iteration can involve multiple reference calls plus the aggregator call.
