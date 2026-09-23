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

MoA presets are selectable on **every Hermes surface**, because MoA is a normal provider in the model system:

- **CLI / gateway / TUI `/model`** — `/model <preset> --provider moa`, or `/model --provider moa` for the default preset. A bare `/model <preset>` also works when the name exactly matches a configured preset.
- **`hermes model`** and the **Dashboard model picker** — a `Mixture of Agents` provider row appears with your preset names as its models.
- **Desktop GUI app** — the model dropdown shows an `MoA presets` section; selecting one (`MoA: <preset>`) switches the active model to that preset. The Desktop settings panel also creates and edits presets.

Configured presets therefore show up wherever you would pick any other model.

## Slash command shortcut

`/moa` is one-shot convenience sugar. It runs a single prompt through the **default** MoA preset, then restores whatever model you were on:

```bash
/moa design and implement a migration plan for this flaky test cluster
```

Hermes temporarily switches to the default MoA preset for that one turn, sends the prompt, then restores your previous model afterward. The whole argument is the prompt — `/moa` no longer interprets it as a preset name.

```bash
/moa
```

Bare `/moa` (no prompt) just prints usage.

To **switch** to a MoA preset for the rest of the session, select it from the model picker — MoA presets appear under a `Mixture of Agents` provider in every model-selection surface (see above). `/moa` is deliberately not a model switch, so a normal prompt can never accidentally change your model.

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
      # Optional: pin sampling temperatures. When omitted (the default),
      # temperature is NOT sent and each model uses its provider default —
      # the same behavior as a single-model Hermes agent.
      # reference_temperature: 0.6
      # aggregator_temperature: 0.4

      enabled: true
```

Default preset:

- reference: `openai-codex:gpt-5.5`
- reference: `openrouter:deepseek/deepseek-v4-pro`
- aggregator / acting model: `openrouter:anthropic/claude-opus-4.8`

### Advisor output

MoA uses provider-owned output limits. Preset and per-slot output-token cap
settings are no longer supported. Provider defaults vary; omission does not
always mean the model maximum. Native protocols that require an output limit
receive an internal value from Hermes.

### Advisor cadence with `fanout`

By default the advisors run **once per user turn** (`fanout: user_turn`) —
they synthesize plan-level advice on the first message of the turn, then the
acting aggregator works through the rest of the tool loop alone. This is the
cheapest cadence: advisor cost does not multiply with the number of tool
calls in a turn. Two alternative cadences trade cost for advice freshness:

- `fanout: per_iteration` — advisors re-run on **every tool iteration**, so
  their advice always tracks the latest tool results — at the cost of
  multiplying advisor latency and spend by the number of tool calls in a
  turn.
- `fanout: every_n:3` — the middle ground: advisors run on the **first**
  iteration of each user turn and then every **3rd** tool iteration (any
  `N >= 2` works). Iterations in between reuse the cached guidance from the
  last advisor run, so the aggregator still gets advice on every step — it is
  just refreshed every N steps instead of every step. The counter resets on
  each new user message, so every turn starts with fresh advice. The mapping
  form `fanout: {mode: every_n, n: 3}` is also accepted and normalized to
  the string form.

```yaml
moa:
  presets:
    fresh:
      reference_models:
        - provider: openrouter
          model: anthropic/claude-opus-4.8
      aggregator:
        provider: openrouter
        model: openai/gpt-5.5
      fanout: per_iteration   # advisors refresh on every tool iteration
```

Unknown or malformed values fall back to `user_turn`.

:::note Default change
Prior to July 2026 the default cadence was `per_iteration`. The default is
now `user_turn` — the cheapest, lowest-impact cadence — until per-mode
benchmarks justify a costlier default. Presets that want per-step advising
back set `fanout: per_iteration` explicitly.
:::

### Privacy filter for advisor outputs

Advisor outputs can echo sensitive data from the conversation — emails,
formatted phone numbers, API keys, JWTs — into the reference blocks shown in
the UI, saved MoA traces, and the aggregator prompt. `moa.privacy_filter`
(off by default) redacts those surfaces:

```yaml
moa:
  privacy_filter: display   # or: full
```

- `display` — redacts **user-visible surfaces only**: the labelled reference
  blocks rendered in the UI and the records written by `save_traces`. The
  aggregator still receives the raw advisor text, so answer quality is
  unaffected.
- `full` — additionally redacts the advisor text injected into the
  aggregator prompt (and the one-shot `/moa` synthesis input).

Credential shapes (API-key prefixes, JWTs, private keys, DB connection
strings) are masked by Hermes' central secret redactor; the MoA filter adds
email and clearly formatted phone-number redaction on top. Patterns are
deliberately conservative for code-review-style advice: bare digit runs, line
numbers, timestamps, git SHAs, and IP addresses are never touched — only
delimited phone formats like `(555) 123-4567` or `555-123-4567` match.

### Per-slot reasoning effort

Reference and aggregator slots may also set `reasoning_effort`. Use this when
you want the same model to contribute at different depths, or when the
aggregator should think harder than the advisory references. Valid values match
Hermes' normal reasoning controls: `none`, `minimal`, `low`, `medium`, `high`,
`xhigh`, `max`, and `ultra`.

```yaml
moa:
  presets:
    deep_review:
      reference_models:
        - provider: openai-codex
          model: gpt-5.6-sol
          reasoning_effort: low
        - provider: openai-codex
          model: gpt-5.6-sol
          reasoning_effort: xhigh
        - provider: xai-oauth
          model: grok-4.5
      aggregator:
        provider: openai-codex
        model: gpt-5.6-sol
        reasoning_effort: high
```

Omit `reasoning_effort` to use the provider/Hermes default for that slot.

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

## Prompt caching

MoA is built so the **main conversation's prompt cache is never broken**. Selecting a MoA preset is a normal model selection: it does not mutate past context, swap toolsets, or rebuild the system prompt mid-conversation. Your conversation history, system prompt, and tool schema stay byte-stable, so the cached prefix every other model relies on is preserved exactly as it would be for a plain model. Switching to or away from a MoA preset costs the same cache invalidation as any other `/model` switch — no more.

Both internal call types cache normally:

- **Reference models** receive a trimmed, deterministic view of the conversation (system prompt and tool transcript stripped — see the loop above). Because that view is a stable function of the stable history, a reference model's prompt prefix repeats across iterations and caches normally. References are short advisory calls with no tools.
- **The aggregator** is the acting model. The reference outputs are appended to the *end* of the latest user turn as private guidance. Because that text sits at the tail — below the entire stable prefix (system prompt + prior history) — it does not invalidate any cached prefix: the aggregator gets a cache hit on everything above the injection, and only the freshly appended tail is new. That is exactly how every normal turn behaves, where each new user message is also uncached tail tokens.

So MoA does not sacrifice prompt caching on either call type. Its only real cost is the extra reference calls per iteration — you pay for multiple model perspectives, not for broken caches. The long-lived conversation prefix shared with the rest of Hermes is fully intact.

## Notes

- MoA is no longer listed under `hermes tools`; there is no `moa` toolset to enable.
- Setting `enabled: false` on a preset disables the reference fan-out for that preset: the aggregator acts alone, exactly as if you selected it as a plain model. This is the per-preset off switch surfaced in the dashboard and desktop settings.
- A preset's aggregator cannot be another MoA preset. Recursive MoA trees are intentionally blocked.
- For ordinary fixed presets, credential failures on one reference model do not abort the turn. Hermes includes the failure in the reference context and continues with whatever models returned. Enforced managed slots are required gates: their denial or failure aborts the cohort instead of silently dropping the reference.
- MoA increases model-call count. A single model iteration can involve multiple reference calls plus the aggregator call.

## Guided model routing for MoA slots (opt-in, per preset)

A reference or aggregator slot in a preset can carry `routing_role` to have that
slot's actual provider/model/reasoning resolved by the same guided-routing selector
and policy store the Kanban and delegation adapters use, instead of a fixed
`provider`/`model` pin. A slot without `routing_role` is entirely unaffected — this
does not change any existing fixed preset.

The [complete fictional preset example](/examples/guided-routing/moa.json) shows
the `moa` configuration object with two references and an aggregator. Even managed
slots must retain nonempty static `provider` and `model` fields: the normalizer
requires them, and shadow uses them. Enforced selection replaces those static
identities; they are never a fallback after denial. The example is tested through
normalization, not activated or qualified for inference. It requires a separately
approved policy with `moareference`/`moaaggregator` rankings and two eligible
reference makers; the builder-only diagnostic policy is not that policy.

Set `routing_mode: shadow` beside `routing_role` to record the route the policy would
recommend while continuing to call the preset's fixed provider/model. Shadow slots are
never required cohort gates, and a missing policy or rejected recommendation is logged
without blocking the legacy MoA call.

- **The cohort is resolved once per MoA run and pinned.** All managed slots in a
  run resolve together and the resulting routes are bound to that run's client; a
  config edit to the named preset mid-run cannot change an already-pinned cohort,
  and restarting a failed cohort is a new attempt, not a resume.
- **Every enforced managed slot is required.** Managed reference selection excludes
  earlier managed reference makers, so two managed references require different
  approved makers. The aggregator may share a reference maker unless its own
  independent-review requirements exclude it. A denied or missing required slot
  fails the attempt — there is no optional managed mode or silent aggregator-only
  fallback when a required reference cannot resolve.
- **The virtual maker `moa` cannot satisfy an independence requirement** — cohort
  diversity is checked against each slot's actual resolved maker, never the
  aggregation mechanism itself.
- **Diversity does not replace contributor provenance.** Each review slot retains
  its original frozen SHA, verifier, completeness and contributor makers. Earlier
  reference makers narrow the later reference's eligibility separately; a missing
  review manifest cannot be filled in by the diversity mechanism.
- **Input estimates must be explicit.** Missing or zero input/reserve estimates
  block before dispatch. Managed references and aggregators check assembled text
  against verified route capacity rather than silently trimming it to fit.
- **Model-proposed scope cannot lower quality.** Missing host-side classification
  authority retains the deep tier, even for `task_class: established-pattern`.
  See [classification authority](./kanban-worker-lanes.md#classification-authority)
  for exact-execution attestations and operator precedence.
- **Failure is a denied slot, not silent unmanaged.** A managed slot whose route is
  denied or mismatched at the call boundary raises before that call is made, rather
  than falling back to whatever the preset's static config would otherwise pick.
- **Streaming health follows consumption.** For a managed acting aggregator, route health
  remains unknown until the returned stream is exhausted. A disconnect or timeout raised
  while consuming the stream is recorded as that route's failure, never as an early
  success merely because stream construction returned.

See [Kanban worker lanes → Guided model routing](./kanban-worker-lanes.md#guided-model-routing-opt-in-per-task)
for the `hermes kanban routing` commands that manage the shared policy this resolves
against. Live activation and remote route qualification are separate operator
steps; local fixture success is not proof of remote model access.
