---
sidebar_position: 8
title: "Model Router"
description: "A classifier reads each prompt and routes it to the matching tier's model — local for casual chat, a strong model for heavy tasks, with fallbacks"
---

# Model Router

The Model Router is a virtual model provider — the inverse of [Mixture of Agents](/user-guide/features/mixture-of-agents). Instead of fanning a prompt out to several models and aggregating, a small **classifier** call reads each incoming prompt and decides which execution **tier** it needs:

- `simple` — casual conversation, quick questions, texting-style replies → typically a **local model** (free, private)
- `complex` — coding, multi-step tasks, heavy tool use, long documents → typically a **strong metered model**

The whole turn then runs on that tier's model. If it fails (server down, rate limit, quota), the router walks the preset's **fallback chain** — local-first by default — so a dead local server degrades gracefully instead of failing the turn.

Use the router when you chat with Hermes on channels like WhatsApp where most messages are trivial, but you occasionally hand it real work: the trivial 90% stays on a free local model, and only the heavy 10% spends your metered provider's usage.

## Select a Router preset as your model

Router presets are selectable on **every Hermes surface**, because the router is a normal provider in the model system:

```bash
/model default --provider router
```

- **CLI / gateway / TUI `/model`** — `/model <preset> --provider router`, or `/model --provider router` for the default preset. A bare `/model <preset>` also works when the name exactly matches a configured preset (MoA preset names win ties; disambiguate with `--provider router`).
- **`hermes model`** and the **Dashboard model picker** — a `Model Router` provider row appears with your preset names as its models.
- **Desktop GUI app** — the model dropdown shows a `Model Router presets` section; selecting one (`Router: <preset>`) switches the active model to that preset. The Desktop Model settings panel and the dashboard's Models page also create and edit presets — classifier, dispatch tiers (simple/complex), and the fallback chain.

In the desktop chat, every routed turn shows a small centered `routed · <tier> · <model>` note in the transcript (and a `rerouted · a → b` note per fallback hop), so you can always see which model served a turn. Messaging channels (WhatsApp/Telegram) don't get these notes.

## Configuration

```yaml title="~/.hermes/config.yaml"
router:
  default_preset: default
  save_traces: false          # true → audit each routing decision to a JSONL
  presets:
    default:
      enabled: true            # false → skip the classifier, always use default_route
      classifier: {provider: openai-codex, model: gpt-5.5}
      classifier_max_tokens: 16
      classifier_context_messages: 4
      default_route: simple    # fail-open target when the classifier errors
      routes:
        simple:  {provider: lmstudio,     model: google/gemma-4-e4b}
        complex: {provider: openai-codex, model: gpt-5.5}
      fallbacks:               # walked in order on acting-model failure
        - {provider: lmstudio,     model: qwen/qwen3-4b-thinking-2507}
        - {provider: openai-codex, model: gpt-5.5}
      channel_hints:           # platform → tier bias fed to the classifier
        whatsapp: simple
```

Key semantics:

- **The classifier is tiny.** Its output contract is one word (`simple` or `complex`), so even a metered classifier costs a few hundred input tokens and ~1 output token per user turn. Keep it on a strong model — routing quality is the whole feature.
- **Fail-open.** If the classifier call errors, times out, or returns garbage, the turn routes to `default_route` (usually the local tier) so chat keeps working when the classifier's provider is down. `enabled: false` skips the classifier entirely and always uses `default_route`.
- **Sticky per user turn.** The classifier runs once per user message; every tool-loop iteration of that turn reuses the decision. A task never swaps models mid-flight. A failed candidate is also remembered for the rest of the turn.
- **Fallbacks walk in order** on any call-time failure (connection, 429, 5xx, auth, quota). When the chain is exhausted the error surfaces to the agent's own [fallback providers](/user-guide/features/fallback-providers), which still backstop the router.
- **Channel hints** bias the classifier per platform — e.g. WhatsApp leans `simple` unless the message is clearly a coding/long task.
- **Recursion guard**: a route/fallback/classifier slot can never point at `router` or `moa`.
- **Small local models are allowed as tiers.** Routed slots bypass the 64K minimum-context gate that applies to a primary model; `hermes router list` prints a soft warning for tiers below 64K (long sessions rely on conversation compression).

## Per-platform default (e.g. WhatsApp)

Make the router the default for one platform, while other platforms keep the global default, with a `"*"` catch-all channel override:

```yaml title="~/.hermes/config.yaml"
platforms:
  whatsapp:
    channel_overrides:
      "*": {provider: router, model: default}
```

Priority per session: `/model` override → exact channel override → `"*"` catch-all → global `model.default`. A user can still `/model` their WhatsApp session onto anything else.

## Terminal CLI

```bash
hermes router list                 # show presets (classifier, tiers, fallbacks)
hermes router configure [name]     # interactively pick classifier/tiers/fallbacks
hermes router delete <name>
hermes router test "hey how's it going"                 # classifier dry-run
hermes router test --platform whatsapp "fix my flask app"
```

`hermes router test` runs the real classifier once and prints the raw output, parsed verdict, resolved route, and fallback chain — the quickest way to sanity-check routing quality.

In a chat session, `/router` shows the active preset and the session's last routing decision.

## Routing visibility

Every surface shows a one-line note when a turn is routed (`⇢ routed simple → lmstudio:google/gemma-4-e4b`) and a warning per fallback hop.

With `save_traces: true`, every routed turn appends a JSONL record to `~/.hermes/router-traces/<session_id>.jsonl`: the classifier's exact input and raw verdict (or the failure that made it fail open), the chosen route, every fallback hop, and the acting model's output. Turn it on for a few days to audit which prompts go where.

## See also

- [Mixture of Agents](/user-guide/features/mixture-of-agents) — the fan-out/aggregate sibling of this feature
- [Fallback providers](/user-guide/features/fallback-providers) — agent-level failover that backstops the router's own chain
