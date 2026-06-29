---
sidebar_position: 3
---

# Configuring Models

Hermes uses two kinds of model slots:

- **Main model** — what the agent thinks with. Every user message, every tool-call loop, every streamed response goes through this model.
- **Auxiliary models** — smaller side-jobs the agent offloads. Context compression, vision (image analysis), web-page summarization, approval scoring, MCP tool routing, session-title generation, and skill search. Each has its own slot and can be overridden independently.

This page covers configuring both from the dashboard. If you prefer config files or the CLI, jump to [Alternative methods](#alternative-methods) at the bottom.

:::tip Fastest path: Nous Portal
[Nous Portal](/user-guide/features/tool-gateway) provides 300+ models under one subscription. On a fresh install, run `hermes setup --portal` to log in and set Nous as your provider in one command. Inspect what's wired up with `hermes portal info`.

- Portal subscribers also get **10% off token-billed providers**.
:::

:::note `model:` schema — empty string vs. mapping
On a brand-new install the bundled default config has `model: ""` (an empty string sentinel meaning "not configured yet"). The first time you run `hermes setup` or `hermes model`, that key is upgraded in-place to a mapping with `provider`, `default`, `base_url`, and `api_mode` sub-keys — the shape shown throughout this page and in [`profiles.md`](./profiles.md) / [`configuration.md`](./configuration.md). If you ever see an empty string in `config.yaml`, run `hermes model` (or click **Change** in the dashboard) and Hermes will write the dict form for you.
:::

## The Models page

Open the dashboard and click **Models** in the sidebar. You get two sections:

1. **Model Settings** — the top panel, where you assign models to slots.
2. **Usage analytics** — ranked cards showing every model that ran a session in the selected period, with token counts, cost, and capability badges.

![Models page overview](/img/docs/dashboard-models/overview.png)

The top card is the **Model Settings** panel. The main row always shows what the agent will spin up for new sessions. Click **Change** to open the picker.

## Setting the main model

Click **Change** on the Main model row:

![Model picker dialog](/img/docs/dashboard-models/picker-dialog.png)

The picker has two columns:

- **Left** — authenticated providers. Only providers you've set up (API key set, OAuth'd, or defined as a custom endpoint) show up here. If a provider is missing, head to **Keys** and add its credential.
- **Right** — the curated model list for the selected provider. These are the agentic models Hermes recommends for that provider, not the raw `/models` dump (which on OpenRouter includes 400+ models including TTS, image generators, and rerankers).

Type in the filter box to narrow by provider name, slug, or model ID.

Pick a model, hit **Switch**, and Hermes writes it to `~/.hermes/config.yaml` under the `model` section. **This applies to new sessions only** — any chat tab you already have open keeps running whatever model it started with. To hot-swap the current chat, use the `/model` slash command inside it.

### Mid-session switches and context warnings

When you switch models **inside an active session** (Herm TUI model picker, `hermes` CLI, or `/model` on Telegram/Discord), Hermes estimates whether your **next message** will run **preflight context compression** against the new model's window. If the session is already near or above that model's compression threshold (see [Context Compression](./configuration.md#context-compression)), the switch reply includes a warning — the same `warning_message` path used for expensive-model notices. The switch still applies immediately; compression runs on the **first user message after the switch**, before the model answers.

## Setting auxiliary models

Click **Show auxiliary** to reveal the 11 task slots:

![Auxiliary panel expanded](/img/docs/dashboard-models/auxiliary-expanded.png)

Every auxiliary task defaults to `auto` — meaning Hermes tries your main model for that job too. If that route is unavailable or hits a capacity-style failure, `auto` follows any task-specific `auxiliary.<task>.fallback_chain`, then the main `fallback_providers` / `fallback_model` chain, then Hermes' built-in auxiliary discovery chain. Override a specific task when you want a cheaper or faster model for a side-job.

### Common override patterns

| Task | When to override |
|---|---|
| **Title Gen** | Almost always. A $0.10/M flash model writes session titles as well as Opus. Default config sets this to `google/gemini-3-flash-preview` on OpenRouter. |
| **Vision** | When your main model lacks vision support. Point it at `google/gemini-2.5-flash` or `gpt-4o-mini`. |
| **Compression** | When you're burning reasoning tokens on Opus/M2.7 just to summarize context. A fast chat model does the job at 1/50th the cost. |
| **Approval** | For `approval_mode: smart` — a fast/cheap model (haiku, flash, gpt-5-mini) decides whether to auto-approve low-risk commands. Expensive models here are waste. |
| **Web Extract** | When you use `web_extract` heavily. Same logic as compression — summarization doesn't need reasoning. |
| **Skills Hub** | `hermes skills search` uses this. Usually fine at `auto`. |
| **MCP** | MCP tool routing. Usually fine at `auto`. |
| **Triage Specifier** | Routes the Kanban triage specifier (`hermes kanban specify`) that expands a rough one-liner into a concrete spec. A cheap, capable model works well. |
| **Kanban Decomposer** | Routes Kanban task decomposition — splits a triage task into a graph of child tasks for specialist profiles. |
| **Profile Describer** | Routes profile-description generation (`hermes profile describe --auto` / the dashboard auto-generate button). Short, cheap call. |
| **Curator** | Routes the curator skill-usage review pass. Can run for minutes on reasoning models, so a cheaper aux model is often worthwhile. |

### Per-task override

Click **Change** on any auxiliary row. Same picker opens, same behavior — pick provider + model, hit Switch. The row updates to show `provider · model` instead of `auto (use main model)`.

### Reset all to auto

If you've over-tuned and want to start over, click **Reset all to auto** at the top of the auxiliary section. Every slot goes back to using your main model.

## The "Use as" shortcut

Every model card on the page has a **Use as** dropdown. This is the fast path — pick a model you see in your analytics, click **Use as**, and assign it to the main slot or any specific auxiliary task in one click:

![Use as dropdown](/img/docs/dashboard-models/use-as-dropdown.png)

The dropdown has:

- **Main model** — same as clicking Change on the main row.
- **All auxiliary tasks** — assigns this model to all 11 aux slots at once. Useful when you just want every side-job on a cheap flash model.
- **Individual task options** — Vision, Web Extract, Compression, etc. The currently-assigned model for each task is marked `current`.

Cards are badged with `main` or `aux · <task>` when they're currently assigned to something — so you can see at a glance which of your historical models are wired in where.

## What gets written to `config.yaml`

When you save via the dashboard, Hermes writes to `~/.hermes/config.yaml`:

**Main model:**
```yaml
model:
  provider: openrouter
  default: anthropic/claude-opus-4.7
  base_url: ''        # cleared on provider switch
  api_mode: chat_completions
```

**Auxiliary override (example — vision on gemini-flash):**
```yaml
auxiliary:
  vision:
    provider: openrouter
    model: google/gemini-2.5-flash
    base_url: ''
    api_key: ''
    timeout: 120
    extra_body: {}
    download_timeout: 30
```

**Auxiliary on auto (default):**
```yaml
auxiliary:
  compression:
    provider: auto
    model: ''
    base_url: ''
    # ... other fields unchanged
```

`provider: auto` with `model: ''` tells Hermes to use the main model for that task, while still honoring fallback policy if the main route cannot serve the auxiliary call.

Optional task-specific fallback chains live under the same auxiliary task:

```yaml
auxiliary:
  title_generation:
    provider: auto
    model: ''
    fallback_chain:
      - provider: openrouter
        model: inclusionai/ring-2.6-1t:free
```

When `fallback_chain` is absent, `auto` uses the top-level `fallback_providers` chain before the built-in auxiliary discovery chain.

## When does it take effect?

- **CLI** (`hermes chat`): next `hermes chat` invocation.
- **Gateway** (Telegram, Discord, Slack, etc.): next *new* session. Existing sessions keep their model. Restart the gateway (`hermes gateway restart`) if you want to force all sessions to pick up the change.
- **Dashboard chat tab** (`/chat`): next new PTY. The currently-open chat keeps its model — use `/model` inside it to hot-swap.

Changes never invalidate prompt caches on running sessions. That's deliberate: swapping the main model inside a session requires a cache reset (the system prompt contains model-specific content), and we reserve that for the explicit `/model` slash command inside chat.

## Troubleshooting

### "No authenticated providers" in the picker

Hermes lists a provider only if it has a working credential. Check **Keys** in the sidebar — you should see one of: an API key, a successful OAuth, or a custom endpoint URL. If the provider you want isn't there, run `hermes setup` to wire it up, or go to **Keys** and add the env var.

### Main model didn't change in my running chat

Expected. The dashboard writes `config.yaml`, which new sessions read. The currently-open chat is a live agent process — it keeps whatever model it was spawned with. Use `/model <name>` inside the chat to hot-swap that specific session.

### Auxiliary override "didn't take effect"

Three things to check:

1. **Did you start a new session?** Existing chats don't re-read config.
2. **Is `provider` set to something other than `auto`?** If the field shows `auto`, the task is still using your main model. Click **Change** and pick a real provider.
3. **Is the provider authenticated?** If you assigned `minimax` to a task but don't have a MiniMax API key, that task falls back to the openrouter default and logs a warning in `agent.log`.

### I picked a model but Hermes switched providers on me

On OpenRouter (or any aggregator), bare model names resolve *within* the aggregator first. So `claude-sonnet-4` on OpenRouter becomes `anthropic/claude-sonnet-4.6`, staying on your OpenRouter auth. But if you typed `claude-sonnet-4` on a native Anthropic auth, it would stay as `claude-sonnet-4-6`. If you see an unexpected provider switch, check that your current provider is what you expect — the picker always shows the current main at the top of the dialog.

## Alternative methods

### CLI slash command

Inside any `hermes chat` session:

```
/model gpt-5.4 --provider openrouter             # session-only
/model gpt-5.4 --provider openrouter --global    # also persists to config.yaml
```

`--global` does the same thing the dashboard's **Change** button does, plus it switches the running session in-place.

### Custom aliases

Define your own short names for models you reach for often, then use `/model <alias>` in the CLI or any messaging platform. There are two equivalent formats — pick whichever fits your workflow.

**Canonical (top-level `model_aliases:`)** — full control over provider + base_url:

```yaml
# ~/.hermes/config.yaml
model_aliases:
  fav:
    model: claude-sonnet-4.6
    provider: anthropic
  grok:
    model: grok-4
    provider: x-ai
```

**Short string form (`model.aliases.<name>: provider/model`)** — convenient from the shell because `hermes config set` only writes scalar values, but it can't carry a custom `base_url`:

```bash
hermes config set model.aliases.fav anthropic/claude-opus-4.6
hermes config set model.aliases.grok x-ai/grok-4
```

Both paths feed the same loader (`hermes_cli/model_switch.py`). Entries declared in `model_aliases:` take precedence over `model.aliases:` entries with the same name.

Then `/model fav` or `/model grok` in chat. User aliases shadow built-in short names (`sonnet`, `kimi`, `opus`, etc.). See [Custom model aliases](/reference/slash-commands#custom-model-aliases) for the full reference.

### `hermes model` subcommand

```bash
hermes model            # Interactive provider + model picker (the canonical way to switch defaults)
```

`hermes model` walks you through picking a provider, authenticating (OAuth flows open a browser; API-key providers prompt for the key), and then choosing a specific model from that provider's curated catalog. The choice is written to `model.provider` and `model.model` in `~/.hermes/config.yaml`.

To list providers/models without launching the picker, use the dashboard or the REST endpoints below. To inspect what the CLI will actually use right now: `hermes config show | grep '^model\.'` and `hermes status`.

### Direct config edit

Edit `~/.hermes/config.yaml` and restart whatever reads it. See the [Configuration reference](./configuration.md) for the full schema.

### REST API

The dashboard uses three endpoints. Useful for scripting:

```bash
# List authenticated providers + curated model lists
curl -H "X-Hermes-Session-Token: $TOKEN" http://localhost:PORT/api/model/options

# Read current main + auxiliary assignments
curl -H "X-Hermes-Session-Token: $TOKEN" http://localhost:PORT/api/model/auxiliary

# Set the main model
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"main","provider":"openrouter","model":"anthropic/claude-opus-4.7"}' \
  http://localhost:PORT/api/model/set

# Override a single auxiliary task
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"vision","provider":"openrouter","model":"google/gemini-2.5-flash"}' \
  http://localhost:PORT/api/model/set

# Assign one model to every auxiliary task
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"","provider":"openrouter","model":"google/gemini-2.5-flash"}' \
  http://localhost:PORT/api/model/set

# Reset all auxiliary tasks to auto
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"__reset__","provider":"","model":""}' \
  http://localhost:PORT/api/model/set
```

The session token is injected into the dashboard HTML at startup and rotates on every server restart. Grab it from the browser devtools (`window.__HERMES_SESSION_TOKEN__`) if you're scripting against a running dashboard.
