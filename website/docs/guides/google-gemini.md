---
sidebar_position: 16
title: "Google Gemini"
description: "Use Hermes Agent with Google Gemini — native AI Studio API, API-key setup, tool calling, streaming, and quota guidance"
---

# Google Gemini

Hermes Agent supports Google Gemini as a native provider using the **Google AI Studio / Gemini API** — not the OpenAI-compatible endpoint. This lets Hermes translate its internal OpenAI-shaped message and tool loop into Gemini's native `generateContent` API while preserving tool calling, streaming, multimodal inputs, and Gemini-specific response metadata.

## Model catalog and thinking controls

The model picker reads Google's native model list, including pagination, and
uses Hermes' cached [models.dev](https://models.dev) metadata to keep text-output
models with function calling. A successful discovery replaces the curated list;
on failure Hermes retains its existing fallback behavior. Models that require
the dedicated Computer Use tool are excluded from this general chat picker.
Malformed individual entries are skipped; a failed page still invalidates the
discovery result, so a truncated catalog never replaces the fallback.
Being listed does not guarantee that your project has access to a model.

Desktop shows the thinking controls described for each model:

- **Effort:** only the model's declared levels.
- **Thinking off:** only when disabling thinking is declared.
- **Thinking budget:** a token count within the model's bounds, with **Dynamic**
  for Google's `thinkingBudget: -1` mode.

The numeric input accepts positive thinking budgets. When the source range
starts at zero, the input starts at one: zero means **Thinking off**, available
only when the model declares that capability. Use that control (or `none` in
configuration) instead of `budget:0`. Dynamic remains a separate choice;
positive model-specific minimum budgets are preserved.

When no explicit level or budget is set, the picker leaves the options
unselected rather than displaying a synthetic "Provider default" effort.
Omitting the override differs from Dynamic, particularly for models whose
default is to leave thinking off.

Missing or unrecognized metadata leaves control to Google and is shown as
unverified. A model's name containing “Flash” does not enable a Fast switch.
The picker and the AI Studio request adapter use the same cached description;
catalog refreshes do not make generation requests.

For configuration, `agent.reasoning_effort` additionally accepts `auto` and
`budget:N` (including `budget:-1`). Explicit unsupported session choices are
rejected before saving. Inherited settings from another model are normalized
to a supported level or provider default. These controls apply to AI Studio;
Vertex retains its separate policy. See Google's
[generateContent thinking guide](https://ai.google.dev/gemini-api/docs/generate-content/thinking).

## Prerequisites

- **Google AI Studio API key** — create one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- **Billing-enabled Google Cloud project** — recommended for agent use. Gemini's free tier is too small for long-running agent sessions because Hermes may make several model calls per user turn.
- **Hermes installed** — no extra Python package is required for the native Gemini provider.

:::tip API key path
Set `GOOGLE_API_KEY` or `GEMINI_API_KEY`. Hermes checks both names for the `gemini` provider.
:::

## Quick Start

```bash
# Add your Gemini API key
echo "GOOGLE_API_KEY=..." >> ~/.hermes/.env

# Select Gemini as your provider
hermes model
# → Choose "More providers..." → "Google AI Studio"
# → Hermes checks your key tier and shows Gemini models
# → Select a model

# Start chatting
hermes chat
```

If you prefer direct config editing, use the native Gemini API base URL:

```yaml
model:
  default: gemini-3.7-flash
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

## Configuration

After running `hermes model`, your `~/.hermes/config.yaml` will contain:

```yaml
model:
  default: gemini-3.7-flash
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

And in `~/.hermes/.env`:

```bash
GOOGLE_API_KEY=...
```

### Native Gemini API

The recommended endpoint is:

```text
https://generativelanguage.googleapis.com/v1beta
```

Hermes detects this endpoint and creates its native Gemini adapter. Internally, Hermes still keeps the agent loop in OpenAI-shaped messages, then translates each request to Gemini's native schema:

- `messages[]` → Gemini `contents[]`
- system prompts → Gemini `systemInstruction`
- tool schemas → Gemini `functionDeclarations`
- tool results → Gemini `functionResponse` parts
- streaming responses → OpenAI-shaped stream chunks for the Hermes loop

Tool parameter type arrays such as `"type": ["number", "null"]` are translated
into Gemini's scalar type plus `nullable` form. Multi-type unions keep every
alternative through `anyOf`, including nested properties and array items. This
happens automatically; no MCP server or provider configuration change is needed.

:::note Gemini 3 thought signatures
For Gemini 3 tool use, Hermes preserves the `thoughtSignature` values attached to function-call parts and replays them on the next tool turn. That covers the validation-critical path for multi-step agent workflows.

Gemini 3 may also attach thought signatures to other response parts. Hermes' native adapter is optimized for agent tool loops today, so it does not yet replay every non-tool-call signature with full part-level fidelity.
:::

### Prefer the Native Endpoint

Google also exposes an OpenAI-compatible endpoint:

```text
https://generativelanguage.googleapis.com/v1beta/openai/
```

For Hermes agent sessions, prefer the native Gemini endpoint above. Hermes includes a native Gemini adapter so it can map multi-turn tool use, tool-call results, streaming, multimodal inputs, and Gemini response metadata directly onto Gemini's `generateContent` API. The OpenAI-compatible endpoint is still useful when you specifically need OpenAI API compatibility.

If you previously set `GEMINI_BASE_URL` to the `/openai` URL, remove it or change it:

```bash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

Host-root base URLs on the Google host are normalized automatically: if the URL
doesn't end with an API version segment (`v1beta`, `v1alpha`, `v1`, ...), Hermes
appends `/v1beta` for you, so `GEMINI_BASE_URL=https://generativelanguage.googleapis.com`
works the same as spelling out the `/v1beta` suffix. The same normalization
applies to the Gemini TTS base URL (`tts.gemini.base_url`). Chat requests only
take the native Gemini path when the base URL points at
`generativelanguage.googleapis.com` or the Vertex AI express host below; a proxy on
another host is treated as an OpenAI-compatible endpoint, so configure it with its
`/openai`-style URL.

### Vertex AI Express Mode Keys

Google now issues `AQ.…`-prefixed keys for **both** Google AI Studio and Vertex AI
express mode (the legacy `AIza…` Studio format is being phased out), so a key's
prefix no longer identifies its surface. Hermes never reroutes by key shape: the
configured base URL decides the surface. Set `GEMINI_API_KEY` and leave
`GEMINI_BASE_URL` unset for the default AI Studio host; set `GEMINI_BASE_URL` to
`https://aiplatform.googleapis.com` (with or without `/v1beta1`) for a Vertex AI
express-mode key, and Hermes completes it to the `publishers/google` form. Each
surface only accepts its own keys — a `403 PERMISSION_DENIED` usually means the
key/host pairing is crossed, and Hermes appends guidance naming the other surface.
A base URL on any other host (a proxy) is never rewritten. Express keys are
separate from the OAuth-based
[Vertex AI provider](./google-vertex.md), which needs no API key.

:::warning Upgrade note for existing express-key users
Earlier Hermes releases detected the `AQ.` prefix and rerouted such keys to
`aiplatform.googleapis.com` automatically, so the documented setup was "set
`GEMINI_API_KEY` to the express key and leave `GEMINI_BASE_URL` unset". That
automatic reroute is gone: with `GEMINI_BASE_URL` unset, every request — chat,
`hermes doctor`, TTS — now goes to the AI Studio host and a Vertex express key
gets `403 PERMISSION_DENIED` there. Add `GEMINI_BASE_URL=https://aiplatform.googleapis.com`
to `~/.hermes/.env` (or set `base_url` on the provider) once and restart.
:::

## Available Models

The `hermes model` picker shows Gemini models maintained in Hermes' provider registry. Common choices include:

| Model | ID | Notes |
|-------|----|-------|
| Gemini 3.8 Flash | `gemini-3.8-flash` | Most capable Flash model for long-horizon agentic and coding work |
| Gemini 3.7 Flash | `gemini-3.7-flash` | Recommended default balance of speed, capability, and multimodal understanding |
| Gemini 3.1 Pro Preview | `gemini-3.1-pro-preview` | Most capable reasoning, math, and coding model |
| Gemini 3.5 Flash Lite | `gemini-3.5-flash-lite` | Fastest and lowest-cost option for lightweight tasks |
| Gemini 2.5 Flash | `gemini-2.5-flash` | Previous generation fast model with thinking capabilities |
| Gemini 2.5 Pro | `gemini-2.5-pro` | Previous generation complex reasoning model |

Model availability changes over time. If a model disappears or is not enabled for your key, run `hermes model` again and pick one from the current list.

:::info Model IDs
Use Gemini's native model IDs such as `gemini-3.7-flash`, not OpenRouter-style IDs like `google/gemini-3.7-flash`, when `provider: gemini`.
:::

### Latest Aliases

Google publishes moving aliases for the Pro and Flash Gemini families. `gemini-pro-latest` and `gemini-flash-latest` are useful when you want Google to advance the model automatically without changing your Hermes config. Note that your usage charges may be affected if newer models introduce different rates.

| Alias | Currently tracks | Notes |
|-------|------------------|-------|
| `gemini-pro-latest` | Latest Gemini Pro model | Best when you want Google's current Pro default |
| `gemini-flash-latest` | Latest Gemini Flash model | Best when you want Google's current Flash default |

```yaml
model:
  default: gemini-pro-latest
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

If you need strict reproducibility, prefer explicit model IDs such as `gemini-3.1-pro-preview` or `gemini-3.7-flash`.

### Gemma via the Gemini API

Google also exposes Gemma models through the Gemini API. Hermes recognizes these as Google models, but hides very low-throughput Gemma entries from the default model picker so new users do not accidentally select an evaluation-tier model for a long-running agent session.

Useful evaluation IDs include:

| Model | ID | Notes |
|-------|----|-------|
| Gemma 4 31B IT | `gemma-4-31b-it` | Larger Gemma model; useful for compatibility and quality evaluation |
| Gemma 4 26B A4B IT | `gemma-4-26b-a4b-it` | Smaller active-parameter variant when available |

These models are best treated as evaluation options on Gemini API keys. Google's Gemma API pricing is free-tier-only and the usage caps are low compared with production Gemini models, so sustained Hermes agent use should normally move to a paid Gemini model, a self-hosted deployment, or another provider with appropriate quota.

To use a Gemma model that is hidden from the picker, set it directly:

```yaml
model:
  default: gemma-4-31b-it
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

## Switching Models Mid-Session

Use the `/model` command during a conversation:

```text
/model gemini-3.7-flash
/model gemini-flash-latest
/model gemini-3.1-pro-preview
/model gemini-pro-latest
/model gemma-4-31b-it
/model gemini-3.1-flash-lite-preview
```

If you have not configured Gemini yet, exit the session and run `hermes model` first. `/model` switches among already-configured providers and models; it does not collect new API keys.

## Diagnostics

```bash
hermes doctor
```

The doctor checks:

- Whether `GOOGLE_API_KEY` or `GEMINI_API_KEY` is available
- Whether configured provider credentials can be resolved

## Gateway (Messaging Platforms)

Gemini works with all Hermes gateway platforms (Telegram, Discord, Slack, WhatsApp, LINE, Feishu, etc.). Configure Gemini as your provider, then start the gateway normally:

```bash
hermes gateway setup
hermes gateway start
```

The gateway reads `config.yaml` and uses the same Gemini provider configuration.

## Troubleshooting

### "Gemini native client requires an API key"

Hermes could not find a usable API key. Add one of these to `~/.hermes/.env`:

```bash
GOOGLE_API_KEY=...
# or
GEMINI_API_KEY=...
```

Then run `hermes model` again.

### "This Google API key is on the free tier"

Hermes probes Gemini API keys during setup. Free-tier quotas can be exhausted after a handful of agent turns because tool use, retries, compression, and auxiliary tasks may require multiple model calls.

Enable billing on the Google Cloud project attached to your key, regenerate the key if needed, then run:

```bash
hermes model
```

### "404 model not found"

The selected model is not available for your account, region, or key. Run `hermes model` again and pick another Gemini model from the current list.

### Gemma model is not shown in `hermes model`

Hermes may hide low-throughput Gemma models from the picker by default. If you intentionally want to evaluate one, set the model ID directly in `~/.hermes/config.yaml`.

### "429 quota exceeded" on Gemma

Gemma models exposed through the Gemini API are useful for evaluation, but their Gemini API free-tier caps are low. Use them for compatibility testing, then switch to a paid Gemini model or another provider for sustained agent sessions.

### OpenAI-compatible endpoint is configured

Check `~/.hermes/.env` for:

```bash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

Change it to the native endpoint or remove the override:

```bash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

### Tool calling fails with schema errors

Upgrade Hermes and rerun `hermes model`. The native Gemini adapter sanitizes tool schemas for Gemini's stricter function-declaration format; older builds or custom endpoints may not.

## Related

- [AI Providers](../integrations/providers.md)
- [Configuration](../user-guide/configuration.md)
- [Fallback Providers](../user-guide/features/fallback-providers.md)
- [AWS Bedrock](./aws-bedrock.md) — native cloud-provider integration using AWS credentials
