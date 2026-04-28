---
sidebar_position: 15
title: "Azure AI Foundry"
description: "Use Hermes Agent with Azure AI Foundry — OpenAI-style and Anthropic-style endpoints, auto-detection of transport and deployed models"
---

# Azure AI Foundry

Hermes Agent supports Azure AI Foundry (and Azure OpenAI) as a first-class provider. A single Azure resource can host models with two different wire formats:

- **OpenAI-style** — `POST /v1/chat/completions` on endpoints like `https://<resource>.openai.azure.com/openai/v1`. Used for GPT-4.x, GPT-5.x, Llama, Mistral, and most open-weight models.
- **Anthropic-style** — `POST /v1/messages` on endpoints like `https://<resource>.services.ai.azure.com/anthropic`. Used when Azure Foundry serves Claude models via the Anthropic Messages API format.

The setup wizard probes your endpoint and auto-detects which transport it uses, which deployments are available, and each model's context length.

## Prerequisites

- An Azure AI Foundry or Azure OpenAI resource with at least one deployment
- An API key for that resource (available in the Azure Portal under "Keys and Endpoint")
- The deployment's endpoint URL

## Quick Start

```bash
hermes model
# → Select "Azure Foundry"
# → Enter your endpoint URL
# → Enter your API key
# Hermes probes the endpoint and auto-detects transport + models
# → Pick a model from the list (or type a deployment name manually)
```

The wizard will:

1. **Sniff the URL path** — URLs ending in `/anthropic` are recognised as Azure Foundry Claude routes.
2. **Probe `GET <base>/models`** — if the endpoint returns an OpenAI-shaped model list, Hermes switches to `chat_completions` and prefills a picker with the returned deployment IDs.
3. **Probe Anthropic Messages shape** — fallback for endpoints that do not expose `/models` but do accept the Anthropic Messages format.
4. **Fall back to manual entry** — private/gated endpoints that reject every probe still work; you pick the API mode and type a deployment name by hand.

Context length for the chosen model is resolved via Hermes' standard metadata chain (`models.dev`, provider metadata, and hardcoded family fallbacks) and stored in `config.yaml` so the model can size its own context window correctly.

## Configuration (written to `config.yaml`)

After running the wizard you'll see something like this:

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.openai.azure.com/openai/v1
  api_mode: chat_completions         # or "anthropic_messages"
  default: gpt-5.4-mini              # your deployment / model name
  context_length: 400000             # auto-detected
```

And in `~/.hermes/.env`:

```
AZURE_FOUNDRY_API_KEY=<your-azure-key>
```

## OpenAI-style endpoints (GPT, Llama, etc.)

Azure OpenAI's v1 GA endpoint accepts the standard `openai` Python client with minimal changes:

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.openai.azure.com/openai/v1
  api_mode: chat_completions
  default: gpt-5.4
```

Important behaviour:

- **GPT-5.x, codex, and o-series auto-route to the Responses API.** Azure Foundry deploys GPT-5 / codex / o1 / o3 / o4 models as Responses-API-only — calling `/chat/completions` against them returns `400 "The requested operation is unsupported."`. Hermes detects these model families by name and upgrades `api_mode` to `codex_responses` transparently, even when `config.yaml` still reads `api_mode: chat_completions`. GPT-4, GPT-4o, Llama, Mistral, and other deployments stay on `/chat/completions`.
- **`max_completion_tokens` is used automatically.** Azure OpenAI (like direct OpenAI) requires `max_completion_tokens` for gpt-4o, o-series, and gpt-5.x models. Hermes sends the right parameter based on the endpoint.
- **Pre-v1 endpoints that require `api-version`.** If you have a legacy base URL like `https://<resource>.openai.azure.com/openai?api-version=2025-04-01-preview`, Hermes extracts the query string and forwards it via `default_query` on every request (the OpenAI SDK otherwise drops it when joining paths).

## Anthropic-style endpoints (Claude via Azure Foundry)

For Claude deployments, use the Anthropic-style route:

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.services.ai.azure.com/anthropic
  api_mode: anthropic_messages
  default: claude-sonnet-4-6
```

Important behaviour:

- **`/v1` is stripped from the base URL.** The Anthropic SDK appends `/v1/messages` to every request URL — Hermes removes any trailing `/v1` before handing the URL to the SDK to avoid double-`/v1` paths.
- **`api-version` is sent via `default_query`, not appended to the URL.** Azure Anthropic requires an `api-version` query string. Baking it into the base URL produces malformed paths like `/anthropic?api-version=.../v1/messages` and returns 404. Hermes passes `api-version=2025-04-15` via the Anthropic SDK's `default_query` instead.
- **OAuth token refresh is disabled.** Azure deployments use static API keys. The `~/.claude/.credentials.json` OAuth token refresh loop that applies to Anthropic Console is explicitly skipped for Azure endpoints to prevent the Claude Code OAuth token from overwriting your Azure key mid-session.

## Alternative: `provider: anthropic` + Azure base URL

If you already have `provider: anthropic` configured and just want to point it at Azure AI Foundry for Claude, you can skip the `azure-foundry` provider entirely:

```yaml
model:
  provider: anthropic
  base_url: https://my-resource.services.ai.azure.com/anthropic
  api_key_env: AZURE_ANTHROPIC_KEY
  default: claude-sonnet-4-6
```

With `AZURE_ANTHROPIC_KEY` set in `~/.hermes/.env`. Hermes detects `azure.com` in the base URL and short-circuits around the Claude Code OAuth token chain so the Azure key is used directly with `x-api-key` auth.

## Model discovery

Azure does **not** expose a pure-API-key endpoint to list your *deployed* model deployments. Deployment enumeration requires Azure Resource Manager authentication (`az cognitiveservices account deployment list`) with an Azure AD principal, not the inference API key.

What Hermes can do:

- Azure OpenAI v1 endpoints (`<resource>.openai.azure.com/openai/v1`) expose `GET /models` with the resource's **available** model catalog. Hermes uses this list to prefill the model picker.
- Azure Foundry `/anthropic` routes: detected via URL path, model name entered manually.
- Private / firewalled endpoints: manual entry with a friendly "couldn't probe" message.

You can always type a deployment name directly — Hermes does not validate against the returned list.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `AZURE_FOUNDRY_API_KEY` | Primary API key for Azure AI Foundry / Azure OpenAI |
| `AZURE_FOUNDRY_BASE_URL` | Endpoint URL (set via `hermes model`; env var is used as a fallback) |
| `AZURE_ANTHROPIC_KEY` | Used by `provider: anthropic` + Azure base URL (alternative to `ANTHROPIC_API_KEY`) |

## Troubleshooting

**401 Unauthorized on gpt-5.x deployments.**
Azure serves gpt-5.x on `/chat/completions`, not `/responses`. Hermes handles this automatically when the URL contains `openai.azure.com`, but if you see a 401 with an `Invalid API key` body, check that `api_mode` in your `config.yaml` is `chat_completions`.

**404 on `/v1/messages?api-version=.../v1/messages`.**
This is the malformed-URL bug from pre-fix Azure Anthropic setups. Upgrade Hermes — the `api-version` parameter is now passed via `default_query` rather than baked into the base URL, so the SDK can't corrupt it during URL joining.

**Wizard says "Auto-detection incomplete."**
The endpoint rejected both the `/models` probe and the Anthropic Messages probe. This is normal for private endpoints behind a firewall or with an IP allow-list. Fall back to manual API mode selection and type your deployment name — everything still works, Hermes just can't prefill the picker.

**Wrong transport picked.**
Run `hermes model` again and the wizard will re-probe. If the probe still picks the wrong mode, you can edit `config.yaml` directly:

```yaml
model:
  provider: azure-foundry
  api_mode: anthropic_messages   # or chat_completions
```

## Related

- [Environment variables](/docs/reference/environment-variables)
- [Configuration](/docs/user-guide/configuration)
- [AWS Bedrock](/docs/guides/aws-bedrock) — the other major cloud provider integration
