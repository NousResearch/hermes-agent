---
sidebar_position: 15
title: "Google Vertex AI"
description: "Use Hermes Agent with Google Cloud Vertex AI — Gemini via the OpenAI-compatible endpoint and Claude via the AnthropicVertex SDK, both with OAuth2 / ADC authentication"
---

# Google Vertex AI

Hermes Agent supports **Google Cloud Vertex AI** as a native dual-model-family provider:

- **Gemini** models — via Vertex's OpenAI-compatible endpoint with OAuth2 access tokens.
- **Claude** models — via Anthropic's **AnthropicVertex SDK** for full feature parity (prompt caching, thinking budgets).

Both paths authenticate through Google Cloud's credential chain (service-account JSON or ADC) — there is **no static API key**. Hermes auto-detects which SDK path to use based on the model name.

:::info OAuth2, not an API key
Vertex has **no static API key** for the standard endpoint. Every request needs a short-lived **OAuth2 access token** (≈1 hour TTL) minted from either a service-account JSON or Application Default Credentials (ADC). Hermes mints and **auto-refreshes** these tokens for you — you never paste a token by hand.
:::

## Prerequisites

- **A Google Cloud project** with the **Vertex AI API enabled** and billing active.
- **Credentials**, one of:
  - a **service-account JSON** key file with the `roles/aiplatform.user` role, or
  - **Application Default Credentials** via `gcloud auth application-default login` (or the metadata server when running on a GCP VM).
- **Claude access** (if using Claude models) — request access to Anthropic models in the [Vertex AI Model Garden](https://console.cloud.google.com/vertex-ai/model-garden).
- **Dependencies** — installed automatically the first time you select Vertex (lazy install), or explicitly:
  - Gemini: `pip install 'hermes-agent[vertex]'` (installs `google-auth`)
  - Claude: `pip install 'anthropic[vertex]>=0.87.0'`

:::tip GCE / GKE / Cloud Run
On Google Cloud compute, attach a service account with the `Vertex AI User` role and you're done. No API keys, no `.env` configuration — Hermes detects the credentials automatically via ADC.
:::

## Quick Start

```bash
# Option A — service account JSON (recommended for servers / gateways)
echo "VERTEX_CREDENTIALS_PATH=/path/to/service-account.json" >> ~/.hermes/.env

# Option B — Application Default Credentials (good for local dev)
gcloud auth application-default login

# Select Vertex as your provider
hermes model
# → Choose "Google Vertex AI"
# → Enter your GCP project ID (or leave blank to use the one in your credentials)
# → Choose a region (default: global)
# → Select a model (Gemini or Claude)

# Start chatting
hermes chat
```

## Configuration

Vertex splits its settings by sensitivity:

- The **credential path** is a pointer to a secret and lives in `~/.hermes/.env`.
- **Project ID and region** are non-secret routing settings and live in `~/.hermes/config.yaml`.

`~/.hermes/.env`:

```bash
# One of these (checked in this order); omit both to use ADC:
VERTEX_CREDENTIALS_PATH=/path/to/service-account.json
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

`~/.hermes/config.yaml`:

```yaml
model:
  default: google/gemini-3-flash-preview   # or claude-sonnet-4-6
  provider: vertex

vertex:
  project_id: my-gcp-project   # blank → use the project embedded in the credentials
  region: global               # "global" is required for the Gemini 3.x previews
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `VERTEX_CREDENTIALS_PATH` | No | Path to service account JSON (highest priority) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Fallback | Standard GCP credentials path |
| `VERTEX_PROJECT_ID` | No | Override project ID (Gemini path) |
| `ANTHROPIC_VERTEX_PROJECT_ID` | No | Override project ID (Claude path, highest priority) |
| `GOOGLE_CLOUD_PROJECT` | Fallback | Standard GCP project variable |
| `GCLOUD_PROJECT` | Fallback | Legacy GCP project variable |
| `VERTEX_REGION` | No | Override region (Gemini path, default: `global`) |
| `CLOUD_ML_REGION` | No | Override region (Claude path, default: `global`) |

:::tip Environment variables win over config.yaml
`VERTEX_PROJECT_ID` / `ANTHROPIC_VERTEX_PROJECT_ID` and `VERTEX_REGION` / `CLOUD_ML_REGION` override the `vertex.project_id` / `vertex.region` values in `config.yaml`. Use them for per-shell overrides; keep the durable settings in `config.yaml`.
:::

### How authentication works

**Gemini path:**

1. Hermes resolves credentials: `VERTEX_CREDENTIALS_PATH` → `GOOGLE_APPLICATION_CREDENTIALS` → ADC.
2. It mints an OAuth2 access token (`cloud-platform` scope) and caches it, refreshing when the token is within 5 minutes of expiry.
3. The token is handed to a standard OpenAI client pointed at the Vertex endpoint:
   ```text
   https://aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{region}/endpoints/openapi
   ```
   Regional locations use a `{region}-aiplatform.googleapis.com` host instead.
4. If a session runs longer than the token lifetime and a request returns `401`, Hermes re-mints the token and retries automatically.

**Claude path:**

1. Hermes resolves `project_id` and `region` from env vars → `config.yaml` → defaults.
2. It builds an `AnthropicVertex` client that uses ADC / `GOOGLE_APPLICATION_CREDENTIALS` internally.
3. The SDK handles token lifecycle (minting, caching, refresh) transparently.

## Available Models

### Gemini

Vertex requires the `google/` vendor prefix on Gemini model IDs. The `hermes model` picker offers:

| Model | ID |
|-------|-----|
| Gemini 3.1 Pro Preview | `google/gemini-3.1-pro-preview` |
| Gemini 3 Pro Preview | `google/gemini-3-pro-preview` |
| Gemini 3 Flash Preview | `google/gemini-3-flash-preview` |
| Gemini 3.1 Flash Lite Preview | `google/gemini-3.1-flash-lite-preview` |
| Gemini 2.5 Pro | `google/gemini-2.5-pro` |
| Gemini 2.5 Flash | `google/gemini-2.5-flash` |

:::note `global` region for Gemini 3.x
The Gemini 3.x preview models are served through the `global` endpoint. Regional endpoints (`us-central1`, etc.) may 404 them. Leave `region: global` unless you have a specific reason to pin a region.
:::

### Claude

Claude models use the same model IDs as the Anthropic API:

| Model | ID | Notes |
|-------|-----|-------|
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | Recommended — best balance of speed and capability |
| Claude Opus 4.6 | `claude-opus-4-6` | Most capable |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | Fastest, cheapest |

## Switching Models Mid-Session

```text
/model google/gemini-3-pro-preview
/model claude-sonnet-4-6
```

`/model` switches among already-configured providers and models; it does not collect new credentials. Configure Vertex with `hermes model` first. Switching between Gemini and Claude models works seamlessly — Hermes detects the model family and routes to the correct SDK path automatically.

## Reasoning / Thinking

Both model families support reasoning/thinking:

- **Gemini**: Hermes maps its reasoning-effort setting onto `extra_body.google.thinking_config` automatically.
- **Claude**: The AnthropicVertex SDK supports thinking budgets natively, the same as direct Anthropic API access.

## Gateway (Messaging Platforms)

Vertex works with all Hermes gateway platforms (Telegram, Discord, Slack, etc.). Configure Vertex as your provider, then start the gateway normally:

```bash
hermes gateway setup
hermes gateway start
```

## Compatibility with Claude Code

Vertex AI configuration is compatible with Claude Code's environment variables:

```bash
export CLAUDE_CODE_USE_VERTEX=1
export ANTHROPIC_VERTEX_PROJECT_ID=your-project
export CLOUD_ML_REGION=us-east5
```

## Diagnostics

```bash
hermes doctor
```

The doctor reports whether Vertex credentials can be resolved (service-account path or ADC) and whether the provider is configured.

## Troubleshooting

### "Vertex AI credentials could not be resolved"

Hermes found neither a service-account JSON nor working ADC. Either set `VERTEX_CREDENTIALS_PATH` in `~/.hermes/.env`, or run `gcloud auth application-default login`. If your project isn't embedded in the credentials, set `vertex.project_id` in `config.yaml`.

### "No Vertex AI project configured"

Set one of the project environment variables, or set `vertex.project_id` in `config.yaml`:

```bash
echo "ANTHROPIC_VERTEX_PROJECT_ID=your-project" >> ~/.hermes/.env
```

### `google-auth` not installed

Install the extra: `pip install 'hermes-agent[vertex]'`. Hermes also lazy-installs it the first time you select the Vertex provider.

### `anthropic[vertex]` not installed

Install the Anthropic Vertex extra: `pip install 'anthropic[vertex]>=0.87.0'`. Required for Claude models on Vertex.

### 404 on Gemini 3.x models

You are probably on a regional endpoint. Set `region: global` in the `vertex:` section of `config.yaml` (or unset `VERTEX_REGION`).

### 403 / permission denied

The service account (or your ADC identity) needs the `roles/aiplatform.user` role on the project, and the Vertex AI API must be enabled for that project. For Claude models, also ensure you have access to Anthropic models in the Vertex AI Model Garden.

## Related

- [Google Gemini (AI Studio)](/guides/google-gemini) — static-API-key Gemini without GCP
- [AWS Bedrock](/guides/aws-bedrock) — another native cloud-provider integration
- [AI Providers](/integrations/providers)
- [Configuration](/user-guide/configuration)
