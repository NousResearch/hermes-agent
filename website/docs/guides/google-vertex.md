---
sidebar_position: 15
title: "Google Vertex AI"
description: "Use Hermes Agent with Gemini on Google Cloud Vertex AI — OAuth2 service account or ADC, GCP billing and quotas, no static API key"
---

# Google Vertex AI

Hermes Agent supports **Gemini models on Google Cloud Vertex AI** through Vertex's OpenAI-compatible endpoint. Unlike the [Google AI Studio provider](/guides/google-gemini) (which uses a static API key against `generativelanguage.googleapis.com`), Vertex gives you **enterprise-grade rate limits and GCP billing/credits**, and is the right choice when you want Gemini usage to draw on your Google Cloud account rather than an AI Studio key.

:::info Vertex authenticates with OAuth2, not an API key
Vertex has **no static API key** for the standard endpoint. Every request needs a short-lived **OAuth2 access token** (≈1 hour TTL) minted from either a service-account JSON or Application Default Credentials (ADC). Hermes mints and **auto-refreshes** these tokens for you — you never paste a token by hand. This is why pasting a temporary token into a custom provider's `api_key` field does not work: it expires mid-session.
:::

## Prerequisites

- **A Google Cloud project** with the **Vertex AI API enabled** and billing active.
- **Credentials**, one of:
  - a **service-account JSON** key file with the `roles/aiplatform.user` role, or
  - **Application Default Credentials** via `gcloud auth application-default login` (or the metadata server when running on a GCP VM).
- **`google-auth`** — installed automatically the first time you select Vertex (lazy install), or explicitly with `pip install 'hermes-agent[vertex]'`.

## Quick Start

```bash
# Option A — service account JSON (recommended for servers / gateways)
echo "VERTEX_CREDENTIALS_PATH=/path/to/service-account.json" >> ~/.hermes/.env

# Option B — Application Default Credentials (good for local dev)
gcloud auth application-default login

# Select Vertex as your provider
hermes model
# → Choose "More providers..." → "Google Vertex AI"
# → Enter your GCP project ID (or leave blank to use the one in your credentials)
# → Choose a region (default: global)
# → Select a Gemini model

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
  default: google/gemini-3-flash-preview
  provider: vertex

vertex:
  project_id: my-gcp-project   # blank → use the project embedded in the credentials
  region: global               # "global" is required for the Gemini 3.x previews
```

:::tip Environment variables win over config.yaml
`VERTEX_PROJECT_ID` and `VERTEX_REGION` override the `vertex.project_id` / `vertex.region` values in `config.yaml`. Use them for per-shell overrides; keep the durable settings in `config.yaml`.
:::

### How authentication works

1. Hermes resolves credentials in this order: `VERTEX_CREDENTIALS_PATH` → `GOOGLE_APPLICATION_CREDENTIALS` → ADC.
2. It mints an OAuth2 access token (`cloud-platform` scope) and caches it, refreshing when the token is within 5 minutes of expiry.
3. The token is handed to a standard OpenAI client pointed at the Vertex endpoint:
   ```text
   https://aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{region}/endpoints/openapi
   ```
   Regional locations use a `{region}-aiplatform.googleapis.com` host instead.
4. If a session runs longer than the token lifetime and a request returns `401`, Hermes re-mints the token and retries automatically. On a long-running gateway, if ADC's refresh token has itself expired, Hermes falls back to the service-account JSON when one is configured.

## Available Models

Vertex requires the `google/` vendor prefix on model IDs. The `hermes model` picker offers:

| Model | ID |
|-------|----|
| Gemini 3.1 Pro Preview | `google/gemini-3.1-pro-preview` |
| Gemini 3 Pro Preview | `google/gemini-3-pro-preview` |
| Gemini 3 Flash Preview | `google/gemini-3-flash-preview` |
| Gemini 3.1 Flash Lite Preview | `google/gemini-3.1-flash-lite-preview` |
| Gemini 2.5 Pro | `google/gemini-2.5-pro` |
| Gemini 2.5 Flash | `google/gemini-2.5-flash` |

:::note `global` region for Gemini 3.x
The Gemini 3.x preview models are served through the `global` endpoint. Regional endpoints (`us-central1`, etc.) may 404 them. Leave `region: global` unless you have a specific reason to pin a region.
:::

## Switching Models Mid-Session

```text
/model google/gemini-3-pro-preview
/model google/gemini-3-flash-preview
```

`/model` switches among already-configured providers and models; it does not collect new credentials. Configure Vertex with `hermes model` first.

## Reasoning / Thinking

Vertex exposes Gemini's thinking budget through the OpenAI-compatible surface. Hermes maps its reasoning-effort setting onto `extra_body.google.thinking_config` automatically, so `reasoning_effort` works the same way it does on other Gemini surfaces.

## Diagnostics

```bash
hermes doctor
```

The doctor reports whether Vertex credentials can be resolved (service-account path or ADC) and whether the provider is configured.

## Troubleshooting

### "Vertex AI credentials could not be resolved"

Hermes found neither a service-account JSON nor working ADC. Either set `VERTEX_CREDENTIALS_PATH` in `~/.hermes/.env`, or run `gcloud auth application-default login`. If your project isn't embedded in the credentials, set `vertex.project_id` in `config.yaml`.

### `google-auth` not installed

Install the extra: `pip install 'hermes-agent[vertex]'`. Hermes also lazy-installs it the first time you select the Vertex provider.

### 404 on Gemini 3.x models

You are probably on a regional endpoint. Set `region: global` in the `vertex:` section of `config.yaml` (or unset `VERTEX_REGION`).

### 403 / permission denied

The service account (or your ADC identity) needs the `roles/aiplatform.user` role on the project, and the Vertex AI API must be enabled for that project.

## Related

- [Google Gemini (AI Studio)](/guides/google-gemini) — static-API-key Gemini without GCP
- [AWS Bedrock](/guides/aws-bedrock) — another native cloud-provider integration
- [AI Providers](/integrations/providers)
- [Configuration](/user-guide/configuration)
