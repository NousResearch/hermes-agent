---
sidebar_position: 15
title: "Google Vertex AI"
description: "Use Hermes Agent with Gemini on Google Cloud Vertex AI — API key (Express Mode, recommended) or OAuth2 service account / ADC"
---

# Google Vertex AI

Hermes Agent supports **Gemini models on Google Cloud Vertex AI** through Vertex's OpenAI-compatible endpoint. Unlike the [Google AI Studio provider](/guides/google-gemini) (which uses a static API key against `generativelanguage.googleapis.com`), Vertex gives you **enterprise-grade rate limits and GCP billing/credits**, and is the right choice when you want Gemini usage to draw on your Google Cloud account rather than an AI Studio key.

## Authentication

Vertex supports two authentication methods, chosen automatically:

### Method 1: API Key (Express Mode) — Recommended

Vertex now supports **API key authentication** through Express Mode. This is the simplest way to connect — no OAuth, no service-account JSON, no ADC setup.

1. **Create an API key** in the [Google Cloud Console](https://console.cloud.google.com/apis/credentials). You can use an existing API key with the Vertex AI API enabled.

2. **Set the environment variables** in `~/.hermes/.env`:
   ```bash
   GOOGLE_VERTEX_API_KEY=AIza...
   GOOGLE_VERTEX_PROJECT=my-gcp-project
   GOOGLE_VERTEX_LOCATION=us-central1   # optional, default: us-central1
   ```

3. **Select Vertex** as your provider:
   ```bash
   hermes model
   # → Choose "More providers..." → "Google Vertex AI"
   # → Verify API key is detected
   # → Enter your GCP project ID
   # → Models will be picked from the curated Vertex list
   #    (no models.list endpoint for API keys)
   # → Select a model
   ```

4. **Start chatting:**
   ```bash
   hermes chat
   ```

:::note API key model discovery
Vertex's public API exposes no `models.list` endpoint for Express Mode API keys (the publisher models endpoint 404s), so Hermes always uses its curated Vertex model list with API key auth. Model availability may vary by region and project; unavailable models fail with a 404 at request time.
:::

### Method 2: OAuth2 / Service Account (Legacy)

Traditional OAuth2 authentication using a service account JSON file or Application Default Credentials (ADC).

#### Prerequisites

- **A Google Cloud project** with the **Vertex AI API enabled** and billing active.
- **Credentials**, one of:
  - a **service-account JSON** key file with the `roles/aiplatform.user` role, or
  - **Application Default Credentials** via `gcloud auth application-default login` (or the metadata server when running on a GCP VM).
- **`google-auth`** — installed automatically the first time you select Vertex (lazy install). Run `hermes setup` to repair a managed install if that fails.

#### Quick Start

```bash
# Option A — service account JSON (recommended for servers / gateways)
echo "VERTEX_CREDENTIALS_PATH=/path/to/service-account.json" >> ~/.hermes/.env

# Option B — Application Default Credentials (good for local dev)
gcloud auth application-default login

# Select Vertex as your provider
hermes model
# → Choose "More providers..." → "Google Vertex AI"
# → Enter your GCP project ID (or leave blank to use the one in your credentials)
# → Choose a region (default: us-central1)
# → Select a Gemini model

# Start chatting
hermes chat
```

## Configuration

Vertex splits its settings by sensitivity:

- **Credential paths and API keys** go in `~/.hermes/.env`.
- **Project ID and region** are non-secret routing settings and live in `~/.hermes/config.yaml`.

`~/.hermes/.env`:

```bash
# Method 1 — API Key (recommended):
GOOGLE_VERTEX_API_KEY=AIza...
GOOGLE_VERTEX_PROJECT=my-gcp-project
GOOGLE_VERTEX_LOCATION=us-central1

# Method 2 — Service account (legacy):
# VERTEX_CREDENTIALS_PATH=/path/to/service-account.json
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

`~/.hermes/config.yaml`:

```yaml
model:
  default: google/gemini-3-flash-preview
  provider: vertex

vertex:
  project_id: my-gcp-project   # blank → use the project from env or credentials
  region: us-central1           # default region
```

:::tip Environment variables win over config.yaml
`GOOGLE_VERTEX_PROJECT`, `GOOGLE_VERTEX_LOCATION`, `VERTEX_PROJECT_ID`, and `VERTEX_REGION` override the `vertex.project_id` / `vertex.region` values in `config.yaml`. Use them for per-shell overrides; keep the durable settings in `config.yaml`.
:::

### How authentication works

For **API Key (Express Mode)**:
1. Hermes reads your `GOOGLE_VERTEX_API_KEY` from the environment.
2. It passes the key in the `x-goog-api-key` header to Vertex's native `:generateContent` API endpoint (`https://aiplatform.googleapis.com/v1/publishers/google`).
3. Requests route through `GeminiNativeClient` which handles native Gemini payload formatting, system prompt injection, and bare model name resolution.
4. No token refresh needed — API keys don't expire.

For **OAuth2 / ADC**:
1. Hermes resolves credentials in this order: `VERTEX_CREDENTIALS_PATH` → `GOOGLE_APPLICATION_CREDENTIALS` → ADC.
2. It mints an OAuth2 access token (`cloud-platform` scope) and caches it, refreshing when the token is within 5 minutes of expiry.
3. The token is handed to a standard OpenAI client pointed at the Vertex endpoint.
4. If a session runs longer than the token lifetime and a request returns `401`, Hermes re-mints the token and retries automatically.

## Available Models

Vertex's public API exposes no `models.list` endpoint for Express Mode API keys (the publisher models endpoint 404s), so Hermes always uses its curated model list with API key auth. The picker falls back to the same curated list for OAuth2 / ADC:

| Model | ID |
|-------|-----|
| Gemini 3.6 Flash | `google/gemini-3.6-flash` |
| Gemini 3.5 Flash | `google/gemini-3.5-flash` |
| Gemini 3.5 Flash Lite | `google/gemini-3.5-flash-lite` |
| Gemini 3.1 Pro Preview | `google/gemini-3.1-pro-preview` |
| Gemini 3.1 Flash Lite | `google/gemini-3.1-flash-lite` |
| Gemini 3 Flash Preview | `google/gemini-3-flash-preview` |
| Gemini 3 Pro Preview | `google/gemini-3-pro-preview` |
| Gemini 2.5 Pro | `google/gemini-2.5-pro` |
| Gemini 2.5 Flash | `google/gemini-2.5-flash` |
| Gemini 2.5 Flash Lite | `google/gemini-2.5-flash-lite` |
| Gemini Flash Latest | `google/gemini-flash-latest` |
| Gemini Flash Lite Latest | `google/gemini-flash-lite-latest` |

:::note Region-specific availability
Models must be available in the region you configure. For example, `us-central1` has the broadest model availability, while `europe-west4` may not have all preview models.
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

The doctor reports whether Vertex credentials can be resolved (API key, service-account path, or ADC) and whether the provider is configured.

## Troubleshooting

### "Vertex AI credentials could not be resolved"

Hermes couldn't find valid credentials. Either:

- **Set up API key** (recommended): Set `GOOGLE_VERTEX_API_KEY`, `GOOGLE_VERTEX_PROJECT`, and optionally `GOOGLE_VERTEX_LOCATION` in `~/.hermes/.env`.

- **Set up OAuth2**: Set `VERTEX_CREDENTIALS_PATH` in `~/.hermes/.env`, or run `gcloud auth application-default login`. If your project isn't embedded in the credentials, set `vertex.project_id` in `config.yaml`.

### API key found but no project ID

When using API key auth, you must also set `GOOGLE_VERTEX_PROJECT` in `~/.hermes/.env` to tell Vertex which GCP project to bill.

### `google-auth` not installed

Only needed for OAuth2 / ADC auth. API key auth does not require `google-auth`. Hermes lazy-installs `google-auth` the first time you select Vertex with OAuth2. If that fails, run `hermes setup` or `pip install 'hermes-agent[vertex]'` to repair the installation.

### 404 on Gemini 3.x models

You may be on a regional endpoint that doesn't host the model. Switch to a region that supports it (`us-central1` has the broadest coverage), or check the [Vertex AI model page](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models) for regional availability.

### 403 / permission denied

For API key auth: ensure the Vertex AI API is enabled and that billing is active on your project. For OAuth2: the service account (or your ADC identity) needs the `roles/aiplatform.user` role on the project.

## Related

- [Google Gemini (AI Studio)](/guides/google-gemini) — static-API-key Gemini without GCP
- [AWS Bedrock](/guides/aws-bedrock) — another native cloud-provider integration
- [AI Providers](/integrations/providers)
- [Configuration](/user-guide/configuration)
