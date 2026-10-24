---
sidebar_position: 15
title: "Google Vertex AI"
description: "Use Hermes Agent with Gemini on Google Cloud Vertex AI — OAuth2 service account or ADC, GCP billing and quotas, no static API key"
---

# Google Vertex AI

Hermes Agent supports **Gemini models on Google Cloud Vertex AI** through Vertex's OpenAI-compatible endpoint. Unlike the [Google AI Studio provider](./google-gemini.md) (which uses a static API key against `generativelanguage.googleapis.com`), Vertex gives you **enterprise-grade rate limits and GCP billing/credits**, and is the right choice when you want Gemini usage to draw on your Google Cloud account rather than an AI Studio key.

:::info Vertex authenticates with OAuth2, not an API key
Vertex has **no static API key** for the standard endpoint. Every request needs a short-lived **OAuth2 access token** (≈1 hour TTL) minted from either a service-account JSON or Application Default Credentials (ADC). Hermes mints and **auto-refreshes** these tokens for you — you never paste a token by hand. This is why pasting a temporary token into a custom provider's `api_key` field does not work: it expires mid-session.
:::

## Prerequisites

- **A Google Cloud project** with the **Vertex AI API enabled** and billing active.
- **Credentials**, one of:
  - a **service-account JSON** key file with the `roles/aiplatform.user` role, or
  - **Application Default Credentials** via `gcloud auth application-default login` (or the metadata server when running on a GCP VM).
- A **Google Cloud project ID** — either embedded in the credentials or set under `vertex.project_id` in `config.yaml`.
- **`google-auth`** — installed automatically the first time you select Vertex (lazy install). Run `hermes setup` to repair a managed install if that fails.

## Quick Start

```bash
# Option A — service account JSON (recommended for servers / gateways)
echo "VERTEX_CREDENTIALS_PATH=/path/to/service-account.json" >> ~/.hermes/.env

# Option B — gcloud Application Default Credentials (good for local dev)
gcloud auth application-default login
#   ↳ writes an "authorized_user" ADC JSON at
#     ~/.config/gcloud/application_default_credentials.json
#   point Hermes at it explicitly (it accepts this file as-is):
echo "VERTEX_CREDENTIALS_PATH=$HOME/.config/gcloud/application_default_credentials.json" >> ~/.hermes/.env

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
#   VERTEX_CREDENTIALS_PATH         -> a service-account JSON *or* a gcloud "authorized_user" ADC file
#   GOOGLE_APPLICATION_CREDENTIALS -> same file kinds
VERTEX_CREDENTIALS_PATH=/path/to/service-account.json
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

:::info Two credential file kinds are accepted
`VERTEX_CREDENTIALS_PATH` / `GOOGLE_APPLICATION_CREDENTIALS` accept **either**:

| File | What it is | How to get it |
|------|------------|---------------|
| **service-account JSON** | A key with `"type": "service_account"` and a `client_email` / `private_key` | `gcloud iam service-accounts keys create <name>.json --iam-account=<sa>@<project>.iam.gserviceaccount.com` |
| **gcloud `authorized_user` JSON** | The ADC file `gcloud auth application-default login` writes, with `"type": "authorized_user"` and a refresh token | `gcloud auth application-default login` (writes `~/.config/gcloud/application_default_credentials.json`) |

The two are **not interchangeable**: an `authorized_user` file is a user OAuth credential, and a
service-account JSON is a key-pair credential. Point `VERTEX_CREDENTIALS_PATH` at the file you
actually have — do not rename one into the other. If a credentials file is read as the wrong kind,
`google-auth` rejects it with:

```
MalformedError: Service account info was not in the expected format, missing fields client_email, token_uri.
```

For a service account, remember to grant the key's `client_email` the `roles/aiplatform.user` role
on the project.
:::

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

1. Hermes resolves credentials in this order: `VERTEX_CREDENTIALS_PATH` → `GOOGLE_APPLICATION_CREDENTIALS` → ADC. The first two accept **either** a service-account JSON **or** a gcloud `authorized_user` ADC file (see "Two credential file kinds are accepted" above); the ADC fallback uses the metadata server or the well-known gcloud ADC file.
2. It mints an OAuth2 access token (`cloud-platform` scope) and caches it, refreshing when the token is within 5 minutes of expiry.
3. The token is handed to a standard OpenAI client pointed at the Vertex endpoint:
   ```text
   https://aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{region}/endpoints/openapi
   ```
   Regional locations use a `{region}-aiplatform.googleapis.com` host instead.
4. If a session runs longer than the token lifetime and a request returns `401`, Hermes re-mints the token and retries automatically. On a long-running gateway, if ADC's refresh token has itself expired, Hermes falls back to the service-account JSON when one is configured.

## Available Models

Vertex requires the `google/` vendor prefix on model IDs. The following ids are live on the Vertex
`global` endpoint (verified against the publisher model list) and are safe to use:

| Model | ID |
|-------|----|
| Gemini 4 Argon | `google/gemini-4-argon` |
| Gemini 3.8 Flash | `google/gemini-3.8-flash` |
| Gemini 3.7 Flash | `google/gemini-3.7-flash` |
| Gemini 3.6 Flash | `google/gemini-3.6-flash` |
| Gemini 3.5 Flash | `google/gemini-3.5-flash` |
| Gemini 3.5 Flash Lite | `google/gemini-3.5-flash-lite` |
| Gemini 3.1 Pro Preview | `google/gemini-3.1-pro-preview` |
| Gemini 3.1 Flash Lite | `google/gemini-3.1-flash-lite` |
| Gemini 3 Flash Preview | `google/gemini-3-flash-preview` |
| Gemini 2.5 Pro | `google/gemini-2.5-pro` |
| Gemini 2.5 Flash | `google/gemini-2.5-flash` |

:::note Retired model ids
Google has retired some older Gemini 3 preview ids. They now return
`404 Publisher model ... was not found` on the Vertex endpoint. Prefer the current ids above —
`google/gemini-3.1-pro-preview` for the Pro tier and `google/gemini-3.1-flash-lite` for the Lite
tier. Every id in the table above was verified live against the `global` endpoint.
:::

:::note `global` region for Gemini 3.x
The Gemini 3.x preview models are served through the `global` endpoint. Regional endpoints (`us-central1`, etc.) may 404 them. Leave `region: global` unless you have a specific reason to pin a region.
:::

## Switching Models Mid-Session

```text
/model google/gemini-3.1-pro-preview
/model google/gemini-3-flash-preview
```

`/model` switches among already-configured providers and models; it does not collect new credentials. Configure Vertex with `hermes model` first.

## Reasoning / Thinking

Vertex exposes Gemini's thinking budget through the OpenAI-compatible surface. Hermes maps its reasoning-effort setting onto `extra_body.google.thinking_config` automatically, so `reasoning_effort` works the same way it does on other Gemini surfaces.

## Priority PayGo

Vertex's shared Standard PayGo pool serves every customer, so a busy pool answers some requests with `429 RESOURCE_EXHAUSTED`. Google's other consumption options are Provisioned Throughput (reserved capacity, an upfront commitment) and Priority PayGo.

Priority PayGo sends the `X-Vertex-AI-LLM-Shared-Request-Type: priority` header, which prioritises your traffic over the shared pool without an upfront commitment. **It is billed at a higher per-token rate than Standard PayGo**, so it is off by default and must be enabled explicitly:

```yaml
vertex:
  project_id: my-gcp-project
  region: global
  priority_paygo: true
```

It is supported on the `global`, `us` and `eu` endpoints. Leave it off unless the capacity 429s are actually costing you more than the rate difference.

## Diagnostics

```bash
hermes doctor
```

The doctor reports whether Vertex credentials can be resolved (service-account path or ADC) and whether the provider is configured.

## Troubleshooting

### "Vertex AI credentials could not be resolved"

Hermes emits this error verbatim when it can resolve neither a credentials file nor working ADC:

> Vertex AI credentials could not be resolved. Vertex uses OAuth2 (not a static API key): point GOOGLE_APPLICATION_CREDENTIALS (or VERTEX_CREDENTIALS_PATH) in ~/.hermes/.env at a credential file - a service-account JSON, or the gcloud 'authorized_user' ADC file - or run 'gcloud auth application-default login' for ADC. Set the GCP project/region under vertex: in config.yaml if they aren't embedded in the credentials. Run `hermes setup` to install Vertex support.

Fix it by pointing `VERTEX_CREDENTIALS_PATH` in `~/.hermes/.env` at **either** a service-account JSON
**or** the gcloud `authorized_user` ADC file (see "Two credential file kinds are accepted" above), and
by setting `vertex.project_id` in `config.yaml` if the project isn't embedded in the credentials.

### `MalformedError: Service account info was not in the expected format, missing fields client_email, token_uri.`

The file at `VERTEX_CREDENTIALS_PATH` was read as a service-account key but is not one — most often
it is the gcloud `authorized_user` ADC file from `gcloud auth application-default login`. Use a real
service-account JSON here, or point the variable at the credential file that matches its type.

### `google-auth` not installed

Hermes lazy-installs it the first time you select the Vertex provider. If that fails, run `hermes setup` to repair the managed install.

### 404 on Gemini 3.x models

You are probably on a regional endpoint. Set `region: global` in the `vertex:` section of `config.yaml` (or unset `VERTEX_REGION`).

### 403 / permission denied

The service account (or your ADC identity) needs the `roles/aiplatform.user` role on the project, and the Vertex AI API must be enabled for that project.

## Related

- [Google Gemini (AI Studio)](./google-gemini.md) — static-API-key Gemini without GCP
- [AWS Bedrock](./aws-bedrock.md) — another native cloud-provider integration
- [AI Providers](../integrations/providers.md)
- [Configuration](../user-guide/configuration.md)
