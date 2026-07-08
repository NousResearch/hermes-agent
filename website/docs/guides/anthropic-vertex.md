---
sidebar_position: 16
title: "Anthropic on Google Vertex AI"
description: "Use Hermes Agent with Anthropic Claude models on Vertex AI — OAuth2 via ADC, GCP billing, no Anthropic API key required"
---

# Anthropic on Google Vertex AI

Hermes Agent supports **Anthropic Claude models on Google Cloud Vertex AI**. This is the third way to run Claude from Hermes, alongside the native [Anthropic provider](/guides/anthropic) (API key from `console.anthropic.com`, billed by Anthropic) and the [AWS Bedrock provider](/guides/aws-bedrock) (Claude via AWS, billed by AWS). Same Anthropic Messages API surface as native Anthropic — same feature set (prompt caching, adaptive thinking, tool-use streaming, xhigh effort) — but authenticated with a Google Cloud OAuth2 token and billed through your GCP account.

:::info Vertex is one Hermes provider hosting two model families
Anthropic Claude on Vertex uses the **same `vertex` provider** as Gemini on Vertex — they run on the same GCP platform under the same ADC auth and share every routing config. Hermes chooses the correct wire (AnthropicVertex vs. Vertex's OpenAI-compat aggregator) automatically based on the model name: `anthropic/…` and bare `claude-…` names go through Anthropic's SDK; everything else routes to the aggregator. You never paste a Claude API key when using this provider.
:::

## Prerequisites

- **A Google Cloud project** with the **Vertex AI API enabled** and billing active.
- **Anthropic models enabled in Vertex Model Garden.** Claude models on Vertex are partner models — they require a one-time console click per model to accept Anthropic's terms and start the Google Cloud Marketplace subscription. Go to [Vertex AI Model Garden](https://console.cloud.google.com/vertex-ai/model-garden), filter by **Anthropic**, and click **Enable** on each Claude SKU you want reachable (Opus, Sonnet, Haiku). Without this, every Claude request returns `HTTP 404: Publisher model … was not found` — the same 404 shape you'd see for a genuinely unknown model.
- **Credentials**, one of:
  - a **service-account JSON** key file with the `roles/aiplatform.user` role, or
  - **Application Default Credentials** via `gcloud auth application-default login` (or the metadata server when running on a GCP VM).
- **`anthropic>=0.39.0`** and **`google-auth`** — installed automatically the first time you select the provider (lazy install), or explicitly with `pip install 'anthropic>=0.39.0' 'hermes-agent[anthropic]'`.

## Quick Start

```bash
# Option A — service account JSON (recommended for servers / gateways)
echo "VERTEX_CREDENTIALS_PATH=/path/to/service-account.json" >> ~/.hermes/.env

# Option B — Application Default Credentials (good for local dev)
gcloud auth application-default login

# Set the routing config
cat >> ~/.hermes/config.yaml <<'YAML'
model:
  default: anthropic/claude-opus-4-8
  provider: vertex

vertex:
  project_id: my-gcp-project
  region: global
YAML

# Start chatting — no API key prompts
hermes chat
```

## Configuration

Anthropic on Vertex is a routing mode of the shared `vertex` provider — it shares its credential and routing configuration with [Gemini-on-Vertex](/guides/google-vertex). If you already have Gemini on Vertex working, the same `vertex.project_id` / `vertex.region` values apply here; only the `model.default` value changes.

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
  default: anthropic/claude-opus-4-8   # ``anthropic/`` prefix picks the AnthropicVertex path
  provider: vertex

vertex:
  project_id: my-gcp-project   # blank → use the project embedded in the credentials
  region: global               # or a regional endpoint that serves the model
```

The `anthropic/` prefix is what Hermes uses to dispatch the request onto the AnthropicVertex SDK. The prefix is **required** — Vertex Model Garden is a multi-vendor surface, so a bare `claude-opus-4-8` has no unambiguous meaning and Hermes deliberately does not guess. Names without the `anthropic/` prefix fall through to the OpenAI-compat aggregator path, where Vertex will 404 with a message pointing at the missing prefix. Gemini models continue to use the `google/gemini-…` naming (required by Vertex's OpenAI-compat aggregator) on the same provider.

:::tip Environment variables win over config.yaml
`VERTEX_PROJECT_ID` and `VERTEX_REGION` override the `vertex.project_id` / `vertex.region` values in `config.yaml`. Use them for per-shell overrides; keep the durable settings in `config.yaml`.
:::

### How dispatch works

Hermes classifies the requested model at runtime:

| Model shape                        | Wire route                                    | SDK                          |
|------------------------------------|-----------------------------------------------|------------------------------|
| `anthropic/claude-…`               | `publishers/anthropic/models/<model>:rawPredict` | `anthropic.AnthropicVertex` |
| `google/gemini-…`, everything else | `endpoints/openapi/…` (OpenAI-compat)         | OpenAI SDK against Vertex   |
| bare `claude-…` (missing prefix)   | falls through to OpenAI-compat, Vertex 404s   | user-facing error            |

The `anthropic/` vendor prefix is stripped internally before hitting the wire — the AnthropicVertex SDK constructs its URL as `.../publishers/anthropic/models/{model}:rawPredict` and expects a bare model name in the request body.

### How authentication works

1. Hermes resolves Google credentials in this order: `VERTEX_CREDENTIALS_PATH` → `GOOGLE_APPLICATION_CREDENTIALS` → Application Default Credentials.
2. Hermes constructs an `anthropic.AnthropicVertex(project_id, region, credentials=…)` client and hands it the Credentials object.
3. The Anthropic SDK's own transport mints and refreshes access tokens on demand via `google-auth`, and constructs the correct `publishers/anthropic/models/<model>:rawPredict` URL for every request.
4. Because the Anthropic SDK handles auth internally, mid-session token expiry is transparent — no separate refresh logic on the Hermes side.

## Available Models

Vertex uses Anthropic's own model IDs, with or without a `@YYYYMMDD` version suffix. Prefix with `anthropic/` in your Hermes config so the dispatcher routes through the AnthropicVertex SDK.

| Model | `model.default` value in `config.yaml`      |
|-------|----------------------------------------------|
| Claude Opus 4.8 | `anthropic/claude-opus-4-8`        |
| Claude Opus 4.5 | `anthropic/claude-opus-4-5`        |
| Claude Opus 4.1 | `anthropic/claude-opus-4-1`        |
| Claude Opus 4 | `anthropic/claude-opus-4`            |
| Claude Sonnet 4.5 | `anthropic/claude-sonnet-4-5`    |
| Claude Sonnet 4 | `anthropic/claude-sonnet-4`        |
| Claude Haiku 4.5 | `anthropic/claude-haiku-4-5`      |
| Claude 3.7 Sonnet | `anthropic/claude-3-7-sonnet`    |
| Claude 3.5 Sonnet v2 | `anthropic/claude-3-5-sonnet-v2` |
| Claude 3.5 Haiku | `anthropic/claude-3-5-haiku`      |

:::warning Per-region model availability
Not every Anthropic SKU is served in every region. `global` serves the broadest set (all currently-listed frontier models); regional endpoints (`us-east5`, `europe-west1`, `asia-southeast1`, …) may 404 on newer models. Check the [Anthropic Claude on Vertex model reference](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/use-claude) for the current region matrix. When in doubt, pin `region: global` — it routes to the best-available regional replica automatically.
:::

## Feature Parity

Claude on Vertex uses Anthropic's `AnthropicVertex` SDK, which speaks the same wire protocol as `anthropic.Anthropic`. All Claude features work identically:

- **Prompt caching** (input token cost reduction on multi-turn conversations)
- **Adaptive thinking** (`reasoning_effort` maps to `output_config.effort`)
- **`xhigh` effort level** on Opus 4.7+
- **Tool-use streaming** (`fine-grained-tool-streaming` beta)
- **Interleaved thinking** across tool calls
- **Extended thinking budgets** on Opus 4.5 and older
- **1M-token context window** on Opus 4.6+ and Sonnet 4.6+ (see below)

### 1M context — no header required

The 1M-token context window is generally available on Vertex-hosted Opus 4.6+ / Sonnet 4.6+ as of March 2026 ([Anthropic GA announcement](https://claude.com/blog/1m-context-ga)). Requests over 200K tokens work automatically on this wire — no beta header, no code change, no per-project opt-in beyond the standard Vertex Model Garden enablement for the SKU. If you still send `context-1m-2025-08-07` (e.g. from code that talks to native Anthropic in a mixed-backend setup) the Vertex path silently ignores it.

Hermes deliberately does not attach the header on the Vertex path — sending a no-op is misleading and would break the header-based gating story on native Anthropic backends that DO still require it. There is nothing to configure to unlock 1M through this provider.

## Switching Models Mid-Session

```text
/model anthropic/claude-opus-4-8
/model anthropic/claude-sonnet-4-5
/model anthropic/claude-haiku-4-5
```

`/model` switches among already-configured providers and models; it does not collect new credentials. Configure the provider once via `hermes model` (or by editing `config.yaml` directly), then use `/model` to hot-swap. Switching between Anthropic and Gemini models on the same `vertex` provider is a single `/model` call — no provider swap, no re-auth.

## Diagnostics

```bash
hermes doctor
```

The doctor reports whether Vertex credentials can be resolved (service-account path or ADC) and whether the provider is configured.

### Common failure modes

- **`HTTP 404: Publisher model … was not found`** — the specific Claude model is not enabled in your project's Vertex Model Garden. Enable it via the console (see Prerequisites).
- **`HTTP 400: Publisher model … not available in region`** — the model is enabled but not served in the region you pinned. Switch `region` to `global` or a region listed in the [Anthropic on Vertex model reference](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/use-claude).
- **`google.auth.exceptions.DefaultCredentialsError`** — no service-account JSON and no ADC. Run `gcloud auth application-default login` or set `VERTEX_CREDENTIALS_PATH` in `~/.hermes/.env`.
- **`HTTP 401`** after a long-idle gateway session — ADC refresh token expired. Configure a service-account JSON via `VERTEX_CREDENTIALS_PATH` for long-running gateways; ADC is a developer-workflow credential.

## Related

- [Google Vertex AI (Gemini)](/guides/google-vertex) — same platform, Gemini models via the OpenAI-compatible endpoint. Same `vertex` provider, same `vertex.project_id` / `vertex.region` config; the model name (`google/gemini-…` vs. `anthropic/claude-…`) picks the wire.
- [Anthropic (native)](/guides/anthropic) — Claude via Anthropic's own API. Different auth (`x-api-key`), different billing.
- [AWS Bedrock](/guides/aws-bedrock) — Claude via AWS. `AnthropicBedrock` SDK, IAM credentials.
