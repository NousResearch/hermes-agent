---
sidebar_position: 15
title: "Google Vertex AI"
description: "Use Hermes Agent with Google Vertex AI -- Claude models via the AnthropicVertex SDK with Application Default Credentials"
---

# Google Vertex AI

Hermes Agent supports Google Vertex AI as a native provider using Anthropic's **AnthropicVertex SDK** -- the same `.messages.create()` interface as the regular Anthropic client. This gives you full Claude feature parity (prompt caching, thinking budgets) while authenticating through Google Cloud's credential chain.

:::info Claude models only (for now)
The Vertex provider currently supports **Claude models only** via the AnthropicVertex SDK. Gemini and other model support on Vertex is planned for a future release. For Gemini via Google AI Studio, use the `gemini` provider instead.
:::

## Prerequisites

- **GCP project** with the [Vertex AI API](https://console.cloud.google.com/apis/api/aiplatform.googleapis.com) enabled
- **Claude model access** -- request access to Anthropic models in the [Vertex AI Model Garden](https://console.cloud.google.com/vertex-ai/model-garden)
- **Google credentials** -- any source supported by [Application Default Credentials (ADC)](https://cloud.google.com/docs/authentication/provide-credentials-adc):
  - `gcloud auth application-default login` (local development)
  - `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json`
  - Attached service account on GCE, GKE, Cloud Run (zero config)

:::tip GCE / GKE / Cloud Run
On Google Cloud compute, attach a service account with the `Vertex AI User` role and you're done. No API keys, no `.env` configuration -- Hermes detects the credentials automatically via ADC.
:::

## Quick Start

```bash
# Select Vertex as your provider
hermes model
# → Choose "Google Vertex AI"
# → Enter your GCP project ID and region

# Start chatting
hermes chat
```

## Configuration

After running `hermes model`, your `~/.hermes/config.yaml` will contain:

```yaml
model:
  default: claude-sonnet-4-6
  provider: vertex

vertex:
  project_id: your-gcp-project
  region: global
```

### Environment Variables

Set these in `~/.hermes/.env` or your shell:

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_VERTEX_PROJECT_ID` | **Yes** | GCP project ID (highest priority) |
| `GOOGLE_CLOUD_PROJECT` | Fallback | Standard GCP project variable |
| `GCLOUD_PROJECT` | Fallback | Legacy GCP project variable |
| `CLOUD_ML_REGION` | No | Vertex AI region (default: `global`) |
| `GOOGLE_APPLICATION_CREDENTIALS` | No | Path to service account JSON |

Project ID resolution order: `ANTHROPIC_VERTEX_PROJECT_ID` > `GOOGLE_CLOUD_PROJECT` > `GCLOUD_PROJECT`.

### Region

Vertex AI supports multiple regions. The default is `global`, which routes to the nearest available region. You can also specify a specific region:

```bash
export CLOUD_ML_REGION=us-east5
```

Common regions: `global`, `us-east5`, `us-central1`, `europe-west1`, `europe-west4`, `asia-southeast1`.

## Available Models

Vertex AI serves Claude models using the same model IDs as the Anthropic API:

| Model | ID | Notes |
|-------|-----|-------|
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | Recommended -- best balance of speed and capability |
| Claude Opus 4.6 | `claude-opus-4-6` | Most capable |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | Fastest, cheapest |

## Switching Models Mid-Session

Use the `/model` command during a conversation:

```
/model claude-opus-4-6
/model claude-haiku-4-5-20251001
```

## Diagnostics

```bash
hermes doctor
```

The doctor checks whether the Vertex project/region can be resolved from environment variables. Live ADC validation happens at request time.

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

## Troubleshooting

### "No Vertex Anthropic project configured"

Set one of the project environment variables:

```bash
echo "ANTHROPIC_VERTEX_PROJECT_ID=your-project" >> ~/.hermes/.env
```

### "google.auth" import error

The `google-auth` package is required. Hermes will try to install it automatically, or install manually:

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[anthropic-vertex]"
```

### "Model is not a supported Claude model"

The Vertex provider currently supports Claude models only. If you need Gemini on Vertex, use the `gemini` provider with a Google AI Studio API key, or configure a custom provider pointing at the Vertex OpenAI-compatible endpoint.

### Permission denied / 403

Ensure your GCP project has:
1. The Vertex AI API enabled
2. Access to the requested Claude model in the Model Garden
3. The authenticated identity has the `Vertex AI User` role (or `roles/aiplatform.user`)
