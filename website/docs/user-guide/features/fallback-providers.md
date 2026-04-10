---
title: Fallback Providers
description: Configure automatic failover to backup LLM providers when your primary model is unavailable.
sidebar_label: Fallback Providers
sidebar_position: 8
---

# Fallback Providers

Hermes Agent has three layers of resilience that keep your sessions running when providers hit issues:

1. **[Credential pools](./credential-pools.md)** — rotate across multiple API keys for the *same* provider (tried first)
2. **Primary model fallback** — automatically switches to a *different* provider:model when your main model fails
3. **Auxiliary task fallback** — independent provider resolution for side tasks like vision, compression, and web extraction

Credential pools handle same-provider rotation (e.g., multiple OpenRouter keys). This page covers cross-provider fallback. Both are optional and work independently.

## Primary Model Fallback

When your main LLM provider encounters errors — rate limits, server overload, auth failures, connection drops — Hermes can automatically switch to a backup provider:model pair mid-session without losing your conversation.

### Configuration

Add a `fallback_model` section to `~/.hermes/config.yaml`:

```yaml
fallback_model:
  provider: openrouter
  model: anthropic/claude-sonnet-4
```

Both `provider` and `model` are **required**. If either is missing, the fallback is disabled.

### Supported Providers

| Provider | Value | Requirements |
|----------|-------|-------------|
| AI Gateway | `ai-gateway` | `AI_GATEWAY_API_KEY` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| Nous Portal | `nous` | `hermes auth` (OAuth) |
| OpenAI Codex | `openai-codex` | `hermes model` (ChatGPT OAuth) |
| GitHub Copilot | `copilot` | `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, or `GITHUB_TOKEN` |
| GitHub Copilot ACP | `copilot-acp` | External process (editor integration) |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` or Claude Code credentials |
| z.ai / GLM | `zai` | `GLM_API_KEY` |
| Kimi / Moonshot | `kimi-coding` | `KIMI_API_KEY` |
| MiniMax | `minimax` | `MINIMAX_API_KEY` |
| MiniMax (China) | `minimax-cn` | `MINIMAX_CN_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| OpenCode Zen | `opencode-zen` | `OPENCODE_ZEN_API_KEY` |
| OpenCode Go | `opencode-go` | `OPENCODE_GO_API_KEY` |
| Kilo Code | `kilocode` | `KILOCODE_API_KEY` |
| Alibaba / DashScope | `alibaba` | `DASHSCOPE_API_KEY` |
| Hugging Face | `huggingface` | `HF_TOKEN` |
| Custom endpoint | `custom` | `base_url` + `api_key_env` (see below) |

### Custom Endpoint Fallback

For a custom OpenAI-compatible endpoint, add `base_url` and optionally `api_key_env`:

```yaml
fallback_model:
  provider: custom
  model: my-local-model
  base_url: http://localhost:8000/v1
  api_key_env: MY_LOCAL_KEY          # env var name containing the API key
```

### When Fallback Triggers

The fallback activates automatically when the primary model fails with:

- **Rate limits** (HTTP 429) — after exhausting retry attempts
- **Server errors** (HTTP 500, 502, 503) — after exhausting retry attempts
- **Auth failures** (HTTP 401, 403) — immediately (no point retrying)
- **Not found** (HTTP 404) — immediately
- **Invalid responses** — when the API returns malformed or empty responses repeatedly

When triggered, Hermes:

1. Resolves credentials for the fallback provider
2. Builds a new API client
3. Swaps the model, provider, and client in-place
4. Resets the retry counter and continues the conversation

The switch is seamless — your conversation history, tool calls, and context are preserved. The agent continues from exactly where it left off, just using a different model.

:::info One-Shot
Fallback activates **at most once** per session. If the fallback provider also fails, normal error handling takes over (retries, then error message). This prevents cascading failover loops.
:::

### Examples

**OpenRouter as fallback for Anthropic native:**
```yaml
model:
  provider: anthropic
  default: claude-sonnet-4-6

fallback_model:
  provider: openrouter
  model: anthropic/claude-sonnet-4
```

**Nous Portal as fallback for OpenRouter:**
```yaml
model:
  provider: openrouter
  default: anthropic/claude-opus-4

fallback_model:
  provider: nous
  model: nous-hermes-3
```

**Local model as fallback for cloud:**
```yaml
fallback_model:
  provider: custom
  model: llama-3.1-70b
  base_url: http://localhost:8000/v1
  api_key_env: LOCAL_API_KEY
```

**Codex OAuth as fallback:**
```yaml
fallback_model:
  provider: openai-codex
  model: gpt-5.3-codex
```

### Where Fallback Works

| Context | Fallback Supported |
|---------|-------------------|
| CLI sessions | ✔ |
| Messaging gateway (Telegram, Discord, etc.) | ✔ |
| Subagent delegation | ✘ (subagents do not inherit fallback config) |
| Cron jobs | ✘ (run with a fixed provider) |
| Auxiliary tasks (vision, compression) | ✘ (use their own provider chain — see below) |

:::tip
There are no environment variables for `fallback_model` — it is configured exclusively through `config.yaml`. This is intentional: fallback configuration is a deliberate choice, not something a stale shell export should override.
:::

---

## Auxiliary Task Fallback

Hermes uses separate lightweight models for side tasks. Each task has its own provider resolution chain that acts as a built-in fallback system.

### Tasks with Independent Provider Resolution

| Task | What It Does | Config Key |
|------|-------------|-----------|
| Vision | Image analysis, browser screenshots | `auxiliary.vision` |
| Web Extract | Web page summarization | `auxiliary.web_extract` |
| Compression | Context compression summaries | `auxiliary.compression` or `compression.summary_provider` |
| Session Search | Past session summarization | `auxiliary.session_search` |
| Skills Hub | Skill search and discovery | `auxiliary.skills_hub` |
| MCP | MCP helper operations | `auxiliary.mcp` |
| Memory Flush | Memory consolidation | `auxiliary.flush_memories` |

### Auto-Detection Chain

When a task's provider is set to `"auto"` (the default), Hermes tries providers in order until one works:

**For text tasks (compression, web extract, etc.):**

```text
OpenRouter → Nous Portal → Custom endpoint → Codex OAuth →
API-key providers (z.ai, Kimi, MiniMax, Hugging Face, Anthropic) → give up
```

**For vision tasks:**

```text
Main provider (if vision-capable) → OpenRouter → Nous Portal →
Codex OAuth → Anthropic → Custom endpoint → give up
```

If the resolved provider fails at call time, Hermes also has an internal retry: if the provider is not OpenRouter and no explicit `base_url` is set, it tries OpenRouter as a last-resort fallback.

### Configuring Auxiliary Providers

Each task can be configured independently in `config.yaml`:

```yaml
auxiliary:
  vision:
    provider: "auto"              # auto | openrouter | nous | codex | main | anthropic
    model: ""                     # e.g. "openai/gpt-4o"
    base_url: ""                  # direct endpoint (takes precedence over provider)
    api_key: ""                   # API key for base_url

  web_extract:
    provider: "auto"
    model: ""

  compression:
    provider: "auto"
    model: ""

  session_search:
    provider: "auto"
    model: ""

  skills_hub:
    provider: "auto"
    model: ""

  mcp:
    provider: "auto"
    model: ""

  flush_memories:
    provider: "auto"
    model: ""
```

Every task above follows the same **provider / model / base_url** pattern. Context compression uses its own top-level block:

```yaml
compression:
  summary_provider: main                             # Same provider options as auxiliary tasks
  summary_model: google/gemini-3-flash-preview
  summary_base_url: null                             # Custom OpenAI-compatible endpoint
```

And the fallback model uses:

```yaml
fallback_model:
  provider: openrouter
  model: anthropic/claude-sonnet-4
  # base_url: http://localhost:8000/v1               # Optional custom endpoint
```

All three — auxiliary, compression, fallback — work the same way: set `provider` to pick who handles the request, `model` to pick which model, and `base_url` to point at a custom endpoint (overrides provider).

### Provider Options for Auxiliary Tasks

These options apply to `auxiliary:`, `compression:`, and `fallback_model:` configs only — `"main"` is **not** a valid value for your top-level `model.provider`. For custom endpoints, use `provider: custom` in your `model:` section (see [AI Providers](/docs/integrations/providers)).

| Provider | Description | Requirements |
|----------|-------------|-------------|
| `"auto"` | Try providers in order until one works (default) | At least one provider configured |
| `"openrouter"` | Force OpenRouter | `OPENROUTER_API_KEY` |
| `"nous"` | Force Nous Portal | `hermes auth` |
| `"codex"` | Force Codex OAuth | `hermes model` → Codex |
| `"main"` | Use whatever provider the main agent uses (auxiliary tasks only) | Active main provider configured |
| `"anthropic"` | Force Anthropic native | `ANTHROPIC_API_KEY` or Claude Code credentials |

### Direct Endpoint Override

For any auxiliary task, setting `base_url` bypasses provider resolution entirely and sends requests directly to that endpoint:

```yaml
auxiliary:
  vision:
    base_url: "http://localhost:1234/v1"
    api_key: "local-key"
    model: "qwen2.5-vl"
```

`base_url` takes precedence over `provider`. Hermes uses the configured `api_key` for authentication, falling back to `OPENAI_API_KEY` if not set. It does **not** reuse `OPENROUTER_API_KEY` for custom endpoints.

---

## Context Compression Fallback

Context compression has a legacy configuration path in addition to the auxiliary system:

```yaml
compression:
  summary_provider: "auto"                    # auto | openrouter | nous | main
  summary_model: "google/gemini-3-flash-preview"
```

This is equivalent to configuring `auxiliary.compression.provider` and `auxiliary.compression.model`. If both are set, the `auxiliary.compression` values take precedence.

If no provider is available for compression, Hermes drops middle conversation turns without generating a summary rather than failing the session.

---

## Delegation Provider Override

Subagents spawned by `delegate_task` do **not** use the primary fallback model. However, they can be routed to a different provider:model pair for cost optimization:

```yaml
delegation:
  provider: "openrouter"                      # override provider for all subagents
  model: "google/gemini-3-flash-preview"      # override model
  # base_url: "http://localhost:1234/v1"      # or use a direct endpoint
  # api_key: "local-key"
```

See [Subagent Delegation](/docs/user-guide/features/delegation) for full configuration details.

---

## Cron Job Providers

Cron jobs run with whatever provider is configured at execution time. They do not support a fallback model. To use a different provider for cron jobs, configure `provider` and `model` overrides on the cron job itself:

```python
cronjob(
    action="create",
    schedule="every 2h",
    prompt="Check server status",
    provider="openrouter",
    model="google/gemini-3-flash-preview"
)
```

See [Scheduled Tasks (Cron)](/docs/user-guide/features/cron) for full configuration details.

---

## Summary

| Feature | Fallback Mechanism | Config Location |
|---------|-------------------|----------------|
| Main agent model | `fallback_model` in config.yaml — one-shot failover on errors | `fallback_model:` (top-level) |
| Vision | Auto-detection chain + internal OpenRouter retry | `auxiliary.vision` |
| Web extraction | Auto-detection chain + internal OpenRouter retry | `auxiliary.web_extract` |
| Context compression | Auto-detection chain, degrades to no-summary if unavailable | `auxiliary.compression` or `compression.summary_provider` |
| Session search | Auto-detection chain | `auxiliary.session_search` |
| Skills hub | Auto-detection chain | `auxiliary.skills_hub` |
| MCP helpers | Auto-detection chain | `auxiliary.mcp` |
| Memory flush | Auto-detection chain | `auxiliary.flush_memories` |
| Delegation | Provider override only (no automatic fallback) | `delegation.provider` / `delegation.model` |
| Cron jobs | Per-job provider override only (no automatic fallback) | Per-job `provider` / `model` |
