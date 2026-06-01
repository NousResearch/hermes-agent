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

The easiest path is the interactive manager:

```bash
hermes fallback
```

`hermes fallback` reuses the provider picker from `hermes model` — same provider list, same credential prompts, same validation. Use the subcommands `add`, `list` (alias `ls`), `remove` (alias `rm`), and `clear` to manage the chain. Changes persist under the top-level `fallback_providers:` list in `config.yaml`.

If you'd rather edit the YAML directly, add a top-level `fallback_providers` list to `~/.hermes/config.yaml`:

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

Each entry requires both `provider` and `model`. Entries missing either field are ignored.

:::note `fallback_model` vs `fallback_providers`
`fallback_providers` (plural, list) is the current config shape and supports multiple fallbacks tried in order. `fallback_model` (singular) is the legacy single-fallback key — Hermes still honors it for back-compat, but `hermes fallback` writes the current `fallback_providers` key and migrates legacy config on write. When both are set, `fallback_providers` takes priority.
:::

### Supported Providers

| Provider | Value | Requirements |
|----------|-------|-------------|
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| Nous Portal | `nous` | `hermes setup --portal` (fresh) or `hermes auth add nous` (OAuth) |
| OpenAI Codex | `openai-codex` | `hermes model` (ChatGPT OAuth) |
| GitHub Copilot | `copilot` | `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, or `GITHUB_TOKEN` |
| GitHub Copilot ACP | `copilot-acp` | External process (editor integration) |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` or Claude Code credentials |
| z.ai / GLM | `zai` | `GLM_API_KEY` |
| Kimi / Moonshot | `kimi-coding` | `KIMI_API_KEY` |
| MiniMax | `minimax` | `MINIMAX_API_KEY` |
| MiniMax (China) | `minimax-cn` | `MINIMAX_CN_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| NVIDIA NIM | `nvidia` | `NVIDIA_API_KEY` (optional: `NVIDIA_BASE_URL`) |
| GMI Cloud | `gmi` | `GMI_API_KEY` (optional: `GMI_BASE_URL`) |
| StepFun | `stepfun` | `STEPFUN_API_KEY` (optional: `STEPFUN_BASE_URL`) |
| Ollama Cloud | `ollama-cloud` | `OLLAMA_API_KEY` |
| Google Gemini (OAuth) | `google-gemini-cli` | `hermes model` (Google OAuth; optional: `HERMES_GEMINI_PROJECT_ID`) |
| Google AI Studio | `gemini` | `GOOGLE_API_KEY` (alias: `GEMINI_API_KEY`) |
| xAI (Grok) | `xai` (alias `grok`) | `XAI_API_KEY` (optional: `XAI_BASE_URL`) |
| xAI Grok OAuth (SuperGrok) | `xai-oauth` (alias `grok-oauth`) | `hermes model` → xAI Grok OAuth (browser login; SuperGrok subscription) |
| AWS Bedrock | `bedrock` | Standard boto3 auth (`AWS_REGION` + `AWS_PROFILE` or `AWS_ACCESS_KEY_ID`) |
| Qwen Portal (OAuth) | `qwen-oauth` | `hermes model` (Qwen Portal OAuth; optional: `HERMES_QWEN_BASE_URL`) |
| MiniMax (OAuth) | `minimax-oauth` | `hermes model` (MiniMax portal OAuth) |
| OpenCode Zen | `opencode-zen` | `OPENCODE_ZEN_API_KEY` |
| OpenCode Go | `opencode-go` | `OPENCODE_GO_API_KEY` |
| Kilo Code | `kilocode` | `KILOCODE_API_KEY` |
| Xiaomi MiMo | `xiaomi` | `XIAOMI_API_KEY` |
| Arcee AI | `arcee` | `ARCEEAI_API_KEY` |
| GMI Cloud | `gmi` | `GMI_API_KEY` |
| Alibaba / DashScope | `alibaba` | `DASHSCOPE_API_KEY` |
| Alibaba Coding Plan | `alibaba-coding-plan` | `ALIBABA_CODING_PLAN_API_KEY` (falls back to `DASHSCOPE_API_KEY`) |
| Kimi / Moonshot (China) | `kimi-coding-cn` | `KIMI_CN_API_KEY` |
| StepFun | `stepfun` | `STEPFUN_API_KEY` |
| Tencent TokenHub | `tencent-tokenhub` | `TOKENHUB_API_KEY` |
| Microsoft Foundry | `azure-foundry` | `AZURE_FOUNDRY_API_KEY` + `AZURE_FOUNDRY_BASE_URL` |
| LM Studio (local) | `lmstudio` | `LM_API_KEY` (or none for local) + `LM_BASE_URL` |
| Hugging Face | `huggingface` | `HF_TOKEN` |
| Custom endpoint | `custom` | `base_url` + `key_env` (see below) |

### Custom Endpoint Fallback

For a custom OpenAI-compatible endpoint, add `base_url` and optionally `key_env`:

```yaml
fallback_providers:
  - provider: custom
    model: my-local-model
    base_url: http://localhost:8000/v1
    key_env: MY_LOCAL_KEY            # env var name containing the API key
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

:::info Per-Turn, Not Per-Session
Fallback is **turn-scoped**: each new user message starts with the primary model restored. If the primary fails mid-turn, fallback activates for that turn only. On the next message, Hermes tries the primary again. Within a single turn, fallback activates at most once — if the fallback also fails, normal error handling takes over (retries, then error message). This prevents cascading failover loops within a turn while giving the primary model a fresh chance every turn.
:::

### Examples

**OpenRouter as fallback for Anthropic native:**
```yaml
model:
  provider: anthropic
  default: claude-sonnet-4-6

fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

**Nous Portal as fallback for OpenRouter:**
```yaml
model:
  provider: openrouter
  default: anthropic/claude-opus-4

fallback_providers:
  - provider: nous
    model: nous-hermes-3
```

**Local model as fallback for cloud:**
```yaml
fallback_providers:
  - provider: custom
    model: llama-3.1-70b
    base_url: http://localhost:8000/v1
    key_env: LOCAL_API_KEY
```

**Codex OAuth as fallback:**
```yaml
fallback_providers:
  - provider: openai-codex
    model: gpt-5.3-codex
```

### Where Fallback Works

| Context | Fallback Supported |
|---------|-------------------|
| CLI sessions | ✔ |
| Messaging gateway (Telegram, Discord, etc.) | ✔ |
| Subagent delegation | ✔ (subagents inherit the parent fallback chain) |
| Cron jobs | ✔ (cron agents inherit configured fallback providers) |
| Auxiliary tasks (vision, compression) | ✘ (use their own provider chain — see below) |

:::tip
There are no environment variables for the primary fallback chain — configure it exclusively through `config.yaml` or `hermes fallback`. This is intentional: fallback configuration is a deliberate choice, not something a stale shell export should override.
:::

---

## Auxiliary Task Fallback

Hermes uses separate lightweight models for side tasks. Each task has its own provider resolution chain that acts as a built-in fallback system.

### Tasks with Independent Provider Resolution

| Task | What It Does | Config Key |
|------|-------------|-----------|
| Vision | Image analysis, browser screenshots | `auxiliary.vision` |
| Web Extract | Web page summarization | `auxiliary.web_extract` |
| Compression | Context compression summaries | `auxiliary.compression` |
| Skills Hub | Skill search and discovery | `auxiliary.skills_hub` |
| MCP | MCP helper operations | `auxiliary.mcp` |
| Approval | Smart command-approval classification | `auxiliary.approval` |
| Title Generation | Session title summaries | `auxiliary.title_generation` |
| Triage Specifier | `hermes kanban specify` / dashboard ✨ button — fleshes out a one-liner triage task into a real spec | `auxiliary.triage_specifier` |

### Auto-Detection Chain

When a task's provider is set to `"auto"` (the default), Hermes tries providers in order until one works:

**For text tasks (compression, web extract, etc.):**

```text
OpenRouter → Nous Portal → Custom endpoint → Codex OAuth →
API-key providers (z.ai, Kimi, MiniMax, Xiaomi MiMo, Hugging Face, Anthropic) → give up
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

  skills_hub:
    provider: "auto"
    model: ""

  mcp:
    provider: "auto"
    model: ""
```

Every task above follows the same **provider / model / base_url** pattern. Context compression is configured under `auxiliary.compression`:

```yaml
auxiliary:
  compression:
    provider: main                                    # Same provider options as other auxiliary tasks
    model: google/gemini-3-flash-preview
    base_url: null                                    # Custom OpenAI-compatible endpoint
```

And the primary fallback chain uses:

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
    # base_url: http://localhost:8000/v1             # Optional custom endpoint
```

All three — auxiliary, compression, fallback — work the same way: set `provider` to pick who handles the request, `model` to pick which model, and `base_url` to point at a custom endpoint (overrides provider).

### Provider Options for Auxiliary Tasks

These options apply to `auxiliary:`, `compression:`, and `fallback_providers:` entries only — `"main"` is **not** a valid value for your top-level `model.provider`. For custom endpoints, use `provider: custom` in your `model:` section (see [AI Providers](/integrations/providers)).

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

## Auxiliary Capacity-Error Fallback

When you set an explicit auxiliary provider (e.g. `auxiliary.vision.provider: glm`), Hermes treats that as your preferred choice — but if the provider literally cannot serve the request because of a **capacity error** (HTTP 402 payment required, HTTP 429 daily-quota exhaustion, connection failure), Hermes falls back through a layered chain instead of failing silently:

1. **Primary aux provider** — the one you configured (tried first, always)
2. **`auxiliary.<task>.fallback_chain`** — your per-task override list, if you wrote one
3. **Main agent provider + model** — last-resort safety net (always tried, even if you didn't write a chain)
4. **Warn + re-raise** — if every layer fails, Hermes logs `Auxiliary <task>: ... all fallbacks exhausted` at WARNING level and re-raises the original error

Transient HTTP 429 rate limits (`Retry-After: ...`) are treated as request constraints, not capacity problems — they respect your explicit provider choice and do **not** trigger the fallback ladder. Only daily/monthly quota exhaustion, payment errors, and connection failures bypass the explicit-provider gate.

For users on `provider: auto` (no explicit aux provider), the existing auto-detection chain runs in place of steps 2–3. Its first step is already the main agent model, so `auto` users get the same outcome with zero config.

### Optional: per-task fallback chain

If you want a different fallback ordering than "main agent model first", configure `fallback_chain` explicitly. Each entry needs at least `provider`; `model`, `base_url`, and `api_key` are optional.

```yaml
auxiliary:
  vision:
    provider: glm
    model: glm-4v-flash
    fallback_chain:
      - provider: openrouter
        model: google/gemini-3-flash-preview
      - provider: nous
        model: anthropic/claude-sonnet-4

  compression:
    provider: openrouter
    fallback_chain:
      - provider: openai
        model: gpt-4o-mini
```

You do **not** need to configure `fallback_chain` to get fallback — the main-agent safety net runs regardless. Use it only when you specifically want a different order than the default.

### Provider quota errors that trigger fallback

Hermes recognizes these as capacity-equivalent to 402 credit exhaustion (not transient rate limits):

- Bedrock / LiteLLM: `Too many tokens per day`, `daily limit`, `tokens per day`
- Vertex AI / GCP: `quota exceeded`, `resource exhausted`, `RESOURCE_EXHAUSTED`
- Generic: `daily quota`, `quota_exceeded`

If your provider returns a different phrase for daily-quota exhaustion and Hermes doesn't trigger fallback, that's a bug — open an issue with the exact error string.

---

## Context Compression Fallback

Context compression uses the `auxiliary.compression` config block to control which model and provider handles summarization:

```yaml
auxiliary:
  compression:
    provider: "auto"                              # auto | openrouter | nous | main
    model: "google/gemini-3-flash-preview"
```

:::info Legacy migration
Older configs with `compression.summary_model` / `compression.summary_provider` / `compression.summary_base_url` are automatically migrated to `auxiliary.compression.*` on first load (config version 17).
:::

If no provider is available for compression, Hermes drops middle conversation turns without generating a summary rather than failing the session.

---

## Delegation Provider Override

Subagents spawned by `delegate_task` inherit the parent agent's primary fallback chain. You can still route subagents to a different primary provider:model pair for cost optimization:

```yaml
delegation:
  provider: "openrouter"                      # override provider for all subagents
  model: "google/gemini-3-flash-preview"      # override model
  # base_url: "http://localhost:1234/v1"      # or use a direct endpoint
  # api_key: "local-key"
```

See [Subagent Delegation](/user-guide/features/delegation) for full configuration details.

---

## Cron Job Providers

Cron jobs inherit your configured `fallback_providers` chain (or legacy `fallback_model`) when they create an agent. To use a different primary provider for a cron job, configure `provider` and `model` overrides on the cron job itself:

```python
cronjob(
    action="create",
    schedule="every 2h",
    prompt="Check server status",
    provider="openrouter",
    model="google/gemini-3-flash-preview"
)
```

See [Scheduled Tasks (Cron)](/user-guide/features/cron) for full configuration details.

---

## Summary

| Feature | Fallback Mechanism | Config Location |
|---------|-------------------|----------------|
| Main agent model | `fallback_providers` in config.yaml — per-turn failover on errors (primary restored each turn) | `fallback_providers:` (top-level list) |
| Auxiliary tasks (any) — auto users | Full auto-detection chain (main agent model first, then provider chain) on capacity errors | `auxiliary.<task>.provider: auto` |
| Auxiliary tasks (any) — explicit provider | `fallback_chain` (if set) → main agent model → warn + raise, on capacity errors only | `auxiliary.<task>.fallback_chain` |
| Vision | Layered (see above) + internal OpenRouter retry | `auxiliary.vision` |
| Web extraction | Layered (see above) + internal OpenRouter retry | `auxiliary.web_extract` |
| Context compression | Layered (see above); degrades to no-summary if all layers unavailable | `auxiliary.compression` |
| Skills hub | Layered (see above) | `auxiliary.skills_hub` |
| MCP helpers | Layered (see above) | `auxiliary.mcp` |
| Approval classification | Layered (see above) | `auxiliary.approval` |
| Title generation | Layered (see above) | `auxiliary.title_generation` |
| Triage specifier | Layered (see above) | `auxiliary.triage_specifier` |
| Delegation | Provider override only (no automatic fallback) | `delegation.provider` / `delegation.model` |
| Cron jobs | Per-job provider override only (no automatic fallback) | Per-job `provider` / `model` |
