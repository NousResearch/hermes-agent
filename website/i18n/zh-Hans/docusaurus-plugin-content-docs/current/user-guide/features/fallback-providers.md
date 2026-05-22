---
title: Fallback Providers
description: 当主模型不可用时，配置自动故障转移到备用 LLM provider。
sidebar_label: Fallback Providers
sidebar_position: 8
---

# Fallback Providers

Hermes Agent 有三层弹性机制，可在 provider 出现问题时保持会话运行：

1. **[Credential pools](./credential-pools.md)** — 为*同一* provider 轮换多个 API key（优先尝试）
2. **Primary model fallback** — 当主模型失败时，自动切换到*不同*的 provider:model
3. **Auxiliary task fallback** — 为 vision、compression 和 web extraction 等辅助任务提供独立的 provider 解析

Credential pools 处理同 provider 轮换（例如多个 OpenRouter key）。本页介绍跨 provider fallback。两者均为可选，且独立工作。

## Primary Model Fallback

当主 LLM provider 遇到错误时 — 如 rate limit、服务器过载、认证失败、连接中断 — Hermes 可以在会话中自动切换到备用 provider:model，而不会丢失对话。

### Configuration

最简单的方式是使用交互式管理器：

```bash
hermes fallback
```

`hermes fallback` 复用了 `hermes model` 的 provider 选择器 — 相同的 provider 列表、相同的凭证提示、相同的验证。使用子命令 `add`、`list`（别名 `ls`）、`remove`（别名 `rm`）和 `clear` 来管理链。更改会持久化保存在 `config.yaml` 顶层的 `fallback_providers:` 列表中。

如果你更喜欢直接编辑 YAML，可以在 `~/.hermes/config.yaml` 中添加 `fallback_model` 部分：

```yaml
fallback_model:
  provider: openrouter
  model: anthropic/claude-sonnet-4
```

`provider` 和 `model` 均为**必填**。如果缺少任一字段，fallback 将被禁用。

:::note `fallback_model` vs `fallback_providers`
`fallback_model`（单数）是遗留的单 fallback 键 — Hermes 仍保留它以兼容旧版本。`fallback_providers`（复数，列表）支持按顺序尝试多个 fallback；`hermes fallback` 写入此键。当两者都设置时，Hermes 会合并它们，且 `fallback_providers` 优先。
:::

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
| NVIDIA NIM | `nvidia` | `NVIDIA_API_KEY` (optional: `NVIDIA_BASE_URL`) |
| GMI Cloud | `gmi` | `GMI_API_KEY` (optional: `GMI_BASE_URL`) |
| StepFun | `stepfun` | `STEPFUN_API_KEY` (optional: `STEPFUN_BASE_URL`) |
| Ollama Cloud | `ollama-cloud` | `OLLAMA_API_KEY` |
| Google Gemini (OAuth) | `google-gemini-cli` | `hermes model` (Google OAuth; optional: `HERMES_GEMINI_PROJECT_ID`) |
| Google AI Studio | `gemini` | `GOOGLE_API_KEY` (alias: `GEMINI_API_KEY`) |
| xAI (Grok) | `xai` (alias `grok`) | `XAI_API_KEY` (optional: `XAI_BASE_URL`) |
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
| Azure AI Foundry | `azure-foundry` | `AZURE_FOUNDRY_API_KEY` + `AZURE_FOUNDRY_BASE_URL` |
| LM Studio (local) | `lmstudio` | `LM_API_KEY` (or none for local) + `LM_BASE_URL` |
| Hugging Face | `huggingface` | `HF_TOKEN` |
| Custom endpoint | `custom` | `base_url` + `key_env` (see below) |

### Custom Endpoint Fallback

对于自定义的 OpenAI-compatible endpoint，添加 `base_url` 和可选的 `key_env`：

```yaml
fallback_model:
  provider: custom
  model: my-local-model
  base_url: http://localhost:8000/v1
  key_env: MY_LOCAL_KEY              # 包含 API key 的环境变量名
```

### When Fallback Triggers

当主模型遇到以下错误时，fallback 会自动激活：

- **Rate limits** (HTTP 429) — 在重试次数耗尽后
- **Server errors** (HTTP 500, 502, 503) — 在重试次数耗尽后
- **Auth failures** (HTTP 401, 403) — 立即触发（无需重试）
- **Not found** (HTTP 404) — 立即触发
- **Invalid responses** — 当 API 反复返回格式错误或空响应时

触发后，Hermes 会：

1. 解析 fallback provider 的凭证
2. 构建新的 API client
3. 原地替换 model、provider 和 client
4. 重置重试计数器并继续对话

切换是无缝的 — 你的对话历史、tool call 和上下文都会被保留。agent 会从断点继续，只是使用了不同的 model。

:::info Per-Turn, Not Per-Session
Fallback 是**按 turn 生效**的：每条新的用户消息都会先恢复使用主模型。如果主模型在 turn 中失败，fallback 仅在该 turn 中激活。在下一条消息时，Hermes 会再次尝试主模型。在单个 turn 内，fallback 最多激活一次 — 如果 fallback 也失败了，则进入正常的错误处理流程（重试，然后报错）。这可以防止在单个 turn 内发生级联故障转移循环，同时让每个 turn 都给予主模型一次新的机会。
:::

### Examples

**OpenRouter 作为 Anthropic native 的 fallback：**
```yaml
model:
  provider: anthropic
  default: claude-sonnet-4-6

fallback_model:
  provider: openrouter
  model: anthropic/claude-sonnet-4
```

**Nous Portal 作为 OpenRouter 的 fallback：**
```yaml
model:
  provider: openrouter
  default: anthropic/claude-opus-4

fallback_model:
  provider: nous
  model: nous-hermes-3
```

**本地模型作为云模型的 fallback：**
```yaml
fallback_model:
  provider: custom
  model: llama-3.1-70b
  base_url: http://localhost:8000/v1
  key_env: LOCAL_API_KEY
```

**Codex OAuth 作为 fallback：**
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
| Subagent delegation | ✘ (subagents 不继承 fallback config) |
| Cron jobs | ✘ (使用固定 provider 运行) |
| Auxiliary tasks (vision, compression) | ✘ (使用它们自己的 provider chain — 见下文) |

:::tip
`fallback_model` 没有环境变量 — 它完全通过 `config.yaml` 配置。这是有意为之：fallback 配置是一个深思熟虑的选择，不应被过时的 shell export 覆盖。
:::

---

## Auxiliary Task Fallback

Hermes 为辅助任务使用独立的轻量级 model。每个任务都有自己的 provider 解析链，充当内置的 fallback 系统。

### Tasks with Independent Provider Resolution

| Task | What It Does | Config Key |
|------|-------------|-----------|
| Vision | Image analysis, browser screenshots | `auxiliary.vision` |
| Web Extract | Web page summarization | `auxiliary.web_extract` |
| Compression | Context compression summaries | `auxiliary.compression` |
| Session Search | Past session summarization | `auxiliary.session_search` |
| Skills Hub | Skill search and discovery | `auxiliary.skills_hub` |
| MCP | MCP helper operations | `auxiliary.mcp` |
| Approval | Smart command-approval classification | `auxiliary.approval` |
| Title Generation | Session title summaries | `auxiliary.title_generation` |
| Triage Specifier | `hermes kanban specify` / dashboard ✨ button — 将一行 triage task 充实为完整 spec | `auxiliary.triage_specifier` |

### Auto-Detection Chain

当任务的 provider 设置为 `"auto"`（默认值）时，Hermes 会按顺序尝试 provider，直到有一个可用：

**对于文本任务（compression、web extract 等）：**

```text
OpenRouter → Nous Portal → Custom endpoint → Codex OAuth →
API-key providers (z.ai, Kimi, MiniMax, Xiaomi MiMo, Hugging Face, Anthropic) → give up
```

**对于 vision 任务：**

```text
Main provider (if vision-capable) → OpenRouter → Nous Portal →
Codex OAuth → Anthropic → Custom endpoint → give up
```

如果解析出的 provider 在调用时失败，Hermes 还会进行内部重试：如果该 provider 不是 OpenRouter 且没有显式设置 `base_url`，它会尝试将 OpenRouter 作为最后的 fallback。

### Configuring Auxiliary Providers

每个任务都可以在 `config.yaml` 中独立配置：

```yaml
auxiliary:
  vision:
    provider: "auto"              # auto | openrouter | nous | codex | main | anthropic
    model: ""                     # e.g. "openai/gpt-4o"
    base_url: ""                  # 直接 endpoint（优先于 provider）
    api_key: ""                   # base_url 的 API key

  web_extract:
    provider: "auto"
    model: ""

  compression:
    provider: "auto"
    model: ""

  session_search:
    provider: "auto"
    model: ""
    timeout: 30
    max_concurrency: 3
    extra_body: {}

  skills_hub:
    provider: "auto"
    model: ""

  mcp:
    provider: "auto"
    model: ""
```

上述每个任务都遵循相同的 **provider / model / base_url** 模式。Context compression 在 `auxiliary.compression` 下配置：

```yaml
auxiliary:
  compression:
    provider: main                                    # 与其他 auxiliary task 相同的 provider 选项
    model: google/gemini-3-flash-preview
    base_url: null                                    # 自定义 OpenAI-compatible endpoint
```

而 fallback model 使用：

```yaml
fallback_model:
  provider: openrouter
  model: anthropic/claude-sonnet-4
  # base_url: http://localhost:8000/v1               # 可选的自定义 endpoint
```

对于 `auxiliary.session_search`，Hermes 还支持：

- `max_concurrency` 限制同时运行的 session summary 数量
- `extra_body` 在 summarization 调用中透传 provider 特定的 OpenAI-compatible 请求字段

Example：

```yaml
auxiliary:
  session_search:
    provider: main
    model: glm-4.5-air
    max_concurrency: 2
    extra_body:
      enable_thinking: false
```

如果你的 provider 不支持原生的 OpenAI-compatible reasoning-control 字段，`extra_body` 对该部分无效；此时 `max_concurrency` 仍然有助于减少请求突发的 429 错误。

以上三者 — auxiliary、compression、fallback — 工作方式相同：设置 `provider` 选择由谁处理请求，`model` 选择使用哪个 model，`base_url` 指向自定义 endpoint（覆盖 provider）。

### Provider Options for Auxiliary Tasks

这些选项仅适用于 `auxiliary:`、`compression:` 和 `fallback_model:` 配置 — `"main"`**不是**顶层 `model.provider` 的有效值。对于自定义 endpoint，在 `model:` 部分使用 `provider: custom`（参见 [AI Providers](/integrations/providers)）。

| Provider | Description | Requirements |
|----------|-------------|-------------|
| `"auto"` | 按顺序尝试 provider，直到有一个可用（默认） | 至少配置了一个 provider |
| `"openrouter"` | 强制使用 OpenRouter | `OPENROUTER_API_KEY` |
| `"nous"` | 强制使用 Nous Portal | `hermes auth` |
| `"codex"` | 强制使用 Codex OAuth | `hermes model` → Codex |
| `"main"` | 使用主 agent 使用的 provider（仅限 auxiliary tasks） | 已配置活跃的主 provider |
| `"anthropic"` | 强制使用 Anthropic native | `ANTHROPIC_API_KEY` or Claude Code credentials |

### Direct Endpoint Override

对于任何 auxiliary task，设置 `base_url` 可以完全绕过 provider 解析，直接将请求发送到该 endpoint：

```yaml
auxiliary:
  vision:
    base_url: "http://localhost:1234/v1"
    api_key: "local-key"
    model: "qwen2.5-vl"
```

`base_url` 优先于 `provider`。Hermes 使用配置的 `api_key` 进行认证，如果未设置则回退到 `OPENAI_API_KEY`。对于自定义 endpoint，它**不会**复用 `OPENROUTER_API_KEY`。

---

## Context Compression Fallback

Context compression 使用 `auxiliary.compression` 配置块来控制哪个 model 和 provider 处理 summarization：

```yaml
auxiliary:
  compression:
    provider: "auto"                              # auto | openrouter | nous | main
    model: "google/gemini-3-flash-preview"
```

:::info Legacy migration
带有 `compression.summary_model` / `compression.summary_provider` / `compression.summary_base_url` 的旧配置会在首次加载时自动迁移到 `auxiliary.compression.*`（config version 17）。
:::

如果没有可用的 provider 进行 compression，Hermes 会丢弃中间的对话 turn 而不生成 summary，而不是导致会话失败。

---

## Delegation Provider Override

由 `delegate_task` 生成的 subagents **不会**使用主 fallback model。但是，它们可以被路由到不同的 provider:model 对以优化成本：

```yaml
delegation:
  provider: "openrouter"                      # 为所有 subagents 覆盖 provider
  model: "google/gemini-3-flash-preview"      # 覆盖 model
  # base_url: "http://localhost:1234/v1"      # 或使用直接 endpoint
  # api_key: "local-key"
```

完整的配置详情参见 [Subagent Delegation](/user-guide/features/delegation)。

---

## Cron Job Providers

Cron jobs 使用执行时配置的 provider 运行。它们不支持 fallback model。要为 cron jobs 使用不同的 provider，在 cron job 本身上配置 `provider` 和 `model` 覆盖：

```python
cronjob(
    action="create",
    schedule="every 2h",
    prompt="Check server status",
    provider="openrouter",
    model="google/gemini-3-flash-preview"
)
```

完整的配置详情参见 [Scheduled Tasks (Cron)](/user-guide/features/cron)。

---

## Summary

| Feature | Fallback Mechanism | Config Location |
|---------|-------------------|----------------|
| Main agent model | `fallback_model` in config.yaml — 出错时按 turn 故障转移（每个 turn 恢复主模型） | `fallback_model:` (top-level) |
| Vision | Auto-detection chain + 内部 OpenRouter 重试 | `auxiliary.vision` |
| Web extraction | Auto-detection chain + 内部 OpenRouter 重试 | `auxiliary.web_extract` |
| Context compression | Auto-detection chain，如果不可用则降级为不生成 summary | `auxiliary.compression` |
| Session search | Auto-detection chain | `auxiliary.session_search` |
| Skills hub | Auto-detection chain | `auxiliary.skills_hub` |
| MCP helpers | Auto-detection chain | `auxiliary.mcp` |
| Approval classification | Auto-detection chain | `auxiliary.approval` |
| Title generation | Auto-detection chain | `auxiliary.title_generation` |
| Triage specifier | Auto-detection chain | `auxiliary.triage_specifier` |
| Delegation | 仅 provider override（无自动 fallback） | `delegation.provider` / `delegation.model` |
| Cron jobs | 仅按 job 的 provider override（无自动 fallback） | Per-job `provider` / `model` |
