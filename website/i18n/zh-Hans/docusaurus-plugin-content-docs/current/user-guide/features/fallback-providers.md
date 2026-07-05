---
title: 备用提供商
description: 配置自动故障转移，在主模型不可用时切换到备用 LLM 提供商。
sidebar_label: 备用提供商
sidebar_position: 8
---

# 备用提供商

Hermes Agent 具备三层弹性机制，在提供商出现问题时保持会话正常运行：

1. **[凭据池](./credential-pools.md)** — 在*同一*提供商的多个 API 密钥之间轮换（优先尝试）
2. **主模型备用** — 当主模型失败时，自动切换到*不同*的提供商:模型
3. **辅助任务备用** — 针对视觉、压缩、网页提取等附属任务的独立提供商解析

凭据池处理同一提供商内的轮换（例如多个 OpenRouter 密钥）。本页介绍跨提供商的备用机制。两者均为可选，且相互独立。

## 主模型备用

当主 LLM 提供商遇到错误——速率限制、服务器过载、认证失败、连接中断——Hermes 可以在会话中途自动切换到备用提供商:模型对，且不会丢失对话内容。

### 配置

最简便的方式是使用交互式管理器：

```bash
hermes fallback
```

`hermes fallback` 复用 `hermes model` 的提供商选择器——相同的提供商列表、相同的凭据提示、相同的验证流程。使用子命令 `add`、`list`（别名 `ls`）、`remove`（别名 `rm`）和 `clear` 来管理备用链。更改会持久化到 `config.yaml` 顶层的 `fallback_providers:` 列表中。

如果你更倾向于直接编辑 YAML，可在 `~/.hermes/config.yaml` 中添加 `fallback_model` 部分：

```yaml
fallback_model:
  provider: openrouter
  model: anthropic/claude-sonnet-4
```

`provider` 和 `model` 均为**必填项**。若任一缺失，备用功能将被禁用。

:::note `fallback_model` 与 `fallback_providers`
`fallback_model`（单数）是旧版单备用键——Hermes 仍支持以保持向后兼容。`fallback_providers`（复数，列表）支持按顺序尝试多个备用；`hermes fallback` 写入此键。当两者同时设置时，Hermes 会合并它们，`fallback_providers` 优先。
:::

### 支持的提供商

| 提供商 | 值 | 要求 |
|----------|-------|-------------|
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| Nous Portal | `nous` | `hermes setup --portal`（全新安装）或 `hermes auth add nous`（OAuth） |
| OpenAI Codex | `openai-codex` | `hermes model`（ChatGPT OAuth） |
| GitHub Copilot | `copilot` | `COPILOT_GITHUB_TOKEN`、`GH_TOKEN` 或 `GITHUB_TOKEN` |
| GitHub Copilot ACP | `copilot-acp` | 外部进程（编辑器集成） |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` 或 Claude Code 凭据 |
| z.ai / GLM | `zai` | `GLM_API_KEY` |
| Kimi / Moonshot | `kimi-coding` | `KIMI_API_KEY` |
| MiniMax | `minimax` | `MINIMAX_API_KEY` |
| MiniMax（中国）| `minimax-cn` | `MINIMAX_CN_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| NVIDIA NIM | `nvidia` | `NVIDIA_API_KEY`（可选：`NVIDIA_BASE_URL`） |
| GMI Cloud | `gmi` | `GMI_API_KEY`（可选：`GMI_BASE_URL`） |
| StepFun | `stepfun` | `STEPFUN_API_KEY`（可选：`STEPFUN_BASE_URL`） |
| Ollama Cloud | `ollama-cloud` | `OLLAMA_API_KEY` |
| Google AI Studio | `gemini` | `GOOGLE_API_KEY`（别名：`GEMINI_API_KEY`） |
| xAI（Grok） | `xai`（别名 `grok`） | `XAI_API_KEY`（可选：`XAI_BASE_URL`） |
| xAI Grok OAuth（SuperGrok） | `xai-oauth`（别名 `grok-oauth`） | `hermes model` → xAI Grok OAuth（浏览器登录；需 SuperGrok 订阅） |
| AWS Bedrock | `bedrock` | 标准 boto3 认证（`AWS_REGION` + `AWS_PROFILE` 或 `AWS_ACCESS_KEY_ID`） |
| Qwen Portal（OAuth） | `qwen-oauth` | `hermes model`（Qwen Portal OAuth；可选：`HERMES_QWEN_BASE_URL`） |
| MiniMax（OAuth） | `minimax-oauth` | `hermes model`（MiniMax 门户 OAuth） |
| OpenCode Zen | `opencode-zen` | `OPENCODE_ZEN_API_KEY` |
| OpenCode Go | `opencode-go` | `OPENCODE_GO_API_KEY` |
| Kilo Code | `kilocode` | `KILOCODE_API_KEY` |
| Xiaomi MiMo | `xiaomi` | `XIAOMI_API_KEY` |
| Arcee AI | `arcee` | `ARCEEAI_API_KEY` |
| GMI Cloud | `gmi` | `GMI_API_KEY` |
| Alibaba / DashScope | `alibaba` | `DASHSCOPE_API_KEY` |
| Alibaba Coding Plan | `alibaba-coding-plan` | `ALIBABA_CODING_PLAN_API_KEY`（回退到 `DASHSCOPE_API_KEY`） |
| Kimi / Moonshot（中国） | `kimi-coding-cn` | `KIMI_CN_API_KEY` |
| StepFun | `stepfun` | `STEPFUN_API_KEY` |
| Tencent TokenHub | `tencent-tokenhub` | `TOKENHUB_API_KEY` |
| Microsoft Foundry | `azure-foundry` | `AZURE_FOUNDRY_API_KEY` + `AZURE_FOUNDRY_BASE_URL` |
| LM Studio（本地） | `lmstudio` | `LM_API_KEY`（本地可不填）+ `LM_BASE_URL` |
| Hugging Face | `huggingface` | `HF_TOKEN` |
| 自定义端点 | `custom` | `base_url` + `key_env`（见下文） |

### 自定义端点备用

对于兼容 OpenAI 的自定义端点，添加 `base_url` 并可选填 `key_env`：

```yaml
fallback_model:
  provider: custom
  model: my-local-model
  base_url: http://localhost:8000/v1
  key_env: MY_LOCAL_KEY              # 包含 API 密钥的环境变量名
```

### 备用触发条件

当主模型出现以下失败时，备用机制自动激活：

- **速率限制**（HTTP 429）——耗尽重试次数后
- **服务器错误**（HTTP 500、502、503）——耗尽重试次数后
- **认证失败**（HTTP 401、403）——立即触发（重试无意义）
- **未找到**（HTTP 404）——立即触发
- **无效响应**——API 多次返回格式错误或空响应时

触发后，Hermes 将：

1. 解析备用提供商的凭据
2. 构建新的 API 客户端
3. 就地替换模型、提供商和客户端
4. 重置重试计数器并继续对话

切换是无感知的——对话历史、工具调用和上下文均被保留。Agent 从中断处继续，只是使用了不同的模型。

:::info 按轮次，而非按会话
备用机制的**作用域为单次轮次**：每条新用户消息都从主模型重新开始。若主模型在某轮次中途失败，备用仅对该轮次生效。下一条消息时，Hermes 会再次尝试主模型。在单次轮次内，备用最多激活一次——若备用也失败，则进入常规错误处理流程（重试，然后返回错误消息）。这既防止了单轮次内的级联故障转移循环，又让主模型在每轮次都有重新尝试的机会。
:::

### 示例

**以 OpenRouter 作为 Anthropic 原生的备用：**
```yaml
model:
  provider: anthropic
  default: claude-sonnet-4-6

fallback_model:
  provider: openrouter
  model: anthropic/claude-sonnet-4
```

**以 Nous Portal 作为 OpenRouter 的备用：**
```yaml
model:
  provider: openrouter
  default: anthropic/claude-opus-4

fallback_model:
  provider: nous
  model: nous-hermes-3
```

**以本地模型作为云端的备用：**
```yaml
fallback_model:
  provider: custom
  model: llama-3.1-70b
  base_url: http://localhost:8000/v1
  key_env: LOCAL_API_KEY
```

**以 Codex OAuth 作为备用：**
```yaml
fallback_model:
  provider: openai-codex
  model: gpt-5.3-codex
```

### 备用适用范围

| 场景 | 是否支持备用 |
|---------|-------------------|
| CLI 会话 | ✔ |
| 消息网关（Telegram、Discord 等） | ✔ |
| 子 Agent 委派 | ✔（子 Agent 继承父 Agent 的备用链） |
| Cron 任务 | ✔（Cron Agent 继承配置的备用提供商） |
| 辅助任务（视觉、压缩等） | ✘（使用各自的提供商链——见下文） |

:::tip
没有针对主备用链的环境变量——只能通过 `config.yaml` 或 `hermes fallback` 进行配置。这是有意为之：备用配置是一个经过深思熟虑的选择，不应被过期的 shell 导出变量覆盖。
:::

---

## 辅助任务备用

Hermes 为附属任务使用独立的轻量级模型。每个任务都有自己的提供商解析链，充当内置的备用系统。

### 具有独立提供商解析的任务

| 任务 | 功能说明 | 配置键 |
|------|-------------|-----------|
| 视觉 | 图像分析、浏览器截图 | `auxiliary.vision` |
| 网页提取 | 网页内容摘要 | `auxiliary.web_extract` |
| 压缩 | 上下文压缩摘要 | `auxiliary.compression` |
| Skills Hub | 技能搜索与发现 | `auxiliary.skills_hub` |
| MCP | MCP 辅助操作 | `auxiliary.mcp` |
| 审批 | 智能命令审批分类 | `auxiliary.approval` |
| 标题生成 | 会话标题摘要 | `auxiliary.title_generation` |
| Triage Specifier | `hermes kanban specify` / 看板（kanban）✨ 按钮——将单行 triage 任务扩展为完整规格 | `auxiliary.triage_specifier` |

### 自动检测链

当任务的提供商设置为 `"auto"`（默认值）时，Hermes 按顺序尝试各提供商，直到找到可用的：

**文本任务（压缩、网页提取等）：**

```text
OpenRouter → Nous Portal → 自定义端点 → Codex OAuth →
API 密钥提供商（z.ai、Kimi、MiniMax、Xiaomi MiMo、Hugging Face、Anthropic）→ 放弃
```

**视觉任务：**

```text
主提供商（若支持视觉）→ OpenRouter → Nous Portal →
Codex OAuth → Anthropic → 自定义端点 → 放弃
```

若解析到的提供商在调用时失败，Hermes 还有内部重试机制：若该提供商不是 OpenRouter 且未设置显式 `base_url`，则尝试以 OpenRouter 作为最后备用。

### 配置辅助提供商

每个任务可在 `config.yaml` 中独立配置：

```yaml
auxiliary:
  vision:
    provider: "auto"              # auto | openrouter | nous | codex | main | anthropic
    model: ""                     # 例如 "openai/gpt-4o"
    base_url: ""                  # 直接端点（优先于 provider）
    api_key: ""                   # base_url 的 API 密钥

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

以上每个任务均遵循相同的 **provider / model / base_url** 模式。上下文压缩在 `auxiliary.compression` 下配置：

```yaml
auxiliary:
  compression:
    provider: main                                    # 与其他辅助任务相同的提供商选项
    model: google/gemini-3-flash-preview
    base_url: null                                    # 自定义 OpenAI 兼容端点
```

备用模型使用：

```yaml
fallback_model:
  provider: openrouter
  model: anthropic/claude-sonnet-4
  # base_url: http://localhost:8000/v1               # 可选自定义端点
```

三者——辅助任务、压缩、备用——工作方式相同：设置 `provider` 指定处理请求的提供商，`model` 指定使用的模型，`base_url` 指向自定义端点（会覆盖 provider）。

### 辅助任务的提供商选项

以下选项仅适用于 `auxiliary:`、`compression:` 和 `fallback_model:` 配置——`"main"` **不是**顶层 `model.provider` 的有效值。对于自定义端点，请在 `model:` 部分使用 `provider: custom`（参见 [AI 提供商](/integrations/providers)）。

| 提供商 | 说明 | 要求 |
|----------|-------------|-------------|
| `"auto"` | 按顺序尝试各提供商直到找到可用的（默认） | 至少配置一个提供商 |
| `"openrouter"` | 强制使用 OpenRouter | `OPENROUTER_API_KEY` |
| `"nous"` | 强制使用 Nous Portal | `hermes auth` |
| `"codex"` | 强制使用 Codex OAuth | `hermes model` → Codex |
| `"main"` | 使用主 Agent 当前的提供商（仅限辅助任务） | 已配置活跃的主提供商 |
| `"anthropic"` | 强制使用 Anthropic 原生 | `ANTHROPIC_API_KEY` 或 Claude Code 凭据 |

### 直接端点覆盖

对于任意辅助任务，设置 `base_url` 将完全绕过提供商解析，直接向该端点发送请求：

```yaml
auxiliary:
  vision:
    base_url: "http://localhost:1234/v1"
    api_key: "local-key"
    model: "qwen2.5-vl"
```

`base_url` 优先于 `provider`。Hermes 使用配置的 `api_key` 进行认证，若未设置则回退到 `OPENAI_API_KEY`。对于自定义端点，**不会**复用 `OPENROUTER_API_KEY`。

---

## 辅助任务容量错误备用

当你设置了显式的辅助提供商（例如 `auxiliary.vision.provider: glm`）时，Hermes 将其视为首选——但若该提供商因**容量错误**（HTTP 402 付款要求、HTTP 429 每日配额耗尽、连接失败）而无法处理请求，Hermes 会通过分层链进行备用，而不是静默失败：

1. **主辅助提供商** — 你配置的那个（始终优先尝试）
2. **`auxiliary.<task>.fallback_chain`** — 你的每任务覆盖列表（若已配置）
3. **主 Agent 提供商 + 模型** — 最后的安全网（始终尝试，即使未配置链）
4. **警告 + 重新抛出** — 若所有层均失败，Hermes 以 WARNING 级别记录 `Auxiliary <task>: ... all fallbacks exhausted` 并重新抛出原始错误

瞬时 HTTP 429 速率限制（`Retry-After: ...`）被视为请求约束，而非容量问题——它们遵守你的显式提供商选择，**不会**触发备用链。只有每日/每月配额耗尽、付款错误和连接失败才会绕过显式提供商限制。

对于使用 `provider: auto`（无显式辅助提供商）的用户，现有的自动检测链将替代步骤 2–3 运行。其第一步已经是主 Agent 模型，因此 `auto` 用户无需任何配置即可获得相同效果。

### 可选：每任务备用链

若你希望使用与"主 Agent 模型优先"不同的备用顺序，可显式配置 `fallback_chain`。每个条目至少需要 `provider`；`model`、`base_url` 和 `api_key` 为可选。

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

你**不需要**配置 `fallback_chain` 才能获得备用功能——主 Agent 安全网无论如何都会运行。仅当你明确希望使用与默认不同的顺序时才需配置。

### 触发备用的提供商配额错误

Hermes 将以下情况识别为等同于 402 额度耗尽的容量错误（而非瞬时速率限制）：

- Bedrock / LiteLLM：`Too many tokens per day`、`daily limit`、`tokens per day`
- Vertex AI / GCP：`quota exceeded`、`resource exhausted`、`RESOURCE_EXHAUSTED`
- 通用：`daily quota`、`quota_exceeded`

若你的提供商对每日配额耗尽返回不同的错误信息，而 Hermes 未触发备用，这是一个 bug——请附上确切的错误字符串提交 issue。

---

## 上下文压缩备用

上下文压缩使用 `auxiliary.compression` 配置块来控制处理摘要的模型和提供商：

```yaml
auxiliary:
  compression:
    provider: "auto"                              # auto | openrouter | nous | main
    model: "google/gemini-3-flash-preview"
```

:::info 旧版迁移
旧版配置中的 `compression.summary_model` / `compression.summary_provider` / `compression.summary_base_url` 会在首次加载时自动迁移到 `auxiliary.compression.*`（配置版本 17）。
:::

若压缩没有可用的提供商，Hermes 会直接丢弃中间对话轮次而不生成摘要，而不是让会话失败。

---

## 委派提供商覆盖

由 `delegate_task` 生成的子 Agent 会继承父 Agent 的主备用链。你仍然可以将子 Agent 路由到不同的主提供商:模型对以进行成本优化：

```yaml
delegation:
  provider: "openrouter"                      # 覆盖所有子 Agent 的提供商
  model: "google/gemini-3-flash-preview"      # 覆盖模型
  # base_url: "http://localhost:1234/v1"      # 或使用直接端点
  # api_key: "local-key"
```

完整配置详情参见[子 Agent 委派](/user-guide/features/delegation)。

---

## Cron 任务提供商

Cron 任务在创建 Agent 时会继承你配置的 `fallback_providers` 链（或旧版 `fallback_model`）。要为 Cron 任务使用不同的主提供商，请在 Cron 任务本身配置 `provider` 和 `model` 覆盖：

```python
cronjob(
    action="create",
    schedule="every 2h",
    prompt="Check server status",
    provider="openrouter",
    model="google/gemini-3-flash-preview"
)
```

完整配置详情参见[定时任务（Cron）](/user-guide/features/cron)。

---

## 总结

| 功能 | 备用机制 | 配置位置 |
|---------|-------------------|----------------|
| 主 Agent 模型 | `fallback_providers`（config.yaml 中）——出错时按轮次故障转移（每轮次恢复主模型） | `fallback_providers:`（顶层列表） |
| 辅助任务（任意）— auto 用户 | 容量错误时完整自动检测链（主 Agent 模型优先，然后提供商链） | `auxiliary.<task>.provider: auto` |
| 辅助任务（任意）— 显式提供商 | `fallback_chain`（若已设置）→ 主 Agent 模型 → 警告 + 抛出，仅在容量错误时触发 | `auxiliary.<task>.fallback_chain` |
| 视觉 | 分层（见上文）+ 内部 OpenRouter 重试 | `auxiliary.vision` |
| 网页提取 | 分层（见上文）+ 内部 OpenRouter 重试 | `auxiliary.web_extract` |
| 上下文压缩 | 分层（见上文）；所有层不可用时降级为无摘要 | `auxiliary.compression` |
| Skills Hub | 分层（见上文） | `auxiliary.skills_hub` |
| MCP 辅助 | 分层（见上文） | `auxiliary.mcp` |
| 审批分类 | 分层（见上文） | `auxiliary.approval` |
| 标题生成 | 分层（见上文） | `auxiliary.title_generation` |
| Triage Specifier | 分层（见上文） | `auxiliary.triage_specifier` |
| 委派 | 仅提供商覆盖（无自动备用） | `delegation.provider` / `delegation.model` |
| Cron 任务 | 仅每任务提供商覆盖（无自动备用） | 每任务 `provider` / `model` |