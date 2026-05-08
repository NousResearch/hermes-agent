---
title: 备用提供商
description: "配置当主 LLM 提供商不可用时自动故障转移到备用 LLM 提供商。"
sidebar_label: 备用提供商
sidebar_position: 8
---

# 备用提供商

Hermes Agent 有三层弹性机制，在提供商出现问题时保持会话运行：

1. **[凭证池](./credential-pools.md)** — 在*同一*提供商中轮换多个 API 密钥（最先尝试）
2. **主模型回退** — 当主模型失败时自动切换到*不同*的 provider:model 对
3. **辅助任务回退** — 用于视觉、压缩和网页提取等辅助任务的独立提供商解析

凭证池处理同提供商轮换（例如多个 OpenRouter 密钥）。本页涵盖跨提供商回退。两者都是可选的，独立工作。

## 主模型回退

当您的 LLM 主提供商遇到错误——限流、服务器过载、认证失败、连接断开——Hermes 可以在不丢失对话的情况下自动切换到备用 provider:model 对。

### 配置

最简单的方式是交互式管理器：

```bash
hermes fallback
```

`hermes fallback` 重用 `hermes model` 的提供商选择器——相同的提供商列表、相同的凭证提示、相同的验证。按 `a` 添加备用，`↑`/`↓` 重新排序，`d` 删除，`q` 保存并退出。变更持久化到 `config.yaml` 的 `model.fallback_providers` 下。

如果您更喜欢直接编辑 YAML，在 `~/.hermes/config.yaml` 中添加 `fallback_model` 部分：

```yaml
fallback_model:
  provider: openrouter
  model: anthropic/claude-sonnet-4
```

`provider` 和 `model` 都是**必需的**。如果任一缺失，备用被禁用。

:::note `fallback_model` vs `fallback_providers`
`fallback_model`（单数）是旧的单一回退键——Hermes 仍为向后兼容而遵循它。`fallback_providers`（复数，列表）支持多个按顺序尝试的回退；`hermes fallback` 写入此键。当两者都设置时，Hermes 合并它们，`fallback_providers` 优先。
:::

### 支持的提供商

| 提供商 | 值 | 要求 |
|----------|-------|-------------|
| AI Gateway | `ai-gateway` | `AI_GATEWAY_API_KEY` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| Nous Portal | `nous` | `hermes auth`（OAuth） |
| OpenAI Codex | `openai-codex` | `hermes model`（ChatGPT OAuth） |
| GitHub Copilot | `copilot` | `COPILOT_GITHUB_TOKEN`、`GH_TOKEN` 或 `GITHUB_TOKEN` |
| GitHub Copilot ACP | `copilot-acp` | 外部进程（编辑器集成） |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` 或 Claude Code 凭证 |
| z.ai / GLM | `zai` | `GLM_API_KEY` |
| Kimi / Moonshot | `kimi-coding` | `KIMI_API_KEY` |
| MiniMax | `minimax` | `MINIMAX_API_KEY` |
| MiniMax（中国） | `minimax-cn` | `MINIMAX_CN_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| NVIDIA NIM | `nvidia` | `NVIDIA_API_KEY`（可选：`NVIDIA_BASE_URL`） |
| GMI Cloud | `gmi` | `GMI_API_KEY`（可选：`GMI_BASE_URL`） |
| StepFun | `stepfun` | `STEPFUN_API_KEY`（可选：`STEPFUN_BASE_URL`） |
| Ollama Cloud | `ollama-cloud` | `OLLAMA_API_KEY` |
| Google Gemini（OAuth） | `google-gemini-cli` | `hermes model`（Google OAuth；可选：`HERMES_GEMINI_PROJECT_ID`） |
| Google AI Studio | `gemini` | `GOOGLE_API_KEY`（别名：`GEMINI_API_KEY`） |
| xAI（Grok） | `xai`（别名 `grok`） | `XAI_API_KEY`（可选：`XAI_BASE_URL`） |
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
| 腾讯 TokenHub | `tencent-tokenhub` | `TOKENHUB_API_KEY` |
| Azure AI Foundry | `azure-foundry` | `AZURE_FOUNDRY_API_KEY` + `AZURE_FOUNDRY_BASE_URL` |
| LM Studio（本地） | `lmstudio` | `LM_API_KEY`（或本地无需）+ `LM_BASE_URL` |
| Hugging Face | `huggingface` | `HF_TOKEN` |
| 自定义端点 | `custom` | `base_url` + `key_env`（见下文） |

### 自定义端点备用

对于自定义 OpenAI 兼容端点，添加 `base_url` 和可选的 `key_env`：

```yaml
fallback_model:
  provider: custom
  model: my-local-model
  base_url: http://localhost:8000/v1
  key_env: MY_LOCAL_KEY              # 包含 API 密钥的环境变量名
```

### 回退触发时机

当主模型失败时自动激活备用：

- **限流**（HTTP 429）——重试耗尽后
- **服务器错误**（HTTP 500、502、503）——重试耗尽后
- **认证失败**（HTTP 401、403）——立即（重试无意义）
- **未找到**（HTTP 404）——立即
- **无效响应**——API 重复返回格式错误或空响应时

触发时，Hermes：

1. 为备用提供商解析凭证
2. 构建新的 API 客户端
3. 原地交换模型、提供商和客户端
4. 重置重试计数器并继续对话

切换是无缝的——您的对话历史、工具调用和上下文都被保留。Agent 从中断处继续，只是使用不同的模型。

:::info 逐轮，而非逐会话
回退是**逐轮作用域**：每个新用户消息以恢复的主模型开始。如果主模型在轮次中途失败，该轮次仅激活备用。下一条消息时，Hermes 再次尝试主模型。在单轮内，备用最多激活一次——如果备用也失败，正常错误处理接管（重试，然后错误消息）。这防止轮内级联故障转移循环，同时让主模型在每轮都有新机会。
:::

### 示例

**OpenRouter 作为 Anthropic 原生的备用：**
```yaml
model:
  provider: anthropic
  default: claude-sonnet-4-6

fallback_model:
  provider: openrouter
  model: anthropic/claude-sonnet-4
```

**Nous Portal 作为 OpenRouter 的备用：**
```yaml
model:
  provider: openrouter
  default: anthropic/claude-opus-4

fallback_model:
  provider: nous
  model: nous-hermes-3
```

**本地模型作为云端备用：**
```yaml
fallback_model:
  provider: custom
  model: llama-3.1-70b
  base_url: http://localhost:8000/v1
  key_env: LOCAL_API_KEY
```

**Codex OAuth 作为备用：**
```yaml
fallback_model:
  provider: openai-codex
  model: gpt-5.3-codex
```

### 回退适用范围

| 场景 | 支持回退 |
|---------|-------------------|
| CLI 会话 | ✔ |
| 消息网关（Telegram、Discord 等） | ✔ |
| 子代理委托 | ✘（子代理不继承备用配置） |
| Cron 任务 | ✘（使用固定提供商运行） |
| 辅助任务（视觉、压缩） | ✘（使用自己的提供商链——见下文） |

:::tip
没有 `fallback_model` 的环境变量——它仅通过 `config.yaml` 配置。这是故意的：备用配置是一个深思熟虑的选择，不应该被过时的 shell 导出覆盖。
:::

---

## 辅助任务备用

Hermes 对辅助任务使用单独的轻量级模型。每个任务有自己的提供商解析链，充当内置备用系统。

### 具有独立提供商解析的任务

| 任务 | 功能 | 配置键 |
|------|-------------|-----------|
| Vision | 图像分析、浏览器截图 | `auxiliary.vision` |
| Web Extract | 网页摘要 | `auxiliary.web_extract` |
| Compression | 上下文压缩摘要 | `auxiliary.compression` |
| Session Search | 过去会话摘要 | `auxiliary.session_search` |
| Skills Hub | 技能搜索和发现 | `auxiliary.skills_hub` |
| MCP | MCP 辅助操作 | `auxiliary.mcp` |
| Approval | 智能命令批准分类 | `auxiliary.approval` |
| Title Generation | 会话标题摘要 | `auxiliary.title_generation` |
| Triage Specifier | `hermes kanban specify` / dashboard ✨ 按钮——将分类待办项扩展为真实规格 | `auxiliary.triage_specifier` |

### 自动检测链

当任务的提供商设置为 `"auto"`（默认值）时，Hermes 按顺序尝试提供商直到成功：

**文本任务（压缩、网页提取等）：**

```text
OpenRouter → Nous Portal → 自定义端点 → Codex OAuth →
API 密钥提供商（z.ai、Kimi、MiniMax、Xiaomi MiMo、Hugging Face、Anthropic） → 放弃
```

**视觉任务：**

```text
主提供商（如果支持视觉） → OpenRouter → Nous Portal →
Codex OAuth → Anthropic → 自定义端点 → 放弃
```

如果解析的提供商在调用时失败，Hermes 也有内部重试：如果提供商不是 OpenRouter 且未设置显式 `base_url`，它会尝试 OpenRouter 作为最后手段备用。

### 配置辅助提供商

每个任务可以在 `config.yaml` 中独立配置：

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

每个上述任务遵循相同的 **provider / model / base_url** 模式。上下文压缩在 `auxiliary.compression` 下配置：

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

对于 `auxiliary.session_search`，Hermes 还支持：

- `max_concurrency` 限制同时运行的会话摘要数
- `extra_body` 传递提供商特定的 OpenAI 兼容请求字段到摘要调用

示例：

```yaml
auxiliary:
  session_search:
    provider: main
    model: glm-4.5-air
    max_concurrency: 2
    extra_body:
      enable_thinking: false
```

如果您的提供商不支持原生 OpenAI 兼容的推理控制字段，`extra_body` 对那部分没有帮助；这种情况下 `max_concurrency` 仍有助于减少请求突发 429。

所有三个——辅助、压缩、备用——以相同方式工作：设置 `provider` 选择谁处理请求，`model` 选择哪个模型，`base_url` 指向自定义端点（覆盖 provider）。

### 辅助任务的提供商选项

这些选项仅适用于 `auxiliary:`、`compression:` 和 `fallback_model:` 配置——`"main"` 对顶级 `model.provider` **不是有效值**。对于自定义端点，在 `model:` 部分使用 `provider: custom`（参见 [AI 提供商](/docs/integrations/providers)）。

| 提供商 | 描述 | 要求 |
|----------|-------------|-------------|
| `"auto"` | 按顺序尝试提供商直到成功（默认） | 至少配置一个提供商 |
| `"openrouter"` | 强制使用 OpenRouter | `OPENROUTER_API_KEY` |
| `"nous"` | 强制使用 Nous Portal | `hermes auth` |
| `"codex"` | 强制使用 Codex OAuth | `hermes model` → Codex |
| `"main"` | 使用主 agent 使用的提供商（仅辅助任务） | 配置了活动主提供商 |
| `"anthropic"` | 强制使用 Anthropic 原生 | `ANTHROPIC_API_KEY` 或 Claude Code 凭证 |

### 直接端点覆盖

对于任何辅助任务，设置 `base_url` 完全绕过提供商解析，直接发送请求到该端点：

```yaml
auxiliary:
  vision:
    base_url: "http://localhost:1234/v1"
    api_key: "local-key"
    model: "qwen2.5-vl"
```

`base_url` 优先于 `provider`。Hermes 使用配置的 `api_key` 进行认证，如果未设置则回退到 `OPENAI_API_KEY`。它**不会**为自定义端点重用 `OPENROUTER_API_KEY`。

---

## 上下文压缩备用

上下文压缩使用 `auxiliary.compression` 配置块控制哪个模型和提供商处理摘要：

```yaml
auxiliary:
  compression:
    provider: "auto"                              # auto | openrouter | nous | main
    model: "google/gemini-3-flash-preview"
```

:::info 旧配置迁移
旧配置中的 `compression.summary_model` / `compression.summary_provider` / `compression.summary_base_url` 在首次加载时自动迁移到 `auxiliary.compression.*`（配置版本 17）。
:::

如果没有可用的压缩提供商，Hermes 在不生成摘要的情况下丢弃中间对话轮次，而不是让会话失败。

---

## 委托提供商覆盖

通过 `delegate_task` 生成的子代理**不使用**主备用模型。但是，它们可以路由到不同的 provider:model 对以优化成本：

```yaml
delegation:
  provider: "openrouter"                      # 覆盖所有子代理的提供商
  model: "google/gemini-3-flash-preview"      # 覆盖模型
  # base_url: "http://localhost:1234/v1"      # 或使用直接端点
  # api_key: "local-key"
```

参见[子代理委托](/docs/user-guide/features/delegation)获取完整配置详情。

---

## Cron 任务提供商

Cron 任务使用执行时配置的提供商运行。它们不支持备用模型。要为 cron 任务使用不同提供商，请在 cron 任务本身上配置 `provider` 和 `model` 覆盖：

```python
cronjob(
    action="create",
    schedule="every 2h",
    prompt="Check server status",
    provider="openrouter",
    model="google/gemini-3-flash-preview"
)
```

参见[定时任务（Cron）](/docs/user-guide/features/cron)获取完整配置详情。

---

## 总结

| 功能 | 备用机制 | 配置位置 |
|---------|-------------------|----------------|
| 主 agent 模型 | `fallback_model` 在 config.yaml 中——错误时逐轮故障转移（每轮恢复主模型） | `fallback_model:`（顶级） |
| Vision | 自动检测链 + 内部 OpenRouter 重试 | `auxiliary.vision` |
| 网页提取 | 自动检测链 + 内部 OpenRouter 重试 | `auxiliary.web_extract` |
| 上下文压缩 | 自动检测链，不可用意时降级为无摘要 | `auxiliary.compression` |
| 会话搜索 | 自动检测链 | `auxiliary.session_search` |
| Skills hub | 自动检测链 | `auxiliary.skills_hub` |
| MCP 辅助 | 自动检测链 | `auxiliary.mcp` |
| 批准分类 | 自动检测链 | `auxiliary.approval` |
| 标题生成 | 自动检测链 | `auxiliary.title_generation` |
| 分类规格器 | 自动检测链 | `auxiliary.triage_specifier` |
| 委托 | 仅提供商覆盖（无自动备用） | `delegation.provider` / `delegation.model` |
| Cron 任务 | 仅按任务提供商覆盖（无自动备用） | 按任务 `provider` / `model` |
