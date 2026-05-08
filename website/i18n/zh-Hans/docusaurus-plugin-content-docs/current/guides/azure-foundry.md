---
sidebar_position: 15
title: "Azure AI Foundry"
description: "使用 Hermes Agent 与 Azure AI Foundry — OpenAI 风格和 Anthropic 风格的端点、传输和已部署模型的自动检测"
---

# Azure AI Foundry

Hermes Agent 支持 Azure AI Foundry（和 Azure OpenAI）作为一流提供商。单个 Azure 资源可以托管具有两种不同传输格式（wire formats）的模型：

- **OpenAI-style** — 在类似 `https://<resource>.openai.azure.com/openai/v1` 的端点上使用 `POST /v1/chat/completions`。用于 GPT-4.x、GPT-5.x、Llama、Mistral 和大多数开放权重模型。
- **Anthropic-style** — 在类似 `https://<resource>.services.ai.azure.com/anthropic` 的端点上使用 `POST /v1/messages`。当 Azure Foundry 通过 Anthropic Messages API 格式提供 Claude 模型时使用。

设置向导会探测你的端点并自动检测它使用哪种传输、哪些部署可用以及每个模型的上下文长度。

## 前置条件

- 具有至少一个部署的 Azure AI Foundry 或 Azure OpenAI 资源
- 该资源的 API 密钥（在 Azure 门户中的"Keys and Endpoint"下可用）
- 部署的端点 URL

## 快速开始

```bash
hermes model
# → 选择 "Azure Foundry"
# → 输入你的端点 URL
# → 输入你的 API 密钥
# Hermes 探测端点并自动检测传输 + 模型
# → 从列表中选择一个模型（或手动输入部署名称）
```

向导将：

1. **嗅探 URL 路径** — 以 `/anthropic` 结尾的 URL 被识别为 Azure Foundry Claude 路由。
2. **探测 `GET <base>/models`** — 如果端点返回 OpenAI 形状的模型列表，Hermes 切换到 `chat_completions` 并用返回的部署 ID 预填充选择器。
3. **探测 Anthropic Messages 形状** — 用于不公开 `/models` 但确实接受 Anthropic Messages 格式的端点的回退。
4. **回退到手动输入** — 拒绝每个探测的私有/受保护端点仍然有效；你选择 API 模式并手动输入部署名称。

所选模型的上下文长度通过 Hermes 的标准元数据链（`models.dev`、提供商元数据、硬编码的系列回退）解析，并存储在 `config.yaml` 中，以便模型可以正确调整其自己的上下文窗口大小。

## 配置（写入 `config.yaml`）

运行向导后，你将看到类似以下内容：

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.openai.azure.com/openai/v1
  api_mode: chat_completions         # 或 "anthropic_messages"
  default: gpt-5.4-mini              # 你的部署 / 模型名称
  context_length: 400000             # 自动检测
```

在 `~/.hermes/.env` 中：

```
AZURE_FOUNDRY_API_KEY=<your-azure-key>
```

## OpenAI 风格端点（GPT、Llama 等）

Azure OpenAI 的 v1 GA 端点接受带有最小更改的标准 `openai` Python 客户端：

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.openai.azure.com/openai/v1
  api_mode: chat_completions
  default: gpt-5.4
```

重要行为：

- **GPT-5.x、codex 和 o-series 自动路由到 Responses API。** Azure Foundry 将 GPT-5 / codex / o1 / o3 / o4 模型部署为仅 Responses-API — 对它们调用 `/chat/completions` 返回 `400 "The requested operation is unsupported."`。Hermes 通过名称检测这些模型系列，并透明地将 `api_mode` 升级到 `codex_responses`，即使 `config.yaml` 仍显示为 `api_mode: chat_completions`。GPT-4、GPT-4o、Llama、Mistral 和其他部署保留在 `/chat/completions` 上。
- **自动使用 `max_completion_tokens`。** Azure OpenAI（如直接 OpenAI）需要对 gpt-4o、o-series 和 gpt-5.x 模型使用 `max_completion_tokens`。Hermes 根据端点发送正确的参数。
- **需要 `api-version` 的 pre-v1 端点。** 如果你有类似 `https://<resource>.openai.azure.com/openai?api-version=2025-04-01-preview` 的旧版基础 URL，Hermes 会提取查询字符串并通过每个请求上的 `default_query` 转发它（否则 OpenAI SDK 在连接路径时会丢弃它）。

## Anthropic 风格端点（通过 Azure Foundry 的 Claude）

对于 Claude 部署，使用 Anthropic 风格路由：

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.services.ai.azure.com/anthropic
  api_mode: anthropic_messages
  default: claude-sonnet-4-6
```

重要行为：

- **`/v1` 从基础 URL 中剥离。** Anthropic SDK 将 `/v1/messages` 附加到每个请求 URL — Hermes 在将 URL 交给 SDK 之前删除任何尾随 `/v1`，以避免双 `/v1` 路径。
- **`api-version` 通过 `default_query` 发送，而不是附加到 URL。** Azure Anthropic 需要 `api-version` 查询字符串。将其嵌入基础 URL 会产生格式错误的路径，如 `/anthropic?api-version=.../v1/messages` 并返回 404。Hermes 改为通过 Anthropic SDK 的 `default_query` 传递 `api-version=2025-04-15`。
- **OAuth token 刷新被禁用。** Azure 部署使用静态 API 密钥。适用于 Anthropic Console 的 `~/.claude/.credentials.json` OAuth token 刷新循环被明确跳过用于 Azure 端点，以防止 Claude Code OAuth token 在会话中期覆盖你的 Azure 密钥。

## 替代方案：`provider: anthropic` + Azure 基础 URL

如果你已经配置了 `provider: anthropic` 并且只想将其指向 Azure AI Foundry 以使用 Claude，则可以完全跳过 `azure-foundry` 提供商：

```yaml
model:
  provider: anthropic
  base_url: https://my-resource.services.ai.azure.com/anthropic
  key_env: AZURE_ANTHROPIC_KEY
  default: claude-sonnet-4-6
```

在 `~/.hermes/.env` 中设置 `AZURE_ANTHROPIC_KEY`。Hermes 检测基础 URL 中的 `azure.com` 并短路绕过 Claude Code OAuth token 链，以便 Azure 密钥直接用于 `x-api-key` 认证。

`key_env` 是规范的 snake_case 字段名称；`api_key_env`（和 camelCase `keyEnv` / `apiKeyEnv`）作为别名被接受。如果同时设置了 `key_env` 和 `AZURE_ANTHROPIC_KEY`/`ANTHROPIC_API_KEY`，则以 `key_env` 命名的 env var 优先。

## 模型发现

Azure **不**公开纯 API 密钥端点来列出你的*已部署*模型部署。部署枚举需要带有 Azure AD 主体的 Azure Resource Manager 认证（`az cognitiveservices account deployment list`），而不是推理 API 密钥。

Hermes 可以做什么：

- Azure OpenAI v1 端点（`<resource>.openai.azure.com/openai/v1`）使用资源的**可用**模型目录公开 `GET /models`。Hermes 使用此列表预填充模型选择器。
- Azure Foundry `/anthropic` 路由：通过 URL 路径检测，手动输入模型名称。
- 私有 / 受防火墙保护的端点：手动输入，附带友好的「无法探测」提示消息。

你始终可以直接输入部署名称 — Hermes 不会针对返回列表进行验证。

## 环境变量

| 变量 | 用途 |
|----------|---------|
| `AZURE_FOUNDRY_API_KEY` | Azure AI Foundry / Azure OpenAI 的主 API 密钥 |
| `AZURE_FOUNDRY_BASE_URL` | 端点 URL（通过 `hermes model` 设置；env var 用作回退） |
| `AZURE_ANTHROPIC_KEY` | 由 `provider: anthropic` + Azure 基础 URL 使用（作为 `ANTHROPIC_API_KEY` 的替代） |

## 故障排除

**在 gpt-5.x 部署上 401 Unauthorized。**
Azure 在 `/chat/completions` 上提供 gpt-5.x，而不是 `/responses`。当 URL 包含 `openai.azure.com` 时，Hermes 自动处理此问题，但如果你看到带有 `Invalid API key` 正文的 401，请检查 `config.yaml` 中的 `api_mode` 是否为 `chat_completions`。

**在 `/v1/messages?api-version=.../v1/messages` 上 404。**
这是来自预修复 Azure Anthropic 设置的格式错误 URL bug。升级 Hermes — `api-version` 参数现在通过 `default_query` 传递，而不是嵌入基础 URL，因此 SDK 在 URL 连接期间无法损坏它。

**向导显示"Auto-detection incomplete。"**
端点拒绝了 `/models` 探测和 Anthropic Messages 探测。对于防火墙后或具有 IP 允许列表的私有端点，这是正常的。回退到手动 API 模式选择并输入你的部署名称 — 一切仍然有效，Hermes 只是无法预填充选择器。

**选择了错误的传输。**
再次运行 `hermes model`，向导将重新探测。如果探测仍然选择错误的模式，你可以直接编辑 `config.yaml`：

```yaml
model:
  provider: azure-foundry
  api_mode: anthropic_messages   # 或 chat_completions
```

## 相关

- [环境变量](/docs/reference/environment-variables)
- [配置](/docs/user-guide/configuration)
- [AWS Bedrock](/docs/guides/aws-bedrock) — 另一个主要的云提供商集成
