---
sidebar_position: 16
title: "Google Gemini"
description: "使用 Hermes 代理与 Google Gemini——原生 AI Studio API、API 密钥设置、OAuth 选项、工具调用、流式传输和配额指南"
---

# Google Gemini

Hermes 代理支持 Google Gemini 作为原生提供商，使用 **Google AI Studio / Gemini API**——不是 OpenAI 兼容端点。这让 Hermes 可以将其内部的 OpenAI 格式消息和工具循环转换为 Gemini 的原生 `generateContent` API，同时保留工具调用、流式传输、多模态输入和 Gemini 特定的响应元数据。

Hermes 还支持一个独立的 **Google Gemini (OAuth)** 提供商，它使用与 Google 的 Gemini CLI 相同的 Cloud Code Assist 后端。对于最低风险的官方 API 路径，请使用 API 密钥提供商（`gemini`）。

## 先决条件

- **Google AI Studio API 密钥**——在 [aistudio.google.com/apikey](https://aistudio.google.com/apikey) 创建
- **已启用计费的 Google Cloud 项目**——推荐用于代理使用。Gemini 的免费层级对于长时间运行的代理会话来说太小，因为 Hermes 可能每次用户轮次进行多次模型调用。
- **已安装 Hermes**——原生 Gemini 提供商不需要额外的 Python 包。

:::tip API 密钥路径
设置 `GOOGLE_API_KEY` 或 `GEMINI_API_KEY`。Hermes 会为 `gemini` 提供商检查这两个名称。
:::

## 快速开始

```bash
# 添加你的 Gemini API 密钥
echo "GOOGLE_API_KEY=..." >> ~/.hermes/.env

# 选择 Gemini 作为你的提供商
hermes model
# → 选择"更多提供商..." → "Google AI Studio"
# → Hermes 检查你的密钥层级并显示 Gemini 模型
# → 选择一个模型

# 开始聊天
hermes chat
```

如果你喜欢直接编辑配置，请使用原生 Gemini API 基础 URL：

```yaml
model:
  default: gemini-3-flash-preview
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

## 配置

运行 `hermes model` 后，你的 `~/.hermes/config.yaml` 将包含：

```yaml
model:
  default: gemini-3-flash-preview
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

在 `~/.hermes/.env` 中：

```bash
GOOGLE_API_KEY=...
```

### 原生 Gemini API

推荐的端点是：

```text
https://generativelanguage.googleapis.com/v1beta
```

Hermes 检测此端点并创建其原生 Gemini 适配器。在内部，Hermes 仍将代理循环保持在 OpenAI 格式的消息中，然后将每个请求转换为 Gemini 的原生架构：

- `messages[]` → Gemini `contents[]`
- 系统提示 → Gemini `systemInstruction`
- 工具架构 → Gemini `functionDeclarations`
- 工具结果 → Gemini `functionResponse` 部分
- 流式响应 → Hermes 循环的 OpenAI 格式流块

:::note Gemini 3 思维签名
对于 Gemini 3 工具使用，Hermes 保留附加到函数调用部分的 `thoughtSignature` 值，并在下一个工具轮次重放它们。这涵盖了多步骤代理工作流的验证关键路径。

Gemini 3 也可能将思维签名附加到其他响应部分。Hermes 的原生适配器目前针对代理工具循环进行了优化，因此尚未以完整的部分级保真度重放每个非工具调用签名。
:::

### 优先使用原生端点

Google 也公开了一个 OpenAI 兼容端点：

```text
https://generativelanguage.googleapis.com/v1beta/openai/
```

对于 Hermes 代理会话，优先使用上面的原生 Gemini 端点。Hermes 包含一个原生 Gemini 适配器，因此它可以映射多轮工具使用、工具调用结果、流式传输、多模态输入和 Gemini 响应元数据直接到 Gemini 的 `generateContent` API。当你特别需要 OpenAI API 兼容性时，OpenAI 兼容端点仍然有用。
如果你之前将 `GEMINI_BASE_URL` 设置为 `/openai` URL，请删除它或更改它：

```bash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

### OAuth 提供商

Hermes 还有一个 `google-gemini-cli` 提供商：

```bash
hermes model
# → 选择"Google Gemini (OAuth)"
```

这使用浏览器 PKCE 登录和 Cloud Code Assist 后端。这对于想要 Gemini CLI 风格 OAuth 的用户很有用，但 Hermes 会显示明确警告，因为 Google 可能会将来自第三方软件的 Gemini CLI OAuth 客户端使用视为违反政策。对于生产或最低风险使用，请优先使用上面的 API 密钥提供商。

## 可用模型

`hermes model` 选择器显示 Hermes 提供商注册表中维护的 Gemini 模型。常见选择包括：

| 模型 | ID | 说明 |
|-------|----|-------|
| Gemini 3.1 Pro Preview | `gemini-3.1-pro-preview` | 可用时最强大的预览模型 |
| Gemini 3 Pro Preview | `gemini-3-pro-preview` | 强大的推理和编码模型 |
| Gemini 3 Flash Preview | `gemini-3-flash-preview` | 推荐的速度和能力平衡默认值 |
| Gemini 3.1 Flash Lite Preview | `gemini-3.1-flash-lite-preview` | 可用时最快/成本最低的选项 |

模型可用性随时间变化。如果模型消失或未为你的密钥启用，请再次运行 `hermes model` 并从当前列表中选择一个。

:::info 模型 ID
当 `provider: gemini` 时，使用 Gemini 的原生模型 ID，如 `gemini-3-flash-preview`，而不是 OpenRouter 风格的 ID，如 `google/gemini-3-flash-preview`。
:::

### 最新别名

Google 为 Pro 和 Flash Gemini 系列发布移动别名。`gemini-pro-latest` 和 `gemini-flash-latest` 在你希望 Google 自动推进模型而无需更改 Hermes 配置时很有用。

| 别名 | 当前跟踪 | 说明 |
|-------|------------------|-------|
| `gemini-pro-latest` | 最新的 Gemini Pro 模型 | 当你想要 Google 当前的 Pro 默认值时最好 |
| `gemini-flash-latest` | 最新的 Gemini Flash 模型 | 当你想要 Google 当前的 Flash 默认值时最好 |

```yaml
model:
  default: gemini-pro-latest
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

如果需要严格的再现性，优先使用明确的模型 ID，如 `gemini-3.1-pro-preview` 或 `gemini-3-flash-preview`。

### 通过 Gemini API 的 Gemma

Google 也通过 Gemini API 公开 Gemma 模型。Hermes 将这些识别为 Google 模型，但从默认模型选择器中隐藏非常低吞吐量的 Gemma 条目，这样新用户就不会意外地为长时间运行的代理会话选择评估层级模型。

有用的评估 ID 包括：

| 模型 | ID | 说明 |
|-------|----|-------|
| Gemma 4 31B IT | `gemma-4-31b-it` | 更大的 Gemma 模型；对兼容性和质量评估有用 |
| Gemma 4 26B A4B IT | `gemma-4-26b-a4b-it` | 可用时较小的活跃参数变体 |

这些模型最好被视为 Gemini API 密钥上的评估选项。Google 的 Gemma API 定价仅限免费层级，与生产的 Gemini 模型相比，使用上限很低，因此持续的 Hermes 代理使用通常应该转移到付费 Gemini 模型、自托管部署或具有适当配额的其他提供商。
要使用从选择器中隐藏的 Gemma 模型，请直接设置：

```yaml
model:
  default: gemma-4-31b-it
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

## 会话中切换模型

在对话期间使用 `/model` 命令：

```text
/model gemini-3-flash-preview
/model gemini-flash-latest
/model gemini-3-pro-preview
/model gemini-pro-latest
/model gemma-4-31b-it
/model gemini-3.1-flash-lite-preview
```

如果你尚未配置 Gemini，请退出会话并首先运行 `hermes model`。`/model` 在已配置的提供商和模型之间切换；它不收集新的 API 密钥。

## 诊断

```bash
hermes doctor
```

doctor 检查：

- `GOOGLE_API_KEY` 或 `GEMINI_API_KEY` 是否可用
- `google-gemini-cli` 是否存在 Gemini OAuth 凭据
- 是否可解析已配置的提供商凭据

对于 OAuth 配额使用，在 Hermes 会话中运行：

```text
/gquota
```

`/gquota` 适用于 `google-gemini-cli` OAuth 提供商，而不是 AI Studio API 密钥提供商。

## 网关（消息传递平台）

Gemini 适用于所有 Hermes 网关平台（Telegram、Discord、Slack、WhatsApp、LINE、Feishu 等）。将 Gemini 配置为你的提供商，然后正常启动网关：

```bash
hermes gateway setup
hermes gateway start
```

网关读取 `config.yaml` 并使用相同的 Gemini 提供商配置。

## 故障排除

### "Gemini native client requires an API key"

Hermes 找不到可用的 API 密钥。将其中之一添加到 `~/.hermes/.env`：

```bash
GOOGLE_API_KEY=...
# 或
GEMINI_API_KEY=...
```

然后再次运行 `hermes model`。

### "This Google API key is on the free tier"

Hermes 在设置期间探测 Gemini API 密钥。免费层级配额可能在少量代理轮次后耗尽，因为工具使用、重试、压缩和辅助任务可能需要多次模型调用。

在附加到你的密钥的 Google Cloud 项目上启用计费，根据需要重新生成密钥，然后运行：

```bash
hermes model
```

### "404 model not found"

所选模型不适用于你的帐户、区域或密钥。再次运行 `hermes model` 并从当前列表中选择另一个 Gemini 模型。

### Gemma 模型未在 `hermes model` 中显示

Hermes 可能默认从选择器中隐藏低吞吐量的 Gemma 模型。如果你有意想要评估一个，请在 `~/.hermes/config.yaml` 中直接设置模型 ID。

### Gemma 上的 "429 quota exceeded"

通过 Gemini API 公开的 Gemma 模型对评估有用，但其 Gemini API 免费层级上限很低。将它们用于兼容性测试，然后为持续的代理会话切换到付费 Gemini 模型或其他提供商。

### 配置了 OpenAI 兼容端点

检查 `~/.hermes/.env` 中是否有：

```bash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

将其更改为原生端点或删除覆盖：

```bash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

### OAuth 登录警告

`google-gemini-cli` 提供商使用 Gemini CLI / Cloud Code Assist OAuth 流程。Hermes 在启动它之前会警告，因为这不同于官方 AI Studio API 密钥路径。将 `provider: gemini` 与 `GOOGLE_API_KEY` 用于官方 API 密钥集成。

### 工具调用因架构错误而失败

升级 Hermes 并重新运行 `hermes model`。原生 Gemini 适配器为 Gemini 更严格的函数声明格式清理工具架构；较旧的构建或自定义端点可能不会。

## 相关

- [AI 提供商](/docs/integrations/providers)
- [配置](/docs/user-guide/configuration)
- [后备提供商](/docs/user-guide/features/fallback-providers)
- [AWS Bedrock](/docs/guides/aws-bedrock)——使用 AWS 凭据的原生云提供商集成
