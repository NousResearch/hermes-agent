---
sidebar_position: 16
title: "Google Gemini"
description: "将 Hermes Agent 与 Google Gemini 配合使用——原生 AI Studio API、API 密钥配置、工具调用、流式传输及配额说明"
---

# Google Gemini

Hermes Agent 通过 **Google AI Studio / Gemini API** 原生支持 Google Gemini——而非 OpenAI 兼容端点。这使 Hermes 能够将其内部 OpenAI 格式的消息和工具循环转换为 Gemini 原生的 `generateContent` API，同时保留工具调用、流式传输、多模态输入以及 Gemini 特有的响应元数据。

## 前提条件

- **Google AI Studio API 密钥** — 在 [aistudio.google.com/apikey](https://aistudio.google.com/apikey) 创建
- **已启用计费的 Google Cloud 项目** — 推荐用于 Agent 场景。Gemini 免费层级对长时间运行的 Agent 会话而言配额过小，因为 Hermes 每次用户交互可能发起多次模型调用。
- **已安装 Hermes** — 原生 Gemini provider 无需额外安装 Python 包。

:::tip API 密钥路径
设置 `GOOGLE_API_KEY` 或 `GEMINI_API_KEY`。Hermes 对 `gemini` provider 会同时检查这两个名称。
:::

## 快速开始

```bash
# 添加 Gemini API 密钥
echo "GOOGLE_API_KEY=..." >> ~/.hermes/.env

# 选择 Gemini 作为 provider
hermes model
# → 选择 "More providers..." → "Google AI Studio"
# → Hermes 检查密钥层级并显示 Gemini 模型列表
# → 选择一个模型

# 开始对话
hermes chat
```

如果你偏好直接编辑配置文件，请使用原生 Gemini API 基础 URL：

```yaml
model:
  default: gemini-3-flash-preview
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

## 配置

运行 `hermes model` 后，`~/.hermes/config.yaml` 将包含：

```yaml
model:
  default: gemini-3-flash-preview
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

`~/.hermes/.env` 中：

```bash
GOOGLE_API_KEY=...
```

### 原生 Gemini API

推荐使用的端点为：

```text
https://generativelanguage.googleapis.com/v1beta
```

Hermes 检测到该端点后会创建原生 Gemini 适配器。在内部，Hermes 仍以 OpenAI 格式维护 Agent 循环，然后将每个请求转换为 Gemini 原生 schema：

- `messages[]` → Gemini `contents[]`
- 系统提示（system prompt）→ Gemini `systemInstruction`
- 工具 schema → Gemini `functionDeclarations`
- 工具结果 → Gemini `functionResponse` 部分
- 流式响应 → 供 Hermes 循环使用的 OpenAI 格式流式数据块

:::note Gemini 3 思维签名
对于 Gemini 3 的工具调用，Hermes 会保留附加在函数调用部分的 `thoughtSignature` 值，并在下一个工具轮次中重放。这覆盖了多步骤 Agent 工作流中验证关键路径的需求。

Gemini 3 也可能在其他响应部分附加思维签名。Hermes 的原生适配器目前针对 Agent 工具循环进行了优化，尚未以完整的部分级保真度重放所有非工具调用签名。
:::

### 优先使用原生端点

Google 还提供了 OpenAI 兼容端点：

```text
https://generativelanguage.googleapis.com/v1beta/openai/
```

对于 Hermes Agent 会话，请优先使用上述原生 Gemini 端点。Hermes 内置原生 Gemini 适配器，可将多轮工具调用、工具调用结果、流式传输、多模态输入以及 Gemini 响应元数据直接映射到 Gemini 的 `generateContent` API。OpenAI 兼容端点在你明确需要 OpenAI API 兼容性时仍然有用。

如果你之前将 `GEMINI_BASE_URL` 设置为 `/openai` URL，请将其删除或修改：

```bash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

## 可用模型

`hermes model` 选择器显示 Hermes provider 注册表中维护的 Gemini 模型。常见选项包括：

| 模型 | ID | 说明 |
|------|----|------|
| Gemini 3.1 Pro Preview | `gemini-3.1-pro-preview` | 可用时最强大的预览模型 |
| Gemini 3 Pro Preview | `gemini-3-pro-preview` | 强大的推理和编码模型 |
| Gemini 3 Flash Preview | `gemini-3-flash-preview` | 推荐的默认选项，速度与能力均衡 |
| Gemini 3.1 Flash Lite Preview | `gemini-3.1-flash-lite-preview` | 可用时速度最快、成本最低的选项 |

模型可用性会随时间变化。如果某个模型消失或未对你的密钥启用，请重新运行 `hermes model` 并从当前列表中选择。

:::info 模型 ID
当 `provider: gemini` 时，请使用 Gemini 原生模型 ID，如 `gemini-3-flash-preview`，而非 OpenRouter 风格的 ID（如 `google/gemini-3-flash-preview`）。
:::

### 最新别名

Google 为 Pro 和 Flash Gemini 系列发布了滚动别名。当你希望 Google 自动升级模型而无需修改 Hermes 配置时，`gemini-pro-latest` 和 `gemini-flash-latest` 非常实用。

| 别名 | 当前指向 | 说明 |
|------|----------|------|
| `gemini-pro-latest` | 最新 Gemini Pro 模型 | 需要 Google 当前 Pro 默认值时的最佳选择 |
| `gemini-flash-latest` | 最新 Gemini Flash 模型 | 需要 Google 当前 Flash 默认值时的最佳选择 |

```yaml
model:
  default: gemini-pro-latest
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

如果需要严格的可复现性，请优先使用明确的模型 ID，如 `gemini-3.1-pro-preview` 或 `gemini-3-flash-preview`。

### 通过 Gemini API 使用 Gemma

Google 也通过 Gemini API 提供 Gemma 模型。Hermes 将这些模型识别为 Google 模型，但会在默认模型选择器中隐藏吞吐量极低的 Gemma 条目，以防新用户在长时间运行的 Agent 会话中意外选择评估层级的模型。

常用评估 ID 包括：

| 模型 | ID | 说明 |
|------|----|------|
| Gemma 4 31B IT | `gemma-4-31b-it` | 较大的 Gemma 模型；适用于兼容性和质量评估 |
| Gemma 4 26B A4B IT | `gemma-4-26b-a4b-it` | 可用时的较小活跃参数变体 |

这些模型最适合作为 Gemini API 密钥的评估选项。Google 的 Gemma API 定价仅限免费层级，与生产级 Gemini 模型相比使用上限较低，因此持续的 Hermes Agent 使用通常应切换到付费 Gemini 模型、自托管部署或具有适当配额的其他 provider。

如需使用选择器中隐藏的 Gemma 模型，请直接在配置中指定：

```yaml
model:
  default: gemma-4-31b-it
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

## 会话中途切换模型

在对话中使用 `/model` 命令：

```text
/model gemini-3-flash-preview
/model gemini-flash-latest
/model gemini-3-pro-preview
/model gemini-pro-latest
/model gemma-4-31b-it
/model gemini-3.1-flash-lite-preview
```

如果尚未配置 Gemini，请退出会话并先运行 `hermes model`。`/model` 用于在已配置的 provider 和模型之间切换，不会收集新的 API 密钥。

## 诊断

```bash
hermes doctor
```

doctor 命令检查：

- `GOOGLE_API_KEY` 或 `GEMINI_API_KEY` 是否可用
- 已配置的 provider 凭据是否可以解析

## Gateway（消息平台）

Gemini 可与所有 Hermes gateway 平台配合使用（Telegram、Discord、Slack、WhatsApp、LINE、飞书等）。将 Gemini 配置为你的 provider，然后正常启动 gateway：

```bash
hermes gateway setup
hermes gateway start
```

gateway 读取 `config.yaml` 并使用相同的 Gemini provider 配置。

## 故障排查

### "Gemini native client requires an API key"

Hermes 找不到可用的 API 密钥。请将以下任一项添加到 `~/.hermes/.env`：

```bash
GOOGLE_API_KEY=...
# 或
GEMINI_API_KEY=...
```

然后重新运行 `hermes model`。

### "This Google API key is on the free tier"

Hermes 在设置期间会探测 Gemini API 密钥。由于工具调用、重试、压缩和辅助任务可能需要多次模型调用，免费层级配额在少数几轮 Agent 交互后即可耗尽。

请为与密钥关联的 Google Cloud 项目启用计费，必要时重新生成密钥，然后运行：

```bash
hermes model
```

### "404 model not found"

所选模型对你的账号、地区或密钥不可用。重新运行 `hermes model` 并从当前列表中选择其他 Gemini 模型。

### Gemma 模型未显示在 `hermes model` 中

Hermes 默认可能会在选择器中隐藏低吞吐量的 Gemma 模型。如果你有意评估某个模型，请直接在 `~/.hermes/config.yaml` 中设置模型 ID。

### Gemma 出现 "429 quota exceeded"

通过 Gemini API 提供的 Gemma 模型适合评估使用，但其 Gemini API 免费层级上限较低。请将其用于兼容性测试，然后切换到付费 Gemini 模型或其他 provider 以进行持续的 Agent 会话。

### 已配置 OpenAI 兼容端点

检查 `~/.hermes/.env` 中是否存在：

```bash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

将其修改为原生端点或删除该覆盖项：

```bash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

### 工具调用因 schema 错误而失败

升级 Hermes 并重新运行 `hermes model`。原生 Gemini 适配器会针对 Gemini 更严格的函数声明格式对工具 schema 进行清理；旧版本或自定义端点可能不支持此功能。

## 相关链接

- [AI Providers](/integrations/providers)
- [Configuration](/user-guide/configuration)
- [Fallback Providers](/user-guide/features/fallback-providers)
- [AWS Bedrock](/guides/aws-bedrock) — 使用 AWS 凭据的原生云 provider 集成