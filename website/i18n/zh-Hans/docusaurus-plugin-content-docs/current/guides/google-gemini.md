---
sidebar_position: 16
title: "Google Gemini"
description: "在 Hermes Agent 中使用 Google Gemini：原生 AI Studio API、API key 配置、OAuth 选项、工具调用、流式输出和配额建议"
---

# Google Gemini

Hermes Agent 支持把 Google Gemini 作为原生 provider 使用，走的是 **Google AI Studio / Gemini API**，而不是 OpenAI-compatible endpoint。这样 Hermes 可以把内部的 OpenAI 形态消息循环转换成 Gemini 原生 `generateContent` API，同时保留工具调用、流式输出、多模态输入和 Gemini 特定的响应元数据。

Hermes 也支持单独的 **Google Gemini (OAuth)** provider，使用的是 Google Gemini CLI 背后的 Cloud Code Assist backend。对于最稳妥的官方 API 方式，推荐使用 API key provider（`gemini`）。

## 前置条件

- Google AI Studio API key
- 建议开启计费的 Google Cloud 项目
- 已安装 Hermes

:::tip API key 路径
设置 `GOOGLE_API_KEY` 或 `GEMINI_API_KEY`，Hermes 会同时检查这两个名字。
:::

## 快速开始

```bash
echo "GOOGLE_API_KEY=..." >> ~/.hermes/.env
hermes model
hermes chat
```

如果你更喜欢直接编辑配置，可以使用原生 Gemini API base URL：

```yaml
model:
  default: gemini-3-flash-preview
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

## 配置

运行 `hermes model` 后，`~/.hermes/config.yaml` 会包含 provider 和 base URL 配置。

### 原生 Gemini API

推荐的 endpoint 是：

```text
https://generativelanguage.googleapis.com/v1beta
```

Hermes 会检测这个 endpoint 并创建原生 Gemini adapter。

### OAuth provider

Hermes 也提供 `google-gemini-cli` provider：

```bash
hermes model
```

它使用浏览器 PKCE 登录和 Cloud Code Assist backend。

## 可用模型

常见选择包括：

| 模型 | ID | 说明 |
|---|---|---|
| Gemini 3.1 Pro Preview | `gemini-3.1-pro-preview` | 预览版中最强的选项之一 |
| Gemini 3 Pro Preview | `gemini-3-pro-preview` | 推理和编码都很强 |
| Gemini 3 Flash Preview | `gemini-3-flash-preview` | 默认平衡速度与能力 |
| Gemini 3.1 Flash Lite Preview | `gemini-3.1-flash-lite-preview` | 更快、成本更低 |

## 运行中切换模型

```text
/model gemini-3-flash-preview
/model gemini-flash-latest
/model gemini-3-pro-preview
/model gemini-pro-latest
```

## 排障

### "Gemini native client requires an API key"

把 `GOOGLE_API_KEY` 或 `GEMINI_API_KEY` 加到 `~/.hermes/.env`。

### "404 model not found"

模型对你的账号、地区或 key 不可用。重新运行 `hermes model` 选择别的模型。

### 结语

如果你追求最稳妥的官方路径，用 API key provider；如果你需要 Gemini CLI 风格登录，再考虑 OAuth provider。
