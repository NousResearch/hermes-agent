---
sidebar_position: 15
title: "订阅代理"
description: "将你的 Nous Portal 订阅（或其他 OAuth 提供商）用作外部应用的 OpenAI 兼容端点"
---

# 订阅代理

订阅代理是一个本地 HTTP 服务器，让外部应用——OpenViking、Karakeep、Open WebUI，以及任何支持 OpenAI 兼容聊天补全（chat completions）的应用——能够将你的 Hermes 托管提供商订阅用作其 LLM 端点。代理会自动附加正确的凭据（并在需要时自动刷新），因此应用无需静态 API 密钥。

这与 [API 服务器](./api-server.md) 不同：

| | API 服务器 | 订阅代理 |
|---|---|---|
| 服务内容 | 你的 Agent（完整工具集、记忆、技能） | 原始模型推理 |
| 使用场景 | "将 Hermes 用作聊天后端" | "从其他应用使用我的 Portal 订阅" |
| 认证 | 你的 `API_SERVER_KEY` | 任意 bearer（代理附加真实凭据） |
| 工具调用 | 是——Agent 执行工具 | 否——仅透传 |

当你需要将 **Agent** 作为后端时，使用 API 服务器。当你只需要通过订阅访问**模型**时，使用代理。

## 快速开始

### 1. 登录你的提供商（仅需一次）

```bash
hermes portal
```

这会打开浏览器进行 Nous Portal OAuth 流程。Hermes 将刷新令牌存储在 `~/.hermes/auth.json` 中——与所有 Hermes 提供商登录信息存放在同一位置。

### 2. 启动代理

```bash
hermes proxy start
```

```
Starting Hermes proxy for Nous Portal
  Listening on:  http://127.0.0.1:8645/v1
  Forwarding to: (resolved per-request from your subscription)
  Use any bearer token in the client — the proxy attaches your real credential.
```

保持在前台运行。如需在注销后继续运行，请使用 `tmux`、`nohup` 或 systemd 单元。

### 3. 将你的应用指向代理

任何 OpenAI 兼容应用的配置都使用相同的三元组：

```
Base URL:   http://127.0.0.1:8645/v1
API key:    任意值（例如 "sk-unused"）
Model:      Hermes-4-70B    # 或 Hermes-4.3-36B、Hermes-4-405B
```

代理会忽略来自你应用的 `Authorization` 请求头，并将你真实的 Portal 凭据附加到上游请求中。当 bearer 令牌临近过期时，刷新会自动进行。

## 可用提供商

```bash
hermes proxy providers
```

当前已内置：`nous`（Nous Portal）。更多 OAuth 提供商可通过在 `hermes_cli/proxy/adapters/` 中实现 `UpstreamAdapter` 接口来添加。

## 检查状态

```bash
hermes proxy status
```

```
Hermes proxy upstream adapters

  [nous    ] Nous Portal — ready (bearer expires 2026-05-15T06:43:21Z)
```

如果显示 `not logged in`，请运行 `hermes portal`。如果显示 `credentials need attention`，说明你的刷新令牌已被撤销（较少见——通常发生在你从 Portal Web UI 退出登录时）——重新运行 `hermes portal` 即可。

## 允许的路径

代理仅转发上游实际提供的路径。对于 Nous Portal：

| 路径 | 用途 |
|------|---------|
| `/v1/chat/completions` | 聊天补全（流式与非流式） |
| `/v1/completions` | 旧版文本补全 |
| `/v1/embeddings` | Embeddings（嵌入） |
| `/v1/models` | 模型列表 |

其他路径（`/v1/images/generations`、`/v1/audio/speech` 等）将返回 404，并附带明确的错误信息指向允许的路径。这可防止游离客户端向上游发送异常请求。

## 配置 OpenViking 使用 Portal

[OpenViking](https://github.com/volcengine/OpenViking) 是一个上下文数据库，需要 LLM 提供商来支持其 VLM（用于提取记忆的视觉/语言模型）和 embedding 模型。通过代理，你可以将其 `vlm.api_base` 指向本地代理：

编辑 `~/.openviking/ov.conf`：

```json
{
  "vlm": {
    "provider": "openai",
    "model": "Hermes-4-70B",
    "api_base": "http://127.0.0.1:8645/v1",
    "api_key": "unused-proxy-attaches-real-creds"
  }
}
```

然后在终端中与 `openviking-server` 一起启动代理：

```bash
# 终端 1
hermes proxy start

# 终端 2
openviking-server
```

OpenViking 的 VLM 调用现在将通过你的 Portal 订阅进行。Embedding 模型侧仍需要自己的提供商——Portal 确实提供 `/v1/embeddings`，但模型选择取决于你的套餐所支持的内容；请查看 `portal.nousresearch.com/models`。

## 配置 Karakeep（或任何书签/摘要应用）

[Karakeep](https://karakeep.app/) 使用 OpenAI 兼容 API 进行书签摘要。在其配置中：

```bash
# Karakeep .env
OPENAI_API_BASE_URL=http://127.0.0.1:8645/v1
OPENAI_API_KEY=any-non-empty-string
INFERENCE_TEXT_MODEL=Hermes-4-70B
```

同样的方式适用于 Open WebUI、LobeHub、NextChat 或任何其他 OpenAI 兼容客户端。

## 在局域网上暴露

默认情况下，代理绑定 `127.0.0.1`（仅限本机）。若要让网络中的其他机器使用：

```bash
hermes proxy start --host 0.0.0.0 --port 8645
```

⚠ **注意：** 你网络中的任何人现在都可以使用你的 Portal 订阅。代理本身没有认证机制——它接受任意 bearer。如果你将其暴露在可信网络之外，请使用防火墙、VPN 或带有适当认证的反向代理。

## 速率限制

你的 Portal 套餐的 RPM/TPM 限制适用于整个代理。代理不进行扇出或连接池——它是单个 bearer，使用你的完整订阅配额。请在 [portal.nousresearch.com](https://portal.nousresearch.com) 监控使用情况。

## 架构

代理设计上尽量精简。每个请求的处理流程：

1. 从你的应用接收 `POST /v1/chat/completions`
2. 查找适配器的当前凭据（如临近过期则刷新）
3. 原样转发请求体，附加 `Authorization: Bearer <minted-key>`
4. 将响应原样流式返回（SSE 保持不变）

无转换。不记录请求体。无 Agent 循环。代理是一个附加凭据的透传通道。

## 未来：更多 OAuth 提供商

适配器系统是可插拔的。添加新提供商（例如 HuggingFace、GitHub Copilot 的聊天端点、通过 OAuth 接入的 Anthropic）需要在 `hermes_cli/proxy/adapters/<provider>.py` 中实现 `UpstreamAdapter`，并在 `adapters/__init__.py` 中注册。协议层面不兼容 OpenAI 的提供商（例如 Anthropic Messages API）需要额外的转换层，这超出了当前版本的范围。