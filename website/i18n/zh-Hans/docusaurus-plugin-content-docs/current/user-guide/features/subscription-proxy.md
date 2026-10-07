---
sidebar_position: 15
title: "订阅代理"
description: "将你的 xAI OAuth 登录用作外部应用的 OpenAI 兼容端点"
---

# 订阅代理

订阅代理是一个本地 HTTP 服务器，让外部应用——OpenViking、Karakeep、Open WebUI，以及任何支持 OpenAI 兼容聊天补全（chat completions）的应用——能够将你的 Rabbit 托管 OAuth 提供商登录用作其 LLM 端点。代理会自动附加正确的凭据（并在需要时自动刷新），因此应用无需静态 API 密钥。

这与 [API 服务器](./api-server.md) 不同：

| | API 服务器 | 订阅代理 |
|---|---|---|
| 服务内容 | 你的 Agent（完整工具集、记忆、技能） | 原始模型推理 |
| 使用场景 | "将 Rabbit 用作聊天后端" | "从其他应用使用我的 xAI 登录" |
| 认证 | 你的 `API_SERVER_KEY` | 任意 bearer（代理附加真实凭据） |
| 工具调用 | 是——Agent 执行工具 | 否——仅透传 |

当你需要将 **Agent** 作为后端时，使用 API 服务器。当你只需要通过 OAuth 登录访问**模型**时，使用代理。

## 快速开始

### 1. 登录你的提供商（仅需一次）

```bash
rabbit auth add xai-oauth --type oauth
```

这会运行 xAI OAuth 流程。Rabbit 将凭据存储在 `~/.rabbit/auth.json` 中——与所有 Rabbit 提供商登录信息存放在同一位置。

### 2. 启动代理

```bash
rabbit proxy start
```

```
Starting Rabbit proxy for xAI Grok OAuth
  Listening on:  http://127.0.0.1:8645/v1
  Forwarding to: (resolved per-request from your OAuth credential)
  Use any bearer token in the client — the proxy attaches your real credential.
```

保持在前台运行。如需在注销后继续运行，请使用 `tmux`、`nohup` 或 systemd 单元。

### 3. 将你的应用指向代理

任何 OpenAI 兼容应用的配置都使用相同的三元组：

```
Base URL:   http://127.0.0.1:8645/v1
API key:    任意值（例如 "sk-unused"）
Model:      grok-4    # 或你的 xAI 账户可用的任意模型
```

代理会忽略来自你应用的 `Authorization` 请求头，并将你真实的 xAI 凭据附加到上游请求中。当 bearer 令牌临近过期时，刷新会自动进行。

## 可用提供商

```bash
rabbit proxy providers
```

当前内置：`xai`（xAI / Grok OAuth）。可通过在 `rabbit_cli/proxy/adapters/` 中实现 `UpstreamAdapter` 接口来添加更多 OAuth 提供商。

## 查看状态

```bash
rabbit proxy status
```

```
Rabbit proxy upstream adapters

  [xai     ] xAI Grok OAuth — ready
```

如果看到 `not logged in`，请运行 `rabbit auth add xai-oauth --type oauth`。如果看到 `credentials need attention`，说明凭据已被撤销或过期——重新运行同一命令即可。

## 允许的路径

代理只转发上游实际提供的路径。对于 xAI：

| 路径 | 用途 |
|------|---------|
| `/v1/chat/completions` | 聊天补全（流式 + 非流式） |
| `/v1/responses` | Responses API |
| `/v1/completions` | 传统文本补全 |
| `/v1/embeddings` | 嵌入 |
| `/v1/models` | 模型列表 |

其他路径（`/v1/images/generations`、`/v1/audio/speech` 等）会返回 404 并附带指向允许路径的清晰错误。这可防止杂散客户端向上游泄漏异常请求。

## 配置 OpenViking 使用代理

[OpenViking](https://github.com/volcengine/OpenViking) 是一个上下文数据库，其 VLM（用于提取记忆的视觉/语言模型）和嵌入模型需要 LLM 提供商。借助代理，你可以将其 `vlm.api_base` 指向本地代理：

编辑 `~/.openviking/ov.conf`：

```json
{
  "vlm": {
    "provider": "openai",
    "model": "grok-4",
    "api_base": "http://127.0.0.1:8645/v1",
    "api_key": "unused-proxy-attaches-real-creds"
  }
}
```

然后在 `openviking-server` 旁边的终端中启动代理：

```bash
# 终端 1
rabbit proxy start

# 终端 2
openviking-server
```

OpenViking 的 VLM 调用现在会通过你的 xAI 登录进行。嵌入模型一侧仍需单独的提供商——代理确实提供 `/v1/embeddings`，但可用的模型取决于你的 xAI 账户支持什么。

## 配置 Karakeep（或任何书签/摘要应用）

[Karakeep](https://karakeep.app/) 使用 OpenAI 兼容 API 进行书签摘要。在其配置中：

```bash
# Karakeep .env
OPENAI_API_BASE_URL=http://127.0.0.1:8645/v1
OPENAI_API_KEY=any-non-empty-string
INFERENCE_TEXT_MODEL=grok-4
```

同样的模式适用于 Open WebUI、LobeChat、NextChat 或任何其他 OpenAI 兼容客户端。

## 暴露到局域网

默认情况下，代理绑定 `127.0.0.1`（仅 localhost）。它会拒绝 `Host` 头不是其自身地址（`localhost`、`127.0.0.1`、`[::1]` 或绑定的 IP）的请求，以及来自其他站点的浏览器请求（`Origin` 非自身，或 `Sec-Fetch-Site` 为 `cross-site`/`same-site`），因此浏览器中打开的网页无法使用它。SDK 和 `curl` 等客户端不发送这些头，不受影响。如需让网络上的其他机器使用：

```bash
rabbit proxy start --host 0.0.0.0 --port 8645
```

⚠ **请注意：** 你网络上的任何人现在都可以使用你的 xAI 登录。通配符绑定会跳过 `Host` 检查（任何名称都可访问），因此代理会拒绝网页发出的一切请求（任何 `Origin`，或 `Sec-Fetch-Site` 非 `none`）：浏览器无法使用通配符绑定的代理。代理本身没有认证——它接受任意 bearer。如需在受信网络之外暴露，请使用防火墙、VPN 或带有适当认证的反向代理。

## 速率限制

你的 xAI 账户的速率限制适用于整个代理。代理不会发散或池化——它是使用你账户全部配额的单个凭据。

## 架构

代理刻意保持极简。每个请求：

1. 接收来自应用的 `POST /v1/chat/completions`
2. 查找适配器的当前凭据（临近过期则刷新）
3. 原样转发请求体，附带 `Authorization: Bearer <credential>`
4. 原样流式返回响应（保留 SSE）

不做转换。不记录请求体。没有 agent 循环。代理只是一个附加凭据的透传通道。

## 未来：更多 OAuth 提供商

适配器系统是可插拔的。添加新的提供商（例如 HuggingFace、GitHub Copilot 的聊天端点、通过 OAuth 的 Anthropic）需要在 `rabbit_cli/proxy/adapters/<provider>.py` 中实现 `UpstreamAdapter` 并在 `adapters/__init__.py` 中注册。协议层面不兼容 OpenAI 的提供商（例如 Anthropic Messages API）需要转换层，这超出了当前形态的范围。
