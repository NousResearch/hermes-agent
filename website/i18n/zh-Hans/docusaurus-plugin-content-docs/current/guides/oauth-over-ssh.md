---
sidebar_position: 17
title: "SSH / 远程主机上的 OAuth"
description: "当 Hermes 运行在远程机器、容器或跳板机后面时，如何完成基于浏览器的 OAuth（Spotify、MCP 服务器）"
---

# SSH / 远程主机上的 OAuth

部分 Hermes 提供商——**Spotify** 和 **远程 MCP 服务器**（Linear、Sentry、Atlassian、Asana、Figma 等）——使用*回环重定向（loopback redirect）* OAuth 流程。认证服务器将浏览器重定向到 `http://127.0.0.1:<port>/callback`，由 Hermes 启动的小型 HTTP 监听器获取授权码。

当 Hermes 和浏览器在同一台机器上时，这一切运行正常。一旦两者不在同一台机器上就会出问题：你笔记本上的浏览器试图访问**你笔记本**上的 `127.0.0.1`，但监听器绑定的是**远程服务器**上的 `127.0.0.1`。

解决方法是一行 SSH 本地端口转发。对于交互式终端上的 MCP 服务器，通常也可以直接粘贴重定向 URL（无需隧道）。

**xAI Grok OAuth（`xai-oauth`）使用 OAuth 设备代码**，不是回环回调——在任意浏览器中打开打印的验证 URL，Hermes 轮询直到批准即可，无需 SSH 隧道。请参阅 [xAI Grok OAuth](./xai-grok-oauth.md)。

## 快速概览

```bash
# 在你的本地机器（笔记本）上，另开一个终端：
ssh -N -L 43827:127.0.0.1:43827 user@remote-host

# 在远程机器的现有 SSH 会话中：
hermes auth add spotify --no-browser
# → Hermes 打印授权 URL，在笔记本的浏览器中打开。
# → 浏览器重定向到 127.0.0.1:43827/callback，隧道转发到远程监听器，登录完成。
```

Hermes 会在 `Waiting for callback on ...` 一行打印实际绑定的端口——从那里复制。Spotify 默认端口为 `43827`。

## 哪些提供商需要此操作

| 提供商 | 回环端口 | 需要隧道？ |
|----------|---------------|----------------|
| Spotify | `43827`（默认） | 是，当 Hermes 在远程时 |
| MCP 服务器（`auth: oauth`） | 每台服务器自动选择 | 是（或粘贴重定向 URL） |
| `xai-oauth`（Grok SuperGrok） | 不适用 | 否——设备代码流程 |
| `anthropic`（Claude Pro/Max） | 不适用 | 否——粘贴代码流程 |
| `openai-codex`（ChatGPT Plus/Pro） | 不适用 | 否——设备码流程 |
| `minimax`、`nous-portal` | 不适用 | 否——设备码流程 |

如果你的提供商不在表中，则不需要隧道。

## 为什么监听器不能直接绑定 0.0.0.0

Spotify 和大多数 MCP OAuth 服务器会根据白名单验证 `redirect_uri` 参数，并要求回环形式（`http://127.0.0.1:<精确端口>/callback`）。将监听器绑定到 `0.0.0.0` 或使用不同端口会导致认证服务器以 redirect_uri 不匹配为由拒绝请求。SSH 隧道可以端到端保持回环 URI 不变。

## 分步操作：单次 SSH 跳转

### 1. 从本地机器启动隧道

```bash
# Spotify（端口 43827）
ssh -N -L 43827:127.0.0.1:43827 user@remote-host
```

`-N` 表示「不打开远程 shell，仅保持隧道」。登录期间保持此终端运行。

### 2. 在另一个 SSH 会话中运行认证命令

```bash
ssh user@remote-host
hermes auth add spotify --no-browser
```

Hermes 检测到 SSH 会话，跳过自动打开浏览器，并打印授权 URL 以及 `Waiting for callback on http://127.0.0.1:<port>/callback`。

### 3. 在本地浏览器中打开 URL

从远程终端复制授权 URL，粘贴到笔记本的浏览器中。批准同意后，认证服务器重定向到 `http://127.0.0.1:<port>/callback`。浏览器经隧道访问，请求转发到远程监听器，Hermes 打印 `Login successful!`。

看到成功提示后即可关闭隧道（在第一个终端按 Ctrl+C）。

## 通过跳板机

如果通过堡垒机 / 跳板机访问 Hermes，使用 SSH 内置的 `-J`（ProxyJump）：

```bash
ssh -N -L 43827:127.0.0.1:43827 -J jump-user@jump-host user@final-host
```

## 故障排除

### `bind [127.0.0.1]:43827: Address already in use`

笔记本上已有进程占用该端口。结束占用进程后重试 `ssh -L`。

### 等待本地回调超时

重定向未到达远程监听器。确认隧道仍在运行，并使用最新一次 `Waiting for callback on ...` 中的端口（首选端口被占用时 Hermes 可能自动递增）。

## 另请参阅

- [xAI Grok OAuth](./xai-grok-oauth.md)——设备代码；无需 SSH 隧道
- [Spotify（SSH 上运行）](../user-guide/features/spotify.md#running-over-ssh--in-a-headless-environment)
- [原生 MCP 客户端（OAuth 部分）](../user-guide/features/mcp.md#oauth-authenticated-http-servers)
