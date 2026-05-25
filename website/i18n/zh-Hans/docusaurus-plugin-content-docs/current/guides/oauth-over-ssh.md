---
sidebar_position: 17
title: "SSH / 远程主机上的 OAuth"
description: "当 Hermes 运行在远程机器、容器或跳板机后面时，如何完成基于浏览器的 OAuth（xAI、Spotify）"
---

# SSH / 远程主机上的 OAuth

部分 Hermes 提供商——目前是 **xAI Grok OAuth** 和 **Spotify**——使用*回环重定向（loopback redirect）* OAuth 流程。认证服务器（xAI、Spotify）将浏览器重定向到 `http://127.0.0.1:<port>/callback`，由 `hermes auth ...` 命令启动的一个小型 HTTP 监听器来获取授权码。

当 Hermes 和浏览器在同一台机器上时，这一切运行正常。一旦两者不在同一台机器上就会出问题：你笔记本上的浏览器试图访问**你笔记本**上的 `127.0.0.1`，但监听器绑定的是**远程服务器**上的 `127.0.0.1`。

解决方法是一行 SSH 本地端口转发——**或者**，当你没有真正的 SSH 客户端时（GCP Cloud Shell、GitHub Codespaces、EC2 Instance Connect、Gitpod、基于浏览器的 Web IDE），使用 [#26923](https://github.com/NousResearch/hermes-agent/issues/26923) 中引入的新 `--manual-paste` 标志。

## 快速概览

```bash
# 在你的本地机器（笔记本）上，另开一个终端：
ssh -N -L 56121:127.0.0.1:56121 user@remote-host

# 在远程机器的现有 SSH 会话中：
hermes auth add xai-oauth --no-browser
# → Hermes 打印一个授权 URL，在笔记本的浏览器中打开它。
# → 浏览器重定向到 127.0.0.1:56121/callback，隧道将请求转发
#   到远程监听器，登录完成。
```

`56121` 是 xAI OAuth 使用的端口。Spotify 请将其替换为 `43827`。Hermes 会在 `Waiting for callback on ...` 这一行打印它实际绑定的端口——从那里复制。

## 仅限浏览器的远程环境（Cloud Shell / Codespaces / EC2 Instance Connect）

如果你没有常规的 SSH 客户端——例如你在 GCP Cloud Shell、GitHub Codespaces、AWS EC2 Instance Connect、Gitpod 或其他基于浏览器的控制台中运行 Hermes——上述 SSH 隧道不可用。请改用 `--manual-paste`：

```bash
hermes auth add xai-oauth --manual-paste
# → Hermes 打印一个授权 URL，在笔记本的浏览器中打开它。
# → 在浏览器中批准。重定向到 127.0.0.1:56121/callback 会加载失败
#   ——这是预期行为。
# → 从失败页面的地址栏复制完整 URL。
# → 在终端的 "Callback URL:" 提示处粘贴。
```

同样的标志也适用于集成模型选择器的 `hermes model --manual-paste`。如果不想粘贴完整 URL，也可以只接受裸的 `?code=...&state=...` 查询片段。

Hermes 对两种路径使用**相同的 PKCE verifier、state 和 nonce**，因此上游 OAuth 流程在字节层面完全一致——`--manual-paste` 纯粹是回调跳转的传输方式变更，不会降低安全性。

## 哪些提供商需要此操作

| 提供商 | 回环端口 | 需要隧道？ |
|----------|---------------|----------------|
| `xai-oauth`（Grok SuperGrok） | `56121` | 是，当 Hermes 在远程时 |
| Spotify | `43827` | 是，当 Hermes 在远程时 |
| `anthropic`（Claude Pro/Max） | 不适用 | 否——粘贴代码流程 |
| `openai-codex`（ChatGPT Plus/Pro） | 不适用 | 否——设备码流程 |
| `minimax`、`nous-portal` | 不适用 | 否——设备码流程 |

如果你的提供商不在表中，则不需要隧道。

## 为什么监听器不能直接绑定 0.0.0.0

xAI 和 Spotify 都会根据白名单验证 `redirect_uri` 参数。两者都要求回环形式（`http://127.0.0.1:<exact-port>/callback`）。将监听器绑定到 `0.0.0.0` 或不同端口会导致认证服务器以 redirect_uri 不匹配为由拒绝请求。SSH 隧道可以端到端保持回环 URI 不变。

## 分步说明：单跳 SSH

### 1. 从本地机器启动隧道

```bash
# xAI Grok OAuth（端口 56121）
ssh -N -L 56121:127.0.0.1:56121 user@remote-host

# 或 Spotify（端口 43827）
ssh -N -L 43827:127.0.0.1:43827 user@remote-host
```

`-N` 表示"不打开远程 shell，只保持隧道开启"。在登录期间保持此终端运行。

### 2. 在另一个 SSH 会话中运行认证命令

```bash
ssh user@remote-host
hermes auth add xai-oauth --no-browser
# 或 Spotify：
# hermes auth add spotify --no-browser
```

Hermes 检测到 SSH 会话后，跳过自动打开浏览器，打印授权 URL 以及 `Waiting for callback on http://127.0.0.1:<port>/callback` 这一行。

### 3. 在本地浏览器中打开 URL

从远程终端复制授权 URL，粘贴到笔记本的浏览器中。批准同意页面。认证服务器重定向到 `http://127.0.0.1:<port>/callback`。浏览器访问隧道，请求被转发到远程监听器，Hermes 打印 `Login successful!`。

看到成功提示后，可以关闭隧道（在第一个终端按 Ctrl+C）。

## 分步说明：通过跳板机

如果你通过堡垒机 / 跳板机访问 Hermes，使用 SSH 内置的 `-J`（ProxyJump）：

```bash
ssh -N -L 56121:127.0.0.1:56121 -J jump-user@jump-host user@final-host
```

这会通过跳板机链式建立 SSH 连接，而不会将回环端口暴露在跳板机上。你笔记本上的本地 `127.0.0.1:56121` 直接隧道到最终远程主机上的 `127.0.0.1:56121`。

对于不支持 `-J` 的旧版 OpenSSH，完整写法为：

```bash
ssh -N \
    -o "ProxyCommand=ssh -W %h:%p jump-user@jump-host" \
    -L 56121:127.0.0.1:56121 \
    user@final-host
```

## Mosh、tmux、ssh ControlMaster

隧道是底层 SSH 连接的属性。如果你在 mosh 会话中的 `tmux` 里运行 Hermes，mosh 的漫游不会携带 `-L` 转发。**单独**开一个普通 SSH 会话**仅用于** `-L` 隧道——这个连接必须在整个认证流程期间保持存活。你的交互式 mosh/tmux 会话可以继续正常运行 Hermes。

如果你使用 `ssh -o ControlMaster=auto`，多路复用连接上的端口转发共享主连接的生命周期。如果隧道未能建立，重启主连接：

```bash
ssh -O exit user@remote-host
ssh -N -L 56121:127.0.0.1:56121 user@remote-host
```

## 故障排查

### `bind [127.0.0.1]:56121: Address already in use`

你笔记本上已有某个程序占用了该端口。可能是上一个隧道没有正常关闭，或者本地也有一个 Hermes 在监听。找到并终止占用进程：

```bash
# macOS / Linux
lsof -iTCP:56121 -sTCP:LISTEN
kill <PID>
```

然后重试 `ssh -L` 命令。

### "Could not establish connection. We couldn't reach your app."（xAI）

当 xAI 重定向到 `127.0.0.1:<port>/callback` 未能到达监听器时，xAI 的授权页面会显示此错误。可能是隧道未运行、端口错误，或者你使用的是 Hermes 上一次运行时打印的端口（如果首选端口被占用，端口可能会自动递增——始终以最新的 `Waiting for callback on ...` 行为准）。

### `xAI authorization timed out waiting for the local callback`

与上述原因相同——重定向从未返回。检查隧道是否仍然存活（`ssh -N` 不显示输出，查看启动它的终端），必要时重启，然后重新运行 `hermes auth add xai-oauth --no-browser`。

### Token 写入了错误的 `~/.hermes`

Token 写入运行 `hermes auth add ...` 的 Linux 用户目录下。如果你的网关 / systemd 服务以不同用户（如 `root` 或专用的 `hermes` 用户）运行，请以**该**用户身份进行认证，使 token 写入其 `~/.hermes/auth.json`。使用 `sudo -u hermes -i` 或等效命令。

## 另请参阅

- [xAI Grok OAuth](./xai-grok-oauth.md)
- [Spotify（`通过 SSH 运行`）](../user-guide/features/spotify.md#running-over-ssh--in-a-headless-environment)
- [SSH `-J` / ProxyJump（man 手册）](https://man.openbsd.org/ssh#J)