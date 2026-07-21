---
sidebar_position: 16
title: "xAI Grok OAuth（SuperGrok / X Premium+）"
description: "使用 SuperGrok 或 X Premium+ 订阅登录，在 Hermes Agent 中使用 Grok 模型——无需 API 密钥"
---

# xAI Grok OAuth（SuperGrok / X Premium+）

Hermes Agent 通过基于浏览器的 OAuth 登录流程支持 xAI Grok，认证服务器为 [accounts.x.ai](https://accounts.x.ai)，支持 **SuperGrok 订阅**（[grok.com](https://x.ai/grok)）或 **X Premium+ 订阅**（已关联的 X 账号）。无需 `XAI_API_KEY`——登录一次后，Hermes 会在后台自动刷新会话。

当你使用拥有 Premium+ 的 X 账号登录时，xAI 会自动将订阅状态关联到你的 xAI 会话，因此 OAuth 流程与直接 SuperGrok 订阅者的体验完全相同。

该传输层复用 `codex_responses` 适配器（xAI 暴露了 Responses 风格的端点），因此推理、工具调用、流式传输和 prompt（提示词）缓存无需任何适配器改动即可正常工作。

同一 OAuth bearer token 也会被 Hermes 中所有直连 xAI 的功能复用——TTS、图像生成、视频生成和转录——因此单次登录即可覆盖全部四项功能。

## 概览

| 项目 | 值 |
|------|-------|
| Provider ID | `xai-oauth` |
| 显示名称 | xAI Grok OAuth (SuperGrok / X Premium+) |
| 认证类型 | 浏览器 OAuth 2.0 PKCE（回环回调） |
| 传输层 | xAI Responses API（`codex_responses`） |
| 默认模型 | `grok-build-0.1` |
| 端点 | `https://api.x.ai/v1` |
| 认证服务器 | `https://accounts.x.ai` |
| 需要环境变量 | 否（此 provider 不使用 `XAI_API_KEY`） |
| 订阅要求 | [SuperGrok](https://x.ai/grok) 或 [X Premium+](https://x.com/i/premium_sign_up)——见下方说明 |

## 前提条件

- Python 3.9+
- 已安装 Hermes Agent
- 你的 xAI 账号拥有有效的 **SuperGrok** 订阅，**或**你登录所用的 X 账号拥有 **X Premium+** 订阅（xAI 会自动关联订阅）
- 本地机器上有可用的浏览器（远程会话可使用 `--no-browser`）

:::warning xAI 可能按套餐限制 OAuth API 访问
xAI 的后端对 OAuth API 接口维护自己的白名单，已有记录显示即使应用内订阅处于激活状态，标准 SuperGrok 订阅者也会收到 `HTTP 403`（见 issue [#26847](https://github.com/NousResearch/hermes-agent/issues/26847)）。如果浏览器中 OAuth 登录成功但推理返回 403，请设置 `XAI_API_KEY` 并切换到 API 密钥路径（`provider: xai`）——该接口目前不受相同限制。
:::

## 快速开始

```bash
# 启动 provider 和模型选择器
hermes model
# → 从 provider 列表中选择 "xAI Grok OAuth (SuperGrok / X Premium+)"
# → Hermes 在浏览器中打开 accounts.x.ai
# → 在浏览器中批准访问
# → 选择模型（grok-build-0.1 在列表顶部）
# → 开始对话

hermes
```

首次登录后，凭据存储在 `~/.hermes/auth.json` 中，并在过期前自动刷新。

## 手动登录

你可以不经过模型选择器直接触发登录：

```bash
hermes auth add xai-oauth
```

### 远程 / 无头会话

在没有浏览器的服务器、容器或 SSH 会话中，Hermes 会检测到远程环境并打印授权 URL，而不是打开浏览器。

**重要：** 回环监听器仍在远程机器的 `127.0.0.1:56121` 上运行。xAI 的重定向需要到达*该*监听器，因此在你的笔记本上打开 URL 会失败（`Could not establish connection. We couldn't reach your app.`），除非你转发端口：

```bash
# 在本地机器的另一个终端中：
ssh -N -L 56121:127.0.0.1:56121 user@remote-host

# 然后在远程机器的 SSH 会话中：
hermes auth add xai-oauth --no-browser
# 在本地浏览器中打开打印出的授权 URL。
```

通过跳板机 / 堡垒机：添加 `-J jump-user@jump-host`。

完整步骤（包括 ProxyJump 链、mosh/tmux 和 ControlMaster 注意事项）请参阅 [OAuth over SSH / Remote Hosts](./oauth-over-ssh.md)。

### 仅限浏览器的远程环境（Cloud Shell、Codespaces、EC2 Instance Connect）

如果你没有常规 SSH 客户端（例如在 GCP Cloud Shell、GitHub Codespaces、AWS EC2 Instance Connect、Gitpod 或其他基于浏览器的控制台中运行 Hermes），上述 `ssh -L` 方案不可用。请改用 `--manual-paste`——Hermes 跳过回环监听器，让你直接从浏览器粘贴失败的回调 URL：

```bash
hermes auth add xai-oauth --manual-paste
# 或通过模型选择器：
hermes model --manual-paste
```

完整操作说明请参阅 [OAuth over SSH / Remote Hosts](./oauth-over-ssh.md#browser-only-remote-cloud-shell--codespaces--ec2-instance-connect)。此为 [#26923](https://github.com/NousResearch/hermes-agent/issues/26923) 的回归修复。

## 登录流程说明

1. Hermes 在浏览器中打开 `accounts.x.ai`。
2. 你登录（或确认现有会话）并批准访问。
3. xAI 重定向回 Hermes，token 保存到 `~/.hermes/auth.json`。
4. 此后，Hermes 在后台刷新 access token——你将保持登录状态，直到执行 `hermes auth logout xai-oauth` 或在 xAI 账号设置中撤销访问。

## 检查登录状态

```bash
hermes doctor
```

`◆ Auth Providers` 部分将显示每个 provider 的当前状态，包括 `xai-oauth`。

## 切换模型

```bash
hermes model
# → 选择 "xAI Grok OAuth (SuperGrok / X Premium+)"
# → 从模型列表中选择（grok-build-0.1 固定在顶部）
```

或直接设置模型：

```bash
hermes config set model.default grok-build-0.1
hermes config set model.provider xai-oauth
```

## 配置参考

登录后，`~/.hermes/config.yaml` 将包含：

```yaml
model:
  default: grok-build-0.1
  provider: xai-oauth
  base_url: https://api.x.ai/v1
```

### Provider 别名

以下所有别名均解析为 `xai-oauth`：

```bash
hermes --provider xai-oauth        # 规范名称
hermes --provider grok-oauth       # 别名
hermes --provider x-ai-oauth       # 别名
hermes --provider xai-grok-oauth   # 别名
```

## 直连 xAI 工具（TTS / 图像 / 视频 / 转录 / X 搜索）

通过 OAuth 登录后，每个直连 xAI 的工具都会自动复用同一 bearer token——**无需单独配置**，除非你更倾向于使用 API 密钥。

为每个工具选择后端：

```bash
hermes tools
# → Text-to-Speech       → "xAI TTS"
# → Image Generation     → "xAI Grok Imagine (image)"
# → Video Generation     → "xAI Grok Imagine"
# → X (Twitter) Search   → "xAI Grok OAuth (SuperGrok / X Premium+)"
```

如果 OAuth token 已存储，选择器会确认并跳过凭据提示。如果既没有 OAuth 也没有设置 `XAI_API_KEY`，选择器会提供三选一菜单：OAuth 登录、粘贴 API 密钥或跳过。

:::note 视频生成默认关闭
`video_gen` 工具集默认禁用。在 `hermes tools` → `🎬 Video Generation`（按空格键）中启用后，agent 才能调用 `video_generate`。否则 agent 可能回退到内置的 ComfyUI 技能，该技能同样标记为视频生成。
:::

:::note 配置 xAI 凭据后 X 搜索自动启用
只要配置了 xAI 凭据（SuperGrok / X Premium+ OAuth token 或 `XAI_API_KEY`），`x_search` 工具集就会自动启用。如不需要，请通过 `hermes tools` → `🐦 X (Twitter) Search`（按空格键）显式禁用。该工具通过 xAI 内置的 `x_search` Responses API 路由——支持 **SuperGrok / X Premium+ OAuth 登录**或付费 `XAI_API_KEY`，两者同时配置时优先使用 OAuth（消耗订阅配额而非 API 费用）。未配置任何 xAI 凭据时，无论工具集是否启用，工具 schema 都对模型隐藏。
:::

### 模型

| 工具 | 模型 | 说明 |
|------|-------|-------|
| 对话 | `grok-build-0.1` | 默认；通过 OAuth 登录时自动选择 |
| 对话 | `grok-4.3` | 之前的默认 |
| 对话 | `grok-4.20-0309-reasoning` | 推理变体 |
| 对话 | `grok-4.20-0309-non-reasoning` | 非推理变体 |
| 对话 | `grok-4.20-multi-agent-0309` | 多 agent 变体 |
| 图像 | `grok-imagine-image` | 默认；约 5–10 秒 |
| 图像 | `grok-imagine-image-quality` | 更高保真度；约 10–20 秒 |
| 视频 | `grok-imagine-video` | 文本转视频 |
| 视频 | `grok-imagine-video-1.5-preview` | 图像转视频；日期别名 `grok-imagine-video-1.5-2026-05-30` |
| TTS | （默认音色） | xAI `/v1/tts` 端点 |

对话模型目录从磁盘上的 `models.dev` 缓存实时获取；缓存刷新后，新的 xAI 模型会自动出现。`grok-build-0.1` 始终固定在列表顶部。

## 环境变量

| 变量 | 作用 |
|----------|--------|
| `XAI_BASE_URL` | 覆盖默认的 `https://api.x.ai/v1` 端点（极少需要）。 |

要将 xAI 设为活跃 provider，请在 `config.yaml` 中设置 `model.provider: xai-oauth`（使用 `hermes setup` 进行引导配置），或在单次调用时传入 `--provider xai-oauth`。

## 故障排查

### Token 过期——未自动重新登录

Hermes 在每次会话前刷新 token，并在收到 401 时响应式地再次刷新。如果刷新因 `invalid_grant` 失败（刷新 token 被撤销或账号已轮换），Hermes 会显示类型化的重新认证消息，而不是崩溃。

当刷新失败是终态时（HTTP 4xx、`invalid_grant`、授权被撤销等），Hermes 将刷新 token 标记为失效并在本地隔离——后续调用跳过注定失败的刷新尝试，而不是反复重放同一个 401。agent 显示一条"需要重新认证"消息，并在你再次登录前保持等待。

**修复方法：** 再次运行 `hermes auth add xai-oauth` 开始全新登录。下次成功交换后隔离状态自动清除。

### 授权超时

回环监听器有有限的过期窗口（默认 180 秒）。如果你未在时限内批准登录，Hermes 会抛出超时错误。

**修复方法：** 重新运行 `hermes auth add xai-oauth`（或 `hermes model`）。流程重新开始。

### State 不匹配（可能的 CSRF）

Hermes 检测到授权服务器返回的 `state` 值与发送的不匹配。

**修复方法：** 重新运行登录。如果问题持续，检查是否有代理或重定向在修改 OAuth 响应。

### 从远程服务器登录

在 SSH 或容器会话中，Hermes 打印授权 URL 而不是打开浏览器。回环回调监听器仍绑定在远程主机的 `127.0.0.1:56121`——你笔记本上的浏览器无法访问它，除非进行 SSH 本地端口转发：

```bash
# 本地机器，另一个终端：
ssh -N -L 56121:127.0.0.1:56121 user@remote-host

# 远程机器：
hermes auth add xai-oauth --no-browser
```

完整操作说明（跳板机、mosh/tmux、端口冲突）：[OAuth over SSH / Remote Hosts](./oauth-over-ssh.md)。

### 登录成功后 HTTP 403（套餐 / 权限问题）

浏览器中 OAuth 完成，token 已保存，但推理或 token 刷新返回 `HTTP 403`，消息类似于 *"The caller does not have permission to execute the specified operation"*。

这**不是** token 过期问题——重新运行 `hermes model` 不会改变结果。xAI 的后端已被观察到将 OAuth API 访问限制在特定 SuperGrok 套餐，即使应用内订阅处于激活状态（issue [#26847](https://github.com/NousResearch/hermes-agent/issues/26847)）。

**修复方法：** 设置 `XAI_API_KEY` 并切换到 API 密钥路径：

```bash
export XAI_API_KEY=xai-...
hermes config set model.provider xai
```

或在 [x.ai/grok](https://x.ai/grok) 升级订阅（如果必须使用 OAuth 路径）。

### 运行时出现"No xAI credentials found"错误

auth 存储中没有 `xai-oauth` 条目，也未设置 `XAI_API_KEY`。你尚未登录，或凭据文件已被删除。

**修复方法：** 运行 `hermes model` 并选择 xAI Grok OAuth provider，或运行 `hermes auth add xai-oauth`。

## 退出登录

删除所有已存储的 xAI Grok OAuth 凭据：

```bash
hermes auth logout xai-oauth
```

这会清除 `auth.json` 中的单例 OAuth 条目以及 `xai-oauth` 的所有凭据池行。如果只想删除单个池条目，请使用 `hermes auth remove xai-oauth <index|id|label>`（运行 `hermes auth list xai-oauth` 查看列表）。

## 另请参阅

- [OAuth over SSH / Remote Hosts](./oauth-over-ssh.md) — 如果 Hermes 与浏览器不在同一台机器上，必读
- [AI Providers 参考](../integrations/providers.md)
- [环境变量](../reference/environment-variables.md)
- [配置](../user-guide/configuration.md)
- [语音与 TTS](../user-guide/features/tts.md)
