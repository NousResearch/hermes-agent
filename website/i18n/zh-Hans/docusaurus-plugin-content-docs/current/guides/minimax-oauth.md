---
sidebar_position: 15
title: "MiniMax OAuth"
description: "通过浏览器 OAuth 登录 MiniMax 并在 Hermes Agent 中使用 MiniMax-M2.7 模型 — 无需 API key"
---

# MiniMax OAuth

Hermes Agent 支持通过基于浏览器的 OAuth 登录流程使用 **MiniMax**，使用与 [MiniMax 门户](https://www.minimax.io) 相同的凭证。无需 API key 或信用卡 — 登录一次，Hermes 自动刷新会话。

传输层重用 `anthropic_messages` 适配器（MiniMax 在 `/anthropic` 暴露 Anthropic Messages 兼容端点），因此所有现有工具调用、流式传输和上下文功能无需任何适配器更改即可工作。

## 概览

| 项目 | 值 |
|------|-------|
| 提供商 ID | `minimax-oauth` |
| 显示名称 | MiniMax (OAuth) |
| 认证类型 | 浏览器 OAuth（PKCE device-code 流程） |
| 传输 | Anthropic Messages 兼容（`anthropic_messages`） |
| 模型 | `MiniMax-M2.7`、`MiniMax-M2.7-highspeed` |
| 全球端点 | `https://api.minimax.io/anthropic` |
| 中国端点 | `https://api.minimaxi.com/anthropic` |
| 需要环境变量 | 否（`MINIMAX_API_KEY` **不**用于此提供商） |

## 前置条件

- Python 3.9+
- 已安装 Hermes Agent
- [minimax.io](https://www.minimax.io)（全球）或 [minimaxi.com](https://www.minimaxi.com)（中国）的 MiniMax 账户
- 本地有浏览器可用（或对远程会话使用 `--no-browser`）

## 快速开始

```bash
# 启动提供商和模型选择器
hermes model
# → 从提供商列表中选择 "MiniMax (OAuth)"
# → Hermes 在浏览器中打开 MiniMax 授权页面
# → 在浏览器中批准访问
# → 选择模型（MiniMax-M2.7 或 MiniMax-M2.7-highspeed）
# → 开始聊天

hermes
```

首次登录后，凭证存储在 `~/.hermes/auth.json`，每个会话前自动刷新。

## 手动登录

无需通过模型选择器即可触发登录：

```bash
hermes auth add minimax-oauth
```

### 中国区域

如果你的账户在中国平台（`minimaxi.com`），传入 `--region cn`：

```bash
hermes auth add minimax-oauth --region cn
```

### 远程/无头会话

在没有浏览器的服务器或容器上：

```bash
hermes auth add minimax-oauth --no-browser
```

Hermes 会打印验证 URL 和用户码 — 在任何设备上打开 URL，提示时输入代码。

## OAuth 流程

Hermes 针对 MiniMax OAuth 端点实现 PKCE device-code 流程：

1. Hermes 生成 PKCE 验证器/挑战对和随机状态值。
2. POST 到 `{base_url}/oauth/code`，附带挑战并接收 `user_code` 和 `verification_uri`。
3. 浏览器打开 `verification_uri`。如果提示，输入 `user_code`。
4. Hermes 轮询 `{base_url}/oauth/token` 直到收到 token（或截止时间过去）。
5. Token（`access_token`、`refresh_token`、过期时间）保存到 `~/.hermes/auth.json` 的 `minimax-oauth` 键下。

Token 刷新（标准 OAuth `refresh_token` grant）在每次会话开始时 access token 距离过期 60 秒内时自动运行。

## 检查登录状态

```bash
hermes doctor
```

`◆ Auth Providers` 部分会显示：

```
✓ MiniMax OAuth  (已登录, region=global)
```

如果未登录：

```
⚠ MiniMax OAuth  (未登录)
```

## 切换模型

```bash
hermes model
# → 选择 "MiniMax (OAuth)"
# → 从模型列表中选择
```

或直接设置模型：

```bash
hermes config set model MiniMax-M2.7
hermes config set provider minimax-oauth
```

## 配置参考

登录后，`~/.hermes/config.yaml` 会包含类似条目：

```yaml
model:
  default: MiniMax-M2.7
  provider: minimax-oauth
  base_url: https://api.minimax.io/anthropic
```

### `--region` 参数

| 值 | 门户 | 推理端点 |
|-------|--------|-------------------|
| `global`（默认） | `https://api.minimax.io` | `https://api.minimax.io/anthropic` |
| `cn` | `https://api.minimaxi.com` | `https://api.minimaxi.com/anthropic` |

### 提供商别名

以下所有都解析为 `minimax-oauth`：

```bash
hermes --provider minimax-oauth    # 规范
hermes --provider minimax-portal   # 别名
hermes --provider minimax-global   # 别名
hermes --provider minimax_oauth    # 别名（下划线形式）
```

## 环境变量

`minimax-oauth` 提供商**不使用** `MINIMAX_API_KEY` 或 `MINIMAX_BASE_URL`。这些变量仅用于基于 API key 的 `minimax` 和 `minimax-cn` 提供商。

| 变量 | 作用 |
|----------|--------|
| `MINIMAX_API_KEY` | 仅被 `minimax` 提供商使用 — 对 `minimax-oauth` 忽略 |
| `MINIMAX_CN_API_KEY` | 仅被 `minimax-cn` 提供商使用 — 对 `minimax-oauth` 忽略 |

在运行时强制使用 `minimax-oauth` 提供商：

```bash
HERMES_INFERENCE_PROVIDER=minimax-oauth hermes
```

## 模型

| 模型 | 适用场景 |
|-------|----------|
| `MiniMax-M2.7` | 长上下文推理、复杂工具调用 |
| `MiniMax-M2.7-highspeed` | 更低延迟、轻量级任务、辅助调用 |

两个模型都支持最高 200,000 tokens 上下文。

当 `minimax-oauth` 作为主要提供商时，`MiniMax-M2.7-highspeed` 也自动用作视觉和委托任务的辅助模型。

## 故障排除

### Token 过期 — 未自动重新登录

如果 access token 距离过期 60 秒内，Hermes 在每次会话开始时刷新。如果 access token 已过期（例如，长时间离线后），下一次请求时自动刷新。如果刷新失败并出现 `refresh_token_reused` 或 `invalid_grant`，Hermes 标记会话需要重新登录。

**修复：** 再次运行 `hermes auth add minimax-oauth` 开始新的登录。

### 授权超时

device-code 流程有有限的有效期窗口。如果未及时批准登录，Hermes 抛出超时错误。

**修复：** 重新运行 `hermes auth add minimax-oauth`（或 `hermes model`）。流程重新开始。

### 状态不匹配（可能的 CSRF）

Hermes 检测到授权服务器返回的 `state` 值与发送的不匹配。

**修复：** 重新运行登录。如果持续出现，检查修改 OAuth 响应的代理或重定向。

### 从远程服务器登录

如果 `hermes` 无法打开浏览器窗口，使用 `--no-browser`：

```bash
hermes auth add minimax-oauth --no-browser
```

Hermes 打印 URL 和代码。在任何设备上打开 URL 并在那里完成流程。

### 运行时出现"未登录 MiniMax OAuth"错误

auth 存储中没有 `minimax-oauth` 的凭证。你尚未登录，或凭证文件被删除。

**修复：** 运行 `hermes model` 并选择 MiniMax (OAuth)，或运行 `hermes auth add minimax-oauth`。

## 登出

删除存储的 MiniMax OAuth 凭证：

```bash
hermes auth remove minimax-oauth
```

## 另见

- [AI 提供商参考](../integrations/providers.md)
- [环境变量](../reference/environment-variables.md)
- [配置](../user-guide/configuration.md)
- [hermes doctor](../reference/cli-commands.md)
