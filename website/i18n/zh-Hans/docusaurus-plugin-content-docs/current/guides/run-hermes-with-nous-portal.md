---
sidebar_position: 1
title: "通过 Nous Portal 运行 Hermes Agent"
description: "完整操作指南：订阅、配置、切换模型、启用 gateway 工具并验证路由"
---

# 通过 Nous Portal 运行 Hermes Agent

本指南带你从头到尾完成在 [Nous Portal](https://portal.nousresearch.com) 订阅下运行 Hermes Agent 的全过程——从注册账号到验证每个工具的路由是否正确。如果你只想了解 Portal 的概述及订阅内容，请参阅 [Nous Portal 集成页面](/integrations/nous-portal)。本页是操作步骤脚本。

## 前提条件

- 已安装 Hermes Agent（[快速入门](/getting-started/quickstart)）
- 在你正在配置的机器上有可用的浏览器（或 SSH 端口转发——参见 [OAuth over SSH](/guides/oauth-over-ssh)）
- 约 5 分钟时间

你**不需要**：OpenAI 密钥、Anthropic 密钥、Firecrawl 账号、FAL 账号、Browser Use 账号，或任何其他按供应商分配的凭证。这正是 Portal 的意义所在。

## 1. 获取订阅

打开 [portal.nousresearch.com/manage-subscription](https://portal.nousresearch.com/manage-subscription)，注册并选择一个套餐。

已订阅？跳至第 2 步。

## 2. 运行一键配置

```bash
hermes setup --portal
```

这条命令会完成五件事：

1. 打开浏览器跳转至 portal.nousresearch.com 进行 OAuth 登录
2. 将 refresh token 存储至 `~/.hermes/auth.json`
3. 在 `~/.hermes/config.yaml` 中设置 `model.provider: nous`
4. 选择一个默认的 agentic 模型（`anthropic/claude-sonnet-4.6` 或类似模型）
5. 为网页搜索、图像生成、TTS 和浏览器自动化开启 Tool Gateway

命令执行完毕后，你将回到终端，可以直接开始对话。

### 如果我通过 SSH 连接到服务器怎么办？

OAuth 需要浏览器，但 loopback 回调运行在 Hermes 所在的机器上。有两种方案：

```bash
# 方案 A：SSH 端口转发（推荐）
ssh -N -L 8642:127.0.0.1:8642 user@remote-host    # 在本地终端执行
hermes setup --portal                              # 在远程机器上执行，在本地浏览器中打开打印出的 URL

# 方案 B：手动粘贴（适用于 Cloud Shell、Codespaces、EC2 Instance Connect）
hermes auth add nous --type oauth --manual-paste
# 然后重新运行 `hermes setup --portal` 以连接 provider + gateway
```

完整操作说明（包括 ProxyJump 链、mosh/tmux 和 ControlMaster 注意事项）请参阅 [OAuth over SSH / 远程主机](/guides/oauth-over-ssh)。

## 3. 验证配置是否成功

```bash
hermes portal info
```

你应该看到：

```
  Nous Portal
  ───────────
  Auth:    ✓ logged in
  Portal:  https://portal.nousresearch.com
  Model:   ✓ using Nous as inference provider

  Tool Gateway
  ────────────
  Web search & extract  via Nous Portal
  Image generation      via Nous Portal
  Text-to-speech        via Nous Portal
  Browser automation    via Nous Portal
```

如果任何一行显示的不是"via Nous Portal"，或者 auth 行显示"not logged in"，请跳至下方的[故障排查](#troubleshooting)。

## 4. 运行第一次对话

```bash
hermes chat
```

尝试一个同时调用模型和 Tool Gateway 的请求：

```
Hey, search the web for "Hermes Agent release notes" and summarize the top 3 hits.
```

你应该看到 Hermes 调用 `web_search`（通过 gateway 由 Firecrawl 提供支持）并返回摘要。如果搜索正常执行且响应内容合理，说明配置完成——Portal 已端到端连通。

## 5. 选择你实际需要的模型

`hermes setup --portal` 会在设置过程中让你选择模型，但订阅的意义在于可以访问完整的模型目录——随时可在会话中使用 `/model` 切换：

```bash
/model anthropic/claude-sonnet-4.6     # 最佳通用 agentic 模型
/model openai/gpt-5.4                  # 强推理 + 工具调用
/model google/gemini-2.5-pro           # 超大上下文窗口
/model deepseek/deepseek-v3.2          # 高性价比编程模型
/model anthropic/claude-opus-4.6       # 处理复杂问题的重量级模型
```

或者打开选择器浏览：

```bash
/model
```

永久设置不同的默认模型：

```bash
# 在终端中，在任何会话之外执行
hermes config set model.default anthropic/claude-sonnet-4.6
```

### 不要在 agent 任务中使用 Hermes-4

Hermes-4-70B 和 Hermes-4-405B 在 Portal 上以大幅折扣提供，但它们是**对话/推理模型**，并非针对工具调用优化的模型。它们在多步骤 agent 循环中表现不佳。请通过 [Nous Chat](https://chat.nousresearch.com) 将它们用于对话/研究工作，或通过[订阅代理](/user-guide/features/subscription-proxy)从非 agent 工具中使用。对于 Hermes Agent 本身，请坚持使用上述前沿 agentic 模型。

Portal 的[信息页面](https://portal.nousresearch.com/info)也有此说明——这是 Nous 官方指导，并非仅代表 Hermes 一方的意见。

## 6. （可选）自定义 Tool Gateway 路由

gateway 是按工具选择启用的，而非全部开启或全部关闭。如果你已有 Browserbase 账号并希望继续使用，同时将网页搜索和图像生成路由至 Nous，这是支持的：

```bash
hermes tools
# → Web search       → "Nous Subscription"     （推荐）
# → Image generation → "Nous Subscription"     （推荐）
# → Browser          → "Browserbase"           （你自己的密钥）
# → TTS              → "Nous Subscription"     （推荐）
```

使用以下命令验证你的混合配置：

```bash
hermes portal tools
```

你将看到每个工具的路由情况——通过订阅路由的工具显示 `via Nous Portal`，使用你自己密钥的工具显示合作方名称（`browserbase`、`firecrawl` 等）。

## 7. （可选）启用语音模式

由于 Tool Gateway 包含 OpenAI TTS，无需单独的 OpenAI 密钥即可使用[语音模式](/user-guide/features/voice-mode)：

```bash
hermes setup voice
# → 为 TTS 选择 "Nous Subscription"
# → 选择语音转文字后端（本地 faster-whisper 免费，无需配置）
```

之后在任何消息平台会话中（Telegram、Discord、Signal 等），发送语音消息，Hermes 将转录内容、生成回复并以合成语音回复——全部通过你的 Portal 订阅完成。

## 8. （可选）Cron 定时任务与常驻工作流

Portal 订阅对 [cron 定时任务](/user-guide/features/cron)和[批处理](/user-guide/features/batch-processing)的支持方式与交互式对话相同——OAuth refresh token 会自动复用。无需额外配置，直接安排 cron 任务，费用将计入你的订阅。

```bash
hermes cron add "Daily AI news summary" "every day at 9am" \
  "Search the web for top AI news and summarize the 5 most important stories"
```

该 cron 任务无人值守运行，调用模型、网页搜索和摘要生成，全部通过你的 Portal 订阅完成。

## Profiles 与多用户配置

如果你使用 [Hermes profiles](/user-guide/profiles)（例如每个项目单独一套配置），Portal refresh token 会通过共享 token 存储自动在所有 profiles 之间共享。在任意 profile 上登录一次，其余 profiles 会自动获取。

对于多人共用一台机器的团队场景，每个人有自己的 Portal 账号 → 每个 home 目录保存各自的 `~/.hermes/auth.json` → 用户之间不共享 token。这是正确的边界划分。

## 故障排查

### 运行 `hermes setup --portal` 后，`hermes portal info` 显示"not logged in"

OAuth 流程未完成。重新运行：

```bash
hermes portal
```

如果浏览器未打开或回调失败，你可能在远程/无头主机上——参见 [OAuth over SSH](/guides/oauth-over-ssh) 了解端口转发和手动粘贴的解决方案。

### "Model: currently openrouter"（或其他 provider）而非"using Nous as inference provider"

本地配置发生了偏移。OAuth 成功，但 `model.provider` 仍指向其他 provider。修复方法：

```bash
hermes config set model.provider nous
```

或以交互方式：

```bash
hermes model
# 选择 Nous Portal
```

使用 `hermes portal info` 重新验证。

### Tool Gateway 工具显示合作方名称而非"via Nous Portal"

按工具的配置覆盖了 gateway 设置。运行：

```bash
hermes tools
# 对需要通过 gateway 路由的工具选择 "Nous Subscription"
```

部分用户会有意混合使用——例如网页搜索通过 Nous 路由，但浏览器使用自己的 Browserbase 密钥。如果这是有意为之，保持不变即可。如果不是，此命令可修复。

### 会话中途出现"Re-authentication required"

你的 Portal refresh token 已失效（密码更改、手动撤销、会话过期）。该 token 现已在本地被隔离，以防 Hermes 无限重试。重新登录即可：

```bash
hermes auth add nous
```

成功重新登录后，隔离状态会自动解除。

### 我想要的模型不在 `/model` 选择器中

Portal 目录镜像了 OpenRouter 的模型列表（300+ 个）。如果某个模型缺失，尝试直接输入 OpenRouter 风格的 slug：

```bash
/model anthropic/claude-opus-4.6
/model openai/o1-2025-12-17
```

如果某个模型确实不可用，请[提交 issue](https://github.com/NousResearch/hermes-agent/issues)——大多数缺失是我们可以更新的路由配置问题。

### 账单未出现在我的 Portal 账号中

`hermes portal info` 会告诉你是否真的在通过 Portal 路由，还是使用了其他 provider。常见原因：

- `model.provider` 设置为 `openrouter`/`anthropic`/等，而非 `nous`
- OAuth refresh 失败后回退到了其他已配置的 provider
- 存在多个 Hermes profiles，你使用的是错误的那个（检查 `hermes profile current`）

### 想要撤销并重新开始

```bash
hermes auth remove nous       # 清除本地 refresh token
# 然后重新运行 setup，或在 Portal 网页界面取消订阅
```

## 用具体数字说明 Portal 的价值

| 不使用 Portal | 使用 Portal |
|----------------|-------------|
| 1 个 OpenRouter / Anthropic / OpenAI 密钥写入 `.env` | 1 个 OAuth refresh token，无需 `.env` 密钥 |
| 1 个 Firecrawl 密钥用于网页搜索 | 网页搜索通过 gateway 路由 |
| 1 个 FAL 密钥用于图像生成 | 图像生成通过 gateway 路由 |
| 1 个 Browser Use / Browserbase 密钥用于浏览器 | 浏览器通过 gateway 路由 |
| 1 个 OpenAI 密钥用于 TTS / 语音模式 | TTS 通过 gateway 路由 |
| 5 个独立的控制台、充值、发票 | 1 个订阅，1 张发票 |
| 跨机器：复制全部 5 个密钥 | 跨机器：重新 OAuth 一次 |

这就是 Portal 的价值。如果你本来就在使用其中两个以上的后端，订阅费用自然就回来了。

## 另请参阅

- **[Nous Portal 集成页面](/integrations/nous-portal)** — 订阅内容概述
- **[Tool Gateway](/user-guide/features/tool-gateway)** — 每个 gateway 路由工具的完整说明
- **[订阅代理](/user-guide/features/subscription-proxy)** — 在非 Hermes 工具中使用你的 Portal 订阅
- **[语音模式](/user-guide/features/voice-mode)** — 在 Portal 订阅上配置语音对话
- **[OAuth over SSH](/guides/oauth-over-ssh)** — 远程/无头主机登录方案
- **[Profiles](/user-guide/profiles)** — 在多个 Hermes 配置之间共享一个 Portal 登录