---
sidebar_position: 1
title: "Nous Portal"
description: "一个订阅，300+ 前沿模型，Tool Gateway，以及 Nous Chat —— 运行 Hermes Agent 的推荐方式"
---

# Nous Portal

[Nous Portal](https://portal.nousresearch.com) 是 Nous Research 的统一订阅网关，也是**运行 Hermes Agent 的推荐方式**。一次 OAuth 登录，即可替代原本需要手动配置的各模型厂商独立账号、API 密钥和计费关系。

如果你只有时间配置一件事，就配置这个。最快路径：

```bash
hermes setup --portal
```

这条命令会完成 Portal OAuth 认证，让你选择一个 Nous 模型，在 `config.yaml` 中将 Nous 设为推理提供商，并开启 Tool Gateway。完成后即可立即运行 `hermes chat`。

还没有订阅？前往 [portal.nousresearch.com/manage-subscription](https://portal.nousresearch.com/manage-subscription) 注册，然后回来运行上面的命令。

## 订阅包含的内容

### 300+ 前沿模型，统一账单

Portal 代理了来自整个生态系统的精选 agentic 模型目录——统一计入你的 Nous 订阅，而非每个厂商单独充值。

| 系列 | 模型 |
|--------|--------|
| **Anthropic Claude** | Opus、Sonnet、Haiku（4.x 系列） |
| **OpenAI** | GPT-5.4、o 系列推理模型 |
| **Google Gemini** | 2.5 Pro、2.5 Flash |
| **DeepSeek** | DeepSeek V3.2、DeepSeek-R1 |
| **Qwen** | Qwen3 系列、Qwen Coder |
| **Kimi / Moonshot** | Kimi-K2、Kimi-Latest |
| **GLM / Zhipu** | GLM-4.6、GLM-4-Plus |
| **MiniMax** | M2.7、M1 |
| **xAI** | Grok-4、Grok-3 |
| **Hermes** | Hermes-4-70B、Hermes-4-405B（对话，见[下方说明](#a-note-on-hermes-4)） |
| **+ 其他所有模型** | 240+ 额外模型——完整的 agentic 前沿生态 |

底层路由通过 OpenRouter 实现，因此模型可用性和故障转移行为与使用 OpenRouter 密钥一致——只是计费走你的 Nous 订阅。在会话中途用 `/model` 即可在 Claude Sonnet 4.6（适合代码）和 Gemini 2.5 Pro（适合长上下文）之间切换——无需新凭证，无需充值，不会遇到余额为零的意外报错。

### Nous Tool Gateway

同一订阅还解锁了 [Tool Gateway](/user-guide/features/tool-gateway)，将 Hermes Agent 的工具调用路由至 Nous 托管的基础设施。五个后端，一次登录：

| 工具 | 合作方 | 功能说明 |
|------|---------|--------------|
| **网页搜索与抓取** | Firecrawl | Agent 级搜索与整页内容提取。无需 Firecrawl API 密钥，无需管理速率限制。 |
| **图像生成** | FAL | 单一端点下的九个模型：FLUX 2 Klein 9B、FLUX 2 Pro、Z-Image Turbo、Nano Banana Pro（Gemini 3 Pro Image）、GPT Image 1.5、GPT Image 2、Ideogram V3、Recraft V4 Pro、Qwen Image。 |
| **文字转语音** | OpenAI TTS | 无需独立 OpenAI 密钥的高质量 TTS。在各消息平台上启用[语音模式](/user-guide/features/voice-mode)。 |
| **云端浏览器自动化** | Browser Use | 用于 `browser_navigate`、`browser_click`、`browser_type`、`browser_vision` 的无头 Chromium 会话。无需 Browserbase 账号。 |
| **云端终端沙箱** | Modal | 用于代码执行的无服务器终端沙箱（可选附加项）。 |

不使用 gateway 的话，接入上述每项服务意味着：一个 Firecrawl 账号、一个 FAL 账号、一个 Browser Use 账号、一个 OpenAI 密钥、一个 Modal 账号——五次独立注册、五个独立控制台、五套独立充值流程。使用 gateway 后，所有内容通过一个订阅统一路由。

你也可以只启用特定的 gateway 工具（例如只开启网页搜索，不开启图像生成）——详见下方[将 gateway 与自有后端混用](#mixing-the-gateway-with-your-own-backends)。

### Nous Chat

你的 Portal 账号同样覆盖 [chat.nousresearch.com](https://chat.nousresearch.com)——Nous Research 的网页对话界面，使用相同的模型目录。适合离开终端时使用，或用于非 agent 的普通对话场景。

### 凭证不落入 dotfiles

由于所有请求都通过一个经 OAuth 认证的 Portal 会话路由，你不会积累一个包含十几个长期 API 密钥的 `.env` 文件。磁盘上唯一的凭证是 `~/.hermes/auth.json` 中的 refresh token（刷新令牌），Hermes 会在每次请求时从中生成短期 JWT——详见下方[令牌处理](#token-handling)。

### 跨平台一致性

[原生 Windows](/user-guide/windows-native) 上，逐个配置 API 密钥是其最大痛点——在 Windows 上分别安装 Firecrawl 账号、FAL 账号、Browser Use 账号、OpenAI 密钥，是整个 agent 配置过程中摩擦最高的部分。Portal 订阅消除了这一问题：一次 OAuth 覆盖模型和所有 gateway 工具，Windows 用户无需手动配置四个后端，即可获得与 macOS/Linux 相同的体验。

## 关于 Hermes 4 的说明

Nous Research 自家的 **Hermes 4** 系列（Hermes-4-70B、Hermes-4-405B）通过 Portal 提供，享有大幅折扣。这些是**前沿混合推理对话模型**——在数学、科学、指令遵循、schema 遵从、角色扮演和长文写作方面表现出色。

但**不建议在 Hermes Agent 内部使用它们**。Hermes 4 针对对话和推理进行了调优，而非 agent 所依赖的高频工具调用循环。请将它们用于 [Nous Chat](https://chat.nousresearch.com)、研究工作流，或通过[订阅代理](/user-guide/features/subscription-proxy)从其他工具调用——但在 agent 场景下，请从目录中选择前沿 agentic 模型：

```bash
/model anthropic/claude-sonnet-4.6     # 最佳通用 agentic 模型
/model openai/gpt-5.4                  # 强推理 + 工具调用
/model google/gemini-2.5-pro           # 超大上下文窗口
/model deepseek/deepseek-v3.2          # 高性价比代码模型
```

Portal 自身的[模型信息页](https://portal.nousresearch.com/info)也有相同警告，因此这不是 Hermes 侧的主观意见——这是 Nous Research 的官方指导。

## 配置

### 全新安装——一条命令

```bash
hermes setup --portal
```

一次性完成全部配置：

1. 打开浏览器跳转至 portal.nousresearch.com 进行 OAuth 登录
2. 将 refresh token 存储至 `~/.hermes/auth.json`
3. 让你从精选列表中选择一个 Nous 模型（也可跳过以保留当前模型）
4. 在 `~/.hermes/config.yaml` 中将 Nous 设为推理提供商（当你选择模型时）
5. 开启 Tool Gateway（网页、图像、TTS、浏览器路由）
6. 返回终端，即可运行 `hermes chat`

如果还没有订阅，请先在 [portal.nousresearch.com/manage-subscription](https://portal.nousresearch.com/manage-subscription) 注册。

### 已有安装——在现有提供商旁添加 Portal

如果你已经配置了 OpenRouter、Anthropic 或其他提供商，想在此基础上添加 Portal：

```bash
hermes model
# 从提供商列表中选择 "Nous Portal"
# 浏览器打开，登录，完成
```

你现有的提供商配置保持不变。可以在会话中途用 `/model` 切换，或在会话间用 `hermes model` 切换——Portal 成为你的可用提供商之一，而非唯一选项。

### 无头环境 / SSH / 远程配置

OAuth 需要浏览器，但回调的 loopback 运行在 Hermes 所在的机器上。对于远程主机，请参阅 [OAuth over SSH / 远程主机](/guides/oauth-over-ssh)——与其他基于 OAuth 的提供商相同的方式同样适用于 Portal（`ssh -L` 端口转发，或在 Cloud Shell / Codespaces 等纯浏览器环境中使用 `--manual-paste`）。

### Profile 配置

如果你使用 [Hermes profiles（配置文件）](/user-guide/profiles)，Portal 的 refresh token 会通过共享令牌存储自动在所有 profile 间共享。在任意 profile 上登录一次，其余 profile 自动获取——无需为每个 profile 重复 OAuth 流程。

## 日常使用 Portal

### 查看当前配置状态

```bash
hermes portal            # 登录 Nous Portal 并完成配置（一键引导）
hermes portal info       # 登录状态、订阅信息、模型与 gateway 路由
hermes portal tools      # 详细的 Tool Gateway 目录及每个工具的路由信息
hermes portal open       # 在浏览器中打开订阅管理页面
```

`hermes portal`（不带子命令）是 `hermes auth add nous --type oauth` 的易记别名——它会登录、让你选择 Nous 模型、把 Nous 设为推理服务商，并提供 Tool Gateway 启用选项（与 `hermes setup --portal` 等价，与首次快速设置走的是同一套 Nous 流程）。

`hermes portal info` 给出高层概览：

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
  Cloud terminal        not configured
```

### 切换模型

在会话中：

```bash
/model anthropic/claude-sonnet-4.6
/model openai/gpt-5.4
/model google/gemini-2.5-pro
```

或打开选择器：

```bash
/model
# 方向键选择，回车确认
```

在会话外（完整配置向导，适合添加新提供商时使用）：

```bash
hermes model
```

### 将 gateway 与自有后端混用

如果你已有 Browserbase 账号并希望继续使用，同时通过 Nous 路由网页搜索和图像生成，这是支持的。使用 `hermes tools` 为每个工具单独选择后端：

```bash
hermes tools
# → 网页搜索       → "Nous Subscription"
# → 图像生成       → "Nous Subscription"
# → 浏览器         → "Browserbase"（你的现有密钥）
# → TTS            → "Nous Subscription"
```

Tool Gateway 是按工具单独选择启用的，而非全部或全不。完整的每工具配置矩阵请参阅 [Tool Gateway 文档](/user-guide/features/tool-gateway)。

### 订阅管理

随时管理套餐、查看用量或升级/取消：

- **网页端：** [portal.nousresearch.com/manage-subscription](https://portal.nousresearch.com/manage-subscription)
- **CLI 快捷方式：** `hermes portal open`（在默认浏览器中打开同一页面）

## 配置参考

运行 `hermes setup --portal` 后，`~/.hermes/config.yaml` 将如下所示：

```yaml
model:
  provider: nous
  default: anthropic/claude-sonnet-4.6     # 或你选择的其他模型
  base_url: https://inference.nousresearch.com/v1
```

Tool Gateway 设置位于各自工具的配置节下：

```yaml
web:
  backend: nous       # 网页搜索/抓取通过 Tool Gateway 路由

image_gen:
  provider: nous

tts:
  provider: nous

browser:
  backend: nous
```

OAuth refresh token 单独存储在 `~/.hermes/auth.json`（不在 `config.yaml` 中——凭证与配置有意分开存放）。

## 令牌处理

Hermes 在每次推理调用时从存储的 Portal refresh token 生成短期 JWT，而非复用长期 API 密钥。令牌生命周期完全自动管理——刷新、生成、在瞬时 401 时重试——你无需关心这些细节。

如果 Portal 使 refresh token 失效（修改密码、手动撤销、会话过期），失效的 refresh token 会被**本地隔离**，Hermes 停止重放该令牌，你不会看到一连串相同的 401 错误。下一次调用会显示清晰的"需要重新认证"提示。运行 `hermes auth add nous` 重新登录；隔离状态在下次成功登录时自动清除。

## 故障排查

### `hermes portal info` 显示"not logged in"

你尚未完成 OAuth 流程，或 refresh token 已被清除。运行：

```bash
hermes portal
```

或使用 `hermes model` 重新选择 Nous Portal。

### 会话中途收到"需要重新认证"提示

你的 Portal refresh token 已失效（修改密码、手动撤销或会话过期）。运行 `hermes auth add nous`，下一次请求将使用新凭证。旧令牌的隔离状态在成功重新登录后自动清除。

### 想使用 Portal 未暴露的特定提供商模型

Portal 通过 OpenRouter 代理，因此 OpenRouter 支持的所有模型通常都可用。如果某个模型未出现在 `/model` 中，可直接尝试 OpenRouter 风格的 slug：

```bash
/model anthropic/claude-opus-4.6
```

如果某个模型确实缺失，请[提交 issue](https://github.com/NousResearch/hermes-agent/issues)——我们将 Portal 目录同步至 Hermes，缺口通常意味着可以更新的路由配置。

### 账单未出现在我的 Portal 账号中

先检查 `hermes portal info`——如果显示你正在使用其他提供商（`Model: currently openrouter` 而非 `using Nous as inference provider`），说明本地配置已偏离。运行 `hermes model`，选择 Nous Portal，下一次请求将通过你的订阅路由。

## 另请参阅

- **[Tool Gateway](/user-guide/features/tool-gateway)** —— 每个 gateway 工具的完整详情、每工具配置及定价
- **[订阅代理](/user-guide/features/subscription-proxy)** —— 在非 Hermes 工具（其他 agent、脚本、第三方客户端）中使用你的 Portal 订阅
- **[语音模式](/user-guide/features/voice-mode)** —— 使用 Portal 的 OpenAI TTS 进行语音对话
- **[AI 提供商](/integrations/providers)** —— 完整提供商目录，供对比参考
- **[OAuth over SSH](/guides/oauth-over-ssh)** —— 从远程主机或纯浏览器环境登录
- **[Profiles](/user-guide/profiles)** —— 多个 Hermes 配置共享一个 Portal 登录