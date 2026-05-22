---
sidebar_position: 1
title: "快速上手"
description: "在 5 分钟内完成 Hermes Agent 的第一次对话 - 从安装到开始聊天"
---

# 快速上手

这篇指南会带你从零开始，完成一个真正可用、能应对实际场景的 Hermes 配置。安装、选择提供商、验证聊天是否正常，然后知道出问题时该怎么处理。

## 想先看视频？

**Onchain AI Garage** 做了一版安装、配置和基础命令的 Masterclass 演示视频 - 如果你更喜欢边看边学，它可以作为这篇页面的补充。更多内容见完整的 [Hermes Agent Tutorials & Use Cases](https://www.youtube.com/channel/UCqB1bhMwGsW-yefBxYwFCCg) 播放列表。

<div style={{position: 'relative', paddingBottom: '56.25%', height: 0, overflow: 'hidden', maxWidth: '100%', marginBottom: '1.5rem'}}>
  <iframe
    style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%'}}
    src="https://www.youtube-nocookie.com/embed/R3YOGfTBcQg"
    title="Hermes Agent Masterclass: Installation, Setup, Basic Commands"
    frameBorder="0"
    allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowFullScreen
  ></iframe>
</div>

## 适合谁

- 刚开始使用，想走最短路径把东西跑起来
- 想切换提供商，但不想在配置上浪费时间
- 要给团队、机器人或常驻工作流部署 Hermes
- 受够了“装是装好了，但还是没反应”

## 最快路径

按你的目标选择对应行：

| 目标 | 先做这个 | 然后做这个 |
|---|---|---|
| 我只想让 Hermes 在机器上跑起来 | `hermes setup` | 跑一次真实对话，确认它能回复 |
| 我已经知道要用哪个提供商 | `hermes model` | 保存配置，然后开始聊天 |
| 我要做机器人或常驻在线方案 | CLI 正常后运行 `hermes gateway setup` | 连接 Telegram、Discord、Slack 或其他平台 |
| 我要本地或自托管模型 | `hermes model` → 自定义端点 | 验证端点、模型名和上下文长度 |
| 我要多提供商回退 | 先运行 `hermes model` | 只有基础聊天正常后，再加路由和回退 |

**经验法则：** 如果 Hermes 连一次普通聊天都完成不了，就先不要加更多功能。先把一个干净的对话跑通，再叠加网关、cron、技能、语音或路由。

---

## 1. 安装 Hermes Agent

运行一行安装器：

```bash
# Linux / macOS / WSL2 / Android (Termux)
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

:::tip Android / Termux
如果你是在手机上安装，请参阅专门的 [Termux 指南](./termux.md)，里面有经过测试的手动路径、支持的额外依赖和当前 Android 限制。
:::

:::tip Windows 用户
先安装 [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)，然后在 WSL2 终端里运行上面的命令。
:::

安装完成后，重新加载 shell：

```bash
source ~/.bashrc   # 或 source ~/.zshrc
```

更详细的安装选项、前提条件和故障排查，请查看 [安装指南](./installation.md)。

## 2. 选择提供商

这是最重要的配置步骤。用 `hermes model` 交互式完成选择：

```bash
hermes model
```

推荐默认项：

| 提供商 | 它是什么 | 如何配置 |
|----------|-----------|---------------|
| **Nous Portal** | 订阅制、零配置 | 在 `hermes model` 里登录 OAuth |
| **OpenAI Codex** | 使用 Codex 模型的 ChatGPT OAuth | 在 `hermes model` 里通过 device code 授权 |
| **Anthropic** | 直接使用 Claude 模型 - Max 方案 + 额外使用额度（OAuth），或 API key 按量付费 | `hermes model` → OAuth 登录（需要 Max + 额外额度），或使用 Anthropic API key |
| **OpenRouter** | 跨很多模型的多提供商路由 | 输入你的 API key |
| **Z.AI** | GLM / Zhipu 托管模型 | 设置 `GLM_API_KEY` / `ZAI_API_KEY` |
| **Kimi / Moonshot** | Moonshot 托管的编程和聊天模型 | 设置 `KIMI_API_KEY`（或 Kimi-Coding 专用 `KIMI_CODING_API_KEY`） |
| **Kimi / Moonshot China** | 中国区 Moonshot 端点 | 设置 `KIMI_CN_API_KEY` |
| **Arcee AI** | Trinity 模型 | 设置 `ARCEEAI_API_KEY` |
| **GMI Cloud** | 多模型直连 API | 设置 `GMI_API_KEY` |
| **MiniMax (OAuth)** | 通过浏览器 OAuth 使用 MiniMax-M2.7 - 无需 API key | `hermes model` → MiniMax (OAuth) |
| **MiniMax** | 国际版 MiniMax 端点 | 设置 `MINIMAX_API_KEY` |
| **MiniMax China** | 中国区 MiniMax 端点 | 设置 `MINIMAX_CN_API_KEY` |
| **Alibaba Cloud** | 通过 DashScope 使用 Qwen 模型 | 设置 `DASHSCOPE_API_KEY` |
| **Hugging Face** | 20+ 个开放模型的统一路由（Qwen、DeepSeek、Kimi 等） | 设置 `HF_TOKEN` |
| **AWS Bedrock** | 通过原生 Converse API 使用 Claude、Nova、Llama、DeepSeek | IAM role 或 `aws configure`（[指南](/guides/aws-bedrock)） |
| **Kilo Code** | KiloCode 托管模型 | 设置 `KILOCODE_API_KEY` |
| **OpenCode Zen** | 按量付费访问精选模型 | 设置 `OPENCODE_ZEN_API_KEY` |
| **OpenCode Go** | 每月 10 美元的开放模型订阅 | 设置 `OPENCODE_GO_API_KEY` |
| **DeepSeek** | 直接访问 DeepSeek API | 设置 `DEEPSEEK_API_KEY` |
| **NVIDIA NIM** | 通过 build.nvidia.com 或本地 NIM 使用 Nemotron 模型 | 设置 `NVIDIA_API_KEY`（可选：`NVIDIA_BASE_URL`） |
| **GitHub Copilot** | GitHub Copilot 订阅（GPT-5.x、Claude、Gemini 等） | 在 `hermes model` 里 OAuth，或使用 `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` |
| **GitHub Copilot ACP** | Copilot ACP 代理后端（启动本地 `copilot` CLI） | `hermes model`（需要 `copilot` CLI + `copilot login`） |
| **Vercel AI Gateway** | Vercel AI Gateway 路由 | 设置 `AI_GATEWAY_API_KEY` |
| **Custom Endpoint** | VLLM、SGLang、Ollama，或任何兼容 OpenAI 的 API | 设置 base URL + API key |

第一次使用时，建议先选一个提供商，默认值先不要改，除非你清楚自己为什么要改。完整的提供商目录、环境变量和配置步骤见 [Providers](/integrations/providers) 页面。

:::caution 最低上下文：64K tokens
Hermes Agent 需要至少 **64,000 tokens** 的上下文长度。上下文窗口更小的模型无法为多步工具调用工作流保留足够的工作记忆，会在启动时被拒绝。大多数托管模型（Claude、GPT、Gemini、Qwen、DeepSeek）都很容易满足。如果你用的是本地模型，请把上下文至少设到 64K（例如 llama.cpp 用 `--ctx-size 65536`，Ollama 用 `-c 65536`）。
:::

:::tip
你可以随时用 `hermes model` 切换提供商 - 没有锁定。若想查看所有受支持提供商及其配置细节，请参见 [AI Providers](/integrations/providers)。
:::

### 设置如何存储

Hermes 把机密信息和普通配置分开存储：

- **机密和 token** → `~/.hermes/.env`
- **非机密设置** → `~/.hermes/config.yaml`

最简单的设置方式是通过 CLI：

```bash
hermes config set model anthropic/claude-opus-4.6
hermes config set terminal.backend docker
hermes config set OPENROUTER_API_KEY sk-or-...
```

对应的值会自动写入正确的文件。

## 3. 运行第一次聊天

```bash
hermes            # 经典 CLI
hermes --tui      # 现代 TUI（推荐）
```

你会看到欢迎横幅，其中显示模型、可用工具和技能。请使用一个具体且容易验证的提示词：

:::tip 选择界面
Hermes 附带两种终端界面：经典的 `prompt_toolkit` CLI，以及更新的 [TUI](../user-guide/tui.md)，支持模态覆盖层、鼠标选择和非阻塞输入。二者共享同一套会话、斜杠命令和配置 - 可以用 `hermes` 和 `hermes --tui` 都试一下。
:::

```
请用 5 条要点概括这个仓库，并告诉我主要入口文件是什么。
```

```
检查我当前目录，并告诉我看起来哪个文件最像主项目文件。
```

```
帮我为这个代码库设计一个干净的 GitHub PR 工作流。
```

**成功的表现：**

- 横幅显示你选择的模型 / 提供商
- Hermes 能正常回复
- 它在需要时能使用工具（终端、文件读取、网页搜索）
- 对话可以正常继续到第二轮以上

如果这一步通了，最难的部分就已经过去了。

## 4. 验证会话功能

在继续之前，先确认 resume 可以工作：

```bash
hermes --continue    # 恢复最近一次会话
hermes -c            # 简写形式
```

这应该会把你带回刚才那次会话。如果没有，检查你是否在同一个 profile 里，以及会话是否真的保存了。以后你在不同设置或机器之间切换时，这一点很重要。

## 5. 试试关键功能

### 使用终端

```
❯ 我的磁盘使用情况是多少？请显示最大的 5 个目录。
```

智能体会代你运行终端命令并显示结果。

### 斜杠命令

输入 `/` 可以看到所有命令的自动补全下拉框：

| 命令 | 作用 |
|---------|-------------|
| `/help` | 显示所有可用命令 |
| `/tools` | 列出可用工具 |
| `/model` | 交互式切换模型 |
| `/personality pirate` | 尝试一个有趣的个性 |
| `/save` | 保存对话 |

### 多行输入

按 `Alt+Enter`、`Ctrl+J` 或 `Shift+Enter` 可以添加新行。`Shift+Enter` 需要终端能把它作为独立序列发送（Kitty / foot / WezTerm / Ghostty 默认支持；iTerm2 / Alacritty / VS Code terminal 需启用 Kitty 键盘协议）。`Alt+Enter` 和 `Ctrl+J` 在所有终端都可用。

### 中断智能体

如果智能体花太久，直接输入新消息并按 Enter - 它会中断当前任务并切换到你的新指令。`Ctrl+C` 也可以。

## 6. 加上下一层能力

只有在基础聊天可用之后再做。按你的需要选择：

### 机器人或共享助手

```bash
hermes gateway setup    # 交互式平台配置
```

连接 [Telegram](/user-guide/messaging/telegram)、[Discord](/user-guide/messaging/discord)、[Slack](/user-guide/messaging/slack)、[WhatsApp](/user-guide/messaging/whatsapp)、[Signal](/user-guide/messaging/signal)、[Email](/user-guide/messaging/email) 或 [Home Assistant](/user-guide/messaging/homeassistant)，也可以连接 [Microsoft Teams](/user-guide/messaging/teams)。

### 自动化和工具

- `hermes tools` - 调整每个平台的工具访问权限
- `hermes skills` - 浏览并安装可复用工作流
- Cron - 只在机器人或 CLI 设置稳定后再用

### 沙箱终端

为了安全，可以把智能体运行在 Docker 容器里或远程服务器上：

```bash
hermes config set terminal.backend docker    # Docker 隔离
hermes config set terminal.backend ssh       # 远程服务器
```

### 语音模式

```bash
# 在 Hermes 安装目录中（Linux/macOS 上安装器默认放在 ~/.hermes/hermes-agent，
# Windows 上放在 %LOCALAPPDATA%\\hermes\\hermes-agent）：
cd ~/.hermes/hermes-agent
uv pip install -e ".[voice]"
# 包含 faster-whisper，可免费做本地语音转文本
```

然后在 CLI 中输入：`/voice on`。按 `Ctrl+B` 开始录音。参见 [Voice Mode](../user-guide/features/voice-mode.md)。

### Skills

```bash
hermes skills search kubernetes
hermes skills install openai/skills/k8s
```

或者在聊天会话内使用 `/skills`。---
sidebar_position: 1
title: "快速开始"
description: "与 Hermes Agent 的第一次对话 — 从安装到聊天，5 分钟内完成"
---

# 快速开始

本指南将带你从零开始，搭建一个能在实际使用中稳定运行的 Hermes 环境。完成安装、选择 provider、验证聊天功能，并清楚了解出问题时该如何处理。

## 更喜欢看视频？

**Onchain AI Garage** 制作了一套关于安装、设置和基础命令的 Masterclass  walkthrough — 如果你更喜欢跟着视频操作，这是本页面的好搭档。更多内容请查看完整的 [Hermes Agent 教程与用例](https://www.youtube.com/channel/UCqB1bhMwGsW-yefBxYwFCCg) 播放列表。

<div style={{position: 'relative', paddingBottom: '56.25%', height: 0, overflow: 'hidden', maxWidth: '100%', marginBottom: '1.5rem'}}>
  <iframe
    style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%'}}
    src="https://www.youtube-nocookie.com/embed/R3YOGfTBcQg"
    title="Hermes Agent Masterclass: Installation, Setup, Basic Commands"
    frameBorder="0"
    allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowFullScreen
  ></iframe>
</div>

## 本文适合谁

- 全新用户，希望找到最快捷的可用路径
- 正在切换 provider，不想在配置错误上浪费时间
- 为团队、机器人或常驻工作流设置 Hermes
- 厌倦了 "装好了，但什么都没发生"

## 最快路径

选择符合你目标的行：

| 目标 | 首先执行 | 然后执行 |
|---|---|---|
| 我只想在我的机器上运行 Hermes | `hermes setup` | 运行一次真实聊天并验证它能响应 |
| 我已经知道我的 provider | `hermes model` | 保存配置，然后开始聊天 |
| 我想要一个机器人或常驻设置 | CLI 正常工作后执行 `hermes gateway setup` | 连接 Telegram、Discord、Slack 或其他平台 |
| 我想要本地或自托管模型 | `hermes model` → custom endpoint | 验证 endpoint、模型名称和上下文长度 |
| 我想要多 provider 故障转移 | 先执行 `hermes model` | 基础聊天正常工作后再添加路由和故障转移 |

**经验法则：** 如果 Hermes 无法完成一次正常聊天，先不要添加更多功能。先让一个干净的对话正常工作，然后再叠加 gateway、cron、skills、voice 或 routing。

---

## 1. 安装 Hermes Agent

运行一行命令安装器：

```bash
# Linux / macOS / WSL2 / Android (Termux)
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

:::tip Android / Termux
如果你在手机上安装，请参阅专门的 [Termux 指南](./termux.md)，了解经过测试的手动路径、支持的扩展功能以及当前 Android 特有的限制。
:::

:::tip Windows 用户
先安装 [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)，然后在 WSL2 终端中运行上面的命令。
:::

安装完成后，重新加载你的 shell：

```bash
source ~/.bashrc   # 或 source ~/.zshrc
```

有关详细的安装选项、前置要求和故障排除，请参阅 [安装指南](./installation.md)。

## 2. 选择 Provider

这是设置中最重要的一步。使用 `hermes model` 以交互方式完成选择：

```bash
hermes model
```

推荐默认选项：

| Provider | 说明 | 设置方式 |
|----------|-----------|---------------|
| **Nous Portal** | 订阅制，零配置 | 通过 `hermes model` 进行 OAuth 登录 |
| **OpenAI Codex** | ChatGPT OAuth，使用 Codex 模型 | 通过 `hermes model` 进行设备码认证 |
| **Anthropic** | 直接使用 Claude 模型 — Max 套餐 + 额外使用额度 (OAuth)，或按 token 付费的 API key | `hermes model` → OAuth 登录（需要 Max + 额外额度），或 Anthropic API key |
| **OpenRouter** | 跨多个模型的多 provider 路由 | 输入你的 API key |
| **Z.AI** | GLM / Zhipu 托管模型 | 设置 `GLM_API_KEY` / `ZAI_API_KEY` |
| **Kimi / Moonshot** | Moonshot 托管的编程和聊天模型 | 设置 `KIMI_API_KEY`（或 Kimi-Coding 专用的 `KIMI_CODING_API_KEY`） |
| **Kimi / Moonshot China** | 中国区 Moonshot endpoint | 设置 `KIMI_CN_API_KEY` |
| **Arcee AI** | Trinity 模型 | 设置 `ARCEEAI_API_KEY` |
| **GMI Cloud** | 多模型直连 API | 设置 `GMI_API_KEY` |
| **MiniMax (OAuth)** | MiniMax-M2.7 通过浏览器 OAuth — 无需 API key | `hermes model` → MiniMax (OAuth) |
| **MiniMax** | 国际版 MiniMax endpoint | 设置 `MINIMAX_API_KEY` |
| **MiniMax China** | 中国区 MiniMax endpoint | 设置 `MINIMAX_CN_API_KEY` |
| **Alibaba Cloud** | 通过 DashScope 使用 Qwen 模型 | 设置 `DASHSCOPE_API_KEY` |
| **Hugging Face** | 通过统一路由器使用 20+ 开源模型（Qwen、DeepSeek、Kimi 等） | 设置 `HF_TOKEN` |
| **AWS Bedrock** | 通过原生 Converse API 使用 Claude、Nova、Llama、DeepSeek | IAM 角色或 `aws configure`（[指南](/guides/aws-bedrock)） |
| **Kilo Code** | KiloCode 托管模型 | 设置 `KILOCODE_API_KEY` |
| **OpenCode Zen** | 按量付费，访问精选模型 | 设置 `OPENCODE_ZEN_API_KEY` |
| **OpenCode Go** | $10/月订阅，使用开源模型 | 设置 `OPENCODE_GO_API_KEY` |
| **DeepSeek** | 直连 DeepSeek API | 设置 `DEEPSEEK_API_KEY` |
| **NVIDIA NIM** | 通过 build.nvidia.com 或本地 NIM 使用 Nemotron 模型 | 设置 `NVIDIA_API_KEY`（可选：`NVIDIA_BASE_URL`） |
| **GitHub Copilot** | GitHub Copilot 订阅（GPT-5.x、Claude、Gemini 等） | 通过 `hermes model` 进行 OAuth，或 `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` |
| **GitHub Copilot ACP** | Copilot ACP agent 后端（生成本地 `copilot` CLI） | `hermes model`（需要 `copilot` CLI + `copilot login`） |
| **Vercel AI Gateway** | Vercel AI Gateway 路由 | 设置 `AI_GATEWAY_API_KEY` |
| **Custom Endpoint** | VLLM、SGLang、Ollama 或任何兼容 OpenAI 的 API | 设置 base URL + API key |

对大多数首次用户：选择一个 provider，除非你清楚为什么要更改，否则接受默认设置。完整的 provider 目录、环境变量和设置步骤见 [Providers](/integrations/providers) 页面。

:::caution 最低上下文：64K token
Hermes Agent 要求模型至少支持 **64,000 token** 的上下文。上下文窗口较小的模型无法为多步骤 tool-calling 工作流维持足够的工作内存，将在启动时被拒绝。大多数托管模型（Claude、GPT、Gemini、Qwen、DeepSeek）轻松满足此要求。如果你运行本地模型，请将其上下文大小设置为至少 64K（例如 llama.cpp 使用 `--ctx-size 65536` 或 Ollama 使用 `-c 65536`）。
:::

:::tip
你可以随时通过 `hermes model` 切换 provider — 无锁定。有关所有支持的 provider 和设置详情，请参阅 [AI Providers](/integrations/providers)。
:::

### 设置如何存储

Hermes 将 secrets 与普通配置分开：

- **Secrets 和 token** → `~/.hermes/.env`
- **非 secret 设置** → `~/.hermes/config.yaml`

最简便的正确设置方式是通过 CLI：

```bash
hermes config set model anthropic/claude-opus-4.6
hermes config set terminal.backend docker
hermes config set OPENROUTER_API_KEY [REDACTED]
```

正确的值会自动写入正确的文件。

## 3. 运行你的第一次聊天

```bash
hermes            # 经典 CLI
hermes --tui      # 现代 TUI（推荐）
```

你会看到一个欢迎横幅，显示你的模型、可用工具和 skills。使用一个具体且易于验证的 prompt：

:::tip 选择你的界面
Hermes 提供两种终端界面：经典的 `prompt_toolkit` CLI 和较新的 [TUI](../user-guide/tui.md)，后者支持模态覆盖层、鼠标选择和非阻塞输入。两者共享相同的 session、斜杠命令和配置 — 分别用 `hermes` 和 `hermes --tui` 尝试。
:::

```
Summarize this repo in 5 bullets and tell me what the main entrypoint is.
```

```
Check my current directory and tell me what looks like the main project file.
```

```
Help me set up a clean GitHub PR workflow for this codebase.
```

**成功的标志：**

- 横幅显示你选择的模型/provider
- Hermes 无错误回复
- 需要时可以使用工具（terminal、file read、web search）
- 对话可以正常进行多轮

如果以上都正常，最困难的部分已经过去了。

## 4. 验证 Session 是否正常工作

在继续之前，确保恢复功能正常：

```bash
hermes --continue    # 恢复最近的 session
hermes -c            # 简写形式
```

这应该能把你带回刚才的 session。如果不行，检查你是否在同一个 profile 中，以及 session 是否确实已保存。当你需要管理多个设置或机器时，这一点很重要。

## 5. 尝试关键功能

### 使用 terminal

```
❯ What's my disk usage? Show the top 5 largest directories.
```

Agent 会代你运行 terminal 命令并显示结果。

### 斜杠命令

输入 `/` 查看所有命令的自动补全下拉列表：

| Command | 功能 |
|---------|-------------|
| `/help` | 显示所有可用命令 |
| `/tools` | 列出可用工具 |
| `/model` | 交互式切换模型 |
| `/personality pirate` | 尝试一个有趣的人格 |
| `/save` | 保存对话 |

### 多行输入

按 `Alt+Enter`、`Ctrl+J` 或 `Shift+Enter` 换行。`Shift+Enter` 需要终端将其作为独立序列发送（Kitty / foot / WezTerm / Ghostty 默认支持；iTerm2 / Alacritty / VS Code terminal 需要启用 Kitty keyboard protocol 后支持）。`Alt+Enter` 和 `Ctrl+J` 在所有终端中都可用。

### 中断 Agent

如果 Agent 耗时过长，输入新消息并按 Enter — 它会中断当前任务并切换到你的新指令。`Ctrl+C` 也有效。

## 6. 添加下一层

仅在基础聊天正常工作之后。选择你需要的：

### 机器人或共享助手

```bash
hermes gateway setup    # 交互式平台配置
```

连接 [Telegram](/user-guide/messaging/telegram)、[Discord](/user-guide/messaging/discord)、[Slack](/user-guide/messaging/slack)、[WhatsApp](/user-guide/messaging/whatsapp)、[Signal](/user-guide/messaging/signal)、[Email](/user-guide/messaging/email)、[Home Assistant](/user-guide/messaging/homeassistant) 或 [Microsoft Teams](/user-guide/messaging/teams)。

### 自动化和工具

- `hermes tools` — 按平台调整工具访问权限
- `hermes skills` — 浏览和安装可复用工作流
- Cron — 仅在机器人或 CLI 设置稳定后使用

### 沙盒 terminal

为了安全，在 Docker 容器或远程服务器中运行 agent：

```bash
hermes config set terminal.backend docker    # Docker 隔离
hermes config set terminal.backend ssh       # 远程服务器
```

### 语音模式

```bash
# 从 Hermes 安装目录执行（curl 安装器将其放在
# Linux/macOS 的 ~/.hermes/hermes-agent 或 Windows 的 %LOCALAPPDATA%\hermes\hermes-agent）：
cd ~/.hermes/hermes-agent
uv pip install -e ".[voice]"
# 包含免费的本地 faster-whisper 语音转文字
```

然后在 CLI 中：`/voice on`。按 `Ctrl+B` 录音。请参阅 [Voice Mode](../user-guide/features/voice-mode.md)。

### Skills

```bash
hermes skills search kubernetes
hermes skills install openai/skills/k8s
```

或在聊天 session 中使用 `/skills`。

### MCP 服务器

```yaml
# 添加到 ~/.hermes/config.yaml
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "[REDACTED]"
```

### 编辑器集成 (ACP)

ACP 支持随标准 `[all]` extras 一起提供，因此 curl 安装器已经包含它。只需运行：

```bash
hermes acp
```

（如果你未使用 `[all]` 安装，请先运行 `cd ~/.hermes/hermes-agent && uv pip install -e ".[acp]"`。）

请参阅 [ACP Editor Integration](../user-guide/features/acp.md)。

---

## 常见故障模式

以下是最浪费时间的问题：

| 症状 | 可能原因 | 解决方法 |
|---|---|---|
| Hermes 能打开但回复为空或错乱 | Provider 认证或模型选择错误 | 重新运行 `hermes model` 并确认 provider、模型和认证信息 |
| Custom endpoint "能用" 但返回乱码 | Base URL、模型名称错误，或实际不兼容 OpenAI | 先在独立的客户端中验证 endpoint |
| Gateway 启动但无人可发消息 | Bot token、allowlist 或平台设置不完整 | 重新运行 `hermes gateway setup` 并检查 `hermes gateway status` |
| `hermes --continue` 找不到旧 session | 切换了 profile 或 session 从未保存 | 检查 `hermes sessions list` 并确认你在正确的 profile 中 |
| 模型不可用或出现奇怪的故障转移行为 | Provider routing 或故障转移设置过于激进 | 在基础 provider 稳定前保持 routing 关闭 |
| `hermes doctor` 标记配置问题 | 配置值缺失或过时 | 修复配置，在添加功能前先重新测试一次普通聊天 |

## 恢复工具包

当感觉不对劲时，按以下顺序操作：

1. `hermes doctor`
2. `hermes model`
3. `hermes setup`
4. `hermes sessions list`
5. `hermes --continue`
6. `hermes gateway status`

这个序列能让你从 "感觉坏了" 快速回到已知正常状态。

---

## 快速参考

| Command | 说明 |
|---------|-------------|
| `hermes` | 开始聊天 |
| `hermes model` | 选择你的 LLM provider 和模型 |
| `hermes tools` | 配置每个平台启用的工具 |
| `hermes setup` | 完整设置向导（一次性配置所有内容） |
| `hermes doctor` | 诊断问题 |
| `hermes update` | 更新到最新版本 |
| `hermes gateway` | 启动消息 gateway |
| `hermes --continue` | 恢复上次 session |

## 下一步

- **[CLI 指南](../user-guide/cli.md)** — 掌握终端界面
- **[配置](../user-guide/configuration.md)** — 自定义你的设置
- **[Messaging Gateway](/user-guide/messaging)** — 连接 Telegram、Discord、Slack、WhatsApp、Signal、Email、Home Assistant、Teams 等
- **[Tools & Toolsets](../user-guide/features/tools.md)** — 探索可用功能
- **[AI Providers](/integrations/providers)** — 完整 provider 列表和设置详情
- **[Skills System](../user-guide/features/skills.md)** — 可复用工作流和知识
- **[Tips & Best Practices](/guides/tips)** — 高级用户技巧
