---
sidebar_position: 1
title: "快速入门"
description: "与 Hermes Agent 的第一次对话——从安装到开始聊天，5 分钟内完成"
---

# 快速入门

本指南带你从零开始搭建一个能够应对实际使用的 Hermes 环境。完成安装、选择 provider（服务提供商）、验证对话正常运行，并了解出现问题时的处理方法。

## 更喜欢看视频？

**Onchain AI Garage** 制作了一套涵盖安装、配置和基本命令的 Masterclass 演示视频——如果你更习惯跟着视频操作，这是本页的绝佳补充。更多内容请查看完整的 [Hermes Agent 教程与使用案例](https://www.youtube.com/channel/UCqB1bhMwGsW-yefBxYwFCCg) 播放列表。

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

## 适用人群

- 全新用户，想以最短路径完成可用配置
- 正在切换 provider，不想因配置错误浪费时间
- 为团队、机器人或长期运行的工作流配置 Hermes
- 厌倦了"安装成功但什么都做不了"的情况

## 最快路径

根据你的目标选择对应行：

| 目标 | 先做这步 | 再做这步 |
|---|---|---|
| 只想让 Hermes 在本机跑起来 | `hermes setup` | 运行一次真实对话并验证有响应 |
| 已知道要用哪个 provider | `hermes model` | 保存配置，然后开始聊天 |
| 想搭建机器人或长期运行的服务 | CLI 正常后运行 `hermes gateway setup` | 接入 Telegram、Discord、Slack 或其他平台 |
| 想使用本地或自托管模型 | `hermes model` → 自定义 endpoint | 验证 endpoint、模型名称和上下文长度 |
| 想要多 provider 故障转移 | 先运行 `hermes model` | 基础对话正常后再添加路由和故障转移 |

**经验法则：** 如果 Hermes 无法完成一次正常对话，暂时不要添加更多功能。先让一次完整对话跑通，再逐步叠加 gateway、cron、skills、语音或路由。

---

## 1. 安装 Hermes Agent

### 在 macOS 或 Windows 上使用 Hermes Desktop 安装器（推荐）

如需同时安装命令行与桌面应用，请从我们的官网[下载 Hermes Desktop 安装器](https://hermes-agent.nousresearch.com/)并运行。

### 不使用 Hermes Desktop：

仅安装命令行版本（跟踪 main 分支）：

```bash
# Linux / macOS / WSL2 / Android (Termux)
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

安装脚本会在 `~/.hermes/hermes-agent` 创建一个受管理的隔离环境（独立的 uv 托管解释器和 venv），这是唯一受支持的安装方式 —— 包括开发用途。请勿使用 `pip install hermes-agent`。

:::tip Android / Termux
如果你在手机上安装，请参阅专门的 [Termux 指南](./termux.md)，其中包含经过测试的手动安装步骤、支持的扩展功能以及当前 Android 特有的限制。
:::

:::tip Windows 用户
请先安装 [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)，然后在 WSL2 终端中运行上述命令。
:::

安装完成后，重新加载 shell：

```bash
source ~/.bashrc   # 或 source ~/.zshrc
```

详细的安装选项、前置条件和故障排查，请参阅 [安装指南](./installation.md)。

## 2. 选择 Provider

这是最重要的配置步骤。使用 `hermes model` 以交互方式完成选择：

```bash
hermes model
```

:::tip 最简路径：Nous Portal
一个订阅涵盖 300+ 个模型，以及 [Tool Gateway](../user-guide/features/tool-gateway.md)（网页搜索、图像生成、TTS、云端浏览器）。全新安装时：

```bash
hermes setup --portal
```

该命令一次性完成登录、设置 Nous 为 provider 并开启 Tool Gateway。
:::

推荐默认选项：

| Provider | 说明 | 配置方式 |
|----------|-----------|---------------|
| **Nous Portal** | 订阅制，零配置 | 通过 `hermes model` 进行 OAuth 登录 |
| **OpenAI Codex** | ChatGPT OAuth，使用 Codex 模型 | 通过 `hermes model` 进行设备码认证 |
| **Anthropic** | 直接使用 Claude 模型——Max 计划 + 额外用量积分（OAuth），或按 token 付费的 API key | `hermes model` → OAuth 登录（需要 Max + 额外积分），或 Anthropic API key |
| **OpenRouter** | 跨多个 provider 的多模型路由 | 输入 API key |
| **Z.AI** | GLM / Zhipu 托管模型 | 设置 `GLM_API_KEY` / `ZAI_API_KEY` |
| **Kimi / Moonshot** | Moonshot 托管的编程和对话模型 | 设置 `KIMI_API_KEY`（或 Kimi-Coding 专用的 `KIMI_CODING_API_KEY`） |
| **Kimi / Moonshot China** | 中国区 Moonshot endpoint | 设置 `KIMI_CN_API_KEY` |
| **Arcee AI** | Trinity 模型 | 设置 `ARCEEAI_API_KEY` |
| **GMI Cloud** | 多模型直连 API | 设置 `GMI_API_KEY` |
| **MiniMax (OAuth)** | 通过浏览器 OAuth 使用 MiniMax-M2.7，无需 API key | `hermes model` → MiniMax (OAuth) |
| **MiniMax** | 国际版 MiniMax endpoint | 设置 `MINIMAX_API_KEY` |
| **MiniMax China** | 中国区 MiniMax endpoint | 设置 `MINIMAX_CN_API_KEY` |
| **Alibaba Cloud** | 通过 DashScope 使用 Qwen 模型 | 设置 `DASHSCOPE_API_KEY` |
| **Hugging Face** | 通过统一路由器使用 20+ 开源模型（Qwen、DeepSeek、Kimi 等） | 设置 `HF_TOKEN` |
| **AWS Bedrock** | 通过原生 Converse API 使用 Claude、Nova、Llama、DeepSeek | IAM 角色或 `aws configure`（[指南](../guides/aws-bedrock.md)） |
| **Kilo Code** | KiloCode 托管模型 | 设置 `KILOCODE_API_KEY` |
| **OpenCode Zen** | 按需付费访问精选模型 | 设置 `OPENCODE_ZEN_API_KEY` |
| **OpenCode Go** | $10/月订阅，访问开源模型 | 设置 `OPENCODE_GO_API_KEY` |
| **DeepSeek** | 直接访问 DeepSeek API | 设置 `DEEPSEEK_API_KEY` |
| **NVIDIA NIM** | 通过 build.nvidia.com 或本地 NIM 使用 Nemotron 模型 | 设置 `NVIDIA_API_KEY`（可选：`NVIDIA_BASE_URL`） |
| **GitHub Copilot** | GitHub Copilot 订阅（GPT-5.x、Claude、Gemini 等） | 通过 `hermes model` 进行 OAuth，或设置 `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` |
| **GitHub Copilot ACP** | Copilot ACP agent 后端（在本地启动 `copilot` CLI） | `hermes model`（需要 `copilot` CLI + `copilot login`） |
| **Custom Endpoint** | VLLM、SGLang、Ollama 或任何兼容 OpenAI 的 API | 设置 base URL + API key |

对于大多数初次使用的用户：选择一个 provider，接受默认值（除非你明确知道为何要修改）。完整的 provider 目录及环境变量和配置步骤请参阅 [Providers](../integrations/providers.md) 页面。

:::caution 最低上下文要求：64K token
Hermes Agent 要求模型至少具备 **64,000 个 token** 的上下文窗口。上下文窗口较小的模型无法为多步骤工具调用工作流维持足够的工作内存，启动时将被拒绝。大多数托管模型（Claude、GPT、Gemini、Qwen、DeepSeek）均轻松满足此要求。如果你运行本地模型，请将其上下文大小设置为至少 64K（例如 llama.cpp 使用 `--ctx-size 65536`，Ollama 使用 `-c 65536`）。
:::

:::tip
你可以随时通过 `hermes model` 切换 provider——没有锁定。所有支持的 provider 完整列表及配置详情，请参阅 [AI Providers](../integrations/providers.md)。
:::

### 配置的存储方式

Hermes 将密钥与普通配置分开存储：

- **密钥和 token** → `~/.hermes/.env`
- **非密钥配置** → `~/.hermes/config.yaml`

通过 CLI 设置值是最简便的方式，系统会自动将值写入正确的文件：

```bash
hermes config set model anthropic/claude-opus-4.6
hermes config set terminal.backend docker
hermes config set OPENROUTER_API_KEY sk-or-...
```

## 3. 运行第一次对话

```bash
hermes            # 经典 CLI
hermes --tui      # 现代 TUI（推荐）
```

你会看到一个欢迎横幅，显示你的模型、可用工具和 skills。使用一个具体且易于验证的 prompt（提示词）：

:::tip 选择你的界面
Hermes 提供两种终端界面：经典的 `prompt_toolkit` CLI，以及更新的 [TUI](../user-guide/tui.md)（支持模态覆盖层、鼠标选择和非阻塞输入）。两者共享相同的会话、斜杠命令和配置——分别用 `hermes` 和 `hermes --tui` 试试看。
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
- Hermes 无错误地回复
- 需要时能够使用工具（终端、文件读取、网页搜索）
- 对话可以正常进行超过一轮

如果以上都正常，你已经过了最难的部分。

## 4. 验证会话功能

继续之前，确认恢复功能正常：

```bash
hermes --continue    # 恢复最近的会话
hermes -c            # 简写形式
```

这应该会带你回到刚才的会话。如果不行，检查你是否在同一个 profile 下，以及会话是否实际已保存。当你同时管理多个配置或多台机器时，这一点很重要。

## 5. 尝试核心功能

### 使用终端

```
❯ What's my disk usage? Show the top 5 largest directories.
```

Agent 会代你执行终端命令并显示结果。

### 斜杠命令

输入 `/` 查看所有命令的自动补全下拉列表：

| 命令 | 功能 |
|---------|-------------|
| `/help` | 显示所有可用命令 |
| `/tools` | 列出可用工具 |
| `/model` | 交互式切换模型 |
| `/personality pirate` | 尝试一个有趣的人格 |
| `/save` | 保存对话 |

### 多行输入

按 `Alt+Enter`、`Ctrl+J` 或 `Shift+Enter` 换行。`Shift+Enter` 需要终端能将其作为独立序列发送（Kitty / foot / WezTerm / Ghostty 默认支持；iTerm2 / Alacritty / VS Code 终端需启用 Kitty 键盘协议）。`Alt+Enter` 和 `Ctrl+J` 在所有终端中均可使用。

### 中断 Agent

如果 agent 响应时间过长，输入新消息并按 Enter——这会中断当前任务并切换到你的新指令。`Ctrl+C` 同样有效。

## 6. 添加下一层功能

仅在基础对话正常后进行。按需选择：

### 机器人或共享助手

```bash
hermes gateway setup    # 交互式平台配置
```

接入 [Telegram](/user-guide/messaging/telegram)、[Discord](/user-guide/messaging/discord)、[Slack](/user-guide/messaging/slack)、[WhatsApp](/user-guide/messaging/whatsapp)、[Signal](/user-guide/messaging/signal)、[Email](/user-guide/messaging/email)、[Home Assistant](/user-guide/messaging/homeassistant) 或 [Microsoft Teams](/user-guide/messaging/teams)。

### 自动化与工具

- `hermes tools` — 按平台调整工具访问权限
- `hermes skills` — 浏览并安装可复用的工作流
- Cron — 仅在机器人或 CLI 配置稳定后使用

### 沙箱终端

为了安全起见，在 Docker 容器或远程服务器中运行 agent：

```bash
hermes config set terminal.backend docker    # Docker 隔离
hermes config set terminal.backend ssh       # 远程服务器
```

### 语音模式

```bash
# 在 Hermes 安装目录下运行（curl 安装器在 Linux/macOS 上将其放置于
# ~/.hermes/hermes-agent，在 Windows 上为 %LOCALAPPDATA%\hermes\hermes-agent）：
cd ~/.hermes/hermes-agent
uv pip install -e ".[voice]"
# 包含 faster-whisper，用于免费的本地语音转文字
```

然后在 CLI 中输入：`/voice on`。按 `Ctrl+B` 开始录音。参阅 [语音模式](../user-guide/features/voice-mode.md)。

### Skills

```bash
hermes skills search kubernetes
hermes skills install openai/skills/k8s
```

或在聊天会话中使用 `/skills`。

### MCP 服务器

```yaml
# 添加到 ~/.hermes/config.yaml
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxx"
```

### 编辑器集成（ACP）

ACP 支持已包含在标准 `[all]` 扩展中，因此 curl 安装器已默认包含。直接运行：

```bash
hermes acp
```

（如果安装时未包含 `[all]`，请先运行 `cd ~/.hermes/hermes-agent && uv pip install -e ".[acp]"`。）

参阅 [ACP 编辑器集成](../user-guide/features/acp.md)。

---

## 常见故障模式

以下是最容易浪费时间的问题：

| 现象 | 可能原因 | 解决方法 |
|---|---|---|
| Hermes 启动但回复为空或异常 | Provider 认证或模型选择有误 | 重新运行 `hermes model`，确认 provider、模型和认证信息 |
| 自定义 endpoint "可用"但返回乱码 | base URL、模型名称有误，或实际上不兼容 OpenAI | 先用独立客户端验证该 endpoint |
| Gateway 启动但无法收到消息 | Bot token、白名单或平台配置不完整 | 重新运行 `hermes gateway setup` 并检查 `hermes gateway status` |
| `hermes --continue` 找不到旧会话 | 切换了 profile 或会话从未保存 | 检查 `hermes sessions list`，确认你在正确的 profile 下 |
| 模型不可用或出现异常的故障转移行为 | Provider 路由或故障转移设置过于激进 | 在基础 provider 稳定之前关闭路由 |
| `hermes doctor` 标记配置问题 | 配置值缺失或已过期 | 修复配置，在添加功能前重新测试普通对话 |

## 恢复工具包

当感觉有问题时，按以下顺序操作：

1. `hermes doctor`
2. `hermes model`
3. `hermes setup`
4. `hermes sessions list`
5. `hermes --continue`
6. `hermes gateway status`

这个顺序能让你快速从"感觉哪里不对"回到已知的正常状态。

---

## 快速参考

| 命令 | 说明 |
|---------|-------------|
| `hermes` | 开始聊天 |
| `hermes model` | 选择 LLM provider 和模型 |
| `hermes tools` | 配置每个平台启用的工具 |
| `hermes setup` | 完整配置向导（一次性配置所有内容） |
| `hermes doctor` | 诊断问题 |
| `hermes update` | 更新到最新版本 |
| `hermes gateway` | 启动消息 gateway |
| `hermes --continue` | 恢复上次会话 |

## 下一步

- **[CLI 指南](../user-guide/cli.md)** — 掌握终端界面
- **[配置](../user-guide/configuration.md)** — 自定义你的配置
- **[消息 Gateway](../user-guide/messaging/index.md)** — 接入 Telegram、Discord、Slack、WhatsApp、Signal、Email、Home Assistant、Teams 等
- **[工具与工具集](../user-guide/features/tools.md)** — 探索可用功能
- **[AI Providers](../integrations/providers.md)** — 完整 provider 列表及配置详情
- **[Skills 系统](../user-guide/features/skills.md)** — 可复用的工作流与知识
- **[技巧与最佳实践](../guides/tips.md)** — 高级用户技巧