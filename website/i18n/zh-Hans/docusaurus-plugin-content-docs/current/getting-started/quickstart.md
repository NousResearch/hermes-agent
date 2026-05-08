---
sidebar_position: 1
title: "快速入门"
description: "与 Hermes Agent 的第一次对话 — 从安装到聊天的 5 分钟指南"
---

# 快速入门

本指南帮助你从零到拥有一个能在实际使用中保持稳定的 Hermes 设置。安装、选择提供商、验证聊天工作，并准确了解出现问题时该怎么做。

## 宁愿观看？

**Onchain AI Garage** 制作了一个安装、设置和基本命令的 Masterclass 演示 —— 如果你更愿意通过视频学习，这是本页的好伴侣。更多内容请参阅完整的 [Hermes Agent 教程和使用案例](https://www.youtube.com/channel/UCqB1bhMwGsW-yefBxYwFCCg) 播放列表。

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

- 全新的新手，想获得最短路径到可工作的设置
- 切换提供商，不想在配置错误上浪费时间
- 为团队、机器人或常驻工作流设置 Hermes
- 厌倦了"安装了，但还是什么都不做"

## 最快路径

选择与你的目标匹配的行：

| 目标 | 首先这样做 | 然后这样做 |
|---|---|---|
| 我只想在我的机器上让 Hermes 工作 | `hermes setup` | 运行真正的聊天并验证它有响应 |
| 我已经知道我的提供商 | `hermes model` | 保存配置，然后开始聊天 |
| 我想要一个机器人或常驻设置 | CLI 工作后 `hermes gateway setup` | 连接 Telegram、Discord、Slack 或其他平台 |
| 我想要本地或自托管模型 | `hermes model` → 自定义端点 | 验证端点、模型名称和上下文长度 |
| 我想要多提供商回退 | 首先 `hermes model` | 仅在基础聊天工作后添加路由和回退 |

**经验法则：** 如果 Hermes 无法完成正常聊天，暂时不要添加更多功能。先让一次干净的对话工作，然后再叠加网关、cron、技能、语音或路由。

---

## 1. 安装 Hermes Agent

运行一行安装程序：

```bash
# Linux / macOS / WSL2 / Android (Termux)
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

:::tip Android / Termux
如果你在手机上安装，请参阅专用的 [Termux 指南](./termux.md)，了解经过测试的手动路径、支持的功能和当前的 Android 特定限制。
:::

:::tip Windows 用户
首先安装 [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)，然后在你的 WSL2 终端内运行上述命令。
:::

完成后，重新加载你的 shell：

```bash
source ~/.bashrc   # 或者 source ~/.zshrc
```

有关详细的安装选项、前置要求和故障排除，请参阅[安装指南](./installation.md)。

## 2. 选择提供商

最关键的设置步骤。使用 `hermes model` 交互式地完成选择：

```bash
hermes model
```

好的默认值：

| 提供商 | 是什么 | 如何设置 |
|----------|-----------|---------------|
| **Nous Portal** | 订阅制，零配置 | 通过 `hermes model` 进行 OAuth 登录 |
| **OpenAI Codex** | ChatGPT OAuth，使用 Codex 模型 | 通过 `hermes model` 进行设备代码认证 |
| **Anthropic** | 直接使用 Claude 模型 — Max 计划 + 额外使用额度（OAuth），或用于按量付费的 API 密钥 | `hermes model` → OAuth 登录（需要 Max + 额外额度），或 Anthropic API 密钥 |
| **OpenRouter** | 跨多模型的多元提供商路由 | 输入你的 API 密钥 |
| **Z.AI** | GLM / 智谱托管模型 | 设置 `GLM_API_KEY` / `ZAI_API_KEY` |
| **Kimi / Moonshot** | Moonshot 托管的编码和聊天模型 | 设置 `KIMI_API_KEY` |
| **Kimi / Moonshot 中国** | 中国区 Moonshot 端点 | 设置 `KIMI_CN_API_KEY` |
| **Arcee AI** | Trinity 模型 | 设置 `ARCEEAI_API_KEY` |
| **GMI Cloud** | 多模型直接 API | 设置 `GMI_API_KEY` |
| **MiniMax (OAuth)** | MiniMax-M2.7 通过浏览器 OAuth — 无需 API 密钥 | `hermes model` → MiniMax (OAuth) |
| **MiniMax** | 国际 MiniMax 端点 | 设置 `MINIMAX_API_KEY` |
| **MiniMax 中国** | 中国区 MiniMax 端点 | 设置 `MINIMAX_CN_API_KEY` |
| **阿里云** | 通过 DashScope 使用 Qwen 模型 | 设置 `DASHSCOPE_API_KEY` |
| **Hugging Face** | 通过统一路由器访问 20+ 开放模型（Qwen、DeepSeek、Kimi 等） | 设置 `HF_TOKEN` |
| **AWS Bedrock** | 通过原生 Converse API 使用 Claude、Nova、Llama、DeepSeek | IAM 角色或 `aws configure`（[指南](../guides/aws-bedrock.md)） |
| **Kilo Code** | KiloCode 托管的模型 | 设置 `KILOCODE_API_KEY` |
| **OpenCode Zen** | 按量付费访问精选模型 | 设置 `OPENCODE_ZEN_API_KEY` |
| **OpenCode Go** | 开放模型每月 10 美元订阅 | 设置 `OPENCODE_GO_API_KEY` |
| **DeepSeek** | 直接 DeepSeek API 访问 | 设置 `DEEPSEEK_API_KEY` |
| **NVIDIA NIM** | 通过 build.nvidia.com 或本地 NIM 使用 Nemotron 模型 | 设置 `NVIDIA_API_KEY`（可选：`NVIDIA_BASE_URL`） |
| **GitHub Copilot** | GitHub Copilot 订阅（GPT-5.x、Claude、Gemini 等） | 通过 `hermes model` 进行 OAuth，或 `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` |
| **GitHub Copilot ACP** | Copilot ACP Agent 后端（生成本地 `copilot` CLI） | `hermes model`（需要 `copilot` CLI + `copilot login`） |
| **Vercel AI Gateway** | Vercel AI Gateway 路由 | 设置 `AI_GATEWAY_API_KEY` |
| **自定义端点** | VLLM、SGLang、Ollama 或任何 OpenAI 兼容 API | 设置基础 URL + API 密钥 |

对于大多数首次用户：选择一个提供商，除非你知道为什么要更改，否则接受默认值。完整的提供商目录，包含环境变量和设置步骤，请参阅[提供商](../integrations/providers.md)页面。

:::caution 最小上下文：64K tokens
Hermes Agent 需要至少 **64,000 tokens** 上下文的模型。上下文窗口更小的模型无法为多步骤工具调用工作流保持足够的工作内存，将在启动时被拒绝。大多数托管模型（Claude、GPT、Gemini、Qwen、DeepSeek）轻松满足此要求。如果你运行本地模型，请将其上下文大小设置为至少 64K（例如 llama.cpp 的 `--ctx-size 65536` 或 Ollama 的 `-c 65536`）。
:::

:::tip
你可以随时通过 `hermes model` 切换提供商 —— 没有锁定。有关所有支持的提供商的完整列表和设置详情，请参阅 [AI 提供商](../integrations/providers.md)。
:::

### 设置如何存储

Hermes 将秘密与普通配置分开：

- **秘密和令牌** → `~/.hermes/.env`
- **非秘密设置** → `~/.hermes/config.yaml`

通过 CLI 设置值是最简单的方式：

```bash
hermes config set model anthropic/claude-opus-4.6
hermes config set terminal.backend docker
hermes config set OPENROUTER_API_KEY sk-or-...
```

正确的值会自动进入正确的文件。

## 3. 运行你的第一次聊天

```bash
hermes            # 经典 CLI
hermes --tui      # 现代 TUI（推荐）
```

你将看到欢迎横幅，显示你的模型、可用工具和技能。使用具体且易于验证的提示：

:::tip 选择你的界面
Hermes 附带两个终端界面：经典的 `prompt_toolkit` CLI 和更新的带模态叠加、鼠标选择和非阻塞输入的 [TUI](../user-guide/tui.md)。两者共享相同的会话、斜杠命令和配置 —— 用 `hermes` 和 `hermes --tui` 分别尝试。
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

- 横幅显示你选择的模型/提供商
- Hermes 回复无错误
- 如需要可以使用工具（终端、文件读取、网络搜索）
- 对话可以正常继续超过一轮

如果这能正常工作，你就过了最难的部分。

## 4. 验证会话工作

在继续之前，确保恢复功能正常：

```bash
hermes --continue    # 恢复最近的会话
hermes -c            # 短形式
```

这应该能把你带回你刚才的会话。如果不能，检查你是否在同一个配置文件以及会话是否实际保存了。这在你管理多个设置或多台机器时很重要。

## 5. 尝试关键功能

### 使用终端

```
❯ What's my disk usage? Show the top 5 largest directories.
```

Agent 代表你运行终端命令并显示结果。

### 斜杠命令

输入 `/` 查看所有命令的自动完成下拉列表：

| 命令 | 功能 |
|---------|-------------|
| `/help` | 显示所有可用命令 |
| `/tools` | 列出可用工具 |
| `/model` | 交互式切换模型 |
| `/personality pirate` | 尝试有趣的人格 |
| `/save` | 保存对话 |

### 多行输入

按 `Alt+Enter` 或 `Ctrl+J` 添加新行。非常适合粘贴代码或编写详细提示。

### 中断 Agent

如果 Agent 花费时间过长，输入新消息并按 Enter —— 它会中断当前任务并切换到你的新指令。`Ctrl+C` 也可以。

## 6. 添加下一层

仅在基础聊天工作之后。根据需要选择：

### 机器人或共享助手

```bash
hermes gateway setup    # 交互式平台配置
```

连接 [Telegram](/docs/user-guide/messaging/telegram)、[Discord](/docs/user-guide/messaging/discord)、[Slack](/docs/user-guide/messaging/slack)、[WhatsApp](/docs/user-guide/messaging/whatsapp)、[Signal](/docs/user-guide/messaging/signal)、[Email](/docs/user-guide/messaging/email) 或 [Home Assistant](/docs/user-guide/messaging/homeassistant)，或 [Microsoft Teams](/docs/user-guide/messaging/teams)。

### 自动化和工具

- `hermes tools` — 调整每个平台的工具访问
- `hermes skills` — 浏览和安装可重用工作流
- Cron — 仅在你的机器人或 CLI 设置稳定后

### 沙盒终端

为了安全，在 Docker 容器或远程服务器上运行 Agent：

```bash
hermes config set terminal.backend docker    # Docker 隔离
hermes config set terminal.backend ssh       # 远程服务器
```

### 语音模式

```bash
pip install "hermes-agent[voice]"
# 包括免费的本地语音转文字 faster-whisper
```

然后在 CLI 中：`/voice on`。按 `Ctrl+B` 录音。参见[语音模式](../user-guide/features/voice-mode.md)。

### 技能

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

```bash
pip install -e '.[acp]'
hermes acp
```

参见 [ACP 编辑器集成](../user-guide/features/acp.md)。

---

## 常见失败模式

这些问题浪费最多时间：

| 症状 | 可能原因 | 修复 |
|---|---|---|
| Hermes 打开但回复为空或损坏 | 提供商认证或模型选择错误 | 重新运行 `hermes model` 并确认提供商、模型和认证 |
| 自定义端点"工作"但返回垃圾 | 基础 URL、模型名称错误或实际上不是 OpenAI 兼容 | 先在独立客户端验证端点 |
| 网关启动但没有人能发送消息 | Bot 令牌、允许列表或平台设置不完整 | 重新运行 `hermes gateway setup` 并检查 `hermes gateway status` |
| `hermes --continue` 找不到旧会话 | 切换了配置文件或会话从未保存 | 检查 `hermes sessions list` 并确认你在正确的配置文件中 |
| 模型不可用或奇怪的回退行为 | 提供商路由或回退设置过于激进 | 在基础提供商稳定之前保持路由关闭 |
| `hermes doctor` 标记配置问题 | 配置值缺失或过时 | 修复配置，在添加功能之前先测试普通聊天 |

## 恢复工具包

当感觉不对劲时，按这个顺序使用：

1. `hermes doctor`
2. `hermes model`
3. `hermes setup`
4. `hermes sessions list`
5. `hermes --continue`
6. `hermes gateway status`

这个序列可以快速将你从"坏了的感觉"恢复到已知状态。

---

## 快速参考

| 命令 | 描述 |
|---------|-------------|
| `hermes` | 开始聊天 |
| `hermes model` | 选择你的 LLM 提供商和模型 |
| `hermes tools` | 配置每个平台启用哪些工具 |
| `hermes setup` | 完整设置向导（一次性配置所有内容） |
| `hermes doctor` | 诊断问题 |
| `hermes update` | 更新到最新版本 |
| `hermes gateway` | 启动消息网关 |
| `hermes --continue` | 恢复上一个会话 |

## 下一步

- **[CLI 指南](../user-guide/cli.md)** — 掌握终端界面
- **[配置](../user-guide/configuration.md)** — 自定义你的设置
- **[消息网关](../user-guide/messaging/index.md)** — 连接 Telegram、Discord、Slack、WhatsApp、Signal、Email、Home Assistant、Teams 等
- **[工具和工具集](../user-guide/features/tools.md)** — 探索可用功能
- **[AI 提供商](../integrations/providers.md)** — 完整的提供商列表和设置详情
- **[技能系统](../user-guide/features/skills.md)** — 可重用工作流和知识
- **[技巧与最佳实践](../guides/tips.md)** — 高级用户技巧
