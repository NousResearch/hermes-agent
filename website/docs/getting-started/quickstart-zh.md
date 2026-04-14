---
sidebar_position: 2
title: "快速入门"
description: "Hermes Agent 中文使用指南 — 从安装到对话，2分钟上手"
---

# 快速入门

本指南将帮助你安装 Hermes Agent、配置 LLM 提供商，并开始你的第一次对话。完成后，你将了解核心功能和如何进一步探索。

## 1. 安装 Hermes Agent

运行一行命令安装：

```bash
# Linux / macOS / WSL2 / Android (Termux)
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

:::tip Android / Termux
如果你在手机上安装，请查看专门的 [Termux 指南](./termux.md)，了解经过测试的手动安装路径、支持的附加功能和当前的 Android 特定限制。
:::

:::tip Windows 用户
先安装 [WSL2](https://learn.microsoft.com/zh-cn/windows/wsl/install)，然后在 WSL2 终端中运行上面的命令。
:::

安装完成后，重新加载你的 shell：

```bash
source ~/.bashrc   # 或 source ~/.zshrc
```

## 2. 配置 LLM 提供商

安装程序会自动配置你的 LLM 提供商。如果以后需要更改，可以使用以下命令：

```bash
hermes model       # 选择你的 LLM 提供商和模型
hermes tools       # 配置启用的工具
hermes setup       # 或一次性配置所有内容
```

`hermes model` 会引导你选择推理提供商：

| 提供商 | 说明 | 配置方式 |
|--------|------|----------|
| **Nous Portal** | 基于订阅，零配置 | 通过 `hermes model` 进行 OAuth 登录 |
| **OpenAI Codex** | ChatGPT OAuth，使用 Codex 模型 | 通过 `hermes model` 设备码认证 |
| **Anthropic** | 直接使用 Claude 模型（Pro/Max 或 API key） | `hermes model` 配合 Claude Code 认证，或 Anthropic API key |
| **OpenRouter** | 多提供商路由，支持多种模型 | 输入你的 API key |
| **Z.AI** | GLM / 智谱托管模型 | 设置 `GLM_API_KEY` / `ZAI_API_KEY` |
| **Kimi / Moonshot** | Moonshot 托管的编码和聊天模型 | 设置 `KIMI_API_KEY` |
| **MiniMax** | 国际 MiniMax 端点 | 设置 `MINIMAX_API_KEY` |
| **MiniMax 中国** | 中国区 MiniMax 端点 | 设置 `MINIMAX_CN_API_KEY` |
| **阿里云** | 通过 DashScope 使用 Qwen 模型 | 设置 `DASHSCOPE_API_KEY` |
| **Hugging Face** | 20+ 开放模型（Qwen, DeepSeek, Kimi 等） | 设置 `HF_TOKEN` |
| **DeepSeek** | 直接 DeepSeek API 访问 | 设置 `DEEPSEEK_API_KEY` |
| **GitHub Copilot** | GitHub Copilot 订阅（GPT-5.x, Claude, Gemini 等） | 通过 `hermes model` OAuth，或 `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` |
| **自定义端点** | VLLM, SGLang, Ollama，或任何兼容 OpenAI 的 API | 设置基础 URL + API key |

:::caution 最小上下文：64K tokens
Hermes Agent 要求模型至少支持 **64,000 tokens** 的上下文。较小窗口的模型无法为多步骤工具调用工作流维持足够的工作内存，启动时会被拒绝。大多数托管模型（Claude, GPT, Gemini, Qwen, DeepSeek）都能轻松满足。如果你运行本地模型，请将其上下文大小设置为至少 64K（例如 llama.cpp 使用 `--ctx-size 65536`，Ollama 使用 `-c 65536`）。
:::

:::tip
你可以随时使用 `hermes model` 切换提供商 — 无需更改代码，没有锁定。配置自定义端点时，Hermes 会提示输入上下文窗口大小，并在可能时自动检测。详情请参阅 [上下文长度检测](../integrations/providers.md#context-length-detection)。
:::

## 3. 开始对话

```bash
hermes
```

就这样！你会看到一个欢迎横幅，显示你的模型、可用工具和技能。输入消息并按 Enter。

```
❯ 你需要什么帮助？
```

代理拥有网页搜索、文件操作、终端命令等工具 — 开箱即用。

## 4. 体验核心功能

### 让它使用终端

```
❯ 我的磁盘使用情况如何？显示最大的 5 个目录。
```

代理会代表你运行终端命令并显示结果。

### 使用斜杠命令

输入 `/` 查看所有命令的自动补全下拉菜单：

| 命令 | 功能 |
|------|------|
| `/help` | 显示所有可用命令 |
| `/tools` | 列出可用工具 |
| `/model` | 交互式切换模型 |
| `/personality pirate` | 尝试有趣的个性 |
| `/save` | 保存对话 |

### 多行输入

按 `Alt+Enter` 或 `Ctrl+J` 添加新行。非常适合粘贴代码或编写详细提示。

### 中断代理

如果代理运行时间过长，只需输入新消息并按 Enter — 它会中断当前任务并切换到你的新指令。`Ctrl+C` 也有效。

### 恢复会话

退出时，hermes 会打印恢复命令：

```bash
hermes --continue    # 恢复最近的会话
hermes -c            # 简短形式
```

## 5. 进一步探索

以下是一些可以尝试的内容：

### 设置沙箱终端

为安全起见，在 Docker 容器或远程服务器中运行代理：

```bash
hermes config set terminal.backend docker    # Docker 隔离
hermes config set terminal.backend ssh       # 远程服务器
```

### 连接消息平台

通过 Telegram、Discord、Slack、WhatsApp、Signal、邮件或 Home Assistant 从手机或其他设备与 Hermes 聊天：

```bash
hermes gateway setup    # 交互式平台配置
```

### 添加语音模式

想要在 CLI 中使用麦克风输入或在消息中使用语音回复？

```bash
pip install "hermes-agent[voice]"
# 包含 faster-whisper，免费本地语音转文字
```

然后启动 Hermes 并在 CLI 中启用：

```text
/voice on
```

按 `Ctrl+B` 录音，或使用 `/voice tts` 让 Hermes 朗读回复。详情请参阅 [语音模式](../user-guide/features/voice-mode.md)。

### 安排自动化任务

```
❯ 每天早上 9 点，检查 Hacker News 的 AI 新闻并在 Telegram 上发送摘要给我。
```

代理会设置一个定时任务，通过网关自动运行。

### 浏览和安装技能

```bash
hermes skills search kubernetes
hermes skills search react --source skills-sh
hermes skills search https://mintlify.com/docs --source well-known
hermes skills install openai/skills/k8s
hermes skills install official/security/1password
hermes skills install skills-sh/vercel-labs/json-render/json-render-react --force
```

提示：
- 使用 `--source skills-sh` 搜索公共 `skills.sh` 目录。
- 使用 `--source well-known` 加文档/网站 URL，从 `/.well-known/skills/index.json` 发现技能。
- 仅在审查第三方技能后使用 `--force`。它可以覆盖非危险策略块，但不能覆盖 `dangerous` 扫描结果。

或者在聊天中使用 `/skills` 斜杠命令。

### 通过 ACP 在编辑器中使用 Hermes

Hermes 也可以作为 ACP 服务器，供 VS Code、Zed 和 JetBrains 等兼容 ACP 的编辑器使用：

```bash
pip install -e '.[acp]'
hermes acp
```

详情请参阅 [ACP 编辑器集成](../user-guide/features/acp.md)。

### 尝试 MCP 服务器

通过模型上下文协议连接外部工具：

```yaml
# 添加到 ~/.hermes/config.yaml
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxx"
```

---

## 快速参考

| 命令 | 说明 |
|------|------|
| `hermes` | 开始聊天 |
| `hermes model` | 选择 LLM 提供商和模型 |
| `hermes tools` | 按平台配置启用的工具 |
| `hermes setup` | 完整设置向导（一次性配置所有内容） |
| `hermes doctor` | 诊断问题 |
| `hermes update` | 更新到最新版本 |
| `hermes gateway` | 启动消息网关 |
| `hermes --continue` | 恢复上次会话 |

## 下一步

- **[CLI 指南](../user-guide/cli.md)** — 掌握终端界面
- **[配置](../user-guide/configuration.md)** — 自定义你的设置
- **[消息网关](../user-guide/messaging/index.md)** — 连接 Telegram、Discord、Slack、WhatsApp、Signal、邮件或 Home Assistant
- **[工具和工具集](../user-guide/features/tools.md)** — 探索可用功能
