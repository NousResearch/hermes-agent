---
sidebar_position: 9
title: "使用 Ollama 在本地运行 Hermes — 零 API 费用"
description: "使用 Ollama 和 Gemma 4 等开放权重模型在本机完整运行 Hermes Agent 的分步指南，无需云端 API 密钥或付费订阅"
---

# 使用 Ollama 在本地运行 Hermes — 零 API 费用

## 问题所在

云端 LLM API 按 token（令牌）计费。一次高强度的编程会话可能花费 5–20 美元。对于个人项目、学习或隐私敏感的工作，费用会不断累积——而且你的每一段对话都会发送给第三方。

## 本指南解决什么

你将在自己的硬件上完整运行 Hermes Agent，使用 [Ollama](https://ollama.com) 作为模型后端。无需 API 密钥，无需订阅，数据不会离开你的机器。配置完成后，Hermes 的使用体验与 OpenRouter 或 Anthropic 完全一致——终端命令、文件编辑、网页浏览、任务委派——只是模型在本地运行。

完成后，你将拥有：

- Ollama 提供一个或多个开放权重模型的服务
- Hermes 通过自定义端点连接到 Ollama
- 一个可以编辑文件、执行命令、浏览网页的本地 agent
- 可选：由你自己的硬件驱动的 Telegram/Discord 机器人

## 所需条件

| 组件 | 最低配置 | 推荐配置 |
|-----------|---------|-------------|
| **内存** | 8 GB（适用于 3B 模型） | 32+ GB（适用于 27B+ 模型） |
| **存储** | 5 GB 可用空间 | 30+ GB（适用于多个模型） |
| **CPU** | 4 核 | 8+ 核（AMD EPYC、Ryzen、Intel Xeon） |
| **GPU** | 非必需 | 配备 8+ GB 显存的 NVIDIA GPU 可显著提速 |

:::tip 仅 CPU 可用，但响应速度较慢
Ollama 可在纯 CPU 服务器上运行。现代 8 核 CPU 运行 9B 模型约可达 ~10 tokens/sec。31B 模型在 CPU 上更慢（~2–5 tokens/sec）——每次响应需要 30–120 秒，但可以正常工作。GPU 能大幅改善这一情况。对于纯 CPU 环境，通过环境变量（而非 `config.yaml` 键）放宽 API 超时时间：

```bash
# ~/.hermes/.env
HERMES_API_TIMEOUT=1800   # 30 分钟 — 为慢速本地模型留出充裕时间
```
:::

## 第一步：安装 Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

验证是否正在运行：

```bash
ollama --version
curl http://localhost:11434/api/tags   # 应返回 {"models":[]}
```

## 第二步：拉取模型

根据你的硬件选择：

| 模型 | 磁盘占用 | 所需内存 | 工具调用 | 适用场景 |
|-------|-------------|------------|:------------:|----------|
| `gemma4:31b` | ~20 GB | 24+ GB | 支持 | 最佳质量——工具使用和推理能力强 |
| `gemma2:27b` | ~16 GB | 20+ GB | 不支持 | 对话任务，不支持工具使用 |
| `gemma2:9b` | ~5 GB | 8+ GB | 不支持 | 快速问答——无法调用工具 |
| `llama3.2:3b` | ~2 GB | 4+ GB | 不支持 | 仅适合轻量级快速回答 |

:::warning 工具调用至关重要
Hermes 是一个**agentic（智能体）**助手——它通过工具调用来编辑文件、执行命令和浏览网页。不支持工具调用的模型只能进行对话，无法执行操作。要体验完整的 Hermes 功能，请使用支持工具的模型（如 `gemma4:31b`）。
:::

拉取你选择的模型：

```bash
ollama pull gemma4:31b
```

:::info 多个模型
你可以拉取多个模型，并在 Hermes 中使用 `/model` 切换。Ollama 按需将活跃模型加载到内存，并自动卸载空闲模型。
:::

验证模型是否正常工作：

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4:31b",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 50
  }'
```

你应该看到包含模型回复的 JSON 响应。

## 第三步：配置 Hermes

运行 Hermes 设置向导：

```bash
hermes setup
```

当提示选择提供商时，选择 **Custom Endpoint**，并输入：

- **Base URL：** `http://localhost:11434/v1`
- **API Key：** 留空或输入 `no-key`（Ollama 不需要密钥）
- **Model：** `gemma4:31b`（或你拉取的模型）

也可以直接编辑 `~/.hermes/config.yaml`：

```yaml
model:
  default: "gemma4:31b"
  provider: "custom"
  base_url: "http://localhost:11434/v1"
```

## 第四步：开始使用 Hermes

```bash
hermes
```

就这样。你现在运行的是一个完全本地化的 agent。试试看：

```
You: List all Python files in this directory and count the lines of code in each

You: Read the README.md and summarize what this project does

You: Create a Python script that fetches the weather for Ho Chi Minh City
```

Hermes 将使用终端工具、文件操作和你的本地模型——无需任何云端调用。

## 第五步：为任务选择合适的模型

并非每个任务都需要最大的模型。以下是实用指南：

| 任务 | 推荐模型 | 原因 |
|------|-------------------|-----|
| 文件编辑、代码、终端命令 | `gemma4:31b` | 唯一具备可靠工具调用能力的模型 |
| 快速问答（无需工具调用） | `gemma2:9b` | 对话任务响应速度快 |
| 轻量级聊天 | `llama3.2:3b` | 最快，但能力非常有限 |

:::note
对于完整的 agentic 工作（编辑文件、执行命令、浏览网页），`gemma4:31b` 目前是支持工具调用的最佳本地选项。请关注 [Ollama 的模型库](https://ollama.com/library) 以获取更新模型——工具调用支持正在快速扩展。
:::

在会话中即时切换模型：

```
/model gemma2:9b
```

## 第六步：优化速度

### 增大 Ollama 的上下文窗口

默认情况下，Ollama 使用 2048 token 的上下文。对于 agentic 工作（工具调用、长对话），需要更大的上下文：

```bash
# 创建一个扩展上下文的 Modelfile
cat > /tmp/Modelfile << 'EOF'
FROM gemma4:31b
PARAMETER num_ctx 16384
EOF

ollama create gemma4-16k -f /tmp/Modelfile
```

然后将 Hermes 配置中的模型名称更新为 `gemma4-16k`。

### 保持模型常驻内存

默认情况下，Ollama 在模型空闲 5 分钟后将其卸载。对于持久化的 gateway 机器人，保持模型常驻：

```bash
# 将 keep-alive 设置为 24 小时
curl http://localhost:11434/api/generate \
  -d '{"model": "gemma4:31b", "keep_alive": "24h"}'
```

或在 Ollama 的环境变量中全局设置：

```bash
# /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_KEEP_ALIVE=24h"
```

### 使用 GPU 卸载（如有）

如果你有 NVIDIA GPU，Ollama 会自动将层卸载到 GPU。通过以下命令检查：

```bash
ollama ps   # 显示已加载的模型及 GPU 层数
```

对于 12 GB 显存 GPU 上的 31B 模型，你将获得部分卸载（约 40 层在 GPU 上，其余在 CPU 上），仍能带来显著的速度提升。

## 第七步：作为 Gateway 机器人运行（可选）

一旦 Hermes 在 CLI 中本地运行正常，你可以将其作为 Telegram 或 Discord 机器人对外提供服务——仍完全运行在你的硬件上。

### Telegram

1. 通过 [@BotFather](https://t.me/BotFather) 创建机器人并获取 token
2. 添加到 `~/.hermes/config.yaml`：

```yaml
model:
  default: "gemma4:31b"
  provider: "custom"
  base_url: "http://localhost:11434/v1"

platforms:
  telegram:
    enabled: true
    token: "YOUR_TELEGRAM_BOT_TOKEN"
```

3. 启动 gateway：

```bash
hermes gateway
```

现在在 Telegram 上给你的机器人发消息——它将使用你的本地模型进行响应。

### Discord

1. 在 [discord.com/developers](https://discord.com/developers/applications) 创建 Discord 应用
2. 添加到配置：

```yaml
platforms:
  discord:
    enabled: true
    token: "YOUR_DISCORD_BOT_TOKEN"
```

3. 启动：`hermes gateway`

## 第八步：设置回退方案（可选）

本地模型在处理复杂任务时可能力不从心。设置一个仅在本地模型失败时激活的云端回退：

```yaml
model:
  default: "gemma4:31b"
  provider: "custom"
  base_url: "http://localhost:11434/v1"

fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

这样，90% 的使用是免费的（本地），只有困难任务才会调用付费 API。

## 故障排查

### 启动时出现"Connection refused"

Ollama 未在运行。启动它：

```bash
sudo systemctl start ollama
# 或
ollama serve
```

### 响应缓慢

- **检查模型大小与内存：** 如果模型所需内存超过可用内存，会发生磁盘交换。请使用更小的模型或增加内存。
- **检查 `ollama ps`：** 如果没有 GPU 层被卸载，响应受 CPU 限制。这对于纯 CPU 服务器是正常现象。
- **减少上下文：** 长对话会降低推理速度。定期使用 `/compress`，或在配置中设置更低的压缩阈值。

### 模型不遵循工具调用

较小的模型（3B、7B）有时会忽略工具调用指令，输出纯文本而非结构化的函数调用。解决方案：

- **使用更大的模型** —— `gemma4:31b` 或 `gemma2:27b` 处理工具调用的能力远优于 3B/7B 模型。
- **Hermes 具备自动修复功能** —— 它能检测格式错误的工具调用并自动尝试修复。
- **设置回退方案** —— 如果本地模型连续失败 3 次，Hermes 将回退到云端提供商。

### 上下文窗口错误

Ollama 默认上下文（2048 token）对于 agentic 工作来说太小。请参阅[第六步](#step-6-optimize-for-speed)了解如何增大上下文。

## 费用对比

以下是与云端 API 相比，本地运行的节省情况，基于典型编程会话（约 10 万 token 输入，约 2 万 token 输出）：

| 提供商 | 每次会话费用 | 每月费用（每日使用） |
|----------|-----------------|---------------------|
| Anthropic Claude Sonnet | ~$0.80 | ~$24 |
| OpenRouter（GPT-4o） | ~$0.60 | ~$18 |
| **Ollama（本地）** | **$0.00** | **$0.00** |

你唯一的成本是电费——根据硬件不同，每次会话约 $0.01–0.05。

## 本地运行效果好的场景

- **文件编辑和代码生成** —— 9B+ 模型处理效果良好
- **终端命令** —— Hermes 封装命令、执行并读取输出，与模型无关
- **网页浏览** —— 浏览器工具负责抓取内容，模型只需解读结果
- **定时任务（Cron job）和计划任务** —— 与云端设置完全一致
- **多平台 gateway** —— Telegram、Discord、Slack 均可与本地模型配合使用

## 云端模型更具优势的场景

- **非常复杂的多步推理** —— 70B+ 或 Claude Opus 等云端模型明显更强
- **长上下文窗口** —— 云端模型提供 10 万–100 万 token；本地模型通常为 8K–32K
- **大篇幅响应的速度** —— 对于长文本生成，云端推理比纯 CPU 本地运行更快

最佳策略：日常任务使用本地模型，困难任务设置云端回退。