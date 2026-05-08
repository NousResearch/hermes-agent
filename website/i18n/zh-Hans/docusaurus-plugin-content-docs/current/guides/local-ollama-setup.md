---
sidebar_position: 9
title: "用 Ollama 本地运行 Hermes — 零 API 成本"
description: "分步指南：完全在你的机器上使用 Ollama 和开源模型如 Gemma 4 运行 Hermes Agent，无需云 API key 或付费订阅"
---

# 用 Ollama 本地运行 Hermes — 零 API 成本

## 问题

云 LLM API 按 token 收费。重度编码会话可能花费 $5-20。对于个人项目、学习或隐私敏感工作，这会累积 — 而且你将每次对话发送给第三方。

## 本指南解决的问题

你将设置完全在自己的硬件上运行的 Hermes Agent，使用 [Ollama](https://ollama.com) 作为模型后端。无需 API key、无订阅、无数据离开你的机器。配置完成后，Hermes 工作方式与使用 OpenRouter 或 Anthropic 完全相同 — 终端命令、文件编辑、网络浏览、委托 — 但模型在本地运行。

结束时你将拥有：

- Ollama 提供一个或多个开源模型
- Hermes 连接到 Ollama 作为自定义端点
- 可工作的本地智能体，可以编辑文件、运行命令和浏览网络
- 可选：完全由你自己的硬件驱动的 Telegram/Discord 机器人

## 你需要什么

| 组件 | 最低要求 | 推荐 |
|-----------|---------|-------------|
| **RAM** | 8 GB（用于 3B 模型） | 32+ GB（用于 27B+ 模型） |
| **存储** | 5 GB 可用 | 30+ GB（用于多个模型） |
| **CPU** | 4 核 | 8+ 核（AMD EPYC、Ryzen、Intel Xeon） |
| **GPU** | 不需要 | NVIDIA GPU 8+ GB VRAM 显著加速 |

:::tip 纯 CPU 可以工作，但预期响应较慢
Ollama 在纯 CPU 服务器上运行。现代 8 核 CPU 上的 9B 模型给出约 ~10 tokens/秒。CPU 上的 31B 模型较慢（~2–5 tokens/秒）— 每个响应需要 30-120 秒，但它可以工作。GPU 显著改善这个。对于纯 CPU 设置，在配置中增加 API 超时：

```yaml
agent:
  api_timeout: 1800   # 30 分钟 — 对慢速本地模型慷慨
```
:::

## 步骤 1：安装 Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

验证运行：

```bash
ollama --version
curl http://localhost:11434/api/tags   # 应返回 {"models":[]}
```

## 步骤 2：拉取模型

根据你的硬件选择：

| 模型 | 磁盘大小 | 需要 RAM | 工具调用 | 最适合 |
|-------|-------------|------------|:------------:|----------|
| `gemma4:31b` | ~20 GB | 24+ GB | 是 | 最佳质量 — 强大的工具使用和推理 |
| `gemma2:27b` | ~16 GB | 20+ GB | 否 | 对话任务，无工具使用 |
| `gemma2:9b` | ~5 GB | 8+ GB | 否 | 快速聊天、问答 — 无法调用工具 |
| `llama3.2:3b` | ~2 GB | 4+ GB | 否 | 仅轻量级快速答案 |

:::warning 工具调用很重要
Hermes 是一个**智能体**助手 — 它通过工具调用编辑文件、运行命令和浏览网络。不支持工具调用的模型只能聊天；它们不能执行操作。要获得完整的 Hermes 体验，使用支持工具的模型（如 `gemma4:31b`）。
:::

拉取你选择的模型：

```bash
ollama pull gemma4:31b
```

:::info 多个模型
你可以拉取多个模型并在 Hermes 中用 `/model` 在它们之间切换。Ollama 按需将活动模型加载到内存中，自动卸载空闲模型。
:::

验证模型工作：

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

## 步骤 3：配置 Hermes

运行 Hermes 设置向导：

```bash
hermes setup
```

当提示选择提供商时，选择 **Custom Endpoint** 并输入：

- **Base URL:** `http://localhost:11434/v1`
- **API Key:** 留空或输入 `no-key`（Ollama 不需要）
- **Model:** `gemma4:31b`（或你拉取的模型）

或者，直接编辑 `~/.hermes/config.yaml`：

```yaml
model:
  default: "gemma4:31b"
  provider: "custom"
  base_url: "http://localhost:11434/v1"
```

## 步骤 4：开始使用 Hermes

```bash
hermes
```

就这样。你现在运行一个完全本地的智能体。试试看：

```
你：列出此目录中的所有 Python 文件并计算每个文件的代码行数

你：读取 README.md 并总结这个项目做什么

你：创建一个获取胡志明市天气的 Python 脚本
```

Hermes 将使用终端工具、文件操作和你的本地模型 — 无云调用。

## 步骤 5：为你的任务选择正确的模型

并非每个任务都需要最大的模型。这是一个实用指南：

| 任务 | 推荐模型 | 为什么 |
|------|-------------------|-----|
| 文件编辑、代码、终端命令 | `gemma4:31b` | 唯一具有可靠工具调用的模型 |
| 快速问答（不需要工具使用） | `gemma2:9b` | 对话任务的快速响应 |
| 轻量级聊天 | `llama3.2:3b` | 最快，但能力非常有限 |

:::note
对于完整的智能体工作（编辑文件、运行命令、浏览），`gemma4:31b` 目前是支持工具调用的最佳本地选项。查看 [Ollama 的模型库](https://ollama.com/library) 获取更新的模型 — 工具调用支持正在快速扩展。
:::

会话中动态切换模型：

```
/model gemma2:9b
```

## 步骤 6：优化速度

### 增加 Ollama 的上下文窗口

默认，Ollama 使用 2048 token 上下文。对于智能体工作（工具调用、长对话），你需要更多：

```bash
# 创建扩展上下文的 Modelfile
cat > /tmp/Modelfile << 'EOF'
FROM gemma4:31b
PARAMETER num_ctx 16384
EOF

ollama create gemma4-16k -f /tmp/Modelfile
```

然后更新你的 Hermes 配置使用 `gemma4-16k` 作为模型名称。

### 保持模型加载

默认，Ollama 在 5 分钟不活动后卸载模型。对于持久 gateway 机器人，保持加载：

```bash
# 设置保持活动 24 小时
curl http://localhost:11434/api/generate \
  -d '{"model": "gemma4:31b", "keep_alive": "24h"}'
```

或在 Ollama 的环境中全局设置：

```bash
# /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_KEEP_ALIVE=24h"
```

### 使用 GPU 卸载（如果有）

如果你有 NVIDIA GPU，Ollama 自动将层卸载到它。检查：

```bash
ollama ps   # 显示哪个模型已加载以及多少 GPU 层
```

对于 12 GB GPU 上的 31B 模型，你将获得部分卸载（~40 层在 GPU，其余在 CPU），这仍然给出显著加速。

## 步骤 7：作为 Gateway 机器人运行（可选）

一旦 Hermes 在 CLI 中本地工作，你可以将其作为 Telegram 或 Discord 机器人暴露 — 仍然完全在你的硬件上运行。

### Telegram

1. 通过 [@BotFather](https://t.me/BotFather) 创建机器人并获取 token
2. 添加到你的 `~/.hermes/config.yaml`：

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

现在在 Telegram 上向你的机器人发消息 — 它使用你的本地模型回复。

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

## 步骤 8：设置回退（可选）

本地模型可能在复杂任务上吃力。设置仅在本地模型失败时激活的云回退：

```yaml
model:
  default: "gemma4:31b"
  provider: "custom"
  base_url: "http://localhost:11434/v1"

fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

这样，90% 的使用是免费的（本地），只有困难任务才使用付费 API。

## 故障排除

### 启动时"连接被拒绝"

Ollama 未运行。启动它：

```bash
sudo systemctl start ollama
# 或
ollama serve
```

### 响应慢

- **检查模型大小 vs RAM：** 如果你的模型需要的 RAM 超过可用，它会交换到磁盘。使用更小的模型或增加 RAM。
- **检查 `ollama ps`：** 如果没有 GPU 层卸载，响应受 CPU 限制。这对纯 CPU 服务器是正常的。
- **减少上下文：** 大对话减慢推理。定期使用 `/compress`，或在配置中设置更低的压缩阈值。

### 模型不遵循工具调用

较小的模型（3B、7B）有时忽略工具调用指令，产生纯文本而非结构化函数调用。解决方案：

- **使用更大的模型** — `gemma4:31b` 或 `gemma2:27b` 比 3B/7B 模型处理工具调用好得多。
- **Hermes 有自动修复** — 它检测格式错误的工具调用并尝试自动修复。
- **设置回退** — 如果本地模型失败 3 次，Hermes 回退到云提供商。

### 上下文窗口错误

默认 Ollama 上下文（2048 tokens）对智能体工作太小。见[步骤 6](#step-6-optimize-for-speed) 增加它。

## 成本比较

基于典型编码会话（~100K tokens 输入，~20K tokens 输出），这里是与云 API 相比本地运行节省的内容：

| 提供商 | 每会话成本 | 月度（日常使用） |
|----------|-----------------|---------------------|
| Anthropic Claude Sonnet | ~$0.80 | ~$24 |
| OpenRouter (GPT-4o) | ~$0.60 | ~$18 |
| **Ollama（本地）** | **$0.00** | **$0.00** |

你唯一的成本是电费 — 根据硬件，每个会话约 $0.01-0.05。

## 本地运行良好的内容

- **文件编辑和代码生成** — 9B+ 模型处理良好
- **终端命令** — Hermes 包装命令、运行它、读取输出，无论模型如何
- **网络浏览** — 浏览器工具做获取；模型只是解释结果
- **Cron 作业和计划任务** — 与云设置工作方式相同
- **多平台 gateway** — Telegram、Discord、Slack 都可与本地模型一起工作

## 云模型更好的场景

- **非常复杂的多步推理** — 70B+ 或 Claude Opus 等云模型明显更好
- **长上下文窗口** — 云模型提供 100K–1M tokens；本地模型通常 8K–32K
- **大响应的速度** — 对于长生成，云推理比纯 CPU 本地更快

最佳点：日常任务使用本地，为困难任务设置云回退。
