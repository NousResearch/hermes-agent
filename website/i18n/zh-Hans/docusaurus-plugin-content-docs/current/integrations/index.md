---
title: "集成"
sidebar_label: "概述"
sidebar_position: 0
---

# 集成

Hermes Agent 连接外部系统以实现 AI 推理、工具服务器、IDE 工作流、编程式访问等功能。这些集成扩展了 Hermes 的能力范围和使用场景。

## AI 提供商与路由

Hermes 开箱即用地支持多种 AI 推理提供商。使用 `hermes model` 进行交互式配置，或在 `config.yaml` 中直接设置。

- **[AI 提供商](/docs/user-guide/features/provider-routing)** — OpenRouter、Anthropic、OpenAI、Google 以及任何兼容 OpenAI 的端点。Hermes 会自动检测各提供商的能力，如视觉、流式传输和工具使用。
- **[提供商路由](/docs/user-guide/features/provider-routing)** — 精细控制哪个底层提供商处理你的 OpenRouter 请求。通过排序、白名单、黑名单和显式优先级顺序来优化成本、速度或质量。
- **[备用提供商](/docs/user-guide/features/fallback-providers)** — 当主模型遇到错误时自动故障转移到备用 LLM 提供商。包括主模型备用和视觉、压缩、网页提取的辅助任务独立备用。

## 工具服务器（MCP）

- **[MCP 服务器](/docs/user-guide/features/mcp)** — 通过模型上下文协议（Model Context Protocol）将 Hermes 连接到外部工具服务器。无需编写原生 Hermes 工具即可访问 GitHub、数据库、文件系统、浏览器栈、内部 API 等的工具。支持 stdio 和 SSE 传输、按服务器工具过滤以及感知能力的资源/提示注册。

## 网页搜索后端

`web_search` 和 `web_extract` 工具支持四个后端提供商，通过 `config.yaml` 或 `hermes tools` 配置：

| 后端 | 环境变量 | 搜索 | 提取 | 爬取 |
|---------|---------|--------|---------|-------|
| **Firecrawl**（默认） | `FIRECRAWL_API_KEY` | ✔ | ✔ | ✔ |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | — |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ | ✔ |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | — |

快速设置示例：

```yaml
web:
  backend: firecrawl    # firecrawl | parallel | tavily | exa
```

如果未设置 `web.backend`，则会自动从可用的 API 密钥中检测后端。也支持通过 `FIRECRAWL_API_URL` 使用自托管的 Firecrawl。

## 浏览器自动化

Hermes 包含完整的浏览器自动化功能，提供多种后端选项用于浏览网站、填写表单和提取信息：

- **Browserbase** — 托管云浏览器，具有反机器人绕过、CAPTCHA 求解和住宅代理功能
- **Browser Use** — 替代云浏览器提供商
- **通过 CDP 连接本地 Chrome** — 使用 `/browser connect` 连接到正在运行的 Chrome 实例
- **本地 Chromium** — 通过 `agent-browser` CLI 实现无头本地浏览器

详见[浏览器自动化](/docs/user-guide/features/browser)了解设置和使用方法。

## 语音与 TTS 提供商

跨所有消息平台的文本转语音和语音转文本：

| 提供商 | 质量 | 成本 | API 密钥 |
||----------|---------|------|---------|
|| **Edge TTS**（默认） | 良好 | 免费 | 无需 |
|| **ElevenLabs** | 优秀 | 付费 | `ELEVENLABS_API_KEY` |
|| **OpenAI TTS** | 良好 | 付费 | `VOICE_TOOLS_OPENAI_KEY` |
|| **MiniMax** | 良好 | 付费 | `MINIMAX_API_KEY` |
|| **NeuTTS** | 良好 | 免费 | 无需 |

语音转文本支持六个提供商：本地 faster-whisper（免费，设备上运行）、本地命令包装器、Groq、OpenAI Whisper API、Mistral 和 xAI。语音消息转录在 Telegram、Discord、WhatsApp 和其他消息平台上均可工作。详见[语音与 TTS](/docs/user-guide/features/tts)和[语音模式](/docs/user-guide/features/voice-mode)。

## IDE 与编辑器集成

- **[IDE 集成 (ACP)](/docs/user-guide/features/acp)** — 在支持 ACP 的编辑器（如 VS Code、Zed 和 JetBrains）中使用 Hermes Agent。Hermes 作为 ACP 服务器运行，在你的编辑器内呈现聊天消息、工具活动、文件差异和终端命令。

## 编程式访问

- **[API 服务器](/docs/user-guide/features/api-server)** — 将 Hermes 暴露为兼容 OpenAI 的 HTTP 端点。任何支持 OpenAI 格式的前端 — Open WebUI、LobeChat、LibreChat、NextChat、ChatBox — 都可以连接并使用 Hermes 作为后端，并使用其完整工具集。

## 记忆与个性化

- **[内置记忆](/docs/user-guide/features/memory)** — 通过 `MEMORY.md` 和 `USER.md` 文件实现持久化、精心管理的记忆。Agent 维护跨会话存续的个人信息笔记和用户配置文件数据存储。
- **[记忆提供商](/docs/user-guide/features/memory-providers)** — 接入外部记忆后端以实现更深层次的个性化。支持八个提供商：Honcho（对话推理）、OpenViking（分层检索）、Mem0（云提取）、Hindsight（知识图谱）、Holographic（本地 SQLite）、RetainDB（混合搜索）、ByteRover（基于 CLI）和 Supermemory。

## 消息平台

Hermes 在 19+ 个消息平台上作为网关机器人运行，全部通过同一个 `gateway` 子系统配置：

- **[Telegram](/docs/user-guide/messaging/telegram)**、**[Discord](/docs/user-guide/messaging/discord)**、**_liv[Slack](/docs/user-guide/messaging/slack)**、**_liv[WhatsApp](/docs/user-guide/messaging/whatsapp)**、**_liv[Signal](/docs/user-guide/messaging/signal)**、**_liv[Matrix](/docs/user-guide/messaging/matrix)**、**_liv[Mattermost](/docs/user-guide/messaging/mattermost)**、**_liv[Email](/docs/user-guide/messaging/email)**、**_liv[SMS](/docs/user-guide/messaging/sms)**、**_liv[DingTalk](/docs/user-guide/messaging/dingtalk)**、**_liv[Feishu/Lark](/docs/user-guide/messaging/feishu)**、**_liv[WeCom](/docs/user-guide/messaging/wecom)**、**_liv[WeCom Callback](/docs/user-guide/messaging/wecom-callback)**、**_liv[Weixin](/docs/user-guide/messaging/weixin)**、**_liv[BlueBubbles](/docs/user-guide/messaging/bluebubbles)**、**_liv[QQ Bot](/docs/user-guide/messaging/qqbot)**、**_liv[Yuanbao](/docs/user-guide/messaging/yuanbao)**、**_liv[Home Assistant](/docs/user-guide/messaging/homeassistant)**、**_liv[Microsoft Teams](/docs/user-guide/messaging/teams)**、**[Webhooks](/docs/user-guide/messaging/webhooks)**

详见[消息网关概述](/docs/user-guide/messaging)了解平台对比表和设置指南。

## 家庭自动化

- **[Home Assistant](/docs/user-guide/messaging/homeassistant)** — 通过四个专用工具（`ha_list_entities`、`ha_get_state`、`ha_list_services`、`ha_call_service`）控制智能家居设备。配置 `HASS_TOKEN` 后，Home Assistant 工具集会自动激活。

## 插件

- **[插件系统](/docs/user-guide/features/plugins)** — 无需修改核心代码即可使用自定义工具、生命周期钩子和 CLI 命令扩展 Hermes。插件从 `~/.hermes/plugins/`、项目本地 `.hermes/plugins/` 和 pip 安装的入口点中发现。
- **[构建插件](/docs/guides/build-a-hermes-plugin)** — 使用工具、钩子和 CLI 命令创建 Hermes 插件的逐步指南。

## 训练与评估

- **[RL 训练](/docs/user-guide/features/rl-training)** — 从 Agent 会话中生成轨迹数据，用于强化学习和模型微调。支持带有可自定义奖励函数的 AtroPos 环境。
- **[批处理](/docs/user-guide/features/batch-processing)** — 跨数百个提示并行运行 Agent，生成结构化的 ShareGPT 格式轨迹数据，用于训练数据生成或评估。
