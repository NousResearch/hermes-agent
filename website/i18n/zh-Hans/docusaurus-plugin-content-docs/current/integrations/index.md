---
title: "集成"
sidebar_label: "概览"
sidebar_position: 0
---

# 集成

Hermes Agent 可连接外部系统，用于 AI 推理、工具服务器、IDE 工作流、程序化访问等。这些集成扩展了 Hermes 的能力边界与运行环境。

## AI 提供商与路由

Hermes 开箱即支持多个 AI 推理提供商。使用 `hermes model` 进行交互式配置，或在 `config.yaml` 中直接设置。

- **[AI 提供商](/user-guide/features/provider-routing)** — OpenRouter、Anthropic、OpenAI、Google 以及任何兼容 OpenAI 的端点。Hermes 会自动检测每个提供商的能力，包括视觉、流式传输和工具调用。
- **[提供商路由](/user-guide/features/provider-routing)** — 精细控制哪些底层提供商处理你的 OpenRouter 请求。通过排序、白名单、黑名单和显式优先级排序，在成本、速度或质量之间优化。
- **[备用提供商](/user-guide/features/fallback-providers)** — 当主模型遇到错误时，自动故障转移到备用 LLM 提供商。包括主模型回退，以及用于视觉、压缩和网页提取的独立辅助任务回退。

## 工具服务器（MCP）

- **[MCP 服务器](/user-guide/features/mcp)** — 通过 Model Context Protocol 将 Hermes 连接到外部工具服务器。无需编写原生 Hermes 工具，即可访问来自 GitHub、数据库、文件系统、浏览器栈、内部 API 等的工具。支持 stdio 和 SSE 两种传输方式、按服务器过滤工具，以及具备能力感知的资源/prompt 注册。

## 网页搜索后端

`web_search` 和 `web_extract` 工具支持四个后端提供商，通过 `config.yaml` 或 `hermes tools` 配置：

| 后端 | 环境变量 | 搜索 | 提取 | 爬取 |
|---------|---------|--------|---------|-------|
| **Firecrawl**（默认） | `FIRECRAWL_API_KEY` | ✔ | ✔ | ✔ |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | — |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ | ✔ |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | — |

快速配置示例：

```yaml
web:
  backend: firecrawl    # firecrawl | searxng | brave-free | ddgs | tavily | exa | parallel | xai
```

若未设置 `web.backend`，后端将根据可用的 API key 自动检测。也支持通过 `FIRECRAWL_API_URL` 使用自托管的 Firecrawl。

## 浏览器自动化

Hermes 内置完整的浏览器自动化功能，提供多种后端选项，用于网站导航、表单填写和信息提取：

- **Browserbase** — 托管云端浏览器，具备反机器人工具、CAPTCHA 解决和住宅代理
- **Browser Use** — 备选云端浏览器提供商
- **本地 Chromium 系 CDP** — 使用 `/browser connect` 连接正在运行的 Chrome、Brave、Chromium 或 Edge 浏览器
- **本地 Chromium** — 通过 `agent-browser` CLI 使用无头本地浏览器

详见[浏览器自动化](/user-guide/features/browser)的配置与使用说明。

## 语音与 TTS 提供商

跨所有消息平台的文字转语音与语音转文字：

| 提供商 | 质量 | 费用 | API Key |
|----------|---------|------|---------|
| **Edge TTS**（默认） | 良好 | 免费 | 无需 |
| **ElevenLabs** | 优秀 | 付费 | `ELEVENLABS_API_KEY` |
| **OpenAI TTS** | 良好 | 付费 | `VOICE_TOOLS_OPENAI_KEY` |
| **MiniMax** | 良好 | 付费 | `MINIMAX_API_KEY` |
| **NeuTTS** | 良好 | 免费 | 无需 |

语音转文字支持六个提供商：本地 faster-whisper（免费，设备端运行）、本地命令封装器、Groq、OpenAI Whisper API、Mistral 和 xAI。语音消息转录支持 Telegram、Discord、WhatsApp 及其他消息平台。详见[语音与 TTS](/user-guide/features/tts) 和[语音模式](/user-guide/features/voice-mode)。

## IDE 与编辑器集成

- **[IDE 集成（ACP）](/user-guide/features/acp)** — 在兼容 ACP 的编辑器（如 VS Code、Zed 和 JetBrains）中使用 Hermes Agent。Hermes 作为 ACP 服务器运行，在编辑器内渲染聊天消息、工具活动、文件差异和终端命令。

## 程序化访问

- **[API 服务器](/user-guide/features/api-server)** — 将 Hermes 暴露为兼容 OpenAI 的 HTTP 端点。任何支持 OpenAI 格式的前端——Open WebUI、LobeHub、LibreChat、NextChat、ChatBox——均可连接并将 Hermes 作为后端使用，享有其完整工具集。

## 记忆与个性化

- **[内置记忆](/user-guide/features/memory)** — 通过 `MEMORY.md` 和 `USER.md` 文件实现持久化、精选记忆。Agent 维护有界的个人笔记和用户画像数据存储，跨会话保留。
- **[记忆提供商](/user-guide/features/memory-providers)** — 接入外部记忆后端以实现更深度的个性化。支持八个提供商：Honcho（辩证推理）、OpenViking（分层检索）、Mem0（云端提取）、Hindsight（知识图谱）、Holographic（本地 SQLite）、RetainDB（混合搜索）、ByteRover（基于 CLI）和 Supermemory。

## 消息平台

Hermes 可作为 gateway（网关）机器人运行于 19+ 个消息平台，均通过同一 `gateway` 子系统配置：

- **[Telegram](/user-guide/messaging/telegram)**、**[Discord](/user-guide/messaging/discord)**、**[Slack](/user-guide/messaging/slack)**、**[WhatsApp](/user-guide/messaging/whatsapp)**、**[Signal](/user-guide/messaging/signal)**、**[Matrix](/user-guide/messaging/matrix)**、**[Mattermost](/user-guide/messaging/mattermost)**、**[Email](/user-guide/messaging/email)**、**[SMS](/user-guide/messaging/sms)**、**[DingTalk](/user-guide/messaging/dingtalk)**、**[Feishu/Lark](/user-guide/messaging/feishu)**、**[WeCom](/user-guide/messaging/wecom)**、**[WeCom Callback](/user-guide/messaging/wecom-callback)**、**[Weixin](/user-guide/messaging/weixin)**、**[BlueBubbles](/user-guide/messaging/bluebubbles)**、**[QQ Bot](/user-guide/messaging/qqbot)**、**[Yuanbao](/user-guide/messaging/yuanbao)**、**[Home Assistant](/user-guide/messaging/homeassistant)**、**[Microsoft Teams](/user-guide/messaging/teams)**、**[Webhooks](/user-guide/messaging/webhooks)**

平台对比表和配置指南详见[消息 Gateway 概览](/user-guide/messaging)。

## 家庭自动化

- **[Home Assistant](/user-guide/messaging/homeassistant)** — 通过四个专用工具（`ha_list_entities`、`ha_get_state`、`ha_list_services`、`ha_call_service`）控制智能家居设备。配置 `HASS_TOKEN` 后，Home Assistant 工具集将自动激活。

## 插件

- **[插件系统](/user-guide/features/plugins)** — 无需修改核心代码，通过自定义工具、生命周期 hook（钩子）和 CLI 命令扩展 Hermes。插件从 `~/.hermes/plugins/`、项目本地 `.hermes/plugins/` 以及通过 pip 安装的入口点自动发现。
- **[构建插件](/developer-guide/plugins)** — 创建包含工具、hook 和 CLI 命令的 Hermes 插件的分步指南。

## 训练与评估

- **[批处理](/user-guide/features/batch-processing)** — 并行跨数百个 prompt（提示词）运行 Agent，生成结构化的 ShareGPT 格式轨迹数据，用于训练数据生成或评估。