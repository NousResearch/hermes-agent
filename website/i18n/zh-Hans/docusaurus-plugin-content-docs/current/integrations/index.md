---
title: "集成"
sidebar_label: "总览"
sidebar_position: 0
---

# 集成

Hermes Agent 可以连接外部系统，用于 AI 推理、工具服务器、IDE 工作流、程序化访问等。这些集成扩展了 Hermes 的能力边界，以及它能够运行的场景。

## AI 提供商与路由

- **[AI 提供商](/integrations/providers)** - 按平台快速选择和配置推理提供商。
- **[提供商路由](/user-guide/features/provider-routing)** - 精细控制 OpenRouter 请求由哪些底层提供商处理。
- **[备用提供商](/user-guide/features/fallback-providers)** - 当主模型出错时自动切换到备用 LLM 提供商。

## 工具服务器（MCP）

- **[MCP 服务器](/user-guide/features/mcp)** - 通过 Model Context Protocol 连接外部工具服务器，支持 stdio 与 SSE 传输及按服务器过滤工具。

## 网页搜索后端

`web_search` 与 `web_extract` 支持四种后端：

| 后端 | 环境变量 | 搜索 | 提取 | 爬取 |
|------|----------|------|------|------|
| Firecrawl（默认） | `FIRECRAWL_API_KEY` | ✔ | ✔ | ✔ |
| Parallel | `PARALLEL_API_KEY` | ✔ | ✔ | — |
| Tavily | `TAVILY_API_KEY` | ✔ | ✔ | ✔ |
| Exa | `EXA_API_KEY` | ✔ | ✔ | — |

## 浏览器自动化

Hermes 支持 Browserbase、Browser Use、本地 Chrome（CDP）和本地 Chromium。详见 [浏览器自动化](/user-guide/features/browser)。

## 语音与 TTS

语音相关能力覆盖 CLI 与消息平台，支持 Edge TTS、ElevenLabs、OpenAI、MiniMax 等。详见 [Voice & TTS](/user-guide/features/tts) 与 [Voice Mode](/user-guide/features/voice-mode)。

## IDE 与编辑器集成

- **[IDE 集成（ACP）](/user-guide/features/acp)** - 在 VS Code、Zed、JetBrains 等 ACP 编辑器中使用 Hermes。

## 程序化访问

- **[API 服务器](/user-guide/features/api-server)** - 以 OpenAI-compatible HTTP 端点暴露 Hermes。

## 记忆与个性化

- **[内置记忆](/user-guide/features/memory)** - 使用 `MEMORY.md` 与 `USER.md` 实现跨会话记忆。
- **[记忆提供商](/user-guide/features/memory-providers)** - 接入 Honcho、Mem0、Hindsight、Supermemory 等外部记忆后端。

## 消息平台

Hermes 网关支持 Telegram、Discord、Slack、WhatsApp、Signal、Matrix、Mattermost、Email、SMS、DingTalk、Feishu、WeCom、Weixin、QQ Bot、Teams、Webhooks 等平台。详见 [消息网关总览](/user-guide/messaging)。

## 插件

- **[插件系统](/user-guide/features/plugins)** - 通过工具、钩子、CLI 命令扩展 Hermes。
- **[构建插件](/guides/build-a-hermes-plugin)** - 插件开发分步指南。

## 训练与评估

- **[批处理](/user-guide/features/batch-processing)** - 从会话生成训练轨迹。
- **[批量处理](/user-guide/features/batch-processing)** - 并行处理大量提示词并导出结构化结果。
