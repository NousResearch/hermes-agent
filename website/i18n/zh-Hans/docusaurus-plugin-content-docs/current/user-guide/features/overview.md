---
title: "功能概览"
sidebar_label: "概览"
sidebar_position: 1
---

# 功能概览

Hermes Agent 包含一套丰富的能力，远超基础聊天范畴。从持久化记忆、文件感知上下文，到浏览器自动化和语音对话，这些功能协同工作，使 Hermes 成为一个强大的自主助手。

## 核心功能

- **[工具与工具集](tools.md)** — 工具是扩展 Agent 能力的函数。它们被组织成逻辑工具集，可按平台启用或禁用，涵盖网络搜索、终端执行、文件编辑、记忆、委派等功能。
- **[技能系统](skills.md)** — Agent 可按需加载的知识文档。技能遵循渐进式披露模式以最小化 token 用量，并兼容 [agentskills.io](https://agentskills.io/specification) 开放标准。
- **[持久化记忆](memory.md)** — 跨会话持久保存的有界、精选记忆。Hermes 通过 `MEMORY.md` 和 `USER.md` 记住你的偏好、项目、环境及已学习的内容。
- **[上下文文件](context-files.md)** — Hermes 自动发现并加载项目上下文文件（`.hermes.md`、`AGENTS.md`、`CLAUDE.md`、`SOUL.md`、`.cursorrules`），这些文件决定了它在你项目中的行为方式。
- **[上下文引用](context-references.md)** — 输入 `@` 后跟引用内容，可将文件、文件夹、git diff 和 URL 直接注入消息中。Hermes 会内联展开引用并自动附加相应内容。
- **[检查点](../checkpoints-and-rollback.md)** — Hermes 在进行文件更改前自动为工作目录创建快照，提供安全网，可通过 `/rollback` 回滚至出错前的状态。

## 自动化

- **[定时任务（Cron）](cron.md)** — 使用自然语言或 cron 表达式调度自动运行的任务。任务可附加技能、将结果推送至任意平台，并支持暂停/恢复/编辑操作。
- **[子 Agent 委派](delegation.md)** — `delegate_task` 工具可生成具有独立上下文、受限工具集和独立终端会话的子 Agent 实例。默认并发运行 3 个子 Agent（可配置），支持并行工作流。
- **[代码执行](code-execution.md)** — `execute_code` 工具允许 Agent 编写以编程方式调用 Hermes 工具的 Python 脚本，通过沙箱 RPC 执行将多步骤工作流压缩为单次 LLM 调用。
- **[事件 Hook](hooks.md)** — 在关键生命周期节点运行自定义代码。Gateway hook 处理日志、告警和 webhook；plugin hook 处理工具拦截、指标和护栏。
- **[批处理](batch-processing.md)** — 跨数百或数千个 prompt（提示词）并行运行 Hermes Agent，生成 ShareGPT 格式的结构化轨迹数据，用于训练数据生成或评估。

## 媒体与网络

- **[语音模式](voice-mode.md)** — 跨 CLI 和消息平台的完整语音交互。使用麦克风与 Agent 对话，收听语音回复，并在 Discord 语音频道中进行实时语音对话。
- **[浏览器自动化](browser.md)** — 支持多种后端的完整浏览器自动化：Browserbase 云端、Browser Use 云端、通过 CDP 连接的本地 Chrome/Brave/Chromium/Edge，或本地 Chromium。可导航网站、填写表单并提取信息。
- **[视觉与图片粘贴](vision.md)** — 多模态视觉支持。将剪贴板中的图片粘贴到 CLI，并使用任意支持视觉的模型请求 Agent 分析、描述或处理图片。
- **[图像生成](image-generation.md)** — 使用 FAL.ai 从文本 prompt 生成图像。支持九种模型（FLUX 2 Klein/Pro、GPT-Image 1.5/2、Nano Banana Pro、Ideogram V3、Recraft V4 Pro、Qwen、Z-Image Turbo）；可通过 `hermes tools` 选择。
- **[语音与 TTS](tts.md)** — 跨所有消息平台的文字转语音输出和语音消息转录，提供十种原生提供商选项：Edge TTS（免费）、ElevenLabs、OpenAI TTS、MiniMax、Mistral Voxtral、Google Gemini、xAI、NeuTTS、KittenTTS 和 Piper——以及支持任意本地 TTS CLI 的自定义命令提供商。

## 集成

- **[MCP 集成](mcp.md)** — 通过 stdio 或 HTTP 传输连接任意 MCP 服务器。无需编写原生 Hermes 工具，即可访问来自 GitHub、数据库、文件系统和内部 API 的外部工具。支持按服务器过滤工具及 sampling（采样）。
- **[提供商路由](provider-routing.md)** — 对 AI 提供商处理请求的方式进行精细控制。通过排序、白名单、黑名单和优先级排序，在成本、速度或质量之间优化。
- **[备用提供商](fallback-providers.md)** — 当主模型遇到错误时自动故障转移至备用 LLM 提供商，包括针对视觉和压缩等辅助任务的独立备用机制。
- **[凭证池](credential-pools.md)** — 在同一提供商的多个密钥之间分发 API 调用。在触发速率限制或发生故障时自动轮换。
- **[Prompt 缓存](../configuration#prompt-caching)** — 针对原生 Anthropic、OpenRouter 和 Nous Portal 上的 Claude，内置跨会话 1 小时前缀缓存。始终开启，无需配置。
- **[记忆提供商](memory-providers.md)** — 接入外部记忆后端（Honcho、OpenViking、Mem0、Hindsight、Holographic、RetainDB、ByteRover、Supermemory），实现跨会话用户建模和超越内置记忆系统的个性化。
- **[API 服务器](api-server.md)** — 将 Hermes 作为兼容 OpenAI 的 HTTP 端点暴露。连接任何支持 OpenAI 格式的前端——Open WebUI、LobeChat、LibreChat 等。
- **[IDE 集成（ACP）](acp.md)** — 在兼容 ACP 的编辑器（如 VS Code、Zed 和 JetBrains）中使用 Hermes。聊天、工具活动、文件 diff 和终端命令均在编辑器内渲染。
- **[强化学习训练](rl-training.md)** — 从 Agent 会话中生成轨迹数据，用于强化学习和模型微调。

## 自定义

- **[个性与 SOUL.md](personality.md)** — 完全可自定义的 Agent 个性。`SOUL.md` 是主要身份文件——系统提示词中的第一项——你可以在每个会话中切换内置或自定义的 `/personality` 预设。
- **[皮肤与主题](skins.md)** — 自定义 CLI 的视觉呈现：横幅颜色、加载动画图标和动词、响应框标签、品牌文字，以及工具活动前缀。
- **[插件](plugins.md)** — 无需修改核心代码即可添加自定义工具、hook 和集成。三种插件类型：通用插件（工具/hook）、记忆提供商（跨会话知识）和上下文引擎（替代上下文管理）。通过统一的 `hermes plugins` 交互式界面管理。