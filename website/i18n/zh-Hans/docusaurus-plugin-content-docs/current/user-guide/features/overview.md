---
title: "功能特性概览"
sidebar_label: "概览"
sidebar_position: 1
---

# 功能特性概览

Hermes Agent 包含一组丰富的能力，远远超出了基础聊天。从持久化内存和文件感知上下文到浏览器自动化和语音对话，这些功能协同工作，使 Hermes 成为强大的自主助手。

## 核心

- **[工具与工具集](tools.md)** — 工具是扩展智能体能力的函数。它们被组织成可按平台启用或禁用的逻辑工具集，涵盖网络搜索、终端执行、文件编辑、内存、委派等。
- **[技能系统](skills.md)** — 智能体按需加载的知识文档。技能遵循渐进式披露模式以最小化 token 使用量，并与 [agentskills.io](https://agentskills.io/specification) 开放标准兼容。
- **[持久化内存](memory.md)** — 有界、精心策划的内存，跨会话持久化。Hermes 通过 `MEMORY.md` 和 `USER.md` 记住您的偏好、项目、环境和它学到的东西。
- **[上下文文件](context-files.md)** — Hermes 自动发现并加载项目上下文文件（`.hermes.md`、`AGENTS.md`、`CLAUDE.md`、`SOUL.md`、`.cursorrules`），这些文件塑造它在您项目中的行为方式。
- **[上下文引用](context-references.md)** — 输入 `@` 后跟引用来将文件、文件夹、git 差异和 URL 直接注入到您的消息中。Hermes 自动内联展开引用并追加内容。
- **[检查点](../checkpoints-and-rollback.md)** — Hermes 在进行文件更改前自动对工作目录进行快照，为您提供安全网，以便在出现问题时使用 `/rollback` 回滚。

## 自动化

- **[定时任务 (Cron)](cron.md)** — 使用自然语言或 cron 表达式安排任务自动运行。任务可以附加技能、将结果交付到任何平台，并支持暂停/恢复/编辑操作。
- **[子智能体委派](delegation.md)** — `delegate_task` 工具生成具有隔离上下文、受限工具集和自身终端会话的子智能体实例。默认并行运行 3 个并发子智能体（可配置）。
- **[代码执行](code-execution.md)** — `execute_code` 工具让智能体编写以编程方式调用 Hermes 工具的 Python 脚本，通过沙盒 RPC 执行将多步骤工作流压缩为单个 LLM 轮次。
- **[事件钩子](hooks.md)** — 在关键生命周期点运行自定义代码。网关钩子处理日志记录、告警和 webhook；插件钩子处理工具拦截、指标和护栏。
- **[批处理](batch-processing.md)** — 在数百或数千个提示上并行运行 Hermes 智能体，生成结构化 ShareGPT 格式的轨迹数据，用于训练数据生成或评估。

## 媒体与 Web

- **[语音模式](voice-mode.md)** — 跨 CLI 和消息平台的完整语音交互。使用麦克风与智能体交谈，听到语音回复，并在 Discord 语音频道中进行实时语音对话。
- **[浏览器自动化](browser.md)** — 具有多个后端的完整浏览器自动化：Browserbase 云、Browser Use 云、通过 CDP 的本地 Chrome 或本地 Chromium。导航网站、填写表单和提取信息。
- **[视觉与图像粘贴](vision.md)** — 多模态视觉支持。将图像从剪贴板粘贴到 CLI 中，并要求智能体使用任何支持视觉的模型分析、描述或处理它们。
- **[图像生成](image-generation.md)** — 使用 FAL.ai 从文本提示生成图像。支持九种模型（FLUX 2 Klein/Pro、GPT-Image 1.5/2、Nano Banana Pro、Ideogram V3、Recraft V4 Pro、Qwen、Z-Image Turbo）；通过 `hermes tools` 选择一种。
- **[语音与 TTS](tts.md)** — 跨所有消息平台的文本转语音输出和语音消息转录，具有十种原生提供商选项：Edge TTS（免费）、ElevenLabs、OpenAI TTS、MiniMax、Mistral Voxtral、Google Gemini、xAI、NeuTTS、KittenTTS 和 Piper —— 以及用于任何本地 TTS CLI 的自定义命令提供商。

## 集成

- **[MCP 集成](mcp.md)** — 通过 stdio 或 HTTP 传输连接到任何 MCP 服务器。无需编写原生 Hermes 工具即可访问来自 GitHub、数据库、文件系统和内部 API 的外部工具。包括按服务器的工具过滤和采样支持。
- **[提供商路由](provider-routing.md)** — 精细控制哪些 AI 提供商处理您的请求。通过排序、白名单、黑名单和优先级排序来优化成本、速度或质量。
- **[备用提供商](fallback-providers.md)** — 当主模型遇到错误时自动故障转移到备用 LLM 提供商，包括视觉和压缩等辅助任务的独立备用。
- **[凭证池](credential-pools.md)** — 为同一提供商分发多个密钥的 API 调用。在速率限制或失败时自动轮换。
- **[内存提供商](memory-providers.md)** — 插入外部内存后端（Honcho、OpenViking、Mem0、Hindsight、Holographic、RetainDB、ByteRover、Supermemory），用于超出内置内存系统的跨会话用户建模和个性化。
- **[API 服务器](api-server.md)** — 将 Hermes 公开为 OpenAI-compatible HTTP 端点。连接任何使用 OpenAI 格式的前端 —— Open WebUI、LobeChat、LibreChat 等。
- **[IDE 集成 (ACP)](acp.md)** — 在 ACP-compatible 编辑器（如 VS Code、Zed 和 JetBrains）中使用 Hermes。聊天、工具活动、文件差异和终端命令在您的编辑器中渲染。
- **[批处理](batch-processing.md)** — 从智能体会话生成轨迹数据，用于训练数据生成和评估。

## 自定义

- **[个性与 SOUL.md](personality.md)** — 完全可定制的智能体个性。`SOUL.md` 是主身份文件 —— 系统提示中的第一件事 —— 您可以按会话交换内置或自定义 `/personality` 预设。
- **[皮肤与主题](skins.md)** — 自定义 CLI 的视觉呈现：横幅颜色、spinner 表情和动词、响应框标签、品牌文本和工具活动前缀。
- **[插件](plugins.md)** — 无需修改核心代码即可添加自定义工具、钩子和集成。三种插件类型：通用插件（工具/钩子）、内存提供商（跨会话知识）和上下文引擎（替代上下文管理）。通过统一的 `hermes plugins` 交互式 UI 管理。
