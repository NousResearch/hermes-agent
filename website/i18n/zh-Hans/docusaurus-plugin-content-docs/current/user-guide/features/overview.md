---
title: "功能概览"
sidebar_label: "概览"
sidebar_position: 1
---

# 功能概览

Hermes 代理包含丰富的功能集，远超基本聊天。从持久记忆和文件感知上下文到浏览器自动化和语音对话，这些功能协同工作，使 Hermes 成为强大的自主助手。

## 核心

- **[工具和工具集](tools.md)** — 工具是扩展代理能力的函数。它们被组织成逻辑工具集，可以按平台启用或禁用，涵盖网络搜索、终端执行、文件编辑、记忆、委托等。
- **[技能系统](skills.md)** — 按需知识文档，代理可以在需要时加载。技能遵循渐进式披露模式以最小化令牌使用，并与 [agentskills.io](https://agentskills.io/specification) 开放标准兼容。
- **[持久记忆](memory.md)** — 有界的、策划的记忆，跨会话持久。Hermes 记住你的偏好、项目、环境以及它通过 `MEMORY.md` 和 `USER.md` 学到的东西。
- **[上下文文件](context-files.md)** — Hermes 自动发现并加载项目上下文文件（`.hermes.md`、`AGENTS.md`、`CLAUDE.md`、`SOUL.md`、`.cursorrules`），这些文件塑造它在你的项目中的行为方式。
- **[上下文引用](context-references.md)** — 输入 `@` 后跟引用，将文件、文件夹、git 差异和 URL 直接注入你的消息中。Hermes 内联展开引用并自动附加内容。
- **[检查点和回滚](../checkpoints-and-rollback.md)** — Hermes 在进行文件更改前自动快照你的工作目录，如果出现问题，给你一个安全网使用 `/rollback` 回滚。

## 自动化

- **[计划任务（Cron）](cron.md)** — 使用自然语言或 cron 表达式安排任务自动运行。作业可以附加技能、将结果交付到任何平台，并支持暂停/恢复/编辑操作。
- **[子代理委托](delegation.md)** — `delegate_task` 工具生成具有隔离上下文、限制工具集和各自终端会话的子代理实例。默认运行 3 个并发子代理（可配置）用于并行工作流。
- **[代码执行](code-execution.md)** — `execute_code` 工具让代理编写调用 Hermes 工具的 Python 脚本，通过沙箱 RPC 执行将多步骤工作流折叠为单个 LLM 轮次。
- **[事件钩子](hooks.md)** — 在关键生命周期点运行自定义代码。网关钩子处理日志记录、警报和 webhooks；插件钩子处理工具拦截、指标和护栏。
- **[批处理](batch-processing.md)** — 跨数百或数千个提示并行运行 Hermes 代理，生成结构化 ShareGPT 格式的轨迹数据用于训练数据生成或评估。

## 媒体和 Web

- **[语音模式](voice-mode.md)** — 跨 CLI 和消息平台的完整语音交互。使用麦克风与代理对话、听到语音回复，并在 Discord 语音频道中进行实时语音对话。
- **[浏览器自动化](browser.md)** — 多个后端的完整浏览器自动化：Browserbase 云、Browser Use 云、通过 CDP 的本地 Chrome 或本地 Chromium。导航网站、填写表单并提取信息。
- **[视觉和图像粘贴](vision.md)** — 多模态视觉支持。将图像从剪贴板粘贴到 CLI 中，并使用任何有视觉能力的模型要求代理分析、描述或处理它们。
- **[图像生成](image-generation.md)** — 使用 FAL.ai 从文本提示生成图像。支持九种模型（FLUX 2 Klein/Pro、GPT-Image 1.5/2、Nano Banana Pro、Ideogram V3、Recraft V4 Pro、Qwen、Z-Image Turbo）；通过 `hermes tools` 选择一个。
- **[语音和 TTS](tts.md)** — 跨所有消息平台的文本转语音输出和语音消息转录，有十个原生提供商选项：Edge TTS（免费）、ElevenLabs、OpenAI TTS、MiniMax、Mistral Voxtral、Google Gemini、xAI、NeuTTS、KittenTTS 和 Piper——加上任何本地 TTS CLI 的自定义命令提供商。

## 集成

- **[MCP 集成](mcp.md)** — 通过 stdio 或 HTTP 传输连接到任何 MCP 服务器。访问 GitHub、数据库、文件系统和内部 API 的外部工具，无需编写原生 Hermes 工具。包括每个服务器的工具有过滤和采样支持。
- **[提供商路由](provider-routing.md)** — 对处理请求的 AI 提供商进行细粒度控制。通过排序、白名单、黑名单和优先级排序优化成本、速度或质量。
- **[后备提供商](fallback-providers.md)** — 当你的主要模型遇到错误时自动故障转移到备份 LLM 提供商，包括视觉和压缩等辅助任务的独立后备。
- **[凭据池](credential-pools.md)** — 跨同一提供商的多密钥分发 API 调用。在速率限制或失败时自动轮换。
- **[记忆提供商](memory-providers.md)** — 插入外部记忆后端（Honcho、OpenViking、Mem0、Hindsight、Holographic、RetainDB、ByteRover、Supermemory），用于内置记忆系统之外的跨会话用户建模和个性化。
- **[API 服务器](api-server.md)** — 将 Hermes 公开为 OpenAI 兼容的 HTTP 端点。连接任何使用 OpenAI 格式的前端——Open WebUI、LobeChat、LibreChat 等。
- **[IDE 集成（ACP）](acp.md)** — 在 ACP 兼容的编辑器（如 VS Code、Zed 和 JetBrains）中使用 Hermes。聊天、工具活动、文件差异和终端命令在你的编辑器内渲染。
- **[强化学习训练](rl-training.md)** — 从代理会话生成轨迹数据用于强化学习和模型微调。

## 定制

- **[个性和 SOUL.md](personality.md)** — 完全可定制的代理个性。`SOUL.md` 是主要身份文件——系统提示中的第一项——你可以在每个会话中交换内置或自定义的 `/personality` 预设。
- **[皮肤和主题](skins.md)** — 自定义 CLI 的视觉呈现：横幅颜色、旋转器表情和动词、响应框标签、品牌文本和工具活动前缀。
- **[插件](plugins.md)** — 无需修改核心代码即可添加自定义工具、钩子和集成。三种插件类型：通用插件（工具/钩子）、记忆提供商（跨会话知识）和上下文引擎（替代上下文管理）。通过统一的 `hermes plugins` 交互式 UI 管理。
