---
slug: /
sidebar_position: 0
title: "Hermes Agent 文档"
description: "由 Nous Research 构建的自我改进 AI Agent。内置学习循环，从经验中创建技能、在使用中改进技能，并在不同会话间保持记忆。"
hide_table_of_contents: true
displayed_sidebar: docs
---

# Hermes Agent

由 [Nous Research](https://nousresearch.com) 构建的自我改进 AI Agent。唯一具有内置学习循环的 Agent —— 它从经验中创建技能、在使用中改进技能、推动自身保持知识，并在不同会话间建立对你的深入了解。

<div style={{display: 'flex', gap: '1rem', marginBottom: '2rem', flexWrap: 'wrap'}}>
  <a href="/docs/getting-started/installation" style={{display: 'inline-block', padding: '0.6rem 1.2rem', backgroundColor: '#FFD700', color: '#07070d', borderRadius: '8px', fontWeight: 600, textDecoration: 'none'}}>开始使用 →</a>
  <a href="https://github.com/NousResearch/hermes-agent" style={{display: 'inline-block', padding: '0.6rem 1.2rem', border: '1px solid rgba(255,215,0,0.2)', borderRadius: '8px', textDecoration: 'none'}}>在 GitHub 上查看</a>
</div>

## Hermes Agent 是什么？

它不是绑定在 IDE 上的编码副驾驶，也不是围绕单一 API 的聊天机器人包装器。它是一个**自主 Agent**，运行时间越长，能力越强。它可以放在任何地方 —— 5 美元的 VPS、GPU 集群，或者几乎空闲时不产生费用的无服务器基础设施（Daytona、Modal）。你可以在 Telegram 上与它对话，而它正在你从不 SSH 登录的云 VM 上工作。它不依赖你的笔记本电脑。

## 快速链接

| | |
|---|---|
| 🚀 **[安装](/docs/getting-started/installation)** | 在 Linux、macOS 或 WSL2 上 60 秒安装 |
| 📖 **[快速入门教程](/docs/getting-started/quickstart)** | 你的第一次对话和可尝试的关键功能 |
| 🗺️ **[学习路径](/docs/getting-started/learning-path)** | 根据你的经验水平找到合适的文档 |
| ⚙️ **[配置](/docs/user-guide/configuration)** | 配置文件、提供商、模型和选项 |
| 💬 **[消息网关](/docs/user-guide/messaging)** | 设置 Telegram、Discord、Slack、WhatsApp、Teams 等 |
| 🔧 **[工具和工具集](/docs/user-guide/features/tools)** | 68 个内置工具及配置方法 |
| 🧠 **[记忆系统](/docs/user-guide/features/memory)** | 跨会话增长的持久记忆 |
| 📚 **[技能系统](/docs/user-guide/features/skills)** | Agent 创建和重用的程序性记忆 |
| 🔌 **[MCP 集成](/docs/user-guide/features/mcp)** | 连接到 MCP 服务器、过滤其工具并安全扩展 Hermes |
| 🧭 **[在 Hermes 中使用 MCP](/docs/guides/use-mcp-with-hermes)** | 实用的 MCP 设置模式、示例和教程 |
| 🎙️ **[语音模式](/docs/user-guide/features/voice-mode)** | 在 CLI、Telegram、Discord 和 Discord VC 中进行实时语音交互 |
| 🗣️ **[在 Hermes 中使用语音模式](/docs/guides/use-voice-mode-with-hermes)** | Hermes 语音工作流程的动手设置和使用模式 |
| 🎭 **[人格与 SOUL.md](/docs/user-guide/features/personality)** | 通过全局 SOUL.md 定义 Hermes 的默认声音 |
| 📄 **[上下文文件](/docs/user-guide/features/context-files)** | 塑造每次对话的项目上下文文件 |
| 🔒 **[安全](/docs/user-guide/security)** | 命令审批、授权、容器隔离 |
| 💡 **[技巧与最佳实践](/docs/guides/tips)** | 充分利用 Hermes 的快速技巧 |
| 🏗️ **[架构](/docs/developer-guide/architecture)** | 内部工作原理 |
| ❓ **[常见问题与故障排除](/docs/reference/faq)** | 常见问题及解决方案 |

## 关键特性

- **闭环学习系统** —— Agent 精心整理的记忆，带有周期性提醒、自主技能创建、使用中的技能自我改进、FTS5 跨会话召回与大语言模型摘要，以及 [Honcho](https://github.com/plastic-labs/honcho) 辩证用户建模
- **可在任何地方运行，不仅仅是你的笔记本电脑** —— 6 种终端后端：本地、Docker、SSH、Daytona、Singularity、Modal。Daytona 和 Modal 提供无服务器持久化 —— 你的环境在空闲时休眠，几乎不产生费用
- **在你所在的地方生活** —— CLI、Telegram、Discord、Slack、WhatsApp、Signal、Matrix、Mattermost、Email、SMS、DingTalk、Feishu、WeCom、BlueBubbles、Home Assistant、Microsoft Teams —— 一个网关覆盖 15+ 平台
- **由模型训练师构建** —— 由 [Nous Research](https://nousresearch.com) 创建，该实验室是 Hermes、Nomos 和 Psyche 的幕后推手。可与 [Nous Portal](https://portal.nousresearch.com)、[OpenRouter](https://openrouter.ai)、OpenAI 或任何端点配合使用
- **定时自动化** —— 内置 cron，支持投递到任何平台
- **委托与并行化** —— 为并行工作流生成隔离的子 Agent。通过 `execute_code` 进行程序化工具调用，将多步骤管道折叠为单次推理调用
- **开放的标准化技能** —— 与 [agentskills.io](https://agentskills.io) 兼容。技能可移植、可共享，通过技能中心由社区贡献
- **完整的网络控制** —— 搜索、提取、浏览、视觉、图片生成、TTS
- **MCP 支持** —— 连接任何 MCP 服务器以扩展工具能力
- **研究就绪** —— 批处理、轨迹导出、使用 Atropos 进行 RL 训练。由 [Nous Research](https://nousresearch.com) 构建 —— 该实验室是 Hermes、Nomos 和 Psyche 模型的幕后推手

## 适用于大语言模型和编码 Agent

机器可读的文档入口点：

- **[`/llms.txt`](/llms.txt)** —— 每个文档页面的精选索引，包含简短描述。约 17 KB，可安全加载到 LLM 上下文中。
- **[`/llms-full.txt`](/llms-full.txt)** —— 每个文档页面合并为单个 markdown 文件，便于一次性摄取。约 1.8 MB。

这两个文件也可通过 `/docs/llms.txt` 和 `/docs/llms-full.txt` 访问。每次部署时动态生成。
