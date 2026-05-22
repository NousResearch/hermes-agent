---
slug: /
sidebar_position: 0
title: "Hermes Agent 文档"
description: "Nous Research 打造的自我改进 AI 智能体。内置学习循环，可从经验中创建技能、在使用中改进，并跨会话记忆。"
hide_table_of_contents: true
displayed_sidebar: docs
---

# Hermes Agent

由 [Nous Research](https://nousresearch.com) 构建的**自我进化 AI 智能体**。它是唯一拥有内置学习循环的智能体——能从经验中创建技能、在使用过程中不断改进、主动提示自己持久化知识，并在多次会话间构建对你的深度理解模型。

<div style={{display: 'flex', gap: '1rem', marginBottom: '2rem', flexWrap: 'wrap'}}>
  <a href="/getting-started/installation" style={{display: 'inline-block', padding: '0.6rem 1.2rem', backgroundColor: '#FFD700', color: '#07070d', borderRadius: '8px', fontWeight: 600, textDecoration: 'none'}}>开始使用 →</a>
  <a href="https://github.com/NousResearch/hermes-agent" style={{display: 'inline-block', padding: '0.6rem 1.2rem', border: '1px solid rgba(255,215,0,0.2)', borderRadius: '8px', textDecoration: 'none'}}>在 GitHub 上查看</a>
</div>

## 安装

**Linux / macOS / WSL2**

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

**Windows（原生，PowerShell）** — *早期测试版，[详情 →](/user-guide/windows-native)*

```powershell
irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1 | iex
```

**Android（Termux）** — 与 Linux 使用相同的 curl 一键安装命令；安装程序会自动检测 Termux 环境。

查看完整的 **[安装指南](/getting-started/installation)**，了解安装程序的具体行为、单用户与 root 布局差异，以及 Windows 相关注意事项。

## 什么是 Hermes Agent？

它不是一个绑定在 IDE 上的编程辅助工具，也不是围绕单一 API 的聊天机器人包装器。它是一个**自主智能体**，运行时间越长，能力越强。它可以部署在你选择的任何地方——一台 5 美元的 VPS、一个 GPU 集群，或是几乎空闲时零成本的无服务器基础设施（Daytona、Modal）。你可以在 Telegram 上与之对话，而它在云端虚拟机上执行任务，你甚至不需要亲自 SSH 登录。它不绑定在你的笔记本电脑上。

## 快速链接

| | |
|---|---|
| 🚀 **[安装](/getting-started/installation)** | 在 Linux、macOS、WSL2 或原生 Windows（早期测试版）上 60 秒完成安装 |
| 📖 **[快速入门教程](/getting-started/quickstart)** | 你的第一次对话以及值得尝试的关键功能 |
| 🗺️ **[学习路径](/getting-started/learning-path)** | 根据你的经验水平找到合适的文档 |
| ⚙️ **[配置](/user-guide/configuration)** | 配置文件、提供商、模型和选项 |
| 💬 **[消息网关](/user-guide/messaging)** | 设置 Telegram、Discord、Slack、WhatsApp、Teams 等 |
| 🔧 **[工具与工具集](/user-guide/features/tools)** | 70+ 内置工具及其配置方法 |
| 🧠 **[记忆系统](/user-guide/features/memory)** | 跨会话持续增长的持久化记忆 |
| 📚 **[技能系统](/user-guide/features/skills)** | 智能体创建和复用的程序性记忆 |
| 🔌 **[MCP 集成](/user-guide/features/mcp)** | 连接 MCP 服务器，过滤其工具，安全扩展 Hermes |
| 🧭 **[在 Hermes 中使用 MCP](/guides/use-mcp-with-hermes)** | 实用的 MCP 设置模式、示例和教程 |
| 🎙️ **[语音模式](/user-guide/features/voice-mode)** | 在 CLI、Telegram、Discord 和 Discord 语音频道中进行实时语音交互 |
| 🗣️ **[在 Hermes 中使用语音模式](/guides/use-voice-mode-with-hermes)** | Hermes 语音工作流的实践设置和使用模式 |
| 🎭 **[个性与 SOUL.md](/user-guide/features/personality)** | 使用全局 SOUL.md 定义 Hermes 的默认风格 |
| 📄 **[上下文文件](/user-guide/features/context-files)** | 塑造每次对话的项目上下文文件 |
| 🔒 **[安全](/user-guide/security)** | 命令审批、授权、容器隔离 |
| 💡 **[技巧与最佳实践](/guides/tips)** | 快速掌握 Hermes 的实用技巧 |
| 🏗️ **[架构](/developer-guide/architecture)** | 底层工作原理 |
| ❓ **[常见问题与故障排除](/reference/faq)** | 常见问题和解决方案 |

## 核心功能

- **闭环学习系统** — 智能体自主管理的记忆，配合定期提示；自主技能创建、使用中的技能自我改进、基于 FTS5 的跨会话召回配合 LLM 摘要，以及 [Honcho](https://github.com/plastic-labs/honcho) 辩证式用户建模
- **随处运行，不限于笔记本** — 6 种终端后端：本地、Docker、SSH、Daytona、Singularity、Modal。Daytona 和 Modal 提供无服务器持久化——环境在空闲时休眠，几乎零成本
- **融入你的生活** — CLI、Telegram、Discord、Slack、WhatsApp、Signal、Matrix、Mattermost、Email、SMS、钉钉、飞书、企业微信、微信、QQ 机器人、元宝、BlueBubbles、Home Assistant、Microsoft Teams、Google Chat 等——一个网关支持 20+ 平台
- **由模型训练者构建** — 由 [Nous Research](https://nousresearch.com) 创建，Hermes、Nomos 和 Psyche 背后的实验室。支持 [Nous Portal](https://portal.nousresearch.com)、[OpenRouter](https://openrouter.ai)、OpenAI 或任何端点
- **定时自动化** — 内置 cron，可投递到任何平台
- **委派与并行** — 生成隔离的子智能体进行并行工作流。通过 `execute_code` 进行程序化工具调用，将多步管道压缩为单次推理调用
- **开放标准技能** — 兼容 [agentskills.io](https://agentskills.io)。技能可移植、可共享，并通过技能中心社区贡献
- **完整的 Web 控制** — 搜索、提取、浏览、视觉、图像生成、TTS
- **MCP 支持** — 连接任何 MCP 服务器以扩展工具能力
- **研究就绪** — 批处理、轨迹导出、使用 Atropos 进行 RL 训练。由 [Nous Research](https://nousresearch.com) 构建——Hermes、Nomos 和 Psyche 模型背后的实验室

## 面向 LLM 和编程智能体

机器可读的文档入口点：

- **[`/llms.txt`](/llms.txt)** — 每个文档页面的精选索引及简短描述。约 17 KB，适合加载到 LLM 上下文中。
- **[`/llms-full.txt`](/llms-full.txt)** — 所有文档页面合并为单个 Markdown 文件，便于一次性摄取。约 1.8 MB。

两个文件也可在 `/docs/llms.txt` 和 `/docs/llms-full.txt` 访问。每次部署时重新生成。
