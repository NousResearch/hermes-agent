---
sidebar_position: 3
title: '学习路径'
description: '根据您的经验水平和目标，选择适合您的 Hermes Agent 文档学习路径。'
---

# 学习路径

Hermes Agent 功能丰富——CLI 助手、Telegram/Discord 机器人、任务自动化、强化学习训练等。本页帮助您根据自身经验水平和目标，确定从哪里开始、阅读哪些内容。

:::tip 从这里开始
如果您尚未安装 Hermes Agent，请先阅读[安装指南](/getting-started/installation)，然后完成[快速入门](/getting-started/quickstart)。以下内容均假设您已完成安装。
:::

## 如何使用本页

- **已知自己的水平？** 跳转至[按经验水平](#by-experience-level)表格，按照对应层级的阅读顺序进行。
- **有明确目标？** 跳至[按使用场景](#by-use-case)，找到匹配的场景。
- **随便浏览？** 查看[主要功能](#key-features-at-a-glance)表格，快速了解 Hermes Agent 的全部能力。

## 按经验水平

| 水平 | 目标 | 推荐阅读 | 预计时间 |
|---|---|---|---|
| **初级** | 快速上手，进行基本对话，使用内置工具 | [安装](/getting-started/installation) → [快速入门](/getting-started/quickstart) → [CLI 用法](/user-guide/cli) → [配置](/user-guide/configuration) | 约 1 小时 |
| **中级** | 搭建消息机器人，使用记忆、cron 任务、技能等高级功能 | [会话](/user-guide/sessions) → [消息](/user-guide/messaging) → [工具](/user-guide/features/tools) → [技能](/user-guide/features/skills) → [记忆](/user-guide/features/memory) → [Cron](/user-guide/features/cron) | 约 2–3 小时 |
| **高级** | 构建自定义工具、创建技能、使用强化学习训练模型、参与项目贡献 | [架构](/developer-guide/architecture) → [添加工具](/developer-guide/adding-tools) → [创建技能](/developer-guide/creating-skills) → [强化学习训练](/user-guide/features/rl-training) → [贡献指南](/developer-guide/contributing) | 约 4–6 小时 |

## 按使用场景

选择与您目标匹配的场景，每个场景均按推荐顺序链接到相关文档。

### "我想要一个 CLI 编程助手"

将 Hermes Agent 用作交互式终端助手，用于编写、审查和运行代码。

1. [安装](/getting-started/installation)
2. [快速入门](/getting-started/quickstart)
3. [CLI 用法](/user-guide/cli)
4. [代码执行](/user-guide/features/code-execution)
5. [上下文文件](/user-guide/features/context-files)
6. [技巧与窍门](/guides/tips)

:::tip
通过上下文文件将文件直接传入对话。Hermes Agent 可以读取、编辑并运行您项目中的代码。
:::

### "我想要一个 Telegram/Discord 机器人"

将 Hermes Agent 部署为您常用消息平台上的机器人。

1. [安装](/getting-started/installation)
2. [配置](/user-guide/configuration)
3. [消息概览](/user-guide/messaging)
4. [Telegram 配置](/user-guide/messaging/telegram)
5. [Discord 配置](/user-guide/messaging/discord)
6. [语音模式](/user-guide/features/voice-mode)
7. [在 Hermes 中使用语音模式](/guides/use-voice-mode-with-hermes)
8. [安全](/user-guide/security)

完整项目示例请参阅：
- [每日简报机器人](/guides/daily-briefing-bot)
- [团队 Telegram 助手](/guides/team-telegram-assistant)

### "我想自动化任务"

调度周期性任务、运行批处理作业，或将多个 agent 动作串联起来。

1. [快速入门](/getting-started/quickstart)
2. [Cron 调度](/user-guide/features/cron)
3. [批处理](/user-guide/features/batch-processing)
4. [委派](/user-guide/features/delegation)
5. [Hooks](/user-guide/features/hooks)

:::tip
Cron 任务让 Hermes Agent 按计划执行任务——每日摘要、定期检查、自动报告——无需您在场。
:::

### "我想构建自定义工具/技能"

通过自定义工具和可复用技能包扩展 Hermes Agent。

1. [插件](/user-guide/features/plugins)
2. [构建 Hermes 插件](/guides/build-a-hermes-plugin)
3. [工具概览](/user-guide/features/tools)
4. [技能概览](/user-guide/features/skills)
5. [MCP（模型上下文协议）](/user-guide/features/mcp)
6. [架构](/developer-guide/architecture)
7. [添加工具](/developer-guide/adding-tools)
8. [创建技能](/developer-guide/creating-skills)

:::tip
对于大多数自定义工具的创建，建议从插件开始。[添加工具](/developer-guide/adding-tools)页面面向 Hermes 核心内置开发，而非常规用户/自定义工具路径。
:::

### "我想训练模型"

使用强化学习（RL）通过 Hermes Agent 内置的 RL 训练流水线对模型行为进行微调。

1. [快速入门](/getting-started/quickstart)
2. [配置](/user-guide/configuration)
3. [强化学习训练](/user-guide/features/rl-training)
4. [Provider 路由](/user-guide/features/provider-routing)
5. [架构](/developer-guide/architecture)

:::tip
强化学习训练在您已了解 Hermes Agent 如何处理对话和工具调用的基础上效果最佳。如果您是新手，请先完成初级路径。
:::

### "我想将其作为 Python 库使用"

以编程方式将 Hermes Agent 集成到您自己的 Python 应用中。

1. [安装](/getting-started/installation)
2. [快速入门](/getting-started/quickstart)
3. [Python 库指南](/guides/python-library)
4. [架构](/developer-guide/architecture)
5. [工具](/user-guide/features/tools)
6. [会话](/user-guide/sessions)

## 主要功能一览

不确定有哪些功能？以下是主要功能的快速目录：

| 功能 | 说明 | 链接 |
|---|---|---|
| **工具** | Agent 可调用的内置工具（文件 I/O、搜索、Shell 等） | [工具](/user-guide/features/tools) |
| **技能** | 可安装的插件包，用于添加新能力 | [技能](/user-guide/features/skills) |
| **记忆** | 跨会话的持久化记忆 | [记忆](/user-guide/features/memory) |
| **上下文文件** | 将文件和目录传入对话 | [上下文文件](/user-guide/features/context-files) |
| **MCP** | 通过模型上下文协议连接外部工具服务器 | [MCP](/user-guide/features/mcp) |
| **Cron** | 调度周期性 agent 任务 | [Cron](/user-guide/features/cron) |
| **委派** | 生成子 agent 以并行处理工作 | [委派](/user-guide/features/delegation) |
| **代码执行** | 运行以编程方式调用 Hermes 工具的 Python 脚本 | [代码执行](/user-guide/features/code-execution) |
| **浏览器** | 网页浏览与抓取 | [浏览器](/user-guide/features/browser) |
| **Hooks** | 事件驱动的回调与中间件 | [Hooks](/user-guide/features/hooks) |
| **批处理** | 批量处理多个输入 | [批处理](/user-guide/features/batch-processing) |
| **强化学习训练** | 使用强化学习微调模型 | [强化学习训练](/user-guide/features/rl-training) |
| **Provider 路由** | 在多个 LLM provider 之间路由请求 | [Provider 路由](/user-guide/features/provider-routing) |

## 下一步阅读

根据您当前所处阶段：

- **刚完成安装？** → 前往[快速入门](/getting-started/quickstart)，运行您的第一次对话。
- **完成了快速入门？** → 阅读 [CLI 用法](/user-guide/cli)和[配置](/user-guide/configuration)，自定义您的设置。
- **已熟悉基础？** → 探索[工具](/user-guide/features/tools)、[技能](/user-guide/features/skills)和[记忆](/user-guide/features/memory)，释放 agent 的全部能力。
- **为团队部署？** → 阅读[安全](/user-guide/security)和[会话](/user-guide/sessions)，了解访问控制与对话管理。
- **准备好开发了？** → 进入[开发者指南](/developer-guide/architecture)，了解内部机制并开始贡献。
- **想要实际示例？** → 查看[指南](/guides/tips)部分，获取真实项目案例和技巧。

:::tip
您无需阅读所有内容。选择与您目标匹配的路径，按顺序跟随链接，即可快速上手。随时可以回到本页寻找下一步。
:::