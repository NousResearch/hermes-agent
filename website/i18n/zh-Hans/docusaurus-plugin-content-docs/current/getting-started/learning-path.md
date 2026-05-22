---
sidebar_position: 3
title: "学习路径"
description: "根据你的经验水平和目标，选择合适的 Hermes Agent 文档学习路径。"
---

# 学习路径

Hermes Agent 能做很多事 - CLI 助手、Telegram/Discord 机器人、任务自动化、RL 训练等等。这一页会帮你根据经验水平和实际目标，决定从哪里开始、该读什么。

:::tip 从这里开始
如果你还没安装 Hermes Agent，先从 [安装指南](/getting-started/installation) 开始，再走一遍 [快速上手](/getting-started/quickstart)。下面所有内容都默认你已经有了可用安装。
:::

## 如何使用本页

- **已经知道自己的水平？** 直接跳到 [按经验水平](#by-experience-level) 表，按对应顺序阅读。
- **有具体目标？** 直接看 [按使用场景](#by-use-case)，找到与你情况匹配的部分。
- **只是随便看看？** 看 [一览关键功能](#key-features-at-a-glance) 表，快速了解 Hermes Agent 能做什么。

## 按经验水平 {#by-experience-level}

| 水平 | 目标 | 推荐阅读顺序 | 预计耗时 |
|---|---|---|---|
| **初学者** | 跑起来、进行基础对话、使用内置工具 | [安装](/getting-started/installation) → [快速上手](/getting-started/quickstart) → [CLI 用法](/user-guide/cli) → [配置](/user-guide/configuration) | ~1 小时 |
| **中级** | 配置消息机器人、使用记忆、cron 作业和技能等进阶功能 | [会话](/user-guide/sessions) → [消息平台](/user-guide/messaging) → [工具](/user-guide/features/tools) → [技能](/user-guide/features/skills) → [记忆](/user-guide/features/memory) → [Cron](/user-guide/features/cron) | ~2–3 小时 |
| **高级** | 构建自定义工具、创建技能、生成训练数据、为项目做贡献 | [架构](/developer-guide/architecture) → [添加工具](/developer-guide/adding-tools) → [创建技能](/developer-guide/creating-skills) → [批处理](/user-guide/features/batch-processing) → [贡献指南](/developer-guide/contributing) | ~4–6 小时 |

## 按使用场景 {#by-use-case}

选择你想做的事情对应的场景。每个场景都按推荐阅读顺序列出相关文档。

### “我想要一个 CLI 编码助手”

把 Hermes Agent 当成一个交互式终端助手，用于编写、审查和运行代码。

1. [安装](/getting-started/installation)
2. [快速上手](/getting-started/quickstart)
3. [CLI 用法](/user-guide/cli)
4. [代码执行](/user-guide/features/code-execution)
5. [上下文文件](/user-guide/features/context-files)
6. [技巧与窍门](/guides/tips)

:::tip
通过上下文文件可以直接把文件送进对话。Hermes Agent 可以读取、编辑和运行你项目里的代码。
:::

### “我想要一个 Telegram/Discord 机器人”

把 Hermes Agent 部署成你常用消息平台上的机器人。

1. [安装](/getting-started/installation)
2. [配置](/user-guide/configuration)
3. [消息平台总览](/user-guide/messaging)
4. [Telegram 配置](/user-guide/messaging/telegram)
5. [Discord 配置](/user-guide/messaging/discord)
6. [语音模式](/user-guide/features/voice-mode)
7. [在 Hermes 中使用语音模式](/guides/use-voice-mode-with-hermes)
8. [安全性](/user-guide/security)

完整项目示例请看：
- [每日简报机器人](/guides/daily-briefing-bot)
- [团队 Telegram 助手](/guides/team-telegram-assistant)

### “我想自动化任务”

安排重复任务、运行批处理作业，或把多个智能体动作串起来。

1. [快速上手](/getting-started/quickstart)
2. [Cron 调度](/user-guide/features/cron)
3. [批量处理](/user-guide/features/batch-processing)
4. [委派](/user-guide/features/delegation)
5. [钩子](/user-guide/features/hooks)

:::tip
Cron 作业可以让 Hermes Agent 按计划运行任务 - 每日摘要、定期检查、自动报告 - 你不在场也可以执行。
:::

### “我想构建自定义工具 / 技能”

用你自己的工具和可复用技能包扩展 Hermes Agent。

1. [插件](/user-guide/features/plugins)
2. [构建 Hermes 插件](/guides/build-a-hermes-plugin)
3. [工具总览](/user-guide/features/tools)
4. [技能总览](/user-guide/features/skills)
5. [MCP（Model Context Protocol）](/user-guide/features/mcp)
6. [架构](/developer-guide/architecture)
7. [添加工具](/developer-guide/adding-tools)
8. [创建技能](/developer-guide/creating-skills)

:::tip
大多数自定义工具的创建应从插件开始。[添加工具](/developer-guide/adding-tools)页面是给 Hermes core 的内置开发用的，不是普通用户 / 自定义工具的常规路径。
:::

### “我想训练模型”

使用 Hermes Agent 的批处理功能生成训练和评估用的轨迹数据。

1. [快速上手](/getting-started/quickstart)
2. [配置](/user-guide/configuration)
3. [批处理](/user-guide/features/batch-processing)
4. [提供商路由](/user-guide/features/provider-routing)
5. [架构](/developer-guide/architecture)

:::tip
批处理在你已经理解 Hermes Agent 如何处理对话和工具调用的基础上效果最好。如果你是新手，先走一遍初学者路径。
:::

### “我想把它当作 Python 库使用”

把 Hermes Agent 以编程方式集成到你自己的 Python 应用里。

1. [安装](/getting-started/installation)
2. [快速上手](/getting-started/quickstart)
3. [Python 库指南](/guides/python-library)
4. [架构](/developer-guide/architecture)
5. [工具](/user-guide/features/tools)
6. [会话](/user-guide/sessions)

## 一览关键功能 {#key-features-at-a-glance}

不确定有哪些能力？下面是主要功能的快速目录：

| 功能 | 它做什么 | 链接 |
|---|---|---|
| **工具** | 智能体可调用的内置工具（文件 I/O、搜索、shell 等） | [工具](/user-guide/features/tools) |
| **技能** | 可安装的插件包，用来添加新能力 | [技能](/user-guide/features/skills) |
| **记忆** | 跨会话持久化的记忆 | [记忆](/user-guide/features/memory) |
| **上下文文件** | 把文件和目录喂给对话 | [上下文文件](/user-guide/features/context-files) |
| **MCP** | 通过 Model Context Protocol 连接外部工具服务器 | [MCP](/user-guide/features/mcp) |
| **Cron** | 为智能体任务安排周期性调度 | [Cron](/user-guide/features/cron) |
| **委派** | 为并行工作创建子智能体 | [委派](/user-guide/features/delegation) |
| **代码执行** | 运行可以程序化调用 Hermes 工具的 Python 脚本 | [代码执行](/user-guide/features/code-execution) |
| **浏览器** | 网页浏览和抓取 | [浏览器](/user-guide/features/browser) |
| **钩子** | 事件驱动的回调和中间件 | [钩子](/user-guide/features/hooks) |
| **批量处理** | 批量处理多个输入 | [批量处理](/user-guide/features/batch-processing) |
| **批处理** | 生成训练和评估用的轨迹数据 | [批处理](/user-guide/features/batch-processing) |
| **提供商路由** | 在多个 LLM 提供商之间路由请求 | [提供商路由](/user-guide/features/provider-routing) |

## 下一步该读什么

根据你现在所处的位置：

- **刚安装完？** → 去看 [快速上手](/getting-started/quickstart)，完成第一次对话。
- **已经完成快速上手？** → 阅读 [CLI 用法](/user-guide/cli) 和 [配置](/user-guide/configuration)，自定义你的设置。
- **已经掌握基础？** → 探索 [工具](/user-guide/features/tools)、[技能](/user-guide/features/skills) 和 [记忆](/user-guide/features/memory)，解锁智能体的完整能力。
- **要给团队部署？** → 阅读 [安全性](/user-guide/security) 和 [会话](/user-guide/sessions)，了解访问控制和会话管理。
- **准备自己动手构建？** → 进入 [开发者指南](/developer-guide/architecture)，了解内部实现并开始贡献。
- **想看实战示例？** → 去 [指南](/guides/tips) 版块看真实项目和技巧。

:::tip
你不需要把所有内容都读完。选一个与你目标匹配的路径，按顺序点开链接，很快就能开始干活。以后随时可以回到这一页找下一步。
:::
