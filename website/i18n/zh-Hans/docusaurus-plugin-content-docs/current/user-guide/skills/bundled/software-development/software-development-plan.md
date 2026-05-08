---
title: "Plan — 计划模式：将 Markdown 计划写入"
sidebar_label: "Plan"
description: "计划模式：将 Markdown 计划写入"
---

{/* 本页面由 website/scripts/generate-skill-docs.py 从技能的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# Plan

计划模式：将 Markdown 计划写入 .hermes/plans/，不执行。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/software-development/plan` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 标签 | `planning`, `plan-mode`, `implementation`, `workflow` |
| 相关技能 | [`writing-plans`](/docs/user-guide/skills/bundled/software-development/software-development-writing-plans), [`subagent-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-subagent-driven-development) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能被触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# 计划模式

当用户想要计划而非执行时使用此技能。

## 核心行为

本轮你只做计划。

- 不实现代码。
- 不编辑项目文件（计划 Markdown 文件除外）。
- 不运行修改性终端命令、提交、推送或执行外部操作。
- 可以在需要时使用只读命令/工具检查仓库或其他上下文。
- 你的交付物是一个 Markdown 计划，保存在活动工作区下的 `.hermes/plans/` 中。

## 输出要求

编写一个具体且可操作的 Markdown 计划。

在相关时包括：
- 目标
- 当前上下文/假设
- 建议方案
- 分步计划
- 可能变更的文件
- 测试/验证
- 风险、权衡和待决问题

如果任务与代码相关，包括确切的文件路径、可能的测试目标和验证步骤。

## 保存位置

使用 `write_file` 将计划保存在：
- `.hermes/plans/YYYY-MM-DD_HHMMSS-<slug>.md`

将其视为相对于活动工作目录/后端工作区的路径。Hermes 文件工具是后端感知的，因此使用此相对路径会将计划与本地、docker、ssh、modal 和 daytona 后端上的工作区放在一起。

如果运行时提供了特定目标路径，使用该确切路径。
如果没有，在 `.hermes/plans/` 下创建一个合理的带时间戳的文件名。

## 交互方式

- 如果请求足够清晰，直接编写计划。
- 如果 `/plan` 没有附带明确指令，从当前对话上下文中推断任务。
- 如果确实不够明确，提出简短的澄清问题而非猜测。
- 保存计划后，简要回复你计划了什么以及保存的路径。
