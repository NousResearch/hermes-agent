---
title: "Plan — Plan 模式：将 Markdown 计划写入"
sidebar_label: "Plan"
description: "Plan 模式：将 Markdown 计划写入"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Plan

Plan 模式：将 Markdown 计划写入 .hermes/plans/，不执行任何操作。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/software-development/plan` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `planning`, `plan-mode`, `implementation`, `workflow` |
| 相关 skill | [`writing-plans`](/user-guide/skills/bundled/software-development/software-development-writing-plans), [`subagent-driven-development`](/user-guide/skills/bundled/software-development/software-development-subagent-driven-development) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# Plan 模式

当用户需要计划而非执行时，使用此 skill。

## 核心行为

在本轮中，你仅进行规划。

- 不实现代码。
- 不编辑项目文件，计划 Markdown 文件除外。
- 不运行有副作用的终端命令，不提交、不推送，不执行外部操作。
- 必要时可使用只读命令/工具检查仓库或其他上下文。
- 你的交付物是保存在活跃工作区 `.hermes/plans/` 目录下的 Markdown 计划文件。

## 输出要求

编写一份具体且可操作的 Markdown 计划。

在相关时包含以下内容：
- 目标
- 当前上下文 / 假设
- 建议方案
- 分步计划
- 可能变更的文件
- 测试 / 验证
- 风险、权衡与待解问题

如果任务与代码相关，请包含精确的文件路径、可能的测试目标以及验证步骤。

## 保存位置

使用 `write_file` 将计划保存至：
- `.hermes/plans/YYYY-MM-DD_HHMMSS-<slug>.md`

将该路径视为相对于活跃工作目录 / 后端工作区的路径。Hermes 文件工具具备后端感知能力，使用此相对路径可确保计划文件在 local、docker、ssh、modal 和 daytona 后端上均与工作区保持一致。

如果运行时提供了具体的目标路径，则使用该精确路径。
如果没有，则自行在 `.hermes/plans/` 下创建一个合理的带时间戳的文件名。

## 交互风格

- 如果请求足够清晰，直接编写计划。
- 如果 `/plan` 没有附带明确指令，则从当前对话上下文中推断任务。
- 如果任务确实描述不足，提出简短的澄清问题，而非凭空猜测。
- 保存计划后，简要回复你所规划的内容及保存路径。