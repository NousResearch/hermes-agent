---
title: "Code Wiki — 为任何代码库生成 wiki 文档 + Mermaid 图表"
sidebar_label: "Code Wiki"
description: "为任何代码库生成 wiki 文档 + Mermaid 图表"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Code Wiki

为任何代码库生成 wiki 文档 + Mermaid 图表。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/software-development/code-wiki` 安装 |
| 路径 | `optional-skills/software-development/code-wiki` |
| 版本 | `0.1.0` |
| 作者 | Teknium (teknium1)、Hermes Agent |
| 许可证 | MIT |
| 平台 | linux、macos、windows |
| 标签 | `Documentation`、`Mermaid`、`Architecture`、`Diagrams`、`Wiki`、`Code-Analysis` |
| 相关技能 | [`codebase-inspection`](/docs/user-guide/skills/bundled/github/github-codebase-inspection)、[`github-repo-management`](/docs/user-guide/skills/bundled/github/github-github-repo-management) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# Code Wiki 技能

为任何代码库生成全面的 wiki——概述、架构、每个模块的深入分析、Mermaid 类图和时序图。受 Google CodeWiki 启发，但适用于本地仓库、私有仓库和任何语言。仅使用现有 Hermes 工具（`terminal`、`read_file`、`search_files`、`write_file`）；无 Docker、无外部服务、无额外依赖。

## 何时使用

- 用户说"document this codebase"、"generate a wiki"、"make architecture diagrams"
- 接手不熟悉的仓库，想要结构化参考
- 用户指向 GitHub URL 要求文档化
- 需要可在 GitHub 上渲染的稳定制品（markdown + Mermaid）

不要用于单文件文档、单 API 端点参考、策略性叙事或用户正在积极开发的代码库。

## 快速参考

| 步骤 | 操作 |
|------|------|
| 1 | 解析目标 — 本地 cwd、给定路径或 `git clone --depth 50 <url>` 到临时目录 |
| 2 | 扫描结构 — `ls`、`find -maxdepth 3`、清单文件、README |
| 3 | 选择 8-10 个模块记录 |
| 4 | 编写 `README.md`（概述 + 模块图） |
| 5 | 编写 `architecture.md` 含 Mermaid 流程图 |
| 6 | 编写 `modules/` 中的模块文档 |
| 7 | 编写 `diagrams/class-diagram.md` |
| 8 | 编写 `diagrams/sequences.md` |
| 9 | 编写 `getting-started.md` |
| 10 | 编写 `api.md`（如适用） |
| 11 | 编写 `.codewiki-state.json` |
| 12 | 向用户报告路径 |

## 陷阱

- **捏造组件。** 每个图表节点和声称的函数调用必须在源代码中。
- **通用 AI 文本。** "This module is responsible for..." 是无内容的。
- **Mermaid > 50 个节点。** 它们无法清晰渲染。拆分它们。
- **默认是 `~/.hermes/wikis/`。** 仅在用户明确要求时写入仓库。
