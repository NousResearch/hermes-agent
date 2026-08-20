---
title: "Subagent Driven Development — 通过 delegate_task 子代理执行计划（双阶段审核）"
sidebar_label: "Subagent Driven Development"
description: "通过 delegate_task 子代理执行计划（双阶段审核）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Subagent Driven Development

通过 delegate_task 子代理执行计划（双阶段审核）。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/software-development/subagent-driven-development` 安装 |
| 路径 | `optional-skills/software-development/subagent-driven-development` |
| 版本 | `1.1.0` |
| 作者 | Hermes Agent（改编自 obra/superpowers） |
| 许可证 | MIT |
| 平台 | linux、macos、windows |
| 标签 | `delegation`、`subagent`、`implementation`、`workflow`、`parallel` |
| 相关技能 | [`plan`](/docs/user-guide/skills/bundled/software-development/software-development-plan)、[`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review)、[`test-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# 子代理驱动的开发

## 概述

通过为每个任务分派全新的子代理并进行系统性双阶段审核来执行实现计划。

**核心原则：** 每个任务一个全新子代理 + 双阶段审核（规格然后质量）= 高质量，快速迭代。

## 何时使用

当你有一个实现计划（来自 `plan` 技能或用户需求）；任务基本独立；质量和规格合规性很重要；你想要任务间的自动审核时使用。

## 流程

### 1. 读取并解析计划

读取计划文件。预先提取所有任务及其完整文本和上下文。创建 todo 列表。

### 2. 每个任务的工作流

对于计划中的**每个**任务：

#### 步骤 1：分派实现子代理

使用 `delegate_task` 并提供完整上下文。包括任务规格、TDD 指令和项目上下文。

#### 步骤 2：分派规格合规审核者

实现完成后，验证是否符合原始规格。如果发现规格问题：修复差距，然后重新运行规格审核。

#### 步骤 3：分派代码质量审核者

规格合规通过后，审核代码质量。如果发现质量问题：修复问题，重新审核。

#### 步骤 4：标记完成

### 3. 最终审核

所有任务完成后，分派最终集成审核者。

### 4. 验证并提交

```bash
pytest tests/ -q
git diff --stat
git add -A && git commit -m "feat: complete [feature name] implementation"
```

## 任务粒度

**每个任务 = 2-5 分钟的专注工作。**

**太大：** "Implement user authentication system"

**合适大小：** "Create User model with email and password fields"、"Add password hashing function"、"Create login endpoint"

## 红旗 — 绝不做的事

- 没有计划就开始实现
- 跳过审核
- 在未修复的关键/重要问题上继续
- 为涉及相同文件的任务分派多个实现子代理
- 让子代理读取计划文件（直接在上下文中提供完整文本）
- 跳过场景设置上下文
- 忽略子代理的问题
- 接受规格合规"差不多就行"
- 让实现者自审替代实际审核

## 效率说明

**为什么每个任务一个新子代理：** 防止累积状态的上下文污染。

**为什么双阶段审核：** 规格审核尽早发现不足/过度构建，质量审核确保实现良好构建。

## 与其他技能集成

### 与 plan
此技能执行 `plan` 技能创建的计划。

### 与 test-driven-development
实现子代理应遵循 TDD。

### 与其他技能
结合 `requesting-code-review` 进行最终集成审核；遇到 bug 时遵循 `systematic-debugging`。

## 记住

```
每个任务一个新子代理
每次都双阶段审核
规格合规优先
代码质量其次
绝不跳过审核
尽早发现问题
```

**质量不是偶然的。它是系统化流程的结果。**
