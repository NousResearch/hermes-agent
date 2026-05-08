---
title: "Dogfood — Web 应用探索性 QA：查找 bug、证据、报告"
sidebar_label: "Dogfood"
description: "Web 应用探索性 QA：查找 bug、证据、报告"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Dogfood

Web 应用探索性 QA：查找 bug、证据、报告。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/dogfood` |
| 版本 | `1.0.0` |
| 标签 | `qa`, `testing`, `browser`, `web`, `dogfood` |

## 参考：完整 SKILL.md

:::info
以下是 Hermites 加载此技能时的完整技能定义。这是技能激活时 agent 看到的指令。
:::

# Dogfood：系统化 Web 应用 QA 测试

## 概述

此技能指导您使用浏览器工具集对 Web 应用进行系统化探索性 QA 测试。您将导航应用、与元素交互、捕获问题证据，并生成结构化的 bug 报告。

## 先决条件

- 浏览器工具集必须可用（`browser_navigate`、`browser_snapshot`、`browser_click`、`browser_type`、`browser_vision`、`browser_console`、`browser_scroll`、`browser_back`、`browser_press`）
- 用户提供目标 URL 和测试范围

## 输入

用户提供：
1. **目标 URL** — 测试入口点
2. **范围** — 要关注哪些领域/功能（或"全站"进行全面测试）
3. **输出目录**（可选）— 保存截图和报告的位置（默认：`./dogfood-output`）

## 工作流程

按照这个 5 阶段系统化工作流程：

### 阶段 1：计划

1. 创建输出目录结构：
<!-- ascii-guard-ignore -->
   ```
   {output_dir}/
   ├── screenshots/       # 证据截图
   └── report.md          # 最终报告（在阶段 5 生成）
   ```
<!-- ascii-guard-ignore-end -->
2. 根据用户输入确定测试范围。
3. 通过规划要测试的页面和功能来构建粗略的站点地图：
   - 落地页/主页
   - 导航链接（页眉、页脚、侧边栏）
   - 关键用户流程（注册、登录、搜索、结账等）
   - 表单和交互元素
   - 边缘情况（空状态、错误页面、404）

### 阶段 2：探索

对于计划中的每个页面或功能：

1. **导航**到页面：
   ```
   browser_navigate(url="https://example.com/page")
   ```

2. **拍摄快照**以了解 DOM 结构：
   ```
   browser_snapshot()
   ```

3. **检查控制台**是否有 JavaScript 错误：
   ```
   browser_console(clear=true)
   ```
   在每次导航后和每次重要交互后执行此操作。静默 JS 错误是高价值发现。

4. **拍摄带注释的截图**以视觉评估页面并识别交互元素：
   ```
   browser_vision(question="描述页面布局，识别任何视觉问题、破损元素或可访问性关注点", annotate=true)
   ```
   `annotate=true` 标志在交互元素上叠加编号 `[N]` 标签。每个 `[N]` 映射到后续浏览器命令的 ref `@eN`。

5. **系统化测试交互元素**：
   - 点击按钮和链接：`browser_click(ref="@eN")`
   - 填写表单：`browser_type(ref="@eN", text="test input")`
   - 测试键盘导航：`browser_press(key="Tab")`、`browser_press(key="Enter")`
   - 滚动浏览内容：`browser_scroll(direction="down")`
   - 用无效输入测试表单验证
   - 测试空提交

6. **每次交互后**，检查：
   - 控制台错误：`browser_console()`
   - 视觉变化：`browser_vision(question="交互后发生了什么变化？")`
   - 预期与实际行为

### 阶段 3：收集证据

对于发现的每个问题：

1. **拍摄显示问题的截图**：
   ```
   browser_vision(question="捕获并描述此页面上可见的问题", annotate=false)
   ```
   保存响应的 `screenshot_path` — 您将在报告中引用它。

2. **记录详情**：
   - 问题出现的 URL
   - 重现步骤
   - 预期行为
   - 实际行为
   - 控制台错误（如果有）
   - 截图路径

3. **对问题进行分类**使用问题分类法（参见 `references/issue-taxonomy.md`）：
   - 严重性：Critical / High / Medium / Low
   - 类别：Functional / Visual / Accessibility / Console / UX / Content

### 阶段 4：分类

1. 审查所有收集的问题。
2. 去重 — 合并在不同地方表现相同 bug 的问题。
3. 为每个问题分配最终严重性和类别。
4. 按严重性排序（Critical 优先，然后 High、Medium、Low）。
5. 统计执行摘要中按严重性和类别的数量。

### 阶段 5：报告

使用 `templates/dogfood-report-template.md` 中的模板生成最终报告。

报告必须包括：
1. **执行摘要**，包括问题总数、按严重性细分和测试范围
2. **每个问题的部分**：
   - 问题编号和标题
   - 严重性和类别标记
   - 观察到的 URL
   - 问题描述
   - 重现步骤
   - 预期与实际行为
   - 截图引用（使用 `MEDIA:<screenshot_path>` 以便内联图片）
   - 相关控制台错误
3. **所有问题的汇总表**
4. **测试说明** — 测试了什么，什么没有测试，任何阻碍因素

将报告保存到 `{output_dir}/report.md`。

## 工具参考

| 工具 | 用途 |
|------|---------|
| `browser_navigate` | 转到 URL |
| `browser_snapshot` | 获取 DOM 文本快照（可访问性树） |
| `browser_click` | 通过 ref (`@eN`) 或文本点击元素 |
| `browser_type` | 在输入字段中键入 |
| `browser_scroll` | 在页面上向上/向下滚动 |
| `browser_back` | 在浏览器历史中返回 |
| `browser_press` | 按键盘键 |
| `browser_vision` | 截图 + AI 分析；使用 `annotate=true` 获取元素标签 |
| `browser_console` | 获取 JS 控制台输出和错误 |

## 提示

- **在导航后和重要交互后始终检查 `browser_console()`。** 静默 JS 错误是最有价值的发现之一。
- **在需要推理交互元素位置或快照 ref 不清楚时使用 `annotate=true` 配合 `browser_vision`**。
- **用有效和无效输入测试** — 表单验证 bug 很常见。
- **滚动浏览长页面** — 折叠内容下方可能有渲染问题。
- **测试导航流程** — 端到端点击多步骤流程。
- **通过注意截图中可见的布局问题来检查响应行为**。
- **不要忘记边缘情况**：空状态、超长文本、特殊字符、快速点击。
- 向用户报告截图时，包含 `MEDIA:<screenshot_path>` 以便他们可以内联查看证据。
