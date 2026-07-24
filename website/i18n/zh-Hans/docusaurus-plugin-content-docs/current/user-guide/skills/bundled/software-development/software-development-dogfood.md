---
title: "Dogfood — 网页应用探索性 QA：发现缺陷、收集证据、生成报告"
sidebar_label: "Dogfood"
description: "网页应用探索性 QA：发现缺陷、收集证据、生成报告"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Dogfood

网页应用探索性 QA：发现缺陷、收集证据、生成报告。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/software-development/dogfood` |
| 版本 | `1.0.0` |
| 平台 | linux, macos, windows |
| 标签 | `qa`, `testing`, `browser`, `web`, `dogfood` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Dogfood：系统化网页应用 QA 测试

## 概述

本 skill 指导你使用浏览器工具集对网页应用进行系统化探索性 QA 测试。你将浏览应用、与元素交互、收集问题证据，并生成结构化缺陷报告。

## 前提条件

- 浏览器工具集必须可用（`browser_navigate`、`browser_snapshot`、`browser_click`、`browser_type`、`browser_vision`、`browser_console`、`browser_scroll`、`browser_back`、`browser_press`）
- 用户提供目标 URL 和测试范围

## 输入

用户提供：
1. **目标 URL** — 测试入口点
2. **范围** — 需要重点测试的区域/功能（或填写"全站"进行全面测试）
3. **输出目录**（可选）— 截图和报告的保存位置（默认：`./dogfood-output`）

## 工作流程

遵循以下 5 阶段系统化工作流程：

### 阶段 1：规划

1. 创建输出目录结构：
<!-- ascii-guard-ignore -->
   ```
   {output_dir}/
   ├── screenshots/       # 证据截图
   └── report.md          # 最终报告（在阶段 5 生成）
   ```
<!-- ascii-guard-ignore-end -->
2. 根据用户输入确定测试范围。
3. 通过规划待测页面和功能，构建粗略站点地图：
   - 落地页/首页
   - 导航链接（页头、页脚、侧边栏）
   - 关键用户流程（注册、登录、搜索、结账等）
   - 表单和交互元素
   - 边界情况（空状态、错误页面、404 等）

### 阶段 2：探索

针对计划中的每个页面或功能：

1. **导航**至该页面：
   ```
   browser_navigate(url="https://example.com/page")
   ```

2. **获取快照**以了解 DOM 结构：
   ```
   browser_snapshot()
   ```

3. **检查控制台**中的 JavaScript 错误：
   ```
   browser_console(clear=true)
   ```
   每次导航后及每次重要交互后都应执行此操作。静默 JS 错误是高价值发现。

4. **获取带标注的截图**，以直观评估页面并识别交互元素：
   ```
   browser_vision(question="Describe the page layout, identify any visual issues, broken elements, or accessibility concerns", annotate=true)
   ```
   `annotate=true` 标志会在交互元素上叠加编号标签 `[N]`。每个 `[N]` 对应后续浏览器命令中的引用 `@eN`。

5. **系统化测试交互元素**：
   - 点击按钮和链接：`browser_click(ref="@eN")`
   - 填写表单：`browser_type(ref="@eN", text="test input")`
   - 测试键盘导航：`browser_press(key="Tab")`、`browser_press(key="Enter")`
   - 滚动内容：`browser_scroll(direction="down")`
   - 使用无效输入测试表单验证
   - 测试空提交

6. **每次交互后**，检查：
   - 控制台错误：`browser_console()`
   - 视觉变化：`browser_vision(question="What changed after the interaction?")`
   - 预期行为与实际行为

### 阶段 3：收集证据

对于发现的每个问题：

1. **截图**以记录问题：
   ```
   browser_vision(question="Capture and describe the issue visible on this page", annotate=false)
   ```
   保存响应中的 `screenshot_path` — 将在报告中引用它。

2. **记录详情**：
   - 问题发生的 URL
   - 复现步骤
   - 预期行为
   - 实际行为
   - 控制台错误（如有）
   - 截图路径

3. **按问题分类法对问题分类**（参见 `references/issue-taxonomy.md`）：
   - 严重程度：Critical（严重）/ High（高）/ Medium（中）/ Low（低）
   - 类别：Functional（功能）/ Visual（视觉）/ Accessibility（无障碍）/ Console（控制台）/ UX（用户体验）/ Content（内容）

### 阶段 4：分类整理

1. 审查所有收集到的问题。
2. 去重 — 合并在不同位置表现为同一缺陷的问题。
3. 为每个问题分配最终严重程度和类别。
4. 按严重程度排序（Critical 优先，依次为 High、Medium、Low）。
5. 按严重程度和类别统计问题数量，用于执行摘要。

### 阶段 5：报告

使用 `templates/dogfood-report-template.md` 中的模板生成最终报告。

报告必须包含：
1. **执行摘要**，含问题总数、按严重程度的分布情况及测试范围
2. **每个问题的章节**，包含：
   - 问题编号和标题
   - 严重程度和类别标签
   - 观察到问题的 URL
   - 问题描述
   - 复现步骤
   - 预期行为与实际行为
   - 截图引用（使用 `MEDIA:<screenshot_path>` 内联显示图片）
   - 相关控制台错误（如有）
3. **所有问题的汇总表**
4. **测试说明** — 已测试内容、未测试内容及任何阻塞项

将报告保存至 `{output_dir}/report.md`。

## 工具参考

| 工具 | 用途 |
|------|---------|
| `browser_navigate` | 跳转至指定 URL |
| `browser_snapshot` | 获取 DOM 文本快照（无障碍树） |
| `browser_click` | 通过引用（`@eN`）或文本点击元素 |
| `browser_type` | 在输入框中输入文字 |
| `browser_scroll` | 在页面上向上/向下滚动 |
| `browser_back` | 在浏览器历史中后退 |
| `browser_press` | 按下键盘按键 |
| `browser_vision` | 截图 + AI 分析；使用 `annotate=true` 显示元素标签 |
| `browser_console` | 获取 JS 控制台输出和错误 |

## 使用技巧

- **每次导航后及重要交互后，务必执行 `browser_console()`。** 静默 JS 错误是最有价值的发现之一。
- **在需要推断交互元素位置或快照引用不清晰时，对 `browser_vision` 使用 `annotate=true`。**
- **使用有效和无效输入分别测试** — 表单验证缺陷十分常见。
- **滚动浏览长页面** — 折叠线以下的内容可能存在渲染问题。
- **测试导航流程** — 端到端点击多步骤流程。
- **通过截图中可见的布局问题检查响应式行为。**
- **不要忽视边界情况**：空状态、超长文本、特殊字符、快速连续点击。
- 向用户报告截图时，请包含 `MEDIA:<screenshot_path>`，以便他们能内联查看证据。