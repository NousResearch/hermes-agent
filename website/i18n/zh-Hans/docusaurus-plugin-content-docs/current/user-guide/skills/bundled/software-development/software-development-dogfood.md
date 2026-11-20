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

本 skill 指导你使用 `browser_exec` 对网页应用进行系统化探索性 QA 测试。你将浏览应用、与元素交互、收集问题证据，并生成结构化缺陷报告。

## 前提条件

- 必须启用 `browser` 工具集（`browser_exec`）。`browser_exec` 运行 Python，并预先导入了辅助函数（`new_tab`、`goto_url`、`wait_for_load`、`page_info`、`js`、`fill_input`、`click_at_xy`、`capture_screenshot`、`cdp`）；浏览器状态在多次调用之间保留，Python 变量则不保留。
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

将每一步的操作合并到一次 `browser_exec` 调用中。针对计划中的每个页面或功能：

1. **安装控制台钩子后再导航。** 在导航前先在标签页上注册钩子，这样页面加载期间抛出的错误也能被捕获：
   ```python
   # Open page with console-error capture
   HOOK = """window.__qa_errors = [];
   const _e = console.error;
   console.error = (...a) => { window.__qa_errors.push(a.map(String).join(' ')); _e(...a); };
   addEventListener('error', e => window.__qa_errors.push('uncaught: ' + e.message));
   addEventListener('unhandledrejection', e => window.__qa_errors.push('rejection: ' + String(e.reason)));"""
   new_tab('about:blank')  # first call only: the hook binds to the current tab
   cdp('Page.addScriptToEvaluateOnNewDocument', source=HOOK)
   goto_url("https://example.com/page")
   wait_for_load()
   print(page_info())
   print(js('window.__qa_errors || []'))
   ```
   之后任何会导航的调用，都要在开头重新注册钩子。静默 JS 错误是高价值发现。

2. **读取页面结构** — 列出可见的交互元素及其视口中心坐标（需要角色/名称时，使用 `cdp('Accessibility.getFullAXTree')` 并在 Python 中过滤）：
   ```python
   # List interactive elements
   print(js("""(() => [...document.querySelectorAll('a,button,input,select,textarea,[role=button]')]
     .filter(e => e.offsetParent).slice(0, 80).map((e, i) => { const r = e.getBoundingClientRect();
       return {i, tag: e.tagName, text: (e.innerText || e.value || e.getAttribute('aria-label') || '').trim().slice(0, 60),
               id: e.id, name: e.name, x: Math.round(r.x + r.width / 2), y: Math.round(r.y + r.height / 2)}; }))()"""))
   ```

3. **直观评估页面**：调用 `capture_screenshot()`，查看附带的图片，检查布局问题、损坏的元素或无障碍问题。如果你的模型无法查看图片，请将打印出的路径传给 `vision_analyze`。

4. **系统化测试交互元素**：
   - 点击按钮和链接：`js("document.querySelector('#submit').click()")`，或使用第 2 步得到的中心坐标调用 `click_at_xy(x, y)`
   - 填写表单：`fill_input('input[name=email]', 'test input')`
   - 测试键盘导航：`cdp('Input.dispatchKeyEvent', type='keyDown', key='Tab', code='Tab', windowsVirtualKeyCode=9)`，随后发送对应的 `keyUp`，再用 `js('document.activeElement.outerHTML.slice(0, 200)')` 查看焦点落在何处；`key='Enter'`、`code='Enter'`、`windowsVirtualKeyCode=13` 同理
   - 滚动内容：`js('window.scrollBy(0, 800)')`
   - 后退：`js('history.back()')`，然后 `wait_for_load()`
   - 使用无效输入测试表单验证
   - 测试空提交

5. **每次交互后**，在同一次调用中检查：
   - 控制台错误：`print(js('window.__qa_errors || []'))`
   - 视觉变化：`capture_screenshot()` 并与上一张截图对比
   - 预期行为与实际行为（`page_info()`，或用 `js()` 读取相关元素）

### 阶段 3：收集证据

对于发现的每个问题：

1. **截图**以记录问题：`capture_screenshot()` 会打印保存路径。将其复制到 `{output_dir}/screenshots/` 并使用描述性文件名（例如 `issue-03-broken-login.png`）— 将在报告中引用它。

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

以上所有浏览器操作都是 `browser_exec` 调用；下表列出上文用到的辅助函数和惯用写法。

| 操作 | `browser_exec` 惯用写法 |
|------|---------|
| 跳转至指定 URL | 先调用一次 `new_tab(url)`（或在注册钩子前调用 `new_tab('about:blank')`），之后用 `goto_url(url)`；随后调用 `wait_for_load()` |
| 页面状态 | `page_info()` |
| DOM / 无障碍树 | 用 `js(...)` 提取，或 `cdp('Accessibility.getFullAXTree')` 并在 Python 中过滤 |
| 点击 | `js("document.querySelector(sel).click()")` 或 `click_at_xy(x, y)` |
| 在输入框中输入文字 | `fill_input(selector, text)` |
| 滚动 | `js('window.scrollBy(0, 800)')` |
| 后退 | `js('history.back()')` |
| 按下按键 | `cdp('Input.dispatchKeyEvent', type='keyDown', key=..., code=..., windowsVirtualKeyCode=...)` + `keyUp` |
| 截图 | `capture_screenshot()`（图片附在结果中；纯文本模型对该路径使用 `vision_analyze`） |
| JS 控制台错误 | 通过 `cdp('Page.addScriptToEvaluateOnNewDocument', source=HOOK)` 挂钩，用 `js('window.__qa_errors || []')` 读取 |

## 使用技巧

- **每次导航后及重要交互后，务必读取 `window.__qa_errors`。** 静默 JS 错误是最有价值的发现之一。
- **将每个子流程合并到一次 `browser_exec` 调用中**（导航、等待、提取、操作、检查错误）— 减少往返次数，且钩子保持有效。
- **需要推断元素位置或 DOM 提取结果不明确时，执行 `capture_screenshot()`**；`click_at_xy` 使用元素列表中的中心坐标。
- **使用有效和无效输入分别测试** — 表单验证缺陷十分常见。
- **滚动浏览长页面** — 折叠线以下的内容可能存在渲染问题。
- **测试导航流程** — 端到端点击多步骤流程。
- **通过截图中可见的布局问题检查响应式行为。**
- **不要忽视边界情况**：空状态、超长文本、特殊字符、快速连续点击。
- 向用户报告截图时，请包含 `MEDIA:<screenshot_path>`，以便他们能内联查看证据。