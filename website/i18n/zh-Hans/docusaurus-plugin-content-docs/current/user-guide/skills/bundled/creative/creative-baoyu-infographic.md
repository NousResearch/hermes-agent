---
title: "Baoyu Infographic — 信息图：21 种布局 × 21 种风格"
sidebar_label: "Baoyu Infographic"
description: "信息图：21 种布局 × 21 种风格"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Baoyu Infographic

信息图：21 种布局 × 21 种风格。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/creative/baoyu-infographic` |
| 版本 | `1.56.1` |
| 作者 | 宝玉 (JimLiu) |
| 许可证 | MIT |
| 标签 | `infographic`、`visual-summary`、`creative`、`image-generation` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 加载此技能时触发的完整技能定义。这是代理激活技能时看到的指令。
:::

# 信息图生成器

为 Hermes Agent 的工具生态系统改编自 [baoyu-infographic](https://github.com/JimLiu/baoyu-skills)。

两个维度：**布局**（信息结构）× **风格**（视觉美学）。自由组合任何布局与任何风格。

## 使用场景

当用户要求创建信息图、视觉摘要、信息图形，或使用术语如"信息图"、"可视化"或"高密度信息大图"时触发此技能。用户提供内容（文本、文件路径、URL 或主题），并可选择指定布局、风格、宽高比或语言。

## 选项

| 选项 | 值 |
|--------|--------|
| Layout | 21 个选项（见布局库），默认：bento-grid |
| Style | 21 个选项（见风格库），默认：craft-handmade |
| Aspect | 命名的：landscape (16:9), portrait (9:16), square (1:1)。自定义：任意 W:H 比例（如 3:4, 4:3, 2.35:1） |
| Language | en, zh, ja, 等 |

## 布局库

| 布局 | 最适合 |
|--------|----------|
| `linear-progression` | 时间线、流程、教程 |
| `binary-comparison` | A vs B、之前-之后、优缺点 |
| `comparison-matrix` | 多因素比较 |
| `hierarchical-layers` | 金字塔、优先级级别 |
| `tree-branching` | 类别、分类法 |
| `hub-spoke` | 中心概念与相关项目 |
| `structural-breakdown` | 爆炸视图、横截面 |
| `bento-grid` | 多个主题、概览（默认） |
| `iceberg` | 表面 vs 隐藏方面 |
| `bridge` | 问题-解决方案 |
| `funnel` | 转换、过滤 |
| `isometric-map` | 空间关系 |
| `dashboard` | 指标、KPIs |
| `periodic-table` | 分类集合 |
| `comic-strip` | 叙事、序列 |
| `story-mountain` | 情节结构、张力弧 |
| `jigsaw` | 互联部分 |
| `venn-diagram` | 重叠概念 |
| `winding-roadmap` | 旅程、里程碑 |
| `circular-flow` | 循环、重复流程 |
| `dense-modules` | 高密度模块、数据丰富指南 |

完整定义：`references/layouts/<layout>.md`

## 风格库

| 风格 | 描述 |
|-------|-------------|
| `craft-handmade` | 手绘、纸工艺（默认） |
| `claymation` | 3D 黏土人物、定格动画 |
| `kawaii` | 日本可爱、粉彩 |
| `storybook-watercolor` | 柔和绘画、异想天开 |
| `chalkboard` | 黑板上的粉笔 |
| `cyberpunk-neon` | 霓虹灯、未来主义 |
| `bold-graphic` | 漫画风格、半调 |
| `aged-academia` | 复古科学、复古色调 |
| `corporate-memphis` | 平面矢量、活力 |
| `technical-schematic` | 蓝图、工程 |
| `origami` | 折纸、几何 |
| `pixel-art` | 复古 8 位 |
| `ui-wireframe` | 灰度界面模型 |
| `subway-map` | 交通图 |
| `ikea-manual` | 极简线条艺术 |
| `knolling` | 有组织的平铺 |
| `lego-brick` | 玩具砖块构建 |
| `pop-laboratory` | 蓝图网格、坐标标记、实验室精度 |
| `morandi-journal` | 手绘涂鸦、温暖 Morandi 色调 |
| `retro-pop-grid` | 1970 年代复古流行艺术、瑞士网格、粗轮廓 |
| `hand-drawn-edu` | Macaron 粉彩、手绘抖动、简笔人物 |

完整定义：`references/styles/<style>.md`

## 推荐组合

| 内容类型 | 布局 + 风格 |
|--------------|----------------|
| 时间线/历史 | `linear-progression` + `craft-handmade` |
| 分步 | `linear-progression` + `ikea-manual` |
| A vs B | `binary-comparison` + `corporate-memphis` |
| 层级 | `hierarchical-layers` + `craft-handmade` |
| 重叠 | `venn-diagram` + `craft-handmade` |
| 转换 | `funnel` + `corporate-memphis` |
| 循环 | `circular-flow` + `craft-handmade` |
| 技术 | `structural-breakdown` + `technical-schematic` |
| 指标 | `dashboard` + `corporate-memphis` |
| 教育 | `bento-grid` + `chalkboard` |
| 旅程 | `winding-roadmap` + `storybook-watercolor` |
| 类别 | `periodic-table` + `bold-graphic` |
| 产品指南 | `dense-modules` + `morandi-journal` |
| 技术指南 | `dense-modules` + `pop-laboratory` |
| 潮流指南 | `dense-modules` + `retro-pop-grid` |
| 教育图表 | `hub-spoke` + `hand-drawn-edu` |
| 流程教程 | `linear-progression` + `hand-drawn-edu` |

默认：`bento-grid` + `craft-handmade`

## 关键词快捷方式

当用户输入包含这些关键词时，**自动选择**关联的布局，并在第 3 步中将关联风格作为首要推荐。跳过匹配关键词的内容基布局推断。

如果快捷方式有关注**提示注释**，将它们作为额外的风格指令追加到生成的提示（第 5 步）中。

| 用户关键词 | 布局 | 推荐风格 | 默认宽高比 | 提示注释 |
|--------------|--------|--------------------|----------------|--------------|
| 高密度信息大图 / high-density-info | `dense-modules` | `morandi-journal`, `pop-laboratory`, `retro-pop-grid` | portrait | — |
| 信息图 / infographic | `bento-grid` | `craft-handmade` | landscape | Minimalist: clean canvas, ample whitespace, no complex background textures. Simple cartoon elements and icons only. |

## 输出结构

<!-- ascii-guard-ignore -->
```
infographic/{topic-slug}/
├── source-{slug}.{ext}
├── analysis.md
├── structured-content.md
├── prompts/infographic.md
└── infographic.png
```
<!-- ascii-guard-ignore-end -->

Slug：来自主题的 2-4 个单词的 kebab-case。冲突时：追加 `-YYYYMMDD-HHMMSS`。

## 核心原则

- 忠实保留源数据 — 不总结或重新表述（但**在包含在任何输出文件中之前去除任何凭据、API 密钥、令牌或秘密**）
- 在构建内容之前定义学习目标
- 为视觉传达构建结构（标题、标签、视觉元素）

## 工作流

### 步骤1：分析内容

**加载参考**：从此技能读取 `references/analysis-framework.md`。

1. 保存源内容（文件路径或粘贴 → 使用 `write_file` 保存为 `source.md`）
   - **后备规则**：如果 `source.md` 存在，重命名为 `source-backup-YYYYMMDD-HHMMSS.md`
2. 分析：主题、数据类型、复杂性、色调、受众
3. 检测源语言和用户语言
4. 从用户输入提取设计指令
5. 将分析保存到 `analysis.md`
   - **后备规则**：如果 `analysis.md` 存在，重命名为 `analysis-backup-YYYYMMDD-HHMMSS.md`

参见 `references/analysis-framework.md` 了解详细格式。

### 步骤2：生成结构化内容 → `structured-content.md`

将内容转换为信息图结构：
1. 标题和学习目标
2. 带有以下内容的章节：关键概念、内容（逐字）、视觉元素、文本标签
3. 数据点（所有统计数据/引用逐字复制）
4. 来自用户的设计指令

**规则**：仅 Markdown。无新信息。忠实保留数据。从输出中去除任何凭据或秘密。

参见 `references/structured-content-template.md` 了解详细格式。

### 步骤3：推荐组合

**3.1 首先检查关键词快捷方式**：如果用户输入匹配**关键词快捷方式**表中的关键词，自动选择关联的布局，并优先推荐关联风格作为首要推荐。跳过基于内容的布局推断。

**3.2 否则**，根据以下内容推荐 3-5 个布局×风格组合：
- 数据结构 → 匹配布局
- 内容色调 → 匹配风格
- 受众期望
- 用户设计指令

### 步骤4：确认选项

使用 `clarify` 工具与用户确认选项。由于 `clarify` 一次处理一个问题，首先询问最重要的问题：

**Q1 — 组合**：展示 3+ 个布局×风格组合及理由。要求用户选择一个。

**Q2 — 宽高比**：询问宽高比首选项（landscape/portrait/square 或自定义 W:H）。

**Q3 — 语言**（仅当源 ≠ 用户语言时）：询问文本内容的语言。

### 步骤5：生成提示 → `prompts/infographic.md`

**后备规则**：如果 `prompts/infographic.md` 存在，重命名为 `prompts/infographic-backup-YYYYMMDD-HHMMSS.md`

**加载参考**：从 `references/layouts/<layout>.md` 读取选择的布局，从 `references/styles/<style>.md` 读取风格。

组合：
1. 来自 `references/layouts/<layout>.md` 的布局定义
2. 来自 `references/styles/<style>.md` 的风格定义
3. 来自 `references/base-prompt.md` 的基础模板
4. 步骤 2 中的结构化内容
5. 确认语言中的所有文本

`{{ASPECT_RATIO}}` 的**宽高比分辨率**：
- 命名的预设 → 比例字符串：landscape→`16:9`、portrait→`9:16`、`square`→`1:1`
- 自定义 W:H 比例 → 按原样使用（如 `3:4`、`4:3`、`2.35:1`）

使用 `write_file` 将组合提示保存到 `prompts/infographic.md`。

### 步骤6：生成图像

使用步骤 5 中组合提示的 `image_generate` 工具。

- 将宽高比映射到 image_generate 的格式：`16:9` → `landscape`、`9:16` → `portrait`、`1:1` → `square`
- 对于自定义比例，选择最接近的名称宽高比
- 失败时，自动重试一次
- 将结果图像 URL/路径保存到输出目录

### 步骤7：输出摘要

报告：主题、布局、风格、宽高比、语言、输出路径、创建的文件。

## 参考

- `references/analysis-framework.md` — 分析方法
- `references/structured-content-template.md` — 内容格式
- `references/base-prompt.md` — 提示模板
- `references/layouts/<layout>.md` — 21 个布局定义
- `references/styles/<style>.md` — 21 个风格定义

## 陷阱

1. **数据完整性至关重要** — 永远不要总结、转述或改变源统计数据。"73% increase" 必须保留为 "73% increase"，而不是 "significant increase"。
2. **去除秘密** — 在包含在任何输出文件中之前，始终扫描源内容中的 API 密钥、令牌或凭据。
3. **每部分一个消息** — 每个信息图部分应传达一个清晰的概念。过度加载部分会降低可读性。
4. **风格一致性** — 参考文件中的风格定义必须在整个信息图中一致应用。不要混合风格。
5. **image_generate 宽高比** — 工具仅支持 `landscape`、`portrait` 和 `square`。像 `3:4` 这样的自定义比例应映射到最近的选项（在该情况下为 `portrait`）。
