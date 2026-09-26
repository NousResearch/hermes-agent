---
title: "Baoyu Infographic — 信息图：21种布局 × 21种风格（信息图, 可视化）"
sidebar_label: "Baoyu Infographic"
description: "信息图：21种布局 × 21种风格（信息图, 可视化）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Baoyu Infographic

信息图：21种布局 × 21种风格（信息图, 可视化）。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/creative/baoyu-infographic` |
| 版本 | `1.56.1` |
| 作者 | 宝玉 (JimLiu) |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `infographic`, `visual-summary`, `creative`, `image-generation` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# 信息图生成器

改编自 [baoyu-infographic](https://github.com/JimLiu/baoyu-skills)，适配 Hermes Agent 的工具生态系统。

两个维度：**布局**（信息结构）× **风格**（视觉美学）。可自由组合任意布局与风格。

## 使用时机

当用户要求创建信息图、视觉摘要、information graphic，或使用"信息图"、"可视化"、"高密度信息大图"等词语时，触发此 skill。用户提供内容（文本、文件路径、URL 或主题），并可选择指定布局、风格、宽高比或语言。

## 选项

| 选项 | 可选值 |
|--------|--------|
| 布局 | 21个选项（见布局图库），默认：bento-grid |
| 风格 | 21个选项（见风格图库），默认：craft-handmade |
| 宽高比 | 命名预设：landscape（16:9）、portrait（9:16）、square（1:1）。自定义：任意 W:H 比例（如 3:4、4:3、2.35:1） |
| 语言 | en、zh、ja 等 |

## 布局图库

| 布局 | 最适合 |
|--------|----------|
| `linear-progression` | 时间线、流程、教程 |
| `binary-comparison` | A vs B、前后对比、优缺点 |
| `comparison-matrix` | 多因素比较 |
| `hierarchical-layers` | 金字塔、优先级层级 |
| `tree-branching` | 分类、分类体系 |
| `hub-spoke` | 以中心概念辐射相关项 |
| `structural-breakdown` | 爆炸图、截面图 |
| `bento-grid` | 多主题、概览（默认） |
| `iceberg` | 表面与隐藏层面 |
| `bridge` | 问题-解决方案 |
| `funnel` | 转化、筛选 |
| `isometric-map` | 空间关系 |
| `dashboard` | 指标、KPI |
| `periodic-table` | 分类集合 |
| `comic-strip` | 叙事、序列 |
| `story-mountain` | 情节结构、张力弧线 |
| `jigsaw` | 相互关联的部分 |
| `venn-diagram` | 重叠概念 |
| `winding-roadmap` | 旅程、里程碑 |
| `circular-flow` | 循环、周期性流程 |
| `dense-modules` | 高密度模块、数据丰富的指南 |

完整定义：`references/layouts/<layout>.md`

## 风格图库

| 风格 | 描述 |
|-------|-------------|
| `craft-handmade` | 手绘、纸艺（默认） |
| `claymation` | 3D 黏土人物、定格动画 |
| `kawaii` | 日系可爱风、马卡龙色 |
| `storybook-watercolor` | 柔和水彩、奇幻风格 |
| `chalkboard` | 黑板粉笔风 |
| `cyberpunk-neon` | 霓虹发光、未来主义 |
| `bold-graphic` | 漫画风格、半调网点 |
| `aged-academia` | 复古科学、棕褐色调 |
| `corporate-memphis` | 扁平矢量、鲜艳色彩 |
| `technical-schematic` | 蓝图、工程制图 |
| `origami` | 折纸、几何造型 |
| `pixel-art` | 复古 8-bit 像素风 |
| `ui-wireframe` | 灰度界面线框图 |
| `subway-map` | 地铁线路图风格 |
| `ikea-manual` | 极简线条插图 |
| `knolling` | 整齐平铺俯拍 |
| `lego-brick` | 玩具积木构造 |
| `pop-laboratory` | 蓝图网格、坐标标注、实验室精度 |
| `morandi-journal` | 手绘涂鸦、莫兰迪暖色调 |
| `retro-pop-grid` | 1970年代复古波普艺术、瑞士网格、粗轮廓线 |
| `hand-drawn-edu` | 马卡龙色、手绘抖动线条、简笔人物 |

完整定义：`references/styles/<style>.md`

## 推荐组合

| 内容类型 | 布局 + 风格 |
|--------------|----------------|
| 时间线/历史 | `linear-progression` + `craft-handmade` |
| 分步说明 | `linear-progression` + `ikea-manual` |
| A vs B | `binary-comparison` + `corporate-memphis` |
| 层级结构 | `hierarchical-layers` + `craft-handmade` |
| 重叠关系 | `venn-diagram` + `craft-handmade` |
| 转化漏斗 | `funnel` + `corporate-memphis` |
| 循环流程 | `circular-flow` + `craft-handmade` |
| 技术内容 | `structural-breakdown` + `technical-schematic` |
| 指标数据 | `dashboard` + `corporate-memphis` |
| 教育内容 | `bento-grid` + `chalkboard` |
| 旅程路线 | `winding-roadmap` + `storybook-watercolor` |
| 分类集合 | `periodic-table` + `bold-graphic` |
| 产品指南 | `dense-modules` + `morandi-journal` |
| 技术指南 | `dense-modules` + `pop-laboratory` |
| 潮流指南 | `dense-modules` + `retro-pop-grid` |
| 教育图解 | `hub-spoke` + `hand-drawn-edu` |
| 流程教程 | `linear-progression` + `hand-drawn-edu` |

默认：`bento-grid` + `craft-handmade`

## 关键词快捷方式

当用户输入包含以下关键词时，**自动选择**对应布局，并在第3步将关联风格作为首选推荐。匹配到关键词后，跳过基于内容的布局推断。

若某快捷方式包含 **Prompt Notes**，则在生成 prompt（第5步）时将其作为额外风格指令追加。

| 用户关键词 | 布局 | 推荐风格 | 默认宽高比 | Prompt Notes |
|--------------|--------|--------------------|----------------|--------------|
| 高密度信息大图 / high-density-info | `dense-modules` | `morandi-journal`, `pop-laboratory`, `retro-pop-grid` | portrait | — |
| 信息图 / infographic | `bento-grid` | `craft-handmade` | landscape | 极简风格：干净画布、充足留白、无复杂背景纹理。仅使用简单卡通元素和图标。 |

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

Slug：从主题中取 2-4 个单词，使用 kebab-case。冲突时追加 `-YYYYMMDD-HHMMSS`。

## 核心原则

- 忠实保留源数据——不做摘要或改写（但在写入输出文件前，**必须去除所有凭据、API 密钥、token 或密钥**）
- 在构建内容结构前先明确学习目标
- 面向视觉传达进行结构化（标题、标签、视觉元素）

## 工作流程

### 第1步：分析内容

**加载参考文件**：读取此 skill 中的 `references/analysis-framework.md`。

1. 保存源内容（文件路径或粘贴内容 → 使用 `write_file` 写入 `source.md`）
   - **备份规则**：若 `source.md` 已存在，重命名为 `source-backup-YYYYMMDD-HHMMSS.md`
2. 分析：主题、数据类型、复杂度、语气、受众
3. 检测源语言和用户语言
4. 从用户输入中提取设计指令
5. 将分析结果保存至 `analysis.md`
   - **备份规则**：若 `analysis.md` 已存在，重命名为 `analysis-backup-YYYYMMDD-HHMMSS.md`

详细格式见 `references/analysis-framework.md`。

### 第2步：生成结构化内容 → `structured-content.md`

将内容转化为信息图结构：
1. 标题与学习目标
2. 各节包含：核心概念、内容（原文）、视觉元素、文字标签
3. 数据点（所有统计数据/引用原样复制）
4. 用户的设计指令

**规则**：仅使用 Markdown。不添加新信息。忠实保留数据。去除所有凭据或密钥。

详细格式见 `references/structured-content-template.md`。

### 第3步：推荐组合

**3.1 优先检查关键词快捷方式**：若用户输入匹配**关键词快捷方式**表中的关键词，自动选择对应布局，并将关联风格作为首选推荐。跳过基于内容的布局推断。

**3.2 否则**，根据以下因素推荐 3-5 个布局×风格组合：
- 数据结构 → 匹配布局
- 内容语气 → 匹配风格
- 受众期望
- 用户设计指令

### 第4步：确认选项

使用 `clarify` 工具与用户确认选项。由于 `clarify` 每次只处理一个问题，优先提问最重要的问题：

**Q1 — 组合**：展示 3 个以上布局×风格组合及理由，请用户选择。

**Q2 — 宽高比**：询问宽高比偏好（landscape/portrait/square 或自定义 W:H）。

**Q3 — 语言**（仅当源语言 ≠ 用户语言时）：询问文字内容使用哪种语言。

### 第5步：生成 Prompt → `prompts/infographic.md`

**备份规则**：若 `prompts/infographic.md` 已存在，重命名为 `prompts/infographic-backup-YYYYMMDD-HHMMSS.md`

**加载参考文件**：读取所选布局的 `references/layouts/<layout>.md` 和风格的 `references/styles/<style>.md`。

组合以下内容：
1. `references/layouts/<layout>.md` 中的布局定义
2. `references/styles/<style>.md` 中的风格定义
3. `references/base-prompt.md` 中的基础模板
4. 第2步的结构化内容
5. 所有文字使用已确认的语言

**`{{ASPECT_RATIO}}` 宽高比解析**：
- 命名预设 → 比例字符串：landscape→`16:9`，portrait→`9:16`，square→`1:1`
- 自定义 W:H 比例 → 原样使用（如 `3:4`、`4:3`、`2.35:1`）

使用 `write_file` 将组装好的 prompt 保存至 `prompts/infographic.md`。

### 第6步：生成图像

使用 `image_generate` 工具，传入第5步组装的 prompt。

- 将宽高比映射到 image_generate 的格式：`16:9` → `landscape`，`9:16` → `portrait`，`1:1` → `square`
- 自定义比例时，选择最接近的命名宽高比
- 失败时自动重试一次
- 将生成的图像 URL/路径保存至输出目录

### 第7步：输出摘要

报告：主题、布局、风格、宽高比、语言、输出路径、已创建文件。

## 参考文件

- `references/analysis-framework.md` — 分析方法论
- `references/structured-content-template.md` — 内容格式
- `references/base-prompt.md` — Prompt 模板
- `references/layouts/<layout>.md` — 21种布局定义
- `references/styles/<style>.md` — 21种风格定义

## 注意事项

1. **数据完整性至关重要** — 绝不摘要、改写或修改源统计数据。"增长73%"必须保持为"增长73%"，而非"显著增长"。
2. **去除密钥** — 在将源内容写入任何输出文件前，始终扫描 API 密钥、token 或凭据。
3. **每节一个信息点** — 信息图的每个节应传达一个清晰概念。内容过载会降低可读性。
4. **风格一致性** — 参考文件中的风格定义必须在整个信息图中一致应用，不得混用风格。
5. **image_generate 宽高比** — 该工具仅支持 `landscape`、`portrait` 和 `square`。自定义比例如 `3:4` 应映射到最接近的选项（此例为 portrait）。