---
title: "宝玉文章配图 — 文章插图：类型 × 风格 × 配色一致性"
sidebar_label: "宝玉文章配图"
description: "文章插图：类型 × 风格 × 配色一致性"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 宝玉文章配图

文章插图：类型 × 风格 × 配色一致性。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/creative/baoyu-article-illustrator` 安装 |
| 路径 | `optional-skills/creative/baoyu-article-illustrator` |
| 版本 | `1.57.0` |
| 作者 | 宝玉 (JimLiu) |
| 许可证 | MIT |
| 平台 | linux、macos、windows |
| 标签 | `article-illustration`、`creative`、`image-generation` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# 文章配图

改编自 [baoyu-article-illustrator](https://github.com/JimLiu/baoyu-skills)，适配 Hermes Agent 的工具生态。

分析文章，识别插图位置，使用**类型 × 风格 × 配色**一致性生成图像。

## 何时使用

当用户要求为文章配图、添加图片、为内容生成插图，或使用"为文章配图"、"illustrate article"、"add images"等短语时触发。用户提供文章（文件路径或粘贴内容），可选地指定类型、风格、配色或密度。

## 三个维度

| 维度 | 控制 | 示例 |
|------|------|------|
| **类型** | 信息结构 | 信息图、场景、流程图、对比、框架、时间线 |
| **风格** | 渲染方式 | notion、温暖、极简、蓝图、水彩、优雅 |
| **配色** | 配色方案（可选） | 马卡龙、温暖、霓虹 — 覆盖风格的默认颜色 |

自由组合：`type=infographic, style=vector-illustration, palette=macaron`。

或使用预设：`edu-visual` → 一次完成类型+风格+配色。参见 [style-presets.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/style-presets.md)。

## 类型

| 类型 | 最适合 |
|------|--------|
| `infographic` | 数据、指标、技术 |
| `scene` | 叙事、情感 |
| `flowchart` | 流程、工作流 |
| `comparison` | 并排对比、选项 |
| `framework` | 模型、架构 |
| `timeline` | 历史、演进 |

## 风格

参见 [references/styles.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/styles.md) 了解核心风格、完整画廊和类型×风格兼容性。

## 输出结构

```
{output-dir}/
├── source-{slug}.{ext}    # 仅粘贴内容
├── outline.md
├── prompts/
│   └── NN-{type}-{slug}.md
└── NN-{type}-{slug}.png
```

**默认输出目录**：

| 输入 | 输出目录 | Markdown 插入路径 |
|------|----------|-------------------|
| 文章文件路径 | `{article-dir}/imgs/` | `imgs/NN-{type}-{slug}.png` |
| 粘贴内容 | `illustrations/{topic-slug}/`（cwd） | `illustrations/{topic-slug}/NN-{type}-{slug}.png` |

如果用户要求不同的布局（例如图片与文章并列或 `illustrations/` 子目录），照做。

## 核心原则

- **可视化概念而非隐喻** — 如果文章使用隐喻（如"电锯切西瓜"），可视化底层概念，而非字面图像。
- **标签使用文章数据** — 文章中的实际数字、术语和引用，而非通用占位符。
- **提示文件是可复现性记录** — 每个插图在生成任何图像之前必须在 `prompts/` 下有保存的提示文件。
- **清除密钥** — 在写入任何内容到磁盘之前扫描源内容中的 API 密钥、token 或凭据。

## 工作流程

```
- [ ] 步骤 1：检测参考图像（如提供）
- [ ] 步骤 2：分析内容
- [ ] 步骤 3：确认设置（逐个澄清问题）
- [ ] 步骤 4：生成大纲
- [ ] 步骤 5：生成提示
- [ ] 步骤 6：生成图像（image_generate）
- [ ] 步骤 7：完成
```

### 步骤 1：检测参考图像

如果用户提供参考图像（路径、附件或 URL）：

1. 对每个参考，调用 `vision_analyze` 并传入路径/URL 及请求分析风格、配色、构图和主题的问题。通过 `write_file` 将返回的描述记录到 `{output-dir}/references/NN-ref-{slug}.md`。
2. **不要**尝试通过 `write_file` / `read_file` 复制二进制文件——这些仅支持文本。如需本地副本，使用 `terminal`（`cp`）。
3. 由于 `image_generate` 不接受图像输入，视觉描述将在步骤 5 中嵌入提示。

### 步骤 2：分析

| 分析 | 输出 |
|------|------|
| 内容类型 | 技术 / 教程 / 方法论 / 叙事 |
| 目的 | 信息 / 可视化 / 想象 |
| 核心论点 | 2-5 个要点 |
| 位置 | 插图能增加价值的地方 |

### 步骤 3：确认设置

使用 `clarify` 工具。由于每次处理一个问题，先问最重要的。

| 顺序 | 问题 | 选项 |
|------|------|------|
| Q1 | **预设或类型** | [推荐预设]、[备选预设]，或手动：infographic、scene、flowchart、comparison、framework、timeline、mixed |
| Q2 | **密度** | 极简(1-2)、平衡(3-5)、按章节（推荐）、丰富(6+) |
| Q3 | **风格** *（Q1 选了预设则跳过）* | [推荐]、minimal-flat、sci-fi、hand-drawn、editorial、scene、poster |
| Q4 | **配色** *（可选）* | 默认（风格颜色）、macaron、warm、neon |
| Q5 | **语言** *（仅文章语言不明确时）* | 文章语言 / 用户语言 |

### 步骤 4：生成大纲 → `outline.md`

使用 `write_file` 保存 `{output-dir}/outline.md`，包含 frontmatter（type、density、style、palette、image_count）和每条插图的条目。

### 步骤 5：生成提示

**阻塞性要求**：每个插图在生成任何图像之前必须有保存的提示文件——提示文件是可复现性记录。

对于每个插图：
1. 按 [references/prompt-construction.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/prompt-construction.md) 创建提示文件。
2. 使用 `write_file` 保存到 `{output-dir}/prompts/NN-{type}-{slug}.md`。
3. 提示必须使用带结构化部分的类型特定模板（ZONES / LABELS / COLORS / STYLE / ASPECT）。
4. LABELS 必须包含文章特定数据：实际数字、术语、指标、引用。

### 步骤 6：生成图像

对于每个提示文件：
1. 调用 `image_generate(prompt=..., aspect_ratio=...)`。`image_generate` 返回包含图像 URL 的 JSON 结果；它不写入磁盘也不接受输出路径。
2. 将提示的 `ASPECT` 映射到 `image_generate` 的枚举：`16:9` → `landscape`、`9:16` → `portrait`、`1:1` → `square`。
3. 通过 `terminal` 将返回的 URL 下载到 `{output-dir}/NN-{type}-{slug}.png`。
4. 生成失败时自动重试一次。

注意：底层图像生成后端是用户配置的（默认：FAL FLUX 2 Klein 9B），不是代理可选的。

### 步骤 7：完成

在相应段落后插入 `![description](...)`。Alt 文本：文章语言的简洁描述。

## 陷阱

1. **数据完整性至上** — 永远不要摘要、转述或更改源统计数据。"73% increase"保持为"73% increase"。
2. **清除密钥** — 在将源内容包含在任何输出文件中之前扫描 API 密钥、token 或凭据。
3. **不要字面插图隐喻** — 可视化底层概念。
4. **提示文件是强制的** — 没有保存的提示文件就不要生成图像。
5. **`image_generate` 宽高比** — 工具支持 `landscape`、`portrait` 和 `square`。自定义比例映射到最近的选项。
6. **`image_generate` 返回 URL 而非本地文件** — 在将本地图像路径插入文章前始终通过 `terminal`（`curl`）下载。
7. **不能从代理选择后端** — `image_generate` 使用用户配置的任何模型。
