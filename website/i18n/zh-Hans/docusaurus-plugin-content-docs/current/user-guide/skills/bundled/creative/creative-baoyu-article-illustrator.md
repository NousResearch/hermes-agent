---
title: "宝玉文章配图助手 — 文章插图：类型 × 风格 × 调色板一致性"
sidebar_label: "宝玉文章配图助手"
description: "文章插图：类型 × 风格 × 调色板一致性"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 宝玉文章配图助手

文章插图：类型 × 风格 × 调色板一致性。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/creative/baoyu-article-illustrator` |
| 版本 | `1.57.0` |
| 作者 | 宝玉 (JimLiu) |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `article-illustration`, `creative`, `image-generation` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# 文章配图助手

改编自 [baoyu-article-illustrator](https://github.com/JimLiu/baoyu-skills)，适配 Hermes Agent 的工具生态系统。

分析文章，识别插图位置，以 **类型 × 风格 × 调色板** 一致性生成图像。

## 使用时机

当用户要求为文章配图、添加图片、生成插图，或使用"为文章配图"、"illustrate article"、"add images"等短语时，触发此 skill。用户提供文章（文件路径或粘贴内容），并可选择指定类型、风格、调色板或密度。

## 三个维度

| 维度 | 控制内容 | 示例 |
|-----------|----------|----------|
| **类型（Type）** | 信息结构 | infographic、scene、flowchart、comparison、framework、timeline |
| **风格（Style）** | 渲染方式 | notion、warm、minimal、blueprint、watercolor、elegant |
| **调色板（Palette）** | 配色方案（可选） | macaron、warm、neon — 覆盖风格的默认颜色 |

可自由组合：`type=infographic, style=vector-illustration, palette=macaron`。

或使用预设：`edu-visual` → 一次性指定 type + style + palette。参见 [style-presets.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-article-illustrator/references/style-presets.md)。

## 类型

| 类型 | 最适合 |
|------|----------|
| `infographic` | 数据、指标、技术内容 |
| `scene` | 叙事、情感表达 |
| `flowchart` | 流程、工作流 |
| `comparison` | 并排对比、选项比较 |
| `framework` | 模型、架构 |
| `timeline` | 历史、演进 |

## 风格

参见 [references/styles.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-article-illustrator/references/styles.md)，包含核心风格、完整图库及类型 × 风格兼容性说明。

## 输出结构

<!-- ascii-guard-ignore -->
```
{output-dir}/
├── source-{slug}.{ext}    # 仅用于粘贴内容
├── outline.md
├── prompts/
│   └── NN-{type}-{slug}.md
└── NN-{type}-{slug}.png
```
<!-- ascii-guard-ignore-end -->

**默认输出目录**：

| 输入 | 输出目录 | Markdown 插入路径 |
|-------|------------------|----------------------|
| 文章文件路径 | `{article-dir}/imgs/` | `imgs/NN-{type}-{slug}.png` |
| 粘贴内容 | `illustrations/{topic-slug}/`（当前工作目录） | `illustrations/{topic-slug}/NN-{type}-{slug}.png` |

如果用户要求不同的布局（例如图片与文章并排，或使用 `illustrations/` 子目录），请遵从用户要求。

**Slug**：2-4 个单词，kebab-case 格式。**冲突时**：追加 `-YYYYMMDD-HHMMSS`。

## 核心原则

- **可视化概念，而非隐喻** — 若文章使用了隐喻（如"电锯切西瓜"），应插图展示其底层概念，而非字面图像。
- **标签使用文章数据** — 使用文章中的实际数字、术语和引用，而非通用占位符。
- **Prompt 文件是可复现性记录** — 每张插图在生成图像前必须在 `prompts/` 下保存对应的 prompt 文件。
- **清除敏感信息** — 在将任何内容写入磁盘前，扫描源内容中的 API 密钥、token 或凭据。

## 工作流程

```
- [ ] 步骤 1：检测参考图像（如有提供）
- [ ] 步骤 2：分析内容
- [ ] 步骤 3：确认设置（使用 clarify 工具，每次一个问题）
- [ ] 步骤 4：生成大纲
- [ ] 步骤 5：生成 prompt
- [ ] 步骤 6：生成图像（image_generate）
- [ ] 步骤 7：收尾
```

### 步骤 1：检测参考图像

如果用户提供了参考图像（内联粘贴的路径、附件或 URL）：

1. 对每个参考图像，使用路径/URL 调用 `vision_analyze`，询问风格、调色板、构图和主题。将返回的描述通过 `write_file` 记录到 `{output-dir}/references/NN-ref-{slug}.md`。
2. **不要**尝试通过 `write_file` / `read_file` 复制二进制文件 — 这些工具仅支持文本。如需本地副本留存记录，使用 `terminal`（`cp "$src" "{output-dir}/references/NN-ref-{slug}.{ext}"`）。skill 本身无需读取二进制文件；它基于 vision 描述工作。
3. 由于 `image_generate` 不接受图像输入，vision 描述将在步骤 5 中嵌入到 prompt 中。

完整流程：[references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-article-illustrator/references/workflow.md#step-1-detect-reference-images)。

### 步骤 2：分析

| 分析项 | 输出 |
|----------|--------|
| 内容类型 | 技术型 / 教程型 / 方法论型 / 叙事型 |
| 目的 | 信息传递 / 可视化 / 想象力激发 |
| 核心论点 | 2-5 个主要观点 |
| 插图位置 | 插图能增加价值的位置 |

读取源文件（文件路径 → `read_file`，或粘贴文本），并使用 `write_file` 将分析结果写入 `{output-dir}/analysis.md`。

完整流程：[references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-article-illustrator/references/workflow.md#step-2-analyze)。

### 步骤 3：确认设置

使用 `clarify` 工具。由于 `clarify` 每次只处理一个问题，请先问最重要的问题。若用户请求中已包含答案，则跳过对应问题。

| 顺序 | 问题 | 选项 |
|-------|----------|---------|
| Q1 | **预设或类型** | [推荐预设]、[备选预设]，或手动选择：infographic、scene、flowchart、comparison、framework、timeline、mixed |
| Q2 | **密度** | minimal（1-2 张）、balanced（3-5 张）、per-section（推荐）、rich（6+ 张） |
| Q3 | **风格** *(若 Q1 已选预设则跳过)* | [推荐]、minimal-flat、sci-fi、hand-drawn、editorial、scene、poster |
| Q4 | **调色板** *(可选)* | 默认（风格颜色）、macaron、warm、neon |
| Q5 | **语言** *(仅当文章语言不明确时)* | 文章语言 / 用户语言 |

连续 `clarify` 问题不超过 2-3 个。若用户在请求中已指定这些内容，则完全跳过。

完整流程：[references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-article-illustrator/references/workflow.md#step-3-confirm-settings)。

### 步骤 4：生成大纲 → `outline.md`

使用 `write_file` 将 `{output-dir}/outline.md` 保存，包含 frontmatter（type、density、style、palette、image_count）及每张插图的条目：

```yaml
## Illustration 1
**Position**: [section/paragraph]
**Purpose**: [why]
**Visual Content**: [what to show]
**Filename**: 01-infographic-concept-name.png
```

完整模板：[references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-article-illustrator/references/workflow.md#step-4-generate-outline)。

### 步骤 5：生成 Prompt

**阻塞条件**：每张插图必须在生成图像前保存 prompt 文件 — prompt 文件是可复现性记录。

对每张插图：

1. 按照 [references/prompt-construction.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-article-illustrator/references/prompt-construction.md) 创建 prompt 文件。
2. 使用 `write_file` 将文件保存到 `{output-dir}/prompts/NN-{type}-{slug}.md`，包含 YAML frontmatter。
3. Prompt 必须使用特定类型的模板，包含结构化章节（ZONES / LABELS / COLORS / STYLE / ASPECT）。
4. LABELS 必须包含文章特定数据：实际数字、术语、指标、引用。
5. 按 prompt frontmatter 处理参考图像（`direct`/`style`/`palette`）— 对于 `direct` 用法，在 prompt 中嵌入参考图像的文字描述（因为 `image_generate` 不接受参考图像输入）。

### 步骤 6：生成图像

对每个 prompt 文件：

1. 调用 `image_generate(prompt=..., aspect_ratio=...)`。`image_generate` 返回包含图像 URL 的 JSON 结果；它不会写入磁盘，也不接受输出路径参数。
2. 将 prompt 的 `ASPECT` 映射到 `image_generate` 的枚举值：`16:9` → `landscape`，`9:16` → `portrait`，`1:1` → `square`。自定义比例 → 映射到最近的命名比例。
3. 通过 `terminal` 将返回的 URL 下载到 `{output-dir}/NN-{type}-{slug}.png`（例如 `curl -sSL -o "{output-dir}/NN-{type}-{slug}.png" "{url}"`）。
4. 生成失败时，自动重试一次。

注意：底层图像生成后端由用户配置（默认：FAL FLUX 2 Klein 9B），agent 无法通过 `image_generate` 选择后端。不要在 prompt 中写入模型名称并期望其路由生效。

### 步骤 7：收尾

在对应段落后插入 `![描述](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-article-illustrator/{relative-path}/NN-{type}-{slug}.png)`。Alt 文本：用文章语言简洁描述。

报告：

```
Article Illustration Complete!
Article: [path] | Type: [type] | Density: [level] | Style: [style] | Palette: [palette or default]
Images: X/N generated
```

## 修改操作

| 操作 | 步骤 |
|--------|-------|
| 编辑 | 更新 prompt → 重新生成 → 更新引用 |
| 添加 | 确定位置 → 编写 prompt → 生成 → 更新大纲 → 插入 |
| 删除 | 删除文件 → 移除引用 → 更新大纲 |

## 参考文档

| 文件 | 内容 |
|------|---------|
| [references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-article-illustrator/references/workflow.md) | 详细流程 |
| [references/usage.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-article-illustrator/references/usage.md) | 调用示例 |
| [references/styles.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-article-illustrator/references/styles.md) | 风格图库 + 调色板图库 |
| [references/style-presets.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-article-illustrator/references/style-presets.md) | 预设快捷方式（type + style + palette） |
| [references/prompt-construction.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-article-illustrator/references/prompt-construction.md) | Prompt 模板 |

## 常见陷阱

1. **数据完整性至关重要** — 绝不摘要、改写或篡改源统计数据。"73% increase"保持原样。
2. **清除敏感信息** — 在将任何内容写入输出文件前，扫描源内容中的 API 密钥、token 或凭据。
3. **不要字面插图隐喻** — 可视化底层概念，而非字面图像。
4. **Prompt 文件是强制要求** — 没有保存 prompt 文件就不能生成图像。该文件是后续重新生成或切换后端的依据。
5. **`image_generate` 的宽高比** — 该工具支持 `landscape`、`portrait` 和 `square`。自定义比例映射到最近的选项。
6. **`image_generate` 返回 URL，而非本地文件** — 在将本地图像路径插入文章前，始终通过 `terminal`（`curl`）下载。
7. **agent 无法选择后端** — `image_generate` 使用用户配置的模型（默认：FAL FLUX 2 Klein 9B）。不要在 prompt 中写入 `"use <model> to generate this"` 并期望其路由生效。