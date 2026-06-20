---
title: "Baoyu Comic — 知识漫画：教育、传记、教程"
sidebar_label: "Baoyu Comic"
description: "知识漫画：教育、传记、教程"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Baoyu Comic

知识漫画（Knowledge comics）：教育、传记、教程。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/creative/baoyu-comic` |
| 版本 | `1.56.1` |
| 作者 | 宝玉 (JimLiu) |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `comic`, `knowledge-comic`, `creative`, `image-generation` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# 知识漫画创作器

改编自 [baoyu-comic](https://github.com/JimLiu/baoyu-skills)，适配 Hermes Agent 的工具生态系统。

创作具有灵活艺术风格 × 基调组合的原创知识漫画。

## 使用时机

当用户要求创作知识/教育漫画、传记漫画、教程漫画，或使用"知识漫画"、"教育漫画"、"Logicomix 风格"等词语时，触发此 skill。用户提供内容（文本、文件路径、URL 或主题），并可选择指定艺术风格、基调、版式、宽高比或语言。

## 参考图片

Hermes 的 `image_generate` 工具**仅接受 prompt（提示词）**——它接受文本 prompt 和宽高比，并返回图片 URL。它**不**接受参考图片。当用户提供参考图片时，将其用于**以文字提取特征**，并嵌入每页 prompt 中：

**接收方式**：当用户提供文件路径时接受（或在对话中粘贴图片）。
- 文件路径 → 复制到漫画输出目录下的 `refs/NN-ref-{slug}.{ext}`，用于溯源
- 粘贴图片但无路径 → 通过 `clarify` 向用户询问路径，或以文字形式提取风格特征作为备选
- 无参考图片 → 跳过此部分

**使用模式**（每张参考图片）：

| 用途 | 效果 |
|-------|--------|
| `style` | 提取风格特征（线条处理、纹理、氛围），追加到每页 prompt 正文 |
| `palette` | 提取十六进制颜色，追加到每页 prompt 正文 |
| `scene` | 提取场景构图或主体说明，追加到相关页面 |

**存在参考图片时，在每页 prompt 的 frontmatter 中记录**：

```yaml
references:
  - ref_id: 01
    filename: 01-ref-scene.png
    usage: style
    traits: "muted earth tones, soft-edged ink wash, low-contrast backgrounds"
```

角色一致性通过 `characters/characters.md` 中的**文字描述**来驱动（在步骤 3 中编写），并内联嵌入每页 prompt（步骤 5）。步骤 7.1 中可选生成的 PNG 角色表是面向用户的审阅产物，而非 `image_generate` 的输入。

## 选项

### 视觉维度

| 选项 | 可选值 | 说明 |
|--------|--------|-------------|
| 艺术风格 | ligne-claire（默认）、manga、realistic、ink-brush、chalk、minimalist | 艺术风格 / 渲染技术 |
| 基调 | neutral（默认）、warm、dramatic、romantic、energetic、vintage、action | 情绪 / 氛围 |
| 版式 | standard（默认）、cinematic、dense、splash、mixed、webtoon、four-panel | 分格排列方式 |
| 宽高比 | 3:4（默认，竖版）、4:3（横版）、16:9（宽屏） | 页面宽高比 |
| 语言 | auto（默认）、zh、en、ja 等 | 输出语言 |
| 参考图片 | 文件路径 | 用于风格 / 调色板特征提取的参考图片（不传入图像模型）。见上方[参考图片](#reference-images)。 |

### 部分工作流选项

| 选项 | 说明 |
|--------|-------------|
| 仅分镜 | 仅生成分镜，跳过 prompt 和图片 |
| 仅 prompt | 生成分镜 + prompt，跳过图片 |
| 仅图片 | 从现有 prompts 目录生成图片 |
| 重新生成第 N 页 | 仅重新生成指定页面（如 `3` 或 `2,5,8`） |

详情：[references/partial-workflows.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/partial-workflows.md)

### 艺术风格、基调与预设目录

- **艺术风格**（6 种）：`ligne-claire`、`manga`、`realistic`、`ink-brush`、`chalk`、`minimalist`。完整定义见 `references/art-styles/<style>.md`。
- **基调**（7 种）：`neutral`、`warm`、`dramatic`、`romantic`、`energetic`、`vintage`、`action`。完整定义见 `references/tones/<tone>.md`。
- **预设**（5 种），具有超出普通艺术风格+基调的特殊规则：

  | 预设 | 等效组合 | Hook |
  |--------|-----------|------|
  | `ohmsha` | manga + neutral | 视觉隐喻、无纯对话页、道具揭示 |
  | `wuxia` | ink-brush + action | 气效、战斗视觉、氛围感 |
  | `shoujo` | manga + romantic | 装饰元素、眼部细节、浪漫节拍 |
  | `concept-story` | manga + warm | 视觉符号体系、成长弧线、对话与动作平衡 |
  | `four-panel` | minimalist + neutral + four-panel 版式 | 起承转合结构、黑白+点缀色、火柴人角色 |

  完整规则见 `references/presets/<preset>.md`——选择预设时加载对应文件。

- **兼容性矩阵**和**内容信号 → 预设**对照表见 [references/auto-selection.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/auto-selection.md)。在步骤 2 推荐组合前请先阅读。

## 文件结构

输出目录：`comic/{topic-slug}/`
- Slug：从主题中取 2-4 个词，使用 kebab-case（如 `alan-turing-bio`）
- 冲突时：追加时间戳（如 `turing-story-20260118-143052`）

**内容**：
| 文件 | 说明 |
|------|-------------|
| `source-{slug}.md` | 保存的源内容（kebab-case slug 与输出目录一致） |
| `analysis.md` | 内容分析 |
| `storyboard.md` | 含分格说明的分镜脚本 |
| `characters/characters.md` | 角色定义 |
| `characters/characters.png` | 角色参考表（从 `image_generate` 下载） |
| `prompts/NN-{cover\|page}-[slug].md` | 生成 prompt |
| `NN-{cover\|page}-[slug].png` | 生成的图片（从 `image_generate` 下载） |
| `refs/NN-ref-{slug}.{ext}` | 用户提供的参考图片（可选，用于溯源） |

## 语言处理

**检测优先级**：
1. 用户指定语言（显式选项）
2. 用户对话语言
3. 源内容语言

**规则**：对所有交互使用用户的输入语言：
- 分镜大纲和场景描述
- 图片生成 prompt
- 用户选择选项和确认信息
- 进度更新、问题、错误、摘要

技术术语保持英文。

## 工作流

### 进度清单

```
Comic Progress:
- [ ] Step 1: Setup & Analyze
  - [ ] 1.1 Analyze content
  - [ ] 1.2 Check existing directory
- [ ] Step 2: Confirmation - Style & options ⚠️ REQUIRED
- [ ] Step 3: Generate storyboard + characters
- [ ] Step 4: Review outline (conditional)
- [ ] Step 5: Generate prompts
- [ ] Step 6: Review prompts (conditional)
- [ ] Step 7: Generate images
  - [ ] 7.1 Generate character sheet (if needed) → characters/characters.png
  - [ ] 7.2 Generate pages (with character descriptions embedded in prompt)
- [ ] Step 8: Completion report
```

### 流程

```
Input → Analyze → [Check Existing?] → [Confirm: Style + Reviews] → Storyboard → [Review?] → Prompts → [Review?] → Images → Complete
```

### 步骤摘要

| 步骤 | 操作 | 关键输出 |
|------|--------|------------|
| 1.1 | 分析内容 | `analysis.md`、`source-{slug}.md` |
| 1.2 | 检查现有目录 | 处理冲突 |
| 2 | 确认风格、重点、受众、审阅方式 | 用户偏好 |
| 3 | 生成分镜 + 角色 | `storyboard.md`、`characters/` |
| 4 | 审阅大纲（如已请求） | 用户确认 |
| 5 | 生成 prompt | `prompts/*.md` |
| 6 | 审阅 prompt（如已请求） | 用户确认 |
| 7.1 | 生成角色表（如需要） | `characters/characters.png` |
| 7.2 | 生成页面 | `*.png` 文件 |
| 8 | 完成报告 | 摘要 |

### 用户问题

使用 `clarify` 工具确认选项。由于 `clarify` 每次只处理一个问题，请先提出最重要的问题，然后依次进行。完整的步骤 2 问题集见 [references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/workflow.md)。

**超时处理（关键）**：`clarify` 可能返回 `"The user did not provide a response within the time limit. Use your best judgement to make the choice and proceed."` ——这**不是**用户对所有选项使用默认值的同意。

- 仅将其视为**该单个问题**的默认值。继续依次提出步骤 2 的其余问题；每个问题都是独立的确认节点。
- **在下一条消息中向用户明确展示该默认值**，以便其有机会纠正：例如 `"Style: defaulted to ohmsha preset (clarify timed out). Say the word to switch."` ——未报告的默认值与从未询问过无异。
- 在一次超时后，**不要**将步骤 2 折叠为"全部使用默认值"的单次处理。如果用户确实不在，他们对所有五个问题同样不在——但他们可以在回来后纠正可见的默认值，而无法纠正不可见的默认值。

### 步骤 7：图片生成

所有图片渲染均使用 Hermes 内置的 `image_generate` 工具。其 schema 仅接受 `prompt` 和 `aspect_ratio`（`landscape` | `portrait` | `square`）；它**返回 URL**，而非本地文件。因此，每张生成的页面或角色表都必须下载到输出目录。

**Prompt 文件要求（硬性规定）**：在调用 `image_generate` 之前，必须将每张图片的完整最终 prompt 写入 `prompts/` 下的独立文件（命名规则：`NN-{type}-[slug].md`）。Prompt 文件是可复现性记录。

**宽高比映射** ——分镜的 `aspect_ratio` 字段映射到 `image_generate` 的格式如下：

| 分镜比例 | `image_generate` 格式 |
|------------------|-------------------------|
| `3:4`、`9:16`、`2:3` | `portrait` |
| `4:3`、`16:9`、`3:2` | `landscape` |
| `1:1` | `square` |

**下载步骤** ——每次调用 `image_generate` 后：
1. 从工具结果中读取 URL
2. 使用**绝对**输出路径获取图片字节，例如：
   `curl -fsSL "<url>" -o /abs/path/to/comic/<slug>/NN-page-<slug>.png`
3. 在继续下一页之前，验证该文件存在于该确切路径且非空

**永远不要依赖 shell CWD 持久性来指定 `-o` 路径。** 终端工具的持久 shell CWD 可能在批次之间发生变化（会话过期、`TERMINAL_LIFETIME_SECONDS`、失败的 `cd` 导致停留在错误目录）。`curl -o relative/path.png` 是一个隐蔽的陷阱：如果 CWD 已偏移，文件会落在其他地方且不报错。**始终向 `-o` 传递完全限定的绝对路径**，或向终端工具传递 `workdir=<abs path>`。2026 年 4 月事故：一个 10 页漫画的第 06-09 页落在了仓库根目录，而非 `comic/<slug>/`，原因是第 3 批次继承了第 2 批次的过期 CWD，`curl -o 06-page-skills.png` 写入了错误目录。随后 agent 花了数轮声称文件存在于它们实际不在的位置。

**7.1 角色表** ——当漫画为多页且有反复出现的角色时，生成角色表（保存至 `characters/characters.png`，宽高比 `landscape`）。对于简单预设（如 four-panel minimalist）或单页漫画可跳过。在调用 `image_generate` 之前，`characters/characters.md` 中的 prompt 文件必须已存在。渲染出的 PNG 是**面向用户的审阅产物**（供用户直观验证角色设计），也是后续重新生成或手动编辑 prompt 的参考——它**不**驱动步骤 7.2。页面 prompt 已在步骤 5 中根据 `characters/characters.md` 中的**文字描述**编写；`image_generate` 无法接受图片作为视觉输入。

**7.2 页面** ——在调用 `image_generate` 之前，每页的 prompt 必须已存在于 `prompts/NN-{cover|page}-[slug].md`。由于 `image_generate` 仅接受 prompt，角色一致性通过在步骤 5 中**将角色描述（来源于 `characters/characters.md`）内联嵌入每页 prompt** 来保证。无论步骤 7.1 是否生成 PNG 表，嵌入方式均相同；PNG 仅作为审阅/重新生成的辅助工具。

**备份规则**：现有的 `prompts/…md` 和 `…png` 文件 → 在重新生成前，以 `-backup-YYYYMMDD-HHMMSS` 后缀重命名。

完整的逐步工作流（分析、分镜、审阅节点、重新生成变体）：[references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/workflow.md)。

## 参考资料

**核心模板**：
- [analysis-framework.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/analysis-framework.md) - 深度内容分析
- [character-template.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/character-template.md) - 角色定义格式
- [storyboard-template.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/storyboard-template.md) - 分镜结构
- [ohmsha-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/ohmsha-guide.md) - Ohmsha manga 细节

**风格定义**：
- `references/art-styles/` - 艺术风格（ligne-claire、manga、realistic、ink-brush、chalk、minimalist）
- `references/tones/` - 基调（neutral、warm、dramatic、romantic、energetic、vintage、action）
- `references/presets/` - 含特殊规则的预设（ohmsha、wuxia、shoujo、concept-story、four-panel）
- `references/layouts/` - 版式（standard、cinematic、dense、splash、mixed、webtoon、four-panel）

**工作流**：
- [workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/workflow.md) - 完整工作流详情
- [auto-selection.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/auto-selection.md) - 内容信号分析
- [partial-workflows.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/partial-workflows.md) - 部分工作流选项

## 页面修改

| 操作 | 步骤 |
|--------|-------|
| **编辑** | **先更新 prompt 文件** → 重新生成图片 → 下载新 PNG |
| **添加** | 在指定位置创建 prompt → 嵌入角色描述后生成 → 重新编号后续页面 → 更新分镜 |
| **删除** | 删除文件 → 重新编号后续页面 → 更新分镜 |

**重要**：更新页面时，务必**先**更新 prompt 文件（`prompts/NN-{cover|page}-[slug].md`），再重新生成。这确保变更有据可查且可复现。

## 注意事项

- 图片生成：每页 10-30 秒；失败时自动重试一次
- **始终下载** `image_generate` 返回的 URL 到本地 PNG——下游工具（以及用户审阅）期望文件在输出目录中，而非临时 URL
- **`curl -o` 使用绝对路径** ——永远不要依赖持久 shell 的 CWD 跨批次持久性。隐蔽陷阱：文件落在错误目录，随后对预期路径执行 `ls` 显示为空。见步骤 7"下载步骤"。
- 对敏感公众人物使用风格化替代形象
- **步骤 2 确认为必须** ——不可跳过
- **步骤 4/6 为条件性** ——仅在用户于步骤 2 中请求时执行
- **步骤 7.1 角色表** ——推荐用于多页漫画，简单预设可选。PNG 是审阅/重新生成辅助工具；页面 prompt（在步骤 5 中编写）使用 `characters/characters.md` 中的文字描述，而非 PNG。`image_generate` 不接受图片作为视觉输入
- **清除敏感信息** ——在写入任何输出文件之前，扫描源内容中的 API 密钥、token 或凭据