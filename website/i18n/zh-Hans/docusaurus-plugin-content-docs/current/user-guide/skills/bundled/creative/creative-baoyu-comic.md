---
title: "Baoyu Comic — 知识漫画：教育、传记、教程"
sidebar_label: "Baoyu Comic"
description: "知识漫画：教育、传记、教程"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Baoyu Comic

知识漫画：教育、传记、教程。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/creative/baoyu-comic` |
| 版本 | `1.56.1` |
| 作者 | 宝玉 (JimLiu) |
| 许可证 | MIT |
| 标签 | `comic`、`knowledge-comic`、`creative`、`image-generation` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 加载此技能时触发的完整技能定义。这是代理激活技能时看到的指令。
:::

# 知识漫画创建器

为 Hermes Agent 的工具生态系统改编自 [baoyu-comic](https://github.com/JimLiu/baoyu-skills)。

使用灵活的艺术风格 × 色调组合创建原创知识漫画。

## 使用场景

当用户要求创建知识/教育漫画、传记漫画、教程漫画，或使用术语如"知识漫画"、"教育漫画"或"Logicomix-style"时触发此技能。用户提供内容（文本、文件路径、URL 或主题），并可选择指定艺术风格、色调、布局、宽高比或语言。

## 参考图像

Hermes 的 `image_generate` 工具**仅支持提示词** — 它接受文本提示和宽高比，并返回图像 URL。它**不**接受参考图像。当用户提供参考图像时，使用它**以文本形式提取特征**，这些特征被嵌入到每个页面提示中：

**接收**：接受文件路径（或对话中粘贴的图像）。
  - 文件路径 → 复制到漫画输出旁边的 `refs/NN-ref-{slug}.{ext}` 以记录来源
  - 粘贴的图像无路径 → 通过 `clarify` 向用户询问路径，或以文本形式口头提取风格特征作为后备
  - 无参考 → 跳过此部分

**使用模式**（每个参考）：
| 用法 | 效果 |
|-------|--------|
| `style` | 提取风格特征（线条处理、纹理、色调）并追加到每个页面提示正文 |
| `palette` | 提取十六进制颜色并追加到每个页面提示正文 |
| `scene` | 提取场景构图或主体说明并追加到相关页面 |

**在每个页面的提示 frontmatter 中记录** 当 refs 存在时：
```yaml
references:
  - ref_id: 01
    filename: 01-ref-scene.png
    usage: style
    traits: "muted earth tones, soft-edged ink wash, low-contrast backgrounds"
```

角色一致性由第 3 步中写入的 `characters/characters.md` 中的**文本描述**驱动，这些描述嵌入到每个页面提示（第 5 步）中。第 7.1 步生成的可选 PNG 角色表是面向人类的审查工件，不是 `image_generate` 的输入。

## 选项

### 视觉维度

| 选项 | 值 | 描述 |
|--------|--------|-------------|
| Art | ligne-claire (默认), manga, realistic, ink-brush, chalk, minimalist | 艺术风格 / 渲染技术 |
| Tone | neutral (默认), warm, dramatic, romantic, energetic, vintage, action | 色调 / 氛围 |
| Layout | standard (默认), cinematic, dense, splash, mixed, webtoon, four-panel | 面板排列 |
| Aspect | 3:4 (默认，肖像), 4:3 (风景), 16:9 (宽屏) | 页面宽高比 |
| Language | auto (默认), zh, en, ja, 等 | 输出语言 |
| Refs | 文件路径 | 用于风格 / 调色板特征提取的参考图像（不传递给图像模型）。参见上面的[参考图像](#reference-images)。 |

### 部分工作流选项

| 选项 | 描述 |
|--------|-------------|
| 仅故事板 | 仅生成故事板，跳过提示和图像 |
| 仅提示 | 生成故事板 + 提示，跳过图像 |
| 仅图像 | 从现有提示目录生成图像 |
| 重新生成 N | 仅重新生成特定页面（如 `3` 或 `2,5,8`） |

详情：[references/partial-workflows.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/partial-workflows.md)

### 艺术、色调和预设目录

- **艺术风格**（6）：`ligne-claire`、`manga`、`realistic`、`ink-brush`、`chalk`、`minimalist`。完整定义在 `references/art-styles/<style>.md`。
- **色调**（7）：`neutral`、`warm`、`dramatic`、`romantic`、`energetic`、`vintage`、`action`。完整定义在 `references/tones/<tone>.md`。
- **预设**（5）具有超出普通 art+tone 的特殊规则：

  | 预设 | 等效 | 钩子 |
  |--------|-----------|------|
  | `ohmsha` | manga + neutral | 视觉隐喻，无说话头像，小工具揭示 |
  | `wuxia` | ink-brush + action | 气功效果，战斗视觉，氛围感 |
  | `shoujo` | manga + romantic | 装饰元素，眼睛细节，浪漫节拍 |
  | `concept-story` | manga + warm | 视觉符号系统，成长弧，对话+动作平衡 |
  | `four-panel` | minimalist + neutral + 四格布局 | 起承转合结构，B&W + 点色，简笔人物 |

  完整规则在 `references/presets/<preset>.md` — 选择预设时加载文件。

- **兼容性矩阵** 和 **内容信号 → 预设** 表在 [references/auto-selection.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/auto-selection.md)。在第 2 步推荐组合之前阅读它。

## 文件结构

输出目录：`comic/{topic-slug}/`
- Slug：来自主题的 2-4 个单词的 kebab-case（如 `alan-turing-bio`）
- 冲突：追加时间戳（如 `turing-story-20260118-143052`）

**内容**：
| 文件 | 描述 |
|------|-------------|
| `source-{slug}.md` | 保存的源内容（kebab-case slug 匹配输出目录） |
| `analysis.md` | 内容分析 |
| `story_board.md` | 带面板分解的故事板 |
| `characters/characters.md` | 角色定义 |
| `characters/characters.png` | 角色参考表（从 `image_generate` 下载） |
| `prompts/NN-{cover\|page}-[slug].md` | 生成提示 |
| `NN-{cover\|page}-[slug].png` | 生成的图像（从 `image_generate` 下载） |
| `refs/NN-ref-{slug}.{ext}` | 用户提供的参考图像（可选，用于来源证明） |

## 语言处理

**检测优先级**：
1. 用户指定的语言（显式选项）
2. 用户的对话语言
3. 源内容语言

**规则**：对所有交互使用用户的输入语言：
- 故事板大纲和场景描述
- 图像生成提示
- 用户选择选项和确认
- 进度更新、问题、错误、摘要

技术术语保持英文。

## 工作流

### 进度检查清单

```
Comic Progress:
- [ ] 步骤1：设置和将分析
  - [ ] 1.1 分析内容
  - [ ] 1.2 检查现有目录
- [ ] 步骤2：确认 — 风格和选项 ⚠️ 必需
- [ ] 步骤3：生成故事板和角色
- [ ] 步骤4：审查大纲（条件性）
- [ ] 步骤5：生成提示
- [ ] 步骤6：审查提示（条件性）
- [ ] 步骤7：生成图像
  - [ ] 7.1 生成角色表（如需要）→ characters/characters.png
  - [ ] 7.2 生成页面（提示中嵌入角色描述）
- [ ] 步骤8：完成报告
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
| 2 | 确认风格、焦点、受众、审查 | 用户首选项 |
| 3 | 生成故事板和角色 | `story_board.md`、`characters/` |
| 4 | 审查大纲（如请求） | 用户批准 |
| 5 | 生成提示 | `prompts/*.md` |
| 6 | 审查提示（如请求） | 用户批准 |
| 7.1 | 生成角色表（如需要） | `characters/characters.png` |
| 7.2 | 生成页面 | `*.png` 文件 |
| 8 | 完成报告 | 摘要 |

### 用户问题

使用 `clarify` 工具确认选项。由于 `clarify` 一次处理一个问题，首先询问最重要的问题，然后按顺序继续。参见 [references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/workflow.md) 了解完整的第 2 步问题集。

**超时处理（关键）**：`clarify` 可能返回 "The user did not provide a response within the time limit. Use your best judgement to make the choice and proceed." — 这**不是**用户对默认值的同意。

- 将其视为**仅针对该问题**的默认值。继续按顺序询问其余的第 2 步问题；每个问题都是一个独立的同意点。
- **在下一条消息中向用户可见地显示默认值**，以便他们有机会纠正它：例如 `"Style: defaulted to ohmsha preset (clarify timed out). Say the word to switch."` — 未报告的默认值与从未询问过无法区分。
- 在第一次超时后，不要将第 2 步折叠为单个 "使用所有默认值" 通道。如果用户真的不在，他们对所有五个问题都将同样不在；但当他们回来时，他们可以纠正可见的默认值，而无法纠正不可见的默认值。

### 步骤7：图像生成

对所有图像渲染使用 Hermes 内置的 `image_generate` 工具。其 schema 仅接受 `prompt` 和 `aspect_ratio`（`landscape` |`portrait` |`square`）；它**返回 URL**，而不是本地文件。因此，每个生成的页面或角色表必须下载到输出目录。

**提示文件要求（硬性）**：在调用 `image_generate` **之前**，将每个图像的完整最终提示写入 `prompts/` 下的独立文件（命名：`NN-{type}-[slug].md`）。提示文件是可重现性记录。

**宽高比映射** — 故事板中的 `aspect_ratio` 字段映射到 `image_generate` 的格式如下：

| 故事板比例 | `image_generate` 格式 |
|------------------|-------------------------|
| `3:4`、`9:16`、`2:3` | `portrait` |
| `4:3`、`16:9`、`3:2` | `landscape` |
| `1:1` | `square` |

**下载步骤** — 每次 `image_generate` 调用后：
1. 从工具结果读取 URL
2. 使用**绝对** 输出路径获取图像字节，例如。
   `curl -fsSL "<url>" -o /abs/path/to/comic/<slug>/NN-page-<slug>.png`
3. 在继续下一页之前，验证文件在该确切路径存在且非空

**永远不要依赖 shell CWD 持久性来设置 `-o` 路径。** 终端工具的持久 shell CWD 可能在批次之间改变（会话过期、`TERMINAL_LIFETIME_SECONDS`、将你留在错误目录的失败 `cd`）。`curl -o relative/path.png` 是一个静默的 footgun：如果 CWD 漂移，文件会落在其他地方且无错误。**始终传递完全限定的绝对路径到 `-o`**，或将 `workdir=<abs path>` 传递到终端工具。2026 年 4 月的事件：一个 10 页漫画的第 06-09 页落到了仓库根目录而不是 `comic/<slug>/`，因为批次 3 继承了批次 2 的陈旧 CWD，并且 `curl -o 06-page-skills.png` 写入了错误的目录。代理然后花了几轮声称文件存在于它们不在的地方。

**7.1 角色表** — 当漫画是多页且具有重复角色时生成它（到 `characters/characters.png`，宽高比 `landscape`）。对于简单的预设（如四格极简）或单页漫画跳过。在调用 `image_generate` 之前，必须存在 `characters/characters.md` 的提示文件。渲染的 PNG 是**面向人类的审查工件**（以便用户可以视觉验证角色设计）以及以后重新生成或手动提示编辑的参考 — 它**不** 驱动步骤 7.2。页面提示是在步骤 5 中根据 `characters/characters.md` 中的**文本描述**写入的；`image_generate` 无法接受图像作为视觉输入。

**7.2 页面** — 在调用 `image_generate` 之前，每个页面的提示必须已经存在于 `prompts/NN-{cover\|page}-[slug].md`。由于 `image_generate` 仅支持提示，角色一致性由**在步骤 5 期间将角色描述（源自 `characters/characters.md`）内联嵌入到每个页面提示中**来强制执行**。无论是否在 7.1 中生成 PNG 表，嵌入都是统一完成的；PNG 只是审查/重新生成辅助。

**后备规则**：现有的 `prompts/…md` 和 `…png` 文件 → 在重新生成之前用 `-backup-YYYYMMDD-HHMMSS` 后缀重命名。

完整逐步工作流（分析、故事板、审查关卡、重新生成变体）：[references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/workflow.md)。

## 参考

**核心模板**：
- [analysis-framework.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/analysis-framework.md) - 深度内容分析
- [character-template.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/character-template.md) - 角色定义格式
- [storyboard-template.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/storyboard-template.md) - 故事板结构
- [ohmsha-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/ohmsha-guide.md) - Ohmsha 漫画 specifics

**风格定义**：
- `references/art-styles/` - 艺术风格（ligne-claire, manga, realistic, ink-brush, chalk, minimalist）
- `references/tones/` - 色调（neutral, warm, dramatic, romantic, energetic, vintage, action）
- `references/presets/` - 具有特殊规则的预设（ohmsha, wuxia, shoujo, concept-story, four-panel）
- `references/layouts/` - 布局（standard, cinematic, dense, splash, mixed, webtoon, four-panel）

**工作流**：
- [workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/workflow.md) - 完整工作流详情
- [auto-selection.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/auto-selection.md) - 内容信号分析
- [partial-workflows.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/baoyu-comic/references/partial-workflows.md) - 部分工作流选项

## 页面修改

| 操作 | 步骤 |
|--------|-------|
| **编辑** | **先更新提示文件** → 重新生成图像 → 下载新 PNG |
| **添加** | 在位置创建提示 → 用嵌入的角色描述生成 → 重新编号后续 → 更新故事板 |
| **删除** | 移除文件 → 重新编号后续 → 更新故事板 |

**重要**：更新页面时，始终先更新提示文件（`prompts/NN-{cover\|page}-[slug].md`），然后再重新生成。这确保更改被记录且可重现。

## 陷阱

- 图像生成：每页 10-30 秒；失败时自动重试一次
- **始终下载** `image_generate` 返回的 URL 到本地 PNG — 下游工具（和用户审查）期望输出目录中的文件，而不是临时 URL
- **对 `curl -o` 使用绝对路径** — 永远不要依赖跨批次的持久 shell CWD。静默 footgun：文件落在错误的目录中，并且在预期路径上的后续 `ls` 什么也看不到。参见步骤 7"下载步骤"。
- 对敏感的公共人物使用风格化的替代方案
- **步骤 2 确认必需** - 不要跳过
- **步骤 4/6 条件性** - 仅当用户在步骤 2 中请求时
- **步骤 7.1 角色表** - 推荐用于多页漫画，简单预设可选。PNG 是审查/重新生成辅助；页面提示（在步骤 5 中写入）使用 `characters/characters.md` 中的文本描述，而不是 PNG。`image_generate` 不接受图像作为视觉输入
- **去除秘密** — 在写入任何输出文件之前，扫描源内容中的 API 密钥、令牌或凭据
